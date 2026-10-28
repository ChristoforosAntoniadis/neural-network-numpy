"""
Tests for batch normalization, the network's train / eval switch, and
the trainer's handling of both (checkpoints, evaluation mode, divergence).
"""

from typing import Any

import numpy as np
import pytest
from tests.test_gradients import numerical_grad

from nn_numpy.config_loader import ConfigLoader
from nn_numpy.modeling.trainer import Trainer
from nn_numpy.nn.layers import BatchNorm
from nn_numpy.nn.losses import get_loss
from nn_numpy.nn.network import NeuralNetwork
from nn_numpy.nn.optimizers import get_optimizer


def make_bn(num_features: int = 3, seed: int = 0) -> BatchNorm:
    """
    Build a BatchNorm layer with non-trivial gamma and beta.

    Parameters
    ----------
    num_features : int, default=3
        Number of features (columns) the layer normalizes.
    seed : int, default=0
        Seed for the generator that draws gamma and beta.

    Returns
    -------
    BatchNorm
        A layer in training mode whose gamma, of shape (1, num_features),
        is drawn uniformly from [0.5, 1.5) and whose beta, of the same
        shape, is drawn from a standard normal.

    Notes
    -----
    Processing:
    1. Create `BatchNorm(num_features)` (gamma = 1, beta = 0).
    2. Overwrite gamma and beta in place with random values.

    Why: with gamma = 1 and beta = 0 some mistakes (for example, forgetting
    to multiply by gamma) would go unnoticed, so the tests use random values.
    """
    rng = np.random.RandomState(seed)
    bn = BatchNorm(num_features)
    bn.gamma[:] = rng.uniform(0.5, 1.5, size=(1, num_features))
    bn.beta[:] = rng.normal(size=(1, num_features))
    return bn


# ------------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------------


def test_training_mode_normalizes_each_feature_over_the_batch() -> None:
    """
    Training-mode output has per-feature mean 0 and standard deviation 1.

    Notes
    -----
    A fresh BatchNorm (gamma = 1, beta = 0) is applied to a 64x3 batch
    drawn with mean 5 and scale 3. Each output column must have mean 0
    and standard deviation 1 (up to the small eps in the denominator).

    The mean is 0 up to float64 rounding (about 1e-15 here), well inside
    atol=1e-10. The standard deviation is sqrt(var / (var + eps)); with
    var ~ 3^2 = 9 and eps = 1e-5 that is about 1 - 1e-5 / (2 * 9) =
    1 - 5.6e-7, well inside atol=1e-4.
    """
    X = np.random.RandomState(1).normal(loc=5.0, scale=3.0, size=(64, 3))
    bn = BatchNorm(3)  # gamma = 1, beta = 0

    out = bn.forward(X)

    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-4)


def test_running_statistics_follow_the_momentum_rule() -> None:
    """
    One training-mode forward pass updates the running statistics by momentum.

    Notes
    -----
    Starting from running_mean = 0 and running_var = 1 with momentum 0.1,
    one forward pass on a 32x3 batch must give
    running = 0.9 * old + 0.1 * batch statistic, for both the mean and
    the (biased) variance.
    """
    X = np.random.RandomState(1).normal(loc=2.0, size=(32, 3))
    bn = BatchNorm(3, momentum=0.1)

    bn.forward(X)

    np.testing.assert_allclose(bn.running_mean, 0.9 * 0.0 + 0.1 * X.mean(axis=0, keepdims=True))
    np.testing.assert_allclose(bn.running_var, 0.9 * 1.0 + 0.1 * X.var(axis=0, keepdims=True))


def test_eval_mode_uses_running_statistics_and_does_not_update_them() -> None:
    """
    Evaluation mode normalizes with the running statistics and leaves them unchanged.

    Notes
    -----
    The running mean and variance are set to known values and the layer
    is switched to evaluation mode. The output on a 5x3 batch must equal
    gamma * (X - running_mean) / sqrt(running_var + eps) + beta, and the
    running statistics must still hold their original values afterwards.
    """
    bn = make_bn()
    bn.running_mean[:] = [[1.0, -1.0, 0.5]]
    bn.running_var[:] = [[4.0, 1.0, 0.25]]
    bn.training = False
    X = np.random.RandomState(2).normal(size=(5, 3))

    out = bn.forward(X)

    expected = bn.gamma * (X - bn.running_mean) / np.sqrt(bn.running_var + bn.eps) + bn.beta
    np.testing.assert_allclose(out, expected)
    np.testing.assert_array_equal(bn.running_mean, [[1.0, -1.0, 0.5]])
    np.testing.assert_array_equal(bn.running_var, [[4.0, 1.0, 0.25]])


def test_eval_mode_prediction_is_independent_of_the_rest_of_the_batch() -> None:
    """
    In evaluation mode a sample's output does not depend on the other samples in the batch.

    Notes
    -----
    Running statistics are built with one training-mode pass, then the
    layer is switched to evaluation mode. One sample is passed alone and
    again as the first row of a 10-row batch; the two outputs must match
    and be finite.
    """
    bn = make_bn()
    bn.forward(np.random.RandomState(3).normal(size=(50, 3)))  # build running stats
    bn.training = False
    sample = np.random.RandomState(4).normal(size=(1, 3))
    others = np.random.RandomState(5).normal(size=(9, 3))

    alone = bn.forward(sample)
    in_batch = bn.forward(np.vstack([sample, others]))[:1]

    np.testing.assert_allclose(alone, in_batch)
    assert np.all(np.isfinite(alone))


# ------------------------------------------------------------------
# Backward pass vs finite differences
# ------------------------------------------------------------------


@pytest.mark.parametrize("training", [True, False])
def test_batchnorm_gradients_match_numerical(training: bool) -> None:
    """
    BatchNorm's analytic gradients for X, gamma and beta match finite differences.

    Parameters
    ----------
    training : bool
        Mode the layer is tested in: True (batch statistics, where the
        gradient also flows through the batch mean and variance) or False
        (fixed running statistics).

    Notes
    -----
    The scalar loss is L = sum(upstream * Y) for a random `upstream`
    array, so dL/dY = upstream. After one forward and backward pass, the
    gradients for X, gamma and beta are compared with `numerical_grad`
    (central differences) at rtol=1e-4, atol=1e-7.
    """
    rng = np.random.RandomState(6)
    X = rng.normal(loc=1.0, scale=2.0, size=(8, 3))
    upstream = rng.normal(size=(8, 3))  # L = sum(upstream * Y)

    bn = make_bn()
    bn.running_mean[:] = rng.normal(size=(1, 3))
    bn.running_var[:] = rng.uniform(0.5, 2.0, size=(1, 3))
    bn.training = training

    bn.forward(X)
    dX = bn.backward(upstream)
    analytic = {"X": dX, "gamma": bn.dgamma.copy(), "beta": bn.dbeta.copy()}

    def loss_value() -> float:
        """
        Return L = sum(upstream * Y) for the current X, gamma and beta.

        Returns
        -------
        float
            The scalar loss; `numerical_grad` perturbs X, gamma or beta in
            place between calls.
        """
        # The training-mode forward updates the running statistics, but they
        # do not affect the training-mode output, so L is unaffected.
        return np.sum(upstream * bn.forward(X))

    for name, array in [("X", X), ("gamma", bn.gamma), ("beta", bn.beta)]:
        np.testing.assert_allclose(
            analytic[name], numerical_grad(loss_value, array), rtol=1e-4, atol=1e-7
        )


# ------------------------------------------------------------------
# Network train / eval switch
# ------------------------------------------------------------------


def build_bn_network(output_activation: str = "sigmoid") -> NeuralNetwork:
    """
    Build a small 3 -> 5 -> 1 network with batch norm on the hidden layer.

    Parameters
    ----------
    output_activation : str, default="sigmoid"
        Activation name for the single output unit.

    Returns
    -------
    NeuralNetwork
        Network with layers Dense(3, 5), BatchNorm(5), ReLU, Dense(5, 1)
        and the output activation, seeded with random_seed=0.

    Notes
    -----
    Processing:
    1. Create `NeuralNetwork(random_seed=0)`.
    2. Build it from a config whose hidden layer has "batch_norm": True.
    """
    network = NeuralNetwork(random_seed=0)
    network.build_from_config(
        {
            "input_dimension": 3,
            "layers": [
                {"type": "dense", "units": 5, "activation": "relu", "batch_norm": True},
                {"type": "dense", "units": 1, "activation": output_activation},
            ],
        }
    )
    return network


def test_batch_norm_is_placed_between_dense_and_activation() -> None:
    """
    A "batch_norm" layer config puts BatchNorm between Dense and its activation.

    Notes
    -----
    The layer class names of `build_bn_network()` must be exactly
    Dense, BatchNorm, ReLU, Dense, Sigmoid.
    """
    names = [type(layer).__name__ for layer in build_bn_network().layers]
    assert names == ["Dense", "BatchNorm", "ReLU", "Dense", "Sigmoid"]


def test_batch_norm_does_not_change_the_dense_weights() -> None:
    """
    Adding BatchNorm leaves the initial Dense weights unchanged.

    Notes
    -----
    BatchNorm draws no random numbers, so Dense initialization is
    unaffected. Two networks with the same seed are built, one without and
    one with batch norm, and their Dense weight matrices must be
    identical.
    """
    plain = NeuralNetwork(random_seed=0)
    plain.build_from_config(
        {
            "input_dimension": 3,
            "layers": [
                {"type": "dense", "units": 5, "activation": "relu"},
                {"type": "dense", "units": 1, "activation": "sigmoid"},
            ],
        }
    )
    dense_plain = [l for l in plain.layers if type(l).__name__ == "Dense"]
    dense_bn = [l for l in build_bn_network().layers if type(l).__name__ == "Dense"]
    for a, b in zip(dense_plain, dense_bn):
        np.testing.assert_array_equal(a.weights, b.weights)


def test_network_train_and_eval_set_every_layer() -> None:
    """
    NeuralNetwork.eval() and train() set the training flag on every layer.

    Notes
    -----
    After `eval()` every layer's `training` must be False; after
    `train()` every layer's `training` must be True.
    """
    network = build_bn_network()
    network.eval()
    assert all(layer.training is False for layer in network.layers)
    network.train()
    assert all(layer.training is True for layer in network.layers)


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------


def make_classification_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a linearly separable binary classification dataset.

    Parameters
    ----------
    n : int, default=200
        Number of samples.
    seed : int, default=0
        Seed for the random generator.

    Returns
    -------
    X : numpy.ndarray of shape (n, 3), dtype float64
        Standard-normal features.
    y : numpy.ndarray of shape (n, 1), dtype float64
        Labels: 1.0 where X[:, 0] - X[:, 1] > 0, else 0.0.

    Notes
    -----
    Processing:
    1. Draw X from a standard normal distribution.
    2. Label each row by the sign of X[:, 0] - X[:, 1] and reshape the
       labels to a column.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] - X[:, 1] > 0).astype(float).reshape(-1, 1)
    return X, y


def test_trainer_with_batch_norm_learns_and_predicts_single_samples() -> None:
    """
    A batch-norm network trains well and predicts single samples consistently.

    Notes
    -----
    A batch-norm classifier is trained for 30 epochs with AdaBelief on 200
    samples (50 for validation) and evaluated on the last 50. Asserted:
    training did not diverge, the test BCE is below 0.4 (always predicting
    0.5 gives ln 2 = 0.693), every layer is in
    evaluation mode after `fit`, and a sample predicted alone gives the
    same output as when it is the first row of a 10-sample batch.
    """
    X, y = make_classification_data(300)
    network = build_bn_network()
    trainer = Trainer(network, get_loss("bce"), get_optimizer("adabelief", 0.01), "classification")

    history = trainer.fit(
        X[:200], y[:200], X[200:250], y[200:250], epochs=30, batch_size=16, verbose=False
    )
    results = trainer.evaluate(X[250:], y[250:])

    assert history["diverged"] is False
    assert results["test_metric"] < 0.4
    # After fit the network is in evaluation mode, and a single sample gives
    # the same prediction alone as inside a batch.
    assert all(layer.training is False for layer in network.layers)
    np.testing.assert_allclose(trainer.predict(X[250:251]), trainer.predict(X[250:260])[:1])


def test_checkpoint_round_trip_includes_batch_norm_running_stats() -> None:
    """
    Trainer checkpoints save and restore BatchNorm's running statistics as well as gamma.

    Notes
    -----
    After one forward pass moves the running statistics, the state is
    saved with `_get_model_state`. The running mean, running variance and
    gamma are then changed, and `_set_model_state` must put all three back
    to the saved values (BatchNorm is the second trainable layer, index 1).
    """
    network = build_bn_network()
    trainer = Trainer(network, get_loss("bce"), get_optimizer("sgd", 0.1), "classification")
    network.forward(np.random.RandomState(7).normal(size=(20, 3)))  # move running stats

    saved = trainer._get_model_state()
    bn = network.layers[1]
    bn.running_mean += 1.0
    bn.running_var += 1.0
    bn.gamma += 1.0
    trainer._set_model_state(saved)

    np.testing.assert_array_equal(bn.running_mean, saved[1]["buffers"]["running_mean"])
    np.testing.assert_array_equal(bn.running_var, saved[1]["buffers"]["running_var"])
    np.testing.assert_array_equal(bn.gamma, saved[1]["params"]["gamma"])


def test_trainer_stops_and_flags_a_diverging_run() -> None:
    """
    The trainer stops early and sets "diverged" when the loss stops being finite.

    Notes
    -----
    A regression network is trained with SGD at learning rate 10 on
    targets scaled by 1000, which makes the loss overflow. NumPy
    overflow warnings are silenced. Asserted: history["diverged"] is
    True, stop_reason is "diverged", fewer than the 50 requested epochs
    ran, and the last recorded
    training or validation loss is not finite.
    """
    rng = np.random.RandomState(8)
    X = rng.normal(size=(100, 3))
    y = 1000.0 * rng.normal(size=(100, 1))
    network = NeuralNetwork(random_seed=0)
    network.build_from_config(
        {
            "input_dimension": 3,
            "layers": [
                {"type": "dense", "units": 8, "activation": "relu"},
                {"type": "dense", "units": 1, "activation": "linear"},
            ],
        }
    )
    trainer = Trainer(network, get_loss("mse"), get_optimizer("sgd", 10.0), "regression")

    with np.errstate(all="ignore"):
        history = trainer.fit(
            X[:80], y[:80], X[80:], y[80:], epochs=50, batch_size=16, verbose=False
        )

    assert history["diverged"] is True
    assert history["epochs_ran"] < 50
    assert history["stop_reason"] == "diverged"
    assert not np.isfinite(history["train_loss"][-1]) or not np.isfinite(history["val_loss"][-1])


# ------------------------------------------------------------------
# Config validation
# ------------------------------------------------------------------


def make_config(**layer_extra: Any) -> dict[str, Any]:
    """
    Build a minimal valid classification config for ConfigLoader.validate.

    Parameters
    ----------
    **layer_extra : Any
        Extra keys merged into the first (hidden) layer of architecture
        "A", for example `batch_norm=True`.

    Returns
    -------
    dict[str, Any]
        Config with every required top-level, preprocessing and
        experiments key, and one architecture "A": a 2-unit ReLU hidden
        layer (plus `layer_extra`) and a 1-unit sigmoid output layer.

    Notes
    -----
    Processing:
    1. Return a fresh dictionary literal each call, so tests cannot
       affect each other by mutating it.
    """
    return {
        "task_type": "classification",
        "input_dimension": 4,
        "loss": "bce",
        "preprocessing": {"enabled": True, "scale_features": True},
        "architectures": {
            "A": [
                {"type": "dense", "units": 2, "activation": "relu", **layer_extra},
                {"type": "dense", "units": 1, "activation": "sigmoid"},
            ]
        },
        "experiments": {
            "optimizers": ["sgd"],
            "learning_rates": [0.1],
            "batch_sizes": [16],
            "epochs": 10,
            "early_stopping": True,
            "patience": 2,
            "min_delta": 0.0,
            "min_epochs_before_early_stop": 0,
        },
    }


def test_config_accepts_batch_norm_flag() -> None:
    """
    The config validator accepts "batch_norm": true on a layer.

    Notes
    -----
    `validate` must return True for `make_config(batch_norm=True)`.
    """
    assert ConfigLoader().validate(make_config(batch_norm=True))


@pytest.mark.parametrize("value", ["yes", 1])
def test_config_rejects_non_boolean_batch_norm(value: str | int) -> None:
    """
    The config validator rejects a "batch_norm" value that is not a boolean.

    Parameters
    ----------
    value : str or int
        Invalid value for "batch_norm": the string "yes" or the integer 1.

    Notes
    -----
    `validate` must raise ValueError for `make_config(batch_norm=value)`.
    """
    with pytest.raises(ValueError):
        ConfigLoader().validate(make_config(batch_norm=value))
