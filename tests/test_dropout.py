"""
Tests for inverted dropout, its place in the network, the trainer's
evaluation-mode training metric, and the config rules for dropout.
"""

from typing import Any

import numpy as np
import pytest

from nn_numpy.config_loader import ConfigLoader
from nn_numpy.modeling.trainer import Trainer
from nn_numpy.nn.layers import Dropout
from nn_numpy.nn.losses import get_loss
from nn_numpy.nn.network import NeuralNetwork
from nn_numpy.nn.optimizers import get_optimizer

# ------------------------------------------------------------------
# The layer
# ------------------------------------------------------------------


def test_training_mode_drops_about_rate_and_scales_survivors() -> None:
    """
    In training mode about `rate` of the values are zeroed and survivors are scaled by 1 / (1 - rate).

    Notes
    -----
    A Dropout(0.2) layer is applied to a 500x200 array of ones. Asserted:
    the fraction of zeros is 0.2 within 0.01, every non-zero value equals
    1 / 0.8, and the overall mean stays 1.0 within 0.01 (the expected
    value is preserved).

    With 500 * 200 = 100,000 values, the dropped fraction has standard
    deviation sqrt(0.2 * 0.8 / 100000) = 0.0013, so abs=0.01 is about 8
    standard deviations. The mean is 1.25 times the kept fraction, with
    standard deviation 1.25 * 0.0013 = 0.0016, so abs=0.01 is about 6.
    """
    layer = Dropout(0.2, random_state=np.random.RandomState(0))
    X = np.ones((500, 200))

    out = layer.forward(X)

    dropped = np.mean(out == 0)
    assert dropped == pytest.approx(0.2, abs=0.01)
    # Survivors are scaled by 1 / (1 - rate).
    np.testing.assert_allclose(out[out != 0], 1.0 / 0.8)
    # So the expected value of the output matches the input.
    assert out.mean() == pytest.approx(1.0, abs=0.01)


def test_eval_mode_is_the_identity() -> None:
    """
    In evaluation mode dropout passes values and gradients through unchanged.

    Notes
    -----
    With `training = False`, `forward(X)` must equal X exactly and
    `backward(upstream)` must equal upstream exactly.
    """
    layer = Dropout(0.5, random_state=np.random.RandomState(0))
    layer.training = False
    X = np.random.RandomState(1).normal(size=(10, 4))

    np.testing.assert_array_equal(layer.forward(X), X)
    upstream = np.random.RandomState(2).normal(size=(10, 4))
    np.testing.assert_array_equal(layer.backward(upstream), upstream)


def test_backward_reuses_the_forward_mask() -> None:
    """
    The backward pass multiplies the gradient by the mask used in the forward pass.

    Notes
    -----
    Y = X * mask is element-wise, so dL/dX = dL/dY * mask. After one
    training-mode forward and backward pass, asserted: the output equals
    X * layer.mask, the gradient equals upstream * layer.mask, and every
    dropped position receives a zero gradient.
    """
    layer = Dropout(0.3, random_state=np.random.RandomState(0))
    X = np.random.RandomState(1).normal(size=(20, 6))
    upstream = np.random.RandomState(2).normal(size=(20, 6))

    out = layer.forward(X)
    grad = layer.backward(upstream)

    # The training-mode forward() above set the mask, so it is an array, not None.
    np.testing.assert_allclose(out, X * layer.mask)  # type: ignore[operator]  # mask set
    np.testing.assert_allclose(grad, upstream * layer.mask)  # type: ignore[operator]  # mask set
    # Dropped values get no gradient.
    assert np.all(grad[out == 0] == 0)


def test_new_mask_for_every_batch() -> None:
    """
    Each training-mode forward pass draws a new random mask.

    Notes
    -----
    Two forward passes of the same 10x10 array of ones through a
    Dropout(0.5) layer must give different outputs. Two independent
    masks of 100 values agree with probability 0.5^100 = 8e-31.
    """
    layer = Dropout(0.5, random_state=np.random.RandomState(0))
    X = np.ones((10, 10))
    assert not np.array_equal(layer.forward(X), layer.forward(X))


@pytest.mark.parametrize("rate", [-0.1, 1.0, 1.5])
def test_invalid_rate_raises(rate: float) -> None:
    """
    Creating a Dropout layer with a rate outside [0, 1) raises ValueError.

    Parameters
    ----------
    rate : float
        Invalid rate: -0.1 (negative), 1.0 (would drop everything) or
        1.5 (above 1).

    Notes
    -----
    `Dropout(rate)` must raise ValueError.
    """
    with pytest.raises(ValueError):
        Dropout(rate)


# ------------------------------------------------------------------
# In the network
# ------------------------------------------------------------------


def build(dropout: float | None = None, seed: int = 0) -> NeuralNetwork:
    """
    Build a small 3 -> 6 -> 1 classifier, optionally with dropout on the hidden layer.

    Parameters
    ----------
    dropout : float or None, default=None
        Dropout rate for the hidden layer. None leaves the "dropout" key
        out of the layer config, so no Dropout layer is added.
    seed : int, default=0
        Random seed passed to `NeuralNetwork`.

    Returns
    -------
    NeuralNetwork
        Network with layers Dense(3, 6), ReLU, [Dropout], Dense(6, 1),
        Sigmoid, in training mode.

    Notes
    -----
    Processing:
    1. Make the hidden-layer config and add "dropout" if given.
    2. Create `NeuralNetwork(random_seed=seed)` and build it from the
       config.
    """
    hidden = {"type": "dense", "units": 6, "activation": "relu"}
    if dropout is not None:
        hidden["dropout"] = dropout
    network = NeuralNetwork(random_seed=seed)
    network.build_from_config(
        {
            "input_dimension": 3,
            "layers": [hidden, {"type": "dense", "units": 1, "activation": "sigmoid"}],
        }
    )
    return network


def test_dropout_is_placed_after_the_activation() -> None:
    """
    A "dropout" layer config adds Dropout right after the hidden activation.

    Notes
    -----
    The layer class names of `build(0.2)` must be exactly
    Dense, ReLU, Dropout, Dense, Sigmoid.
    """
    names = [type(layer).__name__ for layer in build(0.2).layers]
    assert names == ["Dense", "ReLU", "Dropout", "Dense", "Sigmoid"]


def test_dropout_does_not_change_the_initial_weights() -> None:
    """
    Adding dropout leaves the initial Dense weights unchanged.

    Notes
    -----
    Dropout masks use a separate generator from weight initialization.
    Networks built with the same seed, without and with dropout, must
    have identical weight matrices layer by layer.
    """
    plain = [l.weights for l in build().layers if hasattr(l, "weights")]
    dropped = [l.weights for l in build(0.2).layers if hasattr(l, "weights")]
    for a, b in zip(plain, dropped):
        np.testing.assert_array_equal(a, b)


def test_dropout_masks_are_reproducible_from_the_seed() -> None:
    """
    Two networks built with the same seed draw the same dropout masks.

    Notes
    -----
    Two separate networks with dropout 0.5 and seed 3 run a training-mode
    forward pass on the same input; the outputs must be identical.
    """
    X = np.ones((8, 3))
    out_a = build(0.5, seed=3).forward(X)
    out_b = build(0.5, seed=3).forward(X)
    np.testing.assert_array_equal(out_a, out_b)


def test_dropout_is_not_trainable() -> None:
    """
    Dropout layers are not returned as trainable layers.

    Notes
    -----
    Every layer from `get_trainable_layers()` of a network with dropout
    must be a Dense layer.
    """
    network = build(0.2)
    assert all(type(l).__name__ == "Dense" for l in network.get_trainable_layers())


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------


def make_data(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a linearly separable binary classification dataset.

    Parameters
    ----------
    n : int, default=300
        Number of samples.
    seed : int, default=0
        Seed for the random generator.

    Returns
    -------
    X : numpy.ndarray of shape (n, 3), dtype float64
        Standard-normal features.
    y : numpy.ndarray of shape (n, 1), dtype float64
        Labels: 1.0 where X[:, 0] + 0.5 * X[:, 2] > 0, else 0.0.

    Notes
    -----
    Processing:
    1. Draw X from a standard normal distribution.
    2. Label each row by the sign of X[:, 0] + 0.5 * X[:, 2] and reshape
       the labels to a column.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * X[:, 2] > 0).astype(float).reshape(-1, 1)
    return X, y


def test_trainer_with_dropout_learns_and_predicts_deterministically() -> None:
    """
    A dropout network trains well and gives repeatable predictions after training.

    Notes
    -----
    A classifier with dropout 0.2 is trained for 30 epochs with AdaBelief
    on 200 samples (50 for validation). Asserted: the evaluation-mode BCE
    on the last 50 samples is below 0.4 (always predicting 0.5 gives
    ln 2 = 0.693), and predicting the same inputs
    twice gives identical outputs (dropout is off in evaluation mode).
    """
    X, y = make_data()
    network = build(0.2)
    trainer = Trainer(network, get_loss("bce"), get_optimizer("adabelief", 0.01), "classification")

    trainer.fit(X[:200], y[:200], X[200:250], y[200:250], epochs=30, batch_size=16, verbose=False)

    assert trainer.score(X[250:], y[250:]) < 0.4
    # Evaluation mode switches dropout off, so predictions are repeatable.
    np.testing.assert_array_equal(trainer.predict(X[250:]), trainer.predict(X[250:]))


def test_score_matches_the_test_metric_from_evaluate() -> None:
    """
    Trainer.score returns the same metric as Trainer.evaluate on the same data.

    Notes
    -----
    After 5 epochs of SGD on a dropout network, `score(X, y)` must equal
    `evaluate(X, y)["test_metric"]` (both run in evaluation mode).
    """
    X, y = make_data()
    trainer = Trainer(build(0.2), get_loss("bce"), get_optimizer("sgd", 0.1), "classification")
    trainer.fit(X[:200], y[:200], X[200:250], y[200:250], epochs=5, batch_size=16, verbose=False)

    assert trainer.score(X[250:], y[250:]) == pytest.approx(
        trainer.evaluate(X[250:], y[250:])["test_metric"]
    )


# ------------------------------------------------------------------
# Config validation
# ------------------------------------------------------------------


def make_config(
    hidden_extra: dict[str, Any] | None = None, output_extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Build a minimal valid classification config for ConfigLoader.validate.

    Parameters
    ----------
    hidden_extra : dict[str, Any] or None, default=None
        Extra keys merged into the hidden layer of architecture "A", for
        example {"dropout": 0.2}.
    output_extra : dict[str, Any] or None, default=None
        Extra keys merged into the output layer of architecture "A".

    Returns
    -------
    dict[str, Any]
        Config with every required top-level, preprocessing and
        experiments key, and one architecture "A": a 2-unit ReLU hidden
        layer and a 1-unit sigmoid output layer, each with its extras.

    Notes
    -----
    Processing:
    1. Return a fresh dictionary literal each call, with None extras
       treated as empty dicts.
    """
    return {
        "task_type": "classification",
        "input_dimension": 4,
        "loss": "bce",
        "preprocessing": {"enabled": True, "scale_features": True},
        "architectures": {
            "A": [
                {"type": "dense", "units": 2, "activation": "relu", **(hidden_extra or {})},
                {"type": "dense", "units": 1, "activation": "sigmoid", **(output_extra or {})},
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


@pytest.mark.parametrize("rate", [0, 0.0, 0.2, 0.5])
def test_config_accepts_valid_dropout(rate: float) -> None:
    """
    The config validator accepts dropout rates in [0, 1) on a hidden layer.

    Parameters
    ----------
    rate : int or float
        Valid rate: 0 (int), 0.0, 0.2 or 0.5.

    Notes
    -----
    `validate` must return True when the hidden layer has
    {"dropout": rate}.
    """
    assert ConfigLoader().validate(make_config({"dropout": rate}))


@pytest.mark.parametrize("rate", [-0.1, 1, 1.0, "0.2", True])
def test_config_rejects_invalid_dropout(rate: float | str | bool) -> None:
    """
    The config validator rejects dropout values that are out of range or not numbers.

    Parameters
    ----------
    rate : float, int, str or bool
        Invalid value: -0.1 (negative), 1 and 1.0 (not below 1), "0.2"
        (a string) or True (a boolean, rejected even though Python treats
        it as the number 1).

    Notes
    -----
    `validate` must raise ValueError when the hidden layer has
    {"dropout": rate}.
    """
    with pytest.raises(ValueError):
        ConfigLoader().validate(make_config({"dropout": rate}))


def test_config_rejects_dropout_on_the_output_layer() -> None:
    """
    The config validator rejects dropout on the output layer.

    Notes
    -----
    `validate` must raise ValueError when the last layer has
    {"dropout": 0.2}, because dropping output values would make the
    prediction itself random.
    """
    with pytest.raises(ValueError):
        ConfigLoader().validate(make_config(output_extra={"dropout": 0.2}))
