"""
Tests for the Dense layer's optional bias term and weight initialization,
and for the code that must handle every trainable parameter (optimizers,
early-stopping checkpoints, config validation).
"""

from typing import Any

import numpy as np
import pytest

from nn_numpy.config_loader import ConfigLoader
from nn_numpy.modeling.trainer import Trainer
from nn_numpy.nn.layers import Dense
from nn_numpy.nn.losses import get_loss
from nn_numpy.nn.network import NeuralNetwork
from nn_numpy.nn.optimizers import get_optimizer

# ------------------------------------------------------------------
# Bias term
# ------------------------------------------------------------------


def test_bias_is_added_to_every_sample() -> None:
    """
    A Dense layer with bias adds the same bias row to every sample.

    Notes
    -----
    The bias is set to [[1.0, -2.0]] and a 4-sample batch is passed
    forward. The output must equal X @ W + [[1.0, -2.0]] (broadcast over
    the rows).
    """
    layer = Dense(3, 2, random_state=np.random.RandomState(0), use_bias=True)
    layer.bias[:] = [[1.0, -2.0]]  # type: ignore[index]  # use_bias=True
    X = np.random.RandomState(1).normal(size=(4, 3))

    np.testing.assert_allclose(layer.forward(X), X @ layer.weights + [[1.0, -2.0]])


def test_bias_gradient_is_batch_sum_of_upstream_gradient() -> None:
    """
    The bias gradient is the upstream gradient summed over the batch.

    Notes
    -----
    Because the same bias is added to every sample, dL/db is the sum of
    dL/dZ over the rows. After forward and backward on a 5-sample batch,
    `dbias` must equal upstream.sum(axis=0, keepdims=True).
    """
    layer = Dense(3, 2, random_state=np.random.RandomState(0), use_bias=True)
    X = np.random.RandomState(1).normal(size=(5, 3))
    upstream = np.random.RandomState(2).normal(size=(5, 2))

    layer.forward(X)
    layer.backward(upstream)

    np.testing.assert_allclose(
        layer.dbias,  # type: ignore[arg-type]  # use_bias=True, so dbias is an array
        upstream.sum(axis=0, keepdims=True),
    )


def test_bias_free_layer_exposes_only_weights() -> None:
    """
    A Dense layer without bias exposes only "weights" as parameter and gradient.

    Notes
    -----
    The keys of `get_params()` and `get_grads()` must both be exactly
    {"weights"}.
    """
    layer = Dense(3, 2, random_state=np.random.RandomState(0))
    assert set(layer.get_params()) == {"weights"}
    assert set(layer.get_grads()) == {"weights"}


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------


def test_default_init_matches_original_scheme() -> None:
    """
    The default initialization reproduces the original N(0, 0.1^2) draw exactly.

    Notes
    -----
    A default Dense(4, 3) layer built from RandomState(42) must have the
    same weights as RandomState(42).normal(0.0, 0.1, size=(4, 3)), so
    results from before init schemes were added are unchanged.
    """
    layer = Dense(4, 3, random_state=np.random.RandomState(42))
    expected = np.random.RandomState(42).normal(loc=0.0, scale=0.1, size=(4, 3))
    np.testing.assert_array_equal(layer.weights, expected)


def test_enabling_bias_does_not_change_the_weights() -> None:
    """
    Turning on the bias does not change the initial weights.

    Notes
    -----
    The bias starts at zero and draws no random numbers. Two layers built
    from the same seed, without and with bias, must have identical
    weights, and the bias must be a (1, 3) array of zeros.
    """
    plain = Dense(4, 3, random_state=np.random.RandomState(42))
    biased = Dense(4, 3, random_state=np.random.RandomState(42), use_bias=True)
    np.testing.assert_array_equal(plain.weights, biased.weights)
    np.testing.assert_array_equal(
        biased.bias,  # type: ignore[arg-type]  # use_bias=True, so bias is an array
        np.zeros((1, 3)),
    )


@pytest.mark.parametrize(
    "init,expected_std",
    [
        ("he", np.sqrt(2.0 / 400)),
        ("xavier", np.sqrt(2.0 / (400 + 300))),
        ("normal", 0.1),
    ],
)
def test_init_standard_deviation(init: str, expected_std: float) -> None:
    """
    Each initialization scheme draws weights with the expected spread.

    Parameters
    ----------
    init : str
        Initialization scheme: "he", "xavier" or "normal".
    expected_std : float
        Target standard deviation for a 400 -> 300 layer:
        sqrt(2 / 400) for He, sqrt(2 / (400 + 300)) for Xavier, and 0.1
        for "normal".

    Notes
    -----
    With 120,000 weights the sample statistics are close to the true
    ones, so the test asserts that the mean is within 5% of expected_std
    from zero and the standard deviation is within 2% of expected_std.

    The sample mean has standard error expected_std / sqrt(120000) =
    0.0029 * expected_std, so 5% is about 17 standard errors. The sample
    standard deviation has relative standard error 1 / sqrt(2 * 120000)
    = 0.0020, so 2% is about 10.
    """
    layer = Dense(400, 300, random_state=np.random.RandomState(0), init=init)
    assert abs(layer.weights.mean()) < 0.05 * expected_std
    assert layer.weights.std() == pytest.approx(expected_std, rel=0.02)


def test_unknown_init_raises() -> None:
    """
    An unknown initialization name raises ValueError.

    Notes
    -----
    `Dense(3, 2, init="uniform")` must raise ValueError.
    """
    with pytest.raises(ValueError):
        Dense(3, 2, init="uniform")


# ------------------------------------------------------------------
# Optimizers update every parameter
# ------------------------------------------------------------------


@pytest.mark.parametrize("optimizer_name", ["sgd", "momentum", "adabelief", "muon"])
def test_optimizer_updates_weights_and_bias(optimizer_name: str) -> None:
    """
    Every optimizer updates both the weights and the bias of a Dense layer.

    Parameters
    ----------
    optimizer_name : str
        Optimizer under test: "sgd", "momentum", "adabelief" or "muon".

    Notes
    -----
    After one forward and backward pass (upstream gradient of ones), a
    single `update(layer)` at learning rate 0.1 must change both the
    weights and the bias.
    """
    layer = Dense(3, 2, random_state=np.random.RandomState(0), use_bias=True)
    X = np.random.RandomState(1).normal(size=(5, 3))
    layer.forward(X)
    layer.backward(np.ones((5, 2)))

    weights_before = layer.weights.copy()
    bias_before = layer.bias.copy()  # type: ignore[union-attr]  # use_bias=True
    get_optimizer(optimizer_name, 0.1).update(layer)

    assert not np.allclose(layer.weights, weights_before)
    assert not np.allclose(layer.bias, bias_before)  # type: ignore[arg-type]  # use_bias=True


def test_sgd_bias_update_is_learning_rate_times_gradient() -> None:
    """
    SGD updates the bias by b <- b - learning_rate * dL/db.

    Notes
    -----
    After a forward and backward pass, the expected new bias
    b - 0.1 * dbias is computed, and after one SGD update at learning
    rate 0.1 the bias must equal it.
    """
    layer = Dense(3, 2, random_state=np.random.RandomState(0), use_bias=True)
    layer.forward(np.random.RandomState(1).normal(size=(5, 3)))
    layer.backward(np.random.RandomState(2).normal(size=(5, 2)))
    expected = layer.bias - 0.1 * layer.dbias  # type: ignore[operator]  # use_bias=True

    get_optimizer("sgd", 0.1).update(layer)

    np.testing.assert_allclose(layer.bias, expected)  # type: ignore[arg-type]  # use_bias=True


# ------------------------------------------------------------------
# Early-stopping checkpoints include the bias
# ------------------------------------------------------------------


def test_model_state_round_trip_includes_bias() -> None:
    """
    Trainer checkpoints save and restore the bias as well as the weights.

    Notes
    -----
    For a network whose Dense layers all have biases, the state is saved
    with `_get_model_state`, every weight and bias is shifted by 1.0, and
    `_set_model_state` must restore both to the saved values.
    """
    network = NeuralNetwork(random_seed=0)
    network.build_from_config(
        {
            "input_dimension": 3,
            "layers": [
                {"type": "dense", "units": 4, "activation": "relu", "use_bias": True},
                {"type": "dense", "units": 1, "activation": "sigmoid", "use_bias": True},
            ],
        }
    )
    trainer = Trainer(network, get_loss("bce"), get_optimizer("sgd", 0.1), "classification")

    saved = trainer._get_model_state()
    # get_trainable_layers() is typed list[Dense | BatchNorm], but this network has only
    # Dense layers, all with use_bias=True, hence the ignores below.
    for layer in network.get_trainable_layers():
        layer.weights += 1.0  # type: ignore[union-attr]  # Dense only
        layer.bias += 1.0  # type: ignore[union-attr, operator]  # Dense with bias only
    trainer._set_model_state(saved)

    for layer, state in zip(network.get_trainable_layers(), saved):
        np.testing.assert_array_equal(
            layer.weights,  # type: ignore[union-attr]  # Dense only
            state["params"]["weights"],
        )
        np.testing.assert_array_equal(
            layer.bias,  # type: ignore[union-attr, arg-type]  # Dense with bias only
            state["params"]["bias"],
        )


# ------------------------------------------------------------------
# Config validation
# ------------------------------------------------------------------


def make_config(**layer_extra: Any) -> dict[str, Any]:
    """
    Build a minimal valid classification config for ConfigLoader.validate.

    Parameters
    ----------
    **layer_extra : Any
        Extra keys merged into the single layer of architecture "A", for
        example `use_bias=True` or `init="he"`.

    Returns
    -------
    dict[str, Any]
        Config with every required top-level, preprocessing and
        experiments key, and one architecture "A" made of a single 1-unit
        sigmoid Dense layer plus `layer_extra`.

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
            "A": [{"type": "dense", "units": 1, "activation": "sigmoid", **layer_extra}]
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


@pytest.mark.parametrize(
    "layer_extra",
    [
        {},
        {"use_bias": True},
        {"use_bias": False},
        {"init": "he"},
        {"init": "Xavier"},
    ],
)
def test_config_accepts_valid_bias_and_init(layer_extra: dict[str, Any]) -> None:
    """
    The config validator accepts valid "use_bias" and "init" settings.

    Parameters
    ----------
    layer_extra : dict[str, Any]
        Layer settings to add: none, use_bias True or False, init "he",
        or init "Xavier" (the init name is case-insensitive).

    Notes
    -----
    `validate` must return True.
    """
    assert ConfigLoader().validate(make_config(**layer_extra))


@pytest.mark.parametrize(
    "layer_extra",
    [
        {"use_bias": "yes"},
        {"use_bias": 1},
        {"init": "uniform"},
    ],
)
def test_config_rejects_invalid_bias_and_init(layer_extra: dict[str, Any]) -> None:
    """
    The config validator rejects invalid "use_bias" and "init" settings.

    Parameters
    ----------
    layer_extra : dict[str, Any]
        Invalid layer settings: use_bias "yes" (a string), use_bias 1 (an
        integer, not a boolean), or init "uniform" (unsupported scheme).

    Notes
    -----
    `validate` must raise ValueError.
    """
    with pytest.raises(ValueError):
        ConfigLoader().validate(make_config(**layer_extra))
