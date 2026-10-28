"""
Tests for global-norm gradient clipping and its configuration.
"""

import math
from typing import Any

import numpy as np
import pytest

from nn_numpy.config_loader import ConfigLoader
from nn_numpy.modeling.trainer import Trainer
from nn_numpy.nn.losses import get_loss
from nn_numpy.nn.network import NeuralNetwork
from nn_numpy.nn.optimizers import get_optimizer


def build_network_with_gradients(seed: int = 0) -> NeuralNetwork:
    """
    Build a small network with bias and batch norm and run one backward pass.

    Parameters
    ----------
    seed : int, default=0
        Seed for the network and the random batch.

    Returns
    -------
    NeuralNetwork
        A 3 -> 4 (bias, batch norm, ReLU) -> 1 (linear) network in training
        mode, whose layers hold the gradients of an MSE loss on a random
        batch of 8 samples with targets scaled by 50 (so the gradients are
        large).

    Notes
    -----
    Processing:
    1. Build the network from a config with use_bias and batch_norm on the
       hidden layer.
    2. Forward a random batch, compute the MSE gradient and backpropagate.

    The factor 50 keeps the unclipped global gradient norm above the caps
    the tests clip at (1.0 and 2.0); test_clipping_scales_large_gradients_to_the_cap
    asserts it.
    """
    rng = np.random.RandomState(seed)
    network = NeuralNetwork(random_seed=seed)
    network.build_from_config(
        {
            "input_dimension": 3,
            "layers": [
                {
                    "type": "dense",
                    "units": 4,
                    "activation": "relu",
                    "use_bias": True,
                    "batch_norm": True,
                },
                {"type": "dense", "units": 1, "activation": "linear", "use_bias": True},
            ],
        }
    )
    network.train()
    X = rng.normal(size=(8, 3))
    y = 50.0 * rng.normal(size=(8, 1))
    loss = get_loss("mse")
    network.backward(loss.backward(y, network.forward(X)))
    return network


def all_gradients(network: NeuralNetwork) -> list[np.ndarray]:
    """
    List every gradient array of every trainable layer.

    Parameters
    ----------
    network : NeuralNetwork
        Network after a backward pass.

    Returns
    -------
    list of numpy.ndarray
        The arrays returned by get_grads(), in layer order.

    Notes
    -----
    Processing:
    1. Collect get_grads().values() of each trainable layer.
    """
    return [g for layer in network.get_trainable_layers() for g in layer.get_grads().values()]


def test_gradient_norm_covers_every_parameter() -> None:
    """
    gradient_norm is the L2 norm over weights, biases and batch-norm gamma / beta.

    Notes
    -----
    The network has Dense weights and biases and BatchNorm gamma and beta,
    so 6 gradient arrays. The norm must equal sqrt(sum of all squared
    entries) computed by hand.

    6 = 2 Dense layers x (weights, bias) + 1 BatchNorm x (gamma, beta).
    """
    network = build_network_with_gradients()
    grads = all_gradients(network)
    assert len(grads) == 6
    expected = math.sqrt(sum(float(np.sum(g**2)) for g in grads))
    assert network.gradient_norm() == pytest.approx(expected)


def test_clipping_scales_large_gradients_to_the_cap() -> None:
    """
    clip_gradients rescales an over-long gradient to exactly max_norm, keeping its direction.

    Notes
    -----
    The unclipped norm is well above 1.0. After clipping at 1.0 the norm
    must be 1.0, every array must be the original times the same factor,
    and the return value must be the norm before clipping.

    The common factor is max_norm / norm = 1.0 / norm_before, so each
    clipped array is old / norm_before.
    """
    network = build_network_with_gradients()
    before = [g.copy() for g in all_gradients(network)]
    norm_before = network.gradient_norm()
    assert norm_before > 1.0

    returned = network.clip_gradients(1.0)

    assert returned == pytest.approx(norm_before)
    assert network.gradient_norm() == pytest.approx(1.0)
    for old, new in zip(before, all_gradients(network)):
        np.testing.assert_allclose(new, old / norm_before)


def test_clipping_leaves_small_gradients_untouched() -> None:
    """
    Gradients whose norm is below the cap are not changed at all.

    Notes
    -----
    With a cap far above the gradient norm, every gradient array must be
    bit-for-bit unchanged. The cap 1e12 carries no meaning beyond being
    far above any norm this small network produces.
    """
    network = build_network_with_gradients()
    before = [g.copy() for g in all_gradients(network)]
    network.clip_gradients(1e12)
    for old, new in zip(before, all_gradients(network)):
        np.testing.assert_array_equal(new, old)


def test_clipped_sgd_steps_are_bounded() -> None:
    """
    With plain SGD, a clipped update moves the parameters by at most lr * max_norm.

    Notes
    -----
    SGD subtracts lr * gradient from every parameter, so after clipping at
    max_norm the global norm of the parameter change is exactly
    lr * min(norm, max_norm). Checked with lr 0.5 and max_norm 2.

    The norm is above 2 (targets scaled by 50), so the step is
    0.5 * min(norm, 2) = 0.5 * 2 = 1.0. Any learning rate and any cap below
    the norm would do; these two only make the product easy to check.
    """
    network = build_network_with_gradients()
    params_before = [
        p.copy() for layer in network.get_trainable_layers() for p in layer.get_params().values()
    ]
    network.clip_gradients(2.0)
    optimizer = get_optimizer("sgd", 0.5)
    for layer in network.get_trainable_layers():
        optimizer.update(layer)
    params_after = [
        p for layer in network.get_trainable_layers() for p in layer.get_params().values()
    ]

    step = math.sqrt(sum(float(np.sum((a - b) ** 2)) for a, b in zip(params_after, params_before)))
    assert step == pytest.approx(0.5 * 2.0)


def test_clipping_prevents_a_divergence() -> None:
    """
    A run that diverges without clipping stays finite with a gradient-norm cap.

    Notes
    -----
    The same setup as the batch-norm divergence test: SGD at learning rate
    10 on targets scaled by 1000. Unclipped, the run diverges and records a
    huge or non-finite peak gradient norm. With max_grad_norm 1, every
    step is bounded, so all 20 epochs run with finite losses. Both runs
    record one gradient norm per epoch.
    """
    rng = np.random.RandomState(8)
    X = rng.normal(size=(100, 3))
    y = 1000.0 * rng.normal(size=(100, 1))

    def fit(max_grad_norm: float | None) -> dict[str, Any]:
        """
        Train a fresh 3 -> 8 (ReLU) -> 1 network on the test's data.

        Parameters
        ----------
        max_grad_norm : float or None
            Global gradient-norm cap; None trains without clipping.

        Returns
        -------
        dict[str, Any]
            The history returned by Trainer.fit (keys used here:
            "diverged", "grad_norm", "epochs_ran", "train_loss", "val_loss").

        Notes
        -----
        Processing:
        1. Build the network with seed 0, so both runs start from the same
           weights and differ only in the cap.
        2. Train with SGD at learning rate 10 for 20 epochs on the first
           80 rows, validating on the last 20.
        """
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
        trainer = Trainer(
            network,
            get_loss("mse"),
            get_optimizer("sgd", 10.0),
            "regression",
            max_grad_norm=max_grad_norm,
        )
        # The unclipped run overflows by design; silence NumPy's overflow warnings.
        with np.errstate(all="ignore"):
            return trainer.fit(
                X[:80], y[:80], X[80:], y[80:], epochs=20, batch_size=16, verbose=False
            )

    unclipped = fit(None)
    clipped = fit(1.0)

    assert unclipped["diverged"]
    assert not math.isfinite(max(unclipped["grad_norm"])) or max(unclipped["grad_norm"]) > 1e6
    assert not clipped["diverged"]
    assert clipped["epochs_ran"] == 20
    assert all(math.isfinite(v) for v in clipped["train_loss"] + clipped["val_loss"])
    assert len(unclipped["grad_norm"]) == unclipped["epochs_ran"]
    assert len(clipped["grad_norm"]) == 20


# ------------------------------------------------------------------
# Config: architectures with a gradient-norm cap
# ------------------------------------------------------------------


def make_config(architecture: list[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
    """
    Build a minimal valid regression config with one architecture "A".

    Parameters
    ----------
    architecture : list or dict
        The entry for architecture "A": a layer list, or a dict with
        "layers" and options such as "max_grad_norm".

    Returns
    -------
    dict[str, Any]
        Config with every required key and the given architecture.

    Notes
    -----
    Processing:
    1. Return a fresh dictionary literal each call.
    """
    return {
        "task_type": "regression",
        "input_dimension": 2,
        "loss": "mse",
        "preprocessing": {"enabled": True, "scale_features": True},
        "architectures": {"A": architecture},
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


LAYERS = [{"type": "dense", "units": 1, "activation": "linear"}]


def test_architecture_with_a_cap_is_split_into_layers_and_options() -> None:
    """
    A dict architecture is validated and normalized to a layer list plus options.

    Notes
    -----
    After normalization, config["architectures"]["A"] must be the plain
    layer list (what every consumer expects) and the cap must be in
    config["architecture_options"]["A"]. A plain list architecture gets no
    options.
    """
    config = make_config({"layers": LAYERS, "max_grad_norm": 20})
    assert ConfigLoader().validate(config)
    normalized = ConfigLoader.normalize_architectures(config)
    assert normalized["architectures"]["A"] == LAYERS
    assert normalized["architecture_options"]["A"] == {"max_grad_norm": 20}

    plain = ConfigLoader.normalize_architectures(make_config(LAYERS))
    assert plain["architectures"]["A"] == LAYERS
    assert plain["architecture_options"] == {}


@pytest.mark.parametrize(
    "entry",
    [
        {"layers": LAYERS, "max_grad_norm": 0},
        {"layers": LAYERS, "max_grad_norm": -1.0},
        {"layers": LAYERS, "max_grad_norm": "20"},
        {"layers": LAYERS, "max_grad_norm": True},
        {"layers": LAYERS, "clip": 20},
        {"max_grad_norm": 20},
    ],
)
def test_invalid_architecture_options_are_rejected(entry: dict[str, Any]) -> None:
    """
    Invalid caps, unknown keys and a missing layer list raise ValueError.

    Parameters
    ----------
    entry : dict
        An invalid architecture entry.

    Notes
    -----
    max_grad_norm must be a positive number (not a boolean or string),
    only "layers" and "max_grad_norm" are allowed, and "layers" is
    required.
    """
    with pytest.raises(ValueError):
        ConfigLoader().validate(make_config(entry))
