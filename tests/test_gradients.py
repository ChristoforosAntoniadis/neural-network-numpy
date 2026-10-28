"""
Numerical gradient checks for the NumPy backpropagation.

Every analytic gradient (activations, losses, and full networks) is compared
against a central finite-difference estimate:

    dL/dw ~= (L(w + eps) - L(w - eps)) / (2 * eps)
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from nn_numpy.nn.activations import get_activation
from nn_numpy.nn.losses import get_loss
from nn_numpy.nn.network import NeuralNetwork

# Step of the central differences. Their truncation error is of order EPS^2 = 1e-12, and
# their rounding error of order 2.2e-16 * |f| / EPS = 2e-10 for |f| ~ 1 (2.2e-16 is the
# float64 machine epsilon). The sum of the two is smallest near EPS = (2.2e-16)^(1/3) = 6e-6,
# so EPS = 1e-6 sits close to the optimum.
EPS = 1e-6
# Tolerances for analytic vs. numerical gradients. The finite-difference error above is
# about 1e-10, so RTOL = 1e-5 leaves five orders of magnitude of margin, while a wrong
# gradient (a missing factor or a flipped sign) has a relative error of order 1. ATOL
# handles entries at or near zero, where a relative test is meaningless; 1e-8 is still
# 100 times the finite-difference error.
RTOL = 1e-5
ATOL = 1e-8


def numerical_grad(f: Callable[[], float], x: np.ndarray) -> np.ndarray:
    """
    Estimate the gradient of a scalar function by central finite differences.

    Parameters
    ----------
    f : collections.abc.Callable[[], float]
        Function with no arguments that returns a scalar (float). It must
        read `x` (directly or through an object that holds it), so that
        changing `x` in place changes its result.
    x : numpy.ndarray of any shape, dtype float64
        Array to differentiate with respect to. It is modified in place
        during the computation and restored afterwards.

    Returns
    -------
    numpy.ndarray of the same shape as x, dtype float64
        Estimated gradient df/dx.

    Notes
    -----
    Processing:
    1. Create a zero array `grad` shaped like x.
    2. For each element x[idx]:
       a. set it to original + EPS and evaluate f;
       b. set it to original - EPS and evaluate f;
       c. restore the original value;
       d. store (f_plus - f_minus) / (2 * EPS) in grad[idx].
    3. Return grad.

    The central difference has error of order EPS^2, which is much more
    accurate than the one-sided (f(x + eps) - f(x)) / eps. It costs two
    evaluations of f per element, so it is only used on small arrays.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        # x[idx] is a scalar copy, so writing it back below restores x exactly.
        original = x[idx]
        x[idx] = original + EPS
        f_plus = f()
        x[idx] = original - EPS
        f_minus = f()
        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * EPS)
    return grad


def away_from_zero(
    rng: np.random.RandomState, shape: tuple[int, ...], margin: float = 0.1
) -> np.ndarray:
    """
    Draw random values whose absolute value is at least `margin`.

    Parameters
    ----------
    rng : numpy.random.RandomState
        Generator used for the draws.
    shape : tuple of int
        Shape of the returned array.
    margin : float, default=0.1
        Smallest allowed absolute value.

    Returns
    -------
    numpy.ndarray of shape `shape`, dtype float64
        Values with |x| in [margin, 2.0) and a random sign.

    Notes
    -----
    Processing:
    1. Draw magnitudes uniformly from [margin, 2.0).
    2. Multiply each by a randomly chosen sign, -1.0 or +1.0.

    Why: ReLU has a kink at 0 where its derivative jumps. Keeping inputs
    away from 0 means the finite-difference window (plus or minus EPS)
    never crosses the kink, so the numerical gradient is valid. The
    default margin of 0.1 is 1e5 times EPS. With |x| < 2.0, sigmoid and
    tanh stay out of their flat tails (tanh'(2) = 1 - tanh(2)^2 = 0.07),
    so the compared gradients are not vanishingly small.
    """
    x = rng.uniform(margin, 2.0, size=shape)
    return x * rng.choice([-1.0, 1.0], size=shape)


@pytest.mark.parametrize("name", ["relu", "sigmoid", "tanh", "linear"])
def test_activation_backward_matches_numerical(name: str) -> None:
    """
    Each activation's backward pass matches the finite-difference gradient.

    Parameters
    ----------
    name : str
        Activation under test: "relu", "sigmoid", "tanh" or "linear".

    Notes
    -----
    The scalar loss is L = sum(upstream * activation(x)) for a random
    `upstream`, so dL/dy = upstream. The analytic gradient from
    `backward(upstream)` is compared with `numerical_grad` on x (drawn
    away from 0) at rtol=RTOL, atol=ATOL. A fresh activation object is
    used for the numerical side so the cached state of the first one is
    not disturbed.
    """
    rng = np.random.RandomState(0)
    x = away_from_zero(rng, (5, 3))
    upstream = rng.normal(size=(5, 3))  # arbitrary dL/dy, so L = sum(upstream * y)

    activation = get_activation(name)
    activation.forward(x)
    analytic = activation.backward(upstream)

    numeric = numerical_grad(lambda: np.sum(upstream * get_activation(name).forward(x)), x)

    np.testing.assert_allclose(analytic, numeric, rtol=RTOL, atol=ATOL)


def test_mse_backward_matches_numerical() -> None:
    """
    MSE loss's gradient with respect to the predictions matches finite differences.

    Notes
    -----
    For random 8x1 targets and predictions, `backward(y_true, y_pred)` is
    compared with `numerical_grad` of `forward(y_true, y_pred)` with
    respect to y_pred.
    """
    rng = np.random.RandomState(1)
    y_true = rng.normal(size=(8, 1))
    y_pred = rng.normal(size=(8, 1))
    loss = get_loss("mse")

    analytic = loss.backward(y_true, y_pred)
    numeric = numerical_grad(lambda: loss.forward(y_true, y_pred), y_pred)

    np.testing.assert_allclose(analytic, numeric, rtol=RTOL, atol=ATOL)


def test_bce_backward_matches_numerical() -> None:
    """
    BCE loss's gradient with respect to the predictions matches finite differences.

    Notes
    -----
    For random 0/1 targets and predictions in [0.05, 0.95), away from
    the region where the loss clips its inputs, `backward(y_true,
    y_pred)` is compared with `numerical_grad` of `forward(y_true,
    y_pred)` with respect to y_pred.
    """
    rng = np.random.RandomState(2)
    y_true = rng.randint(0, 2, size=(8, 1)).astype(float)
    y_pred = rng.uniform(0.05, 0.95, size=(8, 1))  # stay clear of the clipping region
    loss = get_loss("bce")

    analytic = loss.backward(y_true, y_pred)
    numeric = numerical_grad(lambda: loss.forward(y_true, y_pred), y_pred)

    np.testing.assert_allclose(analytic, numeric, rtol=RTOL, atol=ATOL)


NETWORK_CASES: list[tuple[str, str, list[tuple[str, int]], str, dict[str, Any]]] = [
    # (id, loss, hidden layers, output activation, extra settings for every Dense layer)
    ("A1-style_bce", "bce", [("sigmoid", 6)], "sigmoid", {}),
    ("A2-style_bce", "bce", [("relu", 6), ("relu", 6), ("relu", 6)], "sigmoid", {}),
    ("A1-style_mse", "mse", [("sigmoid", 6)], "linear", {}),
    ("A2-style_mse", "mse", [("relu", 6), ("relu", 6), ("relu", 6)], "linear", {}),
    ("tanh_mse", "mse", [("tanh", 5), ("tanh", 4)], "linear", {}),
    ("A1-style_bce_bias", "bce", [("sigmoid", 6)], "sigmoid", {"use_bias": True}),
    (
        "A2-style_bce_bias_he",
        "bce",
        [("relu", 6), ("relu", 6), ("relu", 6)],
        "sigmoid",
        {"use_bias": True, "init": "he"},
    ),
    (
        "A2-style_mse_bias_he",
        "mse",
        [("relu", 6), ("relu", 6), ("relu", 6)],
        "linear",
        {"use_bias": True, "init": "he"},
    ),
    (
        "tanh_mse_bias_xavier",
        "mse",
        [("tanh", 5), ("tanh", 4)],
        "linear",
        {"use_bias": True, "init": "xavier"},
    ),
    # Batch norm in training mode: gradients flow through the batch statistics.
    (
        "A2-style_bce_batchnorm",
        "bce",
        [("relu", 6), ("relu", 6), ("relu", 6)],
        "sigmoid",
        {"batch_norm": True},
    ),
    (
        "A2-style_mse_batchnorm",
        "mse",
        [("relu", 6), ("relu", 6), ("relu", 6)],
        "linear",
        {"batch_norm": True},
    ),
]


@pytest.mark.parametrize(
    "loss_name,hidden,output_activation,extra",
    [c[1:] for c in NETWORK_CASES],
    ids=[c[0] for c in NETWORK_CASES],
)
def test_network_weight_gradients_match_numerical(
    loss_name: str, hidden: list[tuple[str, int]], output_activation: str, extra: dict[str, Any]
) -> None:
    """
    Backprop gives the correct gradient for every trainable parameter of a full network.

    Checks every trainable parameter: weights, biases, and batch-norm
    gamma / beta.

    Parameters
    ----------
    loss_name : str
        Loss to train with: "bce" or "mse".
    hidden : list of tuple of (str, int)
        Hidden layers as (activation name, number of units) pairs.
    output_activation : str
        Activation of the single output unit ("sigmoid" or "linear").
    extra : dict[str, Any]
        Extra settings applied to every Dense layer config, such as
        {"use_bias": True, "init": "he"} or {"batch_norm": True}.

    Notes
    -----
    A network with input dimension 4 is built from the case and run on a
    batch of 10 random samples (0/1 targets for BCE, normal targets for
    MSE). After one forward and backward pass the analytic gradients are
    copied. For each trainable layer, the test asserts that it exposes
    exactly the expected parameters ({"gamma", "beta"} for BatchNorm;
    {"weights"} or {"weights", "bias"} for Dense), then compares each
    parameter's gradient with `numerical_grad` of the full-network loss
    at rtol=1e-4, atol=1e-7. The network stays in training mode, so
    batch-norm gradients include the path through the batch statistics.
    """
    rng = np.random.RandomState(3)
    input_dim, batch = 4, 10

    layers = [
        {"type": "dense", "units": units, "activation": act, **extra} for act, units in hidden
    ]
    layers.append({"type": "dense", "units": 1, "activation": output_activation, **extra})

    network = NeuralNetwork(random_seed=42)
    network.build_from_config({"input_dimension": input_dim, "layers": layers})
    loss = get_loss(loss_name)

    X = rng.normal(size=(batch, input_dim))
    if loss_name == "bce":
        y = rng.randint(0, 2, size=(batch, 1)).astype(float)
    else:
        y = rng.normal(size=(batch, 1))

    # Analytic gradients via backprop
    y_pred = network.forward(X)
    network.backward(loss.backward(y, y_pred))
    dense_layers = network.get_trainable_layers()
    analytic = [
        {name: g.copy() for name, g in layer.get_grads().items()} for layer in dense_layers
    ]

    def loss_value() -> float:
        """
        Return the loss of a fresh forward pass with the current parameters.

        Returns
        -------
        float
            The loss of `network` on (X, y); `numerical_grad` perturbs the
            parameters in place between calls.
        """
        return loss.forward(y, network.forward(X))

    # Compare each parameter's analytic gradient with a numerical estimate,
    # after checking the layer exposes exactly the parameters expected.
    dense_params = {"weights", "bias"} if extra.get("use_bias") else {"weights"}
    for layer, analytic_grads in zip(dense_layers, analytic):
        expected_params = (
            {"gamma", "beta"} if type(layer).__name__ == "BatchNorm" else dense_params
        )
        assert set(layer.get_params()) == expected_params
        for name, param in layer.get_params().items():
            numeric_grad = numerical_grad(loss_value, param)
            np.testing.assert_allclose(analytic_grads[name], numeric_grad, rtol=1e-4, atol=1e-7)
