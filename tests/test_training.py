"""
Smoke tests for the optimizers and the training loop on tiny synthetic problems.
"""

import numpy as np
import pytest

from nn_numpy.modeling.trainer import Trainer
from nn_numpy.nn.losses import get_loss
from nn_numpy.nn.network import NeuralNetwork
from nn_numpy.nn.optimizers import get_optimizer


def make_classification_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a linearly separable 2-feature binary classification problem.

    Parameters
    ----------
    n : int, default=200
        Number of samples.
    seed : int, default=0
        Seed for the random generator.

    Returns
    -------
    X : numpy.ndarray of shape (n, 2), dtype float64
        Standard-normal features.
    y : numpy.ndarray of shape (n, 1), dtype float64
        Labels: 1.0 where X[:, 0] + X[:, 1] > 0, else 0.0.

    Notes
    -----
    Processing:
    1. Draw X from a standard normal distribution.
    2. Label each row by the sign of X[:, 0] + X[:, 1] and reshape the
       labels to a column.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    return X, y


def build_classifier() -> NeuralNetwork:
    """
    Build a small 2 -> 8 -> 1 binary classifier.

    Parameters
    ----------
    None
        The architecture and seed (42) are fixed.

    Returns
    -------
    NeuralNetwork
        Network with layers Dense(2, 8), ReLU, Dense(8, 1), Sigmoid, in
        training mode.

    Notes
    -----
    Processing:
    1. Create `NeuralNetwork(random_seed=42)`.
    2. Build it from a two-layer config without bias, batch norm or
       dropout.
    """
    network = NeuralNetwork(random_seed=42)
    network.build_from_config(
        {
            "input_dimension": 2,
            "layers": [
                {"type": "dense", "units": 8, "activation": "relu"},
                {"type": "dense", "units": 1, "activation": "sigmoid"},
            ],
        }
    )
    return network


@pytest.mark.parametrize(
    "optimizer_name,learning_rate",
    [
        ("sgd", 0.5),
        ("momentum", 0.5),
        ("adabelief", 0.01),
        ("muon", 0.05),
    ],
)
def test_optimizer_reduces_loss(optimizer_name: str, learning_rate: float) -> None:
    """
    Each optimizer lowers the training loss on a simple problem.

    Parameters
    ----------
    optimizer_name : str
        Optimizer under test: "sgd", "momentum", "adabelief" or "muon".
    learning_rate : float
        Learning rate for that optimizer (0.5 for SGD and momentum, 0.01
        for AdaBelief, 0.05 for Muon).

    Notes
    -----
    The classifier takes 50 full-batch steps (forward, backward, update
    every trainable layer) on 200 samples. The final BCE must be below
    80% of the initial BCE.
    """
    X, y = make_classification_data()
    network = build_classifier()
    loss = get_loss("bce")
    optimizer = get_optimizer(optimizer_name, learning_rate)

    initial_loss = loss.forward(y, network.forward(X))

    for _ in range(50):
        y_pred = network.forward(X)
        network.backward(loss.backward(y, y_pred))
        for layer in network.get_trainable_layers():
            optimizer.update(layer)

    final_loss = loss.forward(y, network.forward(X))
    assert final_loss < 0.8 * initial_loss


def test_unknown_optimizer_raises() -> None:
    """
    Asking for an unsupported optimizer name raises ValueError.

    Notes
    -----
    `get_optimizer("rmsprop", 0.01)` must raise ValueError.
    """
    with pytest.raises(ValueError):
        get_optimizer("rmsprop", 0.01)


def test_trainer_fits_and_evaluates() -> None:
    """
    Trainer.fit records a decreasing loss history and Trainer.evaluate reports a good test BCE.

    Notes
    -----
    The classifier is trained with AdaBelief for 20 epochs (no early
    stopping) on 200 samples, validated on 50 and tested on 50. Asserted:
    all 20 epochs ran and were recorded, the last training loss is below
    the first, the test metric equals the test loss (both are BCE for
    classification), and the test BCE is below 0.3.
    """
    X, y = make_classification_data(n=300)
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:250], y[200:250]
    X_test, y_test = X[250:], y[250:]

    trainer = Trainer(
        network=build_classifier(),
        loss_fn=get_loss("bce"),
        optimizer=get_optimizer("adabelief", 0.01),
        task_type="classification",
    )
    history = trainer.fit(X_train, y_train, X_val, y_val, epochs=20, batch_size=16, verbose=False)

    assert history["epochs_ran"] == 20
    assert len(history["train_loss"]) == 20
    assert history["train_loss"][-1] < history["train_loss"][0]

    results = trainer.evaluate(X_test, y_test)
    # The evaluation metric for classification is BCE, the same quantity as the loss
    assert results["test_metric"] == pytest.approx(results["test_loss"])
    assert results["test_metric"] < 0.3  # well below ln(2) ~= 0.693, the BCE of a coin flip


def test_a_run_that_reaches_max_epochs_records_why_it_ended() -> None:
    """
    A run that uses every epoch reports stop_reason "max_epochs" and no overfitting.

    Notes
    -----
    Early stopping is on, but patience (100) is longer than the 10
    requested epochs, so training can only end at the epoch limit.
    Asserted: stop_reason is "max_epochs", overfitting_epochs is 0, and
    the run is neither diverged nor stopped early.
    """
    X, y = make_classification_data(n=150)
    trainer = Trainer(
        network=build_classifier(),
        loss_fn=get_loss("bce"),
        optimizer=get_optimizer("adabelief", 0.01),
        task_type="classification",
        early_stopping=True,
        patience=100,
    )
    history = trainer.fit(
        X[:100], y[:100], X[100:], y[100:], epochs=10, batch_size=16, verbose=False
    )

    assert history["stop_reason"] == "max_epochs"
    assert history["overfitting_epochs"] == 0
    assert not history["diverged"] and not history["stopped_early"]


def test_early_stopping_records_the_overfitting_signature() -> None:
    """
    A run whose validation loss rises while training loss falls is stopped
    as overfitting.

    Notes
    -----
    The validation set has the training inputs with flipped labels, so
    every step that lowers the training loss raises the validation loss:
    the best checkpoint is epoch 1, and the run must stop after patience
    (3) epochs without validation improvement (no minimum-epoch guard).
    Asserted: stop_reason is "early_stopping", best_epoch is 1, 4 epochs
    ran, and overfitting_epochs counts the 3 epochs in which training
    improved while validation did not.

    By hand: epochs ran = best epoch 1 + patience 3 = 4.
    """
    X, y = make_classification_data(n=100)
    trainer = Trainer(
        network=build_classifier(),
        loss_fn=get_loss("bce"),
        optimizer=get_optimizer("adabelief", 0.01),
        task_type="classification",
        early_stopping=True,
        patience=3,
        min_delta=0.0,
    )
    history = trainer.fit(X, y, X, 1.0 - y, epochs=50, batch_size=16, verbose=False)

    assert history["stop_reason"] == "early_stopping"
    assert history["best_epoch"] == 1
    assert history["epochs_ran"] == 4
    assert history["overfitting_epochs"] == 3
