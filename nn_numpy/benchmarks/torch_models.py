"""
PyTorch versions of the network for the library comparison.

Input: an architecture name (e.g. "A2-bn") whose layers are read from the
main experiment configs, a grid of optimizers / learning rates / batch
sizes, training settings from ``configs/benchmark_experiments.json``, and
one seed's data split.
Output: one result record per (optimizer, learning rate, batch size), with
validation, test and training metrics, training time and loss histories.

The layer structure mirrors the NumPy network exactly (same units,
activations, bias and batch-norm placement), and everything else is what
PyTorch provides by default: nn.Linear's Kaiming-uniform weight
initialization, torch.optim's SGD / momentum / Adam, BatchNorm1d's own
momentum and epsilon, and float32 computation. PyTorch has no built-in
fit loop or early stopping, so the training loop is written out here, the
way PyTorch code usually is, with the same early-stopping rule as the
NumPy trainer.

Every model is scored with the project's own NumPy metric functions.
"""

from collections.abc import Iterable
import copy
import itertools
import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from nn_numpy.benchmarks.data import load_problem_config, with_dropout
from nn_numpy.nn.metrics import (
    accuracy,
    binary_cross_entropy,
    mean_squared_error,
    r2_score,
)

# Make PyTorch's operations deterministic, so a seed always gives the same
# result (an error is raised for any operation without a deterministic
# version; none is used here).
torch.use_deterministic_algorithms(True)

# PyTorch module for each activation name used in the configs.
ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "linear": nn.Identity,
}


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    List every combination of the values in a grid.

    Parameters
    ----------
    grid : dict of str to list
        Each key maps to the list of values to try, e.g.
        {"optimizer": ["sgd", "adam"], "learning_rate": [0.1, 0.001]}.

    Returns
    -------
    list of dict
        One dict per combination (the Cartesian product), with the keys in
        sorted order. Its keys are the grid's keys, each mapped to one value.

    Notes
    -----
    Processing:
    1. Sort the keys so the order of combinations is stable.
    2. Form the Cartesian product of the value lists with
       itertools.product.
    """
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def build_network(
    layer_configs: list[dict[str, Any]], input_dimension: int, seed: int
) -> nn.Sequential:
    """
    Build a PyTorch model with the same layer structure as a NumPy architecture.

    Parameters
    ----------
    layer_configs : list of dict
        The architecture's layers from the main config. Each has "units"
        (int), "activation" (str) and optionally "use_bias" (bool, default
        False), "batch_norm" (bool, default False) and "dropout" (float,
        default 0). The "init" key is ignored: PyTorch's default
        initialization is used.
    input_dimension : int
        Number of input features.
    seed : int
        Seed for weight initialization (torch.manual_seed).

    Returns
    -------
    torch.nn.Sequential
        The model in float32: for each layer, Linear -> (BatchNorm1d) ->
        activation -> (Dropout).

    Raises
    ------
    ValueError
        If an activation has no PyTorch equivalent in ACTIVATIONS.

    Notes
    -----
    Processing:
    1. Seed PyTorch's global generator, which nn.Linear uses for its
       initial weights.
    2. For each layer config, add a Linear layer (with or without bias),
       then BatchNorm1d if requested, then the activation ("linear" is
       nn.Identity), then Dropout if requested.
    """
    torch.manual_seed(seed)

    # Typed as Module, not Linear: BatchNorm1d, activations and Dropout go in too.
    modules: list[nn.Module] = []
    in_features = input_dimension
    for layer in layer_configs:
        if layer["activation"] not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation for PyTorch: {layer['activation']}")
        modules.append(nn.Linear(in_features, layer["units"], bias=layer.get("use_bias", False)))
        if layer.get("batch_norm", False):
            modules.append(nn.BatchNorm1d(layer["units"]))
        modules.append(ACTIVATIONS[layer["activation"]]())
        if layer.get("dropout", 0.0) > 0.0:
            modules.append(nn.Dropout(layer["dropout"]))
        in_features = layer["units"]
    return nn.Sequential(*modules)


class MuonWithAdam:
    """
    PyTorch's Muon for the weight matrices, paired with Adam for the rest.

    Attributes
    ----------
    muon : torch.optim.Muon
        Optimizer for every 2-D parameter (the nn.Linear weights).
    adam : torch.optim.Adam or None
        Optimizer for every other parameter (biases, batch-norm weight and
        bias), with the same learning rate; None when the model has only
        weight matrices (the bias-free A1 and A2).

    Notes
    -----
    torch.optim.Muon only accepts 2-D parameters, and PyTorch's
    documentation pairs it with an Adam-type optimizer for the others. The
    training loop only calls zero_grad() and step(), so this small wrapper
    is all it needs. Both optimizers keep PyTorch's defaults otherwise,
    including Muon's weight decay of 0.1.
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        learning_rate: float,
        weight_decay: float | None = None,
    ) -> None:
        """
        Split the parameters by dimension and create both optimizers.

        Parameters
        ----------
        parameters : iterable of torch.nn.Parameter
            The model's parameters.
        learning_rate : float
            Step size for both optimizers.
        weight_decay : float or None, default=None
            Weight decay for both optimizers. None keeps PyTorch's defaults
            (0.1 for Muon, none for Adam).

        Returns
        -------
        None
            Sets self.muon and self.adam.

        Notes
        -----
        Processing:
        1. Put 2-D parameters in the Muon group and the rest in the Adam group.
        2. Create torch.optim.Muon on its group, and torch.optim.Adam only if
           its group is not empty (PyTorch rejects an empty parameter list).
        """
        parameters = list(parameters)
        matrices = [p for p in parameters if p.ndim == 2]
        others = [p for p in parameters if p.ndim != 2]
        # Any: the dict is unpacked as keyword arguments into optimizers whose other
        # parameters have different types, so dict[str, float] would not match them.
        decay: dict[str, Any] = {} if weight_decay is None else {"weight_decay": weight_decay}
        self.muon = torch.optim.Muon(matrices, lr=learning_rate, **decay)
        self.adam = torch.optim.Adam(others, lr=learning_rate, **decay) if others else None

    def zero_grad(self) -> None:
        """
        Clear the gradients of every parameter.

        Returns
        -------
        None
        """
        self.muon.zero_grad()
        if self.adam is not None:
            self.adam.zero_grad()

    def step(self) -> None:
        """
        Take one optimizer step on every parameter.

        Returns
        -------
        None
            The parameters are updated in place.
        """
        self.muon.step()
        if self.adam is not None:
            self.adam.step()


def build_optimizer(
    name: str,
    parameters: Iterable[nn.Parameter],
    learning_rate: float,
    weight_decay: float | None = None,
) -> torch.optim.Optimizer | MuonWithAdam:
    """
    Create a torch.optim optimizer.

    Parameters
    ----------
    name : str
        "sgd", "momentum", "adam" or "muon".
    parameters : iterable of torch.nn.Parameter
        The model's parameters.
    learning_rate : float
        Step size.
    weight_decay : float or None, default=None
        Weight decay passed to the optimizer (an L2 penalty for SGD and
        Adam, decoupled for Muon). None keeps PyTorch's default (none, or
        0.1 for Muon); used by hyperparameter tuning.

    Returns
    -------
    torch.optim.Optimizer or MuonWithAdam
        SGD, SGD with momentum 0.9, Adam, or Muon paired with Adam (PyTorch
        defaults otherwise).

    Raises
    ------
    ValueError
        If the name is not one of the four.

    Notes
    -----
    Processing:
    1. Map the name to the torch.optim optimizer.

    PyTorch's momentum uses v = 0.9 * v + g and steps by lr * v, while the
    NumPy network uses an exponential moving average v = 0.9 * v + 0.1 * g,
    so its effective steps are about 10 times smaller for the same learning
    rate. Adam is PyTorch's closest built-in to the NumPy network's
    AdaBelief. For Muon, PyTorch's defaults differ from the NumPy network in
    two ways: weight decay 0.1 (the NumPy network has none) and bfloat16
    Newton-Schulz (the NumPy network uses float64).
    """
    # Any, as in MuonWithAdam: unpacked into optimizers with differently typed parameters.
    decay: dict[str, Any] = {} if weight_decay is None else {"weight_decay": weight_decay}
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, **decay)
    if name == "momentum":
        # 0.9 is the momentum coefficient the NumPy network uses (nn_numpy.nn.optimizers).
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, **decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, **decay)
    if name == "muon":
        return MuonWithAdam(parameters, learning_rate, weight_decay)
    raise ValueError(f"Unsupported PyTorch optimizer: {name}")


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """
    Predict with a PyTorch model in evaluation mode.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    X : numpy.ndarray of shape (n_samples, n_features)
        Inputs.

    Returns
    -------
    numpy.ndarray of shape (n_samples, 1), dtype float64
        The model's outputs.

    Notes
    -----
    Processing:
    1. Switch to evaluation mode (no dropout; batch norm uses its running
       statistics) and turn off gradient tracking.
    2. Convert the input to a float32 tensor, run the model, and convert
       the output back to float64 NumPy.
    """
    model.eval()
    with torch.no_grad():
        output = model(torch.as_tensor(X, dtype=torch.float32))
    return output.numpy().astype(np.float64)


def evaluate(
    model: nn.Module, problem_name: str, X: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    """
    Score a trained PyTorch model with the project's NumPy metrics.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model, with the best weights restored.
    problem_name : str
        "classification" or "regression".
    X : numpy.ndarray of shape (n_samples, n_features), dtype float64
        Inputs.
    y : numpy.ndarray of shape (n_samples, 1)
        True labels (0/1) or targets.

    Returns
    -------
    metric : float
        BCE (classification) or MSE (regression). NaN if the predictions
        are not finite, e.g. after divergence.
    secondary : float
        Accuracy (classification) or R² (regression).

    Notes
    -----
    Processing:
    1. Predict in evaluation mode.
    2. Apply the same metric functions as for every other model.
    """
    predictions = predict(model, X)
    if problem_name == "classification":
        return binary_cross_entropy(y, predictions), accuracy(y, predictions)
    return mean_squared_error(y, predictions), r2_score(y, predictions)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | MuonWithAdam,
    loss_fn: nn.Module,
    splits: tuple[np.ndarray, ...],
    batch_size: int,
    seed: int,
    training: dict[str, Any],
) -> dict[str, Any]:
    """
    Train a PyTorch model with mini-batches and early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train, in place.
    optimizer : torch.optim.Optimizer or MuonWithAdam
        Optimizer over the model's parameters, as built by build_optimizer.
        Only zero_grad() and step() are called, which both types provide.
    loss_fn : torch.nn.Module
        nn.BCELoss or nn.MSELoss.
    splits : tuple of numpy.ndarray
        (X_train, y_train, X_val, y_val, X_test, y_test); only the training
        and validation sets are used.
    batch_size : int
        Mini-batch size.
    seed : int
        Seed for the per-epoch shuffling.
    training : dict
        "epochs" (int) and "early_stopping" (dict with "patience" (int),
        "min_delta" (float) and "min_epochs_before_early_stop" (int)).

    Returns
    -------
    dict
        "train_loss_history" and "val_loss_history" (list of float, one per
        epoch), "best_epoch" (int, 1-based) and "diverged" (bool). The model
        holds the best validation checkpoint on return.

    Notes
    -----
    Processing, per epoch:
    1. Training mode; shuffle the training set with a seeded generator and
       loop over mini-batches: forward, loss, backward, optimizer step.
       The epoch's training loss is the mean of its batch losses.
    2. Evaluation mode; compute the validation loss on the whole
       validation set without gradients.
    3. Stop at once if either loss is not finite (diverged).
    4. Early stopping, the same rule as the NumPy trainer: the best
       checkpoint (lowest validation loss, improving by more than
       min_delta) is tracked from the first epoch; epochs without
       improvement only count after the minimum-epoch guard, and training
       stops after `patience` of them.
    After the loop, the best checkpoint's weights (and batch-norm
    statistics) are loaded back into the model.
    """
    # The fallbacks are the values the NumPy configs (configs/*_experiments.json) use;
    # configs/benchmark_experiments.json sets all four explicitly.
    stopping = training.get("early_stopping", {})
    patience = stopping.get("patience", 10)
    min_delta = stopping.get("min_delta", 1e-4)
    guard = stopping.get("min_epochs_before_early_stop", 20)

    X_train, y_train, X_val, y_val = (torch.as_tensor(a, dtype=torch.float32) for a in splits[:4])
    generator = torch.Generator().manual_seed(seed)

    train_losses, val_losses = [], []
    best_val_loss = math.inf
    # Stay None only if no epoch finishes with a finite validation loss (immediate
    # divergence); the model then keeps its last weights.
    best_state, best_epoch = None, None
    epochs_without_improvement = 0
    diverged = False

    for epoch in range(1, training.get("epochs", 100) + 1):
        model.train()
        order = torch.randperm(len(X_train), generator=generator)
        batch_losses = []
        for start in range(0, len(X_train), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            val_losses.append(loss_fn(model(X_val), y_val).item())

        if not (math.isfinite(train_losses[-1]) and math.isfinite(val_losses[-1])):
            diverged = True
            break

        if val_losses[-1] < best_val_loss - min_delta:
            best_val_loss = val_losses[-1]
            best_epoch = epoch
            # deepcopy: state_dict() returns references to the live tensors, which later
            # steps would overwrite.
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        elif epoch > guard:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "train_loss_history": train_losses,
        "val_loss_history": val_losses,
        "best_epoch": best_epoch,
        "diverged": diverged,
    }


def run_model(
    problem_name: str,
    model_name: str,
    grid: dict[str, list[Any]],
    splits: tuple[np.ndarray, ...],
    seed: int,
    baseline: float,
    training: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Train and score every optimizer / learning-rate / batch-size combination
    of one architecture on one seed's split.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    model_name : str
        Architecture name from the main config, e.g. "A2-bn".
    grid : dict of str to list
        Values for "optimizer" (list of str), "learning_rate" (list of
        float) and "batch_size" (list of int).
    splits : tuple of numpy.ndarray
        (X_train, y_train, X_val, y_val, X_test, y_test) from
        nn_numpy.benchmarks.data.load_splits.
    seed : int
        The split's seed, also used to seed PyTorch.
    baseline : float
        The split's constant-prediction test metric.
    training : dict or None, default=None
        Training settings: "epochs" (int) and "early_stopping" (dict with
        "patience" (int), "min_delta" (float) and
        "min_epochs_before_early_stop" (int)). None uses 100 epochs,
        patience 10, min_delta 1e-4 and a 20-epoch guard.

    Returns
    -------
    list of dict
        One record per combination with keys: "library" ("pytorch"),
        "problem_name", "seed", "model", "params", "val_metric",
        "test_metric", "train_metric", "test_accuracy" or "test_r2",
        "baseline_test_metric", "train_seconds", "epochs_ran",
        "best_epoch", "diverged" (bool) and "train_loss_history" /
        "val_loss_history" (list of float).

    Notes
    -----
    Processing, for each combination:
    1. Build the architecture with this seed, and its optimizer and loss
       (nn.BCELoss on the sigmoid output, or nn.MSELoss).
    2. Train it with early stopping (see train) and time the training.
    3. Score the restored model on the validation, test and training sets.
       The validation metric is used later for model selection, exactly as
       for every other model.
    4. Mark the run as diverged if any training or validation loss, or the
       restored model's validation metric, is not finite.
    """
    training = training or {}
    config = load_problem_config(problem_name)
    layer_configs = config["architectures"][model_name]
    secondary_key = "test_accuracy" if problem_name == "classification" else "test_r2"
    records = []

    for params in expand_grid(grid):
        layers = with_dropout(layer_configs, params.get("dropout"))
        model = build_network(layers, config["input_dimension"], seed)
        optimizer = build_optimizer(
            params["optimizer"],
            model.parameters(),
            params["learning_rate"],
            params.get("weight_decay"),
        )
        loss_fn = nn.BCELoss() if problem_name == "classification" else nn.MSELoss()

        start = time.perf_counter()
        history = train(model, optimizer, loss_fn, splits, params["batch_size"], seed, training)
        train_seconds = time.perf_counter() - start

        X_train, y_train, X_val, y_val, X_test, y_test = splits
        val_metric, _ = evaluate(model, problem_name, X_val, y_val)
        diverged = history["diverged"] or not math.isfinite(val_metric)
        test_metric, test_secondary = evaluate(model, problem_name, X_test, y_test)
        train_metric, _ = evaluate(model, problem_name, X_train, y_train)

        records.append(
            {
                "library": "pytorch",
                "problem_name": problem_name,
                "seed": seed,
                "model": model_name,
                "params": dict(params),
                # A diverged run is never selected: its validation metric is NaN.
                "val_metric": math.nan if diverged else val_metric,
                "test_metric": test_metric,
                "train_metric": train_metric,
                secondary_key: test_secondary,
                "baseline_test_metric": baseline,
                "train_seconds": train_seconds,
                "epochs_ran": len(history["train_loss_history"]),
                "best_epoch": history["best_epoch"],
                "diverged": diverged,
                "train_loss_history": history["train_loss_history"],
                "val_loss_history": history["val_loss_history"],
            }
        )

    return records
