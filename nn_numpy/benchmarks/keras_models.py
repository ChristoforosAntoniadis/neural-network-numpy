"""
TensorFlow / Keras versions of the network for the library comparison.

Input: an architecture name (e.g. "A2-bn") whose layers are read from the
main experiment configs, a grid of optimizers / learning rates / batch
sizes, training settings from ``configs/benchmark_experiments.json``, and
one seed's data split.
Output: one result record per (optimizer, learning rate, batch size), with
validation, test and training metrics, training time and loss histories.

The layer structure mirrors the NumPy network exactly (same units,
activations, bias and batch-norm placement), but everything else is what
Keras provides by default: Glorot-uniform weight initialization, Keras's
SGD / momentum / Adam optimizers, BatchNormalization's own momentum and
epsilon, float32 computation, and the built-in EarlyStopping callback. The
comparison therefore shows how the NumPy network compares with the
standard way of building the same model in TensorFlow.

Every model is scored with the project's own NumPy metric functions.
"""

import itertools
import math
import os
import time
from typing import Any

import numpy as np

# Hide TensorFlow's C++ log messages (start-up notices and harmless graph
# warnings), which would otherwise flood the console.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import tensorflow as tf

from nn_numpy.benchmarks.data import load_problem_config, print_config_progress, with_dropout
from nn_numpy.nn.metrics import (
    accuracy,
    binary_cross_entropy,
    mean_squared_error,
    r2_score,
)

# Make TensorFlow's operations deterministic, so a seed always gives the
# same result (at a small speed cost).
tf.config.experimental.enable_op_determinism()

# Only show TensorFlow's Python-side errors, not its deprecation notices.
tf.get_logger().setLevel("ERROR")


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
) -> keras.Model:
    """
    Build a Keras model with the same layer structure as a NumPy architecture.

    Parameters
    ----------
    layer_configs : list of dict
        The architecture's layers from the main config. Each has "units"
        (int), "activation" (str) and optionally "use_bias" (bool, default
        False), "batch_norm" (bool, default False) and "dropout" (float,
        default 0). The "init" key is ignored: Keras's default
        initialization is used.
    input_dimension : int
        Number of input features.
    seed : int
        Seed for weight initialization (and everything else random in
        Keras, set globally with keras.utils.set_random_seed).

    Returns
    -------
    keras.Model
        The uncompiled model: for each layer, Dense -> (BatchNormalization)
        -> Activation -> (Dropout).

    Notes
    -----
    Processing:
    1. Seed Python, NumPy and TensorFlow with keras.utils.set_random_seed.
    2. For each layer config, add a Dense layer (with or without bias),
       then BatchNormalization if requested, then the activation
       ("linear" is the identity), then Dropout if requested.
    3. Wrap the layers in a functional keras.Model.
    """
    keras.utils.set_random_seed(seed)

    inputs = keras.Input(shape=(input_dimension,))
    x = inputs
    for layer in layer_configs:
        x = keras.layers.Dense(layer["units"], use_bias=layer.get("use_bias", False))(x)
        if layer.get("batch_norm", False):
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(layer["activation"])(x)
        if layer.get("dropout", 0.0) > 0.0:
            x = keras.layers.Dropout(layer["dropout"])(x)
    return keras.Model(inputs, x)


def build_optimizer(
    name: str, learning_rate: float, weight_decay: float | None = None
) -> keras.optimizers.Optimizer:
    """
    Create a Keras optimizer.

    Parameters
    ----------
    name : str
        "sgd", "momentum", "adam" or "muon".
    learning_rate : float
        Step size.
    weight_decay : float or None, default=None
        Decoupled weight decay passed to the optimizer (for Muon, to both its
        Muon and AdamW parts). None keeps Keras's default (none for SGD and
        Adam, 0.004 for Muon); used by hyperparameter tuning.

    Returns
    -------
    keras.optimizers.Optimizer
        SGD, SGD with momentum 0.9, Adam, or Muon (Keras defaults otherwise).

    Raises
    ------
    ValueError
        If the name is not one of the four.

    Notes
    -----
    Processing:
    1. Map the name to the Keras optimizer.

    Keras's momentum uses v = 0.9 * v - lr * g, while the NumPy network uses
    an exponential moving average v = 0.9 * v + 0.1 * g, so its effective
    steps are about 10 times smaller for the same learning rate. Adam is
    Keras's closest built-in to the NumPy network's AdaBelief. Keras's Muon
    applies Muon to every 2-D variable and AdamW to the rest by itself. Its
    defaults differ from the NumPy network: weight decay 0.004, and a step
    scaled by 0.2 * sqrt(max(fan_in, fan_out)) instead of
    sqrt(max(1, fan_out / fan_in)).
    """
    decay = {} if weight_decay is None else {"weight_decay": weight_decay}
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, **decay)
    if name == "momentum":
        # 0.9 is the momentum coefficient the NumPy network uses (nn_numpy.nn.optimizers).
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, **decay)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate, **decay)
    if name == "muon":
        if weight_decay is not None:
            decay["adam_weight_decay"] = weight_decay
        return keras.optimizers.Muon(learning_rate=learning_rate, **decay)
    raise ValueError(f"Unsupported Keras optimizer: {name}")


def evaluate(
    model: keras.Model, problem_name: str, X: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    """
    Score a trained Keras model with the project's NumPy metrics.

    Parameters
    ----------
    model : keras.Model
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
    1. Call the model directly in inference mode (training=False: no
       dropout, batch norm uses its moving averages) and convert the
       output to float64. A direct call avoids model.predict's per-call
       overhead and graph retracing for small arrays.
    2. Apply the same metric functions as for every other model.
    """
    predictions = model(X.astype("float32"), training=False).numpy().astype("float64")
    if problem_name == "classification":
        return binary_cross_entropy(y, predictions), accuracy(y, predictions)
    return mean_squared_error(y, predictions), r2_score(y, predictions)


def run_model(
    problem_name: str,
    model_name: str,
    grid: dict[str, list[Any]],
    splits: tuple[np.ndarray, ...],
    seed: int,
    baseline: float,
    training: dict[str, Any] | None = None,
    progress: bool = False,
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
        float) and "batch_size" (list of int), and optionally
        "weight_decay" (list of float) and "dropout" (list of float, the
        rate after every hidden layer). Without them, Keras's default weight
        decay and the architecture's own dropout are used.
    splits : tuple of numpy.ndarray
        (X_train, y_train, X_val, y_val, X_test, y_test) from
        nn_numpy.benchmarks.data.load_splits.
    seed : int
        The split's seed, also used to seed Keras.
    baseline : float
        The split's constant-prediction test metric.
    training : dict or None, default=None
        Training settings: "epochs" (int) and "early_stopping" (dict with
        "patience" (int), "min_delta" (float) and "start_from_epoch"
        (int)). None uses 100 epochs, patience 10, min_delta 1e-4,
        start_from_epoch 20.
    progress : bool, default=False
        Print one progress line per configuration (make benchmarks does);
        tuning leaves it off, because each trial trains a single
        configuration and prints its own line.

    Returns
    -------
    list of dict
        One record per combination with keys: "library" ("tensorflow"),
        "problem_name", "seed", "model", "params", "val_metric",
        "test_metric", "train_metric", "test_accuracy" or "test_r2",
        "baseline_test_metric", "train_seconds", "epochs_ran",
        "best_epoch", "diverged" (bool) and "train_loss_history" /
        "val_loss_history" (list of float).

    Notes
    -----
    Processing, for each combination:
    1. Build the architecture with this seed and compile it with the
       optimizer and the task's loss (binary cross-entropy or MSE).
    2. Fit on the training set, validating on the validation set each
       epoch, with Keras's EarlyStopping (restoring the best weights) and
       TerminateOnNaN callbacks; time the fit.
    3. Score the restored model on the validation, test and training sets.
       The validation metric is used later for model selection, exactly as
       for every other model.
    4. Mark the run as diverged if any training or validation loss, or the
       restored model's validation metric, is not finite.
    """
    training = training or {}
    # The fallbacks are the values the NumPy configs (configs/*_experiments.json) use;
    # configs/benchmark_experiments.json sets all four explicitly.
    stopping = training.get("early_stopping", {})
    epochs = training.get("epochs", 100)

    X_train, y_train, X_val, y_val, X_test, y_test = splits
    config = load_problem_config(problem_name)
    layer_configs = config["architectures"][model_name]
    loss = "binary_crossentropy" if problem_name == "classification" else "mse"
    secondary_key = "test_accuracy" if problem_name == "classification" else "test_r2"
    records = []

    configs = expand_grid(grid)
    for done, params in enumerate(configs, 1):
        layers = with_dropout(layer_configs, params.get("dropout"))
        model = build_network(layers, config["input_dimension"], seed)
        optimizer = build_optimizer(
            params["optimizer"], params["learning_rate"], params.get("weight_decay")
        )
        model.compile(optimizer=optimizer, loss=loss)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=stopping.get("patience", 10),
            min_delta=stopping.get("min_delta", 1e-4),
            start_from_epoch=stopping.get("start_from_epoch", 20),
            restore_best_weights=True,
        )
        start = time.perf_counter()
        history = model.fit(
            X_train,
            y_train.astype("float32"),
            validation_data=(X_val, y_val.astype("float32")),
            epochs=epochs,
            batch_size=params["batch_size"],
            shuffle=True,
            verbose=0,
            callbacks=[early_stopping, keras.callbacks.TerminateOnNaN()],
        )
        train_seconds = time.perf_counter() - start

        train_losses = [float(v) for v in history.history["loss"]]
        val_losses = [float(v) for v in history.history.get("val_loss", [])]
        val_metric, _ = evaluate(model, problem_name, X_val, y_val)
        # Diverged: a non-finite loss during training, or a restored model
        # whose predictions are not finite (a float32 loss can stay just
        # below overflow while the outputs it is computed from do not).
        diverged = not all(math.isfinite(v) for v in [*train_losses, *val_losses, val_metric])
        test_metric, test_secondary = evaluate(model, problem_name, X_test, y_test)
        train_metric, _ = evaluate(model, problem_name, X_train, y_train)

        records.append(
            {
                "library": "tensorflow",
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
                "epochs_ran": len(train_losses),
                "best_epoch": early_stopping.best_epoch + 1,
                "diverged": diverged,
                "train_loss_history": train_losses,
                "val_loss_history": val_losses,
            }
        )
        if progress:
            print_config_progress(records[-1], done, len(configs))
        keras.backend.clear_session()

    return records
