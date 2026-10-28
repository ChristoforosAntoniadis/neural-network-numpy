"""
Data access for the library comparisons.

Input: the experiment configs in ``configs/`` and the raw CSVs in
``data/raw/``.
Output: for a given problem and seed, exactly the same train / validation /
test split (and scaling) that the main NumPy pipeline uses, plus that
split's constant-prediction baseline.

Reusing the main pipeline's preprocessing, rather than re-implementing it,
guarantees that every library is trained and tested on identical data.
"""

import contextlib
import io
from typing import Any

import numpy as np

from nn_numpy.config_loader import ConfigLoader
from nn_numpy.modeling.train import (
    CONFIG_FILES,
    PREPROCESSORS,
    constant_prediction_baseline,
    get_seeds,
)
from nn_numpy.runlog import format_progress, should_report


def load_problem_config(problem_name: str) -> dict[str, Any]:
    """
    Load and validate the main experiment config for one problem.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".

    Returns
    -------
    dict
        The validated config, as used by the main pipeline. Keys:
        "task_type", "loss", "input_dimension" (int), "architectures" (dict of
        architecture name to list of layer dicts), "architecture_options",
        "preprocessing" (dict with "enabled" and "scale_features") and
        "experiments" (the grid and early-stopping settings).

    Notes
    -----
    Processing:
    1. Look up the config file name in CONFIG_FILES.
    2. Load and validate it with ConfigLoader.
    """
    return ConfigLoader().load_and_validate(CONFIG_FILES[problem_name])


def problem_seeds(problem_name: str) -> list[int]:
    """
    Return the seeds the main pipeline runs for one problem.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".

    Returns
    -------
    list of int
        The config's "seeds" list, or [RANDOM_SEED] if it has none.

    Notes
    -----
    Processing:
    1. Load the problem's config.
    2. Return get_seeds(config), the same function the main pipeline uses,
       so the comparisons always cover the same seeds.
    """
    return get_seeds(load_problem_config(problem_name))


def load_splits(problem_name: str, seed: int) -> tuple[tuple[np.ndarray, ...], float]:
    """
    Build the train / validation / test split for one problem and seed.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    seed : int
        Seed for the split. The same seed in the main pipeline produces the
        same split.

    Returns
    -------
    splits : tuple of numpy.ndarray
        (X_train, y_train, X_val, y_val, X_test, y_test). X arrays have
        shape (n_samples, n_features), dtype float64 (scaled with training
        statistics when the config asks for scaling). y arrays have shape
        (n_samples, 1): 0/1 integer labels for classification, float64
        targets for regression.
    baseline : float
        Test metric of always predicting a constant fitted on the training
        targets (class proportion, or mean), as in the main pipeline.

    Notes
    -----
    Processing:
    1. Load the problem's config to read its preprocessing settings.
    2. Run the problem's preprocessor with this seed, with EDA plots
       switched off and its console output silenced.
    3. Compute the constant-prediction baseline for this split.
    """
    preprocessing = load_problem_config(problem_name)["preprocessing"]

    # The preprocessors print progress; silence it to keep the output readable.
    with contextlib.redirect_stdout(io.StringIO()):
        splits = PREPROCESSORS[problem_name]().get_data(
            show_eda=False,
            preprocessing_enabled=preprocessing["enabled"],
            scale_features=preprocessing["scale_features"],
            random_seed=seed,
            run_eda=False,
        )

    return splits, constant_prediction_baseline(problem_name, splits)


def with_dropout(
    layer_configs: list[dict[str, Any]], dropout: float | None
) -> list[dict[str, Any]]:
    """
    Return an architecture with a given dropout rate after every hidden layer.

    Parameters
    ----------
    layer_configs : list of dict
        The architecture's layers from the main config (see load_problem_config).
    dropout : float or None
        Dropout rate in [0, 1) for every hidden layer. None returns the
        architecture unchanged.

    Returns
    -------
    list of dict
        A copy of layer_configs in which every layer except the output layer
        has "dropout" set to the given rate (0 removes dropout). The output
        layer never gets dropout.

    Notes
    -----
    Processing:
    1. Return layer_configs as is when dropout is None.
    2. Otherwise copy each layer dict and set "dropout" on all but the last.

    Used by hyperparameter tuning, which searches the dropout rate of the
    library models without changing the architecture definitions.
    """
    if dropout is None:
        return layer_configs
    layers = [dict(layer) for layer in layer_configs]
    for layer in layers[:-1]:
        layer["dropout"] = dropout
    return layers


def print_config_progress(record: dict[str, Any], done: int, total: int) -> None:
    """
    Print the progress line of one trained library configuration.

    Parameters
    ----------
    record : dict
        The configuration's result record, as built by a library's
        run_model. Uses "model", "problem_name", "val_metric",
        "train_seconds" and "params".
    done : int
        1-based number of the configuration within its grid.
    total : int
        Number of configurations in the grid.

    Returns
    -------
    None
        Prints e.g. "A2 | Config 3/16 | val MSE: 0.012345 | 9s | Params: {...}",
        or nothing when should_report skips this configuration.

    Notes
    -----
    Processing:
    1. Skip the line unless should_report selects this configuration
       (every one, for grids of up to MAX_PROGRESS_LINES).
    2. Print the validation metric, the one model selection uses, and
       the fit time.
    """
    if not should_report(done, total):
        return
    metric_name = "val BCE" if record["problem_name"] == "classification" else "val MSE"
    print(
        format_progress(
            record["model"],
            "Config",
            done,
            total,
            metric_name,
            record["val_metric"],
            record["train_seconds"],
            record["params"],
        )
    )
