"""
scikit-learn models for the library comparison.

Input: a model family name (e.g. "random_forest"), its hyperparameter grid
from ``configs/benchmark_experiments.json``, and one seed's data split.
Output: one result record per hyperparameter configuration, with the
validation, test and training metrics and the training time.

Every model is scored with the project's own NumPy metric functions
(``nn_numpy.nn.metrics``), not scikit-learn's, so the numbers are
computed exactly like the NumPy network's.
"""

import time
from typing import Any
import warnings

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from nn_numpy.benchmarks.data import print_config_progress
from nn_numpy.nn.metrics import (
    accuracy,
    binary_cross_entropy,
    mean_squared_error,
    r2_score,
)

# The scikit-learn estimator class behind each model name, per problem type.
MODEL_CLASSES = {
    "classification": {
        "logistic_regression": LogisticRegression,
        "svm": SVC,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": HistGradientBoostingClassifier,
        "mlp": MLPClassifier,
    },
    "regression": {
        "ridge": Ridge,
        "svr": SVR,
        "random_forest": RandomForestRegressor,
        "gradient_boosting": HistGradientBoostingRegressor,
        "mlp": MLPRegressor,
    },
}


def expand_grid(grid: dict[str, list[Any]] | list[dict[str, list[Any]]]) -> list[dict[str, Any]]:
    """
    List every hyperparameter combination in a grid.

    Parameters
    ----------
    grid : dict of str to list, or list of dict of str to list
        A parameter grid as written in the config: each key maps to a list
        of values to try. A list of such dicts is a union of grids (used
        when some parameters only make sense together).

    Returns
    -------
    list of dict
        One dict per combination. JSON lists inside values (such as
        hidden_layer_sizes [32, 32, 32]) are converted to tuples, which is
        what scikit-learn expects.

    Notes
    -----
    Processing:
    1. Expand the grid with scikit-learn's ParameterGrid, which forms the
       Cartesian product of each dict's value lists.
    2. Convert list values to tuples.
    """
    combinations = []
    for params in ParameterGrid(grid):
        combinations.append(
            {
                key: tuple(value) if isinstance(value, list) else value
                for key, value in params.items()
            }
        )
    return combinations


def build_model(problem_name: str, model_name: str, params: dict[str, Any], seed: int) -> Any:
    """
    Create an unfitted scikit-learn estimator.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    model_name : str
        A key of MODEL_CLASSES[problem_name], e.g. "svm".
    params : dict
        Hyperparameters for this configuration.
    seed : int
        Random seed. Passed as random_state to estimators that have one, so
        every result is reproducible.

    Returns
    -------
    sklearn.base.BaseEstimator
        The configured, unfitted estimator.

    Raises
    ------
    ValueError
        If model_name is not known for this problem type.

    Notes
    -----
    Processing:
    1. Look up the estimator class.
    2. Create it with the given hyperparameters.
    3. Set random_state to the seed if the estimator accepts one.
    """
    try:
        model_class = MODEL_CLASSES[problem_name][model_name]
    except KeyError:
        raise ValueError(
            f"Unknown {problem_name} model '{model_name}'. "
            f"Known models: {sorted(MODEL_CLASSES[problem_name])}"
        ) from None

    model = model_class(**params)
    if "random_state" in model.get_params():
        model.set_params(random_state=seed)
    return model


def evaluate(model: Any, problem_name: str, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Score a fitted model on one dataset with the project's metrics.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        A fitted estimator.
    problem_name : str
        "classification" or "regression".
    X : numpy.ndarray of shape (n_samples, n_features), dtype float64
        Inputs.
    y : numpy.ndarray of shape (n_samples, 1)
        True labels (0/1) or targets.

    Returns
    -------
    metric : float
        BCE for classification (from predicted probabilities), MSE for
        regression. Lower is better.
    secondary : float
        Accuracy for classification, R² for regression. Higher is better.

    Notes
    -----
    Processing:
    1. Classification: take the predicted probability of class 1 from
       predict_proba, then compute BCE and accuracy on it.
    2. Regression: take predict's output, then compute MSE and R².
    """
    if problem_name == "classification":
        probabilities = model.predict_proba(X)[:, 1]
        return binary_cross_entropy(y, probabilities), accuracy(y, probabilities)

    predictions = model.predict(X)
    return mean_squared_error(y, predictions), r2_score(y, predictions)


def run_model(
    problem_name: str,
    model_name: str,
    grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
    splits: tuple[np.ndarray, ...],
    seed: int,
    baseline: float,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """
    Train and score every configuration of one model on one seed's split.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    model_name : str
        A key of MODEL_CLASSES[problem_name].
    grid : dict or list of dict
        The model's hyperparameter grid from the config.
    splits : tuple of numpy.ndarray
        (X_train, y_train, X_val, y_val, X_test, y_test) from
        nn_numpy.benchmarks.data.load_splits.
    seed : int
        The split's seed, also used as the estimator's random_state.
    baseline : float
        The split's constant-prediction test metric, stored with each
        record.
    progress : bool, default=False
        Print one progress line per configuration (make benchmarks does);
        tuning leaves it off, because each trial trains a single
        configuration and prints its own line.

    Returns
    -------
    list of dict
        One record per configuration with keys: "library" ("sklearn"),
        "problem_name", "seed", "model", "params" (dict), "val_metric",
        "test_metric", "train_metric", "test_accuracy" or "test_r2",
        "baseline_test_metric", "train_seconds" and "converged" (bool:
        False if scikit-learn warned that training did not converge).

    Notes
    -----
    Processing, for each configuration in the expanded grid:
    1. Build the estimator with this seed.
    2. Fit it on the training set (targets flattened to 1-D, as
       scikit-learn expects), timing the fit and recording whether a
       ConvergenceWarning was raised.
    3. Score it on the validation, test and training sets.
    4. Store the results. Model selection (lowest validation metric)
       happens later, in the report, as for the NumPy network.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    secondary_key = "test_accuracy" if problem_name == "classification" else "test_r2"
    records = []

    configs = expand_grid(grid)
    for done, params in enumerate(configs, 1):
        model = build_model(problem_name, model_name, params, seed)

        # Record (rather than print) convergence warnings, e.g. from MLPs
        # that reach max_iter.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            start = time.perf_counter()
            model.fit(X_train, y_train.ravel())
            train_seconds = time.perf_counter() - start
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

        val_metric, _ = evaluate(model, problem_name, X_val, y_val)
        test_metric, test_secondary = evaluate(model, problem_name, X_test, y_test)
        train_metric, _ = evaluate(model, problem_name, X_train, y_train)

        records.append(
            {
                "library": "sklearn",
                "problem_name": problem_name,
                "seed": seed,
                "model": model_name,
                # JSON-friendly copy of the hyperparameters (tuples -> lists).
                "params": {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()},
                "val_metric": val_metric,
                "test_metric": test_metric,
                "train_metric": train_metric,
                secondary_key: test_secondary,
                "baseline_test_metric": baseline,
                "train_seconds": train_seconds,
                "converged": converged,
            }
        )
        if progress:
            print_config_progress(records[-1], done, len(configs))

    return records
