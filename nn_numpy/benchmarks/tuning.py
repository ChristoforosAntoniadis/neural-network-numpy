"""
Tune the TensorFlow (Keras) and PyTorch models' hyperparameters with Optuna.

Input: ``configs/tuning_experiments.json`` (search space, trials, seeds),
the training settings in ``configs/benchmark_experiments.json``, the main
experiment configs (architectures and seeds), and the raw data.

Output, per library:
- ``reports/benchmark_<library>_tuned_results_<stamp>.json``: the best
  configuration of each problem and architecture, trained and tested once
  on every seed's split. It has the same record format as the untuned
  benchmark results, so the comparison report reads it as one more library.
- ``reports/tuning_<library>_trials_<stamp>.json``: every tuning trial.

Run it with ``python -m nn_numpy.benchmarks.tuning [library ...]`` or
``make tune``. Without arguments every library in the tuning config is run.
"""

import json
import math
import os
import sys
import time
from typing import Any

import optuna

from nn_numpy.benchmarks.data import load_splits, problem_seeds
from nn_numpy.benchmarks.run import (
    PROBLEMS,
    get_runner,
    load_benchmark_config,
    save_results,
)
from nn_numpy.config import REPORT_DIR, ROOT_DIR, RUN_STAMP, stamped_filename

# Location of the tuning configuration.
TUNING_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "tuning_experiments.json")

# Hyperparameters drawn from a list of choices; the others are numeric ranges.
CATEGORICAL = {"optimizer", "batch_size"}


def load_tuning_config(path: str = TUNING_CONFIG_PATH) -> dict[str, Any]:
    """
    Load and check the tuning configuration.

    Parameters
    ----------
    path : str, default=TUNING_CONFIG_PATH
        Path of the JSON file.

    Returns
    -------
    dict
        Keys "libraries" (list of str), "architectures" (list of str),
        "tuning_seed" (int), "sampler_seed" (int), "n_trials" (int) and
        "search_space" (dict: "optimizer" and "batch_size" are lists of
        choices; "learning_rate", "weight_decay" and "dropout" are dicts
        with "low", "high" and "log").

    Raises
    ------
    ValueError
        If a required key is missing, n_trials is not a positive integer, or
        a range has low > high.

    Notes
    -----
    Processing:
    1. Read the JSON file and drop the "description" text.
    2. Check the required keys, the trial count and every range.
    """
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    config.pop("description", None)

    required = ["libraries", "architectures", "tuning_seed", "sampler_seed", "n_trials"]
    missing = [key for key in [*required, "search_space"] if key not in config]
    if missing:
        raise ValueError(f"Tuning config is missing: {missing}")
    if not isinstance(config["n_trials"], int) or config["n_trials"] < 1:
        raise ValueError("'n_trials' must be a positive integer")
    for name, spec in config["search_space"].items():
        if name in CATEGORICAL:
            if not isinstance(spec, list) or not spec:
                raise ValueError(f"'{name}' must be a non-empty list of choices")
        elif spec["low"] > spec["high"]:
            raise ValueError(f"'{name}' has low > high")
    return config


def suggest_params(trial: optuna.trial.BaseTrial, space: dict[str, Any]) -> dict[str, Any]:
    """
    Draw one configuration from the search space.

    Parameters
    ----------
    trial : optuna.trial.BaseTrial
        The Optuna trial (or a FixedTrial in tests).
    space : dict
        The config's "search_space".

    Returns
    -------
    dict
        One value per hyperparameter: "optimizer" (str), "batch_size" (int),
        "learning_rate", "weight_decay" and "dropout" (float).

    Notes
    -----
    Processing:
    1. Lists are categorical choices (suggest_categorical).
    2. Ranges are floats (suggest_float), on a log scale when "log" is true.
    """
    params = {}
    for name, spec in space.items():
        if name in CATEGORICAL:
            params[name] = trial.suggest_categorical(name, spec)
        else:
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
    return params


def tune_model(
    library: str,
    problem_name: str,
    model_name: str,
    config: dict[str, Any],
    training: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Search one architecture's hyperparameters on the tuning seed's validation set.

    Parameters
    ----------
    library : str
        "tensorflow" or "pytorch".
    problem_name : str
        "classification" or "regression".
    model_name : str
        Architecture name, e.g. "A2-bn".
    config : dict
        The tuning config (see load_tuning_config).
    training : dict
        The library's training settings from the benchmark config.

    Returns
    -------
    best_params : dict
        The configuration with the lowest validation metric.
    trials : list of dict
        One record per trial, as returned by the library's run_model, plus
        "trial" (int, the trial number).

    Notes
    -----
    Processing:
    1. Load the tuning seed's split once.
    2. Create a minimizing study with a TPE sampler seeded with
       sampler_seed, so the search is reproducible.
    3. Each trial draws a configuration, trains it with the library's
       run_model and returns the validation metric. A diverged run returns
       infinity, so it is never chosen.
    4. Return the best trial's configuration and every trial's record.

    The test set is never used here: only the validation metric guides the
    search.
    """
    run_model = get_runner(library)
    seed = config["tuning_seed"]
    splits, baseline = load_splits(problem_name, seed)
    trials = []

    def objective(trial: optuna.trial.Trial) -> float:
        """
        Train one drawn configuration and return its validation metric for Optuna.

        Parameters
        ----------
        trial : optuna.trial.Trial
            The trial that draws the hyperparameters.

        Returns
        -------
        float
            The validation BCE or MSE of the trained configuration, or infinity if
            the run diverged (its validation metric is NaN).

        Notes
        -----
        Processing:
        1. Draw one value per hyperparameter from the search space.
        2. Wrap each value in a one-element list, so the library's run_model,
           which expects a grid, trains exactly this configuration.
        3. Train and score it on the tuning seed's split, tag the record with
           the trial number and append it to `trials` (from the enclosing scope).
        4. Return the validation metric. NaN is replaced by infinity because
           Optuna marks a trial that returns NaN as failed, and the sampler
           ignores failed trials; infinity keeps it as a completed, worst trial.
        """
        params = suggest_params(trial, config["search_space"])
        grid = {name: [value] for name, value in params.items()}
        record = run_model(
            problem_name, model_name, grid, splits, seed, baseline, training=training
        )[0]
        record["trial"] = trial.number
        trials.append(record)
        value = record["val_metric"]
        return value if math.isfinite(value) else math.inf

    sampler = optuna.samplers.TPESampler(seed=config["sampler_seed"])
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=config["n_trials"])
    return dict(study.best_params), trials


def evaluate_tuned(
    library: str,
    problem_name: str,
    model_name: str,
    params: dict[str, Any],
    training: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Train and test a tuned configuration once on every seed's split.

    Parameters
    ----------
    library : str
        "tensorflow" or "pytorch".
    problem_name : str
        "classification" or "regression".
    model_name : str
        Architecture name.
    params : dict
        The tuned configuration (see suggest_params).
    training : dict
        The library's training settings.

    Returns
    -------
    list of dict
        One record per seed, in the library's run_model format, with
        "tuned": True.

    Notes
    -----
    Processing:
    1. For each seed of the problem, load its split and baseline.
    2. Run the single tuned configuration and mark the record as tuned.
    """
    run_model = get_runner(library)
    grid = {name: [value] for name, value in params.items()}
    records = []
    for seed in problem_seeds(problem_name):
        splits, baseline = load_splits(problem_name, seed)
        record = run_model(
            problem_name, model_name, grid, splits, seed, baseline, training=training
        )[0]
        record["tuned"] = True
        records.append(record)
    return records


def tune_library(
    library: str, config: dict[str, Any], training: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Tune and evaluate every problem and architecture of one library.

    Parameters
    ----------
    library : str
        "tensorflow" or "pytorch".
    config : dict
        The tuning config.
    training : dict
        The library's training settings.

    Returns
    -------
    evaluations : list of dict
        The tuned configurations' records on every seed, with the keys
        listed in the library's run_model.
    trials : list of dict
        Every tuning trial's record: the same keys plus "trial" (int).

    Notes
    -----
    Processing, for each problem and architecture:
    1. Tune on the tuning seed (tune_model) and print the best configuration.
    2. Evaluate it on every seed (evaluate_tuned).
    """
    evaluations, trials = [], []
    for problem_name in PROBLEMS:
        for model_name in config["architectures"]:
            start = time.perf_counter()
            best, model_trials = tune_model(library, problem_name, model_name, config, training)
            trials.extend(model_trials)
            evaluations.extend(evaluate_tuned(library, problem_name, model_name, best, training))
            described = ", ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in best.items()
            )
            print(
                f"{library:10s} {problem_name:15s} {model_name:8s} "
                f"{len(model_trials)} trials  {time.perf_counter() - start:7.1f} s  "
                f"best: {described}",
                flush=True,
            )
    return evaluations, trials


def save_trials(library: str, trials: list[dict[str, Any]]) -> str:
    """
    Save every tuning trial of one library as a stamped JSON file.

    Parameters
    ----------
    library : str
        Library name, used in the file name.
    trials : list of dict
        The trial records.

    Returns
    -------
    str
        Path of the written file: reports/tuning_<library>_trials_<stamp>.json.

    Notes
    -----
    Processing:
    1. Build the stamped file name and write the records as JSON.
    """
    path = os.path.join(REPORT_DIR, stamped_filename(f"tuning_{library}_trials.json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trials, f)
    print(f"Saved {len(trials)} {library} tuning trials to: {path}")
    return path


def main(libraries: list[str] | None = None) -> None:
    """
    Tune and evaluate the requested libraries, then save their results.

    Parameters
    ----------
    libraries : list of str or None, default=None
        Libraries to tune. None tunes every library in the tuning config.

    Returns
    -------
    None
        Writes the tuned results and the trials file of each library.

    Raises
    ------
    ValueError
        If a requested library is not in the tuning config.

    Notes
    -----
    Processing:
    1. Load the tuning config and the benchmark config (training settings).
    2. Check the requested libraries, and silence Optuna's per-trial log.
    3. For each library, tune and evaluate, then save both files. The tuned
       results are saved as library "<library>_tuned", so the comparison
       report picks them up next to the untuned results.
    """
    config = load_tuning_config()
    benchmark_config = load_benchmark_config()
    libraries = libraries or config["libraries"]
    unknown = [lib for lib in libraries if lib not in config["libraries"]]
    if unknown:
        raise ValueError(f"Libraries not in the tuning config: {unknown}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"Run stamp: {RUN_STAMP}")
    for library in libraries:
        training = benchmark_config[library].get("training", {})
        evaluations, trials = tune_library(library, config, training)
        save_results(f"{library}_tuned", evaluations)
        save_trials(library, trials)


# Running this module tunes the libraries named on the command line (or all
# of them) and prints the total wall-clock time.
if __name__ == "__main__":
    start_time = time.perf_counter()
    main(sys.argv[1:] or None)
    print(f"\nTotal execution time: {time.perf_counter() - start_time:.2f} seconds")
