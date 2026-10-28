"""
Train the library models on every problem and seed, and save the results.

Input: ``configs/benchmark_experiments.json`` (models and grids per
library), the main experiment configs (which seeds to run), and the raw
data in ``data/raw/``.
Output: ``reports/benchmark_<library>_results_<stamp>.json``, one record per
model configuration, problem and seed.

Run it with ``python -m nn_numpy.benchmarks.run [library ...]`` or
``make benchmarks``. Without arguments every library in the config is run.
"""

from collections.abc import Callable
import json
import os
import sys
import time
from typing import Any

from nn_numpy.benchmarks.data import load_splits, problem_seeds
from nn_numpy.config import REPORT_DIR, ROOT_DIR, RUN_STAMP, stamped_filename
from nn_numpy.runlog import begin_phase, finish_run, start_run_log

# Location of the benchmark configuration.
BENCHMARK_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "benchmark_experiments.json")

# The problems compared, in report order.
PROBLEMS = ["classification", "regression"]


def load_benchmark_config(path: str = BENCHMARK_CONFIG_PATH) -> dict[str, Any]:
    """
    Load the benchmark configuration.

    Parameters
    ----------
    path : str, default=BENCHMARK_CONFIG_PATH
        Path of the JSON config.

    Returns
    -------
    dict
        Maps each library name to its settings, with non-library keys such
        as "description" removed. Two formats are used:
        - "sklearn": problem name -> {model name: hyperparameter grid}
        - neural-network frameworks ("tensorflow", "pytorch"): "architectures"
          (list of str, names from the main configs, used for both
          problems), "grid" (dict of str to list: optimizer,
          learning_rate, batch_size) and "training" (dict: epochs and
          early-stopping settings).

    Notes
    -----
    Processing:
    1. Read the JSON file.
    2. Drop the "description" entry, leaving only libraries.
    """
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    config.pop("description", None)
    return config


def get_runner(library: str) -> Callable[..., list[dict[str, Any]]]:
    """
    Return the function that trains one model family for a library.

    Parameters
    ----------
    library : str
        Library name from the config, e.g. "sklearn".

    Returns
    -------
    callable
        A function run_model(problem_name, model_name, grid, splits, seed,
        baseline, **options) -> list of dict.

    Raises
    ------
    ValueError
        If the library is not supported.

    Notes
    -----
    Processing:
    1. Import the library's module only when it is needed, so running one
       library doesn't require the others to be installed.
    """
    # Declared once for all three imports: the scikit-learn runner takes no training
    # settings, so the three run_model functions have different signatures.
    run_model: Callable[..., list[dict[str, Any]]]
    if library == "sklearn":
        from nn_numpy.benchmarks.sklearn_models import run_model

        return run_model
    if library == "tensorflow":
        from nn_numpy.benchmarks.keras_models import run_model

        return run_model
    if library == "pytorch":
        from nn_numpy.benchmarks.torch_models import run_model

        return run_model
    raise ValueError(f"Unsupported library: {library}")


def run_library(library: str, library_config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Train every model of one library on every problem and seed.

    Parameters
    ----------
    library : str
        Library name, e.g. "sklearn".
    library_config : dict
        The library's settings, in either format described in
        load_benchmark_config.

    Returns
    -------
    list of dict
        All result records, as returned by the library's run_model (its
        docstring lists the keys).

    Notes
    -----
    Processing:
    1. Work out the models per problem: the config's per-problem models
       for scikit-learn, or every listed architecture with the shared
       grid for a neural-network framework (whose "training" settings are
       passed on as an option).
    2. For each problem and each of its seeds, load the split and its
       constant-prediction baseline once, and run every model's full grid
       on it.
    3. Print a progress line per configuration as it finishes (the
       run_model progress option), and a one-line summary per model
       (flushed at once).
    """
    run_model = get_runner(library)
    records = []
    # Every library prints one progress line per configuration as it finishes.
    options: dict[str, Any] = {"progress": True}
    if "architectures" in library_config:
        options["training"] = library_config.get("training", {})

    for problem_name in PROBLEMS:
        if "architectures" in library_config:
            models = {name: library_config["grid"] for name in library_config["architectures"]}
        else:
            models = library_config.get(problem_name, {})
        for seed in problem_seeds(problem_name):
            splits, baseline = load_splits(problem_name, seed)
            for model_name, grid in models.items():
                start = time.perf_counter()
                model_records = run_model(
                    problem_name, model_name, grid, splits, seed, baseline, **options
                )
                records.extend(model_records)
                print(
                    f"{library:8s} {problem_name:15s} seed {seed}  {model_name:20s} "
                    f"{len(model_records):2d} configs  {time.perf_counter() - start:6.1f} s",
                    # Show progress at once, also when the output goes to a file.
                    flush=True,
                )

    return records


def save_results(library: str, records: list[dict[str, Any]]) -> str:
    """
    Save one library's records as a stamped JSON file in reports/.

    Parameters
    ----------
    library : str
        Library name, used in the file name.
    records : list of dict
        The result records.

    Returns
    -------
    str
        Path of the written file:
        reports/benchmark_<library>_results_<stamp>.json.

    Notes
    -----
    Processing:
    1. Build the stamped file name.
    2. Write the records as indented JSON and print the path.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, stamped_filename(f"benchmark_{library}_results.json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} {library} results to: {path}")
    return path


def main(libraries: list[str] | None = None) -> None:
    """
    Run the requested libraries and save each one's results.

    Parameters
    ----------
    libraries : list of str or None, default=None
        Libraries to run. None runs every library in the config.

    Returns
    -------
    None
        Writes one results file per library and prints progress.

    Raises
    ------
    ValueError
        If a requested library is not in the config.

    Notes
    -----
    Processing:
    1. Load the benchmark config.
    2. Check the requested libraries exist in it.
    3. For each library, as its own run-log phase, train everything and
       save the results.
    """
    config = load_benchmark_config()
    libraries = libraries or list(config)
    unknown = [lib for lib in libraries if lib not in config]
    if unknown:
        raise ValueError(f"Libraries not in the benchmark config: {unknown}")

    print(f"Run stamp: {RUN_STAMP}")
    for library in libraries:
        begin_phase(f"Train {library}")
        save_results(library, run_library(library, config[library]))


# Running this module trains the library models (all libraries, or those
# named on the command line) and saves the results, then prints the time.
if __name__ == "__main__":
    start_run_log("-".join(["benchmarks", *sys.argv[1:]]))
    start_time = time.perf_counter()
    main(sys.argv[1:] or None)
    print(f"\nTotal execution time: {time.perf_counter() - start_time:.2f} seconds")
    finish_run()
