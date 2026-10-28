"""
Compare the NumPy network with the library models.

Input:
- the newest ``reports/main_results_full_<stamp>.json`` (the NumPy network)
- the newest ``reports/benchmark_<library>_results_<stamp>.json`` for each
  library that has results (currently scikit-learn)

Output:
- ``reports/benchmark_report_<stamp>.txt``: comparison tables per task
- ``reports/figures/benchmarks/benchmark_<task>_<stamp>.png``: one bar
  chart per task

Every model is treated the same way: on each seed, the configuration with
the lowest validation metric is selected, and its test metric is reported.
Results are averaged over seeds (mean ± sample standard deviation).

Run it with ``python -m nn_numpy.benchmarks.report`` or
``make benchmark-report``.
"""

from collections import Counter
from collections.abc import Sequence
import json
import math
import os
from statistics import mean, stdev
import sys
import time
from typing import Any

import matplotlib.pyplot as plt

from nn_numpy.config import (
    BENCHMARK_FIGURES_DIR,
    REPORT_DIR,
    RUN_STAMP,
    latest_stamped_file,
    resolve_results_path,
    stamped_filename,
)
from nn_numpy.runlog import begin_phase, finish_run, start_run_log

# Libraries whose results the report looks for, in display order. The
# "_tuned" entries are the Optuna-tuned framework models
# (nn_numpy.benchmarks.tuning); they are optional.
LIBRARIES = ["sklearn", "tensorflow", "tensorflow_tuned", "pytorch", "pytorch_tuned"]

# Display names for libraries and models.
LIBRARY_LABELS = {
    "sklearn": "scikit-learn",
    "tensorflow": "TensorFlow (Keras)",
    "tensorflow_tuned": "TensorFlow (Keras), tuned",
    "pytorch": "PyTorch",
    "pytorch_tuned": "PyTorch, tuned",
}

# Libraries that rebuild the NumPy network's own architectures (as opposed
# to scikit-learn's different model families).
FRAMEWORKS = ["tensorflow", "pytorch"]
MODEL_LABELS = {
    "logistic_regression": "logistic regression",
    "ridge": "ridge regression",
    "svm": "SVM (RBF)",
    "svr": "SVR (RBF)",
    "random_forest": "random forest",
    "gradient_boosting": "gradient boosting",
    "mlp": "MLP",
}

# The NumPy network's baseline architectures, as in the main analysis.
BASELINE_ARCHITECTURES = ["A1", "A2"]

# Metric names per problem: (main metric, secondary metric key, secondary label).
PROBLEM_METRICS = {
    "classification": ("BCE", "test_accuracy", "Accuracy"),
    "regression": ("MSE", "test_r2", "R²"),
}


def load_json(path: str) -> list[dict[str, Any]]:
    """
    Read a JSON results file.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    list of dict
        The records stored in the file: NumPy network runs (keys as written by
        nn_numpy.modeling.train) or library runs (keys as listed in each
        library's run_model).

    Notes
    -----
    Processing:
    1. Open the file as UTF-8 and parse it.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def newest_library_results(library: str) -> str | None:
    """
    Find the newest results file for one library.

    Parameters
    ----------
    library : str
        Library name, e.g. "sklearn".

    Returns
    -------
    str or None
        Path of the newest reports/benchmark_<library>_results_<stamp>.json,
        or None if the library has not been run yet.

    Notes
    -----
    Processing:
    1. Ask latest_stamped_file for the newest matching file.
    2. Return None instead of raising when there is none, so the report
       can include whichever libraries have results.
    """
    try:
        return latest_stamped_file(REPORT_DIR, f"benchmark_{library}_results", ".json")
    except FileNotFoundError:
        return None


def is_diverged(run: dict[str, Any]) -> bool:
    """
    Return True if a NumPy network run diverged.

    Parameters
    ----------
    run : dict
        One record from the main results file.

    Returns
    -------
    bool
        True if the run's "diverged" flag is set, or its best validation
        loss is not a finite number.

    Notes
    -----
    Processing:
    1. Same rule as nn_numpy.analysis.is_diverged: diverged runs are
       never selected.
    """
    return bool(run.get("diverged")) or not math.isfinite(run["best_val_loss"])


def select_per_seed(runs: list[dict[str, Any]], val_key: str) -> list[dict[str, Any]]:
    """
    For each seed, select the run with the lowest validation metric.

    Parameters
    ----------
    runs : list of dict
        Candidate runs for one model (or group of models), across seeds.
        Each has a "seed" and a validation metric under val_key.
    val_key : str
        "best_val_loss" for NumPy network runs, "val_metric" for library
        runs. Both hold the validation BCE or MSE.

    Returns
    -------
    list of dict
        One selected run per seed, sorted by seed. Seeds whose runs all
        have a non-finite validation metric are skipped. Each run is one of
        the input dicts, unchanged, so it has the same keys.

    Notes
    -----
    Processing:
    1. Group the runs by seed.
    2. In each group, drop runs with a non-finite validation metric.
    3. Keep the run with the smallest validation metric.

    The test set is never used for selection.
    """
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        by_seed.setdefault(run["seed"], []).append(run)

    selected = []
    for seed in sorted(by_seed):
        candidates = [r for r in by_seed[seed] if math.isfinite(r[val_key])]
        if candidates:
            selected.append(min(candidates, key=lambda r: r[val_key]))
    return selected


def mean_pm_std(values: Sequence[float | None], digits: int) -> str:
    """
    Format a list of numbers as "mean ± std".

    Parameters
    ----------
    values : sequence of float or None
        One value per seed. None values are ignored. Typed as a Sequence
        (read-only) so a plain list of floats is accepted too.
    digits : int
        Decimal places.

    Returns
    -------
    str
        For example "0.5200 ± 0.1876", or "-" when there are no values.

    Notes
    -----
    Processing:
    1. Drop None values (e.g. metrics missing from older results files).
    2. Format the mean and the sample standard deviation (0 for one value).
    """
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    # The ignores: None was filtered out above, but mypy keeps a re-bound parameter's
    # declared type, Sequence[float | None].
    spread = stdev(values) if len(values) > 1 else 0.0  # type: ignore[type-var]  # no Nones
    return (
        f"{mean(values):.{digits}f} ± "  # type: ignore[type-var]  # no Nones
        f"{spread:.{digits}f}"
    )


def describe_config(run: dict[str, Any]) -> str:
    """
    Short text describing a selected configuration.

    Parameters
    ----------
    run : dict
        A NumPy network run or a library run.

    Returns
    -------
    str
        For a NumPy run, e.g. "A1 · adabelief · LR 0.1 · bs 16", and a
        framework run in the same form, e.g. "A1 · adam · LR 0.1 · bs 16";
        a tuned framework run adds its weight decay and dropout, e.g.
        "A1 · adam · LR 0.0123 · bs 32 · wd 1.2e-05 · dropout 0.10".
        For a scikit-learn run, its hyperparameters, e.g. "C=10,
        gamma=scale".

    Notes
    -----
    Processing:
    1. NumPy runs are recognised by their "architecture" key, framework
       runs by an "optimizer" hyperparameter.
    2. scikit-learn runs list their hyperparameters, leaving out fixed
       settings that are the same for every configuration (max_iter,
       probability, kernel, n_estimators).
    """
    if "architecture" in run:
        return (
            f"{run['architecture']} · {run['optimizer']} · "
            f"LR {run['learning_rate']} · bs {run['batch']}"
        )
    if "optimizer" in run["params"]:
        p = run["params"]
        if "weight_decay" in p:
            return (
                f"{run['model']} · {p['optimizer']} · LR {p['learning_rate']:.3g} · "
                f"bs {p['batch_size']} · wd {p['weight_decay']:.1e} · dropout {p['dropout']:.2f}"
            )
        return (
            f"{run['model']} · {p['optimizer']} · LR {p['learning_rate']} · bs {p['batch_size']}"
        )
    fixed = {"max_iter", "probability", "kernel", "n_estimators"}
    return ", ".join(f"{k}={v}" for k, v in run["params"].items() if k not in fixed) or "default"


def summarize(label: str, selected: list[dict[str, Any]], secondary_key: str) -> dict[str, Any]:
    """
    Summarize one model's selected runs across seeds.

    Parameters
    ----------
    label : str
        Row label for the report, e.g. "scikit-learn · random forest".
    selected : list of dict
        One selected run per seed.
    secondary_key : str
        "test_accuracy" or "test_r2".

    Returns
    -------
    dict
        Keys: "label", "test" (list of float), "secondary" (list of float
        or None), "error_removed" (list of float), "train_seconds" (list of
        float or None) and "config" (str: the configuration chosen most
        often, with how many seeds chose it).

    Notes
    -----
    Processing:
    1. Collect the test metric, secondary metric, training time and
       "error removed" (1 - test / constant-prediction baseline) of each
       selected run.
    2. Find the most frequently selected configuration.
    """
    configs = Counter(describe_config(r) for r in selected)
    config, count = configs.most_common(1)[0]
    return {
        "label": label,
        "test": [r["test_metric"] for r in selected],
        "secondary": [r.get(secondary_key) for r in selected],
        "error_removed": [1 - r["test_metric"] / r["baseline_test_metric"] for r in selected],
        "train_seconds": [r.get("train_seconds") for r in selected],
        "config": f"{config} ({count}/{len(selected)} seeds)",
    }


def build_rows(
    problem_name: str,
    numpy_runs: list[dict[str, Any]],
    library_runs: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """
    Build the comparison rows for one problem.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    numpy_runs : list of dict
        All NumPy network runs for this problem.
    library_runs : dict of str to list of dict
        Library name -> that library's runs for this problem.

    Returns
    -------
    rows : list of dict
        Summaries (see summarize) in display order: the NumPy network
        (baseline architectures, then all architectures), each library
        model, then each library's best model overall.
    selections : dict of str to list of dict
        The selected runs per seed for the two head-to-head entries:
        "numpy" (NumPy network, all architectures) and each library name
        (that library's best model overall).

    Notes
    -----
    Processing:
    1. NumPy network: drop diverged runs, then select per seed among A1 /
       A2, and separately among all architectures.
    2. Each library model: select per seed among its configurations.
    3. Each library overall: select per seed among all its models'
       configurations, i.e. what a practitioner would pick from that
       library using the validation set.
    """
    _, secondary_key, _ = PROBLEM_METRICS[problem_name]
    numpy_ok = [r for r in numpy_runs if not is_diverged(r)]

    numpy_baseline = select_per_seed(
        [r for r in numpy_ok if r["architecture"] in BASELINE_ARCHITECTURES], "best_val_loss"
    )
    numpy_all = select_per_seed(numpy_ok, "best_val_loss")
    n_architectures = len({r["architecture"] for r in numpy_runs})
    rows = [
        summarize("NumPy NN · A1 / A2", numpy_baseline, secondary_key),
        summarize(f"NumPy NN · all {n_architectures} architectures", numpy_all, secondary_key),
    ]
    selections = {"numpy": numpy_all}

    for library, runs in library_runs.items():
        name = LIBRARY_LABELS.get(library, library)
        models = list(dict.fromkeys(r["model"] for r in runs))
        for model in models:
            selected = select_per_seed([r for r in runs if r["model"] == model], "val_metric")
            rows.append(
                summarize(f"{name} · {MODEL_LABELS.get(model, model)}", selected, secondary_key)
            )
        best = select_per_seed(runs, "val_metric")
        best_row = summarize(f"{name} · best model (selected on validation)", best, secondary_key)
        best_row["config"] = ", ".join(
            f"{MODEL_LABELS.get(m, m)} ×{n}"
            for m, n in Counter(r["model"] for r in best).most_common()
        )
        rows.append(best_row)
        selections[library] = best

    return rows, selections


def format_table(problem_name: str, rows: list[dict[str, Any]]) -> str:
    """
    Format one problem's comparison rows as a text table.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    rows : list of dict
        Summaries from build_rows.

    Returns
    -------
    str
        A titled, fixed-width table, followed by the configuration chosen
        most often for each row.

    Notes
    -----
    Processing:
    1. Write one line per row: label, test metric, secondary metric,
       error removed (%) and training time of the selected configuration.
    2. List the most often selected configuration per row underneath.
    """
    metric, _, secondary = PROBLEM_METRICS[problem_name]
    digits = 5 if problem_name == "classification" else 4
    title = f"{problem_name.capitalize()} — test {metric} (lower is better), mean ± std over seeds"
    lines = [title, "-" * len(title)]
    lines.append(
        f"{'Model':<60}{'Test ' + metric:<24}{secondary:<20}{'Error removed %':<20}{'Fit time (s)':<14}"
    )
    for row in rows:
        lines.append(
            f"{row['label']:<60}"
            f"{mean_pm_std(row['test'], digits):<24}"
            f"{mean_pm_std(row['secondary'], 4):<20}"
            f"{mean_pm_std([100 * v for v in row['error_removed']], 2):<20}"
            f"{mean_pm_std(row['train_seconds'], 2):<14}"
        )
    lines.append("")
    lines.append("Most often selected configuration:")
    for row in rows:
        lines.append(f"- {row['label']}: {row['config']}")
    lines.append("")
    return "\n".join(lines)


def format_head_to_head(problem_name: str, selections: dict[str, list[dict[str, Any]]]) -> str:
    """
    Compare the NumPy network with each library's best model, seed by seed.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    selections : dict of str to list of dict
        From build_rows: "numpy" and one entry per library.

    Returns
    -------
    str
        One line per library: on how many seeds the NumPy network's test
        metric was lower (better), and the per-seed test metrics.

    Notes
    -----
    Processing:
    1. Pair the NumPy network's selected run with the library's selected
       run on the same seed.
    2. Count the seeds where the NumPy network's test metric is lower.
    """
    metric, _, _ = PROBLEM_METRICS[problem_name]
    numpy_by_seed = {r["seed"]: r for r in selections["numpy"]}
    lines = [f"Head-to-head per seed — {problem_name} (test {metric}):"]
    for library, selected in selections.items():
        if library == "numpy":
            continue
        pairs = [(numpy_by_seed[r["seed"]], r) for r in selected if r["seed"] in numpy_by_seed]
        wins = sum(n["test_metric"] < lib["test_metric"] for n, lib in pairs)
        detail = "; ".join(
            f"seed {n['seed']}: {n['test_metric']:.4g} vs {lib['test_metric']:.4g}"
            for n, lib in pairs
        )
        lines.append(
            f"- NumPy NN (all architectures) vs {LIBRARY_LABELS.get(library, library)} best: "
            f"NumPy better on {wins}/{len(pairs)} seeds ({detail})"
        )
    lines.append("")
    return "\n".join(lines)


def format_architecture_table(
    problem_name: str,
    numpy_runs: list[dict[str, Any]],
    framework: str,
    framework_runs: list[dict[str, Any]],
) -> str:
    """
    Compare the NumPy network and a framework architecture by architecture.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    numpy_runs : list of dict
        All NumPy network runs for this problem.
    framework : str
        Framework library name, e.g. "tensorflow".
    framework_runs : list of dict
        All of the framework's runs for this problem.

    Returns
    -------
    str
        A table with one row per architecture the framework ran: the test
        metric (mean ± std over seeds) of the NumPy and the framework
        version, each selected per seed on validation among its own
        optimizer / learning-rate / batch-size grid; on how many seeds the
        NumPy version was better; and how many runs of each diverged.

    Notes
    -----
    Processing, for each architecture:
    1. Select the best NumPy run per seed (lowest validation loss, diverged
       runs excluded) and the best framework run per seed (lowest
       validation metric; diverged runs have NaN and are skipped).
    2. Compare the two seed by seed.
    3. Count diverged runs on each side.

    The NumPy grid uses SGD / momentum / AdaBelief / Muon, the frameworks
    SGD / momentum / Adam / Muon, each as implemented by that library.
    """
    metric, _, _ = PROBLEM_METRICS[problem_name]
    digits = 5 if problem_name == "classification" else 4
    label = LIBRARY_LABELS.get(framework, framework)
    title = f"Same architecture, different implementation — {problem_name} (test {metric})"
    lines = [title, "-" * len(title)]
    lines.append(
        f"{'Architecture':<14}{'NumPy NN':<24}{label:<24}"
        f"{'NumPy better':<14}Diverged runs (NumPy / framework)"
    )
    for arch in dict.fromkeys(r["model"] for r in framework_runs):
        arch_numpy = [r for r in numpy_runs if r["architecture"] == arch]
        arch_framework = [r for r in framework_runs if r["model"] == arch]
        numpy_sel = select_per_seed([r for r in arch_numpy if not is_diverged(r)], "best_val_loss")
        frame_sel = select_per_seed(arch_framework, "val_metric")
        frame_by_seed = {r["seed"]: r for r in frame_sel}
        pairs = [(n, frame_by_seed[n["seed"]]) for n in numpy_sel if n["seed"] in frame_by_seed]
        wins = sum(n["test_metric"] < f["test_metric"] for n, f in pairs)
        numpy_div = sum(is_diverged(r) for r in arch_numpy)
        frame_div = sum(bool(r.get("diverged")) for r in arch_framework)
        lines.append(
            f"{arch:<14}"
            f"{mean_pm_std([r['test_metric'] for r in numpy_sel], digits):<24}"
            f"{mean_pm_std([r['test_metric'] for r in frame_sel], digits):<24}"
            f"{f'{wins}/{len(pairs)} seeds':<14}"
            f"{numpy_div}/{len(arch_numpy)} / {frame_div}/{len(arch_framework)}"
        )
    lines.append("")
    return "\n".join(lines)


def format_tuning_table(
    problem_name: str,
    framework: str,
    untuned_runs: list[dict[str, Any]],
    tuned_runs: list[dict[str, Any]],
) -> str:
    """
    Compare a framework's grid-selected models with its Optuna-tuned ones.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    framework : str
        Framework library name, e.g. "tensorflow".
    untuned_runs : list of dict
        The framework's grid runs for this problem.
    tuned_runs : list of dict
        The framework's tuned runs for this problem: one per architecture
        and seed, all of an architecture sharing one tuned configuration.

    Returns
    -------
    str
        A table with one row per tuned architecture: the test metric (mean
        ± std over seeds) of the grid-selected and the tuned model, on how
        many seeds tuning gave the lower test metric, how many tuned runs
        diverged, and the tuned configuration.

    Notes
    -----
    Processing, for each architecture:
    1. Select the grid model per seed on validation (as everywhere else);
       the tuned model has one run per seed (diverged runs are skipped).
    2. Compare the two seed by seed and describe the tuned configuration.

    The tuned configuration was chosen on the tuning seed's validation set
    only, so these test scores are unbiased by the search.
    """
    metric, _, _ = PROBLEM_METRICS[problem_name]
    digits = 5 if problem_name == "classification" else 4
    label = LIBRARY_LABELS.get(framework, framework)
    title = f"{label}: grid vs. Optuna-tuned — {problem_name} (test {metric})"
    lines = [title, "-" * len(title)]
    lines.append(
        f"{'Architecture':<14}{'Grid':<24}{'Tuned':<24}{'Tuned better':<14}"
        f"{'Diverged':<10}Tuned configuration"
    )
    for arch in dict.fromkeys(r["model"] for r in tuned_runs):
        grid_sel = select_per_seed([r for r in untuned_runs if r["model"] == arch], "val_metric")
        arch_tuned = [r for r in tuned_runs if r["model"] == arch]
        tuned_sel = select_per_seed(arch_tuned, "val_metric")
        tuned_by_seed = {r["seed"]: r for r in tuned_sel}
        pairs = [(g, tuned_by_seed[g["seed"]]) for g in grid_sel if g["seed"] in tuned_by_seed]
        wins = sum(t["test_metric"] < g["test_metric"] for g, t in pairs)
        diverged = sum(bool(r.get("diverged")) for r in arch_tuned)
        lines.append(
            f"{arch:<14}"
            f"{mean_pm_std([r['test_metric'] for r in grid_sel], digits):<24}"
            f"{mean_pm_std([r['test_metric'] for r in tuned_sel], digits):<24}"
            f"{f'{wins}/{len(pairs)} seeds':<14}"
            f"{f'{diverged}/{len(arch_tuned)}':<10}"
            f"{describe_config(arch_tuned[0])}"
        )
    lines.append("")
    return "\n".join(lines)


def count_wins(
    numpy_selected: list[dict[str, Any]], other_selected: list[dict[str, Any]]
) -> tuple[int, int]:
    """
    Count on how many seeds the NumPy network beat another model.

    Parameters
    ----------
    numpy_selected : list of dict
        The NumPy network's selected run per seed (at most one per seed),
        each with "seed" (int) and "test_metric" (float).
    other_selected : list of dict
        The other model's selected run per seed, same keys. A seed can be
        missing on either side, e.g. when every run of that seed diverged.

    Returns
    -------
    wins : int
        Number of compared seeds where the NumPy network's test metric was
        lower (better).
    compared : int
        Number of seeds that were compared.

    Notes
    -----
    Processing:
    1. Pair the runs by seed; only seeds present on both sides are
       compared, as in the other per-seed tables.
    2. Count the seeds where the NumPy network's test metric is strictly
       lower; a tie is not a win.
    """
    other_by_seed = {r["seed"]: r for r in other_selected}
    pairs = [(n, other_by_seed[n["seed"]]) for n in numpy_selected if n["seed"] in other_by_seed]
    wins = sum(n["test_metric"] < o["test_metric"] for n, o in pairs)
    return wins, len(pairs)


def format_three_way_table(
    problem_name: str,
    numpy_runs: list[dict[str, Any]],
    framework: str,
    grid_runs: list[dict[str, Any]],
    tuned_runs: list[dict[str, Any]],
) -> str:
    """
    Compare, per architecture, the NumPy network with a framework's grid and tuned models.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    numpy_runs : list of dict
        All NumPy network runs for this problem.
    framework : str
        Framework library name, e.g. "tensorflow".
    grid_runs : list of dict
        The framework's grid runs for this problem.
    tuned_runs : list of dict
        The framework's Optuna-tuned runs for this problem.

    Returns
    -------
    str
        A table with one row per architecture the framework tuned: the test
        metric (mean ± std over seeds) of the NumPy network (selected on
        validation from its grid), the framework's grid-selected model and
        its tuned model, and on how many seeds the NumPy network beat each
        of the two framework versions (see count_wins).

    Notes
    -----
    Processing, for each architecture:
    1. Select per seed on validation: the NumPy network among its
       non-diverged runs (best_val_loss), the framework grid and tuned
       models among theirs (val_metric; diverged runs have NaN).
    2. Write the three means and the two seed-by-seed win counts.

    This puts the three groups side by side in one row; the separate
    architecture and grid-vs-tuned tables give the diverged-run counts and
    the tuned configurations.
    """
    metric, _, _ = PROBLEM_METRICS[problem_name]
    digits = 5 if problem_name == "classification" else 4
    label = LIBRARY_LABELS.get(framework, framework)
    title = f"NumPy NN vs. {label} grid vs. {label} tuned — {problem_name} (test {metric})"
    lines = [title, "-" * len(title)]
    lines.append(
        f"{'Architecture':<14}{'NumPy NN':<24}{'Grid':<24}{'Tuned':<24}"
        f"{'NumPy better than grid / tuned'}"
    )
    for arch in dict.fromkeys(r["model"] for r in tuned_runs):
        numpy_sel = select_per_seed(
            [r for r in numpy_runs if r["architecture"] == arch and not is_diverged(r)],
            "best_val_loss",
        )
        grid_sel = select_per_seed([r for r in grid_runs if r["model"] == arch], "val_metric")
        tuned_sel = select_per_seed([r for r in tuned_runs if r["model"] == arch], "val_metric")
        grid_wins, grid_n = count_wins(numpy_sel, grid_sel)
        tuned_wins, tuned_n = count_wins(numpy_sel, tuned_sel)
        lines.append(
            f"{arch:<14}"
            f"{mean_pm_std([r['test_metric'] for r in numpy_sel], digits):<24}"
            f"{mean_pm_std([r['test_metric'] for r in grid_sel], digits):<24}"
            f"{mean_pm_std([r['test_metric'] for r in tuned_sel], digits):<24}"
            f"{grid_wins}/{grid_n} / {tuned_wins}/{tuned_n} seeds"
        )
    lines.append("")
    return "\n".join(lines)


def plot_learning_curves(
    framework: str,
    curves: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]],
) -> str:
    """
    Plot validation-loss curves of the NumPy network's and a framework's
    selected models (grid and, if available, tuned), one panel per problem.

    Parameters
    ----------
    framework : str
        Framework library name, e.g. "tensorflow".
    curves : dict of str to tuple of (dict, dict, dict or None)
        Problem name -> (selected NumPy run, selected framework grid run,
        selected framework tuned run or None) on the same seed. Each run
        has a "val_loss_history" (list of float).

    Returns
    -------
    str
        Path of the saved PNG in reports/figures/benchmarks/.

    Notes
    -----
    Processing:
    1. For each problem, plot the validation-loss histories against the
       epoch number, labelled with each run's configuration: NumPy (orange),
       framework grid (blue) and framework tuned (green, dashed), when
       tuned results exist.
    2. Use a log scale: the losses fall by orders of magnitude.
    3. Save the figure with the run stamp and close it.
    """
    label = LIBRARY_LABELS.get(framework, framework)
    fig, axes = plt.subplots(1, len(curves), figsize=(6.5 * len(curves), 4.5))
    axes = [axes] if len(curves) == 1 else list(axes)
    for ax, (problem_name, (numpy_run, frame_run, tuned_run)) in zip(axes, curves.items()):
        metric, _, _ = PROBLEM_METRICS[problem_name]
        lines_to_plot = [
            (numpy_run, "NumPy", "tab:orange", "-"),
            (frame_run, label, "tab:blue", "-"),
        ]
        if tuned_run is not None:
            lines_to_plot.append((tuned_run, f"{label}, tuned", "tab:green", "--"))
        for run, name, color, style in lines_to_plot:
            history = run["val_loss_history"]
            ax.plot(
                range(1, len(history) + 1),
                history,
                color=color,
                linestyle=style,
                label=f"{name}: {describe_config(run)}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Validation {metric} (log scale)")
        ax.set_title(f"{problem_name.capitalize()}, seed {numpy_run['seed']}: selected models")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(BENCHMARK_FIGURES_DIR, exist_ok=True)
    name = stamped_filename(f"benchmark_curves_{framework}.png")
    path = os.path.join(BENCHMARK_FIGURES_DIR, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved figure to: {path}")
    return path


def plot_problem(problem_name: str, rows: list[dict[str, Any]]) -> str:
    """
    Draw a horizontal bar chart of the test metric for one problem.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    rows : list of dict
        Summaries from build_rows.

    Returns
    -------
    str
        Path of the saved PNG in reports/figures/benchmarks/.

    Notes
    -----
    Processing:
    1. One bar per row: the mean test metric across seeds, with the
       standard deviation as an error bar. NumPy rows are highlighted.
    2. Use a log scale: the values span orders of magnitude (BCE from
       about 1e-4 to 0.04; MSE from about 0.2 for the best models to 9
       for ridge regression), which a linear axis would flatten.
    3. Save the figure with the run stamp and close it.
    """
    metric, _, _ = PROBLEM_METRICS[problem_name]
    labels = [row["label"] for row in rows]
    means = [mean(row["test"]) for row in rows]
    stds = [stdev(row["test"]) if len(row["test"]) > 1 else 0.0 for row in rows]
    colors = ["tab:orange" if label.startswith("NumPy") else "tab:blue" for label in labels]

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.5))
    positions = list(range(len(rows)))[::-1]
    ax.barh(positions, means, xerr=stds, color=colors, capsize=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel(f"Test {metric} (mean ± std over seeds; lower is better)")
    ax.set_title(f"{problem_name.capitalize()}: NumPy network vs. library models")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    os.makedirs(BENCHMARK_FIGURES_DIR, exist_ok=True)
    path = os.path.join(BENCHMARK_FIGURES_DIR, stamped_filename(f"benchmark_{problem_name}.png"))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved figure to: {path}")
    return path


def main(numpy_results_path: str | None = None) -> None:
    """
    Build the comparison report and figures.

    Parameters
    ----------
    numpy_results_path : str or None, default=None
        A specific main_results_full_<stamp>.json. None uses the newest.

    Returns
    -------
    None
        Writes reports/benchmark_report_<stamp>.txt, one bar chart per
        problem and one learning-curve figure per framework, and prints
        their paths.

    Raises
    ------
    FileNotFoundError
        If no library has results yet, or no NumPy results file exists.

    Notes
    -----
    Processing:
    1. Load the NumPy network's results and each library's newest results.
    2. For each problem, build the rows, the table, the head-to-head lines,
       an architecture-by-architecture table per framework, and the bar
       chart.
    3. Plot the learning curves of the selected models per framework.
    4. Write the report with a header explaining the method and listing
       the input files.
    """
    numpy_path = resolve_results_path(numpy_results_path)
    numpy_results = load_json(numpy_path)
    library_paths = {lib: newest_library_results(lib) for lib in LIBRARIES}
    library_paths = {lib: path for lib, path in library_paths.items() if path}
    if not library_paths:
        raise FileNotFoundError(
            "No library results found. Run python -m nn_numpy.benchmarks.run first."
        )
    library_results = {lib: load_json(path) for lib, path in library_paths.items()}

    print(f"Run stamp: {RUN_STAMP}")
    lines = ["LIBRARY COMPARISON", "==================", ""]
    lines.append(
        "The NumPy network is compared with library models trained on exactly "
        "the same train / validation / test splits (same seeds), scored with the same NumPy "
        "metric functions. For every model, the configuration with the lowest validation "
        "metric is selected on each seed and its test metric is reported; the test set is "
        "never used for selection. 'Error removed' is 1 - test / constant-prediction "
        "baseline. 'Fit time' is the training time of the selected configuration on one "
        "seed (one CPU run each; the NumPy network trains up to 100 epochs with early "
        "stopping)."
    )
    lines.append("")
    lines.append(f"NumPy network results: {os.path.basename(numpy_path)}")
    for lib, path in library_paths.items():
        lines.append(f"{LIBRARY_LABELS.get(lib, lib)} results: {os.path.basename(path)}")
    lines.append("")

    # Learning curves to plot per framework: problem -> (NumPy run, framework run).
    # The tuned run is None when that framework has no tuned results.
    curves: dict[str, dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]]] = {
        fw: {} for fw in FRAMEWORKS if fw in library_results
    }

    for problem_name in PROBLEM_METRICS:
        numpy_runs = [r for r in numpy_results if r["problem_name"] == problem_name]
        library_runs = {
            lib: [r for r in records if r["problem_name"] == problem_name]
            for lib, records in library_results.items()
        }
        rows, selections = build_rows(problem_name, numpy_runs, library_runs)
        lines.append(format_table(problem_name, rows))
        lines.append(format_head_to_head(problem_name, selections))
        for framework, framework_curves in curves.items():
            lines.append(
                format_architecture_table(
                    problem_name, numpy_runs, framework, library_runs[framework]
                )
            )
            # The first seed's selected models: NumPy, framework grid and,
            # if tuned results exist, the framework's tuned model.
            numpy_first = selections["numpy"][0]
            frame_first = next(
                r for r in selections[framework] if r["seed"] == numpy_first["seed"]
            )
            tuned_first = next(
                (
                    r
                    for r in selections.get(f"{framework}_tuned", [])
                    if r["seed"] == numpy_first["seed"]
                ),
                None,
            )
            framework_curves[problem_name] = (numpy_first, frame_first, tuned_first)
        for framework in FRAMEWORKS:
            tuned = f"{framework}_tuned"
            if framework in library_runs and tuned in library_runs:
                lines.append(
                    format_tuning_table(
                        problem_name, framework, library_runs[framework], library_runs[tuned]
                    )
                )
                lines.append(
                    format_three_way_table(
                        problem_name,
                        numpy_runs,
                        framework,
                        library_runs[framework],
                        library_runs[tuned],
                    )
                )
        plot_problem(problem_name, rows)

    for framework, framework_curves in curves.items():
        plot_learning_curves(framework, framework_curves)

    path = os.path.join(REPORT_DIR, stamped_filename("benchmark_report.txt"))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved benchmark report to: {path}")


# Running this module writes the comparison report and figures from the
# newest results (optionally a specific NumPy results file given as the
# first argument).
if __name__ == "__main__":
    start_run_log("benchmark-report")
    begin_phase("Write the comparison report and figures")
    start_time = time.perf_counter()
    main(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"\nTotal execution time: {time.perf_counter() - start_time:.2f} seconds")
    finish_run()
