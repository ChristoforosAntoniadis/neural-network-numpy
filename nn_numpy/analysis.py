"""
Aggregate analysis of the experiment sweep, written as a plain-text report.

Input: a full-results JSON written by nn_numpy.modeling.train (reports/main_results_full_<stamp>.json,
the newest one by default, or a path given on the command line).

Processing: adds derived values to every run (best validation loss, final
training loss, training-loss convergence epoch), then summarizes the runs by
optimizer and by architecture, per problem and combined across problems, to
answer three questions: which optimizer converges fastest, which reaches the
best loss, and how depth affects optimization. Extra sections cover the A2
variants (bias, He initialization, batch normalization), dropout, and, when
several seeds were run, mean ± standard deviation across seeds.

Output: reports/analysis_<stamp>.txt.
"""

from collections.abc import Iterable
import json
import math
import os
from statistics import mean, stdev
import sys
import time
from typing import Any

from nn_numpy.config import (
    RANDOM_SEED,
    REPORT_DIR,
    RUN_STAMP,
    resolve_results_path,
    stamped_filename,
)

# Path where this script will write the analysis report. The run stamp is
# inserted before the extension so earlier reports are never overwritten.
OUTPUT_PATH = os.path.join(REPORT_DIR, stamped_filename("analysis.txt"))

# Preferred display order for optimizer summaries in tables. These are the
# baseline optimizers: every section except "Muon" uses only their runs, so
# adding an optimizer to the configs does not change those numbers.
OPTIMIZER_ORDER = ["sgd", "momentum", "adabelief"]

# Optimizers compared in the Muon section: the baseline ones plus Muon.
MUON_SECTION_OPTIMIZERS = OPTIMIZER_ORDER + ["muon"]

# Preferred display order for architecture summaries in tables.
# These are the baseline architectures: the depth comparison and all
# optimizer / architecture sections are computed from these runs only, so
# adding new architectures to the configs does not change those numbers.
ARCHITECTURE_ORDER = ["A1", "A2"]

# A2 and its variants, which change one thing at a time: a bias term,
# He initialization, both, batch normalization on the hidden layers, or a
# cap on the gradient norm during training. Reported in their own section.
A2_VARIANT_ORDER = ["A2", "A2-bias", "A2-he", "A2-bias-he", "A2-bn", "A2-clip"]

# The regression runs whose 3-layer ReLU network collapses in the baseline
# (plain SGD and momentum at learning rate 0.1). The variant section checks
# whether a bias term or He initialization prevents the collapse.
COLLAPSE_CHECK_RUNS = [("sgd", 16), ("sgd", 64), ("momentum", 16), ("momentum", 64)]
COLLAPSE_CHECK_LEARNING_RATE = 0.1

# Architectures compared with and without dropout, as (without, with) pairs.
DROPOUT_PAIRS = [("A1", "A1-dropout"), ("A2-bn", "A2-bn-dropout")]


def load_results(path: str) -> list[dict[str, Any]]:
    """
    Load the full experiment results JSON from disk.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    list of dict
        One dictionary per experiment run, as written by nn_numpy.modeling.train (keys such
        as "problem_name", "architecture", "optimizer", "learning_rate",
        "batch", "test_metric", "train_loss_history", "val_loss_history").

    Notes
    -----
    Processing:
    1. Open the file as UTF-8 text.
    2. Parse it with json.load and return the result.
    """
    # Open the JSON file and parse it into Python objects.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def convergence_epoch(
    train_loss_history: list[float], relative_tolerance: float = 0.01, absolute_floor: float = 1e-4
) -> int:
    """
    Find the epoch at which one run's training loss has converged.

    Parameters
    ----------
    train_loss_history : list of float
        Training loss recorded at each epoch for one run (must not be empty).
    relative_tolerance : float, default=0.01
        Relative closeness threshold to the final training loss.
        Here, 0.01 means "within 1% of the final training loss." No recorded
        justification for 1%.
    absolute_floor : float, default=1e-4
        Minimum tolerance allowed, so the rule does not become unrealistically
        strict when the final training loss is extremely small. 1e-4 equals
        the configs' min_delta, the smallest loss change the trainer counts
        as an improvement.

    Returns
    -------
    int
        The 1-based number of the first epoch after which the remaining
        training-loss values stay close to the final training loss. Equals
        the number of epochs run if no earlier such epoch exists.

    Notes
    -----
    Processing:
    1. Take the final training loss (the last value in the history).
    2. Compute the tolerance
       max(absolute_floor, relative_tolerance * abs(final_train_loss)).
    3. Scan the epochs from the first one forward, and return the first
       epoch from which every remaining loss is within the tolerance of the
       final loss.

    Definition: convergence epoch = first epoch after which all remaining
    training-loss values stay within a small tolerance of the final training
    loss. It uses TRAINING loss, not validation loss.

    Why: the exercise asks which optimizer converges fastest on average. To
    match that wording, convergence is defined from the training-loss curve,
    not from early stopping or validation loss.
    """
    # The final training loss is the last value in the recorded history.
    final_train_loss = train_loss_history[-1]

    # Build the tolerance band around the final training loss.
    # We use either:
    # - 1% of the final loss, or
    # - an absolute minimum tolerance of 0.0001
    # whichever is larger.
    tolerance = max(absolute_floor, relative_tolerance * abs(final_train_loss))

    # Total number of epochs actually run for this experiment.
    epochs_ran = len(train_loss_history)

    # Scan from the first epoch forward.
    # We are looking for the earliest point after which the remaining
    # training-loss values all stay within the tolerance band.
    for start_idx in range(epochs_ran):
        # Tail of the curve from the current epoch to the end.
        tail = train_loss_history[start_idx:]

        # If every remaining value is sufficiently close to the final loss,
        # then this epoch is our convergence point.
        if all(abs(loss - final_train_loss) <= tolerance for loss in tail):
            # Convert 0-based index to 1-based epoch number.
            return start_idx + 1

    # Fallback: if no earlier stable point is found, treat the final epoch
    # as the convergence epoch.
    return epochs_ran


def add_derived_metrics(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add the derived values needed for analysis to every run.

    Parameters
    ----------
    results : list of dict
        Raw experiment results loaded from the JSON file. Each run needs
        "train_loss_history" and "val_loss_history" (list of float).

    Returns
    -------
    list of dict
        The same list (modified in place), with "final_train_loss" (float)
        and "convergence_epoch" (int) added to each run,
        "generalization_gap" (float) added when the run records
        "train_metric", and "best_val_loss" (float), "stop_reason" (str)
        and "overfitting_epochs" (int or None) added if the run doesn't
        have them.

    Notes
    -----
    Processing, for each run:
    1. best_val_loss: keep the value saved by
       nn_numpy.modeling.train. It is the validation loss of the
       checkpoint that early stopping restored, i.e. of the model whose
       test metric is reported. Only if it is missing, fall back to the
       minimum of the validation-loss history.
    2. final_train_loss = last value of the training-loss history.
    3. convergence_epoch = convergence_epoch(train_loss_history).
    4. generalization_gap = test_metric - train_metric, if train_metric is
       recorded. A large positive gap means overfitting.
    5. For older results files: derive stop_reason from the "diverged" and
       "stopped_early" flags, and set overfitting_epochs to None (not
       recorded).

    Why not always use the minimum of the history: early stopping only
    saves a checkpoint when the validation loss improves by more than
    min_delta. A later epoch can reach a slightly lower validation loss
    without being saved, so that minimum can belong to a model that was
    never tested. Selecting runs on it would pick a run for a score its
    reported model did not achieve.

    For loss quality, we use best validation loss within each run.
    For convergence speed, we use training-loss convergence epoch.
    """
    for run in results:
        # Validation loss of the restored (and tested) checkpoint.
        run.setdefault("best_val_loss", min(run["val_loss_history"]))

        # Final training loss reached by this run.
        run["final_train_loss"] = run["train_loss_history"][-1]

        # Convergence epoch computed from the training-loss history.
        run["convergence_epoch"] = convergence_epoch(run["train_loss_history"])

        # Generalization gap: test minus training metric (evaluation mode).
        if "train_metric" in run:
            run["generalization_gap"] = run["test_metric"] - run["train_metric"]

        # Older results files don't record how a run ended.
        if "stop_reason" not in run:
            if run.get("diverged"):
                run["stop_reason"] = "diverged"
            elif run.get("stopped_early"):
                run["stop_reason"] = "early_stopping"
            else:
                run["stop_reason"] = "max_epochs"
        run.setdefault("overfitting_epochs", None)

    return results


def add_normalized_best_val_loss(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Min-max normalize best validation loss within each problem separately.

    Parameters
    ----------
    results : list of dict
        Experiment results with derived metrics already added (each run
        needs "problem_name" and "best_val_loss").

    Returns
    -------
    list of dict
        The same list (modified in place), with "normalized_best_val_loss"
        (float in [0, 1]) added to each run.

    Notes
    -----
    Processing:
    1. Group the runs by "problem_name".
    2. Inside each group, find the smallest and largest best_val_loss.
    3. Set normalized = (best_val_loss - min) / (max - min), or 0.0 for
       every run if all values in the group are equal.

    Why: this is necessary for the combined overall comparison because
    classification uses BCE and regression uses MSE. These losses are not on
    the same scale, so they should not be averaged directly across tasks.
    Because the min and max come from the runs passed in, the normalized
    values depend on which runs are included.
    """
    # Group runs by problem so normalization happens inside each task only.
    problem_groups: dict[str, list[dict[str, Any]]] = {}
    for run in results:
        problem_groups.setdefault(run["problem_name"], []).append(run)

    # Normalize best validation loss separately inside each problem group.
    for runs in problem_groups.values():
        values = [r["best_val_loss"] for r in runs]
        min_v = min(values)
        max_v = max(values)

        for r in runs:
            # If all values are identical, assign 0.0 to avoid division by zero.
            if max_v == min_v:
                r["normalized_best_val_loss"] = 0.0
            else:
                # Standard min-max normalization.
                r["normalized_best_val_loss"] = (r["best_val_loss"] - min_v) / (max_v - min_v)

    return results


def filter_by_problem(
    results: list[dict[str, Any]], problem_name: str | None = None
) -> list[dict[str, Any]]:
    """
    Filter runs by problem name.

    Parameters
    ----------
    results : list of dict
        Full results list; each run has a "problem_name" key.
    problem_name : str or None, default=None
        Problem name to keep, "classification" or "regression". If None,
        return all runs.

    Returns
    -------
    list of dict
        The runs whose "problem_name" equals `problem_name` (a new list), or
        the input list itself when `problem_name` is None.

    Notes
    -----
    Processing:
    1. If no problem name is given, return the input unchanged.
    2. Otherwise keep only the runs of that problem.
    """
    if problem_name is None:
        return results
    return [r for r in results if r["problem_name"] == problem_name]


def group_by_key(results: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    """
    Group a list of run dictionaries by a chosen key.

    Parameters
    ----------
    results : list of dict
        Runs to group.
    key : str
        Dictionary key to group by, such as 'optimizer' or 'architecture'.
        Every run must have this key.

    Returns
    -------
    dict of str or int to list of dict
        Dictionary mapping each value of `key` (usually a str; an int for
        "seed") to the list of runs with that value, in their original order.

    Notes
    -----
    Processing:
    1. Start from an empty dictionary.
    2. Append each run to the list stored under its value of `key`,
       creating the list the first time a value is seen.
    """
    groups: dict[Any, list[dict[str, Any]]] = {}
    for run in results:
        groups.setdefault(run[key], []).append(run)
    return groups


def summarize_task_group(group_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build a task-level summary for one group of runs.

    Parameters
    ----------
    group_runs : list of dict
        Non-empty list of runs belonging to one group (same optimizer or
        same architecture), all from the same problem. Each needs
        "convergence_epoch", "best_epoch" and "best_val_loss".

    Returns
    -------
    dict
        Summary statistics for that group:
        - "num_runs" : int, number of runs in the group.
        - "avg_convergence_epoch" : float, mean convergence epoch
          (training-loss plateau).
        - "avg_best_epoch" : float, mean epoch of the restored
          (validation-selected) checkpoint.
        - "avg_best_val_loss" : float, mean best validation loss.

    Notes
    -----
    Processing:
    1. Count the runs.
    2. Average their convergence epochs, best epochs and best validation
       losses.

    This is used for the classification and regression tables. At the task
    level, we can report average best validation loss directly, because all
    runs in the table use the same loss scale.
    """
    return {
        # Number of runs contributing to this group average.
        "num_runs": len(group_runs),
        # Average convergence epoch based on training-loss convergence.
        "avg_convergence_epoch": mean(r["convergence_epoch"] for r in group_runs),
        # Average epoch of the checkpoint selected on validation loss.
        "avg_best_epoch": mean(r["best_epoch"] for r in group_runs),
        # Average best validation loss inside this task.
        "avg_best_val_loss": mean(r["best_val_loss"] for r in group_runs),
    }


def summarize_combined_group(group_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build a combined (both problems) summary for one group of runs.

    Parameters
    ----------
    group_runs : list of dict
        Non-empty list of runs belonging to one group (same optimizer or
        same architecture), possibly from both problems. Each needs
        "convergence_epoch", "best_epoch" and "normalized_best_val_loss".

    Returns
    -------
    dict
        Summary statistics for that group:
        - "num_runs" : int, number of runs in the group.
        - "avg_convergence_epoch" : float, mean convergence epoch
          (training-loss plateau).
        - "avg_best_epoch" : float, mean epoch of the restored
          (validation-selected) checkpoint.
        - "avg_normalized_best_val_loss" : float, mean normalized best
          validation loss.

    Notes
    -----
    Processing:
    1. Count the runs.
    2. Average their convergence epochs, best epochs and normalized best
       validation losses.

    This is used for the final overall answers across all experiments.
    Because classification and regression use different loss scales,
    we use normalized best validation loss here instead of raw loss.
    """
    return {
        # Number of runs contributing to this group average.
        "num_runs": len(group_runs),
        # Average convergence epoch across all runs in the group.
        "avg_convergence_epoch": mean(r["convergence_epoch"] for r in group_runs),
        # Average epoch of the checkpoint selected on validation loss.
        "avg_best_epoch": mean(r["best_epoch"] for r in group_runs),
        # Average normalized best validation loss across all runs in the group.
        "avg_normalized_best_val_loss": mean(r["normalized_best_val_loss"] for r in group_runs),
    }


def ordered_keys(summary_dict: dict[str, Any], preferred_order: list[str]) -> list[str]:
    """
    Return keys from a summary dictionary in a preferred display order.

    Parameters
    ----------
    summary_dict : dict
        Summary dictionary whose keys should be ordered.
    preferred_order : list of str
        Desired order, such as OPTIMIZER_ORDER or ARCHITECTURE_ORDER.

    Returns
    -------
    list of str
        Keys that exist in summary_dict, ordered according to preferred_order.

    Notes
    -----
    Processing:
    1. Walk through `preferred_order` and keep each name that is a key of
       `summary_dict`.

    Keys of `summary_dict` that are not in `preferred_order` are left out,
    so they do not appear in the table.
    """
    return [k for k in preferred_order if k in summary_dict]


def format_task_table(
    title: str, summary_dict: dict[str, dict[str, Any]], preferred_order: list[str]
) -> str:
    """
    Format a plain-text table for one task-specific summary section.

    Parameters
    ----------
    title : str
        Section title.
    summary_dict : dict of str to dict
        Summary statistics by optimizer or architecture, each value as
        returned by summarize_task_group.
    preferred_order : list of str
        Display order for the rows. Groups not listed here are not shown.

    Returns
    -------
    str
        A formatted multi-line string representing the table, ending with a
        blank line.

    Notes
    -----
    Processing:
    1. Write the title and an underline of dashes.
    2. Write the header row (Group, Runs, Avg Conv Epoch, Avg Best Epoch,
       Avg Best Val Loss). The convergence epoch is where the training loss
       plateaus; the best epoch is the checkpoint selected on validation
       loss.
    3. Write one fixed-width row per group in `preferred_order`.
    4. Join the lines with newlines.
    """
    # Start the section with a title and underline.
    lines = [title, "-" * len(title)]

    # Add the table header row.
    lines.append(
        f"{'Group':<15}{'Runs':<8}{'Avg Conv Epoch':<18}{'Avg Best Epoch':<18}"
        f"{'Avg Best Val Loss':<20}"
    )

    # Add one row per group in the requested order.
    for group_name in ordered_keys(summary_dict, preferred_order):
        stats = summary_dict[group_name]
        lines.append(
            f"{group_name:<15}"
            f"{stats['num_runs']:<8}"
            f"{stats['avg_convergence_epoch']:<18.2f}"
            f"{stats['avg_best_epoch']:<18.2f}"
            f"{stats['avg_best_val_loss']:<20.6f}"
        )

    # Add a blank line after the table for readability.
    lines.append("")
    return "\n".join(lines)


def is_diverged(run: dict[str, Any]) -> bool:
    """
    Check whether a run's training diverged.

    Parameters
    ----------
    run : dict
        One experiment run. Reads the optional "diverged" flag and
        "best_val_loss" (float).

    Returns
    -------
    bool
        True if the run is flagged as diverged or its best validation loss
        is not finite (NaN or infinity).

    Notes
    -----
    Processing:
    1. Return True if the "diverged" flag is present and true.
    2. Otherwise return True if best_val_loss is NaN or infinite.

    Why both checks: a diverged run can still report a finite (but
    meaningless) test metric: if the loss was finite but enormous before it
    overflowed, early stopping restores that checkpoint, so the trainer's
    flag is needed. Older results files have no flag, so a non-finite best
    validation loss is used for them.
    """
    return bool(run.get("diverged")) or not math.isfinite(run["best_val_loss"])


def format_peak_gradient_norm(run: dict[str, Any]) -> str:
    """
    Describe the largest gradient norm a run reached, for the diverged-run list.

    Parameters
    ----------
    run : dict
        One run; uses "grad_norm_history" (list of float, the largest
        gradient norm per epoch) if present.

    Returns
    -------
    str
        " | peak gradient norm 3.2e+38 (epoch 1)" for the largest finite
        value, " | gradient norm overflowed (epoch 1)" if no value is finite,
        or "" for older results files without the history.

    Notes
    -----
    Processing:
    1. Return "" if the run has no gradient-norm history.
    2. Otherwise find the largest finite per-epoch norm and its epoch, or
       the first epoch if every value overflowed.
    """
    history = run.get("grad_norm_history")
    if not history:
        return ""
    finite = [(value, epoch) for epoch, value in enumerate(history, 1) if math.isfinite(value)]
    if not finite:
        return " | gradient norm overflowed (epoch 1)"
    value, epoch = max(finite)
    return f" | peak gradient norm {value:.1e} (epoch {epoch})"


def format_dropout_table(title: str, problem_runs: list[dict[str, Any]]) -> str:
    """
    Format a table comparing each architecture with its dropout version.

    Parameters
    ----------
    title : str
        Section title.
    problem_runs : list of dict
        All runs of one problem. Each needs "architecture", "optimizer",
        "learning_rate", "batch", "test_metric", "train_metric" and the
        fields read by is_diverged; "seed" is optional (RANDOM_SEED if
        missing).

    Returns
    -------
    str
        A formatted multi-line string: a title, a header row, two rows
        (without / with dropout) for every pair that has matched runs, and a
        trailing blank line.

    Notes
    -----
    Processing:
    1. Index the runs by (architecture, optimizer, learning rate, batch
       size, seed).
    2. For every (without, with) pair in DROPOUT_PAIRS, match each run of
       the plain architecture with the dropout run that has the same
       optimizer, learning rate, batch size and seed. A match is used only
       if both runs exist and neither diverged.
    3. Count the matches in which dropout gave the lower test metric.
    4. For each architecture of the pair, report:
       - average training metric (evaluation mode, restored best model)
       - average test metric
       - average generalization gap, test minus train
       - (dropout row only) in how many matched runs dropout gave the lower
         test metric
    Pairs with no matched runs are skipped.
    """
    # Index every run by its settings so the dropout partner of a run can
    # be found with one dictionary lookup.
    lookup = {
        (
            r["architecture"],
            r["optimizer"],
            r["learning_rate"],
            r["batch"],
            r.get("seed", RANDOM_SEED),
        ): r
        for r in problem_runs
    }

    lines = [title, "-" * len(title)]
    lines.append(
        f"{'Architecture':<16}"
        f"{'Matched':<9}"
        f"{'Avg Train':<13}"
        f"{'Avg Test':<13}"
        f"{'Avg Gap':<13}"
        f"{'Dropout Better':<15}"
    )

    for without, with_dropout in DROPOUT_PAIRS:
        # Collect (without-dropout run, with-dropout run) pairs that share
        # every other setting and where neither run diverged.
        pairs = []
        for (arch, optimizer, lr, batch, seed), run in lookup.items():
            if arch != without:
                continue
            partner = lookup.get((with_dropout, optimizer, lr, batch, seed))
            if partner is None or is_diverged(run) or is_diverged(partner):
                continue
            pairs.append((run, partner))

        if not pairs:
            continue

        # Number of matched runs in which dropout lowered the test metric.
        dropout_wins = sum(p[1]["test_metric"] < p[0]["test_metric"] for p in pairs)

        # One row for the plain architecture (index 0 of each pair) and one
        # for its dropout version (index 1).
        for index, label in [(0, without), (1, with_dropout)]:
            runs = [p[index] for p in pairs]
            train = mean(r["train_metric"] for r in runs)
            test = mean(r["test_metric"] for r in runs)
            better = f"{dropout_wins}/{len(pairs)}" if index == 1 else ""
            lines.append(
                f"{label:<16}"
                f"{len(runs):<9}"
                f"{train:<13.6f}"
                f"{test:<13.6f}"
                f"{test - train:<13.6f}"
                f"{better:<15}"
            )

    lines.append("")
    return "\n".join(lines)


def format_collapse_check_table(title: str, regression_runs: list[dict[str, Any]]) -> str:
    """
    Format the test MSE of the collapse-prone regression runs for each
    A2 variant.

    Parameters
    ----------
    title : str
        Section title.
    regression_runs : list of dict
        All regression runs to look in (normally of a single seed: if
        several runs share the same architecture, optimizer and batch size,
        only the last one is kept).

    Returns
    -------
    str
        A formatted multi-line string: one row per A2 variant, one column
        per (optimizer, batch size) combination in COLLAPSE_CHECK_RUNS at
        learning rate COLLAPSE_CHECK_LEARNING_RATE (0.1), and a trailing
        blank line.

    Notes
    -----
    Processing:
    1. Index the runs at learning rate 0.1 by (architecture, optimizer,
       batch size).
    2. Write the title, underline and a header with one column per
       (optimizer, batch size) pair.
    3. For each architecture in A2_VARIANT_ORDER that has runs, write one
       row. Each cell is the test MSE to 4 decimals, "diverged" if the run
       diverged, or "-" if the run is missing.
    """
    # Index the runs so each table cell is a direct lookup.
    lookup = {
        (r["architecture"], r["optimizer"], r["batch"]): r
        for r in regression_runs
        if r["learning_rate"] == COLLAPSE_CHECK_LEARNING_RATE
    }

    lines = [title, "-" * len(title)]

    # Header: one column per optimizer / batch-size pair.
    header = f"{'Architecture':<15}"
    for optimizer, batch in COLLAPSE_CHECK_RUNS:
        header += f"{f'{optimizer} bs{batch}':<16}"
    lines.append(header)

    for architecture in A2_VARIANT_ORDER:
        if not any(key[0] == architecture for key in lookup):
            continue
        row = f"{architecture:<15}"
        for optimizer, batch in COLLAPSE_CHECK_RUNS:
            run = lookup.get((architecture, optimizer, batch))
            if run is None:
                cell = "-"
            elif is_diverged(run):
                # The loss overflowed: training diverged instead of collapsing.
                cell = "diverged"
            else:
                cell = f"{run['test_metric']:.4f}"
            row += f"{cell:<16}"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def format_combined_table(
    title: str, summary_dict: dict[str, dict[str, Any]], preferred_order: list[str]
) -> str:
    """
    Format a plain-text table for one combined-overall summary section.

    Parameters
    ----------
    title : str
        Section title.
    summary_dict : dict of str to dict
        Summary statistics by optimizer or architecture, each value as
        returned by summarize_combined_group.
    preferred_order : list of str
        Display order for the rows. Groups not listed here are not shown.

    Returns
    -------
    str
        A formatted multi-line string representing the table, ending with a
        blank line.

    Notes
    -----
    Processing:
    1. Write the title and an underline of dashes.
    2. Write the header row (Group, Runs, Avg Conv Epoch, Avg Best Epoch,
       Avg Norm Best Val Loss).
    3. Write one fixed-width row per group in `preferred_order`.
    4. Join the lines with newlines.
    """
    # Start the section with a title and underline.
    lines = [title, "-" * len(title)]

    # Add the table header row.
    lines.append(
        f"{'Group':<15}{'Runs':<8}{'Avg Conv Epoch':<18}{'Avg Best Epoch':<18}"
        f"{'Avg Norm Best Val Loss':<24}"
    )

    # Add one row per group in the requested order.
    for group_name in ordered_keys(summary_dict, preferred_order):
        stats = summary_dict[group_name]
        lines.append(
            f"{group_name:<15}"
            f"{stats['num_runs']:<8}"
            f"{stats['avg_convergence_epoch']:<18.2f}"
            f"{stats['avg_best_epoch']:<18.2f}"
            f"{stats['avg_normalized_best_val_loss']:<24.6f}"
        )

    # Add a blank line after the table for readability.
    lines.append("")
    return "\n".join(lines)


def best_group(summary_dict: dict[str, dict[str, Any]], field_name: str) -> str:
    """
    Return the group name with the smallest value for a chosen summary field.

    Parameters
    ----------
    summary_dict : dict of str to dict
        Non-empty summary statistics by group, e.g. the output of
        summarize_combined_group for each optimizer.
    field_name : str
        Name of the field to minimize, e.g. "avg_convergence_epoch".

    Returns
    -------
    str
        Name of the best group under that criterion. On a tie, the first
        group in the dictionary's order wins.

    Notes
    -----
    Processing:
    1. Compare the groups by summary_dict[group][field_name].
    2. Return the name of the group with the smallest value.

    Smallest is best because a lower convergence epoch means faster
    convergence and a lower loss means better loss quality.
    """
    return min(summary_dict.items(), key=lambda x: x[1][field_name])[0]


def architecture_name_to_depth(architecture_name: str) -> str:
    """
    Convert architecture code names into more readable depth labels.

    Parameters
    ----------
    architecture_name : str
        Architecture identifier, such as A1 or A2.

    Returns
    -------
    str
        Human-readable description of the architecture depth:
        "A1 (1 hidden layer)", "A2 (3 hidden layers)", or the name
        unchanged for any other architecture.

    Notes
    -----
    Processing:
    1. Return a fixed label for "A1" or "A2".
    2. Return any other name as it is.
    """
    if architecture_name == "A1":
        return "A1 (1 hidden layer)"
    if architecture_name == "A2":
        return "A2 (3 hidden layers)"
    return architecture_name


def build_depth_effect_sentence(combined_architecture_summary: dict[str, dict[str, Any]]) -> str:
    """
    Build a short verbal answer for how network depth affects optimization overall.

    Parameters
    ----------
    combined_architecture_summary : dict of str to dict
        Combined summary for architectures across all experiments, each
        value as returned by summarize_combined_group.

    Returns
    -------
    str
        One sentence (or short pair of clauses) describing the depth effect.

    Notes
    -----
    Processing:
    1. Find the architecture with the lowest average convergence epoch.
    2. Find the architecture with the lowest average normalized best
       validation loss.
    3. Turn both names into readable depth labels.
    4. If they are the same architecture, say it wins on both; otherwise
       describe the trade-off.
    """
    # Find the architecture with the lower average convergence epoch.
    faster_arch = best_group(combined_architecture_summary, "avg_convergence_epoch")

    # Find the architecture with the lower average normalized best validation loss.
    better_arch = best_group(combined_architecture_summary, "avg_normalized_best_val_loss")

    # Convert shorthand architecture names into readable labels.
    faster_label = architecture_name_to_depth(faster_arch)
    better_label = architecture_name_to_depth(better_arch)

    # If the same architecture is best on both criteria, say so directly.
    if faster_arch == better_arch:
        return (
            f"Overall, {faster_label} both converges faster on average "
            f"and achieves the better average normalized loss."
        )

    # Otherwise explain the trade-off.
    return (
        f"Overall, {faster_label} converges faster on average, "
        f"while {better_label} achieves the better average normalized loss. "
    )


# ------------------------------------------------------------
# Multi-seed analysis
# ------------------------------------------------------------
# Each seed changes the train / validation / test split, the weight
# initialization, the dropout masks and the shuffling. Reporting the mean
# and standard deviation across seeds shows which conclusions hold up and
# which were down to one lucky (or unlucky) draw.


def mean_pm_std(values: Iterable[float], digits: int = 4) -> str:
    """
    Format values as "mean ± std" (sample standard deviation).

    Parameters
    ----------
    values : iterable of float
        One value per seed (any iterable; it is turned into a list).
    digits : int, default=4
        Decimal places used for both the mean and the standard deviation.

    Returns
    -------
    str
        For example "0.3745 ± 0.0512", or "-" if there are no values.

    Notes
    -----
    Processing:
    1. Convert the values to a list; return "-" if it is empty.
    2. Compute the sample standard deviation, or 0.0 when there is only one
       value (the sample formula needs at least two).
    3. Format the mean and standard deviation with `digits` decimals.
    """
    values = list(values)
    if not values:
        return "-"
    spread = stdev(values) if len(values) > 1 else 0.0
    return f"{mean(values):.{digits}f} ± {spread:.{digits}f}"


def runs_by_seed(runs: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """
    Group runs by their seed.

    Parameters
    ----------
    runs : list of dict
        Experiment runs. The "seed" key is optional.

    Returns
    -------
    dict of int to list of dict
        Maps each seed to the runs with that seed. The runs are shallow
        copies of the input dictionaries; they are run records, with the
        keys listed in nn_numpy.modeling.train.run_single_experiment.

    Notes
    -----
    Processing:
    1. Copy each run, filling in "seed" with RANDOM_SEED when it is missing
       (older results files were produced with a single seed and no seed
       field).
    2. Group the copies by "seed" with group_by_key.
    """
    return group_by_key([{**r, "seed": r.get("seed", RANDOM_SEED)} for r in runs], "seed")


def select_by_validation(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Pick the run with the lowest best validation loss, ignoring diverged runs.

    Parameters
    ----------
    runs : list of dict
        Candidate runs. Each needs "best_val_loss" and the fields read by
        is_diverged.

    Returns
    -------
    dict or None
        The selected run, or None if every run diverged (or the list is
        empty). A run record, with the keys listed in
        nn_numpy.modeling.train.run_single_experiment.

    Notes
    -----
    Processing:
    1. Drop the diverged runs.
    2. Return the remaining run with the smallest best_val_loss.

    This is the project's model-selection rule: the test set is never used
    to choose a model.
    """
    candidates = [r for r in runs if not is_diverged(r)]
    return min(candidates, key=lambda r: r["best_val_loss"]) if candidates else None


def error_removed(run: dict[str, Any]) -> float:
    """
    Compute the fraction of the constant-prediction baseline's error removed.

    Parameters
    ----------
    run : dict
        One experiment run with "test_metric" and "baseline_test_metric"
        (both float).

    Returns
    -------
    float
        1 - test_metric / baseline_test_metric. 1.0 means a perfect model,
        0.0 means no better than the constant baseline, and a negative value
        means worse than the baseline.

    Notes
    -----
    Processing:
    1. Divide the run's test metric by the baseline's test metric.
    2. Subtract the ratio from 1.

    For regression (MSE) this is R²; for classification (BCE), McFadden's
    pseudo-R².
    """
    return 1.0 - run["test_metric"] / run["baseline_test_metric"]


def fails_baseline(run: dict[str, Any]) -> bool:
    """
    Check whether a run is no better than always predicting a constant.

    Parameters
    ----------
    run : dict
        One experiment run with "test_metric" and "baseline_test_metric"
        (both float).

    Returns
    -------
    bool
        True if the run's test metric is greater than or equal to the
        constant-prediction baseline's test metric.

    Notes
    -----
    Processing:
    1. Compare test_metric with baseline_test_metric (lower is better for
       both BCE and MSE).
    """
    return run["test_metric"] >= run["baseline_test_metric"]


def format_selected_model_table(
    title: str, problem_runs: list[dict[str, Any]], architectures: list[str], digits: int
) -> str:
    """
    Format a table of the model selected for each seed.

    Parameters
    ----------
    title : str
        Section title.
    problem_runs : list of dict
        All runs of one problem, from every seed. Each needs "architecture",
        "optimizer", "learning_rate", "batch", "best_val_loss",
        "test_metric" and "baseline_test_metric".
    architectures : list of str
        Architectures the selection may choose from, e.g. ["A1", "A2"].
    digits : int
        Decimal places used for the test and baseline metrics.

    Returns
    -------
    str
        A formatted multi-line string: a title, a header, one row per seed
        (selected run, test metric, baseline, error removed), a "Mean" row
        with mean ± std of the test metric and of the error removed (in
        percent), and a trailing blank line.

    Notes
    -----
    Processing:
    1. Group the runs by seed, in increasing seed order.
    2. For each seed, select the run with the lowest best validation loss
       among the given architectures, ignoring diverged runs. Seeds with no
       usable run are skipped.
    3. Write that run's settings, test metric, baseline and error removed.
    4. Write the mean ± sample standard deviation across the selected runs.
    """
    lines = [title, "-" * len(title)]
    lines.append(
        f"{'Seed':<6}{'Selected run':<48}{'Test':<12}{'Baseline':<12}{'Error removed':<14}"
    )

    selected = []
    for seed, runs in sorted(runs_by_seed(problem_runs).items()):
        best = select_by_validation([r for r in runs if r["architecture"] in architectures])
        if best is None:
            continue
        selected.append(best)
        label = (
            f"{best['architecture']} · {best['optimizer']} · "
            f"LR {best['learning_rate']} · bs {best['batch']}"
        )
        lines.append(
            f"{seed:<6}{label:<48}{best['test_metric']:<12.{digits}f}"
            f"{best['baseline_test_metric']:<12.{digits}f}{error_removed(best):<14.2%}"
        )

    lines.append(
        f"{'Mean':<6}{'':<48}"
        f"{mean_pm_std([r['test_metric'] for r in selected], digits):<24}"
        f"{mean_pm_std([100 * error_removed(r) for r in selected], 2)} %"
    )
    lines.append("")
    return "\n".join(lines)


def format_architecture_seed_table(
    title: str, problem_runs: list[dict[str, Any]], digits: int
) -> str:
    """
    Format a table summarizing each architecture across seeds.

    Parameters
    ----------
    title : str
        Section title.
    problem_runs : list of dict
        All runs of one problem, from every seed. Each needs "architecture",
        "best_val_loss", "convergence_epoch", "best_epoch", "test_metric"
        and "baseline_test_metric".
    digits : int
        Decimal places used for the test metric.

    Returns
    -------
    str
        A formatted multi-line string with one row per architecture and a
        trailing blank line.

    Notes
    -----
    Processing, for each architecture (baseline A1 / A2 first, then the A2
    variants, then the dropout architectures; others are not shown):
    1. Per seed, select the best run by validation loss and report the mean
       ± std of their test metrics across seeds.
    2. Per seed, average the convergence epoch (training-loss plateau) and
       the best epoch (validation-selected checkpoint) over runs that didn't
       diverge, and report the mean ± std of those averages.
    3. Count the non-diverged runs that were no better than the constant
       baseline, and the diverged runs, each out of all its runs.
    """
    # Display order: baseline architectures, then A2 variants, then the
    # dropout architectures.
    order = (
        ARCHITECTURE_ORDER
        + [a for a in A2_VARIANT_ORDER if a not in ARCHITECTURE_ORDER]
        + [with_dropout for _, with_dropout in DROPOUT_PAIRS]
    )
    by_arch = group_by_key(problem_runs, "architecture")

    lines = [title, "-" * len(title)]
    lines.append(
        f"{'Architecture':<15}{'Best run test (mean ± std)':<30}"
        f"{'Avg conv epoch':<22}{'Avg best epoch':<22}{'No better than baseline':<25}"
        f"{'Diverged':<10}"
    )

    for arch in [a for a in order if a in by_arch]:
        runs = by_arch[arch]
        best_per_seed = [select_by_validation(rs) for rs in runs_by_seed(runs).values()]
        best_per_seed = [r for r in best_per_seed if r is not None]

        # Average convergence epoch per seed, over runs that didn't diverge.
        conv_per_seed = [
            mean(r["convergence_epoch"] for r in rs if not is_diverged(r))
            for rs in runs_by_seed(runs).values()
            if any(not is_diverged(r) for r in rs)
        ]
        best_epoch_per_seed = [
            mean(r["best_epoch"] for r in rs if not is_diverged(r))
            for rs in runs_by_seed(runs).values()
            if any(not is_diverged(r) for r in rs)
        ]
        failed = sum(fails_baseline(r) for r in runs if not is_diverged(r))
        diverged = sum(is_diverged(r) for r in runs)

        lines.append(
            f"{arch:<15}{mean_pm_std([r['test_metric'] for r in best_per_seed], digits):<30}"
            f"{mean_pm_std(conv_per_seed, 1):<22}{mean_pm_std(best_epoch_per_seed, 1):<22}"
            f"{f'{failed}/{len(runs)}':<25}{f'{diverged}/{len(runs)}':<10}"
        )

    lines.append("")
    return "\n".join(lines)


def format_stopping_seed_table(title: str, problem_runs: list[dict[str, Any]], digits: int) -> str:
    """
    Format a table of how each architecture's runs ended, across seeds.

    Parameters
    ----------
    title : str
        Section title.
    problem_runs : list of dict
        All runs of one problem, from every seed, with derived metrics added
        (see add_derived_metrics). Each needs "architecture", "stop_reason",
        "overfitting_epochs" (int or None), "best_val_loss" and
        "test_metric", and optionally "generalization_gap".
    digits : int
        Decimal places used for the generalization gap.

    Returns
    -------
    str
        A formatted multi-line string with one row per architecture: how
        many runs ended by early stopping, at the maximum number of epochs
        or by diverging; how many early-stopped runs showed the overfitting
        signature when they stopped; and the generalization gap (test minus
        train metric) of the run selected per seed, as mean ± std. Ends with
        a blank line.

    Notes
    -----
    Processing, for each architecture (same order as the architecture
    table; others are not shown):
    1. Count the runs per stop reason.
    2. Among the early-stopped runs, count those with overfitting_epochs > 0,
       i.e. whose training loss was still improving while the validation
       loss was not. Older results files don't record it ("n/a").
    3. Per seed, select the best run by validation loss and report the mean
       ± std of its generalization gap.

    The overfitting count uses the trainer's min_delta as the size of a
    meaningful improvement, so runs whose losses are already far below
    min_delta can't show the signature; the gap covers those.
    """
    order = (
        ARCHITECTURE_ORDER
        + [a for a in A2_VARIANT_ORDER if a not in ARCHITECTURE_ORDER]
        + [with_dropout for _, with_dropout in DROPOUT_PAIRS]
    )
    by_arch = group_by_key(problem_runs, "architecture")

    lines = [title, "-" * len(title)]
    lines.append(
        f"{'Architecture':<15}{'Early / max / diverged':<26}"
        f"{'Overfitting at stop':<22}{'Gap of selected run (test - train)':<36}"
    )

    for arch in [a for a in order if a in by_arch]:
        runs = by_arch[arch]
        counts = {
            reason: sum(r["stop_reason"] == reason for r in runs)
            for reason in ("early_stopping", "max_epochs", "diverged")
        }
        stops = f"{counts['early_stopping']} / {counts['max_epochs']} / {counts['diverged']}"

        early = [r for r in runs if r["stop_reason"] == "early_stopping"]
        if all(r["overfitting_epochs"] is not None for r in early):
            overfit = f"{sum(r['overfitting_epochs'] > 0 for r in early)} of {len(early)}"
        else:
            overfit = "n/a"

        best_per_seed = [select_by_validation(rs) for rs in runs_by_seed(runs).values()]
        gaps = [
            r["generalization_gap"]
            for r in best_per_seed
            if r is not None and "generalization_gap" in r
        ]

        lines.append(f"{arch:<15}{stops:<26}{overfit:<22}{mean_pm_std(gaps, digits):<36}")

    lines.append("")
    return "\n".join(lines)


def format_optimizer_lr_seed_table(
    title: str,
    problem_runs: list[dict[str, Any]],
    digits: int,
    optimizers: list[str] | None = None,
) -> str:
    """
    Format an optimizer x learning-rate table of test metrics across seeds.

    Parameters
    ----------
    title : str
        Section title.
    problem_runs : list of dict
        All runs of one problem, from every seed. Each needs "architecture",
        "optimizer", "learning_rate" and "test_metric".
    digits : int
        Decimal places used for the test metric.
    optimizers : list of str or None, default=None
        Optimizers to show, one row each, in this order. None means
        OPTIMIZER_ORDER.

    Returns
    -------
    str
        A formatted multi-line string: one row per optimizer, one column per
        learning rate (largest first), and a trailing blank line.

    Notes
    -----
    Processing:
    1. Keep only non-diverged runs of the baseline architectures (A1, A2).
    2. For each optimizer and learning rate, average the test metric of the
       matching runs within each seed (over architectures and batch sizes).
    3. Report the mean ± std of those per-seed averages across seeds ("-"
       if there are no runs for that cell).
    """
    runs = [
        r for r in problem_runs if r["architecture"] in ARCHITECTURE_ORDER and not is_diverged(r)
    ]
    learning_rates = sorted({r["learning_rate"] for r in runs}, reverse=True)

    lines = [title, "-" * len(title)]
    lines.append(f"{'Optimizer':<12}" + "".join(f"{f'LR {lr}':<26}" for lr in learning_rates))

    for optimizer in optimizers or OPTIMIZER_ORDER:
        row = f"{optimizer:<12}"
        for lr in learning_rates:
            cell_runs = [
                r for r in runs if r["optimizer"] == optimizer and r["learning_rate"] == lr
            ]
            per_seed = [
                mean(r["test_metric"] for r in rs) for rs in runs_by_seed(cell_runs).values()
            ]
            row += f"{mean_pm_std(per_seed, digits):<26}"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def format_collapse_seed_table(title: str, regression_runs: list[dict[str, Any]]) -> str:
    """
    Format a table counting collapsed, diverged and learning runs per A2 variant.

    Parameters
    ----------
    title : str
        Section title.
    regression_runs : list of dict
        All regression runs, from every seed. Each needs "architecture",
        "optimizer", "batch", "learning_rate", "test_metric" and
        "baseline_test_metric".

    Returns
    -------
    str
        A formatted multi-line string with one row per A2 variant (columns
        Learned, No better than mean, Diverged, each as count/total) and a
        trailing blank line.

    Notes
    -----
    Processing:
    1. Keep the runs at the settings where the baseline A2 collapses: SGD
       and momentum at LR 0.1, batch 16 and 64 (COLLAPSE_CHECK_RUNS).
    2. Group them by architecture.
    3. For each A2 variant, count across all seeds the runs that diverged,
       the non-diverged runs that failed to beat the mean-prediction
       baseline, and the rest, which learned.
    """
    runs = [
        r
        for r in regression_runs
        if r["learning_rate"] == COLLAPSE_CHECK_LEARNING_RATE
        and (r["optimizer"], r["batch"]) in COLLAPSE_CHECK_RUNS
    ]
    by_arch = group_by_key(runs, "architecture")

    lines = [title, "-" * len(title)]
    lines.append(f"{'Architecture':<15}{'Learned':<12}{'No better than mean':<22}{'Diverged':<10}")
    for arch in [a for a in A2_VARIANT_ORDER if a in by_arch]:
        arch_runs = by_arch[arch]
        diverged = sum(is_diverged(r) for r in arch_runs)
        failed = sum(fails_baseline(r) for r in arch_runs if not is_diverged(r))
        learned = len(arch_runs) - diverged - failed
        total = len(arch_runs)
        lines.append(
            f"{arch:<15}{f'{learned}/{total}':<12}{f'{failed}/{total}':<22}{f'{diverged}/{total}':<10}"
        )
    lines.append("")
    return "\n".join(lines)


def build_multi_seed_section(results: list[dict[str, Any]]) -> str:
    """
    Build the "Multi-Seed Results" section of the report.

    Parameters
    ----------
    results : list of dict
        Runs from every seed and every architecture, with derived metrics
        already added (see add_derived_metrics).

    Returns
    -------
    str
        The section as one block of text.

    Notes
    -----
    Processing:
    1. Write a heading and two paragraphs explaining the seeds, the
       constant-prediction baseline and "error removed".
    2. For each problem whose runs record a baseline, add: the selected
       model per seed (baseline architectures, then all architectures), the
       architecture summary across seeds, how the runs ended (stopping and
       overfitting), and the optimizer x learning-rate table.
    3. Add the regression collapse-check counts across seeds.
    4. If any dropout run records a training metric, add the dropout
       comparison for each problem over all matched runs.
    """
    seeds = sorted({r.get("seed", RANDOM_SEED) for r in results})
    lines = ["Multi-Seed Results", "=================="]
    lines.append(
        f"Every configuration was run with {len(seeds)} seeds ({', '.join(map(str, seeds))}). "
        "Each seed changes the train / validation / test split, the weight initialization, "
        "the dropout masks and the shuffling. All sections above use seed "
        f"{RANDOM_SEED} only; this section reports mean ± sample standard deviation across seeds."
    )
    lines.append("")
    lines.append(
        "The constant-prediction baseline is the test metric of a model that ignores the "
        "features: the training class proportion for classification, the training mean for "
        "regression. 'Error removed' is 1 - test / baseline (R² for regression, McFadden's "
        "pseudo-R² for classification). Models are always selected by validation loss, "
        "never by test score, and diverged runs are excluded from selection and averages."
    )
    lines.append("")

    for problem_name, loss_name, digits in [
        ("classification", "BCE", 5),
        ("regression", "MSE", 4),
    ]:
        problem_runs = filter_by_problem(results, problem_name)
        # Older results files have no baseline, which these tables need.
        if not problem_runs or "baseline_test_metric" not in problem_runs[0]:
            continue
        label = problem_name.capitalize()

        lines.append(
            format_selected_model_table(
                f"Selected Model per Seed — {label} ({loss_name}), baseline architectures A1 / A2",
                problem_runs,
                ARCHITECTURE_ORDER,
                digits,
            )
        )
        lines.append(
            format_selected_model_table(
                f"Selected Model per Seed — {label} ({loss_name}), all architectures",
                problem_runs,
                list({r["architecture"] for r in problem_runs}),
                digits,
            )
        )
        lines.append(
            format_architecture_seed_table(
                f"Architectures Across Seeds — {label} ({loss_name})", problem_runs, digits
            )
        )
        lines.append(
            format_stopping_seed_table(
                f"Stopping and Overfitting Across Seeds — {label} ({loss_name})",
                problem_runs,
                digits,
            )
        )
        lines.append(
            format_optimizer_lr_seed_table(
                f"Optimizer x Learning Rate Across Seeds — {label} ({loss_name}), A1 / A2, average test metric",
                problem_runs,
                digits,
            )
        )

    lines.append(
        format_collapse_seed_table(
            "Collapse Check Across Seeds — Regression, SGD / Momentum at LR 0.1, batch 16 / 64",
            filter_by_problem(results, "regression"),
        )
    )

    dropout_names = {with_dropout for _, with_dropout in DROPOUT_PAIRS}
    if any(r["architecture"] in dropout_names and "train_metric" in r for r in results):
        for problem_name, loss_name in [("classification", "BCE"), ("regression", "MSE")]:
            lines.append(
                format_dropout_table(
                    f"Dropout Across Seeds — {problem_name.capitalize()} ({loss_name}), all matched runs",
                    filter_by_problem(results, problem_name),
                )
            )

    return "\n".join(lines)


def build_muon_section(results: list[dict[str, Any]]) -> str:
    """
    Build the Muon section: how Muon compares with the baseline optimizers.

    Parameters
    ----------
    results : list of dict
        Runs from every seed, architecture and optimizer, with derived
        metrics added. Each needs "problem_name", "architecture",
        "optimizer", "learning_rate", "batch", "seed" and the metrics used
        by the summary tables.

    Returns
    -------
    str
        The section as a multi-line string, or "" when no run uses Muon.

    Notes
    -----
    Processing:
    1. Return "" if there are no Muon runs.
    2. Write the section title and a short explanation.
    3. For each problem:
       a. summarize the first seed's non-diverged A1 / A2 runs by optimizer,
          Muon included, and note how many runs diverged;
       b. add the optimizer x learning-rate table across seeds;
       c. add the selected model per seed over every architecture and
          optimizer, so a Muon run can win.

    The other sections use the baseline optimizers only, so their numbers
    stay comparable with earlier results files.
    """
    if not any(r["optimizer"] == "muon" for r in results):
        return ""

    lines = ["Muon — Orthogonalized Momentum", "------------------------------"]
    lines.append(
        "Muon keeps a Nesterov momentum buffer (0.95) for every Dense weight matrix and "
        "replaces the step direction with an approximately orthogonal matrix (5 Newton-Schulz "
        "steps), so every direction of the matrix gets a step of similar size. Biases and "
        "batch-norm parameters use AdaBelief. These runs are not included in the other "
        "sections; the tables below compare all four optimizers on the same runs."
    )
    lines.append("")

    seed_runs = [r for r in results if r.get("seed", RANDOM_SEED) == RANDOM_SEED]
    for problem_name, loss_name, digits in [
        ("classification", "BCE", 5),
        ("regression", "MSE", 4),
    ]:
        label = problem_name.capitalize()
        baseline_runs = [
            r
            for r in filter_by_problem(seed_runs, problem_name)
            if r["architecture"] in ARCHITECTURE_ORDER
        ]
        finite_runs = [r for r in baseline_runs if not is_diverged(r)]
        summary = {
            name: summarize_task_group(runs)
            for name, runs in group_by_key(finite_runs, "optimizer").items()
        }
        lines.append(
            format_task_table(
                f"Optimizer Summary with Muon — {label} ({loss_name}), A1 / A2, seed {RANDOM_SEED}",
                summary,
                MUON_SECTION_OPTIMIZERS,
            )
        )
        diverged = len(baseline_runs) - len(finite_runs)
        if diverged:
            lines.append(f"Diverged runs excluded from the table above: {diverged}")
            lines.append("")

        problem_runs = filter_by_problem(results, problem_name)
        if len({r.get("seed", RANDOM_SEED) for r in problem_runs}) > 1:
            lines.append(
                format_optimizer_lr_seed_table(
                    f"Optimizer x Learning Rate Across Seeds with Muon — {label} ({loss_name}), A1 / A2, average test metric",
                    problem_runs,
                    digits,
                    MUON_SECTION_OPTIMIZERS,
                )
            )
        if problem_runs and "baseline_test_metric" in problem_runs[0]:
            lines.append(
                format_selected_model_table(
                    f"Selected Model per Seed with Muon — {label} ({loss_name}), all architectures and optimizers",
                    problem_runs,
                    list({r["architecture"] for r in problem_runs}),
                    digits,
                )
            )

    return "\n".join(lines)


def main(results_path: str | None = None) -> None:
    """
    Build the aggregate analysis report and write it to disk.

    Parameters
    ----------
    results_path : str or None, default=None
        Path to a main_results_full_<stamp>.json file. When None, the
        newest stamped results file in reports/ is used.

    Returns
    -------
    None
        Writes the report to OUTPUT_PATH (reports/analysis_<stamp>.txt) and
        prints the results file used and the report path.

    Raises
    ------
    FileNotFoundError
        If no results path is given and reports/ has no stamped results file.

    Notes
    -----
    Processing:
    1. Load the results and add the derived metrics to every run.
    2. Keep the runs of the first seed (RANDOM_SEED) for every section
       except "Multi-Seed Results", and the baseline architectures (A1, A2)
       for the main sections.
    3. Normalize best validation loss within each problem.
    4. Summarize the runs by optimizer and by architecture, per problem and
       combined.
    5. Answer the three questions: fastest optimizer, best optimizer by
       loss, and the effect of depth.
    6. Assemble the report text: method, per-problem tables, combined
       tables, answers, and (when such runs exist) the A2-variant, dropout
       and multi-seed sections.
    7. Write the report to disk.
    """
    # ------------------------------------------------------------
    # 1. Load experiment results from local repo.
    # ------------------------------------------------------------
    results_path = resolve_results_path(results_path)
    print(f"Run stamp: {RUN_STAMP}")
    print(f"Using results file: {results_path}")
    all_results = load_results(results_path)

    # ------------------------------------------------------------
    # 2. Add derived metrics needed for analysis.
    # ------------------------------------------------------------
    # This adds:
    # - best_val_loss
    # - final_train_loss
    # - convergence_epoch
    all_results = add_derived_metrics(all_results)

    # Every section except "Multi-Seed Results" uses the first seed only
    # (RANDOM_SEED, 42), so their numbers stay comparable with single-seed
    # results files. Older files have no "seed" field: they are one seed.
    # Every section except "Muon" uses the baseline optimizers only, so
    # adding Muon to the configs does not change their numbers.
    every_optimizer_results = all_results
    multi_seed_results = [r for r in all_results if r["optimizer"] in OPTIMIZER_ORDER]
    all_results = [r for r in multi_seed_results if r.get("seed", RANDOM_SEED) == RANDOM_SEED]

    # The main sections use the baseline architectures only (A1, A2).
    # Normalization is min-max over the runs it is given, so it must also
    # use only these runs, or the extra architectures would shift the
    # baseline's normalized numbers.
    results = [r for r in all_results if r["architecture"] in ARCHITECTURE_ORDER]

    # ------------------------------------------------------------
    # 3. Add normalized loss values for combined overall comparisons.
    # ------------------------------------------------------------
    results = add_normalized_best_val_loss(results)

    # ------------------------------------------------------------
    # 4. Split runs by task for supporting per-task tables.
    # ------------------------------------------------------------
    classification_runs = filter_by_problem(results, "classification")
    regression_runs = filter_by_problem(results, "regression")

    # ------------------------------------------------------------
    # 5. Build classification summaries by optimizer and architecture.
    # ------------------------------------------------------------
    cls_optimizer_summary = {
        name: summarize_task_group(runs)
        for name, runs in group_by_key(classification_runs, "optimizer").items()
    }
    cls_architecture_summary = {
        name: summarize_task_group(runs)
        for name, runs in group_by_key(classification_runs, "architecture").items()
    }

    # ------------------------------------------------------------
    # 6. Build regression summaries by optimizer and architecture.
    # ------------------------------------------------------------
    reg_optimizer_summary = {
        name: summarize_task_group(runs)
        for name, runs in group_by_key(regression_runs, "optimizer").items()
    }
    reg_architecture_summary = {
        name: summarize_task_group(runs)
        for name, runs in group_by_key(regression_runs, "architecture").items()
    }

    # ------------------------------------------------------------
    # 7. Build combined overall summaries across all experiments.
    # ------------------------------------------------------------
    combined_optimizer_summary = {
        name: summarize_combined_group(runs)
        for name, runs in group_by_key(results, "optimizer").items()
    }
    combined_architecture_summary = {
        name: summarize_combined_group(runs)
        for name, runs in group_by_key(results, "architecture").items()
    }

    # ------------------------------------------------------------
    # 8. Compute the final overall answers.
    # ------------------------------------------------------------
    # Which optimizer converges fastest?
    overall_fastest_optimizer = best_group(combined_optimizer_summary, "avg_convergence_epoch")

    # Which optimizer gives the best loss value?
    # For the combined comparison, this uses normalized best validation loss.
    overall_best_optimizer = best_group(combined_optimizer_summary, "avg_normalized_best_val_loss")

    # How does depth affect optimization?
    depth_effect_sentence = build_depth_effect_sentence(combined_architecture_summary)

    # ------------------------------------------------------------
    # 9. Build the final report text as a list of lines.
    # ------------------------------------------------------------
    lines = []

    # Report title.
    lines.append("ANALYSIS")
    lines.append("================")
    lines.append("")

    # Explain the analysis method used.
    lines.append("Method")
    lines.append("------")
    lines.append(
        "For each run, convergence epoch is the first epoch after which all remaining "
        "training-loss values stay within a small tolerance of the final training loss."
    )
    lines.append("")
    lines.append(
        "Convergence epoch (training-loss plateau) and best epoch (the checkpoint with the "
        "lowest validation loss, which early stopping restores and which is tested) answer "
        "different questions: how fast the optimizer settles, and when the model was best. "
        "Early stopping is driven by validation loss only. Overfitting shows as training "
        "loss still improving while validation loss doesn't, and as a large test - train gap."
    )

    lines.append("")
    lines.append(
        "For overall loss comparison across classification and regression, combined results use "
        "normalized best validation loss within each problem separately."
    )

    lines.append("")

    # Add supporting classification tables.
    lines.append("Supporting Data — Classification")
    lines.append("--------------------------------")
    lines.append("Loss : BCE")
    lines.append("")
    lines.append(
        format_task_table(
            "Optimizer Summary — Classification",
            cls_optimizer_summary,
            OPTIMIZER_ORDER,
        )
    )

    lines.append(
        format_task_table(
            "Architecture Summary — Classification",
            cls_architecture_summary,
            ARCHITECTURE_ORDER,
        )
    )

    # Add supporting regression tables.
    lines.append("Supporting Data — Regression")
    lines.append("----------------------------")
    lines.append("Loss: MSE")
    lines.append("")
    lines.append(
        format_task_table(
            "Optimizer Summary — Regression",
            reg_optimizer_summary,
            OPTIMIZER_ORDER,
        )
    )

    lines.append(
        format_task_table(
            "Architecture Summary — Regression",
            reg_architecture_summary,
            ARCHITECTURE_ORDER,
        )
    )

    # Add combined overall tables used for the final answers.
    lines.append("Supporting Data — Combined Overall")
    lines.append("---------------------------------")
    lines.append("Loss values below are normalized within each problem before averaging.")
    lines.append("")
    lines.append(
        format_combined_table(
            "Optimizer Summary — Combined Overall",
            combined_optimizer_summary,
            OPTIMIZER_ORDER,
        )
    )

    lines.append(
        format_combined_table(
            "Architecture Summary — Combined Overall",
            combined_architecture_summary,
            ARCHITECTURE_ORDER,
        )
    )

    # Add the final direct answers to the three questions.
    lines.append("Answers")
    lines.append("-------")
    lines.append(
        f"1. Fastest optimizer on average over all experiments: {overall_fastest_optimizer}"
    )
    lines.append(
        f"2. Best optimizer by average loss-function value over all experiments using average normalized best validation loss: {overall_best_optimizer} "
    )
    lines.append(f"3. How network depth affects optimization: {depth_effect_sentence}")
    lines.append("")

    # Add the A2 variant section when the results contain any variants.
    variant_runs = [r for r in all_results if r["architecture"] in A2_VARIANT_ORDER[1:]]
    if variant_runs:
        a2_family = [r for r in all_results if r["architecture"] in A2_VARIANT_ORDER]

        lines.append("A2 Variants — Bias Terms, Weight Initialization and Batch Normalization")
        lines.append("----------------------------------------------------------------------")
        lines.append(
            "Each variant changes A2 (3 hidden ReLU layers): "
            "A2-bias adds a bias term to every layer, A2-he uses He initialization "
            "instead of N(0, 0.1^2), A2-bias-he does both, and A2-bn adds batch "
            "normalization between each hidden Dense layer and its ReLU, and A2-clip trains "
            "A2 with global gradient-norm clipping at 20 (every mini-batch gradient is "
            "scaled down to a norm of at most 20 before the update). "
            "These runs are not included in the sections above."
        )
        lines.append("")

        for problem_name, loss_name in [("classification", "BCE"), ("regression", "MSE")]:
            problem_runs = filter_by_problem(a2_family, problem_name)

            # Diverged runs would make every average NaN or meaningless,
            # so they are listed separately instead.
            finite_runs = [r for r in problem_runs if not is_diverged(r)]
            diverged_runs = [r for r in problem_runs if is_diverged(r)]

            family_summary = {
                name: summarize_task_group(runs)
                for name, runs in group_by_key(finite_runs, "architecture").items()
            }
            lines.append(
                format_task_table(
                    f"A2 Variant Summary — {problem_name.capitalize()} ({loss_name})",
                    family_summary,
                    A2_VARIANT_ORDER,
                )
            )

            if diverged_runs:
                lines.append("Diverged runs (loss overflowed; excluded from the averages above):")
                for r in diverged_runs:
                    lines.append(
                        f"- {r['architecture']} | {r['optimizer']} | "
                        f"LR={r['learning_rate']} | Batch={r['batch']}"
                        f"{format_peak_gradient_norm(r)}"
                    )
                lines.append("")

        lines.append(
            format_collapse_check_table(
                "Collapse Check — Regression Test MSE at Learning Rate 0.1",
                filter_by_problem(all_results, "regression"),
            )
        )
        lines.append(
            "In the baseline A2, these four runs collapse: the ReLU units in the deeper hidden "
            "layers die, the network outputs a constant, and the test MSE equals that of always "
            "predicting 0. A collapsed network with bias terms can still learn the output "
            "layer's bias, so its constant moves to roughly the mean target and its test MSE "
            "matches that of always predicting the mean. Either way the hidden layers are dead: "
            "a much lower MSE than the constant-prediction baselines is what indicates a "
            "network that actually learned."
        )
        lines.append("")

    # Add the dropout section when the results contain dropout runs that
    # also record the evaluation-mode training metric.
    dropout_names = {with_dropout for _, with_dropout in DROPOUT_PAIRS}
    dropout_runs = [r for r in all_results if r["architecture"] in dropout_names]
    if dropout_runs and all("train_metric" in r for r in dropout_runs):
        lines.append("Dropout — Regularization and the Generalization Gap")
        lines.append("---------------------------------------------------")
        lines.append(
            "Each architecture is compared with a copy that has dropout (rate 0.2) after every "
            "hidden layer, on runs matched by optimizer, learning rate and batch size. Matches "
            "where either run diverged are left out. 'Avg Train' is the training-set metric in "
            "evaluation mode (no dropout) on the restored best model, so it is directly "
            "comparable with 'Avg Test'. 'Avg Gap' is test minus train: a large positive gap "
            "means overfitting, which is what dropout is meant to reduce. 'Dropout Better' "
            "counts the matched runs where dropout gave the lower test metric."
        )
        lines.append("")
        for problem_name, loss_name in [("classification", "BCE"), ("regression", "MSE")]:
            lines.append(
                format_dropout_table(
                    f"Dropout Comparison — {problem_name.capitalize()} ({loss_name})",
                    filter_by_problem(all_results, problem_name),
                )
            )

    # Add the Muon section when the results contain Muon runs.
    muon_section = build_muon_section(every_optimizer_results)
    if muon_section:
        lines.append(muon_section)

    # Add the multi-seed section when the results contain more than one seed.
    if len({r.get("seed", RANDOM_SEED) for r in multi_seed_results}) > 1:
        lines.append(build_multi_seed_section(multi_seed_results))

    # Join all report lines into one final text block.
    report_text = "\n".join(lines)

    # ------------------------------------------------------------
    # 10. Write the final analysis report to disk.
    # ------------------------------------------------------------
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Saved analysis report to: {OUTPUT_PATH}")


# Running this module writes the analysis report for the newest (or the
# given) results file and prints the total wall-clock time.
if __name__ == "__main__":
    start_time = time.perf_counter()
    # Optional first argument: an explicit results JSON to analyse.
    main(sys.argv[1] if len(sys.argv) > 1 else None)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
