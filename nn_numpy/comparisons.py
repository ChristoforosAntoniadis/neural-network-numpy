"""
Comparison plots and short written analyses built from the experiment results.

Input: a full results JSON written by nn_numpy.modeling.train
(``reports/main_results_full_<stamp>.json``): a list with one dictionary per
training run, holding its settings (problem, architecture, optimizer,
learning rate, batch size, seed) and its loss histories and test metric.

Output (each file name gets the run stamp):
- in ``reports/figures/comparisons/``: loss-curve figures that compare runs
  differing in exactly one setting (optimizer, network depth, learning rate,
  A2 variant, dropout);
- in ``reports/comparisons/``: two text files (depth and learning-rate analyses) that say which run
  converged faster and which reached the lower final losses.

Run it with ``python -m nn_numpy.comparisons [results.json]``.
"""

import json
import math
import os
import sys
import time
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Standard Matplotlib plotting interface.
import matplotlib.pyplot as plt
import numpy as np

# COMPARISONS_FIGURES_DIR / COMPARISONS_TEXT_DIR are the output folders.
# RUN_STAMP / stamped_filename give every output file a unique name, and
# resolve_results_path finds the results JSON produced by the experiment sweep.
from nn_numpy.config import (
    COMPARISONS_FIGURES_DIR,
    COMPARISONS_TEXT_DIR,
    RANDOM_SEED,
    RUN_STAMP,
    resolve_results_path,
    stamped_filename,
)

# Folder where the comparison plots are saved (reports/figures/comparisons/).
PLOTS_DIR = COMPARISONS_FIGURES_DIR

# Folder where the short text analyses are saved (reports/comparisons/).
TEXT_DIR = COMPARISONS_TEXT_DIR

# Top-to-bottom order of the optimizers in the optimizer-comparison figures.
OPTIMIZER_ORDER = ["sgd", "momentum", "adabelief", "muon"]

# Create the output folders if they do not already exist.
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)


def load_results(path: str) -> list[dict[str, Any]]:
    """
    Load the full experiment results JSON.

    Parameters
    ----------
    path : str
        Path to the JSON results file.

    Returns
    -------
    list of dict
        One dictionary per experiment run, with keys such as
        "problem_name", "architecture", "optimizer", "learning_rate",
        "batch", "seed", "train_loss_history", "val_loss_history" and
        "test_metric".

    Notes
    -----
    Processing:
    1. Open the file in text mode.
    2. Parse it with ``json.load`` and return the resulting list.
    """
    # Open the JSON file in read mode.
    with open(path, "r") as f:
        # Parse the JSON contents into Python objects and return them.
        return json.load(f)


def _build_epoch_ticks(runs: list[dict[str, Any]]) -> tuple[list[int], int]:
    """
    Build x-axis tick marks for one comparison figure.

    Parameters
    ----------
    runs : list of dict
        The experiment records shown in the figure. Only each record's
        "train_loss_history" (list of float) is used, for its length.

    Returns
    -------
    ticks : list of int
        Sorted, de-duplicated epoch numbers at which to place ticks.
    max_epoch : int
        The largest number of epochs any run in ``runs`` trained for.

    Notes
    -----
    Processing:
    1. Take each run's final epoch as the length of its training-loss
       history, and the largest of these as ``max_epoch``.
    2. Choose a tick step from ``max_epoch``: 1 up to 20 epochs, 5 up to
       50 epochs, otherwise 10.
    3. Make regular ticks 1, 1 + step, ... up to ``max_epoch``, and add
       ``max_epoch`` itself if it was not hit.
    4. Add every run's final epoch and sort the unique values.

    Why: different runs may end at different epochs because of early
    stopping. We want readable tick spacing and an explicit tick at each
    run's final epoch.
    """
    # Compute the final epoch of each run from the length of its train-loss history.
    end_epochs = [len(run["train_loss_history"]) for run in runs]

    # Find the longest run in this figure.
    max_epoch = max(end_epochs)

    # Choose a readable x-axis tick spacing based on the longest run.
    # Short runs get denser ticks; longer runs get more spread-out ticks.
    if max_epoch <= 20:
        step = 1
    elif max_epoch <= 50:
        step = 5
    else:
        step = 10

    # Create regularly spaced epoch ticks.
    regular_ticks = list(range(1, max_epoch + 1, step))

    # Ensure the maximum epoch is explicitly included.
    if regular_ticks[-1] != max_epoch:
        regular_ticks.append(max_epoch)

    # Also include the final epoch of every run, so early-stopped runs
    # have their stopping point clearly visible on the x-axis.
    ticks = sorted(set(regular_ticks + end_epochs))

    # Return both the tick list and the maximum epoch used for x-axis limits.
    return ticks, max_epoch


def _compute_convergence_metrics(
    run: dict[str, Any], relative_tolerance: float = 0.01, absolute_floor: float = 1e-4
) -> dict[str, Any]:
    """
    Compute simple convergence metrics for one run.

    Parameters
    ----------
    run : dict
        One experiment record. Uses "train_loss_history" and
        "val_loss_history" (each a list of float, one value per epoch).
    relative_tolerance : float, default=0.01
        Tolerance as a fraction of the final training loss (0.01 = 1%). No
        recorded justification for 1%.
    absolute_floor : float, default=1e-4
        Smallest tolerance allowed, whatever the final loss is. 1e-4 equals
        the configs' min_delta, the smallest loss change the trainer counts
        as an improvement.

    Returns
    -------
    dict
        With keys:
        - "epochs_ran" (int): number of epochs actually trained;
        - "final_train_loss" (float): last training-loss value;
        - "final_val_loss" (float): last validation-loss value;
        - "tolerance" (float): the tolerance used for the test below;
        - "convergence_epoch" (int): 1-based epoch from which the
          training loss stays within the tolerance of its final value.

    Notes
    -----
    Processing (the convergence rule):
    1. Take the final training loss:
       final_train_loss = last value in train_loss_history
    2. Define a tolerance around that final value:
       tolerance = max(absolute_floor, relative_tolerance * abs(final_train_loss))
       absolute_floor is the minimum allowed tolerance, used so the
       convergence rule does not become unrealistically strict when the
       final training loss is very small.
    3. Scan the epochs from the start and return the first epoch after
       which ALL remaining training-loss values stay within that tolerance
       of the final training loss. If no epoch qualifies (for example
       when the final loss is NaN), the last epoch is used.

    The rule uses training loss only; validation loss is reported but not
    used to decide convergence.
    """

    # Extract the training-loss history for this run.
    train_history = run["train_loss_history"]

    # Extract the validation-loss history for this run.
    val_history = run["val_loss_history"]

    # Final training loss = last recorded training-loss value.
    final_train_loss = float(train_history[-1])

    # Final validation loss = last recorded validation-loss value.
    final_val_loss = float(val_history[-1])

    # Number of epochs actually run (may be less than max epochs due to early stopping).
    epochs_ran = len(train_history)

    # Define the tolerance used for the convergence test.
    # Usually this is 1% of the final training loss, but we never allow it
    # to become smaller than the absolute floor.
    tolerance = max(absolute_floor, relative_tolerance * abs(final_train_loss))

    # Default convergence epoch to the last epoch, in case the run only settles
    # right at the end.
    convergence_epoch = epochs_ran

    # Scan from the beginning of the run.
    # We are looking for the first epoch after which the rest of the training-loss
    # values stay close to the final training loss.
    for start_idx in range(epochs_ran):
        # Take the "tail" of the training-loss curve from this epoch to the end.
        tail = train_history[start_idx:]

        # Check whether all values in that remaining tail are within the allowed
        # tolerance of the final training loss.
        if all(abs(loss - final_train_loss) <= tolerance for loss in tail):
            # Convert the 0-based list index to a 1-based epoch number.
            convergence_epoch = start_idx + 1
            # Stop as soon as we find the first such epoch.
            break

    # Return all computed metrics in one dictionary so they can be reused
    # in plots and written analyses.
    return {
        "epochs_ran": epochs_ran,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "tolerance": tolerance,
        "convergence_epoch": convergence_epoch,
    }


def _format_metrics_text(run: dict[str, Any]) -> str:
    """
    Format a small block of convergence-related metrics for display inside a plot.

    Parameters
    ----------
    run : dict
        One experiment record with "train_loss_history" and
        "val_loss_history" (lists of float), and optionally "best_epoch"
        (int).

    Returns
    -------
    str
        Final train loss and final validation loss (6 decimals), epochs
        ran, the convergence epoch (training-loss plateau) and, if the run
        records it, the best epoch (the checkpoint selected on validation
        loss).

    Notes
    -----
    Processing:
    1. Compute the metrics with ``_compute_convergence_metrics`` (default
       tolerances).
    2. Format them into a multi-line string with aligned labels.
    """
    # Compute convergence metrics for this run.
    metrics = _compute_convergence_metrics(run)

    # Build a small multi-line string that will be shown inside the figure.
    text = (
        f"Final train loss: {metrics['final_train_loss']:.6f}\n"
        f"Final val loss:   {metrics['final_val_loss']:.6f}\n"
        f"Epochs ran:       {metrics['epochs_ran']}\n"
        f"Conv. epoch (train): {metrics['convergence_epoch']}"
    )
    if run.get("best_epoch") is not None:
        text += f"\nBest epoch (val):    {run['best_epoch']}"
    return text


def _plot_loss_curves(
    ax: Axes, run: dict[str, Any], subplot_title: str, include_metrics_box: bool = False
) -> None:
    """
    Plot one run's train-loss and validation-loss curves on a given subplot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Subplot axis to draw on.
    run : dict
        One experiment record from the JSON results. Uses
        "train_loss_history" and "val_loss_history" (lists of float), and
        "best_epoch" (int) if present.
    subplot_title : str
        Title shown above the subplot (" | End Epoch = N" is appended).
    include_metrics_box : bool, default=False
        Whether to show a small textbox with convergence metrics.
        All comparison figures in this module enable it.

    Returns
    -------
    None
        Draws on ``ax`` in place.

    Notes
    -----
    Processing:
    1. Use epochs 1..N as the x values, where N is the length of the
       training-loss history.
    2. Plot training loss as a solid line and validation loss as a dashed
       line.
    3. Draw a dotted vertical line at the final epoch, so early stopping is
       visible, and a dash-dot line at the best epoch: the checkpoint with
       the lowest validation loss, which is restored and tested.
    4. Set the title and y label, add a light grid and the legend.
    5. If requested, put the convergence-metrics box in the upper-right
       corner (axes coordinates 0.98, 0.98).
    """

    # Build the epoch numbers for the x-axis: 1, 2, 3, ..., final epoch.
    epochs = range(1, len(run["train_loss_history"]) + 1)

    # The final epoch is just the length of the stored training-loss history.
    end_epoch = len(run["train_loss_history"])

    # Plot the training-loss curve as a solid line.
    ax.plot(epochs, run["train_loss_history"], label="Train Loss")

    # Plot the validation-loss curve as a dashed line so it is visually distinct.
    ax.plot(epochs, run["val_loss_history"], linestyle="--", label="Validation Loss")

    # Draw a vertical dotted line at the final epoch.
    # This makes early stopping visible on the plot.
    ax.axvline(
        x=end_epoch,
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label=f"Ended at epoch {end_epoch}",
    )

    # Mark the checkpoint early stopping restored (lowest validation loss).
    # The model reported on the test set is this one, not the last epoch.
    if run.get("best_epoch") is not None:
        ax.axvline(
            x=run["best_epoch"],
            color="tab:green",
            linestyle="-.",
            linewidth=1.2,
            alpha=0.8,
            label=f"Best epoch (val) {run['best_epoch']}",
        )

    # Add the subplot title, including the end epoch for readability.
    ax.set_title(f"{subplot_title} | End Epoch = {end_epoch}", fontsize=11)

    # Label the y-axis.
    ax.set_ylabel("Loss")

    # Add a light background grid to make the curves easier to read.
    ax.grid(True, alpha=0.3)

    # Show the legend for train loss, validation loss, and final-epoch line.
    ax.legend()

    # If requested, place a small metrics summary box inside this subplot.
    # Every comparison figure in this module turns it on.
    if include_metrics_box:
        # Build the metrics text.
        metrics_text = _format_metrics_text(run)

        # Place the text box in the upper-right corner of the subplot.
        ax.text(
            0.98,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )


def _apply_epoch_ticks(axes: np.ndarray, runs: list[dict[str, Any]]) -> None:
    """
    Apply consistent x-axis limits and ticks across all subplots in one figure.

    Parameters
    ----------
    axes : numpy.ndarray of matplotlib.axes.Axes, shape (n_subplots,)
        The subplots of the figure (as returned by ``plt.subplots(n, 1)``).
    runs : list of dict
        The experiment records plotted in the figure, used to choose the
        ticks.

    Returns
    -------
    None
        Modifies the x limits, ticks and tick-label rotation of every
        subplot in place.

    Notes
    -----
    Processing:
    1. Build the ticks and the largest epoch with ``_build_epoch_ticks``.
    2. On every subplot, set the x range to [1, max_epoch], place the
       ticks, and rotate the tick labels by 45 degrees.
    """
    # Build the tick marks and determine the largest epoch shown in this figure.
    ticks, max_epoch = _build_epoch_ticks(runs)

    # Apply the same x-axis configuration to every subplot in the figure.
    for ax in axes:
        # Force all subplots to share the same epoch range.
        ax.set_xlim(1, max_epoch)

        # Apply the chosen tick positions.
        ax.set_xticks(ticks)

        # Rotate tick labels slightly for readability.
        ax.tick_params(axis="x", rotation=45)


def _save_and_show(fig: Figure, filename: str) -> None:
    """
    Save a figure to disk, print the location, then show it interactively.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The finished figure.
    filename : str
        Bare file name such as "depth_classification_sgd_lr01_bs16.png";
        the run stamp is inserted before the extension.

    Returns
    -------
    None
        Writes the PNG to ``reports/figures/comparisons/`` at 300 dpi, prints its
        path, shows the figure in a blocking window, then closes it.

    Notes
    -----
    Processing:
    1. Build the output path with ``stamped_filename``.
    2. Save the figure with ``dpi=300`` and ``bbox_inches="tight"``.
    3. Print the path.
    4. Call ``plt.show(block=True)``, which waits until the window is
       closed (with a non-interactive backend it returns immediately).
    5. Close the figure to free memory.
    """
    # Build the full output path for the figure file, with the run stamp
    # inserted before the extension so earlier runs are never overwritten.
    output_path = os.path.join(PLOTS_DIR, stamped_filename(filename))

    # Save the figure with high resolution and tight layout.
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    # Print the file path so the user can see where it was saved.
    print(f"Saved plot to: {output_path}")

    # Show the plot in a window when running from CMD.
    plt.show(block=True)

    # Close the figure afterward to free memory.
    plt.close(fig)


def _write_text_file(filename: str, content: str) -> None:
    """
    Save a short text analysis file to the comparisons text folder.

    Parameters
    ----------
    filename : str
        Bare file name such as "depth_analysis.txt"; the run stamp is
        inserted before the extension.
    content : str
        The text to write.

    Returns
    -------
    None
        Writes the file to ``reports/comparisons/`` (UTF-8) and prints its
        path.

    Notes
    -----
    Processing:
    1. Build the output path with ``stamped_filename``.
    2. Write ``content`` to it with UTF-8 encoding.
    3. Print the path.
    """
    # Build the full output path for the text file, with the run stamp
    # inserted before the extension so earlier runs are never overwritten.
    output_path = os.path.join(TEXT_DIR, stamped_filename(filename))

    # Open the file for writing using UTF-8 encoding.
    with open(output_path, "w", encoding="utf-8") as f:
        # Write the text content to disk.
        f.write(content)

    # Print the file path so the user knows where the analysis was saved.
    print(f"Saved analysis text to: {output_path}")


def _depth_label(architecture_name: str) -> str:
    """
    Convert architecture code names into human-readable depth labels.

    Parameters
    ----------
    architecture_name : str
        Architecture code from the results, e.g. "A1" or "A2".

    Returns
    -------
    str
        "A1 (1 hidden layer)" for "A1", "A2 (3 hidden layers)" for "A2",
        and the input unchanged for any other name.

    Notes
    -----
    Processing:
    1. Compare the name with "A1" and "A2" and return the matching label.
    2. Fall back to the original name.
    """
    # Map architecture A1 to a more descriptive label.
    if architecture_name == "A1":
        return "A1 (1 hidden layer)"

    # Map architecture A2 to a more descriptive label.
    if architecture_name == "A2":
        return "A2 (3 hidden layers)"

    # Fallback: return the original architecture name if it is neither A1 nor A2.
    return architecture_name


def _build_depth_analysis_text(
    selected_runs: list[dict[str, Any]],
    problem_name: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
) -> str:
    """
    Write the text of the network-depth analysis (A1 vs A2).

    Parameters
    ----------
    selected_runs : list of dict
        Exactly two experiment records, A1 first and A2 second (as returned
        by ``filter_depth_runs``).
    problem_name : str
        "classification" or "regression".
    optimizer : str
        Optimizer name shared by both runs, e.g. "sgd".
    learning_rate : float
        Learning rate shared by both runs.
    batch_size : int
        Batch size shared by both runs.

    Returns
    -------
    str
        A multi-line report: the fixed settings, a one-line definition of
        convergence epoch, the measured values for each architecture, and
        three interpretation sentences.

    Notes
    -----
    Processing:
    1. Unpack the two runs and compute each one's convergence metrics.
    2. Turn the architecture codes into readable labels
       (``_depth_label``).
    3. Compare the two runs on convergence epoch, final training loss and
       final validation loss, and write one sentence for each comparison
       (naming the better architecture, or saying they are equal).
    4. Assemble the full text block.

    The text explains:
    - which architecture converged faster
    - which ended with lower final training loss
    - which ended with lower final validation loss
    """
    # Unpack the two matched runs.
    run_a1, run_a2 = selected_runs

    # Compute convergence metrics for A1.
    metrics_a1 = _compute_convergence_metrics(run_a1)

    # Compute convergence metrics for A2.
    metrics_a2 = _compute_convergence_metrics(run_a2)

    # Build human-readable labels for the two architectures.
    label_a1 = _depth_label(run_a1["architecture"])
    label_a2 = _depth_label(run_a2["architecture"])

    # Determine which architecture converges faster.
    if metrics_a1["convergence_epoch"] < metrics_a2["convergence_epoch"]:
        faster_sentence = (
            f"{label_a1} converges faster by this criterion, "
            f"because it reaches and stays near its final training loss earlier."
        )
    elif metrics_a2["convergence_epoch"] < metrics_a1["convergence_epoch"]:
        faster_sentence = (
            f"{label_a2} converges faster by this criterion, "
            f"because it reaches and stays near its final training loss earlier."
        )
    else:
        faster_sentence = (
            "Both architectures reach their final training-loss region at the same epoch."
        )

    # Compare final training losses.
    if metrics_a1["final_train_loss"] < metrics_a2["final_train_loss"]:
        train_sentence = (
            f"{label_a1} ends with the lower final training loss, "
            f"so it achieves the better final training fit in this matched experiment."
        )
    elif metrics_a2["final_train_loss"] < metrics_a1["final_train_loss"]:
        train_sentence = (
            f"{label_a2} ends with the lower final training loss, "
            f"so it achieves the better final training fit in this matched experiment."
        )
    else:
        train_sentence = "Both architectures end with the same final training loss."

    # Compare final validation losses.
    if metrics_a1["final_val_loss"] < metrics_a2["final_val_loss"]:
        val_sentence = (
            f"{label_a1} ends with the lower final validation loss, "
            f"which suggests better validation performance in this matched experiment."
        )
    elif metrics_a2["final_val_loss"] < metrics_a1["final_val_loss"]:
        val_sentence = (
            f"{label_a2} ends with the lower final validation loss, "
            f"which suggests better validation performance in this matched experiment."
        )
    else:
        val_sentence = "Both architectures end with the same final validation loss."

    # Build and return the final text block that will be saved as a .txt file.
    return (
        "Network Depth Experiment\n"
        "========================\n\n"
        "Selected matched experiment\n"
        "---------------------------\n"
        f"Problem: {problem_name}\n"
        f"Optimizer: {optimizer}\n"
        f"Learning rate: {learning_rate}\n"
        f"Batch size: {batch_size}\n"
        "Only the number of hidden layers changes: A1 vs A2.\n\n"
        "---------------------\n"
        "For each run, convergence epoch is the first epoch after which all remaining\n"
        "training-loss values stay within a small tolerance of the final training loss.\n\n"
        "Measured values\n"
        "---------------\n"
        f"{label_a1}:\n"
        f"  - epochs ran: {metrics_a1['epochs_ran']}\n"
        f"  - final train loss: {metrics_a1['final_train_loss']:.6f}\n"
        f"  - final val loss: {metrics_a1['final_val_loss']:.6f}\n"
        f"  - convergence epoch: {metrics_a1['convergence_epoch']}\n\n"
        f"{label_a2}:\n"
        f"  - epochs ran: {metrics_a2['epochs_ran']}\n"
        f"  - final train loss: {metrics_a2['final_train_loss']:.6f}\n"
        f"  - final val loss: {metrics_a2['final_val_loss']:.6f}\n"
        f"  - convergence epoch: {metrics_a2['convergence_epoch']}\n\n"
        "Interpretation\n"
        "--------------\n"
        f"{faster_sentence}\n"
        f"{train_sentence}\n"
        f"{val_sentence}\n"
    )


def _build_learning_rate_analysis_text(
    selected_runs: list[dict[str, Any]],
    problem_name: str,
    architecture: str,
    optimizer: str,
    batch_size: int,
) -> str:
    """
    Write the text of the learning-rate sensitivity analysis (0.1 vs 0.001).

    Parameters
    ----------
    selected_runs : list of dict
        Exactly two experiment records, learning rate 0.1 first and 0.001
        second (as returned by ``filter_learning_rate_runs``).
    problem_name : str
        "classification" or "regression".
    architecture : str
        Architecture shared by both runs, e.g. "A1".
    optimizer : str
        Optimizer name shared by both runs, e.g. "sgd".
    batch_size : int
        Batch size shared by both runs.

    Returns
    -------
    str
        A multi-line report: the fixed settings, the convergence rule, the
        measured values for each learning rate, and three interpretation
        sentences.

    Notes
    -----
    Processing:
    1. Unpack the two runs and compute each one's convergence metrics.
    2. Build labels such as "LR = 0.1" from each run's learning rate.
    3. Compare the two runs on convergence epoch, final training loss and
       final validation loss, and write one sentence for each comparison
       (naming the better learning rate, or saying they are equal).
    4. Assemble the full text block.
    """
    # Unpack the two matched learning-rate runs.
    run_lr_high, run_lr_low = selected_runs

    # Compute convergence metrics for the first learning-rate run.
    metrics_high = _compute_convergence_metrics(run_lr_high)

    # Compute convergence metrics for the second learning-rate run.
    metrics_low = _compute_convergence_metrics(run_lr_low)

    # Store the actual learning-rate values for labeling.
    lr_high = run_lr_high["learning_rate"]
    lr_low = run_lr_low["learning_rate"]

    # Build human-readable labels for the two learning rates.
    high_label = f"LR = {lr_high}"
    low_label = f"LR = {lr_low}"

    # Determine which learning rate converges faster.
    if metrics_high["convergence_epoch"] < metrics_low["convergence_epoch"]:
        faster_sentence = (
            f"{high_label} converges faster by this criterion, "
            f"because it reaches and stays near its final training loss earlier."
        )
    elif metrics_low["convergence_epoch"] < metrics_high["convergence_epoch"]:
        faster_sentence = (
            f"{low_label} converges faster by this criterion, "
            f"because it reaches and stays near its final training loss earlier."
        )
    else:
        faster_sentence = (
            "Both learning rates reach their final training-loss region at the same epoch."
        )

    # Compare final training losses.
    if metrics_high["final_train_loss"] < metrics_low["final_train_loss"]:
        train_sentence = (
            f"{high_label} ends with the lower final training loss in this matched experiment."
        )
    elif metrics_low["final_train_loss"] < metrics_high["final_train_loss"]:
        train_sentence = (
            f"{low_label} ends with the lower final training loss in this matched experiment."
        )
    else:
        train_sentence = "Both learning rates end with the same final training loss."

    # Compare final validation losses.
    if metrics_high["final_val_loss"] < metrics_low["final_val_loss"]:
        val_sentence = (
            f"{high_label} ends with the lower final validation loss in this matched experiment."
        )
    elif metrics_low["final_val_loss"] < metrics_high["final_val_loss"]:
        val_sentence = (
            f"{low_label} ends with the lower final validation loss in this matched experiment."
        )
    else:
        val_sentence = "Both learning rates end with the same final validation loss."

    # Build and return the final text block that will be saved as a .txt file.
    return (
        "Learning Rate Sensitivity\n"
        "=========================\n\n"
        "Selected matched experiment\n"
        "---------------------------\n"
        f"Problem: {problem_name}\n"
        f"Architecture: {architecture}\n"
        f"Optimizer: {optimizer}\n"
        f"Batch size: {batch_size}\n"
        "Only the learning rate changes: 0.1 vs 0.001.\n\n"
        "Convergence rule used\n"
        "---------------------\n"
        "Convergence is defined using TRAINING loss.\n"
        "For each run, convergence epoch is the first epoch after which all remaining\n"
        "training-loss values stay within a small tolerance of the final training loss.\n\n"
        "Measured values\n"
        "---------------\n"
        f"{high_label}:\n"
        f"  - epochs ran: {metrics_high['epochs_ran']}\n"
        f"  - final train loss: {metrics_high['final_train_loss']:.6f}\n"
        f"  - final val loss: {metrics_high['final_val_loss']:.6f}\n"
        f"  - convergence epoch: {metrics_high['convergence_epoch']}\n\n"
        f"{low_label}:\n"
        f"  - epochs ran: {metrics_low['epochs_ran']}\n"
        f"  - final train loss: {metrics_low['final_train_loss']:.6f}\n"
        f"  - final val loss: {metrics_low['final_val_loss']:.6f}\n"
        f"  - convergence epoch: {metrics_low['convergence_epoch']}\n\n"
        "Interpretation\n"
        "--------------\n"
        f"{faster_sentence}\n"
        f"{train_sentence}\n"
        f"{val_sentence}\n"
    )


# --------------------------------------------------
# 1. Optimizer comparison
# --------------------------------------------------
def filter_optimizer_runs(
    results: list[dict[str, Any]],
    problem_name: str,
    architecture: str,
    learning_rate: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    """
    Select the three optimizer runs for one matched experiment.

    Parameters
    ----------
    results : list of dict
        All experiment records (already reduced to one seed by ``main``).
    problem_name : str
        "classification" or "regression".
    architecture : str
        Architecture code, e.g. "A1" or "A2".
    learning_rate : float
        Learning rate to match exactly, e.g. 0.1.
    batch_size : int
        Batch size to match (the record's "batch" key).

    Returns
    -------
    list of dict
        The matching records, sorted SGD, Momentum, AdaBelief, Muon.
        Normally one per optimizer; the caller checks the count. Each is a
        run record, with the keys listed in
        nn_numpy.modeling.train.run_single_experiment.

    Raises
    ------
    KeyError
        If a matching record has an optimizer other than "sgd",
        "momentum", "adabelief" or "muon" (it has no sort position).

    Notes
    -----
    Processing:
    1. Keep the records whose problem, architecture, learning rate and
       batch size all match.
    2. Sort them into the fixed optimizer order for plotting.

    What stays fixed: problem_name, architecture, learning_rate,
    batch_size. What changes: the optimizer only.
    """
    # Keep only runs that match the fixed parameters for this comparison.
    filtered = [
        run
        for run in results
        if run["problem_name"] == problem_name
        and run["architecture"] == architecture
        and run["learning_rate"] == learning_rate
        and run["batch"] == batch_size
    ]

    # Sort optimizers into a consistent top-to-bottom order for plotting.
    optimizer_order = {name: i for i, name in enumerate(OPTIMIZER_ORDER)}
    filtered.sort(key=lambda x: optimizer_order[x["optimizer"]])

    # Return the matched optimizer runs, one per optimizer.
    return filtered


def plot_optimizer_comparison(
    results: list[dict[str, Any]],
    problem_name: str,
    architecture: str,
    learning_rate: float,
    batch_size: int,
    filename: str,
) -> None:
    """
    Create one figure comparing SGD, Momentum, AdaBelief and Muon
    under one matched parameter setting.

    Parameters
    ----------
    results : list of dict
        All experiment records.
    problem_name : str
        "classification" or "regression".
    architecture : str
        Architecture code, e.g. "A1" or "A2".
    learning_rate : float
        Learning rate shared by the runs.
    batch_size : int
        Batch size shared by the runs.
    filename : str
        Bare PNG file name; the run stamp is added when saving.

    Returns
    -------
    None
        Saves the figure to ``reports/figures/comparisons/`` and shows it.

    Raises
    ------
    ValueError
        If the filter does not find exactly one run per optimizer.

    Notes
    -----
    Processing:
    1. Select the matching runs with ``filter_optimizer_runs`` and check
       there is one per optimizer in OPTIMIZER_ORDER.
    2. Create one stacked subplot per optimizer, sharing the x-axis, each with train/validation loss curves and the
       convergence-metrics box.
    3. Apply shared epoch ticks, label the x-axis, add an overall title.
    4. Save and show the figure.
    """
    # Select the runs for this optimizer comparison.
    selected_runs = filter_optimizer_runs(
        results=results,
        problem_name=problem_name,
        architecture=architecture,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    # Sanity check: the optimizer comparison expects one run per optimizer.
    expected_runs = len(OPTIMIZER_ORDER)
    if len(selected_runs) != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} optimizer runs, found {len(selected_runs)} "
            f"for {problem_name}, {architecture}, lr={learning_rate}, batch={batch_size}"
        )

    # Create a figure with one vertical subplot per optimizer.
    fig, axes = plt.subplots(expected_runs, 1, figsize=(12, 4 * expected_runs), sharex=True)

    # Plot each optimizer run on its own subplot.
    for ax, run in zip(axes, selected_runs):
        optimizer_name = run["optimizer"].upper()
        subplot_title = (
            f"{optimizer_name} Optimizer | "
            f"{problem_name.capitalize()} | {architecture} | "
            f"LR={learning_rate} | Batch={batch_size}"
        )
        _plot_loss_curves(ax, run, subplot_title, include_metrics_box=True)

    # Apply consistent epoch ticks across all subplots.
    _apply_epoch_ticks(axes, selected_runs)

    # Label the x-axis only on the last subplot.
    axes[-1].set_xlabel("Epoch")

    # Add an overall title for the full figure.
    fig.suptitle(
        f"Optimizer Comparison: {problem_name.capitalize()} Problem | "
        f"Architecture {architecture} | LR={learning_rate} | Batch Size={batch_size}",
        fontsize=14,
        fontweight="bold",
    )

    # Adjust spacing so the suptitle and subplots fit cleanly.
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # type: ignore[arg-type]  # a list works at runtime

    # Save and display the figure.
    _save_and_show(fig, filename)


# --------------------------------------------------
# 2. Depth comparison (A1 vs A2)
# --------------------------------------------------
def filter_depth_runs(
    results: list[dict[str, Any]],
    problem_name: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    """
    Select the two runs needed for the depth experiment.

    Parameters
    ----------
    results : list of dict
        All experiment records (already reduced to one seed by ``main``).
    problem_name : str
        "classification" or "regression".
    optimizer : str
        Optimizer name, e.g. "sgd".
    learning_rate : float
        Learning rate to match exactly.
    batch_size : int
        Batch size to match (the record's "batch" key).

    Returns
    -------
    list of dict
        The matching A1 and A2 records, A1 first. Normally two; the caller
        checks the count. Each is a run record, with the keys listed in
        nn_numpy.modeling.train.run_single_experiment.

    Notes
    -----
    Processing:
    1. Keep the records whose problem, optimizer, learning rate and batch
       size match and whose architecture is "A1" or "A2" (variants such
       as "A2-bias" are excluded).
    2. Sort A1 before A2.

    What stays fixed: problem_name, optimizer, learning_rate, batch_size.
    What changes: the architecture only (A1 vs A2).
    """
    # Keep only runs that match the fixed settings for the depth comparison.
    filtered = [
        run
        for run in results
        if run["problem_name"] == problem_name
        and run["optimizer"] == optimizer
        and run["learning_rate"] == learning_rate
        and run["batch"] == batch_size
        and run["architecture"] in ["A1", "A2"]
    ]

    # Sort A1 before A2 so the order is stable and easy to read.
    architecture_order = {"A1": 0, "A2": 1}
    filtered.sort(key=lambda x: architecture_order[x["architecture"]])

    # Return the two matched architecture runs.
    return filtered


def plot_depth_comparison(
    results: list[dict[str, Any]],
    problem_name: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
    filename: str,
    analysis_filename: str,
) -> None:
    """
    Plot A1 vs A2 loss curves and write a short depth-analysis text file.

    Parameters
    ----------
    results : list of dict
        All experiment records.
    problem_name : str
        "classification" or "regression".
    optimizer : str
        Optimizer shared by both runs, e.g. "sgd".
    learning_rate : float
        Learning rate shared by both runs.
    batch_size : int
        Batch size shared by both runs.
    filename : str
        Bare PNG file name for the figure; the run stamp is added.
    analysis_filename : str
        Bare .txt file name for the analysis; the run stamp is added.

    Returns
    -------
    None
        Saves and shows the figure (in ``reports/figures/comparisons/``) and
        writes the text file (in ``reports/comparisons/``).

    Raises
    ------
    ValueError
        If the filter does not find exactly two runs.

    Notes
    -----
    Processing:
    1. Select the A1 and A2 runs with ``filter_depth_runs`` and check
       there are two.
    2. Create two stacked subplots, one per architecture, each with
       training loss, validation loss and the convergence-metrics box.
    3. Apply shared epoch ticks, label the x-axis, add an overall title,
       then save and show the figure.
    4. Build the analysis text with ``_build_depth_analysis_text`` and
       save it. The text says which architecture converged faster and
       which achieved the lower final training and validation losses.
    """
    # Select the two runs for the depth comparison.
    selected_runs = filter_depth_runs(
        results=results,
        problem_name=problem_name,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    # Sanity check: the depth comparison expects exactly 2 architectures.
    if len(selected_runs) != 2:
        raise ValueError(
            f"Expected 2 architecture runs, found {len(selected_runs)} "
            f"for {problem_name}, optimizer={optimizer}, lr={learning_rate}, batch={batch_size}"
        )

    # Create a figure with 2 vertical subplots: one for A1 and one for A2.
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Plot each architecture run on its own subplot.
    for ax, run in zip(axes, selected_runs):
        architecture = run["architecture"]
        subplot_title = (
            f"Architecture {architecture} | "
            f"{problem_name.capitalize()} | {optimizer.upper()} | "
            f"LR={learning_rate} | Batch={batch_size}"
        )
        _plot_loss_curves(ax, run, subplot_title, include_metrics_box=True)

    # Apply consistent epoch ticks across both subplots.
    _apply_epoch_ticks(axes, selected_runs)

    # Label the shared x-axis.
    axes[-1].set_xlabel("Epoch")

    # Add an overall figure title.
    fig.suptitle(
        f"Network Depth Comparison: A1 vs A2 | "
        f"{problem_name.capitalize()} Problem | Optimizer={optimizer.upper()} | "
        f"LR={learning_rate} | Batch Size={batch_size}",
        fontsize=14,
        fontweight="bold",
    )

    # Adjust spacing so labels and title fit properly.
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # type: ignore[arg-type]  # a list works at runtime

    # Save and display the figure.
    _save_and_show(fig, filename)

    # Build the written analysis text for network depth analysis.
    analysis_text = _build_depth_analysis_text(
        selected_runs=selected_runs,
        problem_name=problem_name,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    # Save the written analysis to a text file.
    _write_text_file(analysis_filename, analysis_text)


# --------------------------------------------------
# 3. Learning-rate comparison (0.1 vs 0.001)
# --------------------------------------------------
def filter_learning_rate_runs(
    results: list[dict[str, Any]],
    problem_name: str,
    architecture: str,
    optimizer: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    """
    Select the two runs needed for the learning-rate sensitivity experiment.

    Parameters
    ----------
    results : list of dict
        All experiment records (already reduced to one seed by ``main``).
    problem_name : str
        "classification" or "regression".
    architecture : str
        Architecture code, e.g. "A1".
    optimizer : str
        Optimizer name, e.g. "sgd".
    batch_size : int
        Batch size to match (the record's "batch" key).

    Returns
    -------
    list of dict
        The matching records with learning rate 0.1 and 0.001, 0.1 first.
        Normally two; the caller checks the count. Each is a run record,
        with the keys listed in nn_numpy.modeling.train.run_single_experiment.

    Notes
    -----
    Processing:
    1. Keep the records whose problem, architecture, optimizer and batch
       size match and whose learning rate is 0.1 or 0.001.
    2. Sort 0.1 before 0.001.

    What stays fixed: problem_name, architecture, optimizer, batch_size.
    What changes: the learning rate only (0.1 vs 0.001).
    """
    # Keep only runs that match the fixed settings for the learning-rate comparison.
    filtered = [
        run
        for run in results
        if run["problem_name"] == problem_name
        and run["architecture"] == architecture
        and run["optimizer"] == optimizer
        and run["batch"] == batch_size
        and run["learning_rate"] in [0.1, 0.001]
    ]

    # Sort the two learning rates into a stable order for plotting.
    lr_order = {0.1: 0, 0.001: 1}
    filtered.sort(key=lambda x: lr_order[x["learning_rate"]])

    # Return the two matched learning-rate runs.
    return filtered


def plot_learning_rate_comparison(
    results: list[dict[str, Any]],
    problem_name: str,
    architecture: str,
    optimizer: str,
    batch_size: int,
    filename: str,
    analysis_filename: str,
) -> None:
    """
    Create the learning-rate comparison figure and write a short analysis text file.

    Parameters
    ----------
    results : list of dict
        All experiment records.
    problem_name : str
        "classification" or "regression".
    architecture : str
        Architecture shared by both runs, e.g. "A1".
    optimizer : str
        Optimizer shared by both runs, e.g. "sgd".
    batch_size : int
        Batch size shared by both runs.
    filename : str
        Bare PNG file name for the figure; the run stamp is added.
    analysis_filename : str
        Bare .txt file name for the analysis; the run stamp is added.

    Returns
    -------
    None
        Saves and shows the figure (in ``reports/figures/comparisons/``) and
        writes the text file (in ``reports/comparisons/``).

    Raises
    ------
    ValueError
        If the filter does not find exactly two runs.

    Notes
    -----
    Processing:
    1. Select the LR 0.1 and LR 0.001 runs with
       ``filter_learning_rate_runs`` and check there are two.
    2. Create two stacked subplots, one per learning rate, each with
       training loss, validation loss and the convergence-metrics box.
    3. Apply shared epoch ticks, label the x-axis, add an overall title,
       then save and show the figure.
    4. Build the analysis text with ``_build_learning_rate_analysis_text``
       and save it. The text says which learning rate converged faster and
       which achieved the lower final training and validation losses.
    """
    # Select the two runs for the learning-rate comparison.
    selected_runs = filter_learning_rate_runs(
        results=results,
        problem_name=problem_name,
        architecture=architecture,
        optimizer=optimizer,
        batch_size=batch_size,
    )

    # Sanity check: We expect exactly 2 learning-rate runs.
    if len(selected_runs) != 2:
        raise ValueError(
            f"Expected 2 learning-rate runs, found {len(selected_runs)} "
            f"for {problem_name}, {architecture}, optimizer={optimizer}, batch={batch_size}"
        )

    # Create a figure with 2 vertical subplots: one for each learning rate.
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Plot each learning-rate run on its own subplot.
    for ax, run in zip(axes, selected_runs):
        lr = run["learning_rate"]
        subplot_title = (
            f"Learning Rate = {lr} | "
            f"{problem_name.capitalize()} | {architecture} | "
            f"{optimizer.upper()} | Batch={batch_size}"
        )

        _plot_loss_curves(ax, run, subplot_title, include_metrics_box=True)

    # Apply consistent epoch ticks across both subplots.
    _apply_epoch_ticks(axes, selected_runs)

    # Label the shared x-axis.
    axes[-1].set_xlabel("Epoch")

    # Add an overall figure title.
    fig.suptitle(
        f"Learning-Rate Sensitivity Comparison: LR 0.1 vs 0.001 | "
        f"{problem_name.capitalize()} Problem | Architecture {architecture} | "
        f"Optimizer={optimizer.upper()} | Batch Size={batch_size}",
        fontsize=14,
        fontweight="bold",
    )

    # Adjust spacing so labels and title fit properly.
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # type: ignore[arg-type]  # a list works at runtime

    # Save and display the figure.
    _save_and_show(fig, filename)

    # Build the written analysis text for learning rate comparison.
    analysis_text = _build_learning_rate_analysis_text(
        selected_runs=selected_runs,
        problem_name=problem_name,
        architecture=architecture,
        optimizer=optimizer,
        batch_size=batch_size,
    )

    # Save the written analysis to a text file.
    _write_text_file(analysis_filename, analysis_text)


# --------------------------------------------------
# 4. Architecture variant comparison (A2 variants, dropout)
# --------------------------------------------------
# A2 and its variants, each changing one thing: a bias term, He
# initialization, both, batch normalization on the hidden layers, or
# gradient-norm clipping during training.
A2_VARIANT_ORDER = ["A2", "A2-bias", "A2-he", "A2-bias-he", "A2-bn", "A2-clip"]


def filter_variant_runs(
    results: list[dict[str, Any]],
    problem_name: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
    architectures: list[str] = A2_VARIANT_ORDER,
) -> list[dict[str, Any]]:
    """
    Select one run per architecture for one matched experiment.

    Parameters
    ----------
    results : list of dict
        All experiment records (already reduced to one seed by ``main``).
    problem_name : str
        "classification" or "regression".
    optimizer : str
        Optimizer name, e.g. "sgd".
    learning_rate : float
        Learning rate to match exactly.
    batch_size : int
        Batch size to match (the record's "batch" key).
    architectures : list of str, default=A2_VARIANT_ORDER
        Architectures to keep, in the order they should be plotted.

    Returns
    -------
    list of dict
        The matching records, ordered as in ``architectures``. Normally one
        per architecture; the caller checks the count. Each is a run record,
        with the keys listed in nn_numpy.modeling.train.run_single_experiment.

    Notes
    -----
    Processing:
    1. Keep the records whose problem, optimizer, learning rate and batch
       size match and whose architecture is in ``architectures``.
    2. Sort them by the position of their architecture in
       ``architectures``.

    What stays fixed: problem_name, optimizer, learning_rate, batch_size.
    What changes: the architecture only (by default A2, A2-bias, A2-he,
    A2-bias-he, A2-bn, A2-clip).
    """
    filtered = [
        run
        for run in results
        if run["problem_name"] == problem_name
        and run["optimizer"] == optimizer
        and run["learning_rate"] == learning_rate
        and run["batch"] == batch_size
        and run["architecture"] in architectures
    ]

    # Keep the architectures in the given top-to-bottom order.
    filtered.sort(key=lambda x: architectures.index(x["architecture"]))
    return filtered


def plot_variant_comparison(
    results: list[dict[str, Any]],
    problem_name: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
    filename: str,
    architectures: list[str] = A2_VARIANT_ORDER,
    title: str = "A2 Variants: Bias, He Init, Batch Norm, Gradient Clipping",
) -> None:
    """
    Create one figure comparing several architectures on one matched experiment.

    By default it compares A2 with its bias / initialization / batch-norm
    variants; the dropout comparison passes its own architectures and title.

    Parameters
    ----------
    results : list of dict
        All experiment records. Besides the loss histories, uses
        "test_metric" (float) and, if present, "diverged" (bool).
    problem_name : str
        "classification" or "regression". Decides whether the test
        metric is labelled MSE (regression) or BCE (otherwise).
    optimizer : str
        Optimizer shared by all runs, e.g. "sgd".
    learning_rate : float
        Learning rate shared by all runs.
    batch_size : int
        Batch size shared by all runs.
    filename : str
        Bare PNG file name; the run stamp is added when saving.
    architectures : list of str, default=A2_VARIANT_ORDER
        Architectures to compare, one subplot each, top to bottom.
    title : str, default="A2 Variants: Bias, He Init, Batch Norm, Gradient Clipping"
        First part of the figure title; the shared settings are appended.

    Returns
    -------
    None
        Saves the figure to ``reports/figures/comparisons/`` and shows it.

    Raises
    ------
    ValueError
        If the filter does not find exactly one run per architecture.

    Notes
    -----
    Processing:
    1. Select the runs with ``filter_variant_runs`` and check there is one
       per architecture.
    2. Create one stacked subplot per run (3.75 inches of height each).
    3. For each run, title the subplot with its test metric, or with
       "Diverged" if the run is flagged as diverged or its test metric is
       not finite, and draw the loss curves with the metrics box.
    4. Switch a subplot's y-axis to log scale when the run has at least one
       finite training-loss value.
    5. Apply shared epoch ticks, label the x-axis, add the overall title,
       then save and show the figure.

    Each architecture gets its own subplot with its own y-axis, because a
    collapsed run's loss can be orders of magnitude above a healthy one.
    """
    # Select the matching runs, one per architecture, in plotting order.
    selected_runs = filter_variant_runs(
        results=results,
        problem_name=problem_name,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        architectures=architectures,
    )

    # Sanity check: one run per architecture.
    if len(selected_runs) != len(architectures):
        raise ValueError(
            f"Expected {len(architectures)} runs, found {len(selected_runs)} "
            f"for {problem_name}, optimizer={optimizer}, lr={learning_rate}, batch={batch_size}"
        )

    # One vertical subplot per variant.
    fig, axes = plt.subplots(
        len(selected_runs), 1, figsize=(12, 3.75 * len(selected_runs)), sharex=True
    )

    # Name of the test metric shown in each subplot title.
    metric_name = "MSE" if problem_name == "regression" else "BCE"
    for ax, run in zip(axes, selected_runs):
        # A diverged run's test metric is NaN or meaningless (it can come from
        # a finite but enormous checkpoint), so the title says so instead.
        # Older results files have no "diverged" flag.
        if run.get("diverged") or not math.isfinite(run["test_metric"]):
            subplot_title = f"{run['architecture']} | Diverged (loss overflowed)"
        else:
            subplot_title = (
                f"{run['architecture']} | Test {metric_name} = {run['test_metric']:.4f}"
            )
        _plot_loss_curves(ax, run, subplot_title, include_metrics_box=True)

        # Log scale: an exploding run's first-epoch loss can be hundreds of
        # orders of magnitude above where it settles, which would flatten
        # the rest of the curve on a linear axis. A diverged run has no
        # finite values to put on a log axis, so it keeps the default.
        if any(math.isfinite(v) for v in run["train_loss_history"]):
            ax.set_yscale("log")
            ax.set_ylabel("Loss (log scale)")

    # Apply consistent epoch ticks across all subplots.
    _apply_epoch_ticks(axes, selected_runs)

    # Label the shared x-axis.
    axes[-1].set_xlabel("Epoch")

    # Add an overall figure title with the shared settings.
    fig.suptitle(
        f"{title} | "
        f"{problem_name.capitalize()} | Optimizer={optimizer.upper()} | "
        f"LR={learning_rate} | Batch Size={batch_size}",
        fontsize=14,
        fontweight="bold",
    )

    # Adjust spacing so labels and title fit properly.
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # type: ignore[arg-type]  # a list works at runtime

    # Save and display the figure.
    _save_and_show(fig, filename)


def main(results_path: str | None = None) -> None:
    """
    Generate all required plots and short text analyses.

    Parameters
    ----------
    results_path : str or None, default=None
        Path to a main_results_full_<stamp>.json file. When None, the
        newest stamped results file in reports/ is used.

    Returns
    -------
    None
        Saves (and shows) every comparison figure in
        ``reports/figures/comparisons/`` and writes the two analysis text
        files in ``reports/comparisons/``; prints the run
        stamp, the results file used and each saved path.

    Raises
    ------
    FileNotFoundError
        If ``results_path`` is None and no stamped results file exists.
    ValueError
        If a comparison does not find the runs it expects in the results.

    Notes
    -----
    Processing:
    1. Resolve and load the results file.
    2. Keep only the runs of the first seed (``RANDOM_SEED``); runs with
       no "seed" field (older files) are kept.
    3. Optimizer comparison: three plots, each with the same parameters
       inside the plot and only the optimizer varying.
    4. Network depth experiment: compare A1 vs A2 for one matched
       experiment and write a short text analysis.
    5. Learning-rate sensitivity: compare LR 0.1 vs 0.001 for one matched
       experiment and write a short text analysis.
    6. A2 variants (bias, He initialization, batch norm): two plots, only
       if the results contain variant runs.
    7. Dropout: one plot comparing A1 / A1-dropout and A2-bn /
       A2-bn-dropout, only if the results contain dropout runs.
    """
    # Load the full experiment results from disk.
    results_path = resolve_results_path(results_path)
    print(f"Run stamp: {RUN_STAMP}")
    print(f"Using results file: {results_path}")
    results = load_results(results_path)

    # Each plot shows one run's loss curves, so a multi-seed results file is
    # reduced to its first seed (RANDOM_SEED). Variation across seeds is
    # reported in the analysis instead. Older files have no "seed" field.
    results = [run for run in results if run.get("seed", RANDOM_SEED) == RANDOM_SEED]

    # Hand-picked example settings. To adjust, change the arguments of a plot
    # call to any acceptable value. The acceptable values in the current
    # experiment grid are:
    # - problem_name: classification or regression
    # - architecture: A1 or A2 (the variant and dropout plots also use
    #   A2-bias, A2-he, A2-bias-he, A2-bn, A1-dropout, A2-bn-dropout)
    # - learning_rate: 0.1 or 0.001
    # - batch_size: 16 or 64
    # - optimizer: sgd, momentum, adabelief or muon
    # --------------------------------------------------
    # Optimizer comparison
    # --------------------------------------------------

    # Plot 1:
    # Compare optimizers for the classification problem using architecture A1,
    # learning rate 0.1, and batch size 16.
    plot_optimizer_comparison(
        results=results,
        problem_name="classification",
        architecture="A1",
        learning_rate=0.1,
        batch_size=16,
        filename="optimizers_classification_A1_lr01_bs16.png",
    )

    # Plot 2:
    # Compare optimizers for the classification problem using architecture A2,
    # learning rate 0.1, and batch size 64.
    plot_optimizer_comparison(
        results=results,
        problem_name="classification",
        architecture="A2",
        learning_rate=0.1,
        batch_size=64,
        filename="optimizers_classification_A2_lr01_bs64.png",
    )

    # Plot 3:
    # Compare optimizers for the regression problem using architecture A2,
    # learning rate 0.001, and batch size 64.
    plot_optimizer_comparison(
        results=results,
        problem_name="regression",
        architecture="A2",
        learning_rate=0.001,
        batch_size=64,
        filename="optimizers_regression_A2_lr0001_bs64.png",
    )

    # --------------------------------------------------
    # Network depth experiment
    # --------------------------------------------------

    # Here we keep:
    # - problem fixed
    # - optimizer fixed
    # - learning rate fixed
    # - batch size fixed
    #
    # And vary only:
    # - architecture (A1 vs A2)
    #
    # This produces:
    # - one depth-comparison plot
    # - one short text analysis file
    plot_depth_comparison(
        results=results,
        problem_name="classification",
        optimizer="sgd",
        learning_rate=0.1,
        batch_size=16,
        filename="depth_classification_sgd_lr01_bs16.png",
        analysis_filename="depth_analysis.txt",
    )

    # --------------------------------------------------
    # Learning-rate sensitivity
    # --------------------------------------------------

    # Here we keep:
    # - problem fixed
    # - architecture fixed
    # - optimizer fixed
    # - batch size fixed
    #
    # And vary only:
    # - learning rate (0.1 vs 0.001)
    #
    # This produces:
    # - one learning-rate comparison plot
    # - one short text analysis file
    plot_learning_rate_comparison(
        results=results,
        problem_name="classification",
        architecture="A1",
        optimizer="sgd",
        batch_size=16,
        filename="learning_rate_classification_A1_sgd_bs16.png",
        analysis_filename="learning_rate_analysis.txt",
    )

    # --------------------------------------------------
    # A2 variants: bias terms, He initialization, batch norm
    # --------------------------------------------------

    # Here we keep:
    # - problem fixed (regression)
    # - optimizer fixed (SGD)
    # - learning rate fixed (0.1)
    # - batch size fixed (16)
    #
    # And vary only:
    # - A2 variant (bias and / or He initialization, or batch norm)
    #
    # This is one of the settings where the baseline A2 collapses (its ReLU
    # units die), so the plot shows whether any of the changes prevents it.
    # Skipped for older results files that have no variant runs; variants
    # a file doesn't contain (e.g. A2-clip in older files) are left out.
    present_variants = [
        name for name in A2_VARIANT_ORDER if any(run["architecture"] == name for run in results)
    ]
    if len(present_variants) > 1:
        plot_variant_comparison(
            results=results,
            problem_name="regression",
            optimizer="sgd",
            learning_rate=0.1,
            batch_size=16,
            filename="a2_variants_regression_sgd_lr01_bs16.png",
            architectures=present_variants,
        )

        # Same comparison with momentum and batch size 64: the baseline A2
        # still collapses here, but no variant diverges, so the plot shows
        # each fix's outcome side by side.
        plot_variant_comparison(
            results=results,
            problem_name="regression",
            optimizer="momentum",
            learning_rate=0.1,
            batch_size=64,
            filename="a2_variants_regression_momentum_lr01_bs64.png",
            architectures=present_variants,
        )

    # --------------------------------------------------
    # Dropout
    # --------------------------------------------------

    # Here we keep:
    # - problem fixed (regression)
    # - optimizer fixed (AdaBelief)
    # - learning rate fixed (0.1)
    # - batch size fixed (16)
    #
    # And compare:
    # - A1 vs A1-dropout, and A2-bn vs A2-bn-dropout
    #
    # This is the setting of the best baseline regression run. With dropout,
    # the plotted training loss is measured with dropout active, so it
    # includes dropout noise and typically sits above the validation loss.
    # Skipped for older results files that have no dropout runs.
    dropout_architectures = ["A1", "A1-dropout", "A2-bn", "A2-bn-dropout"]
    if any(run["architecture"] == "A1-dropout" for run in results):
        plot_variant_comparison(
            results=results,
            problem_name="regression",
            optimizer="adabelief",
            learning_rate=0.1,
            batch_size=16,
            filename="dropout_regression_adabelief_lr01_bs16.png",
            architectures=dropout_architectures,
            title="Dropout (rate 0.2)",
        )


# Running this file directly builds every comparison plot and analysis from
# a results JSON (the newest one unless a path is given) and prints the
# total run time.
if __name__ == "__main__":
    start_time = time.perf_counter()
    # Optional first argument: an explicit results JSON to plot from.
    main(sys.argv[1] if len(sys.argv) > 1 else None)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
