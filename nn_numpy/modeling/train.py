"""
Experiment driver: runs the full hyperparameter sweep for both problems.

Input: the two JSON experiment configs in configs/ and the datasets in
data/raw/ (downloaded first if missing).

Processing: for each problem (banknote classification, energy-efficiency
regression) and each seed, the data is preprocessed once, then one network
is trained and tested for every combination of architecture, optimizer,
learning rate and batch size listed in the config.

Output: a console summary table, plus two files in reports/ whose names
carry the run stamp: a compact CSV summary and a full JSON file with every
run's settings, metrics and training histories (read by nn_numpy/analysis.py and
the plotting scripts).
"""

import csv
import json
import os
import time
from typing import Any

import numpy as np

from nn_numpy.config import RANDOM_SEED, REPORT_DIR, RUN_STAMP, stamped_filename
from nn_numpy.config_loader import ConfigLoader
from nn_numpy.dataset import Fetch
from nn_numpy.features import PreprocessBanknote, PreprocessEnergy
from nn_numpy.modeling.trainer import Trainer
from nn_numpy.nn.losses import get_loss
from nn_numpy.nn.metrics import (
    accuracy,
    binary_cross_entropy,
    mean_squared_error,
    r2_score,
)
from nn_numpy.nn.network import NeuralNetwork
from nn_numpy.nn.optimizers import get_optimizer
from nn_numpy.runlog import (
    begin_phase,
    finish_run,
    format_progress,
    should_report,
    start_run_log,
)

# Toggle this to True if you want features.py to generate EDA plots.
# Leaving it False makes the full experiment sweep faster and quieter.
SHOW_EDA = False

# Mapping from problem type to the config file that defines its experiments.
CONFIG_FILES = {
    "classification": "classification_experiments.json",
    "regression": "regression_experiments.json",
}

# Preprocessing class for each problem type.
PREPROCESSORS: dict[str, type[PreprocessBanknote] | type[PreprocessEnergy]] = {
    "classification": PreprocessBanknote,
    "regression": PreprocessEnergy,
}

# Experiment outputs (the summary CSV and the full JSON results) are saved
# in REPORT_DIR, i.e. reports/, imported from nn_numpy.config.
# Create the reports directory if it does not already exist.
os.makedirs(REPORT_DIR, exist_ok=True)


def get_seeds(experiment_config: dict[str, Any]) -> list[int]:
    """
    Return the list of seeds to run for one problem.

    Parameters
    ----------
    experiment_config : dict
        The validated contents of one problem's JSON config file. Only the
        optional "seeds" entry of its "experiments" section is read.

    Returns
    -------
    list of int
        The seeds listed in the config, or [RANDOM_SEED] if the config
        doesn't list any.

    Notes
    -----
    Processing:
    1. Look up experiment_config["experiments"]["seeds"].
    2. Fall back to [RANDOM_SEED] when the key is missing.

    Each seed sets the train / validation / test split, the weight
    initialization, the dropout masks and the per-epoch shuffling, so
    running several seeds measures how much results vary from those
    random choices alone.
    """
    return experiment_config["experiments"].get("seeds", [RANDOM_SEED])


def constant_prediction_baseline(
    problem_name: str, dataset_splits: tuple[np.ndarray, ...]
) -> float:
    """
    Compute the test metric of a model that ignores the features entirely.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression". Any value other than
        "classification" is scored with MSE.
    dataset_splits : tuple of numpy.ndarray
        (X_train, y_train, X_val, y_val, X_test, y_test), as returned by the
        preprocessor's get_data(). Each X has shape (n_samples, n_features),
        dtype float64; each y has shape (n_samples, 1) and holds 0/1 labels
        (integer dtype) for classification or float64 targets for
        regression. Only y_train and y_test are used.

    Returns
    -------
    float
        The constant predictor's test BCE (classification) or MSE
        (regression).

    Notes
    -----
    Processing:
    1. Take the mean of y_train. For 0/1 labels this is the proportion of
       class 1; for regression it is the mean target.
    2. Build a float array the same shape as y_test, filled with that mean.
    3. Score it against y_test with BCE for classification, MSE otherwise.

    Why these constants: the class proportion is the best constant
    probability under BCE, and the mean is the best constant under MSE.

    A trained model is only useful to the extent it beats this. The
    fraction of this error a model removes is its "error removed"
    (R² for regression, McFadden's pseudo-R² for classification).
    """
    _, y_train, _, _, _, y_test = dataset_splits
    constant = np.full_like(y_test, y_train.mean(), dtype=float)
    if problem_name == "classification":
        return float(binary_cross_entropy(y_test, constant))
    return float(mean_squared_error(y_test, constant))


def build_model_config(
    experiment_config: dict[str, Any], architecture_name: str
) -> dict[str, Any]:
    """
    Build the small model-config dictionary needed by NeuralNetwork.build_from_config().

    Parameters
    ----------
    experiment_config : dict
        The validated contents of the current problem's JSON config file.
    architecture_name : str
        The architecture key chosen for the current run, such as "A1" or "A2".

    Returns
    -------
    dict
        A smaller dictionary containing only the fields required by
        NeuralNetwork.build_from_config():
        - "input_dimension" : int, number of input features.
        - "layers" : list of dict, the layer specifications of the chosen
          architecture, taken unchanged from the config.

    Raises
    ------
    KeyError
        If `architecture_name` is not one of the config's architectures.

    Notes
    -----
    Processing:
    1. Copy "input_dimension" from the config.
    2. Look up the layer list of the chosen architecture.
    """
    return {
        "input_dimension": experiment_config["input_dimension"],
        "layers": experiment_config["architectures"][architecture_name],
    }


def run_single_experiment(
    problem_name: str,
    experiment_config: dict[str, Any],
    architecture_name: str,
    optimizer_name: str,
    learning_rate: float,
    batch_size: int,
    dataset_splits: tuple[np.ndarray, ...],
    seed: int = RANDOM_SEED,
    baseline_test_metric: float | None = None,
) -> dict[str, Any]:
    """
    Build, train and test one network for one combination of settings.

    Parameters
    ----------
    problem_name : str
        Either "classification" or "regression". Stored in the summary.
    experiment_config : dict
        Validated config for the current problem. Supplies the input
        dimension, architectures, loss name, task type, preprocessing flags
        and the "experiments" settings (epochs and early stopping).
    architecture_name : str
        Name of the selected architecture from the config, e.g. "A1".
    optimizer_name : str
        Name of the optimizer to use, e.g. "sgd", "momentum", "adabelief" or "muon".
    learning_rate : float
        Learning rate for the optimizer.
    batch_size : int
        Mini-batch size for training.
    dataset_splits : tuple of numpy.ndarray
        (X_train, y_train, X_val, y_val, X_test, y_test). Each X has shape
        (n_samples, input_dimension), dtype float64; each y has shape
        (n_samples, 1) with 0/1 labels (integer dtype) for classification
        or float64 targets for regression.
    seed : int, default=RANDOM_SEED
        Seed for weight initialization, dropout masks and data shuffling.
        The same seed was used to create dataset_splits.
    baseline_test_metric : float or None, default=None
        Test metric of a model that ignores the features (see
        constant_prediction_baseline), recorded with the run for reference.

    Returns
    -------
    dict
        Full experiment summary. Keys:
        - settings: "problem_name", "seed", "architecture", "optimizer",
          "batch", "learning_rate", "epochs", "early_stopping", "patience",
          "min_delta", "min_epochs_before_early_stop", "max_grad_norm"
          (float or None: the architecture's gradient-norm cap),
          "preprocessing_enabled", "scale_features";
        - training outcome: "epochs_ran", "stopped_early", "diverged",
          "stop_reason" ("diverged", "early_stopping" or "max_epochs"),
          "overfitting_epochs", "best_epoch", "best_val_loss";
        - metrics (float): "test_loss", "test_metric", "train_metric",
          "baseline_test_metric" (float or None), and "test_accuracy"
          (classification) or "test_r2" (regression);
        - "train_seconds" (float): wall-clock time of Trainer.fit;
        - histories (list of float, one value per epoch):
          "train_loss_history", "val_loss_history", "val_metric_history",
          and "grad_norm_history" (largest gradient norm per epoch, before
          clipping).

    Raises
    ------
    ValueError
        If the number of columns in X_train differs from the config's
        "input_dimension".

    Notes
    -----
    Processing:
    1. Check that the data's feature count matches the config.
    2. Build the network for the chosen architecture, seeded with `seed`.
    3. Create the loss function, optimizer and Trainer (with the config's
       early-stopping settings and the architecture's gradient-norm cap, if
       it has one in "architecture_options").
    4. Train with Trainer.fit on the training set, validating on the
       validation set each epoch.
    5. Evaluate on the test set (silently), and score the training set in
       evaluation mode (no dropout) for the generalization gap.
    6. Compute test accuracy (classification) or test R² (regression), and
       record the training time.
    7. Collect settings, results and histories into one dictionary.

    It prints nothing: main() prints one progress line per reported run,
    which keeps the 1,440-run log readable.
    """
    # Unpack the dataset splits for this problem
    X_train, y_train, X_val, y_val, X_test, y_test = dataset_splits

    # Safety check: the dataset feature dimension must match what the config says
    if X_train.shape[1] != experiment_config["input_dimension"]:
        raise ValueError(
            f"Config input_dimension={experiment_config['input_dimension']} "
            f"does not match dataset dimension={X_train.shape[1]}"
        )

    # Build the specific model configuration for this architecture
    model_config = build_model_config(experiment_config, architecture_name)

    # Create the neural network and build its layer structure from config
    network = NeuralNetwork(random_seed=seed)
    network.build_from_config(model_config)

    # Create the loss function and optimizer from config choices
    loss_fn = get_loss(experiment_config["loss"])
    optimizer = get_optimizer(name=optimizer_name, learning_rate=learning_rate)

    # Architecture options from the config, e.g. a gradient-norm cap.
    options = experiment_config.get("architecture_options", {}).get(architecture_name, {})
    max_grad_norm = options.get("max_grad_norm")

    # Create the Trainer object, including early stopping settings
    trainer = Trainer(
        network=network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        task_type=experiment_config["task_type"],
        early_stopping=experiment_config["experiments"]["early_stopping"],
        patience=experiment_config["experiments"]["patience"],
        min_delta=experiment_config["experiments"]["min_delta"],
        min_epochs_before_early_stop=experiment_config["experiments"][
            "min_epochs_before_early_stop"
        ],
        random_seed=seed,
        max_grad_norm=max_grad_norm,
    )

    # Train the model and collect history. The wall-clock training time is
    # recorded so it can be compared with other libraries.
    fit_start = time.perf_counter()
    history = trainer.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=experiment_config["experiments"]["epochs"],
        batch_size=batch_size,
        verbose=False,
    )
    train_seconds = time.perf_counter() - fit_start

    # Evaluate on the held-out test set
    results = trainer.evaluate(X_test, y_test, verbose=False)

    # Same metric on the training set, in evaluation mode, for the
    # generalization gap (test - train). The training loss recorded during
    # fit() can't be used for this: with dropout it includes dropout noise.
    train_metric = trainer.score(X_train, y_train)

    # An easier-to-read secondary test metric: accuracy for classification,
    # R² for regression. These use the same NumPy functions as the library
    # comparisons in nn_numpy.benchmarks.
    test_predictions = trainer.predict(X_test)
    if experiment_config["task_type"] == "classification":
        secondary_name, secondary_value = "test_accuracy", accuracy(y_test, test_predictions)
    else:
        secondary_name, secondary_value = "test_r2", r2_score(y_test, test_predictions)

    # Build a summary record for this run.
    # Keep the full JSON informative for later plotting/analysis,
    # even though the CSV summary is intentionally minimal.
    summary = {
        "problem_name": problem_name,
        "seed": seed,
        "architecture": architecture_name,
        "optimizer": optimizer_name,
        "batch": batch_size,
        "learning_rate": learning_rate,
        "epochs": experiment_config["experiments"]["epochs"],
        "early_stopping": experiment_config["experiments"]["early_stopping"],
        "patience": experiment_config["experiments"]["patience"],
        "min_delta": experiment_config["experiments"]["min_delta"],
        "min_epochs_before_early_stop": experiment_config["experiments"][
            "min_epochs_before_early_stop"
        ],
        "max_grad_norm": max_grad_norm,
        "epochs_ran": history["epochs_ran"],
        "stopped_early": history["stopped_early"],
        "diverged": history["diverged"],
        "stop_reason": history["stop_reason"],
        "overfitting_epochs": history["overfitting_epochs"],
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],
        "test_loss": float(results["test_loss"]),
        "test_metric": float(results["test_metric"]),
        "train_metric": train_metric,
        "baseline_test_metric": baseline_test_metric,
        secondary_name: secondary_value,
        "train_seconds": train_seconds,
        "train_loss_history": [float(x) for x in history["train_loss"]],
        "val_loss_history": [float(x) for x in history["val_loss"]],
        "val_metric_history": [float(x) for x in history["val_metric"]],
        "grad_norm_history": [float(x) for x in history["grad_norm"]],
        "preprocessing_enabled": experiment_config["preprocessing"]["enabled"],
        "scale_features": experiment_config["preprocessing"]["scale_features"],
    }

    return summary


def save_summary_csv(results: list[dict[str, Any]], filename: str = "main_summary.csv") -> None:
    """
    Save a compact experiment summary as a CSV file in reports/.

    Parameters
    ----------
    results : list of dict
        One summary dictionary per run, as returned by
        run_single_experiment.
    filename : str, default="main_summary.csv"
        Output CSV filename. The run stamp is inserted before the
        extension so earlier runs are never overwritten.

    Returns
    -------
    None
        Writes reports/<filename with run stamp> and prints its path.

    Notes
    -----
    Processing:
    1. Build the stamped output path inside REPORT_DIR.
    2. Write a header row with the selected column names.
    3. Write one row per run, keeping only those columns (a missing key is
       written as an empty cell).

    This CSV intentionally keeps only the fields needed for a compact
    summary table: problem_name, seed, optimizer, batch, learning_rate,
    architecture, epochs_ran, best_epoch, stop_reason and test_metric.
    """
    output_path = os.path.join(REPORT_DIR, stamped_filename(filename))

    # Minimal CSV fields requested for the experiment summary
    fieldnames = [
        "problem_name",
        "seed",
        "optimizer",
        "batch",
        "learning_rate",
        "architecture",
        "epochs_ran",
        "best_epoch",
        "stop_reason",
        "test_metric",
    ]

    # Write the CSV file
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Keep only the selected CSV fields from each row
        for row in results:
            filtered_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(filtered_row)

    print(f"\nSaved summary CSV to: {output_path}")


def save_full_results_json(
    results: list[dict[str, Any]], filename: str = "main_results_full.json"
) -> None:
    """
    Save the full experiment results as a JSON file in reports/.

    Parameters
    ----------
    results : list of dict
        One summary dictionary per run, as returned by
        run_single_experiment, including the history lists.
    filename : str, default="main_results_full.json"
        Output JSON filename. The run stamp is inserted before the
        extension so earlier runs are never overwritten.

    Returns
    -------
    None
        Writes reports/<filename with run stamp> and prints its path.

    Notes
    -----
    Processing:
    1. Build the stamped output path inside REPORT_DIR.
    2. Dump the whole results list to it as indented JSON.

    Why: unlike the CSV, the JSON preserves every field, including the
    per-epoch histories (train_loss_history, val_loss_history,
    val_metric_history). This makes it the input for later analysis and
    plotting.
    """
    output_path = os.path.join(REPORT_DIR, stamped_filename(filename))

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved full results JSON to: {output_path}")


def print_summary_table(results: list[dict[str, Any]]) -> None:
    """
    Print a compact summary table to the console.

    Parameters
    ----------
    results : list of dict
        One summary dictionary per run, as returned by
        run_single_experiment. Each must have the keys problem_name, seed,
        optimizer, batch, learning_rate, architecture, epochs_ran and
        test_metric.

    Returns
    -------
    None
        Prints the table to standard output.

    Notes
    -----
    Processing:
    1. Print a title and a header row of fixed-width columns.
    2. Print one row per run, with the test metric to 6 decimal places.

    The "Epochs" column here displays epochs_ran,
    meaning the actual number of epochs completed.
    """
    print("\n" + "=" * 110)
    print("FINAL EXPERIMENT SUMMARY")
    print("=" * 110)
    print(
        f"{'Problem':<15}"
        f"{'Seed':<6}"
        f"{'Optimizer':<15}"
        f"{'Batch':<10}"
        f"{'LR':<12}"
        f"{'Arch':<13}"
        f"{'Epochs':<10}"
        f"{'Test Metric':<15}"
    )
    print("-" * 110)

    for row in results:
        print(
            f"{row['problem_name']:<15}"
            f"{row['seed']:<6}"
            f"{row['optimizer']:<15}"
            f"{row['batch']:<10}"
            f"{row['learning_rate']:<12}"
            f"{row['architecture']:<13}"
            f"{row['epochs_ran']:<10}"
            f"{row['test_metric']:<15.6f}"
        )


def main() -> None:
    """
    Run the full experiment sweep for every problem and save the results.

    Parameters
    ----------
    None
        Uses the module-level CONFIG_FILES, PREPROCESSORS, SHOW_EDA and
        REPORT_DIR, and the run stamp from nn_numpy.config.

    Returns
    -------
    None
        Downloads missing datasets, prints progress and a summary table,
        and writes the stamped summary CSV and full-results JSON to reports/.

    Notes
    -----
    Processing:
    1. Download the datasets if they are not already present.
    2. Load and validate the experiment config of every problem.
    3. Count the total number of runs, for progress messages.
    4. For each problem and seed, preprocess the dataset once and compute
       the constant-prediction baseline on that split.
    5. Run one experiment for every architecture x optimizer x learning
       rate x batch size combination on that split.
    6. Print the summary table and save the CSV and JSON files.
    """
    # Every file written by this run carries this stamp in its name.
    print(f"Run stamp: {RUN_STAMP}")

    begin_phase("Load data and configs")

    # Ensure all required datasets are available locally
    fetch = Fetch()
    fetch.download_all()

    # Load and validate the experiment configs first.
    # We do this before preprocessing so preprocessing behavior can be
    # controlled by values stored in the JSON configuration files.
    config_loader = ConfigLoader()
    experiment_configs = {
        problem_name: config_loader.load_and_validate(filename)
        for problem_name, filename in CONFIG_FILES.items()
    }

    # This will hold one summary dictionary per run
    all_results = []

    # Count total number of runs for progress reporting
    total_runs = 0
    for problem_name, config in experiment_configs.items():
        total_runs += (
            len(get_seeds(config))
            * len(config["architectures"])
            * len(config["experiments"]["optimizers"])
            * len(config["experiments"]["learning_rates"])
            * len(config["experiments"]["batch_sizes"])
        )

    run_counter = 0

    # Full nested sweep over:
    # - problem
    # - seed (data split + weight initialization + shuffling + dropout)
    # - architecture
    # - optimizer
    # - learning rate
    # - batch size
    for problem_name, config in experiment_configs.items():
        begin_phase(f"Train {problem_name}")
        metric_name = "val BCE" if problem_name == "classification" else "val MSE"
        seeds = get_seeds(config)
        preprocessing = config["preprocessing"]

        for seed in seeds:
            # Preprocess once per seed, so every experiment with this seed
            # uses the exact same train/validation/test split. EDA plots are
            # only made for the first seed, so they aren't saved again (and
            # overwritten) for every seed.
            print(f"\nPreprocessing {problem_name} dataset (seed {seed})...")
            dataset_splits = PREPROCESSORS[problem_name]().get_data(
                show_eda=SHOW_EDA,
                preprocessing_enabled=preprocessing["enabled"],
                scale_features=preprocessing["scale_features"],
                random_seed=seed,
                run_eda=(seed == seeds[0]),
            )
            # Error of always predicting a constant on this split, stored with
            # every run so later analysis can tell whether a model learned.
            baseline = constant_prediction_baseline(problem_name, dataset_splits)

            for architecture_name in config["architectures"]:
                for optimizer_name in config["experiments"]["optimizers"]:
                    for learning_rate in config["experiments"]["learning_rates"]:
                        for batch_size in config["experiments"]["batch_sizes"]:
                            run_counter += 1
                            run_start = time.perf_counter()

                            result = run_single_experiment(
                                problem_name=problem_name,
                                experiment_config=config,
                                architecture_name=architecture_name,
                                optimizer_name=optimizer_name,
                                learning_rate=learning_rate,
                                batch_size=batch_size,
                                dataset_splits=dataset_splits,
                                seed=seed,
                                baseline_test_metric=baseline,
                            )

                            all_results.append(result)

                            # Bounded: about MAX_PROGRESS_LINES lines over all runs.
                            if should_report(run_counter, total_runs):
                                print(
                                    format_progress(
                                        architecture_name,
                                        "Run",
                                        run_counter,
                                        total_runs,
                                        metric_name,
                                        result["best_val_loss"],
                                        time.perf_counter() - run_start,
                                        {
                                            "problem": problem_name,
                                            "optimizer": optimizer_name,
                                            "lr": learning_rate,
                                            "batch": batch_size,
                                            "seed": seed,
                                        },
                                    )
                                )

    begin_phase("Save results")

    # Print a compact console summary
    print_summary_table(all_results)

    # Save the compact CSV summary
    save_summary_csv(all_results)

    # Save the full JSON results including histories
    save_full_results_json(all_results)


# Running this file runs the whole experiment sweep, saves the results to
# reports/, and prints the total wall-clock time.
if __name__ == "__main__":
    start_run_log("train")
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
    finish_run()
