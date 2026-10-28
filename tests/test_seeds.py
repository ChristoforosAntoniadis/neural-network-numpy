"""
Tests for running experiments with several seeds: seed plumbing, the
constant-prediction baseline, config validation, and the multi-seed
analysis helpers.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from nn_numpy.analysis import (
    error_removed,
    fails_baseline,
    mean_pm_std,
    select_by_validation,
)
from nn_numpy.config import RANDOM_SEED
from nn_numpy.config_loader import ConfigLoader
from nn_numpy.features import train_val_test_split
from nn_numpy.modeling.train import constant_prediction_baseline, get_seeds
from nn_numpy.nn.network import NeuralNetwork

# ------------------------------------------------------------------
# Seed plumbing
# ------------------------------------------------------------------


def test_split_changes_with_the_seed() -> None:
    """
    Different seeds give different train/validation/test splits.

    Notes
    -----
    A 100-row dataframe is split (stratified on "class") with seeds 42
    and 43; the sets of "x" values in the two training splits must differ.
    The two seeds carry no meaning; any two distinct seeds would do.
    """
    df = pd.DataFrame({"x": np.arange(100), "class": np.arange(100) % 2})
    train_a, _, _ = train_val_test_split(df, stratify_col="class", random_seed=42)
    train_b, _, _ = train_val_test_split(df, stratify_col="class", random_seed=43)
    assert set(train_a["x"]) != set(train_b["x"])


def test_network_initialization_follows_the_seed() -> None:
    """
    Network weight initialization is reproducible from the seed and changes with it.

    Notes
    -----
    Three one-layer networks are built from the same config with seeds
    42, 42 and 43. The two seed-42 networks must have identical weights;
    the seed-43 network must have different weights.
    """
    config = {
        "input_dimension": 3,
        "layers": [{"type": "dense", "units": 4, "activation": "relu"}],
    }
    a, b, c = (
        NeuralNetwork(random_seed=42),
        NeuralNetwork(random_seed=42),
        NeuralNetwork(random_seed=43),
    )
    for net in (a, b, c):
        net.build_from_config(config)
    np.testing.assert_array_equal(a.layers[0].weights, b.layers[0].weights)
    assert not np.array_equal(a.layers[0].weights, c.layers[0].weights)


def test_get_seeds_defaults_to_the_project_seed() -> None:
    """
    get_seeds falls back to [RANDOM_SEED] and otherwise returns the configured seeds.

    Notes
    -----
    A config without "seeds" must give [RANDOM_SEED]; a config with
    "seeds": [1, 2] must give [1, 2].
    """
    assert get_seeds({"experiments": {}}) == [RANDOM_SEED]
    assert get_seeds({"experiments": {"seeds": [1, 2]}}) == [1, 2]


# ------------------------------------------------------------------
# Constant-prediction baseline
# ------------------------------------------------------------------


def splits(
    y_train: list[float], y_test: list[float]
) -> tuple[None, np.ndarray, None, None, None, np.ndarray]:
    """
    Build a dataset-splits tuple that holds only training and test targets.

    Parameters
    ----------
    y_train : list of float or list of int
        Training targets.
    y_test : list of float or list of int
        Test targets.

    Returns
    -------
    tuple[None, numpy.ndarray, None, None, None, numpy.ndarray]
        (None, y_train, None, None, None, y_test), where both targets are
        numpy.ndarray of shape (n, 1), dtype float64. This matches the
        (X_train, y_train, X_val, y_val, X_test, y_test) layout that
        `constant_prediction_baseline` expects.

    Notes
    -----
    Processing:
    1. Convert each list to a float64 array and reshape it to a column.
    2. Put them in their positions and fill the unused slots with None,
       since the baseline only reads y_train and y_test.
    """
    y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)
    y_test = np.asarray(y_test, dtype=float).reshape(-1, 1)
    return None, y_train, None, None, None, y_test


def test_regression_baseline_predicts_the_training_mean() -> None:
    """
    The regression baseline predicts the training mean and is scored with MSE.

    Notes
    -----
    With training targets [1, 2, 3] and test targets [1, 5], the baseline
    must return 5.0, the MSE of always predicting 2.
    """
    # Training mean is 2; test targets 1 and 5 -> MSE = ((1-2)^2 + (5-2)^2) / 2 = 5
    assert constant_prediction_baseline(
        "regression",
        splits([1, 2, 3], [1, 5]),  # type: ignore[arg-type]  # only y_train / y_test are read
    ) == pytest.approx(5.0)


def test_classification_baseline_predicts_the_training_proportion() -> None:
    """
    The classification baseline predicts the training class-1 proportion and is scored with BCE.

    Notes
    -----
    With training labels [1, 0, 0, 0] the constant prediction is 0.25.
    For test labels [1, 0] the baseline must return
    -(log(0.25) + log(0.75)) / 2 = (1.3863 + 0.2877) / 2 = 0.8370.
    """
    # Training proportion 0.25; BCE of predicting 0.25 for labels [1, 0]
    expected = -(np.log(0.25) + np.log(0.75)) / 2
    assert constant_prediction_baseline(
        "classification",
        splits([1, 0, 0, 0], [1, 0]),  # type: ignore[arg-type]  # only y_train / y_test are read
    ) == pytest.approx(expected)


# ------------------------------------------------------------------
# Config validation
# ------------------------------------------------------------------


def make_config(**experiments_extra: Any) -> dict[str, Any]:
    """
    Build a minimal valid classification config for ConfigLoader.validate.

    Parameters
    ----------
    **experiments_extra : Any
        Extra keys merged into the "experiments" block, for example
        `seeds=[42, 43]`.

    Returns
    -------
    dict[str, Any]
        Config with every required top-level, preprocessing and
        experiments key (plus `experiments_extra`), and one architecture
        "A" made of a single 1-unit sigmoid Dense layer.

    Notes
    -----
    Processing:
    1. Return a fresh dictionary literal each call, so tests cannot
       affect each other by mutating it.
    """
    return {
        "task_type": "classification",
        "input_dimension": 4,
        "loss": "bce",
        "preprocessing": {"enabled": True, "scale_features": True},
        "architectures": {"A": [{"type": "dense", "units": 1, "activation": "sigmoid"}]},
        "experiments": {
            "optimizers": ["sgd"],
            "learning_rates": [0.1],
            "batch_sizes": [16],
            "epochs": 10,
            "early_stopping": True,
            "patience": 2,
            "min_delta": 0.0,
            "min_epochs_before_early_stop": 0,
            **experiments_extra,
        },
    }


def test_config_accepts_seeds() -> None:
    """
    The config validator accepts a list of distinct non-negative integer seeds.

    Notes
    -----
    `validate` must return True for "seeds": [42, 43, 44].
    """
    assert ConfigLoader().validate(make_config(seeds=[42, 43, 44]))


@pytest.mark.parametrize("seeds", [[], [42, 42], [-1], [1.5], [True], "42", 42])
def test_config_rejects_invalid_seeds(seeds: list[Any] | str | int) -> None:
    """
    The config validator rejects invalid "seeds" values.

    Parameters
    ----------
    seeds : list, str or int
        Invalid value: [] (empty), [42, 42] (duplicate), [-1] (negative),
        [1.5] (not an integer), [True] (a boolean), "42" (a string, not a
        list) or 42 (an integer, not a list).

    Notes
    -----
    `validate` must raise ValueError.
    """
    with pytest.raises(ValueError):
        ConfigLoader().validate(make_config(seeds=seeds))


# ------------------------------------------------------------------
# Multi-seed analysis helpers
# ------------------------------------------------------------------


def run(val: float, test: float, baseline: float = 10.0, diverged: bool = False) -> dict[str, Any]:
    """
    Build a minimal experiment-run dictionary for the analysis helpers.

    Parameters
    ----------
    val : float
        Best validation loss of the run.
    test : float
        Test metric of the run.
    baseline : float, default=10.0
        Test metric of the constant-prediction baseline.
    diverged : bool, default=False
        Whether the run diverged.

    Returns
    -------
    dict[str, Any]
        {"best_val_loss": val, "test_metric": test,
        "baseline_test_metric": baseline, "diverged": diverged}.

    Notes
    -----
    Processing:
    1. Put the arguments under the keys the analysis helpers read.
    """
    return {
        "best_val_loss": val,
        "test_metric": test,
        "baseline_test_metric": baseline,
        "diverged": diverged,
    }


def test_mean_pm_std_uses_the_sample_standard_deviation() -> None:
    """
    mean_pm_std formats "mean ± sample std" and handles one or zero values.

    Notes
    -----
    [1, 2, 3] must give "2.00 ± 1.00" (the sample standard deviation,
    dividing by n - 1, is 1.0). A single value must give a standard
    deviation of 0.0, and an empty list must give "-".

    By hand: mean 2, sample std sqrt(((1 - 2)^2 + 0 + (3 - 2)^2) / (3 - 1))
    = sqrt(1) = 1.
    """
    assert mean_pm_std([1.0, 2.0, 3.0], digits=2) == "2.00 ± 1.00"
    assert mean_pm_std([4.0], digits=1) == "4.0 ± 0.0"
    assert mean_pm_std([]) == "-"


def test_selection_uses_validation_loss_and_skips_diverged_runs() -> None:
    """
    select_by_validation picks the lowest validation loss among runs that did not diverge.

    Notes
    -----
    Of three runs with validation losses 0.5, 0.1 and 0.3, the 0.1 run
    diverged, so the 0.3 run must be chosen (the identical object is
    returned). A list containing only a diverged run must give None.
    """
    runs = [run(val=0.5, test=9.0), run(val=0.1, test=8.0, diverged=True), run(val=0.3, test=1.0)]
    assert select_by_validation(runs) is runs[2]
    assert select_by_validation([run(0.1, 1.0, diverged=True)]) is None


def test_error_removed_and_baseline_failure() -> None:
    """
    error_removed and fails_baseline compare a run's test metric with the baseline.

    Notes
    -----
    A test metric of 2.5 against a baseline of 10.0 must give an error
    removed of 1 - 2.5 / 10 = 0.75. A run whose test metric equals the
    baseline fails it; a run slightly below the baseline (9.9) does not.
    """
    assert error_removed(run(val=0, test=2.5, baseline=10.0)) == pytest.approx(0.75)
    assert fails_baseline(run(val=0, test=10.0, baseline=10.0))
    assert not fails_baseline(run(val=0, test=9.9, baseline=10.0))


def test_derived_metrics_keep_the_checkpoint_validation_loss() -> None:
    """
    add_derived_metrics keeps the saved best_val_loss instead of the history minimum.

    Notes
    -----
    The saved value (0.5) belongs to the checkpoint that was restored and
    tested; a later epoch reached 0.4 without being saved (its improvement
    was below min_delta). Selection must use 0.5. A run without a saved
    value falls back to the history minimum.
    """
    from nn_numpy.analysis import add_derived_metrics

    saved = {
        "best_val_loss": 0.5,
        "val_loss_history": [0.9, 0.5, 0.4],
        "train_loss_history": [1.0, 0.8, 0.7],
    }
    missing = {"val_loss_history": [0.9, 0.5, 0.4], "train_loss_history": [1.0, 0.8, 0.7]}
    add_derived_metrics([saved, missing])
    assert saved["best_val_loss"] == 0.5
    assert missing["best_val_loss"] == 0.4


def test_derived_metrics_add_the_gap_and_fill_in_how_old_runs_ended() -> None:
    """
    add_derived_metrics adds the generalization gap and a stop reason for old results.

    Notes
    -----
    A run with a training metric gets generalization_gap = test - train.
    Runs from results files written before stop_reason existed get it from
    their "diverged" / "stopped_early" flags, and overfitting_epochs None
    (not recorded); a run that records them keeps its values.

    By hand: generalization_gap = 0.3 - 0.1 = 0.2.
    """
    from nn_numpy.analysis import add_derived_metrics

    histories = {"val_loss_history": [0.9, 0.5], "train_loss_history": [1.0, 0.8]}
    new = {
        **histories,
        "test_metric": 0.3,
        "train_metric": 0.1,
        "stop_reason": "max_epochs",
        "overfitting_epochs": 0,
    }
    old_diverged = {**histories, "test_metric": 0.3, "diverged": True, "stopped_early": True}
    old_early = {**histories, "test_metric": 0.3, "diverged": False, "stopped_early": True}
    old_full = {**histories, "test_metric": 0.3, "diverged": False, "stopped_early": False}
    add_derived_metrics([new, old_diverged, old_early, old_full])

    assert new["generalization_gap"] == pytest.approx(0.2)
    assert "generalization_gap" not in old_full
    assert (new["stop_reason"], new["overfitting_epochs"]) == ("max_epochs", 0)
    assert old_diverged["stop_reason"] == "diverged"
    assert old_early["stop_reason"] == "early_stopping"
    assert old_full["stop_reason"] == "max_epochs"
    assert old_full["overfitting_epochs"] is None


def test_stopping_table_counts_stop_reasons_and_overfitting() -> None:
    """
    format_stopping_seed_table counts how runs ended and which early stops overfitted.

    Notes
    -----
    Four A1 runs over two seeds: two early stops (one with the overfitting
    signature), one that ran to the epoch limit and one that diverged. The
    A1 row must read "2 / 1 / 1" and "1 of 2", and the gap column must be
    the mean ± std of the gaps of the run selected per seed (0.1 and 0.3).

    By hand: seed 42 selects the 0.2 run (gap 0.1), seed 43 the 0.3 run
    (gap 0.3; its NaN run diverged). Mean (0.1 + 0.3) / 2 = 0.20, sample
    std sqrt((0.1^2 + 0.1^2) / (2 - 1)) = 0.1414, so "0.20 ± 0.14".
    """
    from nn_numpy.analysis import format_stopping_seed_table

    def a1(seed: int, val: float, reason: str, overfit: int, gap: float) -> dict[str, Any]:
        """
        Build one A1 run record for the stopping table.

        Parameters
        ----------
        seed : int
            Seed of the run.
        val : float
            Best validation loss (NaN for the diverged run).
        reason : str
            stop_reason: "early_stopping", "max_epochs" or "diverged".
        overfit : int
            overfitting_epochs at the stop.
        gap : float
            generalization_gap of the run.

        Returns
        -------
        dict[str, Any]
            The keys of run() plus architecture, seed, stop_reason,
            overfitting_epochs and generalization_gap; diverged is True
            exactly when reason is "diverged".
        """
        return {
            **run(val=val, test=1.0, diverged=reason == "diverged"),
            "architecture": "A1",
            "seed": seed,
            "stop_reason": reason,
            "overfitting_epochs": overfit,
            "generalization_gap": gap,
        }

    runs = [
        a1(42, 0.2, "early_stopping", 2, 0.1),
        a1(42, 0.5, "max_epochs", 0, 0.9),
        a1(43, 0.3, "early_stopping", 0, 0.3),
        a1(43, float("nan"), "diverged", 0, 5.0),
    ]
    row = next(
        line
        for line in format_stopping_seed_table("T", runs, 2).splitlines()
        if line.startswith("A1")
    )
    assert "2 / 1 / 1" in row
    assert "1 of 2" in row
    assert "0.20 ± 0.14" in row
