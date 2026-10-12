"""
Tests for the Optuna hyperparameter tuning of the framework models.

Tests that need Optuna, TensorFlow or PyTorch are skipped automatically when
that library isn't installed.
"""

import json
import math

import pytest

from nn_numpy.benchmarks.data import load_problem_config, with_dropout
from nn_numpy.benchmarks.report import (
    count_wins,
    describe_config,
    format_three_way_table,
    format_tuning_table,
)

# ------------------------------------------------------------------
# Config and search space
# ------------------------------------------------------------------


def write_config(tmp_path, **changes):
    """
    Write a copy of the real tuning config with some keys changed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest's temporary directory.
    **changes
        Top-level keys to replace (a value of None removes the key).

    Returns
    -------
    str
        Path of the written JSON file.

    Notes
    -----
    Processing:
    1. Load configs/tuning_experiments.json.
    2. Apply the changes and write the result to tmp_path.
    """
    from nn_numpy.benchmarks.tuning import TUNING_CONFIG_PATH

    with open(TUNING_CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)
    for key, value in changes.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    path = tmp_path / "tuning.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


def test_the_real_tuning_config_loads():
    """
    configs/tuning_experiments.json passes validation and has every hyperparameter.
    """
    pytest.importorskip("optuna")
    from nn_numpy.benchmarks.tuning import load_tuning_config

    config = load_tuning_config()

    assert "description" not in config
    assert set(config["search_space"]) == {
        "optimizer",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "dropout",
    }
    assert config["n_trials"] >= 1


@pytest.mark.parametrize(
    "changes",
    [
        {"n_trials": 0},
        {"tuning_seed": None},
        {"search_space": {"optimizer": [], "learning_rate": {"low": 0.1, "high": 0.2}}},
        {"search_space": {"learning_rate": {"low": 0.3, "high": 0.1, "log": True}}},
    ],
)
def test_invalid_tuning_configs_are_rejected(tmp_path, changes):
    """
    A non-positive trial count, a missing key, empty choices or low > high raise ValueError.

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest's temporary directory.
    changes : dict
        The invalid change applied to the real config.
    """
    pytest.importorskip("optuna")
    from nn_numpy.benchmarks.tuning import load_tuning_config

    with pytest.raises(ValueError):
        load_tuning_config(write_config(tmp_path, **changes))


def test_suggest_params_draws_one_value_per_hyperparameter():
    """
    suggest_params returns the trial's value for every hyperparameter, with the right types.

    Notes
    -----
    optuna.trial.FixedTrial answers each suggestion with a given value and
    checks that categorical values are among the choices.
    """
    optuna = pytest.importorskip("optuna")
    from nn_numpy.benchmarks.tuning import load_tuning_config, suggest_params

    fixed = {
        "optimizer": "muon",
        "learning_rate": 0.01,
        "batch_size": 32,
        "weight_decay": 1e-4,
        "dropout": 0.1,
    }
    params = suggest_params(optuna.trial.FixedTrial(fixed), load_tuning_config()["search_space"])

    assert params == fixed


# ------------------------------------------------------------------
# Dropout and weight decay in the framework models
# ------------------------------------------------------------------


def test_with_dropout_sets_hidden_layers_only():
    """
    with_dropout sets the rate on every hidden layer, never on the output, and copies.
    """
    layers = load_problem_config("regression")["architectures"]["A2-bn"]

    changed = with_dropout(layers, 0.2)

    assert [layer.get("dropout", 0.0) for layer in changed] == [0.2, 0.2, 0.2, 0.0]
    assert all("dropout" not in layer for layer in layers)  # the original is untouched
    assert with_dropout(layers, None) is layers


def test_keras_optimizers_take_the_weight_decay():
    """
    Keras optimizers get the given weight decay; None keeps the defaults.
    """
    pytest.importorskip("tensorflow")
    from nn_numpy.benchmarks.keras_models import build_optimizer

    assert build_optimizer("adam", 0.01, 1e-3).weight_decay == pytest.approx(1e-3)
    assert build_optimizer("adam", 0.01).weight_decay is None
    muon = build_optimizer("muon", 0.01, 1e-3)
    assert muon.weight_decay == pytest.approx(1e-3)
    assert muon.adam_weight_decay == pytest.approx(1e-3)


def test_torch_optimizers_take_the_weight_decay():
    """
    PyTorch optimizers, including both parts of Muon, get the given weight decay.
    """
    pytest.importorskip("torch")
    from nn_numpy.benchmarks.torch_models import build_network, build_optimizer

    model = build_network(load_problem_config("regression")["architectures"]["A2-bn"], 8, seed=0)
    sgd = build_optimizer("momentum", model.parameters(), 0.01, 1e-3)
    assert sgd.param_groups[0]["weight_decay"] == pytest.approx(1e-3)
    muon = build_optimizer("muon", model.parameters(), 0.01, 1e-3)
    assert muon.muon.param_groups[0]["weight_decay"] == pytest.approx(1e-3)
    assert muon.adam.param_groups[0]["weight_decay"] == pytest.approx(1e-3)
    default = build_optimizer("muon", model.parameters(), 0.01)
    assert default.muon.param_groups[0]["weight_decay"] == pytest.approx(0.1)


def test_torch_run_model_applies_tuned_dropout():
    """
    A "dropout" value in the grid adds Dropout after every hidden layer of the model.

    Notes
    -----
    The record keeps the tuned hyperparameters in "params", and the run is
    trained for 2 epochs only.
    """
    pytest.importorskip("torch")
    from torch import nn

    from nn_numpy.benchmarks import torch_models
    from nn_numpy.benchmarks.data import load_splits

    built = []
    original = torch_models.build_network

    def spy(layer_configs, input_dimension, seed):
        model = original(layer_configs, input_dimension, seed)
        built.append(model)
        return model

    torch_models.build_network = spy
    try:
        splits, baseline = load_splits("regression", 42)
        grid = {
            "optimizer": ["adam"],
            "learning_rate": [0.01],
            "batch_size": [64],
            "weight_decay": [1e-4],
            "dropout": [0.25],
        }
        record = torch_models.run_model(
            "regression", "A1", grid, splits, 42, baseline, {"epochs": 2}
        )[0]
    finally:
        torch_models.build_network = original

    dropouts = [m for m in built[0] if isinstance(m, nn.Dropout)]
    assert len(dropouts) == 1 and dropouts[0].p == pytest.approx(0.25)
    assert record["params"]["dropout"] == pytest.approx(0.25)


# ------------------------------------------------------------------
# Tuning end to end
# ------------------------------------------------------------------


def test_tune_model_and_evaluate_tuned_with_pytorch():
    """
    A tiny tuning run returns the best trial's configuration, and evaluation covers every seed.

    Notes
    -----
    Two trials of 2 epochs on A1 regression. The best configuration must be
    the trial with the lowest validation metric, every trial must be on the
    tuning seed, and the evaluation must have one tuned record per seed.
    """
    pytest.importorskip("optuna")
    pytest.importorskip("torch")
    from nn_numpy.benchmarks.data import problem_seeds
    from nn_numpy.benchmarks.tuning import evaluate_tuned, load_tuning_config, tune_model

    config = load_tuning_config()
    config["n_trials"] = 2
    training = {"epochs": 2}

    best, trials = tune_model("pytorch", "regression", "A1", config, training)

    assert len(trials) == 2
    assert {t["seed"] for t in trials} == {config["tuning_seed"]}
    finite = [t for t in trials if math.isfinite(t["val_metric"])]
    assert best == min(finite, key=lambda t: t["val_metric"])["params"]

    records = evaluate_tuned("pytorch", "regression", "A1", best, training)
    assert [r["seed"] for r in records] == problem_seeds("regression")
    assert all(r["tuned"] and r["params"] == best for r in records)


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------


def make_run(model, seed, test, val, params):
    """
    Build a minimal framework run record for the report helpers.

    Parameters
    ----------
    model : str
        Architecture name.
    seed : int
        Seed of the split.
    test, val : float
        Test and validation metric.
    params : dict
        Hyperparameters.

    Returns
    -------
    dict
        A record with the keys the report reads.
    """
    return {
        "model": model,
        "seed": seed,
        "test_metric": test,
        "val_metric": val,
        "params": params,
        "diverged": False,
    }


def test_describe_config_includes_tuned_weight_decay_and_dropout():
    """
    A tuned run's description adds weight decay and dropout to the usual fields.
    """
    params = {
        "optimizer": "adam",
        "learning_rate": 0.012345,
        "batch_size": 32,
        "weight_decay": 1.2e-5,
        "dropout": 0.1,
    }

    text = describe_config(make_run("A1", 42, 0.1, 0.1, params))

    assert text == "A1 · adam · LR 0.0123 · bs 32 · wd 1.2e-05 · dropout 0.10"


def test_tuning_table_compares_grid_and_tuned_per_seed():
    """
    The tuning table selects the grid model per seed and counts the seeds where tuning won.

    Notes
    -----
    Two seeds: on seed 1 the tuned model is better, on seed 2 worse, so the
    table must show "1/2 seeds" and the tuned configuration.
    """
    grid_params = {"optimizer": "adam", "learning_rate": 0.1, "batch_size": 16}
    tuned_params = {**grid_params, "weight_decay": 1e-4, "dropout": 0.2}
    grid = [
        make_run("A1", 1, test=0.5, val=0.1, params=grid_params),
        make_run("A1", 1, test=0.1, val=0.9, params=grid_params),  # worse on validation
        make_run("A1", 2, test=0.2, val=0.1, params=grid_params),
    ]
    tuned = [
        make_run("A1", 1, test=0.3, val=0.2, params=tuned_params),
        make_run("A1", 2, test=0.4, val=0.2, params=tuned_params),
    ]

    table = format_tuning_table("regression", "pytorch", grid, tuned)

    row = next(line for line in table.splitlines() if line.startswith("A1"))
    assert "1/2 seeds" in row
    assert "0/2" in row  # no tuned run diverged
    assert "wd 1.0e-04 · dropout 0.20" in row


def test_count_wins_pairs_by_seed_and_ignores_ties():
    """
    count_wins compares only seeds on both sides and counts strict improvements.

    Notes
    -----
    Seed 1: NumPy better. Seed 2: a tie (not a win). Seed 3: NumPy worse.
    Seed 4 exists only on the NumPy side, so it is not compared.
    """
    numpy_selected = [
        {"seed": 1, "test_metric": 0.1},
        {"seed": 2, "test_metric": 0.2},
        {"seed": 3, "test_metric": 0.5},
        {"seed": 4, "test_metric": 0.0},
    ]
    other_selected = [
        {"seed": 1, "test_metric": 0.3},
        {"seed": 2, "test_metric": 0.2},
        {"seed": 3, "test_metric": 0.4},
    ]

    assert count_wins(numpy_selected, other_selected) == (1, 3)
    assert count_wins(numpy_selected, []) == (0, 0)


def test_three_way_table_puts_numpy_grid_and_tuned_in_one_row():
    """
    The three-way table selects each group per seed and shows both win counts.

    Notes
    -----
    Two seeds. NumPy (selected on best_val_loss) beats the grid model on
    both seeds and the tuned model on one, so the row ends "2/2 / 1/2 seeds".
    Diverged NumPy runs are left out of the selection.
    """
    numpy_runs = [
        {"architecture": "A1", "seed": 1, "best_val_loss": 0.1, "test_metric": 0.10},
        {
            "architecture": "A1",
            "seed": 1,
            "best_val_loss": 0.05,
            "test_metric": 9.0,
            "diverged": True,
        },
        {"architecture": "A1", "seed": 2, "best_val_loss": 0.1, "test_metric": 0.30},
    ]
    params = {"optimizer": "adam", "learning_rate": 0.1, "batch_size": 16}
    grid = [
        make_run("A1", 1, test=0.20, val=0.1, params=params),
        make_run("A1", 2, test=0.40, val=0.1, params=params),
    ]
    tuned = [
        make_run(
            "A1", 1, test=0.05, val=0.1, params={**params, "weight_decay": 1e-4, "dropout": 0.1}
        ),
        make_run(
            "A1", 2, test=0.35, val=0.1, params={**params, "weight_decay": 1e-4, "dropout": 0.1}
        ),
    ]

    table = format_three_way_table("regression", numpy_runs, "pytorch", grid, tuned)

    assert (
        table.splitlines()[0]
        == "NumPy NN vs. PyTorch grid vs. PyTorch tuned — regression (test MSE)"
    )
    row = next(line for line in table.splitlines() if line.startswith("A1"))
    assert row.endswith("2/2 / 1/2 seeds")
    assert "0.2000 ± 0.1414" in row  # the NumPy mean over seeds 1 and 2 (diverged run skipped)


@pytest.mark.parametrize("with_tuned", [True, False])
def test_learning_curves_include_the_tuned_model_when_available(tmp_path, monkeypatch, with_tuned):
    """
    The curve figure draws NumPy, grid and (if given) tuned lines, and works without tuning.

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest's temporary directory, used as the figure folder.
    monkeypatch : pytest.MonkeyPatch
        Redirects the figure folder and records the plotted lines.
    with_tuned : bool
        Whether a tuned run is passed.

    Notes
    -----
    Each panel must have 3 lines with a tuned run (the third dashed) and 2
    without; the PNG must be written either way.
    """
    import matplotlib

    matplotlib.use("Agg")
    from nn_numpy.benchmarks import report

    monkeypatch.setattr(report, "BENCHMARK_FIGURES_DIR", str(tmp_path))
    drawn = []
    original_close = report.plt.close

    def record_and_close(fig):
        drawn.extend([[line.get_linestyle() for line in ax.get_lines()] for ax in fig.axes])
        original_close(fig)

    monkeypatch.setattr(report.plt, "close", record_and_close)

    params = {"optimizer": "adam", "learning_rate": 0.1, "batch_size": 16}
    numpy_run = {
        "architecture": "A1",
        "optimizer": "adabelief",
        "learning_rate": 0.1,
        "batch": 16,
        "seed": 42,
        "val_loss_history": [1.0, 0.5, 0.25],
    }
    grid_run = {**make_run("A1", 42, 0.1, 0.1, params), "val_loss_history": [1.0, 0.6]}
    tuned_run = None
    if with_tuned:
        tuned_params = {**params, "weight_decay": 1e-4, "dropout": 0.1}
        tuned_run = {**make_run("A1", 42, 0.1, 0.1, tuned_params), "val_loss_history": [0.9, 0.4]}

    path = report.plot_learning_curves("pytorch", {"regression": (numpy_run, grid_run, tuned_run)})

    assert path.startswith(str(tmp_path)) and path.endswith(".png")
    assert drawn == [["-", "-", "--"]] if with_tuned else drawn == [["-", "-"]]
