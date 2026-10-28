# A NumPy Neural Network vs. scikit-learn, TensorFlow and PyTorch: Detailed Documentation

This is the in-depth reference for the project. For a quick overview, results, and plots, see the [repository README](../README.md).

## Contents

1. [Project overview](#1-project-overview)
   - [Pipeline map](#pipeline-map)
   - [Dependencies](#dependencies)
   - [Running the pipeline](#running-the-pipeline)
   - [Tests](#tests)
2. [Experiment configuration](#2-experiment-configuration)
3. [Analysis details](#3-analysis-details)
   - [comparisons.py](#comparisonspy)
   - [analysis.py](#analysispy)
4. [Library comparison](#4-library-comparison)
5. [Knowledge base](#5-knowledge-base)

---

## 1. Project overview

This project implements a feed-forward neural network in plain NumPy for two tasks: binary classification and regression.

1. The pipeline downloads the datasets.
2. It preprocesses them: cleaning, optional scaling, and a train / validation / test split (60 / 20 / 20).
3. It loads the experiment settings (architectures, optimizers, learning rates, batch sizes, early stopping) from JSON configuration files.
4. It trains and evaluates every configured network and saves the results.
5. Separate scripts read the saved results to generate comparison plots and a written analysis.

The repository follows the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) layout: raw data in `data/raw/`, generated results in `reports/` and figures in `reports/figures/`, and all source code in the `nn_numpy` package. Its `nn/` subpackage is the neural-network library and `modeling/` holds the trainer and the experiment sweep. `configs/` (experiment definitions) and `tools/` (repository helpers) are this project's additions to the template, and `data/{external,interim,processed}/`, `models/`, `notebooks/` and `references/` are kept empty, with a `.gitkeep`, for future use.

| Task | Dataset | Target | Loss | Evaluation metric |
|---|---|---|---|---|
| Classification | Banknote Authentication (UCI #267) | `class` (0/1) | Binary cross-entropy (BCE) | BCE |
| Regression | Energy Efficiency (UCI #242) | `Heating_Load` (`Cooling_Load` is dropped) | Mean squared error (MSE) | MSE |

The same quantity is used for training and for evaluation, so the reported metric is directly comparable with the loss curves. The `test_metric` column in `reports/main_summary_<stamp>.csv` holds the **test-set BCE** for classification rows and the **test-set MSE** for regression rows. Lower is better in both cases. The metric is chosen in `nn_numpy/modeling/trainer.py` (`Trainer._compute_metric`) and implemented in `nn_numpy/nn/metrics.py`, which reuses the loss classes from `nn_numpy/nn/losses.py`.

### Pipeline map

| Module | Role | Reads | Writes |
|---|---|---|---|
| `nn_numpy/config.py` | Shared constants: project paths and `RANDOM_SEED`; the per-process `RUN_STAMP` and the `stamped_filename` / `latest_stamped_file` / `resolve_results_path` helpers | – | – |
| `nn_numpy/dataset.py` | Downloads the raw UCI datasets with `ucimlrepo` if they are not cached | – | `data/raw/banknote_auth.csv`, `data/raw/energy_efficiency.csv` |
| `nn_numpy/features.py` | Removes duplicates, splits train/val/test with a NumPy-only `train_val_test_split` (stratified by class for classification), optionally standardizes features (fit on train only), runs EDA plots | `data/raw/*.csv` | `reports/figures/eda/*.png` |
| `nn_numpy/scalers.py` | Custom standard scaler, (x − μ) / σ | – | – |
| `nn_numpy/plots.py` | Exploratory and preprocessing plots used by `features.py` | – | – |
| `nn_numpy/config_loader.py` | Loads and validates the JSON experiment configs | `configs/*.json` | – |
| `nn_numpy/runlog.py` | Timestamped run logs, phase timings and progress lines for the long-running entry points; see [Run logs](#run-logs) | – | `logs/run_<YYYYmmdd_HHMMSS>_<mode>.log` |
| `nn_numpy/nn/layers.py` | Dense layer, Z = X·W (+ b), with its backward pass; optional bias and `normal` / `he` / `xavier` weight initialization. BatchNorm layer with learned scale and shift, running statistics, and a training / evaluation mode. Inverted Dropout layer | – | – |
| `nn_numpy/nn/activations.py` | ReLU, Sigmoid, Tanh, Linear (forward and backward) | – | – |
| `nn_numpy/nn/network.py` | Builds the network from a config; full forward and backward passes | – | – |
| `nn_numpy/nn/losses.py` | BCE and MSE (forward and gradient) | – | – |
| `nn_numpy/nn/optimizers.py` | SGD, Momentum SGD, AdaBelief, Muon (with its Newton–Schulz orthogonalization) | – | – |
| `nn_numpy/nn/metrics.py` | Evaluation metrics: BCE (classification) and MSE (regression) | – | – |
| `nn_numpy/modeling/trainer.py` | Mini-batch training loop, validation, early stopping, best-model restore, divergence detection, test evaluation; switches the network between training and evaluation mode | – | – |
| `nn_numpy/modeling/train.py` | Orchestrates the full experiment grid (every config and seed) | everything above | `reports/main_results_full_<stamp>.json`, `reports/main_summary_<stamp>.csv` |
| `nn_numpy/comparisons.py` | Optimizer, depth, and learning-rate comparison plots with short text analyses | newest `reports/main_results_full_*.json` (or a path given as the first argument) | `reports/figures/comparisons/*_<stamp>.png`, `reports/comparisons/*_<stamp>.txt` |
| `nn_numpy/analysis.py` | Aggregate analysis across all runs | newest `reports/main_results_full_*.json` (or a path given as the first argument) | `reports/analysis_<stamp>.txt` |
| `nn_numpy/benchmarks/run.py` | Trains the library models (scikit-learn, TensorFlow, PyTorch) on the same splits; see [section 4](#4-library-comparison) | `configs/benchmark_experiments.json`, `data/raw/*.csv` | `reports/benchmark_<library>_results_<stamp>.json` |
| `nn_numpy/benchmarks/tuning.py` | Tunes the Keras and PyTorch models with Optuna on seed 42's validation set, then tests the best configuration on every seed; see [Hyperparameter tuning](#hyperparameter-tuning-optuna) | `configs/tuning_experiments.json`, `configs/benchmark_experiments.json`, `data/raw/*.csv` | `reports/benchmark_<library>_tuned_results_<stamp>.json`, `reports/tuning_<library>_trials_<stamp>.json` |
| `nn_numpy/benchmarks/report.py` | Compares the NumPy network with the library models, grid-selected and tuned | newest NumPy and library results | `reports/benchmark_report_<stamp>.txt`, `reports/figures/benchmarks/*.png` |

`nn_numpy/modeling/train.py` has a `SHOW_EDA` flag (default `False`). Set it to `True` to display the preprocessing plots interactively while the pipeline runs. The plots are saved to `reports/figures/eda/` either way.

#### Run stamps

Every file a run writes carries a stamp of the form `name_YYYYMMDD-HHMMSS.ext` (local time), so nothing from an earlier run is ever overwritten. The stamp is `RUN_STAMP` in `nn_numpy/config.py`, computed once when the module is first imported, so all files written by one process share it. The training sweep, `nn_numpy.comparisons` and `nn_numpy.analysis` are separate processes and therefore get their own stamps. `nn_numpy.comparisons` and `nn_numpy.analysis` print which results file they used.

Only the newest version of each output is committed. `.gitignore` ignores all stamped files under `reports/`, and the pre-commit hook in `.githooks/` runs `tools/stage_latest_outputs.py`, which:

- force-adds the newest stamped file of each output (using `latest_stamped_outputs` in `nn_numpy/config.py`)
- removes older stamped files from the index, leaving them on disk
- rewrites stamped output names in `README.md` and `docs/DETAILS.md` to the newest stamps, and stops the commit if either file has unstaged edits, so those edits are never committed by accident

Each output is handled on its own, so the newest analysis report may come from a different run than the newest plots. Enable the hook once per clone with `git config core.hooksPath .githooks`. It runs with `.venv`'s Python if present, otherwise `python`.

#### Run logs

`make train`, `make benchmarks`, `make tune`, `make plots`, `make analysis` and `make benchmark-report` each write a log to `logs/run_<YYYYmmdd_HHMMSS>_<mode>.log` (`nn_numpy/runlog.py`). The log is a tee on the process's stdout and stderr: every line (prints, logger lines, Python warnings, tracebacks, and the TensorFlow and PyTorch backends' output) is prefixed with `YYYY-MM-DD HH:MM:SS | `, shown on the terminal and appended to the file. Logger lines carry no timestamp of their own, so each line has exactly one.

Each run is split into phases (for example `Train classification`, `Train regression` and `Save results` in `make train`, or one phase per library in `make benchmarks` and `make tune`). Each phase is logged with its duration, and the log ends with a table of the phases' seconds and share of the run.

The long loops print one progress line per unit of work, for example:

```text
A2 | Run 87/1440 | val MSE: 0.512345 | 1s | Params: {'problem': 'regression', 'optimizer': 'adabelief', 'lr': 0.1, 'batch': 16, 'seed': 42}
A2 | Config 3/16 | val MSE: 0.412345 | 9s | Params: {'optimizer': 'adam', 'learning_rate': 0.001, 'batch_size': 16}
A2 | Trial 7/50 | val MSE: 0.312345 | 6s | Params: {'optimizer': 'muon', 'learning_rate': 0.02, ...}
```

A loop prints at most about 50 lines: every item when it has up to 50 (a framework's configurations, a study's trials) and every `ceil(n / 50)`-th item otherwise (every 29th of the 1,440 NumPy runs). No single fit takes longer than about 10 seconds, so there are no per-epoch lines. Logs differ per computer and run, so `logs/` is git-ignored.

### Dependencies

Pinned versions are in [`requirements.txt`](../requirements.txt). The project was developed with Python 3.12.

The model, training, metrics, scaling, and data splitting use **NumPy only**. The other packages handle loading the CSVs, plotting, and downloading the datasets.

```
numpy==1.26.4        # all model, training, and preprocessing math
pandas==2.2.2        # loading and cleaning the CSV files
matplotlib==3.9.2    # plots
seaborn==0.13.2      # EDA plots
ucimlrepo==0.0.7     # one-time dataset download
```

### Running the pipeline

Run every command from the repository root. The `Makefile` wraps each step (`make help` lists them all); the equivalent Python commands are shown next to each one.

```bash
make requirements   # pip install -r requirements.txt && pip install -e ".[dev]"

# 1. Load or download the data, preprocess, train all configured experiments.
#    Writes reports/main_results_full_<stamp>.json and reports/main_summary_<stamp>.csv
make train          # python -m nn_numpy.modeling.train

# 2. Optimizer / network-depth / learning-rate / variant / dropout comparisons.
#    Writes stamped plots to reports/figures/comparisons/ and short analyses to reports/comparisons/
make plots          # MPLBACKEND=Agg python -m nn_numpy.comparisons

# 3. Aggregate analysis.
#    Writes reports/analysis_<stamp>.txt
make analysis       # python -m nn_numpy.analysis

# Or all three in order:
make all
```

Steps 2 and 3 read the newest `reports/main_results_full_*.json` and fail if none exists. To analyse a specific run instead, pass its path as the first argument, e.g. `python -m nn_numpy.analysis reports/main_results_full_20261028-135829.json`. The repository already includes a results file, so you can run them immediately. `make plots` sets `MPLBACKEND=Agg` so no plot windows open; set it yourself when running the Python command directly.

`make` is not installed on Windows by default: install it with `winget install ezwinports.make` and open a new terminal.

#### Which command trains what

Three commands train models, and they train different ones. Only `make train` trains the NumPy network; `make benchmarks` and `make tune` train the comparison libraries.

| Command | Trains | How | Writes | Purpose |
|---|---|---|---|---|
| `make train` | **The NumPy network** | The fixed grid: 9 architectures × 4 optimizers × 2 learning rates × 2 batch sizes × 5 seeds = 1,440 runs (about 16 min) | `reports/main_results_full_<stamp>.json`, `main_summary_<stamp>.csv` | The main results and key findings |
| `make benchmarks` | **scikit-learn, Keras and PyTorch** | Each library's fixed grid in `configs/benchmark_experiments.json`, on every seed (about 2 h, mostly TensorFlow) | `reports/benchmark_<library>_results_<stamp>.json` | The standard library comparison |
| `make tune` *(optional)* | **Keras and PyTorch** | An Optuna search: 50 trials per framework, task and architecture on seed 42's validation set, then the best configuration on every seed (about 1.5 h) | `reports/benchmark_<library>_tuned_results_<stamp>.json`, `tuning_<library>_trials_<stamp>.json` | Whether tuned frameworks change the comparison |
| `make benchmark-report` | Nothing | Reads the newest results of all of the above | `reports/benchmark_report_<stamp>.txt`, `reports/figures/benchmarks/` | The comparison report and figures |

`make plots` and `make analysis` also train nothing: they read the newest `make train` results. The comparison report works without `make tune`; it then leaves out the tuned rows and tables. See [Library comparison](#4-library-comparison) and [Hyperparameter tuning](#hyperparameter-tuning-optuna) for details.

### Tests

```bash
make test           # python -m pytest
make lint           # ruff format --check and ruff check (reports only)
make format         # ruff check --fix and ruff format (rewrites files)
```

- `tests/test_gradients.py` compares every analytic gradient against a central finite-difference estimate: each activation, both losses, and the weight and bias gradients of whole A1- and A2-style networks, with and without bias terms and with each initialization scheme.
- `tests/test_layers.py` checks the Dense layer's bias (forward shift, batch-summed gradient), that each initialization scheme has the right standard deviation and that the default reproduces the original weights exactly, that every optimizer updates the bias as well as the weights, that early-stopping checkpoints restore the bias, and that the config loader rejects invalid `use_bias` / `init` values.
- `tests/test_batchnorm.py` checks batch normalization: training mode normalizes over the batch and updates the running statistics; evaluation mode uses them and gives a single sample the same output alone as inside a batch; its gradients (input, gamma, beta) match finite differences in both modes. It also checks the network's `train()` / `eval()` switch, that checkpoints restore the running statistics, and that the trainer stops and flags a diverging run.
- `tests/test_dropout.py` checks inverted dropout: in training mode it drops about `rate` of the values with a new mask per batch and scales the survivors so the expected value is unchanged; backward reuses the same mask; evaluation mode is the identity. It also checks that dropout sits after the activation, doesn't change the initial weights, is reproducible from the seed, and that the config loader rejects invalid rates and dropout on the output layer.
- `tests/test_muon.py` checks the Muon optimizer: Newton–Schulz pushes the singular values towards 1 and keeps the singular vectors, the first step follows the update rule exactly (with and without Nesterov), biases and batch-norm parameters follow AdaBelief exactly, and three steps agree with PyTorch's `torch.optim.Muon` (skipped without PyTorch).
- `tests/test_utils.py` checks the run-stamp helpers used for output file names.
- `tests/test_metrics.py` checks the accuracy and R² metrics.
- `tests/test_tuning.py` checks the tuning: the config validation, that each trial draws one value per hyperparameter, that dropout reaches every hidden layer and weight decay reaches each framework's optimizers (both parts of Muon), that a short tuning run picks the trial with the lowest validation metric and evaluates it on every seed, and the report's grid-vs-tuned and three-way tables (with the seed pairing of `count_wins`) and the tuned line in the learning-curve figures.
- `tests/test_benchmarks.py` checks that the library comparison uses exactly the main pipeline's splits, the model selection and summaries in the report, the scikit-learn wrappers, and the Keras and PyTorch models (layer structure, optimizers including Muon, training records, reproducibility; for PyTorch also the hand-written early stopping, divergence detection, and the split of parameters between Muon and Adam). Library tests are skipped automatically when that library isn't installed.
- `tests/test_seeds.py` checks that the split and the weight initialization follow the seed, the constant-prediction baselines, the `seeds` config validation, and the multi-seed analysis helpers (model selection by validation loss, error removed, mean ± std).
- `tests/test_knowledge_base.py` checks the generated OWL knowledge base: it reads back to the same graph and is written identically twice, no IRI has two kinds and nothing references an undeclared name, every assertion respects its property's domain and range, and the OWL RL reasoner infers the pipeline order, stage membership, module roles and architecture families (see [section 5](#5-knowledge-base)).
- `tests/test_clipping.py` checks gradient clipping: the global gradient norm covers every parameter (weights, biases, batch norm's γ and β), clipping scales an over-long gradient to exactly the cap without changing its direction and leaves smaller ones untouched, a clipped SGD step moves the parameters by at most `lr × cap`, a run that diverges without clipping stays finite with it, and the config loader accepts the `max_grad_norm` option and rejects invalid ones.
- `tests/test_preprocessing.py` checks that the NumPy split has the right sizes, has no overlapping rows, preserves class balance when stratified, and is reproducible from the seed.
- `tests/test_training.py` checks that each optimizer reduces the loss on a toy problem, and that `Trainer.fit` / `Trainer.evaluate` run end to end and report BCE as the classification metric.

---

## 2. Experiment configuration

Two JSON files define the experiment setup:

- `configs/classification_experiments.json`
- `configs/regression_experiments.json`

They control a large part of the pipeline, within the limits of the current implementation. They are configurable, but they don't define a fully general neural-network framework.

### What the JSON controls

**Preprocessing**
- `preprocessing.enabled`: if `true`, the dataset is cleaned and can optionally be scaled. If `false`, cleaning, EDA, and scaling are skipped, but the data is still split into train/validation/test because training requires those splits.
- `preprocessing.scale_features` (`true`/`false`): standardizes input features with the custom `StandardScaler`, using statistics fit on the training split only. If preprocessing is disabled, scaling is skipped too.

**Architectures**
- Architecture names (for example `A1`, `A2`): any JSON key is allowed, so you can add new architectures.
- Number of layers: at least 1.
- `units` per layer: any integer ≥ 1. This is the layer's output size. There's no upper bound beyond compute and memory.
- `activation` per layer: one of `relu`, `sigmoid`, `tanh`, `linear`.
- `use_bias` per layer (optional, default `false`): adds a trainable bias vector, so the layer computes X·W + b. The bias starts at zero.
- `init` per layer (optional, default `normal`): the weight initialization scheme.
  - `normal`: N(0, 0.1²), the original fixed scale
  - `he`: N(0, 2 / fan_in), designed for ReLU
  - `xavier`: N(0, 2 / (fan_in + fan_out)), designed for sigmoid and tanh

  The defaults reproduce the original bias-free layer exactly, including the random weights drawn, so adding these keys to a new architecture never changes the results of the existing ones.

- `batch_norm` per layer (optional, default `false`): inserts a batch-normalization layer between the Dense layer and its activation (Dense → BatchNorm → activation). It draws no random numbers, so it doesn't change any Dense layer's initial weights.

- `dropout` per layer (optional, default `0`): adds inverted dropout with this rate after the layer's activation (Dense → (BatchNorm) → activation → Dropout). It must be in [0, 1) and isn't allowed on the output layer. Dropout masks come from a separate seeded generator, so adding dropout doesn't change any layer's initial weights.

An architecture can also be a dict with its layer list under `layers` and training options; see "Gradient clipping" below.

The configs define nine architectures:
- the baselines `A1` (1 hidden sigmoid layer) and `A2` (3 hidden ReLU layers)
- four variants of A2: `A2-bias` adds bias terms, `A2-he` uses He initialization, `A2-bias-he` does both, and `A2-bn` adds batch normalization to the three hidden layers. `A2-bn` has no Dense biases, because batch norm's β takes over their role.
- two dropout versions: `A1-dropout` and `A2-bn-dropout`, which add dropout 0.2 after every hidden layer of `A1` and `A2-bn`
- `A2-clip`: the layers of `A2`, trained with gradient-norm clipping at 20

#### Gradient clipping

An architecture can be given as a dict instead of a layer list, to add a training option:

```json
"A2-clip": {"layers": [ ...the same layers as A2... ], "max_grad_norm": 20.0}
```

`max_grad_norm` (a positive number) turns on **global-norm gradient clipping** for that architecture: after each mini-batch's backward pass, if the L2 norm of all gradients together (weights, biases, batch norm's γ and β) exceeds the cap, every gradient is multiplied by `max_grad_norm / norm` before the optimizer step (`NeuralNetwork.clip_gradients`). The update keeps its direction and only its length is capped; gradients below the cap are untouched. This is the rule from Pascanu et al. (2013), the same as Keras's `clipnorm` and PyTorch's `clip_grad_norm_`. `ConfigLoader` validates the option and splits it off into `config["architecture_options"]`, so `config["architectures"][name]` is always a layer list. Architectures without the option train exactly as before.

Every run also records `grad_norm_history`: the largest gradient norm of any mini-batch in each epoch, measured before clipping. It shows the cause of the regression failures directly: in the full sweep, every diverged run's gradient norm exceeds 1,000 in its first epoch, and 37 of the 55 overflow outright, while a healthy shallow run (A1 · AdaBelief) peaks around 130 and typically sits near 3. One huge step pushes every ReLU unit into its dead region (the collapse) or overflows the loss (the divergence). The analysis lists each diverged run's peak gradient norm.

**Why a cap of 20.** A prototype on the 30 failing runs (10 regression configurations at LR 0.1 × 3 seeds) and on healthy runs gave:

| Cap | Failing runs: learned / collapsed / diverged | Effect on healthy runs |
|---|---|---|
| none | 1 / 13 / 16 | – |
| 1 or 5 | 30 / 0 / 0 | cripples shallow runs (A1 · SGD: test MSE 1.0 → 7.3 at a cap of 1) |
| 10 | 30 / 0 / 0 | slows deep AdaBelief runs (A2: MSE 2.7 → 8.3) |
| **20** | **30 / 0 / 0** | classification almost unchanged (most of its gradient norms are below 5); shallow regression runs within noise |
| 50 | 21 / 9 / 0 | |
| 100 | 5 / 25 / 0 | |

Any cap prevents the overflow, but only caps of about 20 or less also keep the ReLUs alive. The regression gradients are large even in healthy deep runs (a median norm of 50–150), because the targets are unscaled (6–43), so a much smaller cap would bind on almost every batch. Clipping is a variant here, not the default: the other architectures and the library comparisons are trained without it.

**Result on the full sweep (5 seeds).** On the 20 regression runs at LR 0.1 with SGD or momentum, `A2-clip` learns in 18 and collapses in 2, with no divergence; A2 diverged in 8 and collapsed in the other 12. No `A2-clip` run diverged anywhere. With momentum, the clipped runs reach a test MSE of 5–14; with plain SGD, each step is capped at LR × 20 = 2, and the runs end far behind (63–110, two no better than the mean). The cap also binds on healthy runs: A2's best regression result per seed goes from 1.79 ± 0.80 to 4.50 ± 2.41, because AdaBelief at LR 0.1 is slowed. On classification, 57 of 60 runs are identical to A2's; the 3 that differ are AdaBelief runs at LR 0.1 whose gradients spike (up to 270).

#### Muon

Muon (`nn_numpy/nn/optimizers.py`, class `Muon`) is selected with `"muon"` in the optimizers list. For a Dense layer's weights W of shape (fan_in, fan_out) with gradient g, each step is:

1. buffer = 0.95 · buffer + 0.05 · g
2. u = 0.05 · g + 0.95 · buffer (Nesterov momentum)
3. O = Newton–Schulz(u): five quintic iterations X ← a·X + (b·XXᵀ + c·(XXᵀ)²)·X with (a, b, c) = (3.4445, −4.7750, 2.0315), after scaling u to unit Frobenius norm
4. W ← W − lr · √max(1, fan_out / fan_in) · O

The iteration pushes every singular value of u towards 1 while keeping its singular vectors, so each direction of the weight matrix gets a step of similar size. The shape factor is PyTorch's default ("original") adjustment, written for the NumPy network's (fan_in, fan_out) layout.

Muon is defined for weight matrices only, so the step size depends on the learning rate and not on the gradient's magnitude. Parameters are routed by name: `"weights"` go to Muon, and biases and batch norm's γ / β go to AdaBelief with the same learning rate. Name-based routing is needed because the NumPy network stores biases as (1, n) arrays. As in the other optimizers, there is no weight decay.

`tests/test_muon.py` checks:
- the singular values after Newton–Schulz, and that the singular vectors are kept
- the exact first step
- the routing to AdaBelief, including for batch norm
- agreement with `torch.optim.Muon` over three steps, within 2e-3; PyTorch runs Newton–Schulz in bfloat16 and the NumPy network in float64

#### Training and evaluation mode

Batch norm behaves differently while learning and while being measured, so the network has `train()` and `eval()` methods that set a `training` flag on every layer. The trainer uses training mode for the mini-batch updates and evaluation mode for validation, `evaluate()` and `predict()`, and leaves the network in evaluation mode after `fit()`. In training mode, batch norm normalizes with the current batch's statistics and updates running averages of them. In evaluation mode it uses the running averages, so a prediction depends only on its own input. Dense and activation layers ignore the flag.

#### Per-run seed and constant-prediction baseline

Each run records its `seed` and a `baseline_test_metric`: the test metric of a model that ignores the features. For classification that model always predicts the training class proportion; for regression, the training mean. Because it is computed from that seed's split, "error removed" (1 − test / baseline; R² for regression, McFadden's pseudo-R² for classification) can be reported for every seed.

#### Accuracy, R² and training time

Each run also records `test_accuracy` (classification, at a 0.5 threshold) or `test_r2` (regression), computed with the NumPy functions in `nn_numpy/nn/metrics.py`, and `train_seconds`, the wall-clock time of `Trainer.fit`. They are easier to read than BCE / MSE and are used by the library comparison ([section 4](#4-library-comparison)). BCE and MSE remain the metrics used for training and model selection.

#### Training metric for the generalization gap

Each run also records `train_metric`: the BCE or MSE on the training set, measured in evaluation mode on the restored best model (`Trainer.score`). The training loss recorded during `fit()` can't be used for this, because it's measured in training mode and, with dropout, includes the dropout noise. `test_metric − train_metric` is the generalization gap: a large positive gap means overfitting. The analysis adds it to every run as `generalization_gap`.

#### How a run ends

Each run records `stop_reason`, one of:
- `diverged`: the training or validation loss of an epoch became NaN or infinite, and the trainer stopped immediately (`diverged` is also set). If an earlier epoch was finite, early stopping still restores that checkpoint, so a diverged run can report a finite but meaningless test metric. The analysis therefore uses the flag rather than the metric.
- `early_stopping`: the validation loss didn't improve by more than `min_delta` for `patience` consecutive epochs after the minimum-epoch guard.
- `max_epochs`: training used every epoch.

Each run also records `overfitting_epochs`: the number of consecutive epochs, ending at the stop, in which the training loss still improved by more than `min_delta` while the validation loss did not. That is the overfitting signature, so an early stop with `overfitting_epochs > 0` stopped because the model was starting to overfit, and one with 0 stopped on a plain plateau. It is 0 for runs that didn't end by early stopping. Because it uses `min_delta` (1e-4) as the size of a meaningful improvement, runs whose losses are already far below that can't show it; the generalization gap covers them.

**Experiment sweep values**
- `optimizers`: `sgd`, `momentum` (aliases `momentumsgd`, `momentum_sgd`), `adabelief`, `muon`.
- `learning_rates`: positive numbers, typically 0.1, 0.01, or 0.001.
- `batch_sizes`: positive integers, bounded by the size of the training split.
- `seeds` (optional, default `[42]`): the whole sweep is repeated once per seed. Each seed sets the train / validation / test split, the weight initialization, the dropout masks and the per-epoch shuffling. Both configs use `[42, 43, 44, 45, 46]`. Seed 42 reproduces the original single-seed results exactly, and EDA plots are only made for the first seed.

**Training settings**
- `epochs`: integer ≥ 1.
- `early_stopping`: `true` or `false`.
- `patience`: the number of consecutive epochs without a meaningful validation-loss improvement allowed before stopping.
- `min_delta`: the minimum decrease in validation loss that counts as an improvement. It should be small relative to the scale of the chosen loss.
- `min_epochs_before_early_stop`: the number of epochs that must finish before early stopping can trigger.

**Problem-level settings**
- `task_type`: `classification` or `regression`.
- `loss`: classification accepts `bce`, `binary_crossentropy`, `binary_cross_entropy`. Regression accepts `mse`.
- `input_dimension`: must match the number of dataset features, or execution fails.

### What the JSON does not control

1. **Layer types.** Each layer has a `"type"` field, but only `dense` is implemented; batch norm is attached to a dense layer with `batch_norm`. Batch norm and dropout are attached to a dense layer with `batch_norm` and `dropout`. Convolutional or recurrent layers would need code changes first. (Optimizers and checkpoints already handle any number of parameters per layer, through `get_params()` / `get_grads()`, and non-trainable state through `get_buffers()`.)
2. **Optimizer internals.** Only the learning rate comes from JSON. Momentum uses β = 0.9; AdaBelief uses β₁ = 0.9, β₂ = 0.999, ε = 1e-8; Muon uses Nesterov momentum 0.95 and 5 Newton–Schulz steps, with AdaBelief for biases and batch-norm parameters. All are fixed in `nn_numpy/nn/optimizers.py` (`get_optimizer`).
3. **Evaluation metrics.** The task type determines them: BCE for classification and MSE for regression, matching the training losses.
4. **The default random seed.** `RANDOM_SEED` in `nn_numpy/config.py` (42) is used when a config lists no `seeds`, and it's the seed that the single-seed analysis sections and the comparison plots use.
5. **EDA display.** It's controlled by `SHOW_EDA` in `nn_numpy/modeling/train.py` (see above).

### Parameter choices

Values that the grid sweeps (learning rates, batch sizes, optimizers, architectures, the scikit-learn grids) or that Optuna searches are part of the experiment, so they need no justification. The fixed values are:

| Value | Where | Justification |
|---|---|---|
| 60 / 20 / 20 train / validation / test split | `nn_numpy/features.py` | The project's fixed split, stratified for classification |
| Seeds 42–46 | `seeds` in both configs | The values carry no meaning; 42 reproduces the original single-seed runs. Five seeds: no recorded justification |
| 100 epochs | `epochs` | No recorded justification |
| Patience 10, `min_delta` 1e-4, 20-epoch guard | `experiments` | No recorded justification. The Keras and PyTorch runs copy them, so all three stop by the same rule |
| 32 units in every hidden layer | `architectures` | No recorded justification |
| Dropout rate 0.2 | `A1-dropout`, `A2-bn-dropout` | No recorded justification (dropout is compared on vs. off; the rate is fixed) |
| Gradient-norm cap 20 | `A2-clip` | The cap experiment under [Gradient clipping](#gradient-clipping) |
| `normal` initialization, σ = 0.1 | `nn_numpy/nn/layers.py` | No recorded justification; kept so the original results reproduce |
| Momentum β = 0.9 | `nn_numpy/nn/optimizers.py` | No recorded justification |
| AdaBelief β₁ = 0.9, β₂ = 0.999, ε = 1e-8 | `nn_numpy/nn/optimizers.py` | Adam's recommended defaults (Kingma & Ba, 2015); AdaBelief keeps Adam's β values |
| Muon: Nesterov momentum 0.95, 5 Newton–Schulz steps and their coefficients, ε = 1e-7 | `nn_numpy/nn/optimizers.py` | PyTorch's and Keras's Muon defaults |
| BatchNorm momentum 0.1, ε = 1e-5 | `nn_numpy/nn/layers.py` | PyTorch's `BatchNorm1d` defaults |
| BCE clip 1e-12, scaler floor 1e-8, sigmoid input clip ±500 | `nn_numpy/nn/losses.py`, `nn_numpy/scalers.py`, `nn_numpy/nn/activations.py` | Numerical safety; the calculations are in the code comments |
| Convergence tolerance 1%, floor 1e-4 | `nn_numpy/analysis.py`, `nn_numpy/comparisons.py` | 1%: no recorded justification. The floor equals `min_delta` |
| Accuracy threshold 0.5 | `nn_numpy/nn/metrics.py` | Predicts the more probable class |
| scikit-learn: `max_iter` 5000 (logistic regression), 300 trees, 200 boosting iterations, MLP `max_iter` 500 / 1000 | `configs/benchmark_experiments.json` | No recorded justification |
| scikit-learn: RBF kernel and `gamma="scale"` for SVM / SVR | `configs/benchmark_experiments.json` | scikit-learn's defaults |
| scikit-learn MLPs: 32 sigmoid units, and 3 × 32 ReLU units | `configs/benchmark_experiments.json` | They mirror A1 and A2 |
| Optuna: 50 trials, `sampler_seed` 0, search on seed 42 | `configs/tuning_experiments.json` | The seeds carry no meaning beyond reproducibility. 50 trials: no recorded justification |

### Summary

The JSON files make the experiment setup mostly configurable: architectures, optimizer selection, learning rates, batch sizes, training length, early stopping, and the supported task and loss choices.

The implementation itself supports only:
- feed-forward dense networks
- a fixed set of activations, losses, and optimizers
- fixed optimizer hyperparameters beyond the learning rate

---

## 3. Analysis details

### comparisons.py

`comparisons.py` compares optimizers, network depth, and learning rates. It loads the full JSON results file, selects matched runs, plots their training and validation loss curves, and saves the figures. For the depth and learning-rate comparisons, it also writes a short text analysis saying which setting converged faster and which ended with the lower loss.

Each comparison holds every setting fixed except one:

| Comparison | Fixed | Varies | Output |
|---|---|---|---|
| Optimizers | problem, architecture, learning rate, batch size | optimizer (SGD → Momentum → AdaBelief → Muon, one subplot each) | `optimizers_*_<stamp>.png` |
| Network depth | problem, optimizer, learning rate, batch size | architecture (A1 vs A2) | `depth_*_<stamp>.png`, `depth_analysis_<stamp>.txt` |
| Learning rate | problem, architecture, optimizer, batch size | learning rate (0.1 vs 0.001) | `learning_rate_*_<stamp>.png`, `learning_rate_analysis_<stamp>.txt` |
| A2 variants | problem, optimizer, learning rate, batch size | A2 → A2-bias → A2-he → A2-bias-he → A2-bn → A2-clip, one subplot each, log-scale loss | `a2_variants_*_<stamp>.png` |

There are two A2 variant plots, both regression at LR 0.1, where the baseline A2 collapses:
- SGD, batch 16: shows all three failure modes (collapse to 0, collapse to the mean, divergence)
- momentum, batch 64: no variant diverges, so each fix's outcome is visible side by side

Both are skipped for results files that contain no variant runs.

The **dropout plot** (`dropout_regression_adabelief_lr01_bs16_<stamp>.png`) compares A1 with A1-dropout and A2-bn with A2-bn-dropout for regression, AdaBelief, LR 0.1 and batch 16, the setting of the best baseline regression run. With dropout, the plotted training loss is measured with dropout active, so it sits above the validation loss. That is dropout's expected signature, not a sign of a problem. The plot is skipped for results files without dropout runs.

#### Convergence epoch

Convergence is measured from the training-loss curve:

1. `final_train_loss` is the last value in `train_loss_history`.
2. `tolerance = max(absolute_floor, relative_tolerance * abs(final_train_loss))`, with `relative_tolerance = 0.01` and `absolute_floor = 1e-4`. In practice this is a 1% band around the final loss that never gets narrower than 0.0001.
3. Starting from epoch 1, scan forward to find the first epoch after which all remaining training-loss values stay within that band.

That epoch is the **convergence epoch**, a practical definition of when training has essentially settled. The same helper also records `epochs_ran`, `final_train_loss`, and `final_val_loss`. These values appear in the metrics box on the depth and learning-rate plots, together with the run's **best epoch**.

#### Convergence, early stopping and overfitting

Three related ideas, each measured on a different loss:

- **Convergence** means the optimizer has settled: the *training* loss stops changing. The convergence epoch above measures it. It says how fast an optimizer gets there, not whether the model generalizes.
- **Early stopping** is driven by the *validation* loss only. The trainer tracks the best checkpoint from epoch 1, stops after `patience` epochs without a meaningful validation improvement (after the minimum-epoch guard), and restores that checkpoint. Its epoch is the run's `best_epoch`, and it is the model whose test metric is reported.
- **Overfitting** is training loss still falling while validation loss stops improving or rises, so the gap between them widens. Each run records it as `overfitting_epochs` (see "How a run ends") and as the generalization gap.

The plots mark both epochs: a dotted line where training ended and a dash-dot line at the best epoch. The analysis tables report the average convergence epoch and best epoch side by side.

#### Helper functions

- `load_results(path)`: loads the JSON file
- `filter_optimizer_runs(...)`: selects the 3 matched optimizer runs
- `filter_depth_runs(...)`: selects the 2 matched architecture runs
- `filter_learning_rate_runs(...)`: selects the 2 matched learning-rate runs
- `_compute_convergence_metrics(run)`: computes final losses and the convergence epoch
- `_plot_loss_curves(...)`: draws train/validation curves for one run
- `_save_and_show(...)`: saves the figure and displays it
- `_write_text_file(...)`: saves the short written analysis

`main()` chooses which experiments to visualize by calling the three plot functions with fixed arguments. To visualize different runs, change those arguments. The comparison logic stays the same.

### analysis.py

`analysis.py` answers three questions:

1. Which optimizer converges fastest on average?
2. Which optimizer reaches the best loss?
3. How does network depth affect optimization?

#### Derived metrics

For each run, it computes:
- `best_val_loss`: the validation loss saved with the run, i.e. of the checkpoint that early stopping restored and whose test metric is reported (the minimum of `val_loss_history` only for older files without it). The plain minimum of the history is not used, because a later epoch can dip slightly lower without being saved (its improvement was below `min_delta`), and that model was never tested.
- `final_train_loss`: the last value of `train_loss_history`
- `convergence_epoch`: the same definition as in `comparisons.py`
- `generalization_gap`: `test_metric − train_metric`, when the run records `train_metric`
- for older results files: `stop_reason`, derived from the `diverged` and `stopped_early` flags, and `overfitting_epochs = None` (not recorded)

"Fastest convergence" therefore means the lowest average convergence epoch.

#### Loss normalization

Classification uses BCE and regression uses MSE. These are on different scales, so they can't be averaged together directly. `analysis.py` min-max normalizes the best validation loss **within each problem**:

```
normalized_best_val_loss = (best_val_loss - min_loss_in_problem) / (max_loss_in_problem - min_loss_in_problem)
```

This puts each task's losses on a 0–1 scale before combining them. If every loss within a problem is identical, the script assigns 0.0 to avoid dividing by zero.

#### Grouping

- **Per task** (classification, regression), grouped by optimizer and by architecture: `num_runs`, `avg_convergence_epoch`, `avg_best_epoch`, `avg_best_val_loss`.
- **Combined across tasks**, grouped by optimizer and by architecture: `num_runs`, `avg_convergence_epoch`, `avg_best_epoch`, `avg_normalized_best_val_loss`. Normalized loss is used here because the groups mix classification and regression runs.

The "best" group is the one with the smallest value in the relevant field. For the depth question, the script compares A1 and A2 in the combined architecture summary, on both convergence epoch and normalized loss.

All of the sections above use the baseline architectures (A1, A2) and the baseline optimizers (SGD, momentum, AdaBelief) only, including the min-max normalization. Adding new architectures or optimizers to the configs therefore leaves those numbers unchanged.

#### Muon

The Muon section compares all four optimizers on the same runs. For each task it shows:
- a summary table of the first seed's A1 / A2 runs by optimizer, with diverged runs left out of the averages and counted below the table
- the optimizer × learning-rate table of average test metrics across seeds
- the model selected per seed when every architecture and optimizer, Muon included, is allowed

The rest of the report, including the multi-seed section, uses the baseline optimizers only.

#### A2 variants

A separate section compares A2 with its bias, initialization, batch-norm and gradient-clipping variants:
- one summary table per task (runs, average convergence epoch, average best epoch, average best validation loss)
- a **collapse check**: the regression test MSE of the four runs that collapse in the baseline A2 (SGD and momentum, LR 0.1, batch 16 and 64), for each variant

Diverged runs (see "How a run ends" above) are listed separately, with their peak gradient norm, and left out of the averages. A test MSE of 602.17 means the network outputs a constant 0. One near 101.4 means it outputs a constant close to the mean target, which a dead network can still do through its output bias.

#### Dropout

A final section compares each architecture with its dropout version (A1 vs A1-dropout, A2-bn vs A2-bn-dropout), on runs matched by optimizer, learning rate and batch size; matches where either run diverged are left out. For each task it reports:
- the average training metric (`train_metric`, evaluation mode)
- the average test metric
- the average generalization gap (test − train)
- in how many matched runs dropout gave the lower test metric

The section is skipped for results files without dropout runs or without `train_metric`.

#### Multi-seed results

Every section above uses seed 42 only, so its numbers stay comparable with single-seed results files. When the results contain several seeds, a final **Multi-Seed Results** section reports mean ± sample standard deviation across seeds:

- **Selected model per seed**: for each task and seed, the run with the lowest validation loss (never the test score). It's reported once among the baseline architectures A1 / A2 and once among all architectures, with test metric, constant baseline and error removed.
- **Architectures across seeds**: each architecture's best run per seed (again chosen by validation loss), its average convergence epoch and best epoch, and how many of its runs were no better than the constant baseline or diverged.
- **Stopping and overfitting across seeds**: per architecture, how many runs ended by early stopping, at the epoch limit or by diverging; how many of the early stops showed the overfitting signature; and the generalization gap of the run selected per seed.
- **Optimizer × learning rate** (A1 / A2): the average test metric per seed for every optimizer and learning rate.
- **Collapse check across seeds**: for the regression settings where the baseline A2 collapses, how many runs of each A2 variant learned, were no better than predicting the mean, or diverged.
- **Dropout across seeds**: the dropout comparison pooled over all seeds, with runs matched on seed as well.

Diverged runs are excluded from model selection and averages throughout. The comparison plots also use seed 42 only, because each shows one run's loss curves.

#### Helper functions

- `load_results(path)`: loads the JSON results
- `convergence_epoch(...)`: computes the convergence epoch from the training loss
- `add_derived_metrics(results)`: adds `best_val_loss`, `final_train_loss`, `convergence_epoch` and `generalization_gap`, and fills in `stop_reason` / `overfitting_epochs` for older files
- `add_normalized_best_val_loss(results)`: normalizes the best validation loss within each task
- `filter_by_problem(...)`: separates classification and regression runs
- `group_by_key(...)`: groups runs by optimizer or architecture
- `summarize_task_group(...)`: computes averages within one task
- `summarize_combined_group(...)`: computes averages across all runs using normalized loss
- `best_group(summary_dict, field_name)`: selects the group with the smallest value in the given field

The report (`reports/analysis_<stamp>.txt`) contains a method section, supporting tables per task and combined, direct answers to the three questions, and, when the results include them, the A2 variant section.

### In short

- `comparisons.py` explains individual matched experiments with plots and short text analyses.
- `analysis.py` answers the overall questions using averages across all runs.

---

## 4. Library comparison

The `nn_numpy.benchmarks` subpackage compares the NumPy network with standard libraries: scikit-learn's classic models, and the same network built in TensorFlow (Keras) and in PyTorch. The network itself stays NumPy-only; the libraries are only used here, as an optional install:

```bash
make benchmark-requirements   # pip install -e ".[dev,benchmarks]"
make benchmarks               # python -m nn_numpy.benchmarks.run
make tune                     # python -m nn_numpy.benchmarks.tuning (optional, about 1.5 hours)
make benchmark-report         # python -m nn_numpy.benchmarks.report
```

The `benchmarks` group also installs Optuna 5.0 for the tuning. `scipy` is pinned in the group because newer SciPy releases require NumPy 2, which would replace the project's pinned NumPy 1.26.4. TensorFlow 2.21 works with NumPy 1.26.4 and runs on the CPU on Windows (it no longer supports GPUs on native Windows). PyTorch 2.14 also works with NumPy 1.26.4; on Windows and macOS the PyPI wheel is CPU-only, while on Linux it includes CUDA (a much larger download; `pip install torch --index-url https://download.pytorch.org/whl/cpu` installs the CPU build instead). `make benchmarks` runs every library; `python -m nn_numpy.benchmarks.run pytorch` runs one.

### What is compared

| Library | Classification | Regression |
|---|---|---|
| scikit-learn | logistic regression, SVM (RBF), random forest, gradient boosting (`HistGradientBoostingClassifier`), MLP (`MLPClassifier`) | ridge regression, SVR (RBF), random forest, gradient boosting (`HistGradientBoostingRegressor`), MLP (`MLPRegressor`) |
| TensorFlow (Keras) | A1, A2, A2-bias, A2-bn | A1, A2, A2-bias, A2-bn |
| PyTorch | A1, A2, A2-bias, A2-bn | A1, A2, A2-bias, A2-bn |

Each scikit-learn model has a small hyperparameter grid in `configs/benchmark_experiments.json`. The MLP grid mirrors the NumPy network's A1 (32 sigmoid units) and A2 (3 × 32 ReLU units) with scikit-learn's own Adam optimizer.

#### TensorFlow (Keras)

`benchmarks/keras_models.py` rebuilds A1, A2, A2-bias and A2-bn from the **same architecture definitions** in the main configs: the same units, activations, bias settings and batch-norm placement (Dense → BatchNormalization → Activation). Each is trained over the grid SGD / momentum / Adam / Muon × learning rate 0.1 / 0.001 × batch size 16 / 64, for up to 100 epochs, on every seed.

Everything else is what Keras provides, as a practitioner would use it:

| Aspect | NumPy network | Keras |
|---|---|---|
| Weight initialization | N(0, 0.1²) | Glorot uniform |
| Momentum | v = 0.9·v + 0.1·g (moving average) | v = 0.9·v − lr·g (classical), so about 10× larger steps |
| Adaptive optimizer | AdaBelief | Adam (Keras has no AdaBelief) |
| Muon | Nesterov 0.95, 5 Newton–Schulz steps in float64; step lr · √max(1, fan_out / fan_in); no weight decay; AdaBelief for biases and batch norm | `keras.optimizers.Muon`: the same momentum and Newton–Schulz coefficients; step lr · 0.2 · √max(fan_in, fan_out); weight decay 0.004; AdamW for every variable that isn't 2-D |
| Batch norm | momentum 0.1 (= 0.9 decay), ε = 1e-5 | decay 0.99, ε = 1e-3 |
| Precision | float64 | float32 |
| Early stopping | tracks the best checkpoint from epoch 1; can't stop before epoch 20 | `EarlyStopping(start_from_epoch=20)`: ignores the first 20 epochs entirely, then the same patience 10 / `min_delta` 1e-4, restoring the best weights |
| Divergence | stops at the first non-finite loss | `TerminateOnNaN`; a run also counts as diverged when the restored model's predictions aren't finite (in float32 the loss can stay just below overflow, around 1e34, while the outputs overflow) |

Runs are seeded with `keras.utils.set_random_seed` and TensorFlow's deterministic operations, so a seed always gives the same result. The restored model is scored with the same NumPy metric functions as every other model, and its validation metric is used for selection.

#### PyTorch

`benchmarks/torch_models.py` rebuilds the same four architectures as an `nn.Sequential` (Linear → BatchNorm1d → activation, with `nn.Identity` for a linear output) and trains them over the same grid. PyTorch has no `fit()` method or early-stopping callback, so the training loop is written out, as is usual in PyTorch code: per epoch, shuffle with a seeded `torch.Generator`, loop over mini-batches (`zero_grad` → forward → loss → `backward` → `step`), then compute the validation loss in `eval()` mode under `torch.no_grad()`. Because the loop is our own, early stopping uses **exactly the NumPy trainer's rule**.

| Aspect | NumPy network | PyTorch |
|---|---|---|
| Weight initialization | N(0, 0.1²) | `nn.Linear` default: U(−1/√fan_in, 1/√fan_in) (Kaiming uniform with a = √5) |
| Momentum | v = 0.9·v + 0.1·g (moving average) | v = 0.9·v + g, step lr·v, so about 10× larger steps |
| Adaptive optimizer | AdaBelief | Adam (PyTorch has no AdaBelief) |
| Muon | Nesterov 0.95, 5 Newton–Schulz steps in float64; step lr · √max(1, fan_out / fan_in); no weight decay; AdaBelief for biases and batch norm | `torch.optim.Muon` for the 2-D weights: the same rule and step scaling, Newton–Schulz in bfloat16, weight decay 0.1; `Adam` for the 1-D parameters (biases, batch norm) |
| Batch norm | momentum 0.1, ε = 1e-5 | `BatchNorm1d` default: momentum 0.1, ε = 1e-5 (the same) |
| Precision | float64 | float32 |
| Loss | BCE on clipped probabilities, MSE | `nn.BCELoss` on the sigmoid output (log clamped at −100), `nn.MSELoss` |
| Early stopping | best checkpoint tracked from epoch 1; no stopping before epoch 20; patience 10, `min_delta` 1e-4 | the same rule, restoring the best `state_dict` (weights and batch-norm statistics) |
| Divergence | stops at the first non-finite loss | the same, plus non-finite predictions from the restored model |

Runs are seeded with `torch.manual_seed` (weights) and a seeded generator (shuffling), with `torch.use_deterministic_algorithms(True)`. A seed gives the same result on the same machine; the number of CPU threads PyTorch uses can change the last digits, and through them the epoch early stopping picks, so results on another machine may differ slightly.

### Hyperparameter tuning (Optuna)

The grid above is small and was designed around the NumPy network, so `benchmarks/tuning.py` also tunes the Keras and PyTorch models with [Optuna](https://optuna.org/). This tests whether tuned frameworks change the picture. The NumPy network itself is not tuned beyond its grid.

| Setting | Value (`configs/tuning_experiments.json`) |
|---|---|
| Architectures | A1, A2, A2-bias, A2-bn, per task |
| Search space | optimizer ∈ {SGD, momentum, Adam, Muon}; learning rate 1e-4–0.3 (log scale); batch size ∈ {8, 16, 32, 64, 128}; weight decay 1e-6–1e-2 (log scale); dropout 0–0.3 after every hidden layer |
| Sampler | TPE, seeded (`sampler_seed` 0), so the search is reproducible |
| Budget | 50 trials per framework, task and architecture |
| Search data | the validation set of seed 42's split only |
| Training | each framework's settings in `benchmark_experiments.json` (100 epochs, the same early stopping) |

For each framework, task and architecture:
1. Optuna draws 50 configurations. Each is trained on seed 42's training set and scored on its validation set. A diverged run scores infinity, so it is never chosen.
2. The configuration with the lowest validation metric is trained and tested once on every seed's split, exactly like a grid configuration.

The test set never guides the search, so the tuned test scores are as unbiased as the grid's. Seed 42's validation set does guide it, so on that seed the tuned models had a mild advantage in selection. The other four seeds are fully out of sample.

Weight decay uses each library's own form: decoupled in Keras's optimizers and in both libraries' Muon, and an L2 penalty in PyTorch's SGD and Adam. Without tuning, the libraries' defaults apply (no weight decay, except Muon's).

```bash
make tune                     # python -m nn_numpy.benchmarks.tuning [tensorflow|pytorch]
```

The tuned models are saved like another library (`reports/benchmark_<library>_tuned_results_<stamp>.json`), so the comparison report lists them next to the grid-selected ones. Per framework and task, it adds two tables. One compares the grid and tuned versions of each architecture, with the tuned configuration. The other is a three-way table: per architecture, the test metric of the NumPy network, the framework's grid model and its tuned model, and on how many seeds the NumPy network beat each. Seeds are paired: only seeds with a selected run on both sides are compared, and a tie isn't a win. The learning-curve figures add the tuned model as a dashed line. Every trial is kept in `reports/tuning_<library>_trials_<stamp>.json`.

### How the comparison is kept fair

- **Same data.** `benchmarks/data.py` calls the main pipeline's own preprocessing for each seed in the main configs, so every library sees exactly the same train / validation / test split and scaling. A test checks this element by element.
- **Same metrics.** Every model is scored with the project's NumPy metric functions (`nn_numpy.nn.metrics`): BCE (from predicted probabilities) and accuracy for classification, MSE and R² for regression.
- **Same selection rule.** On each seed, every configuration of a model is trained on the training set, the one with the lowest validation metric is selected, and only its test metric is reported. "Best model (selected on validation)" applies the same rule across all of a library's models, which is what a practitioner would pick.
- **Reproducible.** Every scikit-learn estimator with a `random_state` gets the seed, and Keras and PyTorch are seeded and run deterministically.

Training time is the wall-clock time of the selected configuration's fit, on one CPU run. It is only a rough comparison: the NumPy network trains for up to 100 epochs with early stopping, while scikit-learn's models stop on their own criteria.

### Outputs

- `reports/benchmark_<library>_results_<stamp>.json`: one record per configuration, problem and seed (validation, test and training metrics, secondary metric, training time; for scikit-learn whether training converged; for the frameworks the epochs run, best epoch, whether training diverged and the loss histories).
- `reports/benchmark_report_<stamp>.txt`: per task, the test metric, secondary metric, error removed and training time (mean ± std over seeds) for the NumPy network and every library model, the configuration chosen most often, a seed-by-seed head-to-head between the NumPy network (all architectures) and each library's best model, and, per framework, an architecture-by-architecture table (NumPy vs. framework version of A1, A2, A2-bias, A2-bn, with diverged-run counts).
- `reports/benchmark_<library>_tuned_results_<stamp>.json` and `reports/tuning_<library>_trials_<stamp>.json`: the tuned framework models on every seed, and every tuning trial (see [Hyperparameter tuning](#hyperparameter-tuning-optuna)). With them, the report also lists the tuned models, compares grid and tuned versions architecture by architecture, and adds the three-way NumPy / grid / tuned table.
- `reports/figures/benchmarks/benchmark_<task>_<stamp>.png`: the test metric per model as a bar chart (log scale).
- `reports/figures/benchmarks/benchmark_curves_<framework>_<stamp>.png`: the validation-loss curves of the NumPy network's and the framework's selected models on the first seed, for both tasks, plus the framework's tuned model (dashed) when tuned results exist.

---

## 5. Knowledge base

`knowledge/nn_numpy.owl` describes the repository as an OWL 2 ontology: the nodes and edges of the architecture canvas, plus the concepts the code implements. It opens in Protégé 5.5, and its reasoner infers the architecture from the asserted facts.

### What it contains

- **Structure** (from the canvas): every module, package, artifact and make target is an individual of `Component`, with its path, a description taken from the module's docstring, and its stage. Edges: `reads` / `writes` (module ↔ artifact), `imports` (read from the sources with `ast`, lazy imports included), `partOf` (module → package), `runs` (make target → module), `covers` (test module → the modules it imports), `requiresLibrary`, `implements`.
- **Domain** (from the configs): the two tasks with their datasets, losses, metrics and output activations; the nine architectures with their hidden-layer count, units, activation, techniques (bias terms, initialization, batch norm, dropout, gradient clipping) and the architecture each is a variant of; the sweep (optimizers, learning rates, batch sizes, seeds, early stopping, run count); the libraries from `pyproject.toml` with their versions, split into core, optional (benchmarks) and development; the scikit-learn, Keras and PyTorch models of the comparison.
- **Results**: the result files are individuals with a `filePattern`, `writtenBy` the module that produces them and a description of what they contain. No result numbers are stored.

Every object property has an inverse. Where they fit, properties are transitive (`upstreamOf`, `partOf`, `variantOf`), functional or inverse functional (`writtenBy`, `runs`, `implementedIn`, `hasEarlyStopping`, `usesHiddenActivation`, `usesOutputActivation`, `onDataset`, `usesLoss`, `providedBy`, `mirrorsArchitecture`), symmetric (`frameworkCounterpartOf`, between the Keras and PyTorch rebuilds of one architecture), asymmetric (every simple directed property, so an edge entered backwards is reported as an inconsistency) or irreflexive (`imports`, `frameworkCounterpartOf`). `feeds` and `belongsToStage` are defined by property chains, and OWL 2 DL forbids functional, asymmetric and irreflexive characteristics on such properties (and on transitive ones), so they have none.

Nothing is asserted that a reasoner can derive. `feeds` is inferred from `writes ∘ readBy` (a module feeds the modules that read what it writes), and `upstreamOf` is its transitive closure; a module's stage is inferred from its package through `partOf ∘ belongsToStage`; and the defined classes classify individuals by their edges: `Producer`, `Consumer`, `PipelineStep`, `EntryPoint`, `OptionalModule`, `TestedModule`, `StampedArtifact`, one `…StageComponent` class per stage, `Variant`, `DeepArchitecture` / `ShallowArchitecture`, `NormalizedArchitecture`, `RegularizedArchitecture`, `ClippedArchitecture`, `BiasedArchitecture` and `FrameworkReimplementation`.

### Using it in Protégé

1. Open `knowledge/nn_numpy.owl` (or `extensions.owl`, which imports it from the same folder).
2. Reasoner → HermiT → Start reasoner. The inferred class hierarchy and each individual's inferred types appear in the usual views.
3. To see the inferred edges (`feeds`, `upstreamOf`, `belongsToStage`) on an individual, tick **Object property assertions** under Reasoner → Configure.
4. DL Query examples: `Component and upstreamOf value BenchmarkReport` (everything behind the benchmark report), `Module and belongsToStage value NetworkLibraryStage`, `Architecture and usesTechnique some GradientClipping`, `LibraryModel and not FrameworkReimplementation`.

The generated file is never edited by hand: put manual classes, individuals or axioms in `knowledge/extensions.owl`, which regeneration doesn't touch. Protégé writes a `catalog-v001.xml` next to the files; it is git-ignored.

### Regenerating and testing

`make knowledge-base` (or `python tools/build_knowledge_base.py`) rebuilds the file from the current repository with `rdflib` (a dev dependency). The output is deterministic, so regenerating without changes produces no diff. `tests/test_knowledge_base.py` checks that the file reads back to the same graph, that no IRI has two kinds and nothing references an undeclared name, that every assertion respects its property's domain and range, and, with the pure-Python OWL RL reasoner `owlrl`, that the intended inferences hold (pipeline order, stage membership, roles, architecture families). The two classes defined by a numeric range, `DeepArchitecture` and `ShallowArchitecture`, need a reasoner with datatype facets such as HermiT, so they are only verified in Protégé.
