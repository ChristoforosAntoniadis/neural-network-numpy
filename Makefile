#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = neural-network-numpy
PACKAGE = nn_numpy

# Use the project's virtual environment when it exists, otherwise the Python on PATH.
ifeq ($(OS),Windows_NT)
VENV_PYTHON = .venv/Scripts/python.exe
else
VENV_PYTHON = .venv/bin/python
endif
PYTHON_INTERPRETER = $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create the virtual environment in .venv
.PHONY: create_environment
create_environment:
	python -m venv .venv
	@echo Created .venv. Run make requirements next; the Makefile uses .venv automatically.

## Install the pinned runtime dependencies, the package itself, and the dev tools
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e ".[dev]"

## Download the raw datasets into data/raw/ (skipped if already present)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).dataset

## Run the full experiment sweep (all configs and seeds)
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).modeling.train

## Create the comparison figures from the newest results (no plot windows)
.PHONY: plots
plots: export MPLBACKEND = Agg
plots:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).comparisons

## Write the analysis report from the newest results
.PHONY: analysis
analysis:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).analysis

## Run the whole pipeline: train, then plots, then analysis
.PHONY: all
all: train plots analysis

## Install the libraries the network is compared against (scikit-learn, TensorFlow, PyTorch)
.PHONY: benchmark-requirements
benchmark-requirements:
	$(PYTHON_INTERPRETER) -m pip install -e ".[dev,benchmarks]"

## Train the library models on the same splits as the network
.PHONY: benchmarks
benchmarks:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).benchmarks.run

## Tune the TensorFlow and PyTorch models with Optuna (search on validation, test on every seed)
.PHONY: tune
tune:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).benchmarks.tuning

## Write the comparison report and figures (network vs. libraries)
.PHONY: benchmark-report
benchmark-report: export MPLBACKEND = Agg
benchmark-report:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).benchmarks.report

## Build the OWL knowledge base in knowledge/ from the configs, sources and Makefile
.PHONY: knowledge-base
knowledge-base:
	$(PYTHON_INTERPRETER) tools/build_knowledge_base.py

## Run the test suite
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest

## Check code style and lint with ruff (reports only, changes nothing)
.PHONY: lint
lint:
	$(PYTHON_INTERPRETER) -m ruff format --check
	$(PYTHON_INTERPRETER) -m ruff check

## Check the type hints with mypy (reports only, changes nothing)
.PHONY: typecheck
typecheck:
	$(PYTHON_INTERPRETER) -m mypy

## Fix lint issues and format the code with ruff (rewrites files)
.PHONY: format
format:
	$(PYTHON_INTERPRETER) -m ruff check --fix
	$(PYTHON_INTERPRETER) -m ruff format

## Delete Python caches and compiled files
.PHONY: clean
clean:
	$(PYTHON_INTERPRETER) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts]; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache', '.ruff_cache', '.mypy_cache')]"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

## Show this list of commands
.PHONY: help
help:
	@$(PYTHON_INTERPRETER) tools/make_help.py Makefile
