"""
Build the repository's knowledge base: an OWL 2 ontology for Protégé.

Input: the experiment configs (``configs/*.json``), the Python sources of
``nn_numpy/``, ``tools/`` and ``tests/`` (modules and their imports,
read with ``ast``), the ``Makefile`` (targets and the modules they run),
``pyproject.toml`` (libraries and versions), the raw datasets (row counts)
and the architecture canvas (node ids).
Output: ``knowledge/nn_numpy.owl`` in RDF/XML.

The ontology describes the repository as nodes and edges: modules,
packages, artifacts and make targets are individuals, and what each one
reads, writes, imports or belongs to is asserted. Pipeline order, stage
membership and the roles of modules and architectures are left for a
reasoner to infer (see the defined classes and property chains in
``add_schema``). Run it with ``python tools/build_knowledge_base.py`` or
``make knowledge-base``. Open the result in Protégé and start the reasoner.

The generated file is not meant to be edited by hand: put manual additions
in ``knowledge/extensions.owl``, which imports it.
"""

import ast
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from rdflib import OWL, RDF, RDFS, XSD, BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.term import Node

from nn_numpy.config_loader import ConfigLoader

# Repository root and the files the knowledge base is built from.
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "knowledge" / "nn_numpy.owl"
EXTENSIONS = ROOT / "knowledge" / "extensions.owl"
CANVAS = ROOT / "knowledge" / "nn-architecture.canvas"

# Ontology IRIs. The namespace of every entity is the ontology IRI plus "#".
ONTOLOGY_IRI = "https://github.com/WarmupHero/neural-network-numpy/ontology"
EXTENSIONS_IRI = ONTOLOGY_IRI + "/extensions"
NNFS = Namespace(ONTOLOGY_IRI + "#")

# Which pip package provides which import names.
IMPORT_NAMES = {
    "scikit-learn": ["sklearn"],
    "tensorflow": ["tensorflow", "keras"],
    "torch": ["torch"],
}

# Optimizer names in the configs -> ontology individuals.
OPTIMIZER_NAMES = {
    "sgd": "SGD",
    "momentum": "MomentumSGD",
    "adabelief": "AdaBelief",
    "muon": "Muon",
}

# Stage of every root-level module, artifact and make target; modules in
# sub-packages inherit their package's stage through a property chain.
STAGES = {
    "InputsStage": "Inputs: the experiment configs and the raw datasets.",
    "PreprocessingStage": "Load and preprocess: config validation, cleaning, splitting, scaling, EDA plots.",
    "TrainingStage": "Training: the experiment sweep and the training loop of the NumPy network.",
    "NetworkLibraryStage": "The network library (nn/): layers, activations, losses, optimizers, metrics.",
    "ResultsStage": "Results: the files the training sweep writes.",
    "AnalysisStage": "Analysis and reports: comparison plots and the aggregate analysis.",
    "BenchmarkStage": "Library comparison: the same splits trained with scikit-learn, TensorFlow and PyTorch.",
    "InfrastructureStage": "Infrastructure: configuration helpers, Makefile, git hook, tests, docs.",
}
MODULE_STAGES = {
    "nn_numpy.config": "InfrastructureStage",
    "nn_numpy.config_loader": "PreprocessingStage",
    "nn_numpy.dataset": "InputsStage",
    "nn_numpy.features": "PreprocessingStage",
    "nn_numpy.scalers": "PreprocessingStage",
    "nn_numpy.plots": "PreprocessingStage",
    "nn_numpy.comparisons": "AnalysisStage",
    "nn_numpy.analysis": "AnalysisStage",
}
PACKAGES = {
    "nn_numpy": (
        None,
        "The source package: the NumPy network and everything around it.",
    ),
    "nn_numpy.nn": ("NetworkLibraryStage", "The NumPy-only network library."),
    "nn_numpy.modeling": ("TrainingStage", "The training loop and the experiment sweep."),
    "nn_numpy.benchmarks": (
        "BenchmarkStage",
        "The comparison with scikit-learn, TensorFlow and PyTorch.",
    ),
    "tools": (
        "InfrastructureStage",
        "Helper scripts: Makefile help, the pre-commit hook's staging, this builder.",
    ),
    "tests": ("InfrastructureStage", "The pytest suite."),
}

# Artifacts: local name -> (class, path or pattern, stamped, stage, description).
ARTIFACTS = {
    "ClassificationConfig": (
        "ConfigFile",
        "configs/classification_experiments.json",
        False,
        "InputsStage",
        "Experiment definition for the banknote classification task: architectures, optimizers, learning rates, batch sizes, seeds, epochs, early stopping and preprocessing.",
    ),
    "RegressionConfig": (
        "ConfigFile",
        "configs/regression_experiments.json",
        False,
        "InputsStage",
        "Experiment definition for the heating-load regression task, with the same grid as the classification config.",
    ),
    "BenchmarkConfig": (
        "ConfigFile",
        "configs/benchmark_experiments.json",
        False,
        "InputsStage",
        "The library models compared with the network: scikit-learn hyperparameter grids per task, and the architectures, grid and training settings for TensorFlow and PyTorch.",
    ),
    "TuningConfig": (
        "ConfigFile",
        "configs/tuning_experiments.json",
        False,
        "InputsStage",
        "Hyperparameter tuning of the TensorFlow and PyTorch models with Optuna: the libraries and architectures, the tuning seed, the TPE sampler's seed, the number of trials and the search space (optimizer, learning rate, batch size, weight decay, dropout).",
    ),
    "BanknoteDataset": (
        "Dataset",
        "data/raw/banknote_auth.csv",
        False,
        "InputsStage",
        "UCI Banknote Authentication: wavelet features of banknote images with a genuine / forged label. Committed to the repository; dataset.py downloads it only if missing.",
    ),
    "EnergyDataset": (
        "Dataset",
        "data/raw/energy_efficiency.csv",
        False,
        "InputsStage",
        "UCI Energy Efficiency: eight building-shape features and the simulated heating load used as the regression target. Committed to the repository; dataset.py downloads it only if missing.",
    ),
    "EDAFigures": (
        "Figure",
        "reports/figures/eda/<name>_<stamp>.png",
        True,
        "PreprocessingStage",
        "Exploratory plots of the first seed's training split: class balance, pairplot, correlation heatmap, target distribution and the scaling comparison. Written by plots.py during make train.",
    ),
    "MainResultsJSON": (
        "ResultsFile",
        "reports/main_results_full_<stamp>.json",
        True,
        "ResultsStage",
        "The complete results of the experiment sweep: one record per training run with its settings, test / train / baseline metrics, accuracy or R², best epoch, stop reason, overfitting signature, training time and the per-epoch loss and gradient-norm histories. Written by modeling/train.py (make train); read by comparisons.py, analysis.py and benchmarks/report.py. The newest file is used unless a path is given.",
    ),
    "SummaryCSV": (
        "ResultsFile",
        "reports/main_summary_<stamp>.csv",
        True,
        "ResultsStage",
        "A compact table of the sweep: problem, seed, optimizer, batch size, learning rate, architecture, epochs ran, best epoch, stop reason and test metric, one row per run. Written by modeling/train.py next to the full results JSON.",
    ),
    "ComparisonFigures": (
        "Figure",
        "reports/figures/comparisons/<name>_<stamp>.png",
        True,
        "AnalysisStage",
        "Loss-curve figures of matched runs on seed 42: optimizers, network depth, learning rate, the A2 variants and dropout, each marking the final epoch and the validation-selected best epoch. Written by comparisons.py (make plots) from the newest results JSON.",
    ),
    "ComparisonTexts": (
        "Report",
        "reports/comparisons/<name>_<stamp>.txt",
        True,
        "AnalysisStage",
        "Short written analyses next to the depth and learning-rate figures, saying which setting converged faster and ended lower. Written by comparisons.py.",
    ),
    "AnalysisReport": (
        "Report",
        "reports/analysis_<stamp>.txt",
        True,
        "AnalysisStage",
        "The aggregate analysis: convergence epoch next to best epoch per optimizer and architecture, the answers to the three study questions, the A2 variants and collapse checks, dropout and the generalization gap, and the multi-seed tables (selected model per seed, stopping and overfitting). Written by analysis.py (make analysis) from the newest results JSON.",
    ),
    "BenchmarkResultsJSON": (
        "ResultsFile",
        "reports/benchmark_<library>_results_<stamp>.json",
        True,
        "BenchmarkStage",
        "One file per library (sklearn, tensorflow, pytorch) with one record per model configuration, task and seed: validation, test and training metrics, training time and, for the frameworks, the epochs run and loss histories. Written by benchmarks/run.py (make benchmarks) on exactly the network's splits.",
    ),
    "TunedResultsJSON": (
        "ResultsFile",
        "reports/benchmark_<library>_tuned_results_<stamp>.json",
        True,
        "BenchmarkStage",
        "One file per framework (tensorflow_tuned, pytorch_tuned) in the benchmark results' record format: one record per architecture, task and seed, all using the configuration Optuna tuned on the tuning seed's validation set. Written by benchmarks/tuning.py (make tune); read by benchmarks/report.py as two more libraries.",
    ),
    "TuningTrialsJSON": (
        "ResultsFile",
        "reports/tuning_<library>_trials_<stamp>.json",
        True,
        "BenchmarkStage",
        "Every Optuna trial of one framework: the hyperparameters drawn, and the validation, test and training metrics on the tuning seed's split. Written by benchmarks/tuning.py (make tune).",
    ),
    "BenchmarkReport": (
        "Report",
        "reports/benchmark_report_<stamp>.txt",
        True,
        "BenchmarkStage",
        "The comparison of the network with the library models: per task, the test metric, secondary metric, error removed and fit time of every model (mean ± std over seeds), the configuration selected most often, seed-by-seed head-to-heads and the same-architecture tables against Keras and PyTorch. Written by benchmarks/report.py (make benchmark-report) from the newest NumPy and library results.",
    ),
    "BenchmarkFigures": (
        "Figure",
        "reports/figures/benchmarks/<name>_<stamp>.png",
        True,
        "BenchmarkStage",
        "Bar charts of the test metric per model and the validation-loss curves of the selected NumPy and framework models. Written by benchmarks/report.py.",
    ),
    "Readme": (
        "Document",
        "README.md",
        False,
        "InfrastructureStage",
        "The project overview: highlights, results, key findings, the library comparison and its limitations, quickstart and the project structure.",
    ),
    "Details": (
        "Document",
        "docs/DETAILS.md",
        False,
        "InfrastructureStage",
        "The detailed documentation: pipeline map, configuration reference, analysis details, the library comparison methodology and the knowledge base.",
    ),
    "Makefile": (
        "Document",
        "Makefile",
        False,
        "InfrastructureStage",
        "The command entry points: train, plots, analysis, benchmarks, tests, lint, format and the knowledge base. Each target runs one module.",
    ),
    "PreCommitHook": (
        "Document",
        ".githooks/pre-commit",
        False,
        "InfrastructureStage",
        "Git pre-commit hook that runs tools/stage_latest_outputs.py so that only the newest stamped output of each kind is committed. Enabled per clone with git config core.hooksPath .githooks.",
    ),
    "PyProject": (
        "ConfigFile",
        "pyproject.toml",
        False,
        "InfrastructureStage",
        "Package metadata, the runtime dependencies, the dev and benchmarks extras, and the pytest and ruff settings.",
    ),
    "ArchitectureCanvas": (
        "Document",
        "knowledge/nn-architecture.canvas",
        False,
        "InfrastructureStage",
        "The system architecture as an Obsidian canvas: one node per component, grouped by stage, with arrows for what feeds what. The knowledge base annotates its individuals with these node ids.",
    ),
    "KnowledgeBase": (
        "Document",
        "knowledge/nn_numpy.owl",
        False,
        "InfrastructureStage",
        "This ontology, generated by tools/build_knowledge_base.py from the configs, sources, Makefile and pyproject. Open it in Protégé and run the reasoner to see the inferred pipeline and classifications.",
    ),
}

# Data flow between modules and artifacts: module -> (reads, writes).
DATA_FLOW = {
    "nn_numpy.config_loader": (["ClassificationConfig", "RegressionConfig"], []),
    "nn_numpy.dataset": ([], ["BanknoteDataset", "EnergyDataset"]),
    "nn_numpy.features": (["BanknoteDataset", "EnergyDataset"], []),
    "nn_numpy.plots": ([], ["EDAFigures"]),
    "nn_numpy.modeling.train": ([], ["MainResultsJSON", "SummaryCSV"]),
    "nn_numpy.comparisons": (["MainResultsJSON"], ["ComparisonFigures", "ComparisonTexts"]),
    "nn_numpy.analysis": (["MainResultsJSON"], ["AnalysisReport"]),
    "nn_numpy.benchmarks.run": (["BenchmarkConfig"], ["BenchmarkResultsJSON"]),
    "nn_numpy.benchmarks.tuning": (
        ["TuningConfig", "BenchmarkConfig"],
        ["TunedResultsJSON", "TuningTrialsJSON"],
    ),
    "nn_numpy.benchmarks.report": (
        ["MainResultsJSON", "BenchmarkResultsJSON", "TunedResultsJSON"],
        ["BenchmarkReport", "BenchmarkFigures"],
    ),
    "tools.make_help": (["Makefile"], []),
    "tools.stage_latest_outputs": (["Readme", "Details"], ["Readme", "Details"]),
    "tools.build_knowledge_base": (
        [
            "ClassificationConfig",
            "RegressionConfig",
            "BenchmarkConfig",
            "Makefile",
            "PyProject",
            "ArchitectureCanvas",
        ],
        ["KnowledgeBase"],
    ),
}

# What each module implements, by local name of the domain individual.
IMPLEMENTS = {
    "nn_numpy.nn.optimizers": ["SGD", "MomentumSGD", "AdaBelief", "Muon"],
    "nn_numpy.nn.layers": [
        "DenseLayer",
        "BatchNormLayer",
        "DropoutLayer",
        "BiasTerms",
        "BatchNormalization",
        "Dropout",
        "HeInitialization",
        "XavierInitialization",
        "NormalInitialization",
    ],
    "nn_numpy.nn.activations": ["ReLU", "Sigmoid", "Tanh", "Linear"],
    "nn_numpy.nn.losses": ["BCELoss", "MSELoss"],
    "nn_numpy.nn.metrics": [
        "BinaryCrossEntropyMetric",
        "MeanSquaredErrorMetric",
        "Accuracy",
        "R2Score",
    ],
    "nn_numpy.nn.network": ["GlobalNormClipping"],
    "nn_numpy.modeling.trainer": ["EarlyStopping"],
}

# Canvas node id of each individual (only those that have a node).
CANVAS_NODES = {
    "ClassificationConfig": "configs",
    "RegressionConfig": "configs",
    "BenchmarkConfig": "configs",
    "BanknoteDataset": "raw",
    "EnergyDataset": "raw",
    "nn_numpy.config_loader": "loader",
    "nn_numpy.features": "features",
    "nn_numpy.scalers": "features",
    "nn_numpy.plots": "plots",
    "nn_numpy.modeling.train": "train",
    "nn_numpy.modeling.trainer": "trainer",
    "nn_numpy.nn.network": "network",
    "nn_numpy.nn.layers": "layers",
    "nn_numpy.nn.activations": "actloss",
    "nn_numpy.nn.losses": "actloss",
    "nn_numpy.nn.optimizers": "optim",
    "nn_numpy.nn.metrics": "metrics",
    "MainResultsJSON": "results",
    "SummaryCSV": "summary",
    "nn_numpy.comparisons": "comparisons",
    "nn_numpy.analysis": "analysis",
    "nn_numpy.benchmarks.data": "bdata",
    "nn_numpy.benchmarks.sklearn_models": "sk",
    "nn_numpy.benchmarks.keras_models": "keras",
    "nn_numpy.benchmarks.torch_models": "torch",
    "nn_numpy.benchmarks.run": "brun",
    "BenchmarkResultsJSON": "bresults",
    "nn_numpy.benchmarks.report": "breport",
    "nn_numpy.config": "config",
    "Makefile": "make",
    "PreCommitHook": "hook",
    "tools.stage_latest_outputs": "hook",
    "tests": "tests",
    "Readme": "docs",
    "Details": "docs",
}

# Domain individuals that don't come from the configs: local name -> (class, label, description).
DOMAIN = {
    "BiasTerms": (
        "BiasTerm",
        "bias terms",
        "A trainable bias vector per Dense layer (X·W + b), initialized at zero.",
    ),
    "BatchNormalization": (
        "Normalization",
        "batch normalization",
        "Normalizes each layer's pre-activations over the batch with learned scale and shift, keeping running statistics for evaluation mode.",
    ),
    "Dropout": (
        "Regularization",
        "dropout",
        "Inverted dropout after a hidden layer's activation: drops a fraction of the values in training mode and rescales the rest; the identity in evaluation mode.",
    ),
    "HeInitialization": (
        "Initialization",
        "He initialization",
        "Weights drawn from N(0, 2 / fan_in), designed for ReLU.",
    ),
    "XavierInitialization": (
        "Initialization",
        "Xavier initialization",
        "Weights drawn from N(0, 2 / (fan_in + fan_out)), designed for sigmoid and tanh.",
    ),
    "NormalInitialization": (
        "Initialization",
        "fixed-scale normal initialization",
        "Weights drawn from N(0, 0.1²), the project's default.",
    ),
    "GlobalNormClipping": (
        "GradientClipping",
        "global-norm gradient clipping",
        "After each backward pass, if the L2 norm of all gradients exceeds the cap, every gradient is scaled by cap / norm before the optimizer step.",
    ),
    "DenseLayer": (
        "LayerType",
        "Dense",
        "Fully connected layer Z = X·W (+ b) with its backward pass.",
    ),
    "BatchNormLayer": (
        "LayerType",
        "BatchNorm",
        "Batch-normalization layer with gamma, beta and running statistics.",
    ),
    "DropoutLayer": (
        "LayerType",
        "Dropout",
        "Inverted dropout layer with its own seeded random generator.",
    ),
    "ReLU": ("Activation", "ReLU", "max(0, x)."),
    "Sigmoid": (
        "Activation",
        "sigmoid",
        "1 / (1 + exp(-x)); the output activation for classification.",
    ),
    "Tanh": ("Activation", "tanh", "Hyperbolic tangent."),
    "Linear": ("Activation", "linear", "The identity; the output activation for regression."),
    "BCELoss": (
        "Loss",
        "binary cross-entropy loss",
        "The classification training loss, on clipped probabilities.",
    ),
    "MSELoss": ("Loss", "mean squared error loss", "The regression training loss."),
    "BinaryCrossEntropyMetric": (
        "Metric",
        "BCE (metric)",
        "Binary cross-entropy used as the classification evaluation metric.",
    ),
    "MeanSquaredErrorMetric": (
        "Metric",
        "MSE (metric)",
        "Mean squared error used as the regression evaluation metric.",
    ),
    "Accuracy": ("Metric", "accuracy", "Share of correct predictions at a 0.5 threshold."),
    "R2Score": ("Metric", "R²", "Coefficient of determination."),
    "SGD": (
        "Optimizer",
        "SGD",
        "Plain stochastic gradient descent: parameter -= learning rate × gradient.",
    ),
    "MomentumSGD": (
        "Optimizer",
        "SGD with momentum",
        "Keeps a moving average of the gradients (v = 0.9 v + 0.1 g) and steps along it.",
    ),
    "AdaBelief": (
        "Optimizer",
        "AdaBelief",
        "Adaptive optimizer that scales steps by the variance of the gradient from its moving average, with bias correction.",
    ),
    "Muon": (
        "Optimizer",
        "Muon",
        "Orthogonalizes the Nesterov momentum of each weight matrix with Newton-Schulz iterations, so every direction gets a similar step; biases and batch-norm parameters use AdaBelief.",
    ),
    "ClassificationTask": (
        "Task",
        "binary classification",
        "Detect forged banknotes from four wavelet features.",
    ),
    "RegressionTask": (
        "Task",
        "regression",
        "Predict a building's heating load from eight shape features.",
    ),
}

TASK_LINKS = {
    "ClassificationTask": (
        "BanknoteDataset",
        "BCELoss",
        ["BinaryCrossEntropyMetric", "Accuracy"],
        "ClassificationConfig",
    ),
    "RegressionTask": (
        "EnergyDataset",
        "MSELoss",
        ["MeanSquaredErrorMetric", "R2Score"],
        "RegressionConfig",
    ),
}


class Builder:
    """
    Collects the ontology in an rdflib graph with deterministic blank nodes.

    Parameters
    ----------
    None
        Creates an empty graph with the ontology's namespace bound.

    Notes
    -----
    Processing:
    1. Bind the prefixes used in the output file.
    2. Number blank nodes from a counter, so two builds of the same
       repository state produce byte-identical files.
    """

    def __init__(self) -> None:
        """
        Create the empty graph and bind its prefixes.

        Returns
        -------
        None
            Sets self.graph (an empty rdflib.Graph) and the blank-node
            counter self._blank = 0.

        Notes
        -----
        Processing:
        1. Create the graph and bind the nnfs, owl and xsd prefixes.
        2. Start the blank-node counter at 0, so the first node is b0001.
        """
        self.graph = Graph()
        self.graph.bind("nnfs", NNFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)
        self._blank = 0

    def bnode(self) -> BNode:
        """
        Create the next numbered blank node.

        Returns
        -------
        rdflib.BNode
            A blank node named b0001, b0002, ...

        Notes
        -----
        Processing:
        1. Increment the counter and build the node name from it.
        """
        self._blank += 1
        # Four digits keep alphabetical order equal to creation order (axioms are sorted
        # by str) for up to 9,999 nodes; a build creates 88.
        return BNode(f"b{self._blank:04d}")

    def entity(self, name: str, kind: URIRef, label: str, comment: str | None = None) -> URIRef:
        """
        Declare a class, property or individual with a label and comment.

        Parameters
        ----------
        name : str
            Local name, appended to the ontology namespace.
        kind : rdflib.URIRef
            owl:Class, owl:ObjectProperty, owl:DatatypeProperty,
            owl:AnnotationProperty or owl:NamedIndividual.
        label : str
            rdfs:label.
        comment : str or None, default=None
            rdfs:comment, if any.

        Returns
        -------
        rdflib.URIRef
            The entity's IRI.

        Notes
        -----
        Processing:
        1. Add the type, label and comment triples.
        """
        iri = NNFS[name]
        self.graph.add((iri, RDF.type, kind))
        self.graph.add((iri, RDFS.label, Literal(label)))
        if comment:
            self.graph.add((iri, RDFS.comment, Literal(comment)))
        return iri

    def cls(self, name: str, parent: str | None = None, comment: str | None = None) -> URIRef:
        """
        Declare a class, optionally under a parent class.

        Parameters
        ----------
        name : str
            Local name, also used as the label.
        parent : str or None, default=None
            Local name of the superclass.
        comment : str or None, default=None
            rdfs:comment.

        Returns
        -------
        rdflib.URIRef
            The class IRI.

        Notes
        -----
        Processing:
        1. Declare the class with entity().
        2. Add rdfs:subClassOf if a parent is given.
        """
        iri = self.entity(name, OWL.Class, name, comment)
        if parent:
            self.graph.add((iri, RDFS.subClassOf, NNFS[parent]))
        return iri

    def object_property(
        self,
        name: str,
        domain: str | None,
        range_: str | None,
        comment: str,
        inverse: str | None = None,
        transitive: bool = False,
        parents: list[str] | None = None,
        functional: bool = False,
        inverse_functional: bool = False,
        asymmetric: bool = False,
        irreflexive: bool = False,
        symmetric: bool = False,
    ) -> URIRef:
        """
        Declare an object property.

        Parameters
        ----------
        name : str
            Local name.
        domain, range_ : str or None
            Local names of the domain and range classes (None: unrestricted).
        comment : str
            rdfs:comment.
        inverse : str or None, default=None
            Local name of the inverse property, declared in the same call
            with the domain and range swapped.
        transitive : bool, default=False
            Whether to declare the property transitive.
        parents : list of str or None, default=None
            Local names of super-properties.
        functional : bool, default=False
            Whether each subject has at most one value (the inverse is then
            inverse functional).
        inverse_functional : bool, default=False
            Whether each value belongs to at most one subject (the inverse
            is then functional).
        asymmetric : bool, default=False
            Whether the property can never hold in both directions between
            two individuals (implies irreflexive). The inverse is
            asymmetric too.
        irreflexive : bool, default=False
            Whether no individual is related to itself. The inverse is
            irreflexive too.
        symmetric : bool, default=False
            Whether the property always holds in both directions; such a
            property is its own inverse, so no inverse is declared.

        Returns
        -------
        rdflib.URIRef
            The property IRI.

        Notes
        -----
        Processing:
        1. Declare the property, its domain, range, super-properties and
           characteristics.
        2. Declare the inverse the same way, link the two, and give it the
           mirrored characteristics: transitive stays transitive,
           functional becomes inverse functional and vice versa.

        OWL 2 DL only allows functional, inverse-functional, asymmetric and
        irreflexive characteristics on simple properties: not on transitive
        ones, nor on properties defined by a chain or with such
        sub-properties.
        """
        iri = self.entity(name, OWL.ObjectProperty, name, comment)
        if domain:
            self.graph.add((iri, RDFS.domain, NNFS[domain]))
        if range_:
            self.graph.add((iri, RDFS.range, NNFS[range_]))
        if transitive:
            self.graph.add((iri, RDF.type, OWL.TransitiveProperty))
        if functional:
            self.graph.add((iri, RDF.type, OWL.FunctionalProperty))
        if inverse_functional:
            self.graph.add((iri, RDF.type, OWL.InverseFunctionalProperty))
        if asymmetric:
            self.graph.add((iri, RDF.type, OWL.AsymmetricProperty))
        if irreflexive:
            self.graph.add((iri, RDF.type, OWL.IrreflexiveProperty))
        if symmetric:
            self.graph.add((iri, RDF.type, OWL.SymmetricProperty))
        for parent in parents or []:
            self.graph.add((iri, RDFS.subPropertyOf, NNFS[parent]))
        if inverse:
            inv = self.entity(inverse, OWL.ObjectProperty, inverse, f"Inverse of {name}.")
            if range_:
                self.graph.add((inv, RDFS.domain, NNFS[range_]))
            if domain:
                self.graph.add((inv, RDFS.range, NNFS[domain]))
            self.graph.add((inv, OWL.inverseOf, iri))
            if transitive:
                self.graph.add((inv, RDF.type, OWL.TransitiveProperty))
            if functional:
                self.graph.add((inv, RDF.type, OWL.InverseFunctionalProperty))
            if inverse_functional:
                self.graph.add((inv, RDF.type, OWL.FunctionalProperty))
            if asymmetric:
                self.graph.add((inv, RDF.type, OWL.AsymmetricProperty))
            if irreflexive:
                self.graph.add((inv, RDF.type, OWL.IrreflexiveProperty))
        return iri

    def data_property(self, name: str, domain: str, datatype: URIRef, comment: str) -> URIRef:
        """
        Declare a functional data property.

        Parameters
        ----------
        name : str
            Local name.
        domain : str
            Local name of the domain class.
        datatype : rdflib.URIRef
            XSD datatype of the values.
        comment : str
            rdfs:comment.

        Returns
        -------
        rdflib.URIRef
            The property IRI.

        Notes
        -----
        Processing:
        1. Declare the property as a functional datatype property with its
           domain and range.
        """
        iri = self.entity(name, OWL.DatatypeProperty, name, comment)
        self.graph.add((iri, RDF.type, OWL.FunctionalProperty))
        self.graph.add((iri, RDFS.domain, NNFS[domain]))
        self.graph.add((iri, RDFS.range, datatype))
        return iri

    def individual(self, name: str, cls: str, label: str, comment: str | None = None) -> URIRef:
        """
        Declare a named individual of a class.

        Parameters
        ----------
        name : str
            Local name.
        cls : str
            Local name of its class.
        label : str
            rdfs:label.
        comment : str or None, default=None
            rdfs:comment.

        Returns
        -------
        rdflib.URIRef
            The individual's IRI.

        Notes
        -----
        Processing:
        1. Declare it as owl:NamedIndividual and as a member of the class.
        2. Annotate it with its canvas node id, if it has one.
        """
        iri = self.entity(name, OWL.NamedIndividual, label, comment)
        self.graph.add((iri, RDF.type, NNFS[cls]))
        if name in CANVAS_NODES:
            self.graph.add((iri, NNFS.canvasNodeId, Literal(CANVAS_NODES[name])))
        return iri

    def fact(self, subject: str, prop: str, obj: Any) -> None:
        """
        Add a property assertion between local names, or to a literal.

        Parameters
        ----------
        subject : str
            Local name of the subject individual.
        prop : str
            Local name of the property.
        obj : str or rdflib.Literal
            Local name of the object individual, or a literal value.

        Returns
        -------
        None
            Adds one triple.

        Notes
        -----
        Processing:
        1. Resolve the local names and add the triple.
        """
        self.graph.add((NNFS[subject], NNFS[prop], self.term(obj)))

    def restriction(self, prop: str, *, some: Any = None, value: Any = None) -> BNode:
        """
        Build an owl:Restriction on a property.

        Parameters
        ----------
        prop : str
            Local name of the property.
        some : str, rdflib.URIRef or rdflib.BNode, optional
            Filler of an owl:someValuesFrom restriction (a local class name
            or a class / datatype node).
        value : str or rdflib.Literal, optional
            Filler of an owl:hasValue restriction (a local individual name
            or a literal).

        Returns
        -------
        rdflib.BNode
            The restriction node.

        Notes
        -----
        Processing:
        1. Create the node, set the property, then the filler.
        """
        node = self.bnode()
        self.graph.add((node, RDF.type, OWL.Restriction))
        self.graph.add((node, OWL.onProperty, NNFS[prop]))
        if some is not None:
            self.graph.add((node, OWL.someValuesFrom, self.term(some)))
        if value is not None:
            self.graph.add((node, OWL.hasValue, self.term(value)))
        return node

    @staticmethod
    def term(value: Any) -> Any:
        """
        Resolve a local name to its IRI, leaving rdflib nodes and literals alone.

        Parameters
        ----------
        value : str, rdflib.URIRef, rdflib.BNode or rdflib.Literal
            A local name, or an rdflib term.

        Returns
        -------
        rdflib term
            NNFS[value] for a plain string, otherwise the value itself.

        Notes
        -----
        Processing:
        1. rdflib's BNode, URIRef and Literal are all str subclasses, so
           they are checked first; only a plain str is a local name.
        """
        if isinstance(value, (URIRef, BNode, Literal)):
            return value
        return NNFS[value]

    def intersection(self, *members: Any) -> BNode:
        """
        Build an anonymous class that is the intersection of its members.

        Parameters
        ----------
        *members : str or rdflib node
            Local class names or restriction nodes.

        Returns
        -------
        rdflib.BNode
            The intersection class node.

        Notes
        -----
        Processing:
        1. Create an owl:Class node with an owl:intersectionOf list.
        """
        node = self.bnode()
        self.graph.add((node, RDF.type, OWL.Class))
        items = [self.term(m) for m in members]
        head = self.bnode()
        Collection(self.graph, head, items)
        self.graph.add((node, OWL.intersectionOf, head))
        return node

    def defined_class(self, name: str, parent: str, expression: BNode, comment: str) -> URIRef:
        """
        Declare a class equivalent to an expression, for the reasoner to fill.

        Parameters
        ----------
        name : str
            Local name.
        parent : str
            Local name of the asserted superclass.
        expression : rdflib.BNode
            The anonymous class the named class is equivalent to.
        comment : str
            rdfs:comment describing the definition.

        Returns
        -------
        rdflib.URIRef
            The class IRI.

        Notes
        -----
        Processing:
        1. Declare the class under the parent.
        2. Add the owl:equivalentClass axiom.
        """
        iri = self.cls(name, parent, comment)
        self.graph.add((iri, OWL.equivalentClass, expression))
        return iri

    def integer_range(self, facet: URIRef, bound: int) -> BNode:
        """
        Build a datatype restricting xsd:integer by one facet.

        Parameters
        ----------
        facet : rdflib.URIRef
            xsd:minInclusive or xsd:maxInclusive.
        bound : int
            The bound.

        Returns
        -------
        rdflib.BNode
            The datatype node.

        Notes
        -----
        Processing:
        1. Create an rdfs:Datatype on xsd:integer with one restriction in
           its owl:withRestrictions list.
        """
        node = self.bnode()
        self.graph.add((node, RDF.type, RDFS.Datatype))
        self.graph.add((node, OWL.onDatatype, XSD.integer))
        limit = self.bnode()
        self.graph.add((limit, facet, Literal(bound)))
        head = self.bnode()
        Collection(self.graph, head, [limit])
        self.graph.add((node, OWL.withRestrictions, head))
        return node

    def all_disjoint(self, names: list[str]) -> None:
        """
        Declare a set of classes pairwise disjoint.

        Parameters
        ----------
        names : list of str
            Local class names.

        Returns
        -------
        None
            Adds one owl:AllDisjointClasses axiom.

        Notes
        -----
        Processing:
        1. Create the axiom node with an owl:members list.
        """
        node = self.bnode()
        self.graph.add((node, RDF.type, OWL.AllDisjointClasses))
        head = self.bnode()
        Collection(self.graph, head, [NNFS[n] for n in names])
        self.graph.add((node, OWL.members, head))


def add_schema(b: Builder) -> None:
    """
    Add the classes, properties and definitions.

    Parameters
    ----------
    b : Builder
        The builder to add to.

    Returns
    -------
    None
        Adds the ontology header, the class hierarchy, the object, data
        and annotation properties, the disjointness axioms and the defined
        classes.

    Notes
    -----
    Processing:
    1. Ontology header with label, comment and version.
    2. Structural classes (Component and its kinds, Stage, Library) and
       domain classes (Task, Architecture, Technique, ...).
    3. Object properties, including the chains writes ∘ readBy ⊑ feeds and
       partOf ∘ belongsToStage ⊑ belongsToStage, and the transitive
       upstreamOf.
    4. Data properties and the canvasNodeId annotation.
    5. Defined classes: roles of modules and artifacts, stage membership,
       architecture families, framework reimplementations.
    """
    g = b.graph
    ont = URIRef(ONTOLOGY_IRI)
    g.add((ont, RDF.type, OWL.Ontology))
    g.add((ont, RDFS.label, Literal("neural-network-numpy knowledge base")))
    g.add(
        (
            ont,
            RDFS.comment,
            Literal(
                "The repository as nodes and edges: modules, packages, artifacts and make targets, what they "
                "read, write, import and belong to, and the concepts the code implements (tasks, architectures, "
                "techniques, optimizers, libraries). Generated by tools/build_knowledge_base.py; do not edit by "
                "hand, add to knowledge/extensions.owl instead. Start a reasoner (HermiT) to infer the pipeline "
                "order (upstreamOf), stage membership and the defined classes."
            ),
        )
    )
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    g.add((ont, OWL.versionInfo, Literal(pyproject["project"]["version"])))

    # ---- classes -----------------------------------------------------------
    b.cls(
        "Component",
        None,
        "A node of the architecture: a module, package, artifact or make target.",
    )
    b.cls("Module", "Component", "A Python source file.")
    b.cls("TestModule", "Module", "A pytest module in tests/.")
    b.cls("ToolScript", "Module", "A helper script in tools/.")
    b.cls("Package", "Component", "A Python package or a top-level source folder.")
    b.cls(
        "Artifact",
        "Component",
        "A file that is read or written: configs, data, results, figures, reports, documents.",
    )
    b.cls("ConfigFile", "Artifact", "A JSON or TOML configuration file.")
    b.cls("Dataset", "Artifact", "A raw dataset CSV.")
    b.cls("ResultsFile", "Artifact", "A machine-readable results file written by a run.")
    b.cls("Figure", "Artifact", "A set of PNG figures written by a run.")
    b.cls("Report", "Artifact", "A text report written by a run.")
    b.cls("Document", "Artifact", "Documentation, build files and other hand-written files.")
    b.cls("MakeTarget", "Component", "A command in the Makefile.")
    b.cls("Stage", None, "One of the coloured groups of the architecture canvas.")
    b.cls("Library", None, "A third-party Python package the code depends on.")
    b.cls("CoreLibrary", "Library", "A runtime dependency in pyproject's dependencies.")
    b.cls(
        "OptionalLibrary",
        "Library",
        "A dependency of the benchmarks extra, needed only for the library comparison.",
    )
    b.cls(
        "DevelopmentLibrary",
        "Library",
        "A dependency of the dev extra: tests, linting, this builder.",
    )
    b.cls("Task", None, "A learning task: a dataset, a loss and the metrics it is evaluated with.")
    b.cls("Architecture", None, "A network architecture from the experiment configs.")
    b.cls("Technique", None, "A modelling or training technique an architecture can use.")
    for name in (
        "Normalization",
        "Regularization",
        "Initialization",
        "BiasTerm",
        "GradientClipping",
    ):
        b.cls(name, "Technique")
    b.cls("LayerType", None, "A kind of layer implemented in nn/layers.py.")
    b.cls("Activation", None, "An activation function.")
    b.cls("Loss", None, "A training loss.")
    b.cls("Metric", None, "An evaluation metric.")
    b.cls("Optimizer", None, "A parameter-update rule.")
    b.cls("Sweep", None, "The full experiment grid.")
    b.cls("LearningRate", None, "A learning-rate value of the sweep.")
    b.cls("BatchSize", None, "A mini-batch size of the sweep.")
    b.cls(
        "Seed",
        None,
        "A random seed of the sweep: it sets the split, the initial weights, the dropout masks and the shuffling.",
    )
    b.cls("EarlyStoppingPolicy", None, "The early-stopping settings shared by every run.")
    b.cls("LibraryModel", None, "A model trained by a library in the comparison.")
    b.all_disjoint(["Module", "Package", "Artifact", "MakeTarget"])
    b.all_disjoint(["TestModule", "ToolScript"])
    b.all_disjoint(["ConfigFile", "Dataset", "ResultsFile", "Figure", "Report", "Document"])
    b.all_disjoint(["CoreLibrary", "OptionalLibrary", "DevelopmentLibrary"])
    b.all_disjoint(
        ["Normalization", "Regularization", "Initialization", "BiasTerm", "GradientClipping"]
    )
    b.all_disjoint(
        [
            "Component",
            "Stage",
            "Library",
            "Task",
            "Architecture",
            "Technique",
            "LayerType",
            "Activation",
            "Loss",
            "Metric",
            "Optimizer",
            "Sweep",
            "LearningRate",
            "BatchSize",
            "Seed",
            "EarlyStoppingPolicy",
            "LibraryModel",
        ]
    )

    # ---- object properties -------------------------------------------------
    b.object_property(
        "feeds",
        "Component",
        "Component",
        "Direct data or code flow from one node to another: a module into the artifact it writes, an artifact into the module that reads it, a module into the module that imports it. Also inferred from writes ∘ readBy.",
        inverse="fedBy",
    )
    b.object_property(
        "upstreamOf",
        "Component",
        "Component",
        "Transitive closure of feeds: everything that, directly or through other nodes, contributes to a node.",
        inverse="downstreamOf",
        transitive=True,
    )
    g.add((NNFS.feeds, RDFS.subPropertyOf, NNFS.upstreamOf))
    b.object_property(
        "writes",
        "Module",
        "Artifact",
        "The module produces the artifact. Each artifact is written by one module.",
        inverse="writtenBy",
        parents=["feeds"],
        inverse_functional=True,
        asymmetric=True,
    )
    b.object_property(
        "reads",
        "Module",
        "Artifact",
        "The module consumes the artifact.",
        inverse="readBy",
        asymmetric=True,
    )
    g.add((NNFS.readBy, RDFS.subPropertyOf, NNFS.feeds))
    b.object_property(
        "imports",
        "Module",
        "Module",
        "The module imports the other module (from the source, including imports inside functions).",
        inverse="importedBy",
        asymmetric=True,
        irreflexive=True,
    )
    g.add((NNFS.importedBy, RDFS.subPropertyOf, NNFS.feeds))
    chain = b.bnode()
    Collection(g, chain, [NNFS.writes, NNFS.readBy])
    g.add((NNFS.feeds, OWL.propertyChainAxiom, chain))
    b.object_property(
        "partOf",
        "Component",
        "Package",
        "The module or package is inside the package, directly or through sub-packages.",
        inverse="contains",
        transitive=True,
    )
    b.object_property(
        "belongsToStage",
        "Component",
        "Stage",
        "The stage of the architecture the node belongs to. Asserted for root-level modules, artifacts, make targets and sub-packages; inferred for a sub-package's modules through partOf ∘ belongsToStage. Not declared functional: OWL 2 DL forbids that on a property defined by a chain.",
        inverse="hasStageComponent",
    )
    chain = b.bnode()
    Collection(g, chain, [NNFS.partOf, NNFS.belongsToStage])
    g.add((NNFS.belongsToStage, OWL.propertyChainAxiom, chain))
    b.object_property(
        "runBy",
        "Module",
        "MakeTarget",
        "The make target runs the module as a script. Each target runs one module.",
        inverse="runs",
        inverse_functional=True,
        asymmetric=True,
    )
    b.object_property(
        "testedBy",
        "Module",
        "TestModule",
        "The test module imports, and so exercises, the module.",
        inverse="covers",
        asymmetric=True,
    )
    b.object_property(
        "requiresLibrary",
        "Module",
        "Library",
        "The module imports the library.",
        inverse="requiredBy",
        asymmetric=True,
    )
    b.object_property(
        "implements",
        "Module",
        None,
        "The module contains the implementation of the concept. Each concept is implemented in one module.",
        inverse="implementedIn",
        inverse_functional=True,
        asymmetric=True,
    )
    b.object_property(
        "definedIn",
        None,
        "ConfigFile",
        "The architecture, task, sweep or library model is defined in the config file.",
        inverse="defines",
        asymmetric=True,
    )
    b.object_property(
        "hasArchitecture",
        "Sweep",
        "Architecture",
        "An architecture of the grid.",
        inverse="architectureOfSweep",
        asymmetric=True,
    )
    b.object_property(
        "hasOptimizer",
        "Sweep",
        "Optimizer",
        "An optimizer of the grid.",
        inverse="optimizerOfSweep",
        asymmetric=True,
    )
    b.object_property(
        "hasLearningRate",
        "Sweep",
        "LearningRate",
        "A learning rate of the grid.",
        inverse="learningRateOfSweep",
        asymmetric=True,
    )
    b.object_property(
        "hasBatchSize",
        "Sweep",
        "BatchSize",
        "A batch size of the grid.",
        inverse="batchSizeOfSweep",
        asymmetric=True,
    )
    b.object_property(
        "hasSeed", "Sweep", "Seed", "A seed of the grid.", inverse="seedOfSweep", asymmetric=True
    )
    b.object_property(
        "hasEarlyStopping",
        "Sweep",
        "EarlyStoppingPolicy",
        "The early-stopping settings of every run. A sweep has one policy.",
        inverse="earlyStoppingOfSweep",
        functional=True,
        asymmetric=True,
    )
    b.object_property(
        "onTask",
        "Sweep",
        "Task",
        "The sweep is run on the task.",
        inverse="taskOfSweep",
        asymmetric=True,
    )
    b.object_property(
        "variantOf",
        "Architecture",
        "Architecture",
        "The architecture derives from the other one by one or more changes. Transitive: A2-bn-dropout is a variant of A2-bn, and so of A2.",
        inverse="hasVariant",
        transitive=True,
    )
    b.object_property(
        "usesTechnique",
        "Architecture",
        "Technique",
        "The architecture uses the technique.",
        inverse="usedByArchitecture",
        asymmetric=True,
    )
    b.object_property(
        "usesHiddenActivation",
        "Architecture",
        "Activation",
        "The activation of the hidden layers. An architecture has one.",
        inverse="hiddenActivationOf",
        functional=True,
        asymmetric=True,
    )
    b.object_property(
        "usesOutputActivation",
        "Task",
        "Activation",
        "The activation of the output layer for this task. A task has one.",
        inverse="outputActivationOf",
        functional=True,
        asymmetric=True,
    )
    b.object_property(
        "onDataset",
        "Task",
        "Dataset",
        "The dataset of the task. One dataset per task, and one task per dataset.",
        inverse="datasetOfTask",
        functional=True,
        inverse_functional=True,
        asymmetric=True,
    )
    b.object_property(
        "usesLoss",
        "Task",
        "Loss",
        "The training loss of the task. One loss per task, and one task per loss.",
        inverse="lossOfTask",
        functional=True,
        inverse_functional=True,
        asymmetric=True,
    )
    b.object_property(
        "evaluatedBy",
        "Task",
        "Metric",
        "An evaluation metric of the task.",
        inverse="evaluates",
        asymmetric=True,
    )
    b.object_property(
        "providedBy",
        "LibraryModel",
        "Library",
        "The library that implements the model. A model has one.",
        inverse="provides",
        functional=True,
        asymmetric=True,
    )
    b.object_property(
        "mirrorsArchitecture",
        "LibraryModel",
        "Architecture",
        "The model rebuilds the network's architecture layer for layer. A model mirrors one.",
        inverse="mirroredBy",
        functional=True,
        asymmetric=True,
    )
    b.object_property(
        "forTask",
        "LibraryModel",
        "Task",
        "A task the model is trained on.",
        inverse="hasLibraryModel",
        asymmetric=True,
    )
    b.object_property(
        "frameworkCounterpartOf",
        "LibraryModel",
        "LibraryModel",
        "The two library models rebuild the same architecture in different frameworks (e.g. the Keras A1 and the PyTorch A1). Symmetric, and no model is its own counterpart.",
        symmetric=True,
        irreflexive=True,
    )

    # ---- data and annotation properties ------------------------------------
    b.data_property("path", "Component", XSD.string, "Path relative to the repository root.")
    b.data_property(
        "filePattern", "Artifact", XSD.string, "Path pattern of a stamped or per-library output."
    )
    b.data_property(
        "isStamped",
        "Artifact",
        XSD.boolean,
        "Whether each run writes a new file with a timestamp, so nothing is overwritten.",
    )
    b.data_property("version", "Library", XSD.string, "The pinned version in pyproject.toml.")
    b.data_property("hiddenLayerCount", "Architecture", XSD.integer, "Number of hidden layers.")
    b.data_property(
        "unitsPerHiddenLayer", "Architecture", XSD.integer, "Units in each hidden layer."
    )
    b.data_property(
        "dropoutRate", "Architecture", XSD.decimal, "Dropout rate after each hidden layer, if any."
    )
    b.data_property(
        "maxGradNorm",
        "Architecture",
        XSD.decimal,
        "The gradient-norm cap, if the architecture clips gradients.",
    )
    b.data_property("learningRateValue", "LearningRate", XSD.decimal, "The value.")
    b.data_property("batchSizeValue", "BatchSize", XSD.integer, "The value.")
    b.data_property("seedValue", "Seed", XSD.integer, "The value.")
    b.data_property("maxEpochs", "EarlyStoppingPolicy", XSD.integer, "Maximum number of epochs.")
    b.data_property(
        "patience",
        "EarlyStoppingPolicy",
        XSD.integer,
        "Epochs without a meaningful validation improvement before stopping.",
    )
    b.data_property(
        "minDelta",
        "EarlyStoppingPolicy",
        XSD.decimal,
        "Smallest validation-loss decrease that counts as an improvement.",
    )
    b.data_property(
        "minEpochsBeforeEarlyStop",
        "EarlyStoppingPolicy",
        XSD.integer,
        "Epochs that must finish before early stopping can trigger.",
    )
    b.data_property("sampleCount", "Dataset", XSD.integer, "Rows used after duplicate removal.")
    b.data_property("featureCount", "Dataset", XSD.integer, "Number of input features.")
    b.data_property(
        "runCount", "Sweep", XSD.integer, "Number of training runs in the grid over both tasks."
    )
    b.data_property(
        "testCount", "Package", XSD.integer, "Number of test functions in the package."
    )
    b.entity(
        "canvasNodeId",
        OWL.AnnotationProperty,
        "canvasNodeId",
        "Id of the node that represents the individual in knowledge/nn-architecture.canvas.",
    )

    # ---- defined classes ---------------------------------------------------
    b.defined_class(
        "Producer",
        "Module",
        b.intersection("Module", b.restriction("writes", some="Artifact")),
        "A module that writes an artifact. Inferred.",
    )
    b.defined_class(
        "Consumer",
        "Module",
        b.intersection("Module", b.restriction("reads", some="Artifact")),
        "A module that reads an artifact. Inferred.",
    )
    b.defined_class(
        "PipelineStep",
        "Module",
        b.intersection("Producer", "Consumer"),
        "A module that reads artifacts and writes others: a step of the pipeline. Inferred.",
    )
    b.defined_class(
        "EntryPoint",
        "Module",
        b.intersection("Module", b.restriction("runBy", some="MakeTarget")),
        "A module a make target runs as a script. Inferred.",
    )
    b.defined_class(
        "OptionalModule",
        "Module",
        b.intersection("Module", b.restriction("requiresLibrary", some="OptionalLibrary")),
        "A module that needs the benchmarks extra to be installed. Inferred.",
    )
    b.defined_class(
        "TestedModule",
        "Module",
        b.intersection("Module", b.restriction("testedBy", some="TestModule")),
        "A module that at least one test module imports. Inferred.",
    )
    b.defined_class(
        "StampedArtifact",
        "Artifact",
        b.intersection("Artifact", b.restriction("isStamped", value=Literal(True))),
        "An output written with a run stamp in its name. Inferred.",
    )
    for stage in STAGES:
        b.defined_class(
            f"{stage}Component",
            "Component",
            b.intersection("Component", b.restriction("belongsToStage", value=stage)),
            f"A node of the {stage}. Inferred through belongsToStage, including the partOf chain.",
        )
    b.defined_class(
        "Variant",
        "Architecture",
        b.intersection("Architecture", b.restriction("variantOf", some="Architecture")),
        "An architecture derived from another by one change. Inferred.",
    )
    b.defined_class(
        "DeepArchitecture",
        "Architecture",
        b.intersection(
            "Architecture",
            b.restriction("hiddenLayerCount", some=b.integer_range(XSD.minInclusive, 2)),
        ),
        "An architecture with at least two hidden layers. Inferred from hiddenLayerCount (needs a reasoner with datatype facets, e.g. HermiT).",
    )
    b.defined_class(
        "ShallowArchitecture",
        "Architecture",
        b.intersection(
            "Architecture",
            b.restriction("hiddenLayerCount", some=b.integer_range(XSD.maxInclusive, 1)),
        ),
        "An architecture with at most one hidden layer. Inferred from hiddenLayerCount (needs a reasoner with datatype facets, e.g. HermiT).",
    )
    g.add((NNFS.DeepArchitecture, OWL.disjointWith, NNFS.ShallowArchitecture))
    for name, technique in [
        ("NormalizedArchitecture", "Normalization"),
        ("RegularizedArchitecture", "Regularization"),
        ("ClippedArchitecture", "GradientClipping"),
        ("BiasedArchitecture", "BiasTerm"),
    ]:
        b.defined_class(
            name,
            "Architecture",
            b.intersection("Architecture", b.restriction("usesTechnique", some=technique)),
            f"An architecture that uses a {technique} technique. Inferred.",
        )
    b.defined_class(
        "FrameworkReimplementation",
        "LibraryModel",
        b.intersection("LibraryModel", b.restriction("mirrorsArchitecture", some="Architecture")),
        "A library model that rebuilds one of the network's own architectures (Keras, PyTorch), as opposed to a different model family (scikit-learn). Inferred.",
    )


def python_modules() -> dict[str, Path]:
    """
    Find the repository's Python modules.

    Returns
    -------
    dict of str to pathlib.Path
        Dotted module name (e.g. "nn_numpy.nn.layers",
        "tests.test_layers") -> source file, in sorted order.

    Notes
    -----
    Processing:
    1. Collect every .py file under nn_numpy/, tools/ and tests/.
    2. Leave out __init__.py and anything under __pycache__.
    """
    files = (
        list((ROOT / "nn_numpy").rglob("*.py"))
        + list((ROOT / "tools").glob("*.py"))
        + list((ROOT / "tests").glob("*.py"))
    )
    modules = {}
    for path in sorted(files):
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue
        modules[".".join(path.relative_to(ROOT).with_suffix("").parts)] = path
    return modules


def imports_of(path: Path, modules: dict[str, Path]) -> tuple[set[str], set[str]]:
    """
    List what a module imports.

    Parameters
    ----------
    path : pathlib.Path
        The module's source file.
    modules : dict of str to pathlib.Path
        All repository modules, from python_modules().

    Returns
    -------
    internal : set of str
        Dotted names of the repository modules it imports.
    external : set of str
        Top-level names of the third-party packages it imports.

    Notes
    -----
    Processing:
    1. Parse the file and walk every Import / ImportFrom node, including
       those inside functions (lazy imports).
    2. A name that is a repository module counts as internal; for
       "from package import name", the submodule package.name is tried too.
    3. Other names count as external unless they are in the standard
       library or one of the repository's own top-level packages.
    """
    internal, external = set(), set()

    def resolve(name: str) -> None:
        """
        Record one imported name as internal, external or ignored.

        Parameters
        ----------
        name : str
            Dotted import name, e.g. "nn_numpy.nn.layers" or "torch.nn".

        Returns
        -------
        None
            Adds to the enclosing internal or external set.

        Notes
        -----
        Processing:
        1. A repository module name is added to internal.
        2. Otherwise a name from the repository's own packages or the
           standard library is ignored.
        3. Anything else adds its top-level package to external.

        Step 2 also drops names such as "nn_numpy.nn.layers.Dense" (a class
        imported from a module), which step 1 does not recognize as a module.
        """
        if name in modules:
            internal.add(name)
            return
        top = name.split(".")[0]
        if top in ("nn_numpy", "tools", "tests") or top in sys.stdlib_module_names:
            return
        external.add(top)

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolve(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            resolve(node.module)
            for alias in node.names:
                resolve(f"{node.module}.{alias.name}")
    return internal, external


def module_summary(path: Path) -> str | None:
    """
    First paragraph of a module's docstring.

    Parameters
    ----------
    path : pathlib.Path
        The module's source file.

    Returns
    -------
    str or None
        The first paragraph with whitespace collapsed, or None without a
        docstring.

    Notes
    -----
    Processing:
    1. Read the docstring with ast.get_docstring.
    2. Cut at the first blank line and join the lines with single spaces.
    """
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    if not doc:
        return None
    return " ".join(doc.split("\n\n")[0].split())


def make_targets() -> dict[str, tuple[str, str | None]]:
    """
    Read the Makefile's documented targets.

    Returns
    -------
    dict of str to tuple of (str, str or None)
        Target name -> (description from the "##" comment, dotted name of
        the module its recipe runs, or None).

    Notes
    -----
    Processing:
    1. Walk the lines; remember the latest "## " comment.
    2. A line "name:" (not starting with ".") starts a target with that
       description; its recipe lines are scanned for "-m $(PACKAGE).x"
       or "tools/x.py" to find the module run.
    """
    targets: dict[str, tuple[str, str | None]] = {}
    description, current = "", None
    for line in (ROOT / "Makefile").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            description = line[3:].strip()
            continue
        head = re.match(r"^([A-Za-z_][\w-]*):", line)
        if head:
            current = head.group(1)
            targets.setdefault(current, (description, None))
            continue
        if line.startswith("\t") and current:
            found = re.search(r"-m \$\(PACKAGE\)\.([\w.]+)", line)
            tool = re.search(r"tools/(\w+)\.py", line)
            module = (
                f"nn_numpy.{found.group(1)}"
                if found
                else f"tools.{tool.group(1)}"
                if tool
                else None
            )
            if module:
                targets[current] = (targets[current][0], module)
    return targets


def libraries() -> dict[str, tuple[str, str, list[str]]]:
    """
    Read the dependencies from pyproject.toml.

    Returns
    -------
    dict of str to tuple of (str, str, list of str)
        Pip name -> (library class: CoreLibrary, OptionalLibrary or
        DevelopmentLibrary; version specifier; import names).

    Notes
    -----
    Processing:
    1. Read project.dependencies (core), the benchmarks extra (optional)
       and the dev extra (development).
    2. Split each requirement into name and version specifier.
    3. Map pip names to import names with IMPORT_NAMES (default: the pip
       name itself).
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sections = [
        (pyproject["project"]["dependencies"], "CoreLibrary"),
        (pyproject["project"]["optional-dependencies"]["benchmarks"], "OptionalLibrary"),
        (pyproject["project"]["optional-dependencies"]["dev"], "DevelopmentLibrary"),
    ]
    result = {}
    for requirements, cls in sections:
        for requirement in requirements:
            name = re.split(r"[\s=<>!~]", requirement, maxsplit=1)[0]
            spec = requirement[len(name) :].strip()
            result[name] = (cls, spec, IMPORT_NAMES.get(name, [name]))
    return result


def add_structure(b: Builder) -> None:
    """
    Add the nodes and edges of the repository: stages, packages, modules,
    artifacts, make targets and libraries.

    Parameters
    ----------
    b : Builder
        The builder to add to.

    Returns
    -------
    None
        Adds the individuals and their asserted relations.

    Notes
    -----
    Processing:
    1. Stages and packages, with each sub-package's stage.
    2. Libraries from pyproject.toml.
    3. Every module: class (Module, TestModule or ToolScript), path,
       docstring summary, package, stage (root-level modules only),
       imports, required libraries, data flow and what it implements.
       Test modules also get "covers" edges to the modules they import.
    4. Artifacts from ARTIFACTS, with path or pattern, stamp flag and stage.
    5. Make targets from the Makefile, each running its module.
    """
    for stage, comment in STAGES.items():
        b.individual(
            stage, "Stage", stage.replace("Stage", " stage").lower().capitalize(), comment
        )

    for name, (stage, comment) in PACKAGES.items():
        b.individual(name, "Package", name, comment)
        b.fact(name, "path", Literal(name.replace(".", "/") + "/"))
        if stage:
            b.fact(name, "belongsToStage", stage)
        if "." in name:
            b.fact(name, "partOf", name.rsplit(".", 1)[0])

    import_to_library = {}
    for pip_name, (cls, spec, import_names) in libraries().items():
        local = re.sub(r"[^A-Za-z0-9_]", "_", pip_name)
        b.individual(local, cls, pip_name, f"{pip_name} {spec}".strip())
        if spec:
            b.fact(local, "version", Literal(spec))
        for import_name in import_names:
            import_to_library[import_name] = local

    modules = python_modules()
    test_count = 0
    for name, path in modules.items():
        kind = (
            "TestModule"
            if name.startswith("tests.")
            else "ToolScript"
            if name.startswith("tools.")
            else "Module"
        )
        relative = path.relative_to(ROOT).as_posix()
        b.individual(name, kind, relative, module_summary(path))
        b.fact(name, "path", Literal(relative))
        package = name.rsplit(".", 1)[0]
        b.fact(name, "partOf", package)
        if name in MODULE_STAGES:
            b.fact(name, "belongsToStage", MODULE_STAGES[name])
        internal, external = imports_of(path, modules)
        for other in sorted(internal):
            b.fact(name, "imports", other)
            if kind == "TestModule":
                b.fact(name, "covers", other)
        for import_name in sorted(external):
            if import_name in import_to_library:
                b.fact(name, "requiresLibrary", import_to_library[import_name])
        if kind == "TestModule":
            test_count += sum(
                isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                for node in ast.parse(path.read_text(encoding="utf-8")).body
            )
    b.fact("tests", "testCount", Literal(test_count))

    for module, (reads, writes) in DATA_FLOW.items():
        for artifact in reads:
            b.fact(module, "reads", artifact)
        for artifact in writes:
            b.fact(module, "writes", artifact)
    for module, concepts in IMPLEMENTS.items():
        for concept in concepts:
            b.fact(module, "implements", concept)

    for name, (cls, path, stamped, stage, comment) in ARTIFACTS.items():
        b.individual(name, cls, path, comment)
        b.fact(name, "filePattern" if stamped or "<" in path else "path", Literal(path))
        b.fact(name, "isStamped", Literal(stamped))
        b.fact(name, "belongsToStage", stage)

    for target, (description, module) in make_targets().items():
        name = f"make_{target.replace('-', '_')}"
        b.individual(name, "MakeTarget", f"make {target}", description)
        b.fact(name, "belongsToStage", "InfrastructureStage")
        if module:
            b.fact(name, "runs", module)


def add_domain(b: Builder) -> None:
    """
    Add the concepts the code implements and the experiment grid.

    Parameters
    ----------
    b : Builder
        The builder to add to.

    Returns
    -------
    None
        Adds techniques, layer types, activations, losses, metrics,
        optimizers, tasks, datasets' sizes, the architectures, the sweep and
        the library models.

    Notes
    -----
    Processing:
    1. The fixed domain individuals from DOMAIN.
    2. Tasks: dataset, loss, metrics, output activation (from the config's
       output layer) and config file. Dataset sizes are counted from the
       CSVs after duplicate removal; feature counts come from the configs.
    3. Architectures from the validated configs: hidden layers, units,
       hidden activation, techniques (bias, init, batch norm, dropout,
       gradient clipping), the base architecture they are a variant of
       (by name: "A2-bias" -> "A2"), and the config files.
    4. The sweep: optimizers, learning rates, batch sizes, seeds, early
       stopping, run count, tasks.
    5. Library models from the benchmark config.
    """
    for name, (cls, label, comment) in DOMAIN.items():
        b.individual(name, cls, label, comment)

    configs = {
        "ClassificationTask": ConfigLoader().load_and_validate("classification_experiments.json"),
        "RegressionTask": ConfigLoader().load_and_validate("regression_experiments.json"),
    }
    activation_names = {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh", "linear": "Linear"}
    init_names = {
        "he": "HeInitialization",
        "xavier": "XavierInitialization",
        "normal": "NormalInitialization",
    }
    csv_rows = {
        "BanknoteDataset": len(pd.read_csv(ROOT / "data/raw/banknote_auth.csv").drop_duplicates()),
        "EnergyDataset": len(
            pd.read_csv(ROOT / "data/raw/energy_efficiency.csv").drop_duplicates()
        ),
    }
    for task, (dataset, loss, metrics, config_name) in TASK_LINKS.items():
        config = configs[task]
        b.fact(task, "onDataset", dataset)
        b.fact(task, "usesLoss", loss)
        for metric in metrics:
            b.fact(task, "evaluatedBy", metric)
        output_layer = config["architectures"]["A1"][-1]
        b.fact(task, "usesOutputActivation", activation_names[output_layer["activation"]])
        b.fact(task, "definedIn", config_name)
        b.fact(dataset, "sampleCount", Literal(csv_rows[dataset]))
        b.fact(dataset, "featureCount", Literal(config["input_dimension"]))

    # Architectures: the same in both configs apart from the output activation.
    classification, regression = configs["ClassificationTask"], configs["RegressionTask"]
    assert list(classification["architectures"]) == list(regression["architectures"])
    for name, layers in classification["architectures"].items():
        other = regression["architectures"][name]
        assert [dict(layer, activation=None) for layer in layers[:-1]] == [
            dict(layer, activation=None) for layer in other[:-1]
        ]
        hidden = layers[:-1]
        local = name.replace("-", "_")
        b.individual(
            local,
            "Architecture",
            name,
            describe_architecture(
                name, hidden, classification["architecture_options"].get(name, {})
            ),
        )
        b.fact(local, "hiddenLayerCount", Literal(len(hidden)))
        b.fact(local, "unitsPerHiddenLayer", Literal(hidden[0]["units"]))
        b.fact(local, "usesHiddenActivation", activation_names[hidden[0]["activation"]])
        techniques = set()
        for layer in layers:
            if layer.get("use_bias"):
                techniques.add("BiasTerms")
            techniques.add(init_names[layer.get("init", "normal")])
            if layer.get("batch_norm"):
                techniques.add("BatchNormalization")
            if layer.get("dropout", 0) > 0:
                techniques.add("Dropout")
                b.fact(local, "dropoutRate", Literal(layer["dropout"], datatype=XSD.decimal))
        cap = classification["architecture_options"].get(name, {}).get("max_grad_norm")
        if cap is not None:
            techniques.add("GlobalNormClipping")
            b.fact(local, "maxGradNorm", Literal(cap, datatype=XSD.decimal))
        for technique in sorted(techniques):
            b.fact(local, "usesTechnique", technique)
        base = name
        while "-" in base:
            base = base.rsplit("-", 1)[0]
            if base in classification["architectures"]:
                b.fact(local, "variantOf", base.replace("-", "_"))
                break
        b.fact(local, "definedIn", "ClassificationConfig")
        b.fact(local, "definedIn", "RegressionConfig")

    # The sweep, shared by both tasks.
    experiments = classification["experiments"]
    assert experiments == regression["experiments"]
    b.individual(
        "ExperimentSweep",
        "Sweep",
        "the experiment sweep",
        "Every architecture × optimizer × learning rate × batch size × seed, on both tasks, with the same early-stopping settings.",
    )
    for name in classification["architectures"]:
        b.fact("ExperimentSweep", "hasArchitecture", name.replace("-", "_"))
    for optimizer in experiments["optimizers"]:
        b.fact("ExperimentSweep", "hasOptimizer", OPTIMIZER_NAMES[optimizer])
    for lr in experiments["learning_rates"]:
        local = f"LearningRate_{str(lr).replace('.', '_')}"
        b.individual(local, "LearningRate", f"learning rate {lr}")
        b.fact(local, "learningRateValue", Literal(lr, datatype=XSD.decimal))
        b.fact("ExperimentSweep", "hasLearningRate", local)
    for size in experiments["batch_sizes"]:
        local = f"BatchSize_{size}"
        b.individual(local, "BatchSize", f"batch size {size}")
        b.fact(local, "batchSizeValue", Literal(size))
        b.fact("ExperimentSweep", "hasBatchSize", local)
    for seed in experiments["seeds"]:
        local = f"Seed_{seed}"
        b.individual(local, "Seed", f"seed {seed}")
        b.fact(local, "seedValue", Literal(seed))
        b.fact("ExperimentSweep", "hasSeed", local)
    b.individual(
        "EarlyStopping",
        "EarlyStoppingPolicy",
        "early stopping",
        "Stop after `patience` epochs without a validation-loss improvement of more than `minDelta`, never before `minEpochsBeforeEarlyStop` epochs, and restore the best checkpoint.",
    )
    b.fact("EarlyStopping", "maxEpochs", Literal(experiments["epochs"]))
    b.fact("EarlyStopping", "patience", Literal(experiments["patience"]))
    b.fact("EarlyStopping", "minDelta", Literal(experiments["min_delta"], datatype=XSD.decimal))
    b.fact(
        "EarlyStopping",
        "minEpochsBeforeEarlyStop",
        Literal(experiments["min_epochs_before_early_stop"]),
    )
    b.fact("ExperimentSweep", "hasEarlyStopping", "EarlyStopping")
    runs = (
        len(classification["architectures"])
        * len(experiments["optimizers"])
        * len(experiments["learning_rates"])
        * len(experiments["batch_sizes"])
        * len(experiments["seeds"])
        * len(configs)
    )
    b.fact("ExperimentSweep", "runCount", Literal(runs))
    for task, (_, _, _, config_name) in TASK_LINKS.items():
        b.fact("ExperimentSweep", "onTask", task)
        b.fact("ExperimentSweep", "definedIn", config_name)

    # Library models from the benchmark config.
    benchmark = json.loads(
        (ROOT / "configs/benchmark_experiments.json").read_text(encoding="utf-8")
    )
    tasks = {"classification": "ClassificationTask", "regression": "RegressionTask"}
    for problem, models in benchmark["sklearn"].items():
        for model in models:
            local = f"sklearn_{problem}_{model}"
            b.individual(
                local,
                "LibraryModel",
                f"scikit-learn {model.replace('_', ' ')} ({problem})",
                f"scikit-learn's {model.replace('_', ' ')} for {problem}, tuned over the small grid in configs/benchmark_experiments.json.",
            )
            b.fact(local, "providedBy", "scikit_learn")
            b.fact(local, "forTask", tasks[problem])
            b.fact(local, "definedIn", "BenchmarkConfig")
    for library, pip_name, label in [
        ("tensorflow", "tensorflow", "Keras"),
        ("pytorch", "torch", "PyTorch"),
    ]:
        for arch in benchmark[library]["architectures"]:
            local = f"{library}_{arch.replace('-', '_')}"
            b.individual(
                local,
                "LibraryModel",
                f"{label} {arch}",
                f"The network's {arch} rebuilt in {label} with the framework's defaults, trained on both tasks over the same grid shape.",
            )
            b.fact(local, "providedBy", pip_name)
            b.fact(local, "mirrorsArchitecture", arch.replace("-", "_"))
            if library == "pytorch" and arch in benchmark["tensorflow"]["architectures"]:
                b.fact(local, "frameworkCounterpartOf", f"tensorflow_{arch.replace('-', '_')}")
            for task in tasks.values():
                b.fact(local, "forTask", task)
            b.fact(local, "definedIn", "BenchmarkConfig")


def describe_architecture(name: str, hidden: list[dict[str, Any]], options: dict[str, Any]) -> str:
    """
    One-line description of an architecture.

    Parameters
    ----------
    name : str
        The architecture's name.
    hidden : list of dict
        Its hidden layers from the config.
    options : dict
        Its architecture options (e.g. max_grad_norm).

    Returns
    -------
    str
        E.g. "3 hidden layers of 32 relu units; bias terms; batch norm".

    Notes
    -----
    Processing:
    1. Count the hidden layers and read the first one's units and
       activation.
    2. List the extras any layer uses: bias, init, batch norm, dropout, and
       the gradient cap from the options.
    """
    extras = []
    if any(layer.get("use_bias") for layer in hidden):
        extras.append("bias terms")
    inits = {layer.get("init", "normal") for layer in hidden} - {"normal"}
    extras.extend(f"{init} initialization" for init in sorted(inits))
    if any(layer.get("batch_norm") for layer in hidden):
        extras.append("batch normalization")
    rates = {layer["dropout"] for layer in hidden if layer.get("dropout", 0) > 0}
    extras.extend(f"dropout {rate}" for rate in sorted(rates))
    if "max_grad_norm" in options:
        extras.append(f"gradient-norm clipping at {options['max_grad_norm']}")
    plural = "s" if len(hidden) != 1 else ""
    text = f"{name}: {len(hidden)} hidden layer{plural} of {hidden[0]['units']} {hidden[0]['activation']} units"
    return text + ("; " + "; ".join(extras) if extras else "") + "."


# Element and predicate order in the written file: declarations by kind,
# then a fixed order of the common predicates.
DECLARATION_TYPES = [
    OWL.Ontology,
    OWL.AnnotationProperty,
    OWL.Class,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.NamedIndividual,
]
PREDICATE_ORDER = [
    RDF.type,
    RDFS.label,
    RDFS.comment,
    RDFS.subClassOf,
    OWL.equivalentClass,
    OWL.disjointWith,
    RDFS.subPropertyOf,
    OWL.inverseOf,
    OWL.propertyChainAxiom,
    RDFS.domain,
    RDFS.range,
    OWL.imports,
    OWL.versionInfo,
]
PREFIXES = {
    str(RDF): "rdf",
    str(RDFS): "rdfs",
    str(OWL): "owl",
    str(XSD): "xsd",
    str(NNFS): "nnfs",
}


def qname(iri: Node) -> str:
    """
    Prefixed name of an IRI in one of the file's namespaces.

    Parameters
    ----------
    iri : rdflib.term.Node
        An IRI in the rdf, rdfs, owl, xsd or nnfs namespace. Typed as Node
        because rdflib returns predicates and rdf:type objects as Node; any
        other node (a blank node, a literal) raises ValueError.

    Returns
    -------
    str
        E.g. "owl:Class" or "nnfs:writes".

    Raises
    ------
    ValueError
        If the IRI is in none of the known namespaces.

    Notes
    -----
    Processing:
    1. Match the IRI's start against the known namespaces.
    """
    for namespace, prefix in PREFIXES.items():
        if str(iri).startswith(namespace):
            return f"{prefix}:{str(iri)[len(namespace) :]}"
    raise ValueError(f"No prefix for {iri}")


def write_rdfxml(graph: Graph, path: Path) -> None:
    """
    Write a graph as readable RDF/XML.

    Parameters
    ----------
    graph : rdflib.Graph
        The ontology. Every named subject must be declared with one of
        DECLARATION_TYPES; blank nodes are class expressions, lists,
        datatypes, facets or owl:AllDisjointClasses axioms.
    path : pathlib.Path
        Output file.

    Returns
    -------
    None
        Writes the file.

    Notes
    -----
    Processing:
    1. Named subjects become top-level elements named after their
       declaration type, ordered by kind and then by IRI; blank-node
       axioms (owl:AllDisjointClasses) come last.
    2. Under each subject, predicates follow PREDICATE_ORDER and then
       alphabetical order; objects are references (rdf:resource),
       literals, or nested elements for blank nodes.
    3. A blank node that heads an RDF list is written as
       rdf:parseType="Collection"; any other blank node is written as a
       nested typed element (owl:Restriction, owl:Class, rdfs:Datatype) or
       rdf:Description.

    rdflib's own RDF/XML writer nests named entities inside each other and
    can turn a blank node inside a collection into a named IRI, which would
    corrupt the class expressions; this writer avoids both.
    """
    lines: list[str] = []

    def attr(value: str) -> str:
        """
        Escape a string for use inside a double-quoted XML attribute.

        Parameters
        ----------
        value : str
            Raw attribute value, e.g. an IRI.

        Returns
        -------
        str
            value with &, <, > and " replaced by entities.

        Notes
        -----
        xml.sax.saxutils.escape handles &, < and > only; the extra mapping
        covers the double quote that delimits the attribute.
        """
        return escape(value, {'"': "&quot;"})

    def write_object(predicate: Node, obj: Any, depth: int) -> None:
        """
        Append the XML for one predicate-object pair of a subject.

        Parameters
        ----------
        predicate : rdflib.term.Node
            The property; always an IRI in RDF, but typed as Node because
            rdflib returns predicates as Node.
        obj : rdflib.URIRef, rdflib.Literal or rdflib.BNode
            The object.
        depth : int
            Nesting level; each level indents by two spaces.

        Returns
        -------
        None
            Appends lines to the enclosing lines list.

        Notes
        -----
        Processing:
        1. An IRI becomes an empty element with rdf:resource.
        2. A literal becomes a text element, with rdf:datatype if it has one.
        3. A blank node that heads an RDF list becomes a
           rdf:parseType="Collection" element; named members are
           rdf:Description references, blank members nested nodes.
        4. Any other blank node is nested with write_node.
        """
        pad = "  " * depth
        name = qname(predicate)
        if isinstance(obj, URIRef):
            lines.append(f'{pad}<{name} rdf:resource="{attr(str(obj))}"/>')
        elif isinstance(obj, Literal):
            if obj.datatype is not None:
                datatype = attr(str(obj.datatype))
                lines.append(f'{pad}<{name} rdf:datatype="{datatype}">{escape(str(obj))}</{name}>')
            else:
                lines.append(f"{pad}<{name}>{escape(str(obj))}</{name}>")
        elif (obj, RDF.first, None) in graph:
            lines.append(f'{pad}<{name} rdf:parseType="Collection">')
            for item in Collection(graph, obj):
                if isinstance(item, URIRef):
                    lines.append(f'{pad}  <rdf:Description rdf:about="{attr(str(item))}"/>')
                else:
                    write_node(item, depth + 1)
            lines.append(f"{pad}</{name}>")
        else:
            lines.append(f"{pad}<{name}>")
            write_node(obj, depth + 1)
            lines.append(f"{pad}</{name}>")

    def write_node(node: Any, depth: int) -> None:
        """
        Append the XML element for a subject and all its properties.

        Parameters
        ----------
        node : rdflib.URIRef or rdflib.BNode
            A named subject (written with rdf:about) or a blank node
            (written without an identifier, nested in its parent).
        depth : int
            Nesting level; each level indents by two spaces.

        Returns
        -------
        None
            Appends lines to the enclosing lines list.

        Notes
        -----
        Processing:
        1. Name the element after the node's declaration type, else its
           only rdf:type, else rdf:Description.
        2. Leave out the rdf:type triple the element name already states.
        3. Write an empty element if nothing is left; otherwise write each
           remaining pair with write_object, in a fixed order.
        """
        pad = "  " * depth
        types = sorted(graph.objects(node, RDF.type), key=str)
        declared = next((t for t in DECLARATION_TYPES if t in types), None)
        element_type = declared if declared else types[0] if len(types) == 1 else None
        element = qname(element_type) if element_type else "rdf:Description"
        about = f' rdf:about="{attr(str(node))}"' if isinstance(node, URIRef) else ""
        pairs = [
            (p, o)
            for p, o in graph.predicate_objects(node)
            if not (p == RDF.type and o == element_type)
        ]
        if not pairs:
            lines.append(f"{pad}<{element}{about}/>")
            return
        lines.append(f"{pad}<{element}{about}>")

        def order(pair: tuple[Any, Any]) -> tuple[int, str, int, str]:
            """
            Sort key for a node's predicate-object pairs.

            Parameters
            ----------
            pair : tuple of (rdflib.term.Node, rdflib.term.Node)
                A (predicate, object) pair.

            Returns
            -------
            tuple of (int, str, int, str)
                (rank in PREDICATE_ORDER or after all of them, prefixed
                predicate name, 0 for IRIs / 1 for literals / 2 for blank
                nodes, object as text).

            Notes
            -----
            A total order on everything but blank-node objects keeps the
            output byte-identical between builds; the blank nodes' own
            names are numbered deterministically by Builder.bnode.
            """
            p, o = pair
            rank = PREDICATE_ORDER.index(p) if p in PREDICATE_ORDER else len(PREDICATE_ORDER)
            kind = 0 if isinstance(o, URIRef) else 1 if isinstance(o, Literal) else 2
            return (rank, qname(p), kind, str(o))

        for p, o in sorted(pairs, key=order):
            write_object(p, o, depth + 1)
        lines.append(f"{pad}</{element}>")

    def subject_order(subject: URIRef) -> tuple[int, str]:
        """
        Sort key for the top-level named subjects.

        Parameters
        ----------
        subject : rdflib.URIRef
            A named subject of the graph.

        Returns
        -------
        tuple of (int, str)
            (position of its declaration type in DECLARATION_TYPES, or
            after all of them if undeclared; the IRI), so the file lists
            the ontology header, annotation properties, classes, object
            and data properties, then individuals, each alphabetically.
        """
        types = set(graph.objects(subject, RDF.type))
        kind = next(
            (i for i, t in enumerate(DECLARATION_TYPES) if t in types), len(DECLARATION_TYPES)
        )
        return (kind, str(subject))

    named = sorted({s for s in graph.subjects() if isinstance(s, URIRef)}, key=subject_order)
    referenced = {o for o in graph.objects() if isinstance(o, BNode)}
    axioms = sorted(
        {s for s in graph.subjects() if isinstance(s, BNode) and s not in referenced}, key=str
    )

    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append("<rdf:RDF")
    for namespace, prefix in PREFIXES.items():
        lines.append(f'  xmlns:{prefix}="{namespace}"')
    lines.append(">")
    for subject in named:
        write_node(subject, 1)
    for axiom in axioms:
        write_node(axiom, 1)
    lines.append("</rdf:RDF>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_graph() -> Graph:
    """
    Build the whole ontology.

    Returns
    -------
    rdflib.Graph
        The ontology: schema, structure and domain.

    Notes
    -----
    Processing:
    1. add_schema, add_structure, add_domain on one Builder.
    """
    builder = Builder()
    add_schema(builder)
    add_structure(builder)
    add_domain(builder)
    return builder.graph


def extensions_graph() -> Graph:
    """
    Build the small ontology that imports the generated one.

    Returns
    -------
    rdflib.Graph
        An ontology with only a header and an owl:imports of the knowledge
        base, for manual additions in Protégé.

    Notes
    -----
    Processing:
    1. Declare the ontology, its label and comment, and the import.
    """
    g = Graph()
    g.bind("nnfs", NNFS)
    g.bind("owl", OWL)
    ont = URIRef(EXTENSIONS_IRI)
    g.add((ont, RDF.type, OWL.Ontology))
    g.add((ont, OWL.imports, URIRef(ONTOLOGY_IRI)))
    g.add((ont, RDFS.label, Literal("neural-network-numpy knowledge base: extensions")))
    g.add(
        (
            ont,
            RDFS.comment,
            Literal(
                "Manual additions to the generated knowledge base. Open this file in Protégé (it imports "
                "nn_numpy.owl from the same folder) and add classes, individuals or axioms here; "
                "regenerating the knowledge base never touches this file."
            ),
        )
    )
    return g


def main(output: Path = OUTPUT) -> None:
    """
    Write the knowledge base, and the extensions file if it doesn't exist.

    Parameters
    ----------
    output : pathlib.Path, default=OUTPUT
        Where to write the RDF/XML file.

    Returns
    -------
    None
        Writes the file(s) and prints their paths and the entity counts.

    Notes
    -----
    Processing:
    1. Build the graph and write it with write_rdfxml.
    2. Write knowledge/extensions.owl only if it is missing, so manual
       additions survive regeneration.
    """
    graph = build_graph()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_rdfxml(graph, output)
    counts = {
        kind: sum(isinstance(s, URIRef) for s in set(graph.subjects(RDF.type, iri)))
        for kind, iri in [
            ("classes", OWL.Class),
            ("object properties", OWL.ObjectProperty),
            ("data properties", OWL.DatatypeProperty),
            ("individuals", OWL.NamedIndividual),
        ]
    }
    print(f"Saved knowledge base to: {output}")
    print(
        "  " + ", ".join(f"{n} {kind}" for kind, n in counts.items()) + f", {len(graph)} triples"
    )
    if not EXTENSIONS.exists():
        write_rdfxml(extensions_graph(), EXTENSIONS)
        print(f"Saved extensions file to: {EXTENSIONS}")


# Running this script builds the knowledge base (optionally to the path
# given as the first argument).
if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT)
