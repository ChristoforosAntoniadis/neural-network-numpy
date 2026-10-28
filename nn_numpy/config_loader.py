"""
Loading and validation of the JSON experiment configs.

Defines `ConfigLoader`, which reads a config file from the project's
`configs/` folder and checks its structure and values before any
experiment runs.

Input: a JSON file name in configs/ (classification_experiments.json or
regression_experiments.json).

Output: the parsed config dictionary, validated and with each architecture's
training options split off into config["architecture_options"]. Nothing is
written to disk.
"""

import json
import os
from typing import Any

from nn_numpy.config import ROOT_DIR
from nn_numpy.nn.layers import SUPPORTED_INITS


class ConfigLoader:
    """
    Utility class for loading and validating high-level experiment configs.

    This class is responsible for:
    - locating config files inside the project's configs folder
    - loading JSON files into Python dictionaries
    - validating that the config structure is correct
    - checking that important values are sensible before experiments run

    Attributes
    ----------
    configs_dir : str
        Absolute path of the project's `configs/` folder.
    SUPPORTED_ACTIVATIONS : frozenset of str
        Activation names accepted in a layer's "activation" field.
    SUPPORTED_CLASSIFICATION_LOSSES : frozenset of str
        Loss names accepted when task_type is "classification".
    SUPPORTED_REGRESSION_LOSSES : frozenset of str
        Loss names accepted when task_type is "regression".

    Notes
    -----
    Validation helps catch mistakes early, such as:
    - missing keys
    - unsupported layer types
    - invalid activation names
    - impossible values like negative units or epochs
    - using the wrong loss for the task type
    """

    # Supported activation names used by nn_numpy.nn.activations.get_activation(...)
    # Frozensets, because these are fixed constants shared by every instance.
    SUPPORTED_ACTIVATIONS = frozenset({"relu", "sigmoid", "tanh", "linear"})

    # Supported loss names used by nn_numpy.nn.losses.get_loss(...)
    SUPPORTED_CLASSIFICATION_LOSSES = frozenset(
        {"bce", "binary_crossentropy", "binary_cross_entropy"}
    )
    SUPPORTED_REGRESSION_LOSSES = frozenset({"mse"})

    def __init__(self) -> None:
        """
        Create a ConfigLoader and point it to the configs directory.

        Parameters
        ----------
        None
            Uses `ROOT_DIR` from nn_numpy.config.

        Returns
        -------
        None
            Sets `self.configs_dir` to `<ROOT_DIR>/configs`.

        Notes
        -----
        Processing:
        1. Join the project root with "configs" and store the path.
        """
        self.configs_dir = os.path.join(ROOT_DIR, "configs")

    def load(self, filename: str) -> dict[str, Any]:
        """
        Load a JSON configuration file.

        Parameters
        ----------
        filename : str
            Name of the config file, for example:
            "classification_experiments.json"

        Returns
        -------
        dict
            Parsed JSON config as a Python dictionary. An experiment config
            has the keys "task_type", "input_dimension", "loss",
            "preprocessing", "architectures" and "experiments"; validate()
            checks them.

        Raises
        ------
        FileNotFoundError
            If the requested config file does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.

        Notes
        -----
        Processing:
        1. Build the full path `<configs_dir>/<filename>`.
        2. Raise FileNotFoundError if it does not exist.
        3. Open the file and parse it with `json.load`.

        No validation happens here; see `validate`.
        """
        # Build the full path to the target config file
        config_path = os.path.join(self.configs_dir, filename)

        # Stop early if the file does not exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Open and parse the JSON file
        with open(config_path, "r") as f:
            config = json.load(f)

        return config

    def validate(self, config: dict[str, Any]) -> bool:
        """
        Validate a high-level experiment config.

        Parameters
        ----------
        config : dict
            Configuration dictionary loaded from JSON. Required keys:
            - "task_type" (str): "classification" or "regression"
            - "input_dimension" (int): number of input features, >= 1
            - "loss" (str): a BCE name for classification, "mse" for
              regression (case-insensitive)
            - "architectures" (dict of str to list or dict): named
              architectures, each a non-empty list of layer dicts with
              "type" ("dense"), "units" (int >= 1), "activation" (str),
              and optional "use_bias" (bool), "batch_norm" (bool),
              "dropout" (float in [0, 1), not on the last layer) and
              "init" (str); or a dict with that list under "layers" and
              the optional "max_grad_norm" (number > 0), the gradient-norm
              cap used when training it
            - "experiments" (dict): "optimizers" (list), "learning_rates"
              (list), "batch_sizes" (list), "epochs" (int >= 1),
              "early_stopping" (bool), "patience" (int >= 1), "min_delta"
              (float >= 0), "min_epochs_before_early_stop" (int, 0 to
              epochs), and optional "seeds" (list of int)
            - "preprocessing" (dict): "enabled" (bool) and
              "scale_features" (bool)

        Returns
        -------
        bool
            True if validation succeeds.

        Raises
        ------
        ValueError
            If any required key is missing or any value is invalid. The
            message names the offending key (and architecture/layer).

        Notes
        -----
        Processing:
        1. Check the top-level keys, task_type, input_dimension and that
           architectures is a non-empty dict.
        2. Check the preprocessing block: required keys and boolean
           values.
        3. Check that the loss name matches the task type.
        4. Check every architecture: for the dict form, only the keys
           "layers" and "max_grad_norm" and a positive cap; then every
           layer: required keys, layer type, units, activation, and the
           optional use_bias, batch_norm, dropout and init settings.
        5. Check the experiments block: required keys, non-empty lists,
           positive epochs and patience, non-negative min_delta and
           min_epochs_before_early_stop, the optional seeds list, and
           that the minimum-epoch guard does not exceed epochs.
        6. Return True. Validation stops at the first problem found.

        The contents of the optimizers, learning_rates and batch_sizes
        lists are not checked here, only that they are non-empty lists.
        """

        # ----------------------------
        # Validate top-level structure
        # ----------------------------

        required_top_level_keys = [
            "task_type",
            "input_dimension",
            "loss",
            "architectures",
            "experiments",
            "preprocessing",
        ]

        # Ensure all required top-level keys exist
        for key in required_top_level_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: '{key}'")

        # task_type must be one of the two project modes
        if config["task_type"] not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")

        # input_dimension must be a positive integer
        if not isinstance(config["input_dimension"], int) or config["input_dimension"] < 1:
            raise ValueError("input_dimension must be a positive integer")

        # architectures must be a non-empty dictionary
        if not isinstance(config["architectures"], dict) or len(config["architectures"]) == 0:
            raise ValueError("'architectures' must be a non-empty dictionary")

        # ----------------------------
        # Validate preprocessing block
        # ----------------------------

        preprocessing = config["preprocessing"]

        # preprocessing must be a dictionary
        if not isinstance(preprocessing, dict):
            raise ValueError("'preprocessing' must be a dictionary")

        required_preprocessing_keys = ["enabled", "scale_features"]

        # Ensure all required preprocessing keys exist
        for key in required_preprocessing_keys:
            if key not in preprocessing:
                raise ValueError(f"Missing required preprocessing key: '{key}'")

        # enabled must be a boolean
        if not isinstance(preprocessing["enabled"], bool):
            raise ValueError("'preprocessing.enabled' must be true or false")

        # scale_features must be a boolean
        if not isinstance(preprocessing["scale_features"], bool):
            raise ValueError("'preprocessing.scale_features' must be true or false")

        # ----------------------------
        # Validate loss compatibility
        # ----------------------------

        # Normalize the loss name to lowercase so that values like "MSE" still work
        loss_name = str(config["loss"]).lower()

        # Classification configs should use BCE-style losses
        if config["task_type"] == "classification":
            if loss_name not in self.SUPPORTED_CLASSIFICATION_LOSSES:
                raise ValueError(
                    "For classification, loss must be one of: "
                    "'bce', 'binary_crossentropy', 'binary_cross_entropy'"
                )

        # Regression configs should use MSE
        elif (
            config["task_type"] == "regression"
            and loss_name not in self.SUPPORTED_REGRESSION_LOSSES
        ):
            raise ValueError("For regression, loss must be 'mse'")

        # ----------------------------
        # Validate architectures
        # ----------------------------

        for arch_name, entry in config["architectures"].items():
            # The dict form holds the layer list plus training options.
            if isinstance(entry, dict):
                unknown = set(entry) - {"layers", "max_grad_norm"}
                if unknown:
                    raise ValueError(
                        f"Architecture '{arch_name}' has unknown keys: {sorted(unknown)}"
                    )
                if "layers" not in entry:
                    raise ValueError(f"Architecture '{arch_name}' is missing 'layers'")
                if "max_grad_norm" in entry:
                    cap = entry["max_grad_norm"]
                    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or cap <= 0:
                        raise ValueError(
                            f"Architecture '{arch_name}' must have 'max_grad_norm' "
                            f"as a positive number"
                        )
                layers = entry["layers"]
            else:
                layers = entry

            # Each architecture should be a non-empty list of layer definitions
            if not isinstance(layers, list) or len(layers) == 0:
                raise ValueError(f"Architecture '{arch_name}' must be a non-empty list")

            # Validate each layer entry
            for i, layer in enumerate(layers):
                if "type" not in layer:
                    raise ValueError(f"Architecture '{arch_name}', layer {i} is missing 'type'")
                if "units" not in layer:
                    raise ValueError(f"Architecture '{arch_name}', layer {i} is missing 'units'")
                if "activation" not in layer:
                    raise ValueError(
                        f"Architecture '{arch_name}', layer {i} is missing 'activation'"
                    )

                # This project only supports dense layers
                if layer["type"].lower() != "dense":
                    raise ValueError(
                        f"Architecture '{arch_name}', layer {i} has unsupported type: {layer['type']}"
                    )

                # Validate that units is a positive integer
                if not isinstance(layer["units"], int) or layer["units"] < 1:
                    raise ValueError(
                        f"Architecture '{arch_name}', layer {i} must have "
                        f"'units' as a positive integer"
                    )

                # Validate that activation is supported
                activation_name = str(layer["activation"]).lower()
                if activation_name not in self.SUPPORTED_ACTIVATIONS:
                    raise ValueError(
                        f"Architecture '{arch_name}', layer {i} has unsupported activation: "
                        f"{layer['activation']}. Supported activations are: "
                        f"{sorted(self.SUPPORTED_ACTIVATIONS)}"
                    )

                # Optional: whether the Dense layer has a bias (default false)
                if "use_bias" in layer and not isinstance(layer["use_bias"], bool):
                    raise ValueError(
                        f"Architecture '{arch_name}', layer {i} must have "
                        f"'use_bias' as true or false"
                    )

                # Optional: batch normalization after the Dense layer (default false)
                if "batch_norm" in layer and not isinstance(layer["batch_norm"], bool):
                    raise ValueError(
                        f"Architecture '{arch_name}', layer {i} must have "
                        f"'batch_norm' as true or false"
                    )

                # Optional: dropout rate after the activation (default 0, no dropout)
                if "dropout" in layer:
                    rate = layer["dropout"]
                    if (
                        isinstance(rate, bool)
                        or not isinstance(rate, (int, float))
                        or not 0 <= rate < 1
                    ):
                        raise ValueError(
                            f"Architecture '{arch_name}', layer {i} must have "
                            f"'dropout' as a number in [0, 1)"
                        )
                    # Dropping values of the final prediction would make the
                    # output itself random, so dropout is for hidden layers only.
                    if rate > 0 and i == len(layers) - 1:
                        raise ValueError(
                            f"Architecture '{arch_name}': dropout is not allowed on the "
                            f"output layer (layer {i})"
                        )

                # Optional: weight initialization scheme (default "normal")
                if "init" in layer and str(layer["init"]).lower() not in SUPPORTED_INITS:
                    raise ValueError(
                        f"Architecture '{arch_name}', layer {i} has unsupported init: "
                        f"{layer['init']}. Supported schemes are: {sorted(SUPPORTED_INITS)}"
                    )

        # ----------------------------
        # Validate experiments block
        # ----------------------------

        experiments = config["experiments"]

        required_experiment_keys = [
            "optimizers",
            "learning_rates",
            "batch_sizes",
            "epochs",
            "early_stopping",
            "patience",
            "min_delta",
            "min_epochs_before_early_stop",
        ]

        # Ensure all required experiment keys exist
        for key in required_experiment_keys:
            if key not in experiments:
                raise ValueError(f"Missing required experiments key: '{key}'")

        # optimizers must be a non-empty list
        if not isinstance(experiments["optimizers"], list) or len(experiments["optimizers"]) == 0:
            raise ValueError("'optimizers' must be a non-empty list")

        # learning_rates must be a non-empty list
        if (
            not isinstance(experiments["learning_rates"], list)
            or len(experiments["learning_rates"]) == 0
        ):
            raise ValueError("'learning_rates' must be a non-empty list")

        # batch_sizes must be a non-empty list
        if (
            not isinstance(experiments["batch_sizes"], list)
            or len(experiments["batch_sizes"]) == 0
        ):
            raise ValueError("'batch_sizes' must be a non-empty list")

        # epochs must be a positive integer
        if not isinstance(experiments["epochs"], int) or experiments["epochs"] < 1:
            raise ValueError("'epochs' must be a positive integer")

        # early_stopping must be a boolean
        if not isinstance(experiments["early_stopping"], bool):
            raise ValueError("'early_stopping' must be true or false")

        # patience must be a positive integer
        if not isinstance(experiments["patience"], int) or experiments["patience"] < 1:
            raise ValueError("'patience' must be a positive integer")

        # min_delta must be numeric and non-negative
        if not isinstance(experiments["min_delta"], (int, float)) or experiments["min_delta"] < 0:
            raise ValueError("'min_delta' must be a non-negative number")

        # min_epochs_before_early_stop must be a non-negative integer
        min_epochs_before_early_stop = experiments["min_epochs_before_early_stop"]
        if (
            not isinstance(min_epochs_before_early_stop, int)
            or isinstance(min_epochs_before_early_stop, bool)
            or min_epochs_before_early_stop < 0
        ):
            raise ValueError("'min_epochs_before_early_stop' must be a non-negative integer")

        # Optional: seeds to repeat the whole sweep with (default [RANDOM_SEED]).
        # Must be a non-empty list of distinct, non-negative integers
        # (booleans are rejected even though Python treats them as ints).
        if "seeds" in experiments:
            seeds = experiments["seeds"]
            if (
                not isinstance(seeds, list)
                or len(seeds) == 0
                or not all(
                    isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds
                )
            ):
                raise ValueError("'seeds' must be a non-empty list of non-negative integers")
            if len(set(seeds)) != len(seeds):
                raise ValueError("'seeds' must not contain duplicates")

        # The minimum-epoch guard cannot exceed the configured maximum epochs
        if min_epochs_before_early_stop > experiments["epochs"]:
            raise ValueError("'min_epochs_before_early_stop' cannot be greater than 'epochs'")

        # If all checks pass, the config is valid
        return True

    @staticmethod
    def normalize_architectures(config: dict[str, Any]) -> dict[str, Any]:
        """
        Split architectures with options into a layer list and the options.

        Parameters
        ----------
        config : dict
            A validated config. Each architecture is either a list of layer
            dicts or a dict with "layers" (that list) and options such as
            "max_grad_norm".

        Returns
        -------
        dict
            The same config (modified in place): every
            config["architectures"][name] is a plain layer list, and
            config["architecture_options"] maps the names of architectures
            that had options to those options, e.g.
            {"A2-clip": {"max_grad_norm": 20.0}}.

        Notes
        -----
        Processing:
        1. For each architecture given as a dict, move its options into
           "architecture_options" and replace the entry with its layer
           list.

        Everything that builds networks (training, the benchmarks, the
        tests) reads config["architectures"][name] as a layer list, so the
        options are kept separately and only the training loop reads them.
        """
        options = {}
        for name, entry in config["architectures"].items():
            if isinstance(entry, dict):
                options[name] = {k: v for k, v in entry.items() if k != "layers"}
                config["architectures"][name] = entry["layers"]
        config["architecture_options"] = options
        return config

    def load_and_validate(self, filename: str) -> dict[str, Any]:
        """
        Load and validate a config file.

        Parameters
        ----------
        filename : str
            Name of the JSON config file inside the configs folder, for
            example "regression_experiments.json".

        Returns
        -------
        dict
            Loaded and validated config dictionary, with architectures
            normalized by normalize_architectures (layer lists, plus
            "architecture_options").

        Raises
        ------
        FileNotFoundError
            If the file does not exist (from `load`).
        ValueError
            If the config is invalid (from `validate`).

        Notes
        -----
        Processing:
        1. Read and parse the file with `load(filename)`.
        2. Check it with `validate(config)`.
        3. Normalize the architectures with `normalize_architectures` and
           return the config.
        """
        # First load the config from disk
        config = self.load(filename)

        # Then validate its structure and contents
        self.validate(config)

        # Split architecture options (e.g. a gradient-norm cap) from layers
        return self.normalize_architectures(config)
