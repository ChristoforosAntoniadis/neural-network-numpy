"""
Feed-forward neural network container.

Defines `NeuralNetwork`, which holds an ordered list of layers (Dense,
optional BatchNorm, activation, optional Dropout), runs the forward and
backward passes through them, and can build itself from an architecture
described in a config dictionary.

Input: an architecture (a list of layer dictionaries from a config), the
input dimension and a seed; then NumPy batches and loss gradients.

Output: predictions, the gradients stored on each layer, and the list of
trainable layers the optimizers update. Nothing is written to disk.
"""

from typing import Any

import numpy as np

from nn_numpy.config import RANDOM_SEED
from nn_numpy.nn.activations import get_activation
from nn_numpy.nn.layers import BatchNorm, Dense, Dropout


class NeuralNetwork:
    """
    A simple feed-forward neural network built as an ordered sequence
    of dense layers and activation functions.

    This class is responsible for:
    - storing the network layers
    - running the full forward pass
    - running the full backward pass
    - returning trainable layers for optimization
    - building the architecture from a configuration

    Attributes
    ----------
    layers : list
        Layer objects (Dense, BatchNorm, activation, Dropout) in the order
        the data flows through them.
    random : numpy.random.RandomState
        Generator shared by all Dense layers for weight initialization.
    dropout_random : numpy.random.RandomState
        Separate generator shared by all Dropout layers for their masks.

    Notes
    -----
    - The loss function is handled separately in losses.py
    - Parameter updates are handled separately in optimizers.py
    - The network owns a single random number generator for weight
      initialization, so all layer initializations are reproducible but
      not identical. Dropout uses a second, separate generator.
    """

    def __init__(self, random_seed: int = RANDOM_SEED) -> None:
        """
        Create an empty neural network with its random number generators.

        Parameters
        ----------
        random_seed : int, default=RANDOM_SEED
            Seed for the weight-initialization generator. The dropout
            generator is seeded with `random_seed + 1`.

        Returns
        -------
        None
            Sets `self.layers` to an empty list and creates
            `self.random` and `self.dropout_random`.

        Notes
        -----
        Processing:
        1. Start with an empty layer list.
        2. Create `self.random`, a `numpy.random.RandomState(random_seed)`
           that every Dense layer draws its initial weights from.
        3. Create `self.dropout_random`, a
           `numpy.random.RandomState(random_seed + 1)` that every Dropout
           layer draws its masks from.

        Why one shared generator for the weights: it gives reproducibility
        across runs (same seed, same weights) while still giving each layer
        different weight values, because each layer continues drawing from
        where the previous one stopped.
        """
        # Store all layers in the exact order they are added.
        # Example:
        # [Dense, ReLU, Dense, Sigmoid]
        # Any, not a union of the layer classes: the list is duck-typed. Every layer has
        # forward() / backward(), and train() / eval() set a training flag on all of them,
        # including the Dense and activation layers that never read it.
        self.layers: list[Any] = []

        # Create one seeded random number generator for the whole network.
        # This is passed to each Dense layer so weight initialization
        # is reproducible.
        self.random = np.random.RandomState(random_seed)

        # A second, separate generator shared by all Dropout layers.
        # Keeping dropout masks off the initialization generator means
        # adding dropout to an architecture leaves its initial weights
        # exactly as they were, so comparisons stay like for like.
        self.dropout_random = np.random.RandomState(random_seed + 1)

    def add_dense(
        self, input_dim: int, output_dim: int, use_bias: bool = False, init: str = "normal"
    ) -> None:
        """
        Add a dense (fully connected) layer to the network.

        Parameters
        ----------
        input_dim : int
            Number of input features to the layer.
        output_dim : int
            Number of neurons in the layer.
        use_bias : bool, default=False
            Whether the layer has a trainable bias vector.
        init : str, default="normal"
            Weight initialization scheme: "normal", "he" or "xavier".

        Returns
        -------
        None
            Appends a new `Dense` layer to `self.layers`.

        Notes
        -----
        Processing:
        1. Create a `Dense(input_dim, output_dim)` layer, passing the
           network's shared generator `self.random` plus `use_bias` and
           `init`. The layer draws its initial weights immediately.
        2. Append it to the end of `self.layers`.

        A Dense layer performs the linear transformation:

            Z = X @ W (+ b)

        where:
        - X has shape (batch_size, input_dim)
        - W has shape (input_dim, output_dim)
        - b has shape (1, output_dim), only when use_bias is True
        - Z has shape (batch_size, output_dim)
        """
        # Create a Dense layer and pass in the shared random generator.
        # This keeps initialization reproducible across runs.
        self.layers.append(
            Dense(input_dim, output_dim, random_state=self.random, use_bias=use_bias, init=init)
        )

    def add_batch_norm(self, num_features: int) -> None:
        """
        Add a batch-normalization layer to the network.

        Parameters
        ----------
        num_features : int
            Number of neurons to normalize, i.e. the output size of the
            Dense layer just before it.

        Returns
        -------
        None
            Appends a new `BatchNorm` layer to `self.layers`.

        Notes
        -----
        Processing:
        1. Create a `BatchNorm(num_features)` layer with its default
           momentum and epsilon.
        2. Append it to the end of `self.layers`.

        Batch norm is placed between a Dense layer and its activation:

            Dense -> BatchNorm -> ReLU

        so the activation always receives inputs with a stable scale.
        It draws no random numbers, so adding it does not change the
        weights initialized for any Dense layer.
        """
        self.layers.append(BatchNorm(num_features))

    def add_dropout(self, rate: float) -> None:
        """
        Add a dropout layer to the network.

        Parameters
        ----------
        rate : float
            Probability of dropping each value during training, in [0, 1).

        Returns
        -------
        None
            Appends a new `Dropout` layer to `self.layers`.

        Notes
        -----
        Processing:
        1. Create a `Dropout(rate)` layer that uses the network's dedicated
           dropout generator `self.dropout_random`.
        2. Append it to the end of `self.layers`.

        Dropout is placed after a hidden layer's activation:

            Dense -> (BatchNorm) -> ReLU -> Dropout

        so it randomly removes some of that layer's outputs before they
        reach the next layer. It uses the network's dedicated dropout
        generator, not the weight-initialization one.
        """
        self.layers.append(Dropout(rate, random_state=self.dropout_random))

    def train(self) -> None:
        """
        Put every layer in training mode.

        Parameters
        ----------
        None
            Uses `self.layers`.

        Returns
        -------
        None
            Sets `layer.training = True` on every layer in `self.layers`.

        Notes
        -----
        Processing:
        1. Loop over all layers and set their `training` attribute to True.

        Only layers whose behaviour differs between training and
        evaluation actually use the flag: BatchNorm uses batch statistics
        and updates its running averages, and Dropout drops random values.
        Dense and activation layers behave identically in both modes.
        """
        for layer in self.layers:
            layer.training = True

    def eval(self) -> None:
        """
        Put every layer in evaluation mode.

        Parameters
        ----------
        None
            Uses `self.layers`.

        Returns
        -------
        None
            Sets `layer.training = False` on every layer in `self.layers`.

        Notes
        -----
        Processing:
        1. Loop over all layers and set their `training` attribute to False.

        Used for validation, testing and prediction, so that the same input
        always gives the same output, independent of which other samples are
        in the batch. BatchNorm switches to its running statistics and stops
        updating them, and Dropout passes its input through unchanged.
        """
        for layer in self.layers:
            layer.training = False

    def add_activation(self, activation_name: str) -> None:
        """
        Add an activation layer to the network.

        Parameters
        ----------
        activation_name : str
            Name of the activation function: "relu", "sigmoid", "tanh" or
            "linear" (case-insensitive; see `nn_numpy.nn.activations.get_activation`).

        Returns
        -------
        None
            Appends the matching activation layer to `self.layers`.

        Raises
        ------
        ValueError
            Raised by `get_activation` when the name is not supported.

        Notes
        -----
        Processing:
        1. Ask `get_activation(activation_name)` for a new activation
           object (ReLU, Sigmoid, Tanh or Linear).
        2. Append it to the end of `self.layers`.

        Activation functions are placed after Dense layers to introduce
        nonlinearity. Without them, multiple Dense layers would collapse
        into one overall linear transformation.
        """
        self.layers.append(get_activation(activation_name))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Run a full forward pass through the network.

        Parameters
        ----------
        X : numpy.ndarray of shape (batch_size, input_dim), dtype float64
            Input data, one row per sample.

        Returns
        -------
        numpy.ndarray of shape (batch_size, output_dim), dtype float64
            Final network output after passing through all layers in order
            (output_dim is the number of units in the last layer, 1 in
            this project).

        Notes
        -----
        Processing:
        1. Start with `output = X`.
        2. For each layer in order, replace `output` with
           `layer.forward(output)`. Each layer caches what it needs for
           the backward pass.
        3. Return the final `output`.

        Behaviour depends on the mode set by `train()` / `eval()`
        (BatchNorm and Dropout act differently in each).

        If the layers are:

            Dense -> ReLU -> Dense -> Sigmoid

        then this method computes:

            X
            -> Dense.forward(X)
            -> ReLU.forward(...)
            -> Dense.forward(...)
            -> Sigmoid.forward(...)

        This is how the network produces predictions.
        """
        # Start with the raw input data.
        output = X

        # Pass the current output through each layer in sequence.
        # Each layer's output becomes the next layer's input.
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, grad_loss: np.ndarray) -> np.ndarray:
        """
        Run a full backward pass through the network.

        Parameters
        ----------
        grad_loss : numpy.ndarray of shape (batch_size, output_dim), dtype float64
            Gradient of the loss with respect to the network output
            (dL/dy_pred), as returned by the loss's `backward` method.

        Returns
        -------
        numpy.ndarray of shape (batch_size, input_dim), dtype float64
            Gradient with respect to the network input.
            This is returned mainly for completeness.

        Notes
        -----
        Processing:
        1. Start with `grad = grad_loss`.
        2. For each layer in reverse order, replace `grad` with
           `layer.backward(grad)`. Layers with parameters (Dense,
           BatchNorm) also store their parameter gradients, which the
           optimizer reads afterwards.
        3. Return the final `grad`.

        Must be called after `forward` on the same batch, because each
        layer uses the values it cached during the forward pass.

        Backpropagation works in reverse order.

        If the forward path was:

            Dense -> ReLU -> Dense -> Sigmoid

        then the backward path is:

            Sigmoid.backward(...)
            Dense.backward(...)
            ReLU.backward(...)
            Dense.backward(...)

        Each layer receives a gradient from the layer after it,
        computes its own contribution using the chain rule,
        and passes a new gradient to the layer before it.
        """
        # Start with the gradient coming from the loss function.
        grad = grad_loss

        # Move backward through the layers in reverse order.
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def get_trainable_layers(self) -> list[Dense | BatchNorm]:
        """
        Return the layers that contain trainable parameters.

        Parameters
        ----------
        None
            Uses `self.layers`.

        Returns
        -------
        list of Dense or BatchNorm
            Trainable layers, in the same order as in `self.layers`.

        Notes
        -----
        Processing:
        1. Keep every layer in `self.layers` that has both a `get_params`
           and a `get_grads` method; drop the rest.

        A layer is trainable if it exposes its parameters through
        get_params() and get_grads(). In this project these are the Dense
        layers (weights, and bias when enabled) and the BatchNorm layers
        (gamma and beta).

        Activation and Dropout layers do not have trainable parameters, so
        they are excluded.
        """
        trainable_layers = [
            layer
            for layer in self.layers
            if hasattr(layer, "get_params") and hasattr(layer, "get_grads")
        ]
        return trainable_layers

    def gradient_norm(self) -> float:
        """
        Compute the global L2 norm of all current gradients.

        Parameters
        ----------
        None
            Uses the gradients stored by the last backward pass.

        Returns
        -------
        float
            sqrt of the sum of the squared entries of every gradient array of
            every trainable layer (Dense weights and biases, BatchNorm gamma
            and beta). inf or NaN if a gradient has overflowed.

        Notes
        -----
        Processing:
        1. Collect the arrays from get_grads() of every trainable layer.
        2. Sum their squared entries and take the square root.

        Squaring an exploding gradient can overflow; that warning is
        suppressed and the result is inf, which callers treat as a
        non-finite norm.
        """
        with np.errstate(over="ignore", invalid="ignore"):
            total = sum(
                float(np.sum(grad * grad))
                for layer in self.get_trainable_layers()
                for grad in layer.get_grads().values()
            )
        return float(np.sqrt(total))

    def clip_gradients(self, max_norm: float) -> float:
        """
        Scale all gradients down so their global L2 norm is at most max_norm.

        Parameters
        ----------
        max_norm : float
            Largest allowed global gradient norm (> 0).

        Returns
        -------
        float
            The global gradient norm before clipping.

        Notes
        -----
        Processing:
        1. Compute the global norm with gradient_norm().
        2. If it is finite and larger than max_norm, multiply every gradient
           array in place by max_norm / norm. The optimizer then reads the
           scaled gradients.

        This is global-norm clipping (Pascanu et al., 2013): all gradients
        are scaled by the same factor, so the update keeps its direction and
        only its length is capped. Gradients below the cap are left
        untouched. A non-finite norm is left alone, so the trainer's
        divergence check still sees the run blow up.
        """
        norm = self.gradient_norm()
        if np.isfinite(norm) and norm > max_norm:
            scale = max_norm / norm
            for layer in self.get_trainable_layers():
                for grad in layer.get_grads().values():
                    grad *= scale
        return norm

    def build_from_config(self, config: dict[str, Any]) -> None:
        """
        Build the network automatically from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary describing the network architecture. Keys:
            "input_dimension" (int), the number of input features, and
            "layers" (list of dict), one entry per Dense layer. Each layer
            dict has "type" (str, must be "dense"), "units" (int) and
            "activation" (str), plus optional "use_bias" (bool, default
            False), "init" (str, default "normal"), "batch_norm" (bool,
            default False) and "dropout" (float, default 0.0).

            Expected format:
            {
                "input_dimension": 10,
                "layers": [
                    {"type": "dense", "units": 32, "activation": "relu",
                     "use_bias": true, "init": "he"},
                    {"type": "dense", "units": 16, "activation": "tanh"},
                    {"type": "dense", "units": 1, "activation": "linear"}
                ]
            }

        Returns
        -------
        None
            Appends the built layers to `self.layers`.

        Raises
        ------
        ValueError
            If a layer's "type" is not "dense" (case-insensitive), or an
            activation or init name is not supported.
        KeyError
            If a required key ("input_dimension", "layers", "type",
            "units", "activation") is missing.

        Notes
        -----
        Processing:
        1. Read "input_dimension" as the input size of the first layer.
        2. For each layer entry, in order:
           a. verify the layer type is "dense"
           b. add a Dense layer, with the optional "use_bias" (default
              False) and "init" (default "normal") settings
           c. if "batch_norm" is true (default False), add a BatchNorm layer
           d. add the specified activation layer
           e. if "dropout" is above 0 (default 0), add a Dropout layer
           f. use this layer's "units" as the next layer's input size

        This method does not validate the config beyond the layer type;
        `ConfigLoader.validate` is expected to have checked it already.

        This method lets the architecture be defined outside the code,
        usually in a config file. That makes experiments easier to run
        and compare.
        """
        # Read the input dimension of the first layer.
        current_input_dim = config["input_dimension"]

        # Read the list of layer specifications.
        layer_configs = config["layers"]

        # Build the network one layer at a time.
        for layer_config in layer_configs:
            # Read and normalize the layer type.
            layer_type = layer_config["type"].lower()

            # This project only supports Dense layers.
            if layer_type != "dense":
                raise ValueError(f"Unsupported layer type: {layer_type}")

            # Number of output units in this Dense layer.
            units = layer_config["units"]

            # Activation function to apply after this Dense layer.
            activation = layer_config["activation"]

            # Add the Dense layer. Bias and initialization are optional
            # settings; the defaults reproduce the original bias-free layer.
            self.add_dense(
                current_input_dim,
                units,
                use_bias=layer_config.get("use_bias", False),
                init=layer_config.get("init", "normal").lower(),
            )

            # Optional batch normalization between the Dense layer and
            # its activation.
            if layer_config.get("batch_norm", False):
                self.add_batch_norm(units)

            # Add the matching activation layer.
            self.add_activation(activation)

            # Optional dropout after the activation (hidden layers only;
            # the config loader rejects it on the output layer).
            if layer_config.get("dropout", 0.0) > 0.0:
                self.add_dropout(layer_config["dropout"])

            # The output size of this layer becomes the input size
            # of the next layer.
            current_input_dim = units

    def summary(self) -> None:
        """
        Print a simple summary of the network architecture.

        Parameters
        ----------
        None
            Uses `self.layers`.

        Returns
        -------
        None
            Prints one line per layer to standard output.

        Notes
        -----
        Processing:
        1. Print a header line.
        2. For each layer, print its 1-based position and class name.

        Example output:

            --- Network Architecture ---
            Layer 1: Dense
            Layer 2: ReLU
            Layer 3: Dense
            Layer 4: Sigmoid

        This is only a lightweight helper for inspection.
        """
        print("\n--- Network Architecture ---")
        for i, layer in enumerate(self.layers, start=1):
            print(f"Layer {i}: {layer.__class__.__name__}")
