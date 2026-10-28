"""
Trainable and regularization layers for the NumPy network.

This module defines the building blocks that sit between activations:

- Dense: a fully connected linear layer (Z = X @ W, optionally + b).
- BatchNorm: batch normalization with a learned scale and shift.
- Dropout: inverted dropout, active only in training mode.

Every layer exposes forward(X) and backward(grad_output). Layers with
trainable parameters also expose get_params(), get_grads() and
set_params(), which the optimizers and the early-stopping checkpoint use.

Input: NumPy batches of shape (batch_size, features) and upstream gradients.

Output: the layer outputs and input gradients, plus each layer's parameter
gradients stored as attributes. Nothing is written to disk.
"""

import numpy as np

# Weight initialization schemes supported by Dense.
SUPPORTED_INITS = {"normal", "he", "xavier"}


class Dense:
    """
    A fully connected (dense) layer with an optional bias term.

    This layer performs a linear transformation:

        Z = X @ W          (use_bias=False, the project's original layer)
        Z = X @ W + b      (use_bias=True)

    Attributes
    ----------
    input_dim : int
        Number of input features the layer expects.
    output_dim : int
        Number of output neurons.
    use_bias : bool
        Whether the layer has a trainable bias.
    init : str
        Name of the weight initialization scheme that was used.
    weights : numpy.ndarray of shape (input_dim, output_dim), dtype float64
        Trainable weight matrix W.
    bias : numpy.ndarray of shape (1, output_dim), dtype float64, or None
        Trainable bias b, or None when use_bias is False.
    input : numpy.ndarray of shape (batch_size, input_dim), dtype float64, or None
        Input batch cached by forward() for use in backward().
    dweights : numpy.ndarray of shape (input_dim, output_dim), dtype float64
        Gradient dL/dW from the last backward pass.
    dbias : numpy.ndarray of shape (1, output_dim), dtype float64, or None
        Gradient dL/db from the last backward pass, or None without a bias.

    Notes
    -----
    Shape convention used in this project:

    - X: (batch_size, input_dim)
    - W: (input_dim, output_dim)
    - b: (1, output_dim), broadcast across the batch
    - Z: (batch_size, output_dim)

    The formula is often written as z = W x + b, which usually assumes
    x is a column vector. In this implementation, each sample is stored as
    a row, so the equivalent NumPy operation is X @ W + b.

    Why a bias matters: without a bias, every neuron computes a function
    that passes through the origin, so an all-zero input always produces
    an all-zero pre-activation. A bias lets each neuron shift its
    activation threshold. For ReLU this is what lets a neuron stay active
    (or become active again) when its weighted input is negative.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        random_state: np.random.RandomState | None = None,
        use_bias: bool = False,
        init: str = "normal",
    ) -> None:
        """
        Create a dense layer and initialize its trainable parameters.

        Parameters
        ----------
        input_dim : int
            Number of input features entering the layer.
        output_dim : int
            Number of output neurons produced by the layer.
        random_state : numpy.random.RandomState or None, default=None
            Optional random generator for reproducible weight initialization.
            If None, NumPy's global random generator (the numpy.random
            module) is used.
        use_bias : bool, default=False
            Whether the layer has a trainable bias vector.
            False reproduces the project's original bias-free layer.
        init : str, default="normal"
            Weight initialization scheme:

            - "normal": N(0, 0.1^2), the project's original scheme
            - "he":     N(0, 2 / input_dim), designed for ReLU layers
            - "xavier": N(0, 2 / (input_dim + output_dim)),
              designed for sigmoid / tanh layers

        Returns
        -------
        None
            Sets input_dim, output_dim, use_bias, init, weights, bias,
            input, dweights and dbias on the new object.

        Raises
        ------
        ValueError
            If init is not one of the supported schemes (raised by
            _init_std).

        Notes
        -----
        Processing:

        1. Store the layer dimensions and options.
        2. Pick the random generator: random_state if given, otherwise
           the numpy.random module.
        3. Draw the weight matrix of shape (input_dim, output_dim) from a
           normal distribution with mean 0 and the standard deviation
           returned by _init_std.
        4. Create the bias as zeros of shape (1, output_dim) if use_bias
           is True, otherwise set it to None. No random numbers are used,
           so turning the bias on does not change any drawn weights.
        5. Create zero-filled gradient arrays with the same shapes as the
           parameters, and set the cached input to None.
        """

        # Save the layer dimensions so the object remembers:
        # - how many values it expects as input
        # - how many values it should produce as output
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.init = init

        # Choose which random number generator to use.
        # If a seeded generator is passed in, use it for reproducibility.
        # Otherwise, use NumPy's default random generator.
        rng = np.random if random_state is None else random_state

        # Initialize weights from a zero-mean normal distribution.
        # Only the standard deviation depends on the chosen scheme.
        #
        # Weight matrix shape:
        # (input_dim, output_dim)
        #
        # Example:
        # if input_dim = 4 and output_dim = 3,
        # then weights.shape = (4, 3)
        self.weights = rng.normal(
            loc=0.0,
            scale=self._init_std(init, input_dim, output_dim),
            size=(self.input_dim, self.output_dim),
        )

        # The bias starts at zero. Zero is the standard choice because the
        # random weights already break the symmetry between neurons.
        # Creating it uses no random numbers, so enabling the bias does not
        # change the weights drawn for this or any later layer.
        self.bias = np.zeros((1, self.output_dim)) if use_bias else None

        # This will store the input batch from the forward pass.
        # We need it later during backpropagation to compute dL/dW.
        self.input: np.ndarray = None  # type: ignore[assignment]  # set in forward()

        # Gradients of the loss with respect to the parameters:
        # dL/dW and dL/db. They have the same shapes as the parameters.
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias) if use_bias else None

    @staticmethod
    def _init_std(init: str, input_dim: int, output_dim: int) -> float:
        """
        Return the standard deviation used to draw the initial weights.

        Parameters
        ----------
        init : str
            Initialization scheme: "normal", "he" or "xavier".
        input_dim : int
            Fan-in: number of inputs to each neuron.
        output_dim : int
            Fan-out: number of neurons in the layer.

        Returns
        -------
        float
            Standard deviation of the zero-mean normal distribution the
            weights are drawn from.

        Raises
        ------
        ValueError
            If init is not in SUPPORTED_INITS.

        Notes
        -----
        Processing:

        1. "normal" returns the fixed value 0.1.
        2. "he" returns sqrt(2 / input_dim).
        3. "xavier" returns sqrt(2 / (input_dim + output_dim)).
        4. Any other name raises ValueError.

        Why the scale matters: each pre-activation sums input_dim terms.
        If the weights are too small, signals and gradients shrink layer
        after layer; if too large, they grow. Scaling the variance by the
        layer's fan-in keeps the signal's variance roughly constant
        through the network.

        - He (Kaiming) init uses variance 2 / fan_in. The factor 2
          compensates for ReLU zeroing out about half of its inputs.
        - Xavier (Glorot) init uses variance 2 / (fan_in + fan_out), a
          compromise that keeps both the forward signal and the backward
          gradient stable for symmetric activations such as tanh and
          sigmoid.
        - "normal" is the project's original fixed scale of 0.1, which
          ignores the layer size.
        """
        if init == "normal":
            # No recorded justification for 0.1; it is kept so the original
            # results stay reproducible, and A2-he / A2-bias-he test the
            # size-aware alternative.
            return 0.1
        if init == "he":
            return np.sqrt(2.0 / input_dim)
        if init == "xavier":
            return np.sqrt(2.0 / (input_dim + output_dim))
        raise ValueError(
            f"Unsupported init: {init}. Supported schemes are: {sorted(SUPPORTED_INITS)}"
        )

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the linear transformation to a batch of inputs.

        Parameters
        ----------
        X : numpy.ndarray of shape (batch_size, input_dim), dtype float64
            Input batch, one sample per row.

        Returns
        -------
        numpy.ndarray of shape (batch_size, output_dim), dtype float64
            Pre-activations Z, one row per sample.

        Notes
        -----
        Processing:

        1. Cache X in self.input for the backward pass.
        2. Compute Z = X @ W.
        3. If the layer uses a bias, add the (1, output_dim) bias to
           every row.

        Math: for a batch of inputs, Z = X @ W (+ b).
        """

        # Store the input so it is available during the backward pass.
        self.input = X

        # Apply the linear transformation.
        #
        # The @ operator means matrix multiplication.
        # This is equivalent to:
        #   np.matmul(X, self.weights)
        # or, for 2D arrays:
        #   np.dot(X, self.weights)
        Z = X @ self.weights

        # The (1, output_dim) bias is broadcast and added to every row,
        # so each sample in the batch gets the same shift.
        if self.use_bias:
            Z = Z + self.bias

        return Z

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the parameter gradients and the gradient for the previous layer.

        Parameters
        ----------
        grad_output : numpy.ndarray of shape (batch_size, output_dim), dtype float64
            Gradient of the loss with respect to this layer's output, dL/dZ.

        Returns
        -------
        numpy.ndarray of shape (batch_size, input_dim), dtype float64
            Gradient of the loss with respect to this layer's input, dL/dX.

        Notes
        -----
        Processing:

        1. Compute dL/dW = X^T @ dL/dZ using the input cached by
           forward(), and store it in self.dweights.
        2. If the layer uses a bias, compute dL/db by summing dL/dZ over
           the batch and store it in self.dbias.
        3. Compute and return dL/dX = dL/dZ @ W^T.

        Math: let Z = X @ W + b and grad_output = dL/dZ. Then

            dL/dW = X^T @ (dL/dZ)
            dL/db = sum over the batch of dL/dZ
            dL/dX = (dL/dZ) @ W^T

        The bias gradient is a sum because the same b is added to every
        sample: each sample's dL/dZ contributes to it.

        forward() must have been called first, because self.input is used.
        """

        # Compute gradient with respect to the weights.
        #
        # Shape check:
        # self.input.T  -> (input_dim, batch_size)
        # grad_output   -> (batch_size, output_dim)
        # result        -> (input_dim, output_dim)
        #
        # This matches the shape of self.weights.
        self.dweights = self.input.T @ grad_output

        # Compute gradient with respect to the bias.
        #
        # Shape check:
        # grad_output -> (batch_size, output_dim)
        # result      -> (1, output_dim), matching self.bias
        if self.use_bias:
            self.dbias = np.sum(grad_output, axis=0, keepdims=True)

        # Compute gradient with respect to the input.
        # The bias does not depend on X, so it does not appear here.
        #
        # Shape check:
        # grad_output    -> (batch_size, output_dim)
        # self.weights.T -> (output_dim, input_dim)
        # result         -> (batch_size, input_dim)
        #
        # This tells the previous layer how the loss changes with respect
        # to its outputs.
        grad_input = grad_output @ self.weights.T

        return grad_input

    def get_params(self) -> dict[str, np.ndarray]:
        """
        Return the trainable parameters of the layer.

        Parameters
        ----------
        None
            Uses self.weights, self.bias and self.use_bias.

        Returns
        -------
        dict of str to numpy.ndarray
            "weights" of shape (input_dim, output_dim), plus "bias" of
            shape (1, output_dim) when the layer uses one, both float64.
            The arrays are the layer's own (not copies), so updating them
            in place updates the layer.

        Notes
        -----
        Processing:

        1. Put self.weights under the key "weights".
        2. If use_bias is True, add self.bias under the key "bias".
        """
        params = {"weights": self.weights}
        if self.use_bias:
            params["bias"] = self.bias
        return params

    def get_grads(self) -> dict[str, np.ndarray]:
        """
        Return the gradients of the trainable parameters.

        Parameters
        ----------
        None
            Uses self.dweights, self.dbias and self.use_bias.

        Returns
        -------
        dict of str to numpy.ndarray
            Gradients with the same keys and shapes as get_params():
            "weights" (input_dim, output_dim) and, with a bias,
            "bias" (1, output_dim), both float64.

        Notes
        -----
        Processing:

        1. Put self.dweights under the key "weights".
        2. If use_bias is True, add self.dbias under the key "bias".

        The values are the ones computed by the most recent backward().
        """
        grads = {"weights": self.dweights}
        if self.use_bias:
            grads["bias"] = self.dbias
        return grads

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """
        Replace the trainable parameters with copies of the given arrays.

        Parameters
        ----------
        params : dict of str to numpy.ndarray
            Arrays with the same keys and shapes as get_params():
            "weights" (input_dim, output_dim) and, if the layer uses a
            bias, "bias" (1, output_dim). For example a checkpoint saved
            during early stopping.

        Returns
        -------
        None
            Replaces self.weights and, if use_bias is True, self.bias.

        Raises
        ------
        KeyError
            If a required key ("weights", or "bias" when use_bias is True)
            is missing from params.

        Notes
        -----
        Processing:

        1. Copy params["weights"] into self.weights.
        2. If use_bias is True, copy params["bias"] into self.bias.

        Copies are stored so that later in-place optimizer updates do not
        modify the checkpoint the arrays came from.
        """
        self.weights = params["weights"].copy()
        if self.use_bias:
            self.bias = params["bias"].copy()


class BatchNorm:
    """
    Batch normalization for dense layers (Ioffe & Szegedy, 2015).

    Placed between a Dense layer and its activation, it rescales each
    neuron's pre-activation so that, across the current mini-batch, it has
    mean 0 and variance 1, then applies a learned scale and shift:

        x_hat = (X - mean) / sqrt(var + eps)
        Y     = gamma * x_hat + beta

    Attributes
    ----------
    num_features : int
        Number of neurons (columns) being normalized.
    momentum : float
        Weight of the current batch in the running-statistics update.
    eps : float
        Constant added to the variance before the square root.
    gamma : numpy.ndarray of shape (1, num_features), dtype float64
        Trainable scale, initialized to ones.
    beta : numpy.ndarray of shape (1, num_features), dtype float64
        Trainable shift, initialized to zeros.
    dgamma, dbeta : numpy.ndarray of shape (1, num_features), dtype float64
        Gradients of gamma and beta from the last backward pass.
    running_mean : numpy.ndarray of shape (1, num_features), dtype float64
        Moving average of batch means, used in evaluation mode.
    running_var : numpy.ndarray of shape (1, num_features), dtype float64
        Moving average of batch variances, used in evaluation mode.
    training : bool
        True in training mode, False in evaluation mode.
    x_hat : numpy.ndarray of shape (batch_size, num_features), dtype float64, or None
        Normalized input cached by forward().
    inv_std : numpy.ndarray of shape (1, num_features), dtype float64, or None
        1 / sqrt(var + eps) cached by forward().
    cached_training : bool or None
        Value of training at the time of the last forward() call.

    Notes
    -----
    Shape convention:

    - X, Y:          (batch_size, num_features)
    - gamma, beta:   (1, num_features), trainable
    - running stats: (1, num_features), not trainable

    Training vs evaluation: the layer behaves differently depending on
    self.training.

    - Training: mean and var come from the current batch, so every
      layer receives inputs on a consistent scale however much the
      earlier weights change. The layer also updates exponential
      moving averages of the batch statistics.
    - Evaluation: the moving averages are used instead. A prediction then
      depends only on its own input, and works for a single sample, whose
      own "batch variance" would be zero.

    The network's train() / eval() methods set the flag on every layer.

    Why a Dense bias is redundant before batch norm: subtracting the batch
    mean removes any constant added by the Dense layer's bias, so its
    gradient would always be zero. beta takes over the bias's job of
    shifting the activation.
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5) -> None:
        """
        Create a batch-normalization layer.

        Parameters
        ----------
        num_features : int
            Number of neurons being normalized (the previous layer's output size).
        momentum : float, default=0.1
            Weight given to the current batch when updating the running
            statistics: running = (1 - momentum) * running + momentum * batch.
            0.1 is PyTorch's BatchNorm1d default, which uses the same
            convention.
        eps : float, default=1e-5
            Small constant added to the variance for numerical stability.
            1e-5 is PyTorch's BatchNorm1d default.

        Returns
        -------
        None
            Sets the hyperparameters, gamma, beta, their gradients, the
            running statistics, the training flag and the forward caches.

        Notes
        -----
        Processing:

        1. Store num_features, momentum and eps.
        2. Create gamma as ones and beta as zeros, shape (1, num_features),
           plus zero gradient arrays of the same shape.
        3. Create running_mean as zeros and running_var as ones, shape
           (1, num_features).
        4. Start in training mode and set the forward caches to None.
        """
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Trainable scale and shift. Starting at gamma = 1, beta = 0 means the
        # layer initially outputs exactly the normalized values.
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        # Running statistics used in evaluation mode. They start at the
        # identity transform (mean 0, variance 1).
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        # Layers start in training mode; NeuralNetwork.train() / eval() switch it.
        self.training = True

        # Values cached by forward() for the backward pass.
        self.x_hat: np.ndarray = None  # type: ignore[assignment]  # set in forward()
        self.inv_std: np.ndarray = None  # type: ignore[assignment]  # set in forward()
        # None until the first forward pass; backward() is only called after one.
        self.cached_training: bool | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize, scale and shift a batch.

        Parameters
        ----------
        X : numpy.ndarray of shape (batch_size, num_features), dtype float64
            Input batch, usually the pre-activations of a Dense layer.

        Returns
        -------
        numpy.ndarray of shape (batch_size, num_features), dtype float64
            gamma * x_hat + beta.

        Notes
        -----
        Processing:

        1. In training mode, compute the per-feature mean and variance of
           X over the batch (axis 0) and update running_mean and
           running_var as exponential moving averages with weight
           momentum.
        2. In evaluation mode, use running_mean and running_var instead.
        3. Compute inv_std = 1 / sqrt(var + eps) and
           x_hat = (X - mean) * inv_std, and cache both together with the
           current training flag for backward().
        4. Return gamma * x_hat + beta.

        The variance is the biased (population) variance, np.var with
        ddof=0, for both normalization and the running average.
        """
        if self.training:
            # Statistics of the current batch, per feature.
            mean = X.mean(axis=0, keepdims=True)
            var = X.var(axis=0, keepdims=True)

            # Update the running statistics for evaluation mode.
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # Evaluation: use the statistics accumulated during training.
            mean = self.running_mean
            var = self.running_var

        # Cache what backward() needs.
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.x_hat = (X - mean) * self.inv_std
        self.cached_training = self.training

        return self.gamma * self.x_hat + self.beta

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backpropagate through batch normalization.

        Parameters
        ----------
        grad_output : numpy.ndarray of shape (batch_size, num_features), dtype float64
            Gradient of the loss with respect to this layer's output, dL/dY.

        Returns
        -------
        numpy.ndarray of shape (batch_size, num_features), dtype float64
            Gradient of the loss with respect to this layer's input, dL/dX.

        Notes
        -----
        Processing:

        1. Compute dgamma and dbeta by summing over the batch and store
           them in self.dgamma and self.dbeta.
        2. Compute dx_hat = dL/dY * gamma.
        3. If the last forward() ran in evaluation mode, return
           dx_hat * inv_std.
        4. Otherwise return the full training-mode gradient below, which
           includes the paths through the batch mean and variance.

        Math: the parameter gradients sum over the batch, because gamma
        and beta are shared by every sample:

            dL/dgamma = sum_i dL/dY_i * x_hat_i
            dL/dbeta  = sum_i dL/dY_i

        In training mode, mean and var are themselves functions of every
        sample in the batch, so each input affects every output. Applying
        the chain rule through x_hat, var and mean and simplifying gives:

            dx_hat = dL/dY * gamma
            dL/dX  = (inv_std / N) * (N * dx_hat
                                      - sum(dx_hat)
                                      - x_hat * sum(dx_hat * x_hat))

        The two subtracted terms are the gradient flowing back through the
        batch mean and the batch variance. In evaluation mode the
        statistics are constants, so dL/dX is simply dx_hat * inv_std.

        The branch is chosen by cached_training (the mode used in the last
        forward()), not by the current training flag, so the gradient
        always matches the forward computation it differentiates.
        """
        # Parameter gradients: sum over the batch, shape (1, num_features).
        self.dgamma = np.sum(grad_output * self.x_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(grad_output, axis=0, keepdims=True)

        # Gradient with respect to the normalized input x_hat.
        dx_hat = grad_output * self.gamma

        # Evaluation mode: mean and var were constants, so only the
        # division by the standard deviation remains.
        if not self.cached_training:
            return dx_hat * self.inv_std

        # Training mode: include the gradient flowing through the batch
        # mean and batch variance (formula in the docstring).
        n = grad_output.shape[0]
        return (self.inv_std / n) * (
            n * dx_hat
            - np.sum(dx_hat, axis=0, keepdims=True)
            - self.x_hat * np.sum(dx_hat * self.x_hat, axis=0, keepdims=True)
        )

    def get_params(self) -> dict[str, np.ndarray]:
        """
        Return the trainable parameters: gamma (scale) and beta (shift).

        Parameters
        ----------
        None
            Uses self.gamma and self.beta.

        Returns
        -------
        dict of str to numpy.ndarray
            {"gamma": gamma, "beta": beta}, each of shape
            (1, num_features), dtype float64. The arrays are the layer's
            own, so updating them in place updates the layer.

        Notes
        -----
        Processing:

        1. Build and return a dict referencing self.gamma and self.beta.
        """
        return {"gamma": self.gamma, "beta": self.beta}

    def get_grads(self) -> dict[str, np.ndarray]:
        """
        Return the gradients of gamma and beta.

        Parameters
        ----------
        None
            Uses self.dgamma and self.dbeta.

        Returns
        -------
        dict of str to numpy.ndarray
            {"gamma": dgamma, "beta": dbeta}, each of shape
            (1, num_features), dtype float64, with the same keys as
            get_params().

        Notes
        -----
        Processing:

        1. Build and return a dict referencing the gradients computed by
           the most recent backward().
        """
        return {"gamma": self.dgamma, "beta": self.dbeta}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """
        Replace gamma and beta with copies of the given arrays.

        Parameters
        ----------
        params : dict of str to numpy.ndarray
            Must contain "gamma" and "beta", each of shape
            (1, num_features), dtype float64.

        Returns
        -------
        None
            Replaces self.gamma and self.beta.

        Raises
        ------
        KeyError
            If "gamma" or "beta" is missing from params.

        Notes
        -----
        Processing:

        1. Copy params["gamma"] into self.gamma.
        2. Copy params["beta"] into self.beta.
        """
        self.gamma = params["gamma"].copy()
        self.beta = params["beta"].copy()

    def get_buffers(self) -> dict[str, np.ndarray]:
        """
        Return the non-trainable state: the running statistics.

        Parameters
        ----------
        None
            Uses self.running_mean and self.running_var.

        Returns
        -------
        dict of str to numpy.ndarray
            {"running_mean": ..., "running_var": ...}, each of shape
            (1, num_features), dtype float64. The arrays are the layer's
            own, not copies.

        Notes
        -----
        Processing:

        1. Build and return a dict referencing the running statistics.

        They are not updated by the optimizer, but they are part of the
        model, so early-stopping checkpoints must save and restore them
        together with gamma and beta.
        """
        return {"running_mean": self.running_mean, "running_var": self.running_var}

    def set_buffers(self, buffers: dict[str, np.ndarray]) -> None:
        """
        Replace the running statistics with copies of the given arrays.

        Parameters
        ----------
        buffers : dict of str to numpy.ndarray
            Must contain "running_mean" and "running_var", each of shape
            (1, num_features), dtype float64, for example from
            get_buffers() at an early-stopping checkpoint.

        Returns
        -------
        None
            Replaces self.running_mean and self.running_var.

        Raises
        ------
        KeyError
            If "running_mean" or "running_var" is missing from buffers.

        Notes
        -----
        Processing:

        1. Copy buffers["running_mean"] into self.running_mean.
        2. Copy buffers["running_var"] into self.running_var.
        """
        self.running_mean = buffers["running_mean"].copy()
        self.running_var = buffers["running_var"].copy()


class Dropout:
    """
    Inverted dropout (Srivastava et al., 2014).

    During training, each input value is set to zero with probability
    `rate`, using a fresh random mask for every batch. The values that
    survive are divided by (1 - rate):

        mask = (random >= rate) / (1 - rate)
        Y    = X * mask

    Attributes
    ----------
    rate : float
        Probability of dropping each value, in [0, 1).
    random : numpy.random.RandomState or module numpy.random
        Generator used to draw the masks.
    training : bool
        True in training mode, False in evaluation mode.
    mask : numpy.ndarray of shape (batch_size, num_features), dtype float64, or None
        Scaled mask from the last training-mode forward pass (entries are
        0 or 1 / (1 - rate)), or None when the layer passed X through.

    Notes
    -----
    Why it regularizes: a neuron can't rely on any particular neighbour
    being present, so the network has to spread what it learns across
    many neurons instead of building fragile co-adapted features. Each
    batch effectively trains a different thinned sub-network.

    Why the survivors are scaled up ("inverted" dropout): with rate 0.2,
    only about 80% of the values reach the next layer, so its input sum
    would be about 80% of what it sees when nothing is dropped. Dividing
    the survivors by 0.8 keeps the expected value the same, so evaluation
    can use the plain network with no adjustment.

    Training vs evaluation:

    - Training: random mask, as above.
    - Evaluation: the layer does nothing (Y = X), so predictions are
      deterministic.

    The network's train() / eval() methods set the flag on every layer.
    """

    def __init__(self, rate: float, random_state: np.random.RandomState | None = None) -> None:
        """
        Create a dropout layer.

        Parameters
        ----------
        rate : float
            Probability of dropping each value, in [0, 1).
        random_state : numpy.random.RandomState or None, default=None
            Random generator for the masks. If None, the numpy.random
            module is used. The network passes a generator separate from
            the one used for weight initialization, so adding dropout
            doesn't change any layer's initial weights.

        Returns
        -------
        None
            Sets self.rate, self.random, self.training (True) and
            self.mask (None).

        Raises
        ------
        ValueError
            If rate is not in the interval [0, 1).

        Notes
        -----
        Processing:

        1. Check that 0 <= rate < 1. A rate of 1 would drop everything
           and divide by zero when scaling.
        2. Store the rate and the random generator.
        3. Start in training mode with no mask.
        """
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate}")

        self.rate = rate
        self.random = np.random if random_state is None else random_state

        # Layers start in training mode; NeuralNetwork.train() / eval() switch it.
        self.training = True

        # Mask from the last training-mode forward pass, reused by backward().
        self.mask: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dropout in training mode; pass the input through in evaluation mode.

        Parameters
        ----------
        X : numpy.ndarray of shape (batch_size, num_features), dtype float64
            Input batch, usually the output of an activation.

        Returns
        -------
        numpy.ndarray of shape (batch_size, num_features), dtype float64
            X * mask in training mode; X itself (the same object) in
            evaluation mode or when rate is 0.

        Notes
        -----
        Processing:

        1. If the layer is in evaluation mode or rate is 0, set mask to
           None and return X unchanged.
        2. Otherwise draw uniform random numbers in [0, 1) with the shape
           of X and keep the positions where the number is >= rate
           (probability 1 - rate).
        3. Build mask = keep / (1 - rate), so kept values are scaled up
           and dropped values are 0, and store it for backward().
        4. Return X * mask.
        """
        if not self.training or self.rate == 0.0:
            self.mask = None
            return X

        # Keep each value with probability (1 - rate), and scale the
        # survivors so the expected output equals the input.
        keep = self.random.random_sample(X.shape) >= self.rate
        self.mask = keep / (1.0 - self.rate)
        return X * self.mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backpropagate through dropout.

        Parameters
        ----------
        grad_output : numpy.ndarray of shape (batch_size, num_features), dtype float64
            Gradient of the loss with respect to this layer's output, dL/dY.

        Returns
        -------
        numpy.ndarray of shape (batch_size, num_features), dtype float64
            Gradient of the loss with respect to this layer's input, dL/dX.

        Notes
        -----
        Processing:

        1. If no mask was stored (the last forward() passed X through),
           return grad_output unchanged.
        2. Otherwise return grad_output * mask.

        Y = X * mask is element-wise, so dL/dX = dL/dY * mask: dropped
        values receive no gradient, and surviving ones are scaled by the
        same 1 / (1 - rate) as in the forward pass. In evaluation mode the
        layer is the identity, so the gradient passes through unchanged.
        """
        if self.mask is None:
            return grad_output
        return grad_output * self.mask
