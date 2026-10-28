"""
Element-wise activation functions and a factory to create them by name.

Each activation class has forward(x), which applies the function to every
element, and backward(grad_output), which multiplies the incoming gradient
by the function's derivative (chain rule). Activations have no trainable
parameters; they only cache what backward() needs from forward().

Input: NumPy arrays from the previous layer (forward) and upstream gradients
(backward).

Output: arrays of the same shape. Nothing is written to disk.
"""

import numpy as np

# --------------------
# ReLU
# --------------------


class ReLU:
    """
    Rectified Linear Unit activation.

    Attributes
    ----------
    input : numpy.ndarray or None
        Input cached by forward(), same shape as the last x.

    Notes
    -----
    Forward:
        f(x) = max(0, x)

    Backward:
        f'(x) = 1 if x > 0, else 0

    Meaning:

    - Positive inputs stay unchanged
    - Zero and negative inputs become 0
    """

    def __init__(self) -> None:
        """
        Create a ReLU activation with an empty input cache.

        Parameters
        ----------
        None
            Takes no arguments.

        Returns
        -------
        None
            Sets self.input to None.

        Notes
        -----
        Processing:

        1. Initialize the input cache used by backward().
        """
        # We store the input from the forward pass because
        # the backward pass needs to know which values were
        # positive and which were not.
        self.input: np.ndarray = None  # type: ignore[assignment]  # set in forward()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply ReLU elementwise.

        Parameters
        ----------
        x : numpy.ndarray of any shape, typically (batch_size, num_features), dtype float64
            Pre-activations, usually the output of a Dense or BatchNorm layer.

        Returns
        -------
        numpy.ndarray of the same shape as x, dtype float64
            max(0, x) for each element.

        Notes
        -----
        Processing:

        1. Cache x in self.input for the backward pass.
        2. Return np.maximum(0, x).
        """
        # Save input for use in backward pass
        self.input = x

        # np.maximum compares each element with 0
        # and keeps the larger value
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through ReLU.

        Parameters
        ----------
        grad_output : numpy.ndarray of the same shape as the forward input, dtype float64
            Gradient of the loss with respect to the ReLU output, coming
            from the next layer.

        Returns
        -------
        numpy.ndarray of the same shape as grad_output, dtype float64
            Gradient of the loss with respect to the ReLU input.

        Notes
        -----
        Processing:

        1. Copy grad_output (so the caller's array is not modified).
        2. Set the copy to 0 wherever the cached input was <= 0.
        3. Return the copy.

        ReLU derivative:

        - 1 where input > 0
        - 0 where input <= 0 (the derivative at exactly 0 is taken as 0)

        So we pass the gradient through only where the original input was
        positive.
        """
        # Start with a copy of the incoming gradient
        grad_input = grad_output.copy()

        # Wherever the original input was <= 0,
        # the ReLU derivative is 0, so block the gradient
        grad_input[self.input <= 0] = 0

        return grad_input


# --------------------
# Sigmoid
# --------------------


class Sigmoid:
    """
    Sigmoid activation.

    Attributes
    ----------
    output : numpy.ndarray or None
        Sigmoid output cached by forward(), same shape as the last x.

    Notes
    -----
    Forward:
        f(x) = 1 / (1 + exp(-x))

    Backward:
        f'(x) = sigmoid(x) * (1 - sigmoid(x))

    Meaning:

    - Converts values into the range (0, 1)
    - Commonly used in binary classification output layers
    """

    def __init__(self) -> None:
        """
        Create a sigmoid activation with an empty output cache.

        Parameters
        ----------
        None
            Takes no arguments.

        Returns
        -------
        None
            Sets self.output to None.

        Notes
        -----
        Processing:

        1. Initialize the output cache used by backward().
        """
        # Store the sigmoid output from the forward pass.
        # This is useful because the derivative can be written as:
        # sigmoid(x) * (1 - sigmoid(x))
        self.output: np.ndarray = None  # type: ignore[assignment]  # set in forward()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid elementwise.

        Parameters
        ----------
        x : numpy.ndarray of any shape, typically (batch_size, num_features), dtype float64
            Pre-activations, usually the output of a Dense or BatchNorm layer.

        Returns
        -------
        numpy.ndarray of the same shape as x, dtype float64
            1 / (1 + exp(-x)) for each element, in the range [0, 1]
            (values strictly between 0 and 1 except for extreme inputs).

        Notes
        -----
        Processing:

        1. Clip x to [-500, 500] so exp(-x) cannot overflow.
        2. Compute 1 / (1 + exp(-x_clipped)).
        3. Cache the result in self.output for the backward pass and
           return it.
        """
        # Clip very large or very small values to avoid numerical overflow
        # inside exp(). This does not change the intended behavior in a
        # meaningful way, but makes computation safer.
        # Why 500: exp() overflows float64 above about 709.8, and exp(500) is
        # about 1.4e217, safely below that. sigmoid(500) already rounds to 1.0
        # and sigmoid(-500) is about 7e-218, so the clip changes nothing that
        # float64 can represent in the output.
        x_clipped = np.clip(x, -500, 500)

        # Apply the sigmoid formula
        self.output = 1 / (1 + np.exp(-x_clipped))

        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through sigmoid.

        Parameters
        ----------
        grad_output : numpy.ndarray of the same shape as the forward input, dtype float64
            Gradient of the loss with respect to the sigmoid output, coming
            from the next layer (or from the loss).

        Returns
        -------
        numpy.ndarray of the same shape as grad_output, dtype float64
            Gradient of the loss with respect to the sigmoid input.

        Notes
        -----
        Processing:

        1. Read the cached output s from forward().
        2. Return grad_output * s * (1 - s).

        If s = sigmoid(x), then:
            ds/dx = s * (1 - s)

        By the chain rule:
            grad_input = grad_output * ds/dx
        """
        return grad_output * self.output * (1 - self.output)


# --------------------
# Tanh
# --------------------


class Tanh:
    """
    Hyperbolic tangent activation.

    Attributes
    ----------
    output : numpy.ndarray or None
        tanh output cached by forward(), same shape as the last x.

    Notes
    -----
    Forward:
        f(x) = tanh(x)

    Backward:
        f'(x) = 1 - tanh(x)^2

    Meaning:

    - Maps inputs into the range (-1, 1)
    - Often used in hidden layers
    """

    def __init__(self) -> None:
        """
        Create a tanh activation with an empty output cache.

        Parameters
        ----------
        None
            Takes no arguments.

        Returns
        -------
        None
            Sets self.output to None.

        Notes
        -----
        Processing:

        1. Initialize the output cache used by backward().
        """
        # Store tanh output during forward pass.
        # This makes the derivative easy to compute later.
        self.output: np.ndarray = None  # type: ignore[assignment]  # set in forward()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply tanh elementwise.

        Parameters
        ----------
        x : numpy.ndarray of any shape, typically (batch_size, num_features), dtype float64
            Pre-activations, usually the output of a Dense or BatchNorm layer.

        Returns
        -------
        numpy.ndarray of the same shape as x, dtype float64
            tanh(x) for each element, in the range [-1, 1].

        Notes
        -----
        Processing:

        1. Compute np.tanh(x).
        2. Cache the result in self.output for the backward pass and
           return it.
        """
        # NumPy applies tanh to every element in the array
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through tanh.

        Parameters
        ----------
        grad_output : numpy.ndarray of the same shape as the forward input, dtype float64
            Gradient of the loss with respect to the tanh output, coming
            from the next layer.

        Returns
        -------
        numpy.ndarray of the same shape as grad_output, dtype float64
            Gradient of the loss with respect to the tanh input.

        Notes
        -----
        Processing:

        1. Read the cached output t from forward().
        2. Return grad_output * (1 - t^2).

        If t = tanh(x), then:
            dt/dx = 1 - t^2

        By the chain rule:
            grad_input = grad_output * (1 - t^2)
        """
        return grad_output * (1 - self.output**2)


# --------------------
# Linear - Identity
# --------------------


class Linear:
    """
    Linear (identity) activation.

    Notes
    -----
    Forward:
        f(x) = x

    Backward:
        f'(x) = 1

    Meaning:

    - Leaves the input unchanged
    - Commonly used in regression output layers
    """

    def __init__(self) -> None:
        """
        Create a linear (identity) activation.

        Parameters
        ----------
        None
            Takes no arguments.

        Returns
        -------
        None
            Nothing is stored; the object has no attributes.

        Notes
        -----
        Processing:

        1. Do nothing. The method exists so all activations share the
           same structure.
        """
        # No cached values are needed for linear activation,
        # but we keep the same class structure as the others
        # for consistency.

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Return the input unchanged.

        Parameters
        ----------
        x : numpy.ndarray of any shape, typically (batch_size, num_features), dtype float64
            Pre-activations, usually the output of the last Dense layer.

        Returns
        -------
        numpy.ndarray of the same shape as x, dtype float64
            The same array object x (not a copy).

        Notes
        -----
        Processing:

        1. Return x as it is.
        """
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through linear activation.

        Parameters
        ----------
        grad_output : numpy.ndarray of the same shape as the forward input, dtype float64
            Gradient of the loss with respect to the activation output,
            coming from the next layer (or from the loss).

        Returns
        -------
        numpy.ndarray of the same shape as grad_output, dtype float64
            The same array object grad_output, unchanged.

        Notes
        -----
        Processing:

        1. Return grad_output as it is.

        Since f(x) = x, its derivative is 1.
        So the gradient just passes through as-is.
        """
        return grad_output


def get_activation(name: str) -> ReLU | Sigmoid | Tanh | Linear:
    """
    Return an activation object based on its name.

    Parameters
    ----------
    name : str
        Name of the activation function, case-insensitive. Supported
        names: "relu", "sigmoid", "tanh", "linear".

    Returns
    -------
    ReLU or Sigmoid or Tanh or Linear
        A new instance of the requested activation class.

    Raises
    ------
    ValueError
        If the activation name is not supported.

    Notes
    -----
    Processing:

    1. Convert name to lowercase.
    2. Return a new object of the matching class.
    3. Raise ValueError for any other name.

    This lets configs choose an activation with a plain string.
    """
    # Convert to lowercase so inputs like "ReLU" or "SIGMOID"
    # still work correctly
    name = name.lower()

    # Factory pattern:
    # choose which activation class to create based on the name
    if name == "relu":
        return ReLU()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    elif name == "linear":
        return Linear()
    else:
        raise ValueError(f"Unsupported activation: {name}")
