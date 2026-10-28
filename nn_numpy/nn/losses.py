"""
Loss functions used for training: MSE for regression, BCE for classification.

Each loss class has forward(y_true, y_pred), which returns the scalar loss
averaged over the batch, and backward(y_true, y_pred), which returns
dL/dy_pred, the gradient that starts backpropagation. get_loss() builds a
loss object from a config string.

Input: targets and predictions as NumPy arrays of shape (batch_size, 1).

Output: the batch-mean loss (float) and its gradient with respect to the
predictions. Nothing is written to disk.
"""

import numpy as np


class MSELoss:
    """
    Mean Squared Error loss for regression.

    Notes
    -----
    Formula:
        MSE = (1 / n) * sum((y_true - y_pred)^2)

    This loss measures how far the predictions are from the true values.
    Larger errors are penalized more strongly because the difference is squared.

    This class provides:

    - forward pass: compute scalar loss
    - backward pass: compute dL/dy_pred

    The class has no state, so one object can be reused for any batch.
    """

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the Mean Squared Error loss.

        Parameters
        ----------
        y_true : numpy.ndarray of shape (batch_size, 1), dtype float64
            True target values.
        y_pred : numpy.ndarray of shape (batch_size, 1), dtype float64
            Predicted values, same shape as y_true.

        Returns
        -------
        numpy.float64
            Mean squared error over all elements of the batch.

        Notes
        -----
        Processing:

        1. Compute the element-wise error y_true - y_pred.
        2. Square it.
        3. Average over all elements with np.mean.

        For each sample:
            error = y_true - y_pred

        Then square the error so negative and positive mistakes
        do not cancel each other out.

        Finally, average over the batch.
        """
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of MSE with respect to predictions.

        Parameters
        ----------
        y_true : numpy.ndarray of shape (batch_size, 1), dtype float64
            True target values.
        y_pred : numpy.ndarray of shape (batch_size, 1), dtype float64
            Predicted values, same shape as y_true.

        Returns
        -------
        numpy.ndarray of shape (batch_size, 1), dtype float64
            Gradient dL/dy_pred.

        Notes
        -----
        Processing:

        1. Take n = y_true.shape[0], the number of samples (rows).
        2. Return (2 / n) * (y_pred - y_true).

        For:
            L = (1/n) * sum((y_true - y_pred)^2)

        The derivative is:
            dL/dy_pred = (2/n) * (y_pred - y_true)

        n is the number of rows, so this matches forward() exactly when
        there is one output column, as in this project.

        This tells us how the loss changes if the prediction changes.

        If y_pred is too large:
            gradient is positive
            -> gradient descent will push prediction down

        If y_pred is too small:
            gradient is negative
            -> gradient descent will push prediction up
        """
        # Number of samples in the mini-batch
        n = y_true.shape[0]

        # Gradient of MSE with respect to predictions
        return (2 / n) * (y_pred - y_true)


class BCELoss:
    """
    Binary Cross-Entropy loss for binary classification.

    Notes
    -----
    Formula:
        BCE = -(1 / n) * sum(
            y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)
        )

    Assumes:

    - y_true contains binary labels in {0, 1}
    - y_pred contains probabilities in (0, 1),
      typically the output of a sigmoid activation

    This class provides:

    - forward pass: compute scalar loss
    - backward pass: compute dL/dy_pred

    The class has no state, so one object can be reused for any batch.
    """

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the Binary Cross-Entropy loss.

        Parameters
        ----------
        y_true : numpy.ndarray of shape (batch_size, 1), dtype float64
            True binary labels, 0 or 1.
        y_pred : numpy.ndarray of shape (batch_size, 1), dtype float64
            Predicted probabilities, same shape as y_true.

        Returns
        -------
        numpy.float64
            Binary cross-entropy averaged over all elements of the batch.

        Notes
        -----
        Processing:

        1. Clip y_pred to [1e-12, 1 - 1e-12] so log() never sees 0.
        2. Compute y_true * log(p) + (1 - y_true) * log(1 - p) for each
           element.
        3. Average with np.mean and negate.

        BCE compares:
        - the true class label
        - the predicted probability for that class

        It penalizes confident wrong predictions very strongly.

        Example:
        - if y_true = 1 and y_pred is close to 0, loss becomes very large
        - if y_true = 1 and y_pred is close to 1, loss is small
        """
        # Small constant for numerical stability.
        # Prevents log(0), which is undefined.
        # Why 1e-12: -log(1e-12) = 27.6 caps one sample's loss at a finite
        # value, and 1 - 1e-12 is still distinct from 1 in float64 (whose
        # spacing near 1 is 1.1e-16), so both ends of the clip are exact.
        epsilon = 1e-12

        # Clip probabilities to stay safely inside (0, 1)
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

        # Apply the BCE formula and average over the batch
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )

        return loss

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of BCE with respect to predictions.

        Parameters
        ----------
        y_true : numpy.ndarray of shape (batch_size, 1), dtype float64
            True binary labels, 0 or 1.
        y_pred : numpy.ndarray of shape (batch_size, 1), dtype float64
            Predicted probabilities, same shape as y_true.

        Returns
        -------
        numpy.ndarray of shape (batch_size, 1), dtype float64
            Gradient dL/dy_pred.

        Notes
        -----
        Processing:

        1. Clip y_pred to [1e-12, 1 - 1e-12] so the divisions below never
           divide by zero.
        2. Take n = y_true.shape[0], the number of samples (rows).
        3. Return -(y_true / p - (1 - y_true) / (1 - p)) / n.

        For:
            L = -(1/n) * sum(
                y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)
            )

        The derivative is:
            dL/dy_pred = -(1/n) * [y_true / y_pred - (1 - y_true) / (1 - y_pred)]

        This gradient tells us how changing the predicted probability
        affects the BCE loss.

        BCE expects probabilities, so y_pred should normally come from a
        sigmoid output layer.
        """
        # Small constant for numerical stability.
        # Prevents division by zero when y_pred is exactly 0 or 1.
        epsilon = 1e-12

        # Clip predictions before using them in the gradient formula
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

        # Number of samples in the mini-batch
        n = y_true.shape[0]

        # Apply the derivative formula for BCE
        grad = -((y_true / y_pred_clipped) - ((1 - y_true) / (1 - y_pred_clipped))) / n

        return grad


def get_loss(name: str) -> MSELoss | BCELoss:
    """
    Factory function that returns a loss object by name.

    Parameters
    ----------
    name : str
        Name of the loss function, case-insensitive. Supported names:

        - MSELoss: "mse", "mean_squared_error", "mean_squarederror",
          "meansquared_error"
        - BCELoss: "bce", "binary_crossentropy", "binary_cross_entropy"

    Returns
    -------
    MSELoss or BCELoss
        A new instance of the requested loss class.

    Raises
    ------
    ValueError
        If the loss name is not supported.

    Notes
    -----
    Processing:

    1. Convert name to lowercase.
    2. Return MSELoss() or BCELoss() if the name is in the matching list.
    3. Raise ValueError otherwise.

    This helper lets the rest of the project choose a loss function
    using a string from a config file instead of manually creating
    the object every time.
    """
    # Make the name lowercase so inputs like "MSE" or "BCE" still work
    name = name.lower()

    # Return the requested loss object
    if name in ["mse", "mean_squared_error", "mean_squarederror", "meansquared_error"]:
        return MSELoss()
    elif name in ["bce", "binary_crossentropy", "binary_cross_entropy"]:
        return BCELoss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")
