"""
Evaluation metrics reported for trained models.

The metrics reuse the training losses (BCELoss and MSELoss) so that the
reported numbers are directly comparable with the loss curves. Both
functions accept 1-D or column-vector inputs and return a plain float.

Input: targets and predictions (or probabilities) as array-likes.

Output: one float per call. Nothing is written to disk.
"""

import numpy as np
from numpy.typing import ArrayLike

from nn_numpy.nn.losses import BCELoss, MSELoss


def binary_cross_entropy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute Binary Cross-Entropy (BCE) as the classification evaluation metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, 1)
        True labels, 0 or 1. Converted to float64.
    y_pred : array-like of shape (n_samples,) or (n_samples, 1)
        Predicted probabilities from the network, usually the output
        of a sigmoid activation. Converted to float64. Must have the same
        number of elements as y_true.

    Returns
    -------
    float
        Mean binary cross-entropy. Lower is better.

    Notes
    -----
    Processing:

    1. Convert both inputs to float64 NumPy arrays and reshape them to
       column vectors of shape (n_samples, 1).
    2. Compute the loss with BCELoss().forward, which clips the
       probabilities to avoid log(0).
    3. Convert the result to a Python float.

    The same BCE used as the training loss is used to evaluate the model,
    so the reported metric is directly comparable with the loss curves.

    It computes:
        -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

    Reusing BCELoss keeps a single implementation (including the
    clipping that avoids log(0)).
    """

    # Convert inputs to NumPy arrays and reshape them into column vectors,
    # so shapes like (n_samples,) and (n_samples, 1) are handled the same way.
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1, 1)

    return float(BCELoss().forward(y_true, y_pred))


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute Mean Squared Error (MSE) as the regression evaluation metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, 1)
        True target values. Converted to float64.
    y_pred : array-like of shape (n_samples,) or (n_samples, 1)
        Predicted target values. Converted to float64. Must have the same
        number of elements as y_true.

    Returns
    -------
    float
        Mean squared error. Lower is better.

    Notes
    -----
    Processing:

    1. Convert both inputs to float64 NumPy arrays and reshape them to
       column vectors of shape (n_samples, 1).
    2. Compute the loss with MSELoss().forward.
    3. Convert the result to a Python float.

    The same MSE used as the training loss is used to evaluate the model,
    so the reported metric is directly comparable with the loss curves.

    It computes:
        mean((y_true - y_pred)^2)

    Reusing MSELoss keeps a single implementation.
    """

    # Convert inputs to NumPy arrays and reshape them into column vectors,
    # so shapes like (n_samples,) and (n_samples, 1) are handled the same way.
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1, 1)

    return float(MSELoss().forward(y_true, y_pred))


def accuracy(y_true: ArrayLike, y_prob: ArrayLike, threshold: float = 0.5) -> float:
    """
    Compute classification accuracy from predicted probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, 1)
        True labels, 0 or 1.
    y_prob : array-like of shape (n_samples,) or (n_samples, 1)
        Predicted probability of class 1 for each sample.
    threshold : float, default=0.5
        0.5 predicts whichever class has the higher probability.
        A sample is predicted as class 1 when its probability is at least
        this value.

    Returns
    -------
    float
        Fraction of samples whose predicted class equals the true label,
        between 0 and 1. Higher is better.

    Notes
    -----
    Processing:

    1. Flatten both inputs to 1-D float64 arrays.
    2. Turn probabilities into class predictions: 1 if y_prob >= threshold,
       otherwise 0.
    3. Return the mean of (prediction == label).

    Accuracy is reported next to BCE because it is easier to read, but it
    ignores how confident each prediction was. BCE remains the metric used
    for training and model selection.
    """
    # Flatten so (n,) and (n, 1) inputs are handled the same way.
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    predictions = (y_prob >= threshold).astype(float)
    return float(np.mean(predictions == y_true))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute the coefficient of determination (R²) for regression.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, 1)
        True target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, 1)
        Predicted target values.

    Returns
    -------
    float
        R² = 1 - SS_res / SS_tot. 1 is a perfect fit, 0 matches always
        predicting the mean of y_true, and negative values are worse than
        that. Higher is better.

    Notes
    -----
    Processing:

    1. Flatten both inputs to 1-D float64 arrays.
    2. SS_res = sum((y_true - y_pred)^2), the squared error of the model.
    3. SS_tot = sum((y_true - mean(y_true))^2), the squared error of always
       predicting the mean of the same targets.
    4. Return 1 - SS_res / SS_tot.

    This is the textbook R², which uses the mean of the evaluated targets
    themselves. The "error removed" figure in the analysis instead compares
    against a constant fitted on the training set, so the two can differ
    slightly.
    """
    # Flatten so (n,) and (n, 1) inputs are handled the same way.
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)
