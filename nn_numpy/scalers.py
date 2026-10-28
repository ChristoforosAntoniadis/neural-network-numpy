"""
Feature scaling used during preprocessing.

Provides a NumPy StandardScaler that learns per-column mean and
standard deviation on the training data and applies the same
transformation to any other split.

Input: a NumPy feature matrix of shape (n_samples, n_features), the training
split for fit() and any split for transform().

Output: the scaled matrix, same shape, float64. Nothing is written to disk.
"""

from typing import Self

import numpy as np


class StandardScaler:
    """
    Standardize features by removing the mean and scaling by the standard deviation.

    Attributes
    ----------
    mean_ : numpy.ndarray of shape (n_features,), dtype float64, or None
        Feature-wise mean learned by fit(); None before fitting.
    std_ : numpy.ndarray of shape (n_features,), dtype float64, or None
        Feature-wise standard deviation learned by fit(), with zeros
        replaced by 1e-8; None before fitting.

    Notes
    -----
    Formula:

        X_scaled = (X - mean) / std

    This is done feature-by-feature, meaning each column is scaled separately.

    Why this is useful: standardization helps neural networks train more
    smoothly because

    - features are put on a similar scale
    - very large-valued features do not dominate smaller-valued ones
    - optimization is usually more stable
    """

    def __init__(self) -> None:
        """
        Create an empty scaler.

        Parameters
        ----------
        None
            Takes no arguments.

        Returns
        -------
        None
            Sets self.mean_ and self.std_ to None until fit() is called.

        Notes
        -----
        Processing:

        1. Set mean_ and std_ to None, marking the scaler as unfitted.
        """
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> Self:
        """
        Learn the feature-wise mean and standard deviation from the data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), dtype float64
            Training data, one sample per row.

        Returns
        -------
        StandardScaler
            The same scaler object (self), now fitted, so calls can be
            chained as in fit(X).transform(X).

        Notes
        -----
        Processing:

        1. Compute the mean of each column and store it in self.mean_.
        2. Compute the standard deviation of each column (population
           std, ddof=0) and store it in self.std_.
        3. Replace any zero standard deviation with 1e-8 to avoid
           division by zero later.

        axis=0 means:
        - compute one mean per column
        - compute one standard deviation per column

        So if X has shape (100, 4), then:
        - self.mean_ will have shape (4,)
        - self.std_ will have shape (4,)
        """
        # Compute the mean of each feature column
        self.mean_ = X.mean(axis=0)

        # Compute the standard deviation of each feature column
        self.std_ = X.std(axis=0)

        # Prevent division by zero for constant features.
        #
        # If a feature has std = 0, then all its values are identical.
        # Replacing 0 with a tiny number avoids numerical errors.
        # In practice, (X - mean) will be 0 for that feature anyway,
        # so the transformed values will remain 0.
        # That makes the replacement value arbitrary: any nonzero number gives
        # 0 / value = 0, and 1e-8 is simply a small nonzero placeholder.
        self.std_ = np.where(self.std_ == 0, 1e-8, self.std_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply standardization using the stored mean and standard deviation.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), dtype float64
            Data to scale. n_features must match the data used in fit().

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features), dtype float64
            Standardized version of X (a new array; X is not modified).

        Raises
        ------
        ValueError
            If fit() has not been called first.

        Notes
        -----
        Processing:

        1. Check that mean_ and std_ have been learned.
        2. Return (X - mean_) / std_, broadcasting the (n_features,)
           statistics across all rows.

        In a proper ML workflow:
        - fit() should be called on training data only
        - transform() should then be applied to train/validation/test
          using the same stored training statistics
        """
        # Make sure the scaler has already learned mean and std
        if self.mean_ is None or self.std_ is None:
            raise ValueError("You must call fit() before transform()")

        # Apply the standard scaling formula feature-by-feature
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the scaler on X, then immediately transform X.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), dtype float64
            Training data to learn the statistics from and then scale.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features), dtype float64
            Standardized version of X.

        Notes
        -----
        Processing:

        1. Call fit(X), which stores mean_ and std_.
        2. Call transform(X) on the fitted scaler and return the result.

        This is just a convenience method equivalent to:

            scaler.fit(X)
            X_scaled = scaler.transform(X)
        """
        return self.fit(X).transform(X)
