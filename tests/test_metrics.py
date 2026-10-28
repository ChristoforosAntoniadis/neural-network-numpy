"""
Tests for the secondary evaluation metrics in nn_numpy.nn.metrics:
accuracy (classification) and R² (regression).
"""

import numpy as np
import pytest

from nn_numpy.nn.metrics import accuracy, r2_score


def test_accuracy_thresholds_probabilities_at_one_half() -> None:
    """
    accuracy counts a probability of at least 0.5 as class 1.

    Notes
    -----
    Labels [1, 0, 1, 0] with probabilities [0.9, 0.2, 0.4, 0.5]: the first
    two are right, the third (0.4 -> class 0) and fourth (0.5 -> class 1)
    are wrong, so accuracy is 0.5.
    """
    assert accuracy([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.5]) == pytest.approx(0.5)


def test_accuracy_accepts_column_vectors() -> None:
    """
    accuracy gives the same result for (n,) and (n, 1) inputs.

    Notes
    -----
    The network outputs column vectors, scikit-learn 1-D arrays; both
    shapes must be handled identically.
    """
    y = np.array([1, 0, 1])
    p = np.array([0.8, 0.1, 0.7])
    assert accuracy(y, p) == accuracy(y.reshape(-1, 1), p.reshape(-1, 1)) == 1.0


def test_r2_is_one_for_perfect_predictions_and_zero_for_the_mean() -> None:
    """
    r2_score is 1 for a perfect fit and 0 for always predicting the mean.

    Notes
    -----
    These two reference points define R². For y = [1, 2, 3, 6] the mean is
    3, so SS_tot = 4 + 1 + 0 + 9 = 14. A perfect fit has SS_res = 0, so
    R² = 1 - 0 / 14 = 1; predicting 3 everywhere gives SS_res = SS_tot,
    so R² = 1 - 14 / 14 = 0.
    """
    y = np.array([1.0, 2.0, 3.0, 6.0])
    assert r2_score(y, y) == pytest.approx(1.0)
    assert r2_score(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_r2_matches_the_definition() -> None:
    """
    r2_score equals 1 - SS_res / SS_tot on random data.

    Notes
    -----
    Computed by hand from the formula, with column-vector inputs. The
    noise (scale 0.5) keeps R² strictly between 0 and 1, so the check
    is not satisfied by either reference point alone.
    """
    rng = np.random.RandomState(0)
    y = rng.normal(size=(50, 1))
    pred = y + rng.normal(scale=0.5, size=(50, 1))
    expected = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    assert r2_score(y, pred) == pytest.approx(expected)
