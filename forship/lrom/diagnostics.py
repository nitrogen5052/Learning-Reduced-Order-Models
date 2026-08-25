"""Numerical diagnostics shared by the demonstration notebooks."""

from __future__ import annotations

import numpy as np


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return one relative L2 error for each leading sample."""
    prediction = np.atleast_2d(prediction)
    reference = np.atleast_2d(reference)
    return np.linalg.norm(prediction - reference, axis=1) / np.maximum(
        np.linalg.norm(reference, axis=1), 1.0e-30
    )


def absolute_l2(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return one absolute L2 error for each leading sample."""
    return np.linalg.norm(np.atleast_2d(prediction) - np.atleast_2d(reference), axis=1)
