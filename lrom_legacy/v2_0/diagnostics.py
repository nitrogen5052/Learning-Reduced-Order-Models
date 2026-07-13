"""Scientific error arrays for notebook-native plotting."""

from __future__ import annotations

import numpy as np


def pointwise_absolute(*, prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(prediction) - np.asarray(reference))


def relative_l2(*, prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    numerator = np.linalg.norm(prediction - reference, axis=1)
    denominator = np.maximum(np.linalg.norm(reference, axis=1), 1e-30)
    return numerator / denominator
