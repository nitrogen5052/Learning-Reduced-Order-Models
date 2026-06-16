from __future__ import annotations

import numpy as np


def relative_l2_rows(pred: np.ndarray, ref: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    if pred.shape != ref.shape:
        raise ValueError("pred and ref must have the same shape")
    return np.linalg.norm(pred - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), floor)


def absolute_l2_rows(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    if pred.shape != ref.shape:
        raise ValueError("pred and ref must have the same shape")
    return np.linalg.norm(pred - ref, axis=1)
