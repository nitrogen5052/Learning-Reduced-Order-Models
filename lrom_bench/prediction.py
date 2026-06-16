from __future__ import annotations

import numpy as np

from lrom_bench.rf_lrom import CentralLROM


def predict_coefficients(model: CentralLROM, predictors: np.ndarray) -> np.ndarray:
    predictors = np.asarray(predictors, dtype=np.complex128)
    if predictors.ndim == 1:
        predictors = predictors[np.newaxis, :]
    if predictors.ndim != 2:
        raise ValueError("predictors must have shape (n_samples, K) or (K,)")
    if predictors.shape[1] != model.n_predictors:
        raise ValueError("predictor width does not match model")

    n_basis = model.n_basis
    identity = np.eye(n_basis, dtype=np.complex128)
    coeffs = np.empty((predictors.shape[0], n_basis), dtype=np.complex128)
    for i, p_row in enumerate(predictors):
        matrix = identity + np.einsum("k,kij->ij", p_row, model.matrices)
        rhs = np.einsum("k,kj->j", p_row, model.vectors)
        coeffs[i] = np.linalg.solve(matrix, rhs)
    return coeffs


def reconstruct_from_basis(
    phi0: np.ndarray,
    vectors: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    phi0 = np.asarray(phi0, dtype=np.complex128)
    vectors = np.asarray(vectors, dtype=np.complex128)
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    if phi0.ndim != 1:
        raise ValueError("phi0 must have shape (n_mesh,)")
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape (n_mesh, n_basis)")
    if coeffs.ndim != 2:
        raise ValueError("coeffs must have shape (n_samples, n_basis)")
    if vectors.shape[1] != coeffs.shape[1]:
        raise ValueError("basis width must match between vectors and coeffs")
    if vectors.shape[0] != phi0.size:
        raise ValueError("mesh width must match between phi0 and vectors")
    return phi0[np.newaxis, :] + coeffs @ vectors.T
