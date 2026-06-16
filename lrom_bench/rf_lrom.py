from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CentralLROM:
    name: str
    matrices: np.ndarray
    vectors: np.ndarray
    residual_mse: float
    rank: int
    singular_values: np.ndarray

    @property
    def n_basis(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def n_predictors(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def n_complex_parameters(self) -> int:
        n = self.n_basis
        return self.n_predictors * (n * n + n)


def fit_central_lrom(
    name: str,
    predictors: np.ndarray,
    coeff_targets: np.ndarray,
) -> CentralLROM:
    predictors = np.asarray(predictors, dtype=np.complex128)
    coeff_targets = np.asarray(coeff_targets, dtype=np.complex128)
    if predictors.ndim != 2:
        raise ValueError("predictors must have shape (n_samples, K)")
    if coeff_targets.ndim != 2:
        raise ValueError("coeff_targets must have shape (n_samples, n_basis)")
    if predictors.shape[0] != coeff_targets.shape[0]:
        raise ValueError("predictors and coeff_targets need the same sample count")

    n_samples, k_pred = predictors.shape
    n_basis = coeff_targets.shape[1]
    n_unknown = k_pred * (n_basis * n_basis + n_basis)
    design = np.zeros((n_samples * n_basis, n_unknown), dtype=np.complex128)
    target = -coeff_targets.reshape(-1)

    def matrix_offset(j: int) -> int:
        return j * (n_basis * n_basis + n_basis)

    def vector_offset(j: int) -> int:
        return matrix_offset(j) + n_basis * n_basis

    for sample_idx, (p_row, a_row) in enumerate(zip(predictors, coeff_targets)):
        for row in range(n_basis):
            eq = sample_idx * n_basis + row
            for j, p in enumerate(p_row):
                moff = matrix_offset(j)
                voff = vector_offset(j)
                design[eq, moff + row * n_basis : moff + (row + 1) * n_basis] = p * a_row
                design[eq, voff + row] = -p

    solution, _residuals, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)

    matrices = np.zeros((k_pred, n_basis, n_basis), dtype=np.complex128)
    vectors = np.zeros((k_pred, n_basis), dtype=np.complex128)
    for j in range(k_pred):
        moff = matrix_offset(j)
        voff = vector_offset(j)
        matrices[j] = solution[moff : moff + n_basis * n_basis].reshape(n_basis, n_basis)
        vectors[j] = solution[voff : voff + n_basis]

    residual = design @ solution - target
    return CentralLROM(
        name=name,
        matrices=matrices,
        vectors=vectors,
        residual_mse=float(np.mean(np.abs(residual) ** 2)),
        rank=int(rank),
        singular_values=singular_values,
    )
