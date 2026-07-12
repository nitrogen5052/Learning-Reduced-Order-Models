"""Residual-fit learned reduced-operator model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RFLROMModel:
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


def fit(*, predictors: np.ndarray, coordinates: np.ndarray) -> RFLROMModel:
    """Fit the transformed implicit reduced equation by linear LS."""
    predictors = np.asarray(predictors, dtype=np.complex128)
    coordinates = np.asarray(coordinates, dtype=np.complex128)
    if predictors.ndim != 2 or coordinates.ndim != 2:
        raise ValueError("predictors and coordinates must be two-dimensional")
    if predictors.shape[0] != coordinates.shape[0]:
        raise ValueError("predictors and coordinates require equal sample counts")
    sample_count, predictor_count = predictors.shape
    basis_size = coordinates.shape[1]
    block_size = basis_size * basis_size + basis_size
    design = np.zeros(
        (sample_count * basis_size, predictor_count * block_size),
        dtype=np.complex128,
    )
    target = -coordinates.reshape(-1)
    for sample_index, (feature_row, coordinate_row) in enumerate(
        zip(predictors, coordinates)
    ):
        for equation_row in range(basis_size):
            equation = sample_index * basis_size + equation_row
            for predictor_index, feature in enumerate(feature_row):
                offset = predictor_index * block_size
                matrix_start = offset + equation_row * basis_size
                design[
                    equation, matrix_start : matrix_start + basis_size
                ] = feature * coordinate_row
                design[equation, offset + basis_size * basis_size + equation_row] = -feature
    solution, _residuals, rank, singular_values = np.linalg.lstsq(
        design, target, rcond=None
    )
    matrices = np.empty(
        (predictor_count, basis_size, basis_size), dtype=np.complex128
    )
    vectors = np.empty((predictor_count, basis_size), dtype=np.complex128)
    for predictor_index in range(predictor_count):
        offset = predictor_index * block_size
        matrices[predictor_index] = solution[
            offset : offset + basis_size * basis_size
        ].reshape(basis_size, basis_size)
        vectors[predictor_index] = solution[
            offset + basis_size * basis_size : offset + block_size
        ]
    residual = design @ solution - target
    return RFLROMModel(
        matrices=matrices,
        vectors=vectors,
        residual_mse=float(np.mean(np.abs(residual) ** 2)),
        rank=int(rank),
        singular_values=singular_values,
    )


def solve(*, model: RFLROMModel, predictors: np.ndarray) -> np.ndarray:
    """Solve the small online implicit equation for reduced coordinates."""
    predictors = np.asarray(predictors, dtype=np.complex128)
    if predictors.ndim == 1:
        predictors = predictors[np.newaxis, :]
    if predictors.ndim != 2 or predictors.shape[1] != model.n_predictors:
        raise ValueError("predictor width does not match the RF-LROM model")
    identity = np.eye(model.n_basis, dtype=np.complex128)
    coordinates = np.empty(
        (predictors.shape[0], model.n_basis), dtype=np.complex128
    )
    for index, row in enumerate(predictors):
        matrix = identity + np.einsum("k,kij->ij", row, model.matrices)
        rhs = np.einsum("k,kj->j", row, model.vectors)
        coordinates[index] = np.linalg.solve(matrix, rhs)
    return coordinates
