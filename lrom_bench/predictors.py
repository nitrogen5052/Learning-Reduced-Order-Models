from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


PotentialFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PredictorPack:
    s_points: np.ndarray
    center_values: np.ndarray
    scales: np.ndarray
    singular_values: np.ndarray


def centered_parameter_predictors(
    samples: np.ndarray,
    center: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    center = np.asarray(center, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if np.any(scales == 0.0):
        raise ValueError("scales must be nonzero")
    return (samples - center) / scales


def greedy_maxvol_indices(basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis)
    if basis.ndim != 2:
        raise ValueError("basis must be two-dimensional")
    n_rows, n_cols = basis.shape
    if n_cols > n_rows:
        raise ValueError("basis must have at least as many rows as columns")

    selected: list[int] = []
    residual_basis = basis.copy()
    for _ in range(n_cols):
        row_norms = np.linalg.norm(residual_basis, axis=1)
        row_norms[selected] = -np.inf
        idx = int(np.argmax(row_norms))
        selected.append(idx)
        selected_matrix = basis[selected]
        projection_coeffs = np.linalg.lstsq(selected_matrix.T, basis.T, rcond=None)[0]
        projected = (selected_matrix.T @ projection_coeffs).T
        residual_basis = basis - projected
    return np.array(selected, dtype=int)


def raw_potential_predictors(
    potential: PotentialFn,
    alphas: np.ndarray,
    s_points: np.ndarray,
) -> np.ndarray:
    return np.asarray([potential(s_points, alpha) for alpha in np.asarray(alphas)])


def make_potential_predictor_pack(
    potential: PotentialFn,
    train_alphas: np.ndarray,
    central_alpha: np.ndarray,
    mesh: np.ndarray,
    n_predictors: int,
    min_mesh_value: float = 0.0,
) -> PredictorPack:
    mesh = np.asarray(mesh, dtype=float)
    center = potential(mesh, central_alpha)
    delta = np.asarray([potential(mesh, alpha) - center for alpha in train_alphas]).T
    allowed = np.flatnonzero(mesh >= min_mesh_value)
    if allowed.size < n_predictors:
        raise ValueError("not enough allowed mesh points for predictor selection")
    u, singular_values, _ = np.linalg.svd(delta[allowed], full_matrices=False)
    if n_predictors > u.shape[1]:
        raise ValueError("not enough SVD modes for requested predictors")
    local = greedy_maxvol_indices(u[:, :n_predictors])
    indices = allowed[local]
    s_points = mesh[indices]
    center_values = potential(s_points, central_alpha)
    raw_train = raw_potential_predictors(potential, train_alphas, s_points)
    scales = np.maximum(np.std(raw_train - center_values[np.newaxis, :], axis=0), 1e-12)
    return PredictorPack(
        s_points=s_points,
        center_values=center_values,
        scales=scales,
        singular_values=singular_values,
    )


def centered_potential_predictors(
    potential: PotentialFn,
    alphas: np.ndarray,
    pack: PredictorPack,
    normalize: bool = True,
) -> np.ndarray:
    values = raw_potential_predictors(potential, alphas, pack.s_points)
    centered = values - pack.center_values[np.newaxis, :]
    if normalize:
        centered = centered / pack.scales[np.newaxis, :]
    return centered
