"""Small, explicit LROM utilities for the clean demonstration notebooks.

The exploratory notebooks grew a fairly large helper module.  This file keeps
the public demo version deliberately narrow:

* central formulation with ``phi0 = phi(alpha_c)``;
* gauge-fixed implicit equation ``M0 = I``;
* residual-fit training, so fitting is one linear least-squares solve;
* optional potential predictors chosen by a simple delta-maxvol rule.

The functions here are intentionally boring and readable.  They are not meant
to hide the method, only to keep the notebooks from drowning in repeated loops.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass
class CentralLROM:
    """A fitted central-gauge LROM for one partial-wave channel.

    The model is

        [I + sum_j p_j(alpha) M_j] a = sum_j p_j(alpha) b_j,

    where the predictors are already centered so that p(alpha_c) = 0.
    """

    name: str
    matrices: np.ndarray  # shape (K, n, n)
    vectors: np.ndarray  # shape (K, n)
    train_seconds: float
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


@dataclass
class PredictorPack:
    """Potential-predictor metadata for one partial-wave channel."""

    s_points: np.ndarray
    center_values: np.ndarray
    scales: np.ndarray
    singular_values: np.ndarray


def fit_central_lrom(name: str, predictors: np.ndarray, coeff_targets: np.ndarray) -> CentralLROM:
    """Fit the central-gauge LROM by one complex least-squares solve.

    Parameters
    ----------
    predictors:
        Array with shape ``(n_samples, K)``.  These must be centered, so that
        the central parameter point has predictor value zero.
    coeff_targets:
        Least-squares reduced coefficients with shape ``(n_samples, n)``.

    Notes
    -----
    For each sample ``i``, the residual equation is

        a_i + sum_j p_ij M_j a_i - sum_j p_ij b_j = 0.

    This is linear in the unknown entries of ``M_j`` and ``b_j`` because the
    target coefficients ``a_i`` are known during training.
    """

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
                # Row of M_j times known coefficient vector a_row.
                design[eq, moff + row * n_basis : moff + (row + 1) * n_basis] = p * a_row
                # Minus source vector b_j.
                design[eq, voff + row] = -p

    tic = time.perf_counter()
    solution, residuals, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    train_seconds = time.perf_counter() - tic

    matrices = np.zeros((k_pred, n_basis, n_basis), dtype=np.complex128)
    vectors = np.zeros((k_pred, n_basis), dtype=np.complex128)
    for j in range(k_pred):
        moff = matrix_offset(j)
        voff = vector_offset(j)
        matrices[j] = solution[moff : moff + n_basis * n_basis].reshape(n_basis, n_basis)
        vectors[j] = solution[voff : voff + n_basis]

    residual = design @ solution - target
    residual_mse = float(np.mean(np.abs(residual) ** 2))
    return CentralLROM(
        name=name,
        matrices=matrices,
        vectors=vectors,
        train_seconds=train_seconds,
        residual_mse=residual_mse,
        rank=int(rank),
        singular_values=singular_values,
    )


def predict_coefficients(model: CentralLROM, predictors: np.ndarray) -> np.ndarray:
    """Solve the fitted implicit system for each row of predictors."""

    predictors = np.asarray(predictors, dtype=np.complex128)
    if predictors.ndim == 1:
        predictors = predictors[np.newaxis, :]
    n_basis = model.n_basis
    identity = np.eye(n_basis, dtype=np.complex128)
    coeffs = np.empty((predictors.shape[0], n_basis), dtype=np.complex128)
    for i, p_row in enumerate(predictors):
        matrix = identity + np.einsum("k,kij->ij", p_row, model.matrices)
        rhs = np.einsum("k,kj->j", p_row, model.vectors)
        coeffs[i] = np.linalg.solve(matrix, rhs)
    return coeffs


def reconstruct_from_basis(basis, coeffs: np.ndarray) -> np.ndarray:
    """Reconstruct scaled wavefunctions from a ROSE CustomBasis and coefficients."""

    coeffs = np.asarray(coeffs, dtype=np.complex128)
    return np.array([basis.phi_0 + basis.vectors @ row for row in coeffs])


def relative_l2_rows(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Row-wise relative L2 errors."""

    pred = np.asarray(pred)
    ref = np.asarray(ref)
    return np.linalg.norm(pred - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-300)


def greedy_maxvol_indices(basis: np.ndarray) -> np.ndarray:
    """Greedy row selection used as a compact maxvol-style rule.

    ``basis`` is an array of candidate rows and columns representing dominant
    modes.  The algorithm starts from the largest row norm and repeatedly adds
    the row with the largest residual after projection onto the selected rows.
    """

    basis = np.asarray(basis)
    n_rows, n_cols = basis.shape
    selected: list[int] = []
    residual_basis = basis.copy()
    for _ in range(n_cols):
        row_norms = np.linalg.norm(residual_basis, axis=1)
        idx = int(np.argmax(row_norms))
        selected.append(idx)
        selected_matrix = basis[selected]
        projection_coeffs = np.linalg.lstsq(selected_matrix.T, basis.T, rcond=None)[0]
        projected = (selected_matrix.T @ projection_coeffs).T
        residual_basis = basis - projected
    return np.array(selected, dtype=int)


def delta_maxvol_predictor_pack(
    interaction,
    train_alphas: np.ndarray,
    central_alpha: np.ndarray,
    rho_mesh: np.ndarray,
    n_predictors: int,
    min_s: float = 0.2,
) -> PredictorPack:
    """Choose potential-predictor points from centered operator variations."""

    center = interaction.tilde(rho_mesh, central_alpha)
    delta = np.array([interaction.tilde(rho_mesh, alpha) - center for alpha in train_alphas]).T
    allowed = np.flatnonzero(rho_mesh >= min_s)
    if allowed.size < n_predictors:
        raise ValueError("not enough allowed mesh points for predictor selection")
    delta_allowed = delta[allowed]
    u, singular_values, _ = np.linalg.svd(delta_allowed, full_matrices=False)
    local = greedy_maxvol_indices(u[:, :n_predictors])
    indices = allowed[local]
    s_points = rho_mesh[indices]
    center_values = interaction.tilde(s_points, central_alpha)
    raw_train = raw_potential_predictors(interaction, train_alphas, s_points)
    scales = np.maximum(np.std(raw_train - center_values[np.newaxis, :], axis=0), 1e-12)
    return PredictorPack(
        s_points=s_points,
        center_values=center_values,
        scales=scales,
        singular_values=singular_values,
    )


def raw_potential_predictors(interaction, alphas: np.ndarray, s_points: np.ndarray) -> np.ndarray:
    """Evaluate the scaled potential at fixed operator-grid points ``s_points``."""

    return np.array([interaction.tilde(s_points, alpha) for alpha in np.asarray(alphas)])


def centered_potential_predictors(
    interaction,
    alphas: np.ndarray,
    pack: PredictorPack,
    normalize: bool = True,
) -> np.ndarray:
    """Return centered potential predictors for a fitted predictor pack."""

    values = raw_potential_predictors(interaction, alphas, pack.s_points)
    centered = values - pack.center_values[np.newaxis, :]
    if normalize:
        centered = centered / pack.scales[np.newaxis, :]
    return centered


def centered_parameter_predictors(samples: np.ndarray, center: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Simple centered raw-parameter predictors."""

    return (np.asarray(samples, dtype=float) - np.asarray(center, dtype=float)) / np.asarray(scales, dtype=float)


def one_at_a_time_scan_samples(center: np.ndarray, widths: np.ndarray, n_scan: int, names: tuple[str, ...]):
    """Build one-at-a-time scan samples around a central parameter vector.

    Returns
    -------
    samples:
        Array of shape (len(names) * n_scan, len(center)).
    scan_info:
        List of (name, values, slice) entries so plotting helpers can recover
        each one-parameter scan. The parameter index is inferred from the order
        of names.
    """

    center = np.asarray(center, dtype=float)
    widths = np.asarray(widths, dtype=float)
    samples = []
    info = []
    start = 0
    for j, name in enumerate(names):
        values = np.linspace(center[j] - widths[j], center[j] + widths[j], n_scan)
        block = np.tile(center, (n_scan, 1))
        block[:, j] = values
        samples.append(block)
        stop = start + n_scan
        info.append((name, values, slice(start, stop)))
        start = stop
    return np.vstack(samples), info
