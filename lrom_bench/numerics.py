from __future__ import annotations

import numpy as np


def trapezoid_integral(
    y: np.ndarray,
    x: np.ndarray | None = None,
    dx: float = 1.0,
    axis: int = -1,
) -> np.ndarray:
    y = np.asarray(y)
    if y.shape[axis] < 2:
        return np.sum(y, axis=axis) * 0.0
    moved = np.moveaxis(y, axis, -1)
    if x is None:
        widths = np.asarray(dx)
    else:
        x = np.asarray(x)
        widths = np.diff(x, axis=0 if x.ndim > 1 else -1)
    area = (moved[..., 1:] + moved[..., :-1]) * 0.5
    return np.sum(area * widths, axis=-1)


def trapezoid_sqrt_weights(mesh: np.ndarray) -> np.ndarray:
    mesh = np.asarray(mesh, dtype=float)
    if mesh.ndim != 1 or mesh.size < 2:
        raise ValueError("mesh must be a one-dimensional array with at least two points")
    weights = np.empty_like(mesh, dtype=float)
    weights[1:-1] = 0.5 * (mesh[2:] - mesh[:-2])
    weights[0] = 0.5 * (mesh[1] - mesh[0])
    weights[-1] = 0.5 * (mesh[-1] - mesh[-2])
    return np.sqrt(weights)


def least_squares_basis_coefficients(
    vectors: np.ndarray,
    phi0: np.ndarray,
    wavefunctions: np.ndarray,
    mesh: np.ndarray,
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.complex128)
    phi0 = np.asarray(phi0, dtype=np.complex128)
    wavefunctions = np.asarray(wavefunctions, dtype=np.complex128)
    weights = trapezoid_sqrt_weights(mesh)
    weighted_vectors = weights[:, np.newaxis] * vectors
    coeffs = []
    for phi in wavefunctions:
        rhs = weights * (phi - phi0)
        coeff, *_ = np.linalg.lstsq(weighted_vectors, rhs, rcond=None)
        coeffs.append(coeff)
    return np.asarray(coeffs)
