"""Central-reference reduced basis and weighted LS coordinates."""

from __future__ import annotations

import numpy as np

from .state import BasisState


def _sqrt_trapezoid_weights(radius: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius, dtype=float)
    if radius.ndim != 1 or radius.size < 2 or np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius must be a strictly increasing one-dimensional mesh")
    widths = np.diff(radius)
    weights = np.empty(radius.size, dtype=float)
    weights[0] = widths[0] / 2.0
    weights[-1] = widths[-1] / 2.0
    weights[1:-1] = (widths[:-1] + widths[1:]) / 2.0
    return np.sqrt(weights)


def build_basis(
    *,
    phi0: np.ndarray,
    snapshots: np.ndarray,
    radius: np.ndarray,
    basis_size: int,
) -> BasisState:
    """Build a centered SVD basis around the high-fidelity central state."""
    phi0 = np.asarray(phi0, dtype=np.complex128)
    snapshots = np.asarray(snapshots, dtype=np.complex128)
    radius = np.asarray(radius, dtype=float)
    if phi0.ndim != 1:
        raise ValueError("phi0 must be one-dimensional")
    if snapshots.ndim != 2 or snapshots.shape[1] != phi0.size:
        raise ValueError("snapshots must have shape (sample_count, mesh_size)")
    if radius.shape != phi0.shape:
        raise ValueError("radius must have the same length as phi0")
    if (
        isinstance(basis_size, bool)
        or not isinstance(basis_size, int)
        or basis_size < 1
        or basis_size > min(snapshots.shape)
    ):
        raise ValueError("basis_size must be between 1 and the snapshot rank limit")
    centered = snapshots - phi0[np.newaxis, :]
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    return BasisState(
        phi0=phi0,
        vectors=vh[:basis_size].T,
        radius=radius,
        singular_values=singular_values[:basis_size],
    )


def project_coordinates(
    *, basis: BasisState, wavefunctions: np.ndarray
) -> np.ndarray:
    """Compute trapezoid-weighted least-squares coordinates."""
    wavefunctions = np.asarray(wavefunctions, dtype=np.complex128)
    if wavefunctions.ndim != 2 or wavefunctions.shape[1] != basis.phi0.size:
        raise ValueError("wavefunctions must have shape (sample_count, mesh_size)")
    weights = _sqrt_trapezoid_weights(basis.radius)
    weighted_vectors = basis.vectors * weights[:, np.newaxis]
    centered = (wavefunctions - basis.phi0[np.newaxis, :]) * weights[np.newaxis, :]
    coordinates, _residuals, _rank, _singular_values = np.linalg.lstsq(
        weighted_vectors, centered.T, rcond=None
    )
    return coordinates.T


def reconstruct(*, basis: BasisState, coordinates: np.ndarray) -> np.ndarray:
    """Reconstruct wavefunctions from central state plus basis coordinates."""
    coordinates = np.asarray(coordinates, dtype=np.complex128)
    if coordinates.ndim == 1:
        coordinates = coordinates[np.newaxis, :]
    if coordinates.ndim != 2 or coordinates.shape[1] != basis.basis_size:
        raise ValueError("coordinates must have shape (sample_count, basis_size)")
    return basis.phi0[np.newaxis, :] + coordinates @ basis.vectors.T
