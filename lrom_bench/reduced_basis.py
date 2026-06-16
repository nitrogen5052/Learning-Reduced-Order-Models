from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lrom_bench.numerics import least_squares_basis_coefficients


@dataclass(frozen=True)
class CentralBasisData:
    phi0: np.ndarray
    vectors: np.ndarray
    mesh: np.ndarray

    def __post_init__(self) -> None:
        phi0 = np.asarray(self.phi0)
        vectors = np.asarray(self.vectors)
        mesh = np.asarray(self.mesh)
        if phi0.ndim != 1:
            raise ValueError("phi0 must be one-dimensional")
        if vectors.ndim != 2:
            raise ValueError("vectors must have shape (n_mesh, n_basis)")
        if mesh.ndim != 1:
            raise ValueError("mesh must be one-dimensional")
        if vectors.shape[0] != phi0.size:
            raise ValueError("vectors and phi0 must share the mesh dimension")
        if mesh.size != phi0.size:
            raise ValueError("mesh and phi0 must have the same length")

    @property
    def n_mesh(self) -> int:
        return int(np.asarray(self.phi0).size)

    @property
    def n_basis(self) -> int:
        return int(np.asarray(self.vectors).shape[1])


def build_centered_svd_basis(
    phi0: np.ndarray,
    snapshots: np.ndarray,
    mesh: np.ndarray,
    n_basis: int,
) -> CentralBasisData:
    phi0 = np.asarray(phi0, dtype=np.complex128)
    snapshots = np.asarray(snapshots, dtype=np.complex128)
    mesh = np.asarray(mesh, dtype=float)
    if snapshots.ndim != 2:
        raise ValueError("snapshots must have shape (n_samples, n_mesh)")
    if phi0.ndim != 1:
        raise ValueError("phi0 must be one-dimensional")
    if snapshots.shape[1] != phi0.size:
        raise ValueError("snapshots and phi0 must share the mesh dimension")
    if n_basis < 1:
        raise ValueError("n_basis must be positive")
    if n_basis > min(snapshots.shape):
        raise ValueError("n_basis cannot exceed the snapshot matrix rank limit")

    centered = snapshots - phi0[np.newaxis, :]
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    vectors = vh[:n_basis].T
    return CentralBasisData(phi0=phi0, vectors=vectors, mesh=mesh)


def project_ls_coordinates(
    basis: CentralBasisData,
    wavefunctions: np.ndarray,
) -> np.ndarray:
    return least_squares_basis_coefficients(
        vectors=basis.vectors,
        phi0=basis.phi0,
        wavefunctions=wavefunctions,
        mesh=basis.mesh,
    )
