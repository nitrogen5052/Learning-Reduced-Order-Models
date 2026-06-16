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
