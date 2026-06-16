from __future__ import annotations

import numpy as np

from lrom_bench.reduced_basis import CentralBasisData, project_ls_coordinates


def test_project_ls_coordinates_wraps_central_basis_data() -> None:
    mesh = np.array([0.0, 1.0, 2.0])
    phi0 = np.array([1.0, 1.0, 1.0])
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    basis = CentralBasisData(phi0=phi0, vectors=vectors, mesh=mesh)
    coeff_true = np.array([[2.0, -3.0], [0.5, 4.0]])
    wavefunctions = phi0[np.newaxis, :] + coeff_true @ vectors.T

    coeff = project_ls_coordinates(basis, wavefunctions)

    assert basis.n_basis == 2
    assert basis.n_mesh == 3
    assert np.allclose(coeff, coeff_true)
