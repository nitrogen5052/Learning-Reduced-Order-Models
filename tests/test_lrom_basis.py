from __future__ import annotations

import numpy as np
import pytest

from lrom.basis import build_basis, project_coordinates, reconstruct


def test_centered_basis_projects_and_reconstructs_training_space() -> None:
    radius = np.linspace(0.0, 4.0, 81)
    phi0 = np.exp(1j * radius)
    mode1 = np.sin(radius)
    mode2 = 1j * np.cos(2.0 * radius)
    snapshots = np.asarray(
        [
            phi0 + 0.5 * mode1 - 0.2 * mode2,
            phi0 - 0.4 * mode1 + 0.7 * mode2,
            phi0 + 0.1 * mode1 + 0.3 * mode2,
        ]
    )

    basis = build_basis(
        phi0=phi0,
        snapshots=snapshots,
        radius=radius,
        basis_size=2,
    )
    coordinates = project_coordinates(basis=basis, wavefunctions=snapshots)
    reconstructed = reconstruct(basis=basis, coordinates=coordinates)

    assert basis.vectors.shape == (81, 2)
    assert basis.singular_values.shape == (2,)
    assert coordinates.shape == (3, 2)
    assert np.allclose(reconstructed, snapshots, atol=1e-11)


def test_basis_rejects_invalid_size() -> None:
    radius = np.linspace(0.0, 1.0, 5)
    phi0 = np.zeros(5)
    snapshots = np.zeros((2, 5))

    with pytest.raises(ValueError, match="basis_size"):
        build_basis(phi0=phi0, snapshots=snapshots, radius=radius, basis_size=3)
