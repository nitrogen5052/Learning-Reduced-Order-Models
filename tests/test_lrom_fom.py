from __future__ import annotations

import numpy as np

from lrom import LROM


def test_nuclear_scattering_fom_samples_l0_real_ws() -> None:
    emulator = LROM(
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        l=0,
        potential="ws_1",
    )

    emulator.sampling(
        training_ranges={"Vv": (48.0, 52.0)},
        testing_ranges={"Vv": (46.0, 54.0)},
        training_size=3,
        testing_size=3,
        mesh_size=64,
        strategy="linspace",
        seed=7,
        eim_basis_size=2,
    )

    assert emulator.is_sampled
    assert emulator.partial_waves == (0,)
    assert emulator.mesh.radius.shape == (64,)
    assert np.all(np.diff(emulator.mesh.radius) > 0.0)
    assert emulator.samples.central_wavefunctions[0].shape == (64,)
    assert emulator.samples.training_wavefunctions[0].shape == (3, 64)
    assert emulator.samples.testing_wavefunctions[0].shape == (3, 64)
    assert emulator.samples.training_potentials.shape == (3, 64)
    assert set(emulator.central_parameters) == {"Vv", "Rv", "av"}
    assert set(emulator.full_order_model) == {0}
