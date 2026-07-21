from __future__ import annotations

import numpy as np

import lrom_legacy.v1_2 as v1_2


def _trained_emulator() -> v1_2.LROM:
    emulator = v1_2.LROM(
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        l=0,
        potential="ws_1",
    )
    vv = dict(emulator.central_parameters)["Vv"]
    emulator.sampling(
        training_ranges={"Vv": (0.9 * vv, 1.1 * vv)},
        testing_ranges={"Vv": (0.8 * vv, 1.2 * vv)},
        training_size=5,
        testing_size=5,
        mesh_size=96,
        strategy="linspace",
    )
    emulator.train(basis_size=3, predictor="parameters", predictor_count=1)
    return emulator


def test_train_does_not_compute_or_store_optional_ls_benchmark() -> None:
    emulator = _trained_emulator()
    results = emulator.testing_results

    assert results.ls is None
    assert set(results.coefficients) == {"lrom"}
    assert set(results.metrics["relative_l2"][0]) == {"lrom"}
    assert set(results.metrics["pointwise_absolute"][0]) == {"lrom"}
    assert set(emulator.testing_errors[0]) == {"lrom"}
    assert emulator.testing_case(case_id="test-0000").ls is None


def test_least_squares_baseline_is_an_explicit_basis_projection() -> None:
    emulator = _trained_emulator()
    basis = emulator.basis[0]
    high_fidelity = emulator.samples.testing_wavefunctions[0]

    coordinates, reconstruction = v1_2.least_squares_baseline(
        basis=basis,
        wavefunctions=high_fidelity,
    )

    expected_coordinates = v1_2.project_coordinates(
        basis=basis,
        wavefunctions=high_fidelity,
    )
    assert np.array_equal(coordinates, expected_coordinates)
    assert np.array_equal(
        reconstruction,
        v1_2.reconstruct(basis=basis, coordinates=coordinates),
    )

