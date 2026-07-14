from __future__ import annotations

import numpy as np

from lrom import LROM
from lrom import build_parameter_predictor, build_potential_predictor


def test_parameter_predictor_uses_named_centered_scales() -> None:
    state = build_parameter_predictor(
        parameter_names=("Vv", "Rv", "av"),
        varied_names=("Vv",),
        central=np.array([50.0, 4.0, 0.65]),
        training_values=np.array([[45.0, 4.0, 0.65], [50.0, 4.0, 0.65], [55.0, 4.0, 0.65]]),
        testing_values=np.array([[40.0, 4.0, 0.65], [60.0, 4.0, 0.65]]),
    )

    assert state.kind == "parameters"
    assert state.names == ("Vv",)
    assert state.training_features[:, 0].tolist() == [-1.0, 0.0, 1.0]
    assert state.testing_features[:, 0].tolist() == [-2.0, 2.0]


def test_potential_predictor_selects_physical_radii() -> None:
    radius = np.linspace(0.0, 4.0, 9)
    central = np.zeros(9)
    mode1 = np.sin(radius)
    mode2 = np.cos(2.0 * radius)
    training = np.asarray([central + a * mode1 + b * mode2 for a, b in [(-1, 0), (0, 1), (1, 1), (2, -1)]])
    testing = np.asarray([central + 0.5 * mode1 - 0.25 * mode2])

    state = build_potential_predictor(
        radius=radius,
        central_potential=central,
        training_potentials=training,
        testing_potentials=testing,
        predictor_count=2,
    )

    assert state.kind == "potential"
    assert state.selected_radii.shape == (2,)
    assert np.all(np.isin(state.selected_radii, radius))
    assert state.training_features.shape == (4, 2)
    assert state.testing_features.shape == (1, 2)


def test_real_ws1_training_stores_lrom_state() -> None:
    emulator = LROM(
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        l=0,
        potential="ws_1",
    )
    emulator.sampling(
        training_ranges={"Vv": (47.0, 53.0)},
        testing_ranges={"Vv": (45.0, 55.0)},
        training_size=5,
        testing_size=5,
        mesh_size=96,
        strategy="linspace",
        seed=9,
        eim_basis_size=2,
    )

    emulator.train(basis_size=2, predictor="parameters")

    assert emulator.is_trained
    assert emulator.basis[0].vectors.shape == (96, 2)
    assert emulator.predictors.kind == "parameters"
    assert emulator.rf_lrom[0].n_basis == 2
    assert emulator.testing_results.high_fidelity[0].shape == (5, 96)
    assert emulator.testing_results.lrom[0].shape == (5, 96)
    assert emulator.testing_results.ls[0].shape == (5, 96)
    assert emulator.testing_results.metrics["relative_l2"][0]["lrom"].shape == (5,)
    assert emulator.training_results.high_fidelity[0].shape == (5, 96)
    for results in (emulator.training_results, emulator.testing_results):
        for method in ("lrom", "ls"):
            assert results.coefficients[method][0].shape == (5, 2)
            assert results.metrics["relative_l2"][0][method].shape == (5,)
            assert results.metrics["pointwise_absolute"][0][method].shape == (
                5,
                96,
            )
    assert emulator.testing_errors[0]["lrom"].shape == (5, 96)
    assert emulator.testing_errors[0]["ls"].shape == (5, 96)

    case = emulator.testing_case(case_id="test-0002")
    assert case.case_id == "test-0002"
    assert case.parameters["Vv"] == 50.0
    assert case.radius.shape == (96,)
    assert case.high_fidelity[0].shape == (96,)
    assert case.lrom[0].shape == (96,)
    assert case.ls[0].shape == (96,)

    emulator.predict(parameters={"Vv": 50.5})

    assert emulator.predictions.parameters.shape == (1, 3)
    assert emulator.predictions.coefficients[0].shape == (1, 2)
    assert emulator.predictions.wavefunctions[0].shape == (1, 96)


def test_real_ws3_training_uses_potential_predictors() -> None:
    emulator = LROM(
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        l=0,
        potential="ws_3",
    )
    emulator.sampling(
        training_ranges={
            "Vv": (47.0, 53.0),
            "Rv": (3.8, 4.2),
            "av": (0.58, 0.72),
        },
        testing_ranges={
            "Vv": (45.0, 55.0),
            "Rv": (3.6, 4.4),
            "av": (0.52, 0.78),
        },
        training_size=8,
        testing_size=4,
        mesh_size=96,
        strategy="latin_hypercube",
        seed=11,
        eim_basis_size=2,
    )

    emulator.train(basis_size=2, predictor="potential", predictor_count=2)

    assert emulator.predictors.kind == "potential"
    assert emulator.predictors.selected_radii.shape == (2,)
    assert emulator.predictors.central_values.shape == (2,)
    assert emulator.training_results.high_fidelity[0].shape == (8, 96)
    assert emulator.training_results.metrics["relative_l2"][0]["lrom"].shape == (8,)
    assert emulator.testing_errors[0]["lrom"].shape == (4, 96)


