from __future__ import annotations

import inspect

import numpy as np

import lrom
import lrom_legacy.v2_0 as v2_0


def _build(module):
    emulator = module.LROM(
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        l=0,
        potential="ws_1",
    )
    vv = dict(emulator.central_parameters)["Vv"]
    sampling_options = dict(
        training_ranges={"Vv": (0.9 * vv, 1.1 * vv)},
        testing_ranges={"Vv": (0.8 * vv, 1.2 * vv)},
        training_size=5,
        testing_size=5,
        mesh_size=96,
        strategy="linspace",
        seed=1204,
    )
    sampling_options["high_fidelity_solver"] = "runge_kutta"
    emulator.sampling(**sampling_options)
    emulator.train(basis_size=3, predictor="parameters", predictor_count=1)
    emulator.predict(parameters={"Vv": 0.95 * vv})
    return emulator


def test_v2_shell_identity_and_cross_section_api() -> None:
    assert v2_0.__version__ == "2.0.0"
    for name in ("sampling", "train", "predict", "save"):
        assert callable(getattr(v2_0.LROM, name))
    training_parameters = inspect.signature(v2_0.LROM.train).parameters
    assert "observable" in training_parameters
    assert "angles_degrees" in training_parameters
    sampling_parameters = inspect.signature(v2_0.LROM.sampling).parameters
    assert "eim_basis_size" not in sampling_parameters
    assert sampling_parameters["high_fidelity_solver"].default == "runge_kutta"


def test_v2_shell_matches_v1_2_for_the_shared_wavefunction_workflow() -> None:
    active = _build(lrom)
    parked = _build(v2_0)

    pairs = (
        (active.samples.central_wavefunctions[0], parked.samples.central_wavefunctions[0]),
        (active.samples.training_wavefunctions[0], parked.samples.training_wavefunctions[0]),
        (active.samples.testing_wavefunctions[0], parked.samples.testing_wavefunctions[0]),
        (active.basis[0].vectors, parked.basis[0].vectors),
        (active.rf_lrom[0].matrices, parked.rf_lrom[0].matrices),
        (active.rf_lrom[0].vectors, parked.rf_lrom[0].vectors),
        (active.predictions.wavefunctions[0], parked.predictions.wavefunctions[0]),
    )
    for active_values, parked_values in pairs:
        assert np.array_equal(active_values, parked_values)
