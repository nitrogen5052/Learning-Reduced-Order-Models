from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import lrom_legacy.v1_2 as v1_2


ROOT = Path(__file__).resolve().parents[1]


def test_sampling_exposes_only_the_exact_runge_kutta_solver() -> None:
    parameters = inspect.signature(v1_2.LROM.sampling).parameters

    assert "eim_basis_size" not in parameters
    assert parameters["high_fidelity_solver"].default == "runge_kutta"


def test_sampling_rejects_unknown_high_fidelity_solver() -> None:
    emulator = v1_2.LROM(
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        l=0,
        potential="ws_1",
    )

    with pytest.raises(v1_2.LROMSamplingError, match="runge_kutta"):
        emulator.sampling(
            training_ranges={"Vv": (45.0, 50.0)},
            testing_ranges={"Vv": (44.0, 51.0)},
            training_size=3,
            testing_size=3,
            high_fidelity_solver="unknown",
        )


def test_v1_2_fom_uses_exact_interaction_space() -> None:
    source = (ROOT / "lrom_legacy/v1_2/__init__.py").read_text()

    assert "rose.InteractionSpace(**interaction_options)" in source
    assert "InteractionEIMSpace" not in source
