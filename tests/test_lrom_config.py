from __future__ import annotations

import numpy as np
import pytest

from lrom.config import LROMConfig
from lrom.errors import LROMConfigurationError
from lrom.potentials import KD_PARAMETER_NAMES, resolve_potential


def test_l_is_normalized_without_range_expansion() -> None:
    one = LROMConfig.create(
        target=(40, 20), projectile=(1, 0), lab_energy=14.1, l=3
    )
    many = LROMConfig.create(
        target=(40, 20), projectile=(1, 0), lab_energy=14.1, l=(3, 0, 1)
    )

    assert one.channels == (3,)
    assert many.channels == (0, 1, 3)
    assert "only channel 3" in one.description["l"]


def test_ws_schemas_have_named_parameters() -> None:
    ws1 = resolve_potential("ws_1")
    ws3 = resolve_potential("ws_3")
    full = resolve_potential("woods-saxon")

    assert ws1.parameter_names == ("Vv", "Rv", "av")
    assert ws1.sampleable_names == ("Vv",)
    assert ws3.sampleable_names == ("Vv", "Rv", "av")
    assert full.parameter_names == KD_PARAMETER_NAMES
    assert len(full.sampleable_names) == 15


def test_real_woods_saxon_uses_vv_rv_av_order() -> None:
    ws3 = resolve_potential("ws_3")
    radii = np.array([0.0, 4.0, 8.0])

    values = ws3.function(radii, np.array([50.0, 4.0, 0.5]))

    assert values.shape == radii.shape
    assert values[0] < values[1] < values[2] < 0.0


def test_custom_potential_requires_named_central_parameters() -> None:
    def custom(r: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return r * 0.0 + alpha[0]

    with pytest.raises(LROMConfigurationError, match="central_parameters"):
        LROMConfig.create(
            target=(40, 20),
            projectile=(1, 0),
            lab_energy=14.1,
            potential=custom,
        )

    config = LROMConfig.create(
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        potential=custom,
        central_parameters={"depth": 50.0},
    )

    assert config.parameter_names == ("depth",)
    assert config.central_overrides == {"depth": 50.0}


def test_builtin_overrides_are_named_and_validated() -> None:
    config = LROMConfig.create(
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        potential="ws_1",
        central_parameters={"Rv": 4.1, "av": 0.65},
    )

    assert config.central_overrides == {"Rv": 4.1, "av": 0.65}

    with pytest.raises(LROMConfigurationError, match="unknown"):
        LROMConfig.create(
            target=(40, 20),
            projectile=(1, 0),
            lab_energy=14.1,
            potential="ws_3",
            central_parameters={"radius": 4.1},
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"target": (20, 40)}, "target"),
        ({"projectile": (1, 2)}, "projectile"),
        ({"lab_energy": 0.0}, "lab_energy"),
        ({"l": -1}, "l"),
        ({"l": (0, 0)}, "duplicate"),
        ({"fom": "unknown"}, "fom"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object], message: str) -> None:
    base: dict[str, object] = {
        "target": (40, 20),
        "projectile": (1, 0),
        "lab_energy": 14.1,
    }
    base.update(kwargs)

    with pytest.raises(LROMConfigurationError, match=message):
        LROMConfig.create(**base)
