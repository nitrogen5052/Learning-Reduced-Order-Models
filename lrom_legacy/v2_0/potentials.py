"""Named potential schemas used by the public LROM configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .errors import LROMConfigurationError


PotentialFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]

KD_PARAMETER_NAMES = (
    "Vv",
    "Rv",
    "av",
    "Wv",
    "Rwv",
    "awv",
    "Wd",
    "Rd",
    "ad",
    "Vso",
    "Rso",
    "aso",
    "Wso",
    "Rwso",
    "awso",
)

FULL_WOODS_SAXON_PARAMETER_NAMES = (
    "Vv",
    "Wv",
    "Wd",
    "Vso",
    "Rv",
    "Rd",
    "Rso",
    "av",
    "ad",
    "aso",
)

# Strength values follow the ROSE KD_simple sign convention: Wv and Wd are
# positive and enter the interaction as -1j*Wv*f and +4j*ad*Wd*f' (both
# absorptive). Rso depends on the target mass and is filled in by
# full_woods_saxon_central().
_FULL_WOODS_SAXON_CENTRAL_BASE = {
    "Vv": 46.7238,
    "Wv": 1.72334,
    "Wd": 7.2357,
    "Vso": 6.1,
    "Rv": 4.0538,
    "Rd": 4.4055,
    "av": 0.6718,
    "ad": 0.5379,
    "aso": 0.60,
}


def full_woods_saxon_central(*, target_a: int) -> dict[str, float]:
    """Central full Woods-Saxon parameters with Rso derived from the target."""
    central = dict(_FULL_WOODS_SAXON_CENTRAL_BASE)
    central["Rso"] = 1.01 * float(target_a) ** (1.0 / 3.0)
    return central


def real_woods_saxon(r: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Evaluate the real volume Woods-Saxon term."""
    radius = np.asarray(r, dtype=float)
    vv, rv, av = np.asarray(alpha, dtype=float)
    if av <= 0.0:
        raise ValueError("av must be positive")
    exponent = np.clip((radius - rv) / av, -700.0, 700.0)
    return -vv / (1.0 + np.exp(exponent))


def full_woods_saxon(r: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Evaluate the central complex full Woods-Saxon terms."""
    radius = np.asarray(r, dtype=float)
    vv, wv, wd, _vso, rv, rd, _rso, av, ad, _aso = np.asarray(alpha, dtype=float)
    if av <= 0.0 or ad <= 0.0:
        raise ValueError("av and ad must be positive")
    volume_exponent = np.clip((radius - rv) / av, -700.0, 700.0)
    surface_exponent = np.clip((radius - rd) / ad, -700.0, 700.0)
    volume = 1.0 / (1.0 + np.exp(volume_exponent))
    surface_exp = np.exp(surface_exponent)
    surface_prime = -(surface_exp / ad) / (1.0 + surface_exp) ** 2
    return -vv * volume - 1j * wv * volume + 4j * ad * wd * surface_prime


@dataclass(frozen=True)
class PotentialSpec:
    """Stable name-to-vector contract for a potential."""

    name: str
    function: PotentialFunction | None
    parameter_names: tuple[str, ...]
    sampleable_names: tuple[str, ...]


_BUILTINS = {
    "ws_1": PotentialSpec(
        name="ws_1",
        function=real_woods_saxon,
        parameter_names=("Vv", "Rv", "av"),
        sampleable_names=("Vv",),
    ),
    "ws_3": PotentialSpec(
        name="ws_3",
        function=real_woods_saxon,
        parameter_names=("Vv", "Rv", "av"),
        sampleable_names=("Vv", "Rv", "av"),
    ),
    "woods-saxon": PotentialSpec(
        name="woods-saxon",
        function=None,
        parameter_names=KD_PARAMETER_NAMES,
        sampleable_names=KD_PARAMETER_NAMES,
    ),
    "full_woods-saxon": PotentialSpec(
        name="full_woods-saxon",
        function=full_woods_saxon,
        parameter_names=FULL_WOODS_SAXON_PARAMETER_NAMES,
        sampleable_names=FULL_WOODS_SAXON_PARAMETER_NAMES,
    ),
}


def resolve_potential(potential: str) -> PotentialSpec:
    """Resolve a registered potential name."""
    try:
        return _BUILTINS[potential]
    except KeyError as exc:
        choices = ", ".join(sorted(_BUILTINS))
        raise LROMConfigurationError(
            f"unknown potential {potential!r}; choose one of: {choices}"
        ) from exc


def custom_potential_spec(
    potential: PotentialFunction,
    *,
    parameter_names: tuple[str, ...],
) -> PotentialSpec:
    """Create a named schema for a user-supplied numerical potential."""
    return PotentialSpec(
        name=getattr(potential, "__name__", "custom"),
        function=potential,
        parameter_names=parameter_names,
        sampleable_names=parameter_names,
    )
