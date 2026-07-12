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


def real_woods_saxon(r: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Evaluate the real volume Woods-Saxon term."""
    radius = np.asarray(r, dtype=float)
    vv, rv, av = np.asarray(alpha, dtype=float)
    if av <= 0.0:
        raise ValueError("av must be positive")
    exponent = np.clip((radius - rv) / av, -700.0, 700.0)
    return -vv / (1.0 + np.exp(exponent))


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
