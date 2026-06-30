"""Immutable physical configuration for :class:`lrom.LROM`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from .errors import LROMConfigurationError
from .potentials import PotentialFunction, PotentialSpec, custom_potential_spec, resolve_potential


def _validate_nucleus(name: str, value: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise LROMConfigurationError(f"{name} must be an (A, Z) integer tuple")
    mass, charge = value
    if mass < 1 or charge < 0 or charge > mass:
        raise LROMConfigurationError(
            f"{name} must satisfy A >= 1 and 0 <= Z <= A"
        )
    return value


def _normalize_channels(value: int | tuple[int, ...]) -> tuple[int, ...]:
    channels = (value,) if isinstance(value, int) and not isinstance(value, bool) else value
    if not isinstance(channels, tuple) or not channels:
        raise LROMConfigurationError("l must be a non-negative integer or tuple")
    if any(isinstance(channel, bool) or not isinstance(channel, int) or channel < 0 for channel in channels):
        raise LROMConfigurationError("l channels must be non-negative integers")
    if len(set(channels)) != len(channels):
        raise LROMConfigurationError("l contains a duplicate channel")
    return tuple(sorted(channels))


def _validate_overrides(
    values: Mapping[str, float] | None,
    *,
    allowed: tuple[str, ...],
) -> Mapping[str, float]:
    copied = {} if values is None else dict(values)
    unknown = sorted(set(copied) - set(allowed))
    if unknown:
        raise LROMConfigurationError(f"unknown central parameter names: {unknown}")
    for name, value in copied.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise LROMConfigurationError(f"central parameter {name!r} must be finite")
        copied[name] = float(value)
    return MappingProxyType(copied)


@dataclass(frozen=True)
class LROMConfig:
    """Normalized immutable inputs that define one physical emulator."""

    target: tuple[int, int]
    projectile: tuple[int, int]
    lab_energy: float
    channels: tuple[int, ...]
    fom: str
    potential: PotentialSpec
    central_overrides: Mapping[str, float]
    description: Mapping[str, str]

    @classmethod
    def create(
        cls,
        *,
        target: tuple[int, int],
        projectile: tuple[int, int],
        lab_energy: float,
        l: int | tuple[int, ...] = 0,
        fom: str = "nucl-scatter-eq",
        potential: str | PotentialFunction = "ws_3",
        central_parameters: Mapping[str, float] | None = None,
    ) -> "LROMConfig":
        target = _validate_nucleus("target", target)
        projectile = _validate_nucleus("projectile", projectile)
        if (
            isinstance(lab_energy, bool)
            or not isinstance(lab_energy, (int, float))
            or not math.isfinite(lab_energy)
            or lab_energy <= 0.0
        ):
            raise LROMConfigurationError("lab_energy must be a positive finite value")
        channels = _normalize_channels(l)
        if fom != "nucl-scatter-eq":
            raise LROMConfigurationError(
                "unknown fom; the first implementation supports 'nucl-scatter-eq'"
            )

        if isinstance(potential, str):
            potential_spec = resolve_potential(potential)
            overrides = _validate_overrides(
                central_parameters, allowed=potential_spec.parameter_names
            )
        elif callable(potential):
            if not central_parameters:
                raise LROMConfigurationError(
                    "custom potential requires named central_parameters"
                )
            parameter_names = tuple(central_parameters)
            if any(not isinstance(name, str) or not name for name in parameter_names):
                raise LROMConfigurationError("custom parameter names must be non-empty strings")
            potential_spec = custom_potential_spec(
                potential, parameter_names=parameter_names
            )
            overrides = _validate_overrides(
                central_parameters, allowed=parameter_names
            )
        else:
            raise LROMConfigurationError(
                "potential must be a registered name or callable potential(r, alpha)"
            )

        if len(channels) == 1:
            l_text = f"only channel {channels[0]}"
        else:
            l_text = f"exact channels {channels}"
        description = MappingProxyType(
            {
                "target": "Target nucleus as an (A, Z) tuple.",
                "projectile": "Incident projectile as an (A, Z) tuple.",
                "lab_energy": "Projectile laboratory-frame energy in MeV.",
                "l": f"Orbital angular-momentum selection: {l_text}.",
                "fom": "Full-order equation identifier.",
                "potential": "Registered potential schema or custom callable.",
            }
        )
        return cls(
            target=target,
            projectile=projectile,
            lab_energy=float(lab_energy),
            channels=channels,
            fom=fom,
            potential=potential_spec,
            central_overrides=overrides,
            description=description,
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.potential.parameter_names

    @property
    def sampleable_names(self) -> tuple[str, ...]:
        return self.potential.sampleable_names
