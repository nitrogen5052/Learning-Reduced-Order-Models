"""Small state containers owned by :class:`lrom.LROM`."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParameterCases:
    """One named parameter table with stable case identifiers."""

    case_ids: tuple[str, ...]
    parameter_names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != (len(self.case_ids), len(self.parameter_names)):
            raise ValueError("parameter values shape does not match IDs and names")
        object.__setattr__(self, "values", values)

    def named(self, *, index: int) -> dict[str, float]:
        """Return one row as a name-to-value mapping."""
        return {
            name: float(value)
            for name, value in zip(self.parameter_names, self.values[index])
        }


@dataclass(frozen=True)
class SamplingDesign:
    """Named, reproducible training and testing parameter cases."""

    training: ParameterCases
    testing: ParameterCases
    strategy: str
    seed: int | None


@dataclass(frozen=True)
class Kinematics:
    """Central scattering kinematics resolved by the FOM provider."""

    mu: float
    e_com: float
    k: float
    eta: float
    coulomb_radius: float


@dataclass(frozen=True)
class MeshState:
    """Internal and physical one-dimensional meshes."""

    rho: np.ndarray
    radius: np.ndarray

    def __post_init__(self) -> None:
        rho = np.asarray(self.rho, dtype=float)
        radius = np.asarray(self.radius, dtype=float)
        if rho.ndim != 1 or radius.ndim != 1 or rho.shape != radius.shape:
            raise ValueError("rho and radius must be one-dimensional arrays of equal size")
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "radius", radius)


@dataclass(frozen=True)
class SamplingState:
    """All authoritative arrays and runtime objects produced by sampling."""

    design: SamplingDesign
    central_parameters: Mapping[str, float]
    kinematics: Kinematics
    mesh: MeshState
    central_wavefunctions: Mapping[int, np.ndarray]
    training_wavefunctions: Mapping[int, np.ndarray]
    testing_wavefunctions: Mapping[int, np.ndarray]
    training_potentials: np.ndarray
    full_order_models: Mapping[int, Any]


@dataclass(frozen=True)
class TrainingState:
    """Authoritative reduced models and automatic test evaluation."""

    basis: Mapping[int, Any]
    predictors: Any
    rf_lrom: Mapping[int, Any]
    rose_rbm: Mapping[int, Any]
    testing_results: Any
    testing_errors: Mapping[int, Any]
