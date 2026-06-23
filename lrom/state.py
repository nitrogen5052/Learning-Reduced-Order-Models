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
    central_potential: np.ndarray
    training_potentials: np.ndarray
    testing_potentials: np.ndarray
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


@dataclass(frozen=True)
class BasisState:
    """One central-reference reduced basis and its spectrum."""

    phi0: np.ndarray
    vectors: np.ndarray
    radius: np.ndarray
    singular_values: np.ndarray

    def __post_init__(self) -> None:
        phi0 = np.asarray(self.phi0, dtype=np.complex128)
        vectors = np.asarray(self.vectors, dtype=np.complex128)
        radius = np.asarray(self.radius, dtype=float)
        singular_values = np.asarray(self.singular_values, dtype=float)
        if phi0.ndim != 1 or radius.shape != phi0.shape:
            raise ValueError("phi0 and radius must be one-dimensional with equal size")
        if vectors.ndim != 2 or vectors.shape[0] != phi0.size:
            raise ValueError("basis vectors must have shape (mesh_size, basis_size)")
        object.__setattr__(self, "phi0", phi0)
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "singular_values", singular_values)

    @property
    def basis_size(self) -> int:
        return int(self.vectors.shape[1])


@dataclass(frozen=True)
class RoseRBMState:
    """Live ROSE emulator sharing one authoritative reduced basis."""

    basis: BasisState
    custom_basis: Any
    emulator: Any


@dataclass(frozen=True)
class TestingResults:
    """Testing-set wavefunctions produced by every approved method."""

    high_fidelity: Mapping[int, np.ndarray]
    rose: Mapping[int, np.ndarray]
    lrom: Mapping[int, np.ndarray]
    ls: Mapping[int, np.ndarray]
    coefficients: Mapping[str, Mapping[int, np.ndarray]]
    metrics: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]]


@dataclass(frozen=True)
class PredictionState:
    """Most recent named-parameter inference batch."""

    parameter_names: tuple[str, ...]
    parameters: np.ndarray
    coefficients: Mapping[int, np.ndarray]
    wavefunctions: Mapping[int, np.ndarray]


@dataclass(frozen=True)
class TestingCase:
    """One testing case gathered across all comparison methods."""

    case_id: str
    parameters: Mapping[str, float]
    radius: np.ndarray
    high_fidelity: Mapping[int, np.ndarray]
    rose: Mapping[int, np.ndarray]
    lrom: Mapping[int, np.ndarray]
    ls: Mapping[int, np.ndarray]
