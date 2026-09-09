"""LROM: learned reduced-operator models for nuclear scattering wavefunctions.

The whole version 1 package lives in this single file, organized in the order
a calculation flows: errors, potentials, configuration, state containers,
sampling designs, the reduced basis, diagnostics, predictors, the RF-LROM
core, the ROSE full-order-model boundary, the training engine, portable
artifacts, and finally the public `LROM` object.

Canonical workflow:

    import lrom

    emulator = lrom.LROM(target=(40, 20), projectile=(1, 0),
                         lab_energy=14.1, l=0, potential="ws_3")
    emulator.sampling(training_ranges=..., testing_ranges=...,
                      training_size=..., testing_size=...)
    emulator.train(basis_size=4, predictor="potential", predictor_count=6)
    emulator.predict(parameters={"Vv": 47.0})
    emulator.save(path="model.lrom")

Version 2.0 (cross sections) is parked in `lrom_legacy.v2_0` pending fixes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import platform
from types import MappingProxyType
from typing import Any
import zipfile

import numpy as np
from numba import njit
from scipy.stats import qmc

# ==========================================================================
# 1. Physical configuration and potentials
# Owns the immutable physical question and validates its named potential.
# Does not own solver outputs, training data, or learned operators.
# ==========================================================================

class LROMError(Exception):
    """Base error for the public LROM workflow."""


class LROMConfigurationError(LROMError, ValueError):
    """Invalid immutable physical configuration."""


class LROMSamplingError(LROMError, ValueError):
    """Invalid sampling request or returned sample state."""


class LROMStateError(LROMError, RuntimeError):
    """Workflow method called in an invalid lifecycle state."""


class LROMArtifactError(LROMError, ValueError):
    """Invalid or incompatible portable emulator artifact."""


# -- Potential definitions -------------------------------------------------

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


# -- Immutable physical configuration -------------------------------------

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
        l: int | tuple[int, ...] = 0,  # noqa: E741 - standard partial-wave symbol
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


# ==========================================================================
# 2. Parameter designs and lifecycle state
# Owns named sample cases and the raw NumPy state produced at each stage.
# Does not run ROSE, fit RF-LROM, or mutate the public object.
# ==========================================================================

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
    testing_results: Any
    testing_errors: Mapping[int, Any]
    training_results: Any = None
    training_options: Mapping[str, Any] | None = None


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
class TestingResults:
    """Automatic RF-LROM diagnostics for a sampled data set.

    ``ls`` remains as a compatibility slot, but training deliberately leaves it
    empty.  A least-squares reconstruction is a separate benchmark, not part of
    the learned RF-LROM model; request it with ``least_squares_baseline``.
    """

    high_fidelity: Mapping[int, np.ndarray]
    lrom: Mapping[int, np.ndarray]
    ls: Mapping[int, np.ndarray] | None
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
    lrom: Mapping[int, np.ndarray]
    ls: Mapping[int, np.ndarray] | None


# -- Parameter-design builders --------------------------------------------

def _validate_size(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LROMSamplingError(f"{name} must be a positive integer")
    return value


def _validate_ranges(
    ranges: Mapping[str, tuple[float, float]],
    *,
    parameter_names: tuple[str, ...],
    sampleable_names: tuple[str, ...],
) -> dict[str, tuple[float, float]]:
    copied = dict(ranges)
    unknown = sorted(set(copied) - set(parameter_names))
    if unknown:
        raise LROMSamplingError(f"unknown parameter names: {unknown}")
    unavailable = sorted(set(copied) - set(sampleable_names))
    if unavailable:
        raise LROMSamplingError(
            f"parameters are not sampleable for this potential: {unavailable}"
        )
    if not copied:
        raise LROMSamplingError("at least one parameter range is required")
    validated: dict[str, tuple[float, float]] = {}
    for name, bounds in copied.items():
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise LROMSamplingError(f"range for {name!r} must be a (minimum, maximum) tuple")
        lower, upper = bounds
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or not math.isfinite(lower)
            or not math.isfinite(upper)
        ):
            raise LROMSamplingError(f"range for {name!r} must be finite")
        if lower >= upper:
            raise LROMSamplingError(f"range for {name!r} must be increasing")
        validated[name] = (float(lower), float(upper))
    return validated


def _base_values(
    *,
    count: int,
    parameter_names: tuple[str, ...],
    central: Mapping[str, float],
) -> np.ndarray:
    missing = sorted(set(parameter_names) - set(central))
    if missing:
        raise LROMSamplingError(f"central values missing parameters: {missing}")
    row = np.asarray([central[name] for name in parameter_names], dtype=float)
    if not np.all(np.isfinite(row)):
        raise LROMSamplingError("central parameter values must be finite")
    return np.tile(row, (count, 1))


def _linspace_cases(
    *,
    prefix: str,
    count: int,
    parameter_names: tuple[str, ...],
    central: Mapping[str, float],
    ranges: Mapping[str, tuple[float, float]],
) -> ParameterCases:
    if len(ranges) != 1:
        raise LROMSamplingError("linspace requires exactly one varied parameter")
    values = _base_values(
        count=count, parameter_names=parameter_names, central=central
    )
    name, bounds = next(iter(ranges.items()))
    values[:, parameter_names.index(name)] = np.linspace(*bounds, count)
    return ParameterCases(
        case_ids=tuple(f"{prefix}-{index:04d}" for index in range(count)),
        parameter_names=parameter_names,
        values=values,
    )


def _lhs_cases(
    *,
    prefix: str,
    count: int,
    parameter_names: tuple[str, ...],
    central: Mapping[str, float],
    ranges: Mapping[str, tuple[float, float]],
    rng: np.random.Generator,
) -> ParameterCases:
    values = _base_values(
        count=count, parameter_names=parameter_names, central=central
    )
    varied_names = tuple(name for name in parameter_names if name in ranges)
    lower = np.asarray([ranges[name][0] for name in varied_names])
    upper = np.asarray([ranges[name][1] for name in varied_names])
    unit = qmc.LatinHypercube(d=len(varied_names), seed=rng).random(count)
    sampled = qmc.scale(unit, lower, upper)
    for column, name in enumerate(varied_names):
        values[:, parameter_names.index(name)] = sampled[:, column]
    return ParameterCases(
        case_ids=tuple(f"{prefix}-{index:04d}" for index in range(count)),
        parameter_names=parameter_names,
        values=values,
    )


def _explicit_cases(
    *,
    prefix: str,
    grid: Mapping[str, Sequence[float]],
    parameter_names: tuple[str, ...],
    sampleable_names: tuple[str, ...],
    central: Mapping[str, float],
) -> ParameterCases:
    copied = dict(grid)
    if not copied:
        raise LROMSamplingError("an explicit grid requires at least one parameter")
    unknown = sorted(set(copied) - set(parameter_names))
    if unknown:
        raise LROMSamplingError(f"unknown parameter names: {unknown}")
    unavailable = sorted(set(copied) - set(sampleable_names))
    if unavailable:
        raise LROMSamplingError(
            f"parameters are not sampleable for this potential: {unavailable}"
        )

    columns: dict[str, np.ndarray] = {}
    for name, column in copied.items():
        try:
            values = np.asarray(column, dtype=float)
        except (TypeError, ValueError) as exc:
            raise LROMSamplingError(
                f"explicit grid column {name!r} must be numeric"
            ) from exc
        if values.ndim != 1:
            raise LROMSamplingError(
                f"explicit grid column {name!r} must be one-dimensional"
            )
        if values.size == 0:
            raise LROMSamplingError(
                f"explicit grid column {name!r} requires at least one value"
            )
        if not np.all(np.isfinite(values)):
            raise LROMSamplingError(
                f"explicit grid column {name!r} must be finite"
            )
        columns[name] = values

    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise LROMSamplingError(
            "all columns in an explicit grid must have equal length"
        )
    count = lengths.pop()
    values = _base_values(
        count=count,
        parameter_names=parameter_names,
        central=central,
    )
    for name, column in columns.items():
        values[:, parameter_names.index(name)] = column
    return ParameterCases(
        case_ids=tuple(f"{prefix}-{index:04d}" for index in range(count)),
        parameter_names=parameter_names,
        values=values,
    )


def create_explicit_sampling_design(
    *,
    parameter_names: tuple[str, ...],
    sampleable_names: tuple[str, ...],
    central: Mapping[str, float],
    training_grid: Mapping[str, Sequence[float]],
    testing_grid: Mapping[str, Sequence[float]],
) -> SamplingDesign:
    """Create row-aligned cases from user-supplied named parameter columns."""
    if set(training_grid) != set(testing_grid):
        raise LROMSamplingError(
            "training_grid and testing_grid must use the same parameter names"
        )
    training = _explicit_cases(
        prefix="train",
        grid=training_grid,
        parameter_names=parameter_names,
        sampleable_names=sampleable_names,
        central=central,
    )
    testing = _explicit_cases(
        prefix="test",
        grid=testing_grid,
        parameter_names=parameter_names,
        sampleable_names=sampleable_names,
        central=central,
    )
    return SamplingDesign(
        training=training,
        testing=testing,
        strategy="explicit_grid",
        seed=None,
    )


def create_sampling_design(
    *,
    parameter_names: tuple[str, ...],
    sampleable_names: tuple[str, ...],
    central: Mapping[str, float],
    training_ranges: Mapping[str, tuple[float, float]],
    testing_ranges: Mapping[str, tuple[float, float]],
    training_size: int,
    testing_size: int,
    strategy: str = "latin_hypercube",
    seed: int | None = None,
) -> SamplingDesign:
    """Create named training/testing cases from separate physical domains."""
    training_size = _validate_size("training_size", training_size)
    testing_size = _validate_size("testing_size", testing_size)
    training_ranges = _validate_ranges(
        training_ranges,
        parameter_names=parameter_names,
        sampleable_names=sampleable_names,
    )
    testing_ranges = _validate_ranges(
        testing_ranges,
        parameter_names=parameter_names,
        sampleable_names=sampleable_names,
    )
    if set(training_ranges) != set(testing_ranges):
        raise LROMSamplingError(
            "training_ranges and testing_ranges must use the same parameter names"
        )
    if strategy not in {"linspace", "latin_hypercube"}:
        raise LROMSamplingError(
            "strategy must be 'linspace' or 'latin_hypercube'"
        )

    if strategy == "linspace":
        training = _linspace_cases(
            prefix="train",
            count=training_size,
            parameter_names=parameter_names,
            central=central,
            ranges=training_ranges,
        )
        testing = _linspace_cases(
            prefix="test",
            count=testing_size,
            parameter_names=parameter_names,
            central=central,
            ranges=testing_ranges,
        )
    else:
        sequence = np.random.SeedSequence(seed)
        training_seed, testing_seed = sequence.spawn(2)
        training = _lhs_cases(
            prefix="train",
            count=training_size,
            parameter_names=parameter_names,
            central=central,
            ranges=training_ranges,
            rng=np.random.default_rng(training_seed),
        )
        testing = _lhs_cases(
            prefix="test",
            count=testing_size,
            parameter_names=parameter_names,
            central=central,
            ranges=testing_ranges,
            rng=np.random.default_rng(testing_seed),
        )
    return SamplingDesign(
        training=training,
        testing=testing,
        strategy=strategy,
        seed=seed,
    )


# ==========================================================================
# 3. Centered reduced basis
# Owns compression of spatial wavefunction snapshots around phi0.
# Does not learn parameter dependence or run a full-order solve.
# ==========================================================================

def _sqrt_trapezoid_weights(radius: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius, dtype=float)
    if radius.ndim != 1 or radius.size < 2 or np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius must be a strictly increasing one-dimensional mesh")
    widths = np.diff(radius)
    weights = np.empty(radius.size, dtype=float)
    weights[0] = widths[0] / 2.0
    weights[-1] = widths[-1] / 2.0
    weights[1:-1] = (widths[:-1] + widths[1:]) / 2.0
    return np.sqrt(weights)


def build_basis(
    *,
    phi0: np.ndarray,
    snapshots: np.ndarray,
    radius: np.ndarray,
    basis_size: int,
) -> BasisState:
    """Build a centered SVD basis around the high-fidelity central state."""
    phi0 = np.asarray(phi0, dtype=np.complex128)
    snapshots = np.asarray(snapshots, dtype=np.complex128)
    radius = np.asarray(radius, dtype=float)
    if phi0.ndim != 1:
        raise ValueError("phi0 must be one-dimensional")
    if snapshots.ndim != 2 or snapshots.shape[1] != phi0.size:
        raise ValueError("snapshots must have shape (sample_count, mesh_size)")
    if radius.shape != phi0.shape:
        raise ValueError("radius must have the same length as phi0")
    if (
        isinstance(basis_size, bool)
        or not isinstance(basis_size, int)
        or basis_size < 1
        or basis_size > min(snapshots.shape)
    ):
        raise ValueError("basis_size must be between 1 and the snapshot rank limit")
    centered = snapshots - phi0[np.newaxis, :]
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    return BasisState(
        phi0=phi0,
        vectors=vh[:basis_size].T,
        radius=radius,
        singular_values=singular_values[:basis_size],
    )


def project_coordinates(
    *, basis: BasisState, wavefunctions: np.ndarray
) -> np.ndarray:
    """Express snapshots in the reduced basis for RF-LROM training.

    The radial integral is represented by trapezoid weights.  Multiplying both
    the basis and centered snapshots by the square-root weights turns the
    weighted basis projection into the ordinary system ``A a = b``, where
    ``A = W**(1/2) Phi`` and ``b = W**(1/2) (phi - phi0)``.  ``lstsq``
    solves that problem directly with an SVD-based LAPACK routine; forming and
    solving normal equations would square the condition number and lose
    numerical accuracy.
    """
    wavefunctions = np.asarray(wavefunctions, dtype=np.complex128)
    if wavefunctions.ndim != 2 or wavefunctions.shape[1] != basis.phi0.size:
        raise ValueError("wavefunctions must have shape (sample_count, mesh_size)")
    weights = _sqrt_trapezoid_weights(basis.radius)
    weighted_vectors = basis.vectors * weights[:, np.newaxis]
    centered = (wavefunctions - basis.phi0[np.newaxis, :]) * weights[np.newaxis, :]
    coordinates, _residuals, _rank, _singular_values = np.linalg.lstsq(
        weighted_vectors, centered.T, rcond=None
    )
    return coordinates.T


def reconstruct(*, basis: BasisState, coordinates: np.ndarray) -> np.ndarray:
    """Reconstruct wavefunctions from central state plus basis coordinates."""
    coordinates = np.asarray(coordinates, dtype=np.complex128)
    if coordinates.ndim == 1:
        coordinates = coordinates[np.newaxis, :]
    if coordinates.ndim != 2 or coordinates.shape[1] != basis.basis_size:
        raise ValueError("coordinates must have shape (sample_count, basis_size)")
    return basis.phi0[np.newaxis, :] + coordinates @ basis.vectors.T


# ==========================================================================
# 4. Optional analysis utilities
# Owns explicit benchmarks and raw error arrays used for diagnostics.
# Does not participate in RF-LROM inference unless called by the user.
# ==========================================================================

def least_squares_baseline(
    *, basis: BasisState, wavefunctions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the optional best-in-basis reconstruction benchmark.

    This benchmark sees the target high-fidelity wavefunction, so it answers a
    different question from RF-LROM prediction: how accurately could this
    fixed basis reconstruct the target if its optimal coefficients were known?
    Keeping the call explicit prevents that oracle calculation from being
    mistaken for part of the learned LROM algorithm.
    """
    coordinates = project_coordinates(basis=basis, wavefunctions=wavefunctions)
    wavefunction = reconstruct(basis=basis, coordinates=coordinates)
    return coordinates, wavefunction


def pointwise_absolute(*, prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(prediction) - np.asarray(reference))


def relative_l2(*, prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    numerator = np.linalg.norm(prediction - reference, axis=1)
    denominator = np.maximum(np.linalg.norm(reference, axis=1), 1e-30)
    return numerator / denominator


# ==========================================================================
# 5. Predictor construction
# Owns the low-dimensional features p_j(alpha) used by RF-LROM.
# Does not fit reduced operators or reconstruct wavefunctions.
# ==========================================================================

@dataclass(frozen=True)
class PredictorState:
    """Fitted transformation from physical parameters to RF predictors."""

    kind: str
    names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_indices: np.ndarray
    center: np.ndarray
    scales: np.ndarray
    training_features: np.ndarray
    testing_features: np.ndarray
    selected_indices: np.ndarray
    selected_radii: np.ndarray
    central_values: np.ndarray
    singular_values: np.ndarray


def build_parameter_predictor(
    *,
    parameter_names: tuple[str, ...],
    varied_names: tuple[str, ...],
    central: np.ndarray,
    training_values: np.ndarray,
    testing_values: np.ndarray,
) -> PredictorState:
    """Normalize named parameter deviations by the training-domain extent."""
    indices = np.asarray([parameter_names.index(name) for name in varied_names], dtype=int)
    central = np.asarray(central, dtype=float)
    training_values = np.asarray(training_values, dtype=float)
    testing_values = np.asarray(testing_values, dtype=float)
    delta_training = training_values[:, indices] - central[indices]
    scales = np.max(np.abs(delta_training), axis=0)
    if np.any(scales <= 0.0):
        raise ValueError("varied parameter scales must be nonzero")
    return PredictorState(
        kind="parameters",
        names=varied_names,
        parameter_names=parameter_names,
        parameter_indices=indices,
        center=central,
        scales=scales,
        training_features=delta_training / scales,
        testing_features=(testing_values[:, indices] - central[indices]) / scales,
        selected_indices=np.empty(0, dtype=int),
        selected_radii=np.empty(0),
        central_values=np.empty(0),
        singular_values=np.empty(0),
    )


def _greedy_maxvol_indices(basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis)
    if basis.ndim != 2 or basis.shape[1] > basis.shape[0]:
        raise ValueError("maxvol basis must have shape (points, modes)")
    selected: list[int] = []
    residual = basis.copy()
    for _ in range(basis.shape[1]):
        norms = np.linalg.norm(residual, axis=1)
        if selected:
            norms[selected] = -np.inf
        index = int(np.argmax(norms))
        selected.append(index)
        chosen = basis[selected]
        coefficients = np.linalg.lstsq(chosen.T, basis.T, rcond=None)[0]
        residual = basis - (chosen.T @ coefficients).T
    return np.asarray(selected, dtype=int)


def build_potential_predictor(
    *,
    radius: np.ndarray,
    central_potential: np.ndarray,
    training_potentials: np.ndarray,
    testing_potentials: np.ndarray,
    predictor_count: int,
    minimum_radius: float = 0.0,
) -> PredictorState:
    """Select informative potential locations with SVD and greedy maxvol."""
    radius = np.asarray(radius, dtype=float)
    central = np.asarray(central_potential)
    training = np.asarray(training_potentials)
    testing = np.asarray(testing_potentials)
    if training.ndim != 2 or testing.ndim != 2 or central.shape != radius.shape:
        raise ValueError("potential arrays must use (samples, radius) shapes")
    allowed = np.flatnonzero(radius >= minimum_radius)
    delta = (training - central[np.newaxis, :]).T
    u, singular_values, _vh = np.linalg.svd(delta[allowed], full_matrices=False)
    if predictor_count < 1 or predictor_count > min(u.shape):
        raise ValueError("predictor_count exceeds the available potential rank")
    local = _greedy_maxvol_indices(u[:, :predictor_count])
    selected = allowed[local]
    raw_training = training[:, selected] - central[selected][np.newaxis, :]
    scales = np.maximum(np.std(raw_training, axis=0), 1e-12)
    return PredictorState(
        kind="potential",
        names=tuple(f"U(r={radius[index]:.8g})" for index in selected),
        parameter_names=(),
        parameter_indices=np.empty(0, dtype=int),
        center=np.empty(0),
        scales=scales,
        training_features=raw_training / scales,
        testing_features=(testing[:, selected] - central[selected][np.newaxis, :]) / scales,
        selected_indices=selected,
        selected_radii=radius[selected],
        central_values=central[selected],
        singular_values=singular_values,
    )


def features_for_values(
    *,
    predictor: PredictorState,
    values: np.ndarray,
    potential_function=None,
) -> np.ndarray:
    """Apply a fitted predictor transformation to new ordered parameter rows."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if predictor.kind == "parameters":
        return (
            values[:, predictor.parameter_indices]
            - predictor.center[predictor.parameter_indices][np.newaxis, :]
        ) / predictor.scales[np.newaxis, :]
    if potential_function is None:
        raise ValueError("potential predictor requires a potential function")
    raw = np.asarray(
        [potential_function(predictor.selected_radii, row) for row in values]
    )
    return (raw - predictor.central_values[np.newaxis, :]) / predictor.scales[np.newaxis, :]


# ==========================================================================
# 6. RF-LROM numerical core
# Owns the stacked operator fit and the small online reduced solve.
# Does not generate high-fidelity data or choose the reduced basis.
# ==========================================================================

@dataclass(frozen=True)
class RFLROMModel:
    matrices: np.ndarray
    vectors: np.ndarray
    residual_mse: float
    rank: int
    singular_values: np.ndarray

    @property
    def n_basis(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def n_predictors(self) -> int:
        return int(self.vectors.shape[0])


def fit_rf_lrom(
    *, predictors: np.ndarray, coordinates: np.ndarray
) -> RFLROMModel:
    """Fit one stacked RF-LROM system with complex least squares.

    For each sample, the reduced equation is linear in the unknown entries of
    every predictor matrix ``M_j`` and vector ``b_j`` once the training
    coordinate ``a`` is known.  Stacking all samples and reduced-equation rows
    gives a design matrix with shape
    ``(sample_count * basis_size, predictor_count * (basis_size**2 + basis_size))``.

    One solve keeps the coupled equation visible and avoids independent fits
    that could obscure its shared residual.  ``np.linalg.lstsq`` applies a
    stable LAPACK least-squares solver directly; normal equations are avoided
    because they square the condition number.  The fitted flat vector is then
    unpacked into the small ``M_j`` matrices and ``b_j`` vectors used online.
    """
    predictors = np.asarray(predictors, dtype=np.complex128)
    coordinates = np.asarray(coordinates, dtype=np.complex128)
    if predictors.ndim != 2 or coordinates.ndim != 2:
        raise ValueError("predictors and coordinates must be two-dimensional")
    if predictors.shape[0] != coordinates.shape[0]:
        raise ValueError("predictors and coordinates require equal sample counts")
    sample_count, predictor_count = predictors.shape
    basis_size = coordinates.shape[1]
    block_size = basis_size * basis_size + basis_size
    design = np.zeros(
        (sample_count * basis_size, predictor_count * block_size),
        dtype=np.complex128,
    )
    target = -coordinates.reshape(-1)
    # Each row encodes one component of
    # (I + sum_j p_j M_j) a = sum_j p_j b_j.
    for sample_index, (feature_row, coordinate_row) in enumerate(
        zip(predictors, coordinates)
    ):
        for equation_row in range(basis_size):
            equation = sample_index * basis_size + equation_row
            for predictor_index, feature in enumerate(feature_row):
                offset = predictor_index * block_size
                matrix_start = offset + equation_row * basis_size
                design[
                    equation, matrix_start : matrix_start + basis_size
                ] = feature * coordinate_row
                design[equation, offset + basis_size * basis_size + equation_row] = -feature
    solution, _residuals, rank, singular_values = np.linalg.lstsq(
        design, target, rcond=None
    )
    matrices = np.empty(
        (predictor_count, basis_size, basis_size), dtype=np.complex128
    )
    vectors = np.empty((predictor_count, basis_size), dtype=np.complex128)
    for predictor_index in range(predictor_count):
        offset = predictor_index * block_size
        matrices[predictor_index] = solution[
            offset : offset + basis_size * basis_size
        ].reshape(basis_size, basis_size)
        vectors[predictor_index] = solution[
            offset + basis_size * basis_size : offset + block_size
        ]
    residual = design @ solution - target
    return RFLROMModel(
        matrices=matrices,
        vectors=vectors,
        residual_mse=float(np.mean(np.abs(residual) ** 2)),
        rank=int(rank),
        singular_values=singular_values,
    )


def solve_rf_lrom(
    *, model: RFLROMModel, predictors: np.ndarray
) -> np.ndarray:
    """Perform the online reduced solve for each new predictor row."""
    predictors = np.asarray(predictors, dtype=np.complex128)
    if predictors.ndim == 1:
        predictors = predictors[np.newaxis, :]
    if predictors.ndim != 2 or predictors.shape[1] != model.n_predictors:
        raise ValueError("predictor width does not match the RF-LROM model")
    identity = np.eye(model.n_basis, dtype=np.complex128)
    coordinates = np.empty(
        (predictors.shape[0], model.n_basis), dtype=np.complex128
    )
    for index, row in enumerate(predictors):
        matrix = identity + np.einsum("k,kij->ij", row, model.matrices)
        rhs = np.einsum("k,kj->j", row, model.vectors)
        # This solve is only basis_size by basis_size; no spatial FOM is run.
        coordinates[index] = np.linalg.solve(matrix, rhs)
    return coordinates


# ==========================================================================
# 7. Exact ROSE high-fidelity boundary
# Owns physical-radius meshes and authoritative Runge-Kutta snapshots.
# Does not build ROSE EIMs, reduced bases, predictors, or learned operators.
# ==========================================================================

def _import_rose() -> Any:
    import scipy.special

    if not hasattr(scipy.special, "sph_harm") and hasattr(scipy.special, "sph_harm_y"):
        def sph_harm(m: Any, n: Any, theta: Any, phi: Any) -> Any:
            # legacy sph_harm took (theta=azimuthal, phi=polar);
            # sph_harm_y takes angles in (polar, azimuthal) order
            return scipy.special.sph_harm_y(n, m, phi, theta)

        scipy.special.sph_harm = sph_harm
    try:
        import rose
    except ImportError as exc:
        raise ImportError(
            "LROM sampling with 'nucl-scatter-eq' requires nuclear-rose"
        ) from exc
    return rose


@njit
def _real_ws_interaction(r: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    vv = alpha[0]
    rv = alpha[1]
    av = alpha[2]
    return -vv / (1.0 + np.exp((r - rv) / av))



@dataclass(frozen=True)
class ChannelFOM:
    """Live ROSE solver state for one exact partial-wave channel."""

    channel: int
    interaction: Any
    solver: Any
    base_solver: Any
    rho_mesh: np.ndarray
    radius_mesh: np.ndarray

    def solve(self, *, parameters: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.solver.phi(np.asarray(parameters, dtype=float), self.rho_mesh),
            dtype=np.complex128,
        )


class NuclearScatteringFOM:
    """Construct and sample the built-in ROSE scattering equation."""

    def resolve(
        self, *, config: LROMConfig
    ) -> tuple[Mapping[str, float], Kinematics]:
        rose = _import_rose()
        projectile_lookup = {
            (1, 0): rose.Projectile.neutron,
            (1, 1): rose.Projectile.proton,
        }
        try:
            projectile = projectile_lookup[config.projectile]
        except KeyError as exc:
            raise LROMConfigurationError(
                "nucl-scatter-eq currently supports neutron (1, 0) or proton (1, 1)"
            ) from exc
        mu, e_com, k, eta = rose.kinematics(
            target=config.target,
            projectile=config.projectile,
            E_lab=config.lab_energy,
        )
        kd = rose.koning_delaroche.KDGlobal(projectile)
        coulomb_radius, kd_values = kd.get_params(
            config.target[0], config.target[1], mu, config.lab_energy, k
        )
        if config.potential.name in {"ws_1", "ws_3"}:
            values = np.asarray(kd_values[:3], dtype=float)
        elif config.potential.name == "woods-saxon":
            values = np.asarray(kd_values, dtype=float)
        else:
            values = np.asarray(
                [config.central_overrides[name] for name in config.parameter_names],
                dtype=float,
            )
        central = dict(zip(config.parameter_names, values))
        central.update(config.central_overrides)
        return MappingProxyType(central), Kinematics(
            mu=float(mu),
            e_com=float(e_com),
            k=float(k),
            eta=float(eta),
            coulomb_radius=float(coulomb_radius),
        )

    def sample(
        self,
        *,
        config: LROMConfig,
        design: SamplingDesign,
        mesh_size: int,
        radial_domain: tuple[float, float] | None,
        high_fidelity_solver: str,
        solver_options: Mapping[str, object] | None,
    ) -> SamplingState:
        if isinstance(mesh_size, bool) or not isinstance(mesh_size, int) or mesh_size < 16:
            raise LROMSamplingError("mesh_size must be an integer of at least 16")
        if high_fidelity_solver != "runge_kutta":
            raise LROMSamplingError(
                "high_fidelity_solver must be 'runge_kutta'"
            )
        central, kinematics = self.resolve(config=config)
        central_vector = np.asarray(
            [central[name] for name in config.parameter_names], dtype=float
        )
        rose = _import_rose()

        if radial_domain is None:
            rho_domain = (1e-8, float(8.0 * np.pi))
        else:
            if len(radial_domain) != 2 or radial_domain[0] < 0 or radial_domain[0] >= radial_domain[1]:
                raise LROMSamplingError(
                    "radial_domain must be an increasing non-negative (minimum, maximum) tuple"
                )
            rho_domain = (
                max(1e-8, float(radial_domain[0]) * kinematics.k),
                float(radial_domain[1]) * kinematics.k,
            )
        rho_mesh = np.linspace(*rho_domain, mesh_size)
        radius_mesh = rho_mesh / kinematics.k

        interaction_options: dict[str, Any] = {
            "l_max": max(config.channels),
            "n_theta": len(config.parameter_names),
            "mu": kinematics.mu,
            "energy": kinematics.e_com,
        }
        if config.potential.name == "woods-saxon":
            interaction_options.update(
                coordinate_space_potential=rose.koning_delaroche.KD_simple,
                spin_orbit_term=rose.koning_delaroche.KD_simple_so,
                is_complex=True,
            )
            potential_function = rose.koning_delaroche.KD_simple
        else:
            interaction_options.update(
                coordinate_space_potential=(
                    _real_ws_interaction
                    if config.potential.name in {"ws_1", "ws_3"}
                    else config.potential.function
                ),
                is_complex=False,
            )
            potential_function = config.potential.function
        interactions = rose.InteractionSpace(**interaction_options)

        options = dict(solver_options or {})
        s_0 = float(options.pop("s_0", 6.0 * np.pi))
        rk_tols = options.pop("rk_tols", (1e-9, 1e-9))
        if options:
            raise LROMSamplingError(
                f"unknown solver_options: {sorted(options)}"
            )
        base_solver = rose.SchroedingerEquation.make_base_solver(
            s_0=s_0,
            rk_tols=list(rk_tols),
            domain=np.asarray(rho_domain, dtype=float),
        )
        full_order_models: dict[int, ChannelFOM] = {}
        for channel in config.channels:
            interaction = interactions.interactions[channel][0]
            solver = base_solver.clone_for_new_interaction(interaction)
            full_order_models[channel] = ChannelFOM(
                channel=channel,
                interaction=interaction,
                solver=solver,
                base_solver=base_solver,
                rho_mesh=rho_mesh,
                radius_mesh=radius_mesh,
            )

        central_wavefunctions = {
            channel: model.solve(parameters=central_vector)
            for channel, model in full_order_models.items()
        }
        training_wavefunctions = {
            channel: np.asarray(
                [model.solve(parameters=row) for row in design.training.values]
            )
            for channel, model in full_order_models.items()
        }
        testing_wavefunctions = {
            channel: np.asarray(
                [model.solve(parameters=row) for row in design.testing.values]
            )
            for channel, model in full_order_models.items()
        }
        if potential_function is None:
            raise LROMSamplingError("selected potential cannot be evaluated")
        central_potential = np.asarray(
            potential_function(radius_mesh, central_vector)
        )
        training_potentials = np.asarray(
            [potential_function(radius_mesh, row) for row in design.training.values]
        )
        testing_potentials = np.asarray(
            [potential_function(radius_mesh, row) for row in design.testing.values]
        )
        return SamplingState(
            design=design,
            central_parameters=central,
            kinematics=kinematics,
            mesh=MeshState(rho=rho_mesh, radius=radius_mesh),
            central_wavefunctions=central_wavefunctions,
            training_wavefunctions=training_wavefunctions,
            testing_wavefunctions=testing_wavefunctions,
            central_potential=central_potential,
            training_potentials=training_potentials,
            testing_potentials=testing_potentials,
            full_order_models=full_order_models,
        )


# ==========================================================================
# 8. RF-LROM training orchestration
# Owns the visible sequence basis -> predictors -> coordinates -> RF fit.
# Does not change the public lifecycle or compute an automatic LS baseline.
# ==========================================================================

def _channel_sort_key(channel) -> tuple[int, int]:
    if isinstance(channel, tuple):
        return int(channel[0]), int(channel[1])
    return int(channel), 0


def _trained_channels(emulator) -> tuple:
    return tuple(sorted(emulator.samples.full_order_models, key=_channel_sort_key))


def _centered_basis(*, emulator, channel, basis_size: int) -> BasisState:
    samples = emulator.samples
    return build_basis(
        phi0=np.asarray(samples.central_wavefunctions[channel], dtype=np.complex128),
        snapshots=np.asarray(samples.training_wavefunctions[channel], dtype=np.complex128),
        radius=samples.mesh.radius,
        basis_size=basis_size,
    )


def _predictor(*, emulator, kind: str, count: int) -> PredictorState:
    samples = emulator.samples
    central = np.asarray(
        [emulator.central_parameters[name] for name in emulator.parameter_names]
    )
    if kind == "parameters":
        varying = np.ptp(samples.design.training.values, axis=0) > 0.0
        varied_names = tuple(
            name for name, varied in zip(emulator.parameter_names, varying) if varied
        )
        return build_parameter_predictor(
            parameter_names=emulator.parameter_names,
            varied_names=varied_names,
            central=central,
            training_values=samples.design.training.values,
            testing_values=samples.design.testing.values,
        )
    if kind == "potential":
        return build_potential_predictor(
            radius=samples.mesh.radius,
            central_potential=samples.central_potential,
            training_potentials=samples.training_potentials,
            testing_potentials=samples.testing_potentials,
            predictor_count=count,
            minimum_radius=0.2,
        )
    raise ValueError("predictor must be 'parameters' or 'potential'")


def _evaluate(
    *,
    emulator,
    bases,
    rf_models,
    wavefunctions,
    predictor_features,
) -> TestingResults:
    high_fidelity = dict(wavefunctions)
    lrom_wavefunctions = {}
    lrom_coefficients: dict[object, np.ndarray] = {}
    pointwise_metrics: dict[object, dict[str, np.ndarray]] = {}
    relative_metrics: dict[object, dict[str, np.ndarray]] = {}
    for channel in _trained_channels(emulator):
        basis = bases[channel]
        lrom_coordinates = solve_rf_lrom(
            model=rf_models[channel],
            predictors=predictor_features,
        )
        lrom_coefficients[channel] = lrom_coordinates
        lrom_wavefunctions[channel] = reconstruct(
            basis=basis,
            coordinates=lrom_coordinates,
        )
        pointwise_metrics[channel] = {
            "lrom": pointwise_absolute(
                prediction=lrom_wavefunctions[channel],
                reference=high_fidelity[channel],
            )
        }
        relative_metrics[channel] = {
            "lrom": relative_l2(
                prediction=lrom_wavefunctions[channel],
                reference=high_fidelity[channel],
            )
        }
    return TestingResults(
        high_fidelity=high_fidelity,
        lrom=lrom_wavefunctions,
        ls=None,
        coefficients={"lrom": lrom_coefficients},
        metrics={
            "relative_l2": relative_metrics,
            "pointwise_absolute": pointwise_metrics,
        },
    )


def _reduced_basis_state(*, emulator, basis_size: int) -> TrainingState:
    """Build only the centered basis state for each requested channel."""
    channels = _trained_channels(emulator)
    bases = {
        channel: _centered_basis(
            emulator=emulator, channel=channel, basis_size=basis_size
        )
        for channel in channels
    }
    return TrainingState(
        basis=bases,
        predictors=None,
        rf_lrom={},
        testing_results=None,
        testing_errors={channel: {} for channel in channels},
        training_options={"basis_size": basis_size},
    )


def _train_state(
    *, emulator, basis_size: int, predictor: str, predictor_count: int
) -> TrainingState:
    """Build the basis and predictor, then fit and evaluate RF-LROM."""
    basis_only = _reduced_basis_state(emulator=emulator, basis_size=basis_size)
    predictor_state = _predictor(
        emulator=emulator, kind=predictor, count=predictor_count
    )
    samples = emulator.samples
    bases = basis_only.basis
    rf_models = {}
    for channel in _trained_channels(emulator):
        basis = bases[channel]
        # Required RF-LROM step: the high-fidelity training snapshots become
        # reduced coordinates before the reduced equation can be learned.
        train_coordinates = project_coordinates(
            basis=basis, wavefunctions=samples.training_wavefunctions[channel]
        )
        rf_models[channel] = fit_rf_lrom(
            predictors=predictor_state.training_features,
            coordinates=train_coordinates,
        )
    training_results = _evaluate(
        emulator=emulator,
        bases=bases,
        rf_models=rf_models,
        wavefunctions=samples.training_wavefunctions,
        predictor_features=predictor_state.training_features,
    )
    testing_results = _evaluate(
        emulator=emulator,
        bases=bases,
        rf_models=rf_models,
        wavefunctions=samples.testing_wavefunctions,
        predictor_features=predictor_state.testing_features,
    )
    return TrainingState(
        basis=bases,
        predictors=predictor_state,
        rf_lrom=rf_models,
        testing_results=testing_results,
        testing_errors=testing_results.metrics["pointwise_absolute"],
        training_results=training_results,
        training_options={
            "basis_size": basis_size,
            "predictor": predictor,
            "predictor_count": predictor_count,
        },
    )


# ==========================================================================
# 9. RF-LROM prediction
# Owns conversion of named inputs into features, coefficients, and phi0+Phi*a.
# Does not run ROSE or modify trained operators.
# ==========================================================================

def _parameter_rows(*, emulator, parameters) -> np.ndarray:
    rows = [parameters] if isinstance(parameters, Mapping) else list(parameters)
    if not rows:
        raise ValueError("parameters must contain at least one case")
    central = dict(emulator.central_parameters)
    result = []
    for row in rows:
        unknown = sorted(set(row) - set(emulator.parameter_names))
        if unknown:
            raise ValueError(f"unknown parameter names: {unknown}")
        merged = central | {name: float(value) for name, value in row.items()}
        result.append([merged[name] for name in emulator.parameter_names])
    values = np.asarray(result, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("prediction parameters must be finite")
    return values



def predict(*, emulator, parameters) -> PredictionState:
    """Predict one or more named parameter cases from trained portable state."""
    values = _parameter_rows(emulator=emulator, parameters=parameters)
    predictor = emulator.predictors
    features = features_for_values(
        predictor=predictor,
        values=values,
        potential_function=emulator.config.potential.function,
    )
    coefficients = {
        channel: solve_rf_lrom(model=model, predictors=features)
        for channel, model in emulator.rf_lrom.items()
    }
    wavefunctions = {
        channel: reconstruct(
            basis=emulator.basis[channel], coordinates=coordinates
        )
        for channel, coordinates in coefficients.items()
    }
    return PredictionState(
        parameter_names=emulator.parameter_names,
        parameters=values,
        coefficients=coefficients,
        wavefunctions=wavefunctions,
    )


# ==========================================================================
# 10. Portable artifacts
# Owns safe serialization of prediction-critical arrays and provenance.
# Does not serialize live ROSE objects or full sampling data.
# ==========================================================================

ARTIFACT_SCHEMA = 1


def _json_config(emulator: LROM) -> dict[str, object]:
    potential_name = emulator.config.potential.name
    if potential_name not in {"ws_1", "ws_3", "woods-saxon"}:
        raise LROMArtifactError(
            "portable artifacts require a registered potential name"
        )
    return {
        "target": list(emulator.config.target),
        "projectile": list(emulator.config.projectile),
        "lab_energy": emulator.config.lab_energy,
        "channels": list(emulator.partial_waves),
        "fom": emulator.config.fom,
        "potential": potential_name,
        "central_parameters": dict(emulator.central_parameters),
        "parameter_names": list(emulator.parameter_names),
    }


def save_artifact(*, path: str | Path, emulator: LROM) -> None:
    """Write prediction-critical state without pickle or live ROSE objects."""
    if emulator.mesh is None or emulator.kinematics is None:
        raise LROMArtifactError("trained emulator is missing mesh or kinematics")
    config = _json_config(emulator)
    predictor = emulator.predictors
    arrays: dict[str, np.ndarray] = {
        "mesh_rho": np.asarray(emulator.mesh.rho),
        "mesh_radius": np.asarray(emulator.mesh.radius),
        "predictor_parameter_indices": np.asarray(predictor.parameter_indices),
        "predictor_center": np.asarray(predictor.center),
        "predictor_scales": np.asarray(predictor.scales),
        "predictor_selected_indices": np.asarray(predictor.selected_indices),
        "predictor_selected_radii": np.asarray(predictor.selected_radii),
        "predictor_central_values": np.asarray(predictor.central_values),
        "predictor_singular_values": np.asarray(predictor.singular_values),
    }
    channels: dict[str, dict[str, object]] = {}
    for channel in emulator.partial_waves:
        basis = emulator.basis[channel]
        model = emulator.rf_lrom[channel]
        prefix = f"l{channel}"
        arrays[f"{prefix}_phi0"] = np.asarray(basis.phi0)
        arrays[f"{prefix}_basis_vectors"] = np.asarray(basis.vectors)
        arrays[f"{prefix}_basis_singular_values"] = np.asarray(
            basis.singular_values
        )
        arrays[f"{prefix}_rf_matrices"] = np.asarray(model.matrices)
        arrays[f"{prefix}_rf_vectors"] = np.asarray(model.vectors)
        arrays[f"{prefix}_rf_singular_values"] = np.asarray(
            model.singular_values
        )
        channels[str(channel)] = {
            "residual_mse": model.residual_mse,
            "rank": model.rank,
        }
    config_hash = sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    metadata = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "package_version": "0.1.0",
        "config_hash": config_hash,
        "config": config,
        "kinematics": {
            "mu": emulator.kinematics.mu,
            "e_com": emulator.kinematics.e_com,
            "k": emulator.kinematics.k,
            "eta": emulator.kinematics.eta,
            "coulomb_radius": emulator.kinematics.coulomb_radius,
        },
        "predictor": {
            "kind": predictor.kind,
            "names": list(predictor.names),
            "parameter_names": list(predictor.parameter_names),
        },
        "channels": channels,
        "training_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    array_buffer = io.BytesIO()
    np.savez_compressed(array_buffer, **arrays)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            destination, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr("metadata.json", json.dumps(metadata, indent=2))
            archive.writestr("arrays.npz", array_buffer.getvalue())
    except OSError as exc:
        raise LROMArtifactError(f"could not write artifact {destination}") from exc


def _required_array(arrays, name: str) -> np.ndarray:
    try:
        value = np.asarray(arrays[name])
    except KeyError as exc:
        raise LROMArtifactError(f"artifact is missing array {name!r}") from exc
    if value.dtype == object:
        raise LROMArtifactError(f"artifact array {name!r} has unsafe object dtype")
    return value


def load_artifact(*, path: str | Path) -> LROM:
    """Load a portable prediction-only emulator."""
    source = Path(path)
    try:
        with zipfile.ZipFile(source) as archive:
            if set(archive.namelist()) != {"metadata.json", "arrays.npz"}:
                raise LROMArtifactError("artifact must contain metadata.json and arrays.npz")
            metadata = json.loads(archive.read("metadata.json"))
            array_bytes = archive.read("arrays.npz")
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise LROMArtifactError(f"invalid LROM artifact {source}") from exc
    if metadata.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise LROMArtifactError(
            f"unsupported artifact schema {metadata.get('artifact_schema')!r}"
        )
    try:
        arrays = np.load(io.BytesIO(array_bytes), allow_pickle=False)
        config = metadata["config"]
        emulator = LROM(
            target=tuple(config["target"]),
            projectile=tuple(config["projectile"]),
            lab_energy=float(config["lab_energy"]),
            l=tuple(config["channels"]),
            fom=config["fom"],
            potential=config["potential"],
            central_parameters=config["central_parameters"],
        )
        emulator._central_parameters = MappingProxyType(
            {name: float(value) for name, value in config["central_parameters"].items()}
        )
        kin = metadata["kinematics"]
        emulator._kinematics = Kinematics(
            mu=float(kin["mu"]),
            e_com=float(kin["e_com"]),
            k=float(kin["k"]),
            eta=float(kin["eta"]),
            coulomb_radius=float(kin["coulomb_radius"]),
        )
        emulator._portable_mesh = MeshState(
            rho=_required_array(arrays, "mesh_rho"),
            radius=_required_array(arrays, "mesh_radius"),
        )
        predictor_meta = metadata["predictor"]
        predictor = PredictorState(
            kind=predictor_meta["kind"],
            names=tuple(predictor_meta["names"]),
            parameter_names=tuple(predictor_meta["parameter_names"]),
            parameter_indices=_required_array(
                arrays, "predictor_parameter_indices"
            ).astype(int),
            center=_required_array(arrays, "predictor_center"),
            scales=_required_array(arrays, "predictor_scales"),
            training_features=np.empty((0, 0)),
            testing_features=np.empty((0, 0)),
            selected_indices=_required_array(
                arrays, "predictor_selected_indices"
            ).astype(int),
            selected_radii=_required_array(arrays, "predictor_selected_radii"),
            central_values=_required_array(arrays, "predictor_central_values"),
            singular_values=_required_array(
                arrays, "predictor_singular_values"
            ),
        )
        bases = {}
        models = {}
        for channel_text, model_meta in metadata["channels"].items():
            channel = int(channel_text)
            prefix = f"l{channel}"
            bases[channel] = BasisState(
                phi0=_required_array(arrays, f"{prefix}_phi0"),
                vectors=_required_array(arrays, f"{prefix}_basis_vectors"),
                radius=emulator._portable_mesh.radius,
                singular_values=_required_array(
                    arrays, f"{prefix}_basis_singular_values"
                ),
            )
            models[channel] = RFLROMModel(
                matrices=_required_array(arrays, f"{prefix}_rf_matrices"),
                vectors=_required_array(arrays, f"{prefix}_rf_vectors"),
                residual_mse=float(model_meta["residual_mse"]),
                rank=int(model_meta["rank"]),
                singular_values=_required_array(
                    arrays, f"{prefix}_rf_singular_values"
                ),
            )
        emulator._training_state = TrainingState(
            basis=bases,
            predictors=predictor,
            rf_lrom=models,
            testing_results=None,
            testing_errors={channel: {} for channel in bases},
        )
        emulator._inference_only = True
        emulator._provenance = {
            "artifact_schema": ARTIFACT_SCHEMA,
            "package_version": metadata["package_version"],
            "config_hash": metadata["config_hash"],
            "training_environment": metadata["training_environment"],
        }
        arrays.close()
        return emulator
    except LROMArtifactError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise LROMArtifactError(f"invalid LROM artifact {source}") from exc


# ==========================================================================
# 11. Public LROM lifecycle
# Owns LROM -> sampling -> train -> predict -> save/load state transitions.
# Delegates numerical work to the flat functions above.
# ==========================================================================

class LROM:
    """Stateful learned reduced-operator model workflow."""

    def __init__(
        self,
        *,
        target: tuple[int, int],
        projectile: tuple[int, int],
        lab_energy: float,
        l: int | tuple[int, ...] = 0,  # noqa: E741 - standard partial-wave symbol
        fom: str = "nucl-scatter-eq",
        potential: str | PotentialFunction = "ws_3",
        central_parameters: Mapping[str, float] | None = None,
    ) -> None:
        self._config = LROMConfig.create(
            target=target,
            projectile=projectile,
            lab_energy=lab_energy,
            l=l,
            fom=fom,
            potential=potential,
            central_parameters=central_parameters,
        )
        self._central_parameters: Mapping[str, float] = self._config.central_overrides
        self._kinematics: Kinematics | None = None
        self._sampling_state: SamplingState | None = None
        self._portable_mesh: MeshState | None = None
        self._training_state: TrainingState | None = None
        self._prediction_state: Any = None
        self._fom_provider: Any = None
        self._inference_only = False
        self._provenance: dict[str, Any] = {}

    @property
    def config(self) -> LROMConfig:
        return self._config

    @property
    def kinematics(self) -> Kinematics | None:
        self._ensure_physics_state()
        return self._kinematics

    @property
    def central_parameters(self) -> Mapping[str, float]:
        self._ensure_physics_state()
        return self._central_parameters

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self._config.parameter_names

    @property
    def sampleable_parameters(self) -> tuple[str, ...]:
        return self._config.sampleable_names

    @property
    def partial_waves(self) -> tuple[int, ...]:
        return self._config.channels

    @property
    def description(self) -> Mapping[str, str]:
        return self._config.description

    @property
    def samples(self) -> SamplingState | None:
        return self._sampling_state

    @property
    def mesh(self) -> MeshState | None:
        if self._sampling_state is not None:
            return self._sampling_state.mesh
        return self._portable_mesh

    @property
    def full_order_model(self) -> Mapping[int, Any] | None:
        return (
            None
            if self._sampling_state is None
            else self._sampling_state.full_order_models
        )

    @property
    def basis(self) -> Mapping[int, Any] | None:
        return None if self._training_state is None else self._training_state.basis

    @property
    def predictors(self) -> Any:
        return None if self._training_state is None else self._training_state.predictors

    @property
    def rf_lrom(self) -> Mapping[int, Any] | None:
        return None if self._training_state is None else self._training_state.rf_lrom

    @property
    def testing_results(self) -> Any:
        return None if self._training_state is None else self._training_state.testing_results

    @property
    def training_results(self) -> Any:
        return None if self._training_state is None else self._training_state.training_results

    @property
    def training_options(self) -> Mapping[str, Any] | None:
        return None if self._training_state is None else self._training_state.training_options

    @property
    def testing_errors(self) -> Mapping[int, Any] | None:
        return None if self._training_state is None else self._training_state.testing_errors

    @property
    def predictions(self) -> Any:
        return self._prediction_state

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self._provenance

    @property
    def is_sampled(self) -> bool:
        return self._sampling_state is not None

    @property
    def is_trained(self) -> bool:
        return self._training_state is not None

    @property
    def can_predict(self) -> bool:
        return self.is_trained

    def _provider(self) -> Any:
        if self._fom_provider is None:
            self._fom_provider = NuclearScatteringFOM()
        return self._fom_provider

    def _ensure_physics_state(self) -> None:
        if self._kinematics is not None:
            return
        central, kinematics = self._provider().resolve(config=self._config)
        self._central_parameters = central
        self._kinematics = kinematics

    def _clear_training_state(self) -> None:
        self._training_state = None
        self._clear_prediction_state()

    def _clear_prediction_state(self) -> None:
        self._prediction_state = None

    def sampling(
        self,
        *,
        training_ranges: Mapping[str, tuple[float, float]] | None = None,
        testing_ranges: Mapping[str, tuple[float, float]] | None = None,
        training_size: int | None = None,
        testing_size: int | None = None,
        training_grid: Mapping[str, Sequence[float]] | None = None,
        testing_grid: Mapping[str, Sequence[float]] | None = None,
        mesh_size: int = 900,
        radial_domain: tuple[float, float] | None = None,
        strategy: str | None = None,
        seed: int | None = None,
        high_fidelity_solver: str = "runge_kutta",
        solver_options: Mapping[str, object] | None = None,
    ) -> None:
        if self._inference_only:
            raise LROMStateError("a portable inference artifact cannot run sampling")
        provider = self._provider()
        central, kinematics = provider.resolve(config=self._config)
        grid_mode = training_grid is not None or testing_grid is not None
        if grid_mode:
            if training_grid is None or testing_grid is None:
                raise LROMSamplingError(
                    "training_grid and testing_grid must both be provided"
                )
            if any(
                value is not None
                for value in (
                    training_ranges,
                    testing_ranges,
                    training_size,
                    testing_size,
                    strategy,
                    seed,
                )
            ):
                raise LROMSamplingError(
                    "explicit grids cannot be combined with ranges, sizes, strategy, or seed"
                )
            design = create_explicit_sampling_design(
                parameter_names=self.parameter_names,
                sampleable_names=self.sampleable_parameters,
                central=central,
                training_grid=training_grid,
                testing_grid=testing_grid,
            )
        else:
            if any(
                value is None
                for value in (
                    training_ranges,
                    testing_ranges,
                    training_size,
                    testing_size,
                )
            ):
                raise LROMSamplingError(
                    "range sampling requires training/testing ranges and sizes"
                )
            design = create_sampling_design(
                parameter_names=self.parameter_names,
                sampleable_names=self.sampleable_parameters,
                central=central,
                training_ranges=training_ranges,
                testing_ranges=testing_ranges,
                training_size=training_size,
                testing_size=testing_size,
                strategy=strategy or "latin_hypercube",
                seed=seed,
            )
        state = provider.sample(
            config=self._config,
            design=design,
            mesh_size=mesh_size,
            radial_domain=radial_domain,
            high_fidelity_solver=high_fidelity_solver,
            solver_options=solver_options,
        )
        self._sampling_state = state
        self._central_parameters = state.central_parameters
        self._kinematics = state.kinematics
        self._clear_training_state()

    def reduced_basis(self, *, basis_size: int) -> None:
        if not self.is_sampled:
            raise LROMStateError("call sampling() before reduced_basis()")
        self._training_state = _reduced_basis_state(
            emulator=self, basis_size=basis_size
        )
        self._clear_prediction_state()

    def train(
        self,
        *,
        basis_size: int = 4,
        predictor: str = "potential",
        predictor_count: int = 6,
    ) -> None:
        if self._inference_only:
            raise LROMStateError("a portable inference artifact cannot be retrained")
        if not self.is_sampled:
            raise LROMStateError("call sampling() before train()")
        self._training_state = _train_state(
            emulator=self,
            basis_size=basis_size,
            predictor=predictor,
            predictor_count=predictor_count,
        )
        self._clear_prediction_state()

    def predict(
        self,
        *,
        parameters: Mapping[str, float] | Sequence[Mapping[str, float]],
    ) -> None:
        if not self.can_predict:
            raise LROMStateError("call train() before predict()")
        self._prediction_state = predict(emulator=self, parameters=parameters)

    def testing_case(self, *, case_id: str) -> TestingCase:
        if self._training_state is None or self._sampling_state is None:
            raise LROMStateError("call train() before requesting a testing case")
        try:
            index = self._sampling_state.design.testing.case_ids.index(case_id)
        except ValueError as exc:
            raise LROMStateError(f"unknown testing case_id {case_id!r}") from exc
        results = self._training_state.testing_results
        return TestingCase(
            case_id=case_id,
            parameters=self._sampling_state.design.testing.named(index=index),
            radius=self._sampling_state.mesh.radius,
            high_fidelity={
                channel: values[index]
                for channel, values in results.high_fidelity.items()
            },
            lrom={channel: values[index] for channel, values in results.lrom.items()},
            ls=None,
        )

    def save(self, *, path: str | Path) -> None:
        if not self.can_predict:
            raise LROMStateError("call train() before save()")
        save_artifact(path=path, emulator=self)


# -- Public module surface -------------------------------------------------

__version__ = "1.2.0"


def load(*, path: str | Path) -> LROM:
    """Load a portable trained emulator."""
    return load_artifact(path=path)


__all__ = [
    "BasisState",
    "KD_PARAMETER_NAMES",
    "Kinematics",
    "LROM",
    "LROMArtifactError",
    "LROMConfig",
    "LROMConfigurationError",
    "LROMError",
    "LROMSamplingError",
    "LROMStateError",
    "MeshState",
    "ParameterCases",
    "PotentialSpec",
    "PredictionState",
    "PredictorState",
    "RFLROMModel",
    "SamplingDesign",
    "SamplingState",
    "TestingCase",
    "TestingResults",
    "TrainingState",
    "build_basis",
    "build_parameter_predictor",
    "build_potential_predictor",
    "create_explicit_sampling_design",
    "create_sampling_design",
    "custom_potential_spec",
    "features_for_values",
    "fit_rf_lrom",
    "least_squares_baseline",
    "load",
    "pointwise_absolute",
    "project_coordinates",
    "real_woods_saxon",
    "reconstruct",
    "relative_l2",
    "resolve_potential",
    "solve_rf_lrom",
]
