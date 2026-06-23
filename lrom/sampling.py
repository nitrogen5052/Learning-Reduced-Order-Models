"""Named parameter-space sampling without public positional conventions."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
from scipy.stats import qmc

from .errors import LROMSamplingError
from .state import ParameterCases, SamplingDesign


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
