"""Small state containers owned by :class:`lrom.LROM`."""

from __future__ import annotations

from dataclasses import dataclass

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
