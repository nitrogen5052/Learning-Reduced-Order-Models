"""Sampling and reusable full-order training databases."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .channels import Channel
from .reduced import latin_hypercube


@dataclass(frozen=True)
class InputSpace:
    """Named continuous, integer-valued, and fixed model inputs.

    Integer inputs such as ``A`` and ``Z`` remain ordered numerical variables;
    the distinction only prevents sampling or accepting nonphysical fractions.
    """

    continuous: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    integers: Mapping[str, Sequence[int]] = field(default_factory=dict)
    fixed: Mapping[str, float | int] = field(default_factory=dict)

    def sample(self, count: int, seed: int = 0) -> list[dict[str, float | int]]:
        """Draw reproducible samples and return one named dictionary per case."""
        if count <= 0:
            raise ValueError("count must be positive")
        names = tuple(self.continuous)
        if names:
            bounds = np.asarray([self.continuous[name] for name in names], float)
            continuous_rows = latin_hypercube(
                bounds[:, 0], bounds[:, 1], count, seed
            )
        else:
            continuous_rows = np.empty((count, 0))
        rng = np.random.default_rng(seed + 7919)
        rows: list[dict[str, float | int]] = []
        for index in range(count):
            row = dict(self.fixed)
            row.update({name: float(continuous_rows[index, column])
                        for column, name in enumerate(names)})
            for name, allowed in self.integers.items():
                values = np.asarray(allowed, dtype=int)
                if values.size == 0:
                    raise ValueError(f"integer input {name!r} has no allowed values")
                row[name] = int(rng.choice(values))
            rows.append(row)
        return rows

    def validate(self, samples: Sequence[Mapping[str, float | int]]) -> None:
        """Validate names, ranges, and integrality without changing samples."""
        required = set(self.continuous) | set(self.integers) | set(self.fixed)
        for sample in samples:
            missing = required.difference(sample)
            if missing:
                raise ValueError(f"sample is missing inputs: {sorted(missing)}")
            for name, (lower, upper) in self.continuous.items():
                if not lower <= float(sample[name]) <= upper:
                    raise ValueError(f"{name} lies outside its input range")
            for name, allowed in self.integers.items():
                value = sample[name]
                if int(value) != value or int(value) not in set(map(int, allowed)):
                    raise ValueError(f"{name} must be one of {list(allowed)}")


@dataclass
class TrainingData:
    """Reusable scaled-coordinate FOM solutions and operator snapshots."""

    samples: list[dict[str, float | int]]
    reference_sample: dict[str, float | int]
    s_mesh: np.ndarray
    channels: tuple[Channel, ...]
    wavefunctions: dict[Channel, np.ndarray]
    reference_wavefunctions: dict[Channel, np.ndarray]
    potentials: dict[Channel, np.ndarray]
    reference_potentials: dict[Channel, np.ndarray]
    s_matrices: dict[Channel, np.ndarray]
    reference_s_matrices: dict[Channel, complex]
    physical_radii: np.ndarray | None = None
    reference_radius: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def radius(self) -> np.ndarray:
        """Deprecated alias for the common coordinate mesh.

        The shared LROM coordinate is now ``s=k r``; use :attr:`s_mesh` in new
        code. Physical radii vary by sample and live in ``physical_radii``.
        """
        return self.s_mesh

    def save(self, path: str | Path) -> None:
        """Save the complete database as one compressed, portable NPZ file."""
        arrays: dict[str, np.ndarray] = {"s_mesh": np.asarray(self.s_mesh)}
        if self.physical_radii is not None:
            arrays["physical_radii"] = np.asarray(self.physical_radii)
        if self.reference_radius is not None:
            arrays["reference_radius"] = np.asarray(self.reference_radius)
        for index, channel in enumerate(self.channels):
            arrays[f"wave_{index}"] = self.wavefunctions[channel]
            arrays[f"reference_wave_{index}"] = self.reference_wavefunctions[channel]
            arrays[f"potential_{index}"] = self.potentials[channel]
            arrays[f"reference_potential_{index}"] = self.reference_potentials[channel]
            arrays[f"s_{index}"] = self.s_matrices[channel]
            arrays[f"reference_s_{index}"] = np.asarray(self.reference_s_matrices[channel])
        description = {
            "samples": self.samples,
            "reference_sample": self.reference_sample,
            "channels": [list(channel) for channel in self.channels],
            "metadata": self.metadata,
        }
        arrays["description_json"] = np.asarray(json.dumps(description))
        np.savez_compressed(Path(path), **arrays)

    def subset(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        reference_index: int | None = None,
    ) -> "TrainingData":
        """Return selected cases without repeating any full-order solves.

        ``reference_index`` refers to a case in the original database and must
        also be retained.  Choosing a local reference is useful for an
        energy-window emulator whose basis should be centered in that window.
        """
        selected = np.asarray(indices, dtype=int).reshape(-1)
        if selected.size == 0:
            raise ValueError("a training-data subset cannot be empty")
        if np.any(selected < 0) or np.any(selected >= len(self.samples)):
            raise IndexError("training-data subset index is out of range")

        if reference_index is None:
            reference_sample = dict(self.reference_sample)
            reference_wavefunctions = {
                channel: np.asarray(values).copy()
                for channel, values in self.reference_wavefunctions.items()
            }
            reference_potentials = {
                channel: np.asarray(values).copy()
                for channel, values in self.reference_potentials.items()
            }
            reference_s_matrices = dict(self.reference_s_matrices)
            reference_radius = (
                None if self.reference_radius is None
                else np.asarray(self.reference_radius).copy()
            )
        else:
            reference_index = int(reference_index)
            if reference_index not in set(selected.tolist()):
                raise ValueError("reference_index must be retained by indices")
            reference_sample = dict(self.samples[reference_index])
            reference_wavefunctions = {
                channel: np.asarray(values[reference_index]).copy()
                for channel, values in self.wavefunctions.items()
            }
            reference_potentials = {
                channel: np.asarray(values[reference_index]).copy()
                for channel, values in self.potentials.items()
            }
            reference_s_matrices = {
                channel: complex(values[reference_index])
                for channel, values in self.s_matrices.items()
            }
            reference_radius = (
                None if self.physical_radii is None
                else np.asarray(self.physical_radii[reference_index]).copy()
            )

        return TrainingData(
            samples=[dict(self.samples[index]) for index in selected],
            reference_sample=reference_sample,
            s_mesh=np.asarray(self.s_mesh).copy(),
            channels=tuple(self.channels),
            wavefunctions={
                channel: np.asarray(values[selected]).copy()
                for channel, values in self.wavefunctions.items()
            },
            reference_wavefunctions=reference_wavefunctions,
            potentials={
                channel: np.asarray(values[selected]).copy()
                for channel, values in self.potentials.items()
            },
            reference_potentials=reference_potentials,
            s_matrices={
                channel: np.asarray(values[selected]).copy()
                for channel, values in self.s_matrices.items()
            },
            reference_s_matrices=reference_s_matrices,
            physical_radii=(
                None if self.physical_radii is None
                else np.asarray(self.physical_radii[selected]).copy()
            ),
            reference_radius=reference_radius,
            metadata={**self.metadata, "subset_size": int(selected.size)},
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainingData":
        """Load a database previously written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            description = json.loads(str(archive["description_json"]))
            channels = tuple((int(c[0]), float(c[1])) for c in description["channels"])
            return cls(
                samples=description["samples"],
                reference_sample=description["reference_sample"],
                s_mesh=(archive["s_mesh"] if "s_mesh" in archive else archive["radius"]),
                channels=channels,
                wavefunctions={c: archive[f"wave_{i}"] for i, c in enumerate(channels)},
                reference_wavefunctions={c: archive[f"reference_wave_{i}"] for i, c in enumerate(channels)},
                potentials={c: archive[f"potential_{i}"] for i, c in enumerate(channels)},
                reference_potentials={c: archive[f"reference_potential_{i}"] for i, c in enumerate(channels)},
                s_matrices={c: archive[f"s_{i}"] for i, c in enumerate(channels)},
                reference_s_matrices={c: complex(archive[f"reference_s_{i}"].item()) for i, c in enumerate(channels)},
                physical_radii=(archive["physical_radii"] if "physical_radii" in archive else None),
                reference_radius=(archive["reference_radius"] if "reference_radius" in archive else None),
                metadata=description["metadata"],
            )
