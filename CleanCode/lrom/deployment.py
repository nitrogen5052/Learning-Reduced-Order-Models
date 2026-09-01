"""Low-overhead deployment helpers for hard-routed local LROMs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .emulator import ScatteringLROM
from .observables import ElasticCrossSectionKernel


class HardRoutedScatteringLROM:
    """Deploy independent energy-window LROMs with deterministic hard routing.

    The angular observable kernel is built once. Batched calls are grouped by
    window so every local LROM evaluates all of its assigned samples in one
    stacked numerical operation. Only the selected window is evaluated for a
    scalar query.
    """

    def __init__(
        self,
        models: Sequence[ScatteringLROM],
        energy_edges: Sequence[float],
        angles: Sequence[float],
        maximum_batch_size: int = 48,
        linear_solver=None,
    ):
        self.models = tuple(models)
        self.energy_edges = np.asarray(energy_edges, dtype=float)
        self.angles = np.ascontiguousarray(np.asarray(angles, dtype=float).reshape(-1))
        self.maximum_batch_size = int(maximum_batch_size)
        self.linear_solver = linear_solver
        if not self.models:
            raise ValueError("at least one local LROM is required")
        if len(self.energy_edges) != len(self.models) + 1:
            raise ValueError("energy_edges must contain one more value than models")
        if not np.all(np.diff(self.energy_edges) > 0.0):
            raise ValueError("energy_edges must be strictly increasing")
        if not len(self.angles):
            raise ValueError("at least one scattering angle is required")
        if self.maximum_batch_size < 1:
            raise ValueError("maximum_batch_size must be positive")

        l_max = self.models[0].problem.l_max
        if any(model.problem.l_max != l_max for model in self.models):
            raise ValueError("all local LROMs must use the same l_max")
        if any(not model.channel_models for model in self.models):
            raise RuntimeError("all local LROMs must be trained before deployment")

        # Precompute the energy-independent angular sums and seed each model's
        # cache. This also removes first-query setup noise from online timing.
        self._angle_key = self.angles.tobytes()
        self._angular_kernel = ElasticCrossSectionKernel.build(
            self.angles, 1.0, l_max
        )
        for model in self.models:
            model._angular_kernels[self._angle_key] = self._angular_kernel

    def route_energy(self, energy: float) -> int:
        """Return the hard-window index, including the two endpoint windows."""
        return int(np.searchsorted(
            self.energy_edges[1:-1], float(energy), side="right"
        ))

    def cross_section(self, sample: Mapping[str, float | int]) -> np.ndarray:
        """Evaluate exactly one local LROM for one sample."""
        return self.models[self.route_energy(sample["E"])].cross_section(
            sample, self.angles
        )

    def cross_sections(
        self, samples: Sequence[Mapping[str, float | int]]
    ) -> np.ndarray:
        """Group samples by energy window and evaluate each nonempty group once."""
        if not samples:
            return np.empty((0, len(self.angles)), dtype=float)
        energies = np.fromiter(
            (float(sample["E"]) for sample in samples),
            dtype=float, count=len(samples),
        )
        assignments = np.searchsorted(
            self.energy_edges[1:-1], energies, side="right"
        )
        output = np.empty((len(samples), len(self.angles)), dtype=float)
        for index, model in enumerate(self.models):
            selected = np.flatnonzero(assignments == index)
            for start in range(0, len(selected), self.maximum_batch_size):
                chunk = selected[start:start + self.maximum_batch_size]
                output[chunk] = model.cross_sections(
                    [samples[position] for position in chunk], self.angles
                )
        return output

    def partial_wave_s_matrices(
        self, samples: Sequence[Mapping[str, float | int]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return routed ``S+``, ``S-``, and wave numbers without angle work.

        This is the preferred route when each requested physical case has a
        different experimental angle grid: solve each case once, then apply a
        precomputed observation-specific angular kernel downstream.
        """
        if not samples:
            width = self.models[0].problem.l_max + 1
            return (
                np.empty((0, width), complex), np.empty((0, width), complex),
                np.empty(0, float),
            )
        energies = np.fromiter(
            (float(sample["E"]) for sample in samples),
            dtype=float, count=len(samples),
        )
        assignments = np.searchsorted(
            self.energy_edges[1:-1], energies, side="right"
        )
        width = self.models[0].problem.l_max + 1
        plus = np.empty((len(samples), width), complex)
        minus = np.empty_like(plus)
        wave_numbers = np.empty(len(samples), float)
        for index, model in enumerate(self.models):
            packed = model._packed_online
            if not hasattr(packed, "s_matrices_batch"):
                raise RuntimeError(
                    "partial-wave deployment requires padded/equal-shape packing"
                )
            selected = np.flatnonzero(assignments == index)
            for start in range(0, len(selected), self.maximum_batch_size):
                chunk = selected[start:start + self.maximum_batch_size]
                channels, k_values = packed.s_matrices_batch(
                    model.problem,
                    [samples[position] for position in chunk],
                    model.control_inputs,
                    linear_solver=(
                        None if self.linear_solver is None
                        else self.linear_solver.solve
                    ),
                )
                plus[chunk] = channels[:, packed.plus_indices]
                minus[chunk] = channels[:, packed.minus_indices]
                wave_numbers[chunk] = k_values
        return plus, minus, wave_numbers

    def partial_wave_s_matrices_gpu(
        self,
        samples: Sequence[Mapping[str, float | int]],
        solver,
        maximum_batch_size: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return partial waves with reduced solves delegated to a GPU solver.

        Feature and observable construction remain on the CPU.  This targets
        the batched dense solves, which dominate calibration likelihood time.
        ``solver`` must expose ``solve(matrices, right_sides)``.
        """
        if not samples:
            return self.partial_wave_s_matrices(samples)
        energies = np.fromiter(
            (float(sample["E"]) for sample in samples),
            dtype=float, count=len(samples),
        )
        assignments = np.searchsorted(
            self.energy_edges[1:-1], energies, side="right"
        )
        width = self.models[0].problem.l_max + 1
        plus = np.empty((len(samples), width), complex)
        minus = np.empty_like(plus)
        wave_numbers = np.empty(len(samples), float)
        for index, model in enumerate(self.models):
            packed = model._packed_online
            if not hasattr(packed, "feature_batch"):
                raise RuntimeError(
                    "GPU partial-wave deployment requires padded/equal-shape packing"
                )
            selected = np.flatnonzero(assignments == index)
            if not len(selected):
                continue
            for start in range(0, len(selected), maximum_batch_size):
                chunk = selected[start:start + maximum_batch_size]
                features, k_values = packed.feature_batch(
                    model.problem,
                    [samples[position] for position in chunk],
                    model.control_inputs,
                )
                channels = packed.s_matrices_from_feature_batch(
                    features, linear_solver=solver.solve
                )
                plus[chunk] = channels[:, packed.plus_indices]
                minus[chunk] = channels[:, packed.minus_indices]
                wave_numbers[chunk] = k_values
        return plus, minus, wave_numbers
