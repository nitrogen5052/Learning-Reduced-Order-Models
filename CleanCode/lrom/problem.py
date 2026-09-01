"""High-level scattering problem and independent FOM data generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .channels import Channel, elastic_channels
from .data import TrainingData
from .fom import NumerovFOM, ScatteringSystem, scaled_potential
from .observables import (
    assemble_elastic_cross_sections,
    assemble_varying_energy_cross_sections,
)
from .physics import PotentialModel, potential_model, two_body_kinematics


@dataclass
class ScatteringProblem:
    """Physics definition shared by FOM generation and learned emulators."""

    potential: PotentialModel | str = "full_woods_saxon"
    target_a: int = 40
    target_z: int = 20
    projectile_a: int = 1
    projectile_z: int = 0
    lab_energy: float = 14.1
    l_max: int = 10
    mesh_points: int = 800
    s_max: float = 8.0 * np.pi
    matching_s: float = 6.0 * np.pi
    target_binding_energy: float | None = 342.052184
    reference_omega: np.ndarray | None = None
    included_channels: tuple[Channel, ...] | None = None

    def __post_init__(self) -> None:
        if self.mesh_points < 5:
            raise ValueError("mesh_points must be at least five")
        if not 0.0 < self.matching_s < self.s_max:
            raise ValueError("matching_s must lie strictly inside the scaled mesh")
        if isinstance(self.potential, str):
            self.potential = potential_model(self.potential)
        if self.reference_omega is None:
            self.reference_omega = self.potential.parameters_for(
                self.target_a, self.target_z, self.lab_energy
            )
        self.reference_omega = np.asarray(self.reference_omega, dtype=float)
        if len(self.reference_omega) != len(self.potential.parameter_names):
            raise ValueError("reference_omega does not match the potential model")
        self._system_cache: dict[tuple[int, int, float], ScatteringSystem] = {}

    @property
    def channels(self) -> tuple[Channel, ...]:
        """Return every partial-wave channel included by this problem."""
        if self.included_channels is not None:
            return tuple((int(ell), float(spin)) for ell, spin in self.included_channels)
        return elastic_channels(self.l_max)

    @property
    def reference_sample(self) -> dict[str, float | int]:
        """Return the central named input selection."""
        sample = {
            name: float(value)
            for name, value in zip(self.potential.parameter_names, self.reference_omega)
        }
        sample.update(E=float(self.lab_energy), A=int(self.target_a), Z=int(self.target_z))
        return sample

    def omega(self, sample: Mapping[str, float | int]) -> np.ndarray:
        """Extract the potential parameter vector from a named sample."""
        if all(name in sample for name in self.potential.parameter_names):
            return np.fromiter(
                (sample[name] for name in self.potential.parameter_names),
                dtype=float,
                count=len(self.potential.parameter_names),
            )
        a = int(sample.get("A", self.target_a))
        z = int(sample.get("Z", self.target_z))
        energy = float(sample.get("E", self.lab_energy))
        mapped = self.potential.parameters_for(a, z, energy)
        return np.asarray([
            sample.get(name, mapped[index])
            for index, name in enumerate(self.potential.parameter_names)
        ], dtype=float)

    def system(self, sample: Mapping[str, float | int] | None = None) -> ScatteringSystem:
        """Construct the kinematics and common scaled mesh for one selection."""
        sample = self.reference_sample if sample is None else sample
        a = int(sample.get("A", self.target_a))
        z = int(sample.get("Z", self.target_z))
        energy = float(sample.get("E", self.lab_energy))
        cache_key = (a, z, energy)
        if cache_key in self._system_cache:
            return self._system_cache[cache_key]
        kwargs = {}
        if self.target_binding_energy is not None and a == self.target_a and z == self.target_z:
            mu, e_cm, _ = two_body_kinematics(
                a, z, self.projectile_a, self.projectile_z, energy,
                target_binding_energy=self.target_binding_energy,
            )
            kwargs.update(reduced_mass_mev=mu, com_energy_mev=e_cm)
        system = ScatteringSystem(
            target_a=a, projectile_a=self.projectile_a, lab_energy=energy,
            mesh_points=self.mesh_points, s_max=self.s_max,
            matching_s=self.matching_s, **kwargs,
        )
        self._system_cache[cache_key] = system
        return system

    def scaled_potential(
        self,
        sample: Mapping[str, float | int],
        spin_orbit_factor: float,
        s_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate ``V(s/k)/E_cm`` for one sample and spin channel."""
        system = self.system(sample)
        return scaled_potential(
            system, self.omega(sample), self.potential.evaluate,
            spin_orbit_factor, s_values,
        )

    def generate_training_data(
        self, samples: Sequence[Mapping[str, float | int]], fom=None
    ) -> TrainingData:
        """Solve all channels and retain wavefunctions, potentials, and S matrices."""
        fom = NumerovFOM() if fom is None else fom
        rows = [dict(sample) for sample in samples]
        reference = self.reference_sample
        reference_system = self.system(reference)
        s_mesh = reference_system.s_mesh
        wavefunctions, reference_wavefunctions = {}, {}
        potentials, reference_potentials = {}, {}
        s_matrices, reference_s_matrices = {}, {}
        for channel in self.channels:
            ell, spin = channel
            ref_potential = self.scaled_potential(reference, spin, s_mesh)
            ref_wave, ref_s = fom.solve(self, reference, channel)
            case_waves, case_potentials, case_s = [], [], []
            for sample in rows:
                system = self.system(sample)
                omega = self.omega(sample)
                wave, s_value = fom.solve(self, sample, channel)
                case_waves.append(wave)
                case_s.append(s_value)
                case_potentials.append(self.scaled_potential(sample, spin, s_mesh))
            reference_wavefunctions[channel] = ref_wave
            reference_potentials[channel] = ref_potential
            reference_s_matrices[channel] = ref_s
            wavefunctions[channel] = np.asarray(case_waves)
            potentials[channel] = np.asarray(case_potentials)
            s_matrices[channel] = np.asarray(case_s)
        return TrainingData(
            rows, reference, s_mesh, self.channels, wavefunctions,
            reference_wavefunctions, potentials, reference_potentials,
            s_matrices, reference_s_matrices,
            physical_radii=np.asarray([self.system(sample).radius for sample in rows]),
            reference_radius=reference_system.radius,
            metadata={
                "potential_model": self.potential.name,
                "coordinate": "s=k*r",
                "potential_scaling": "V/E_cm",
                "matching_s": self.matching_s,
            },
        )

    def cross_section(
        self, sample: Mapping[str, float | int], angles: np.ndarray, fom=None
    ) -> np.ndarray:
        """Calculate one independent FOM differential cross section in mb/sr."""
        fom = NumerovFOM() if fom is None else fom
        system = self.system(sample)
        omega = self.omega(sample)
        s_by_channel = {}
        for ell, spin in self.channels:
            _, s_value = fom.solve(self, sample, (ell, spin))
            s_by_channel[(ell, spin)] = np.asarray([s_value])
        return assemble_elastic_cross_sections(
            s_by_channel, angles, system.wave_number
        )[0]

    def cross_sections_from_data(
        self, data: TrainingData, angles: np.ndarray
    ) -> np.ndarray:
        """Assemble FOM cross sections already stored in a training database."""
        if tuple(data.channels) != tuple(self.channels):
            raise ValueError("training channels do not match the problem")
        wave_numbers = np.asarray([
            self.system(sample).wave_number for sample in data.samples
        ])
        return assemble_varying_energy_cross_sections(
            data.s_matrices, angles, wave_numbers
        )
