"""Independent full-order solver on the scaled coordinate ``s = k r``.

The dimensionless radial equation is

    phi''(s) = [l(l+1)/s**2 + V(s/k)/E_cm - 1] phi(s).

Using one common ``s`` mesh removes the energy-dependent free-wave frequency
from a snapshot database. The solver is independent of ROSE, but follows the
same regular-origin and positive-derivative convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from .physics import AMU, HBAR_C

Potential = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


class NumerovFOM:
    """Default independent scaled-coordinate full-order solver."""

    def solve(self, problem, sample, channel):
        """Return ``phi(s)`` and the channel S matrix for one named sample."""
        ell, spin = channel
        return solve_channel(
            problem.system(sample), problem.omega(sample), ell,
            problem.potential.evaluate, spin,
        )


@dataclass(frozen=True)
class ScatteringSystem:
    """Kinematics and the common dimensionless coordinate mesh for one case."""

    target_a: int = 40
    projectile_a: int = 1
    lab_energy: float = 14.1
    mesh_points: int = 800
    s_min: float = 1.0e-8
    s_max: float = 8.0 * np.pi
    matching_s: float = 6.0 * np.pi
    reduced_mass_mev: float | None = None
    com_energy_mev: float | None = None

    @property
    def reduced_mass(self) -> float:
        """Reduced mass in MeV."""
        if self.reduced_mass_mev is not None:
            return float(self.reduced_mass_mev)
        return AMU * self.target_a * self.projectile_a / (self.target_a + self.projectile_a)

    @property
    def com_energy(self) -> float:
        """Center-of-mass scattering energy in MeV."""
        if self.com_energy_mev is not None:
            return float(self.com_energy_mev)
        return self.lab_energy * self.target_a / (self.target_a + self.projectile_a)

    @property
    def wave_number(self) -> float:
        """Center-of-mass wave number in fm^-1."""
        return np.sqrt(2.0 * self.reduced_mass * self.com_energy) / HBAR_C

    @property
    def s_mesh(self) -> np.ndarray:
        """Uniform shared mesh in the dimensionless coordinate ``s=k r``."""
        return np.linspace(self.s_min, self.s_max, self.mesh_points)

    @property
    def radius(self) -> np.ndarray:
        """Physical radius corresponding to this case's ``s`` mesh, in fm."""
        return self.s_mesh / self.wave_number


def neutron_gamow_factor(ell: int) -> float:
    """Return ``C_l=1/(2l+1)!!`` for the neutral regular-origin solution."""
    value = 1.0
    for order in range(1, int(ell) + 1):
        value /= 2 * order + 1
    return value


def regular_start_s(ell: int, threshold: float = 1.0e-10) -> float:
    """Smallest safe coordinate where ``C_l s**(l+1)`` reaches threshold."""
    return float((threshold / neutron_gamow_factor(ell)) ** (1.0 / (ell + 1)))


def scaled_potential(
    system: ScatteringSystem,
    omega: np.ndarray,
    potential: Potential,
    spin_orbit_factor: float = 0.0,
    s_values: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the dimensionless operator potential ``V(s/k)/E_cm``."""
    s = system.s_mesh if s_values is None else np.asarray(s_values, dtype=float)
    return potential(s / system.wave_number, omega, spin_orbit_factor) / system.com_energy


def _riccati_hankel(ell: int, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return incoming and outgoing free Riccati-Hankel functions."""
    s = np.asarray(s, dtype=float)
    j = spherical_jn(ell, s)
    y = spherical_yn(ell, s)
    return s * (j - 1j * y), s * (j + 1j * y)


def _riccati_hankel_derivatives(
    ell: int, s: float
) -> tuple[complex, complex]:
    """Return derivatives of incoming and outgoing functions with respect to s."""
    j = spherical_jn(ell, s)
    y = spherical_yn(ell, s)
    jp = spherical_jn(ell, s, derivative=True)
    yp = spherical_yn(ell, s, derivative=True)
    return (
        (j - 1j * y) + s * (jp - 1j * yp),
        (j + 1j * y) + s * (jp + 1j * yp),
    )


def s_matrix_from_scaled_boundary(
    matching_s: float,
    value: complex,
    derivative: complex,
    ell: int,
) -> complex:
    """Extract S from ``phi`` and ``dphi/ds`` at one matching coordinate."""
    h_minus, h_plus = _riccati_hankel(ell, matching_s)
    dh_minus, dh_plus = _riccati_hankel_derivatives(ell, matching_s)
    log_derivative = derivative / value
    return -(dh_minus - log_derivative * h_minus) / (
        dh_plus - log_derivative * h_plus
    )


def s_matrix_from_scaled_values(
    s_values: np.ndarray, values: np.ndarray, ell: int
) -> complex:
    """Extract an S matrix by fitting values to free waves on a matching tail."""
    s_values = np.asarray(s_values, dtype=float)
    values = np.asarray(values, dtype=complex)
    if s_values.ndim != 1 or values.shape != s_values.shape or len(s_values) < 2:
        raise ValueError("s_values and values must be equal one-dimensional tails")
    h_minus, h_plus = _riccati_hankel(ell, s_values)
    amplitudes, *_ = np.linalg.lstsq(
        np.column_stack([h_minus, h_plus]), values, rcond=None
    )
    return amplitudes[1] / amplitudes[0]


def matching_indices(
    s_mesh: np.ndarray, matching_s: float, matching_points: int = 12
) -> np.ndarray:
    """Select a short tail ending at the configured matching coordinate."""
    end = int(np.searchsorted(s_mesh, matching_s, side="right"))
    end = min(max(end, matching_points), len(s_mesh))
    return np.arange(end - matching_points, end)


def matching_index(s_mesh: np.ndarray, matching_s: float) -> int:
    """Return the mesh index nearest the requested matching coordinate."""
    return int(np.argmin(np.abs(np.asarray(s_mesh) - matching_s)))


def scaled_derivative_at(
    s_mesh: np.ndarray, values: np.ndarray, index: int
) -> np.ndarray:
    """Differentiate along the first axis with a centered five-point stencil."""
    s_mesh = np.asarray(s_mesh, dtype=float)
    values = np.asarray(values)
    if index < 2 or index > len(s_mesh) - 3:
        return np.gradient(values, s_mesh, axis=0, edge_order=2)[index]
    step = s_mesh[1] - s_mesh[0]
    return (
        values[index - 2] - 8.0 * values[index - 1]
        + 8.0 * values[index + 1] - values[index + 2]
    ) / (12.0 * step)


def s_matrix_from_scaled_wavefunction(
    s_mesh: np.ndarray, wavefunction: np.ndarray, ell: int, matching_s: float
) -> complex:
    """Extract S from one full snapshot using its value and derivative at s0."""
    s_mesh = np.asarray(s_mesh, dtype=float)
    wavefunction = np.asarray(wavefunction, dtype=complex)
    index = matching_index(s_mesh, matching_s)
    derivative = scaled_derivative_at(s_mesh, wavefunction, index)
    return s_matrix_from_scaled_boundary(
        s_mesh[index], wavefunction[index], derivative, ell
    )


def solve_channel(
    system: ScatteringSystem,
    omega: np.ndarray,
    ell: int,
    potential: Potential,
    spin_orbit_factor: float = 0.0,
    normalization: str = "shooting",
    phi_threshold: float = 1.0e-10,
) -> tuple[np.ndarray, complex]:
    """Solve one partial wave on ``s`` and return ``phi(s)`` and its S matrix.

    Values below the l-dependent start are zero. At the first two active mesh
    points, the regular solution is initialized as ``C_l s**(l+1)``, whose
    derivative with respect to ``s`` is positive. This avoids propagating
    unresolvably tiny higher-l values from a fixed origin.
    """
    s = system.s_mesh
    u_tilde = scaled_potential(
        system, omega, potential, spin_orbit_factor, s_values=s
    )
    q = ell * (ell + 1) / s**2 + u_tilde - 1.0
    step = s[1] - s[0]
    step2 = step**2

    start_s = max(system.s_min, regular_start_s(ell, phi_threshold))
    start = int(np.searchsorted(s, start_s, side="left"))
    if start + 2 >= len(s):
        raise ValueError("scaled mesh ends before the regular solution can start")

    phi = np.zeros(s.size, dtype=complex)
    c_ell = neutron_gamow_factor(ell)
    phi[start] = c_ell * s[start] ** (ell + 1)
    phi[start + 1] = c_ell * s[start + 1] ** (ell + 1)
    for index in range(start + 1, s.size - 1):
        numerator = (
            2.0 * phi[index] * (1.0 + 5.0 * step2 * q[index] / 12.0)
            - phi[index - 1] * (1.0 - step2 * q[index - 1] / 12.0)
        )
        phi[index + 1] = numerator / (1.0 - step2 * q[index + 1] / 12.0)

    match_index = matching_index(s, system.matching_s)
    s_matrix = s_matrix_from_scaled_wavefunction(
        s, phi, ell, system.matching_s
    )
    if normalization == "shooting":
        return phi, s_matrix
    if normalization == "scattering":
        h_minus, h_plus = _riccati_hankel(ell, s[match_index])
        target = 0.5 * (h_minus + s_matrix * h_plus)
        return phi * (target / phi[match_index]), s_matrix
    raise ValueError("normalization must be 'shooting' or 'scattering'")


def solve_parameter_rows(
    system: ScatteringSystem,
    parameter_rows: np.ndarray,
    ell: int,
    potential: Potential,
    spin_orbit_factor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one channel for an ordered collection of optical parameters."""
    solutions = [
        solve_channel(system, row, ell, potential, spin_orbit_factor)
        for row in np.asarray(parameter_rows)
    ]
    return (
        np.asarray([solution[0] for solution in solutions]),
        np.asarray([solution[1] for solution in solutions]),
    )


def s_matrices_from_scaled_wavefunctions(
    s_mesh: np.ndarray,
    wavefunctions: np.ndarray,
    ell: int,
    matching_s: float = 6.0 * np.pi,
    matching_points: int = 12,
) -> np.ndarray:
    """Extract one S matrix from every ``phi(s)`` row in an array."""
    del matching_points
    return np.asarray([
        s_matrix_from_scaled_wavefunction(s_mesh, wave, ell, matching_s)
        for wave in np.atleast_2d(wavefunctions)
    ])


# Backward-compatible name used by the first draft notebooks.
s_matrix_from_asymptotic_values = s_matrix_from_scaled_values
