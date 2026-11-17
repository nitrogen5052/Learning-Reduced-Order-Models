"""Independent full-order solver for neutron elastic scattering.

The solver integrates the radial Schrödinger equation with SciPy and matches
the numerical logarithmic derivative to free Riccati–Hankel functions.  It is
the source of all training data and FOM reference results in this rewrite.
ROSE is intentionally not imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import eval_legendre, lpmv, spherical_jn, spherical_yn

from .physics import AMU, HBAR_C

Potential = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class ScatteringSystem:
    """Kinematics and a common physical-radius mesh for one target and energy."""

    target_a: int = 40
    projectile_a: int = 1
    lab_energy: float = 14.1
    mesh_points: int = 600
    r_min: float = 1.0e-5
    r_max: float = 20.0
    reduced_mass_mev: float | None = None
    com_energy_mev: float | None = None

    @property
    def reduced_mass(self) -> float:
        if self.reduced_mass_mev is not None:
            return float(self.reduced_mass_mev)
        return AMU * self.target_a * self.projectile_a / (self.target_a + self.projectile_a)

    @property
    def com_energy(self) -> float:
        if self.com_energy_mev is not None:
            return float(self.com_energy_mev)
        return self.lab_energy * self.target_a / (self.target_a + self.projectile_a)

    @property
    def wave_number(self) -> float:
        return np.sqrt(2.0 * self.reduced_mass * self.com_energy) / HBAR_C

    @property
    def radius(self) -> np.ndarray:
        return np.linspace(self.r_min, self.r_max, self.mesh_points)


def _s_matrix_from_boundary(ell: int, k: float, radius: float, log_derivative: complex) -> complex:
    """Match a numerical solution to incoming/outgoing free radial waves."""
    z = k * radius
    j = spherical_jn(ell, z)
    y = spherical_yn(ell, z)
    jp = spherical_jn(ell, z, derivative=True)
    yp = spherical_yn(ell, z, derivative=True)

    h_minus = z * (j - 1j * y)
    h_plus = z * (j + 1j * y)
    dh_minus = k * ((j - 1j * y) + z * (jp - 1j * yp))
    dh_plus = k * ((j + 1j * y) + z * (jp + 1j * yp))
    # With u proportional to H^- + S H^+, the free regular solution has S=1.
    return -(dh_minus - log_derivative * h_minus) / (dh_plus - log_derivative * h_plus)


def solve_channel(
    system: ScatteringSystem,
    omega: np.ndarray,
    ell: int,
    potential: Potential,
    spin_orbit_factor: float = 0.0,
    normalization: str = "shooting",
) -> tuple[np.ndarray, complex]:
    """Solve one partial wave and return its wavefunction and S matrix.

    The default ``"shooting"`` convention matches ROSE: near the origin,
    ``u(r) = (k r)^(ell+1)`` and the derivative with respect to ``k r`` is
    positive. ``"scattering"`` optionally rescales the same solution to unit
    incoming-wave amplitude. The S matrix is independent of this normalization.
    """
    r = system.radius
    factor = 2.0 * system.reduced_mass / HBAR_C**2

    v = potential(r, omega, spin_orbit_factor)
    q = ell * (ell + 1) / r**2 + factor * (v - system.com_energy)
    step = r[1] - r[0]
    step2 = step**2

    # Numerov is especially convenient here: the potential is evaluated once
    # on the visible mesh and no hidden solver grid is introduced.
    u = np.zeros(r.size, dtype=complex)
    rho = system.wave_number * r
    u[0] = rho[0] ** (ell + 1)
    u[1] = rho[1] ** (ell + 1)
    for index in range(1, r.size - 1):
        numerator = (
            2.0 * u[index] * (1.0 + 5.0 * step2 * q[index] / 12.0)
            - u[index - 1] * (1.0 - step2 * q[index - 1] / 12.0)
        )
        u[index + 1] = numerator / (1.0 - step2 * q[index + 1] / 12.0)

    derivative = (
        25.0 * u[-1] - 48.0 * u[-2] + 36.0 * u[-3] - 16.0 * u[-4] + 3.0 * u[-5]
    ) / (12.0 * step)
    s_matrix = _s_matrix_from_boundary(ell, system.wave_number, r[-1], derivative / u[-1])

    if normalization == "shooting":
        return u, s_matrix
    if normalization == "scattering":
        z = system.wave_number * r[-1]
        h_minus = z * (spherical_jn(ell, z) - 1j * spherical_yn(ell, z))
        h_plus = z * (spherical_jn(ell, z) + 1j * spherical_yn(ell, z))
        target_boundary = 0.5 * (h_minus + s_matrix * h_plus)
        return u * (target_boundary / u[-1]), s_matrix
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


def s_matrix_from_wavefunction(
    radius: np.ndarray,
    wavefunction: np.ndarray,
    ell: int,
    wave_number: float,
    matching_points: int = 12,
) -> complex:
    """Extract an S matrix by fitting the asymptotic wavefunction.

    A short least-squares fit is more stable for reconstructed LROM
    wavefunctions than differentiating only the final two mesh points.
    """
    r = np.asarray(radius)[-matching_points:]
    u = np.asarray(wavefunction)[-matching_points:]
    z = wave_number * r
    h_minus = z * (spherical_jn(ell, z) - 1j * spherical_yn(ell, z))
    h_plus = z * (spherical_jn(ell, z) + 1j * spherical_yn(ell, z))
    amplitudes, *_ = np.linalg.lstsq(np.column_stack([h_minus, h_plus]), u, rcond=None)
    return amplitudes[1] / amplitudes[0]


def differential_cross_section(
    angles_degrees: np.ndarray,
    wave_number: float,
    s_plus: np.ndarray,
    s_minus: np.ndarray,
) -> np.ndarray:
    """Spin-1/2 on spin-0 elastic differential cross section in mb/sr."""
    theta = np.deg2rad(np.asarray(angles_degrees))
    x = np.cos(theta)
    amplitude_a = np.zeros(theta.size, dtype=complex)
    amplitude_b = np.zeros(theta.size, dtype=complex)

    for ell in range(len(s_plus)):
        sm = s_minus[ell] if ell > 0 else s_plus[ell]
        amplitude_a += (
            (ell + 1) * (s_plus[ell] - 1.0) + ell * (sm - 1.0)
        ) * eval_legendre(ell, x)
        if ell > 0:
            amplitude_b += (s_plus[ell] - sm) * lpmv(1, ell, x)

    amplitude_a /= 2j * wave_number
    amplitude_b /= 2j * wave_number
    return 10.0 * (np.abs(amplitude_a) ** 2 + np.abs(amplitude_b) ** 2)
