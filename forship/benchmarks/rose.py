"""ROSE calculations used as external benchmarks in the demonstration notebooks.

This module is deliberately outside :mod:`lrom`. Importing the LROM package
never imports ROSE and never uses ROSE to create training data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


def _import_rose():
    import scipy.special

    if not hasattr(scipy.special, "sph_harm") and hasattr(scipy.special, "sph_harm_y"):
        scipy.special.sph_harm = (
            lambda m, n, theta, phi: scipy.special.sph_harm_y(n, m, phi, theta)
        )
    import rose

    return rose


@njit
def _real_woods_saxon(radius, omega):
    return -omega[0] / (1.0 + np.exp((radius - omega[1]) / omega[2]))


@njit
def _full_woods_saxon(radius, omega):
    vv, wv, wd, _vso, rv, rd, _rso, av, ad, _aso = omega
    volume = 1.0 / (1.0 + np.exp((radius - rv) / av))
    exponential = np.exp((radius - rd) / ad)
    derivative = -(exponential / ad) / (1.0 + exponential) ** 2
    return -vv * volume - 1j * wv * volume + 4j * ad * wd * derivative


@njit
def _full_woods_saxon_spin_orbit(radius, omega, ldots):
    vso, rso, aso = omega[3], omega[6], omega[9]
    exponential = np.exp((radius - rso) / aso)
    derivative = -(exponential / aso) / (1.0 + exponential) ** 2
    return 2.0 * vso * ldots * derivative / radius


@dataclass(frozen=True)
class RoseWavefunctionBenchmark:
    """Self-contained ROSE FOM and reduced-basis results."""

    radius: np.ndarray
    training_exact: np.ndarray
    testing_exact: np.ndarray
    training_prediction: np.ndarray
    testing_prediction: np.ndarray
    training_coefficients: np.ndarray
    testing_coefficients: np.ndarray
    basis_vectors: np.ndarray


def wavefunction_benchmark(
    radius: np.ndarray,
    training_rows: np.ndarray,
    testing_rows: np.ndarray,
    *,
    target=(40, 20),
    projectile=(1, 0),
    lab_energy=14.1,
    basis_size=4,
    eim_size=8,
) -> RoseWavefunctionBenchmark:
    """Run an independent ROSE FOM/RBM calculation on supplied parameter rows."""
    rose = _import_rose()
    mu, energy, _relativistic_wave_number, eta = rose.kinematics(
        target=target, projectile=projectile, E_lab=lab_energy
    )
    radius = np.asarray(radius)
    # Fixed-energy ROSE interactions use k=sqrt(2 mu E)/hbar internally.
    wave_number = np.sqrt(2.0 * mu * energy) / 197.3269804
    rho = radius * wave_number
    all_rows = np.vstack([training_rows, testing_rows])
    bounds = np.column_stack([all_rows.min(axis=0), all_rows.max(axis=0)])
    interaction_space = rose.InteractionEIMSpace(
        l_max=0,
        coordinate_space_potential=_real_woods_saxon,
        n_theta=3,
        mu=mu,
        energy=energy,
        is_complex=False,
        training_info=bounds,
        n_basis=eim_size,
        rho_mesh=rho,
    )
    interaction = interaction_space.interactions[0][0]
    exact_interaction = rose.InteractionSpace(
        l_max=0,
        coordinate_space_potential=_real_woods_saxon,
        n_theta=3,
        mu=mu,
        energy=energy,
        is_complex=False,
    ).interactions[0][0]
    base_solver = rose.SchroedingerEquation.make_base_solver(
        s_0=min(6.0 * np.pi, rho[-1] - np.pi),
        rk_tols=[1e-9, 1e-9],
        domain=np.array([rho[0], rho[-1]]),
    )
    exact_solver = base_solver.clone_for_new_interaction(exact_interaction)
    training_exact = np.asarray([exact_solver.phi(row, rho) for row in training_rows])
    testing_exact = np.asarray([exact_solver.phi(row, rho) for row in testing_rows])
    free_reference = np.asarray(
        [rose.free_solutions.phi_free(float(value), 0, eta) for value in rho]
    )
    basis = rose.basis.CustomBasis(
        solutions=training_exact.T.copy(),
        phi_0=free_reference,
        rho_mesh=rho,
        n_basis=basis_size,
        solver=exact_solver,
        subtract_phi0=True,
        use_svd=True,
        center=False,
        scale=False,
    )
    emulator = rose.reduced_basis_emulator.ReducedBasisEmulator(
        interaction, basis, s_0=base_solver.s_0, initialize_emulator=True
    )
    training_coefficients = np.asarray(
        [emulator.coefficients(row) for row in training_rows]
    )
    testing_coefficients = np.asarray(
        [emulator.coefficients(row) for row in testing_rows]
    )
    training_prediction = np.asarray(
        [emulator.emulate_wave_function(row) for row in training_rows]
    )
    testing_prediction = np.asarray(
        [emulator.emulate_wave_function(row) for row in testing_rows]
    )
    return RoseWavefunctionBenchmark(
        radius=radius,
        training_exact=training_exact,
        testing_exact=testing_exact,
        training_prediction=training_prediction,
        testing_prediction=testing_prediction,
        training_coefficients=training_coefficients,
        testing_coefficients=testing_coefficients,
        basis_vectors=np.asarray(basis.vectors),
    )


def exact_cross_section(
    radius: np.ndarray,
    omega: np.ndarray,
    angles_degrees: np.ndarray,
    *,
    l_max=10,
    target=(40, 20),
    projectile=(1, 0),
    lab_energy=14.1,
) -> np.ndarray:
    """Return a final high-fidelity differential cross section from ROSE."""
    rose = _import_rose()
    mu, energy, wave_number, _eta = rose.kinematics(
        target=target, projectile=projectile, E_lab=lab_energy
    )
    rho = np.asarray(radius) * wave_number
    interactions = rose.InteractionSpace(
        l_max=l_max,
        coordinate_space_potential=_full_woods_saxon,
        spin_orbit_term=_full_woods_saxon_spin_orbit,
        n_theta=len(omega),
        mu=mu,
        energy=energy,
        is_complex=True,
    )
    base_solver = rose.SchroedingerEquation.make_base_solver(
        s_0=min(6.0 * np.pi, rho[-1] - np.pi),
        rk_tols=[1e-9, 1e-9],
        domain=np.array([rho[0], rho[-1]]),
    )
    scattering = rose.ScatteringAmplitudeEmulator.HIFI_solver(
        interactions,
        base_solver=base_solver,
        l_max=l_max,
        angles=np.deg2rad(angles_degrees),
        verbose=False,
        s_mesh=rho,
    )
    return np.asarray(scattering.exact_dsdo(np.asarray(omega)))
