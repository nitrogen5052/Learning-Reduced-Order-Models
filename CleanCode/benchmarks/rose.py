"""ROSE calculations used as external benchmarks in the demonstration notebooks.

This module is deliberately outside :mod:`lrom`. Importing the LROM package
never imports ROSE and never uses ROSE to create training data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


def _import_rose():
    """Import ROSE with a compatibility alias for recent SciPy versions."""
    import scipy.special

    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid
    if not hasattr(scipy.special, "sph_harm") and hasattr(scipy.special, "sph_harm_y"):
        scipy.special.sph_harm = (
            lambda m, n, theta, phi: scipy.special.sph_harm_y(n, m, phi, theta)
        )
    import rose

    return rose


@njit
def _real_woods_saxon(radius, omega):
    """Return ROSE's real Woods--Saxon interaction on its radial mesh."""
    return -omega[0] / (1.0 + np.exp((radius - omega[1]) / omega[2]))


@njit
def _full_woods_saxon(radius, omega):
    """Return ROSE's complex central Woods--Saxon interaction."""
    vv, wv, wd, _vso, rv, rd, _rso, av, ad, _aso = omega
    volume = 1.0 / (1.0 + np.exp((radius - rv) / av))
    exponential = np.exp((radius - rd) / ad)
    derivative = -(exponential / ad) / (1.0 + exponential) ** 2
    return -vv * volume - 1j * wv * volume + 4j * ad * wd * derivative


@njit
def _full_woods_saxon_spin_orbit(radius, omega, ldots):
    """Return ROSE's spin-orbit interaction for the supplied l-dot-s value."""
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


@dataclass(frozen=True)
class RoseExactCrossSectionEvaluator:
    """A reusable ROSE high-fidelity cross-section calculation.

    Construction of the interaction and channel solvers is an offline cost.
    ``evaluate`` is the parameter-to-cross-section operation timed in CAT
    comparisons.
    """

    scattering: object

    def evaluate(self, omega: np.ndarray) -> np.ndarray:
        """Return the ROSE FOM cross section for one optical-parameter row."""
        return np.asarray(self.scattering.exact_dsdo(np.asarray(omega)))


@dataclass(frozen=True)
class RoseReducedCrossSectionSet:
    """ROSE scattering emulators indexed by ``(basis size, EIM size)``."""

    emulators: dict[tuple[int, int], object]

    def evaluate(self, configuration: tuple[int, int], omega: np.ndarray) -> np.ndarray:
        """Return one ROSE RBM cross section for the requested configuration."""
        return np.asarray(
            self.emulators[tuple(configuration)].emulate_dsdo(np.asarray(omega))
        )


def reduced_cross_section_emulators(
    radius: np.ndarray,
    angles_degrees: np.ndarray,
    training_rows: np.ndarray,
    configurations: tuple[tuple[int, int], ...],
    *,
    l_max: int = 10,
    target=(40, 20),
    projectile=(1, 0),
    lab_energy: float = 14.1,
) -> RoseReducedCrossSectionSet:
    """Train a consistent family of ROSE cross-section emulators.

    Parameters
    ----------
    radius, angles_degrees
        Physical radial mesh in fm and center-of-mass angles in degrees.
    training_rows
        Optical-potential parameter rows used to build every wavefunction
        basis and empirical-interpolation representation.
    configurations
        ``(N_phi, N_EIM)`` pairs for the wavefunction basis and potential EIM.

    Returns
    -------
    RoseReducedCrossSectionSet
        Reusable ROSE emulators with one entry for every configuration.
    """
    rose = _import_rose()
    configurations = tuple(tuple(map(int, item)) for item in configurations)
    training_rows = np.asarray(training_rows, dtype=float)
    mu, energy, wave_number, _eta = rose.kinematics(
        target=target, projectile=projectile, E_lab=lab_energy
    )
    rho = np.asarray(radius, dtype=float) * wave_number
    base_interactions = rose.InteractionEIMSpace(
        l_max=l_max,
        coordinate_space_potential=_full_woods_saxon,
        spin_orbit_term=_full_woods_saxon_spin_orbit,
        n_theta=training_rows.shape[1],
        mu=mu,
        energy=energy,
        is_complex=True,
        training_info=training_rows,
        explicit_training=True,
        n_basis=max(eim_size for _basis_size, eim_size in configurations),
        rho_mesh=rho,
    )
    base_solver = rose.SchroedingerEquation.make_base_solver(
        s_0=min(6.0 * np.pi, rho[-1] - np.pi),
        rk_tols=[1e-9, 1e-9],
        domain=np.array([rho[0], rho[-1]]),
    )
    _interactions, emulators = rose.training.build_sae_config_set(
        configurations,
        base_interactions,
        training_rows,
        angles=np.deg2rad(np.asarray(angles_degrees, dtype=float)),
        base_solver=base_solver,
        s_mesh=rho,
        scale=True,
        use_svd=True,
        Smatrix_abs_tol=1e-8,
    )
    return RoseReducedCrossSectionSet(dict(zip(configurations, emulators)))


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


def exact_cross_section_evaluator(
    radius: np.ndarray,
    angles_degrees: np.ndarray,
    *,
    l_max=10,
    parameter_count=10,
    target=(40, 20),
    projectile=(1, 0),
    lab_energy=14.1,
) -> RoseExactCrossSectionEvaluator:
    """Build a reusable, independent ROSE high-fidelity evaluator."""
    rose = _import_rose()
    mu, energy, wave_number, _eta = rose.kinematics(
        target=target, projectile=projectile, E_lab=lab_energy
    )
    rho = np.asarray(radius) * wave_number
    interactions = rose.InteractionSpace(
        l_max=l_max,
        coordinate_space_potential=_full_woods_saxon,
        spin_orbit_term=_full_woods_saxon_spin_orbit,
        n_theta=parameter_count,
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
    return RoseExactCrossSectionEvaluator(scattering=scattering)


def exact_kd_cross_section_evaluator(
    radius: np.ndarray,
    angles_degrees: np.ndarray,
    *,
    l_max=10,
    target=(40, 20),
    projectile=(1, 0),
    lab_energy=14.1,
    matching_s=6.0 * np.pi,
    reduced_mass_mev=None,
    com_energy_mev=None,
    wave_number=None,
) -> RoseExactCrossSectionEvaluator:
    """Build an independent ROSE FOM for the 15-parameter KD potential.

    The parameter order matches :func:`lrom.physics.kd_neutron_potential`.
    This helper is kept in the external benchmark module so the LROM package
    remains independent of ROSE.
    """
    rose = _import_rose()
    from rose.koning_delaroche import KD_simple, KD_simple_so

    rose_mu, rose_energy, rose_wave_number, _eta = rose.kinematics(
        target=target, projectile=projectile, E_lab=lab_energy
    )
    mu = rose_mu if reduced_mass_mev is None else float(reduced_mass_mev)
    energy = rose_energy if com_energy_mev is None else float(com_energy_mev)
    wave_number = (
        rose_wave_number if wave_number is None else float(wave_number)
    )
    rho = np.asarray(radius) * wave_number
    interactions = rose.InteractionSpace(
        l_max=l_max,
        coordinate_space_potential=KD_simple,
        spin_orbit_term=KD_simple_so,
        n_theta=15,
        mu=mu,
        energy=energy,
        is_complex=True,
    )
    base_solver = rose.SchroedingerEquation.make_base_solver(
        s_0=min(float(matching_s), rho[-1] - np.pi),
        rk_tols=[1e-10, 1e-10],
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
    return RoseExactCrossSectionEvaluator(scattering=scattering)


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
    """Return one high-fidelity differential cross section from ROSE."""
    evaluator = exact_cross_section_evaluator(
        radius,
        angles_degrees,
        l_max=l_max,
        parameter_count=len(omega),
        target=target,
        projectile=projectile,
        lab_energy=lab_energy,
    )
    return evaluator.evaluate(omega)
