"""Utilities for Phase 3 learned implicit reduced equations.

The notebook keeps the scientific workflow explicit, while this module holds
small reusable mechanics: building one-parameter ROSE datasets, extracting
ROSE reduced matrices, packing complex model parameters, fitting implicit
linear systems, and evaluating errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from numba import njit


ROOT = Path(__file__).resolve().parents[1]
# Portable demo version: use the public PyPI package nuclear-rose, which imports as rose.
# Install with: pip install nuclear-rose

import rose  # noqa: E402
from rose.basis import CustomBasis  # noqa: E402
from rose.reduced_basis_emulator import ReducedBasisEmulator  # noqa: E402


TARGET = (40, 20)
PROJECTILE = (1, 0)
E_LAB = 14.1


@dataclass
class OneParameterDataset:
    name: str
    param_index: int
    central_alpha: np.ndarray
    train_values: np.ndarray
    test_values: np.ndarray
    train_alphas: np.ndarray
    test_alphas: np.ndarray
    emulator: object
    rbe: object
    rho_mesh: np.ndarray
    r_mesh: np.ndarray
    coeff_train: np.ndarray
    coeff_test: np.ndarray
    coeff_rom_train: np.ndarray
    coeff_rom_test: np.ndarray
    phi_hf_train: np.ndarray
    phi_hf_test: np.ndarray
    phi_basis_train: np.ndarray
    phi_basis_test: np.ndarray
    phi_rbm_train: np.ndarray
    phi_rbm_test: np.ndarray


@dataclass
class MultiParameterDataset:
    name: str
    feature_names: list[str]
    feature_scales: np.ndarray
    central_sample: np.ndarray
    central_alpha: np.ndarray
    train_samples: np.ndarray
    test_samples: np.ndarray
    train_alphas: np.ndarray
    test_alphas: np.ndarray
    rbe: object
    rho_mesh: np.ndarray
    coeff_train: np.ndarray
    coeff_test: np.ndarray
    coeff_rom_train: np.ndarray
    coeff_rom_test: np.ndarray
    phi_hf_train: np.ndarray
    phi_hf_test: np.ndarray
    phi_basis_train: np.ndarray
    phi_basis_test: np.ndarray
    phi_rbm_train: np.ndarray
    phi_rbm_test: np.ndarray


@dataclass
class MultiPartialWaveDataset:
    name: str
    feature_names: list[str]
    feature_scales: np.ndarray
    central_sample: np.ndarray
    central_alpha: np.ndarray
    train_samples: np.ndarray
    test_samples: np.ndarray
    train_alphas: np.ndarray
    test_alphas: np.ndarray
    sae: object
    rho_mesh: np.ndarray
    angles: np.ndarray
    coeff_train: list[list[np.ndarray]]
    coeff_test: list[list[np.ndarray]]


@dataclass
class GenericCentralSAEDataset:
    name: str
    feature_names: list[str]
    feature_scales: np.ndarray
    central_sample: np.ndarray
    train_samples: np.ndarray
    test_samples: np.ndarray
    sae: object
    rho_mesh: np.ndarray
    angles: np.ndarray
    coeff_train: list[list[np.ndarray]]
    coeff_test: list[list[np.ndarray]]


@dataclass
class FitResult:
    name: str
    params: np.ndarray
    matrices: list[np.ndarray]
    vectors: list[np.ndarray]
    losses: np.ndarray
    train_seconds: float
    nfev: int
    success: bool
    message: str


@dataclass
class ResidualFitResult:
    name: str
    params: np.ndarray
    matrices: list[np.ndarray]
    vectors: list[np.ndarray]
    train_seconds: float
    residual_mse: float
    rank: int
    singular_values: np.ndarray
    success: bool
    message: str


@dataclass
class OperatorDesign:
    design: np.ndarray
    target: np.ndarray
    keep: np.ndarray
    fixed_complex_index: int
    fixed_value: complex
    shapes: list[tuple[int, ...]]
    n_features: int
    n_basis: int
    n_unknowns: int


def central_kd_parameters():
    mu, e_com, k, eta = rose.kinematics(
        target=TARGET,
        projectile=PROJECTILE,
        E_lab=E_LAB,
    )
    kd = rose.koning_delaroche.KDGlobal(rose.Projectile.neutron)
    r_c, alpha = kd.get_params(TARGET[0], TARGET[1], mu, E_LAB, k)
    return mu, e_com, k, eta, r_c, alpha


@njit
def real_volume_woods_saxon(r, alpha):
    """Real central Woods-Saxon potential with alpha = [Vv, Rv, av]."""
    vv, rv, av = alpha
    return -vv * rose.koning_delaroche.woods_saxon_safe(r, rv, av)


def central_real_ws_parameters():
    """Central [Vv, Rv, av] values extracted from the KD calcium setup."""
    mu, e_com, k, eta, r_c, alpha = central_kd_parameters()
    return mu, e_com, k, eta, r_c, alpha[:3].copy()


def energized_alpha_from_sample(
    sample: np.ndarray,
    optical_alpha0: np.ndarray | None = None,
):
    """Build ROSE's energy-aware alpha vector from [Vv, Rv, av, E_lab]."""
    if optical_alpha0 is None:
        *_unused, optical_alpha0 = central_kd_parameters()
    vv, rv, av, e_lab = sample
    mu, e_com, k, _eta = rose.kinematics(
        target=TARGET,
        projectile=PROJECTILE,
        E_lab=float(e_lab),
    )
    optical_alpha = np.array(optical_alpha0, dtype=float).copy()
    optical_alpha[0] = vv
    optical_alpha[1] = rv
    optical_alpha[2] = av
    return np.concatenate([[e_com, mu, k], optical_alpha])


def energized_alphas_from_samples(
    samples: np.ndarray,
    optical_alpha0: np.ndarray | None = None,
):
    return np.array(
        [energized_alpha_from_sample(sample, optical_alpha0) for sample in samples]
    )


def make_alphas(alpha0: np.ndarray, param_index: int, values: np.ndarray):
    alphas = np.tile(alpha0, (len(values), 1))
    alphas[:, param_index] = values
    return alphas


def multiparameter_feature_values(
    samples: np.ndarray,
    central_sample: np.ndarray,
    feature_scales: np.ndarray,
):
    """Return [1, normalized deltas...] for the central LROM equation."""
    deltas = (samples - central_sample) / feature_scales
    return np.column_stack([np.ones(samples.shape[0]), deltas])


def trapezoid_sqrt_weights(mesh: np.ndarray):
    weights = np.empty_like(mesh, dtype=float)
    weights[1:-1] = 0.5 * (mesh[2:] - mesh[:-2])
    weights[0] = 0.5 * (mesh[1] - mesh[0])
    weights[-1] = 0.5 * (mesh[-1] - mesh[-2])
    return np.sqrt(weights)


def least_squares_basis_coefficients(basis, wavefunctions: np.ndarray, mesh: np.ndarray):
    """Project FOM wavefunctions onto phi0 + span(Phi) in weighted L2."""
    weights = trapezoid_sqrt_weights(mesh)
    weighted_vectors = weights[:, np.newaxis] * basis.vectors
    coeffs = []
    for phi in wavefunctions:
        rhs = weights * (phi - basis.phi_0)
        coeff, *_ = np.linalg.lstsq(weighted_vectors, rhs, rcond=None)
        coeffs.append(coeff)
    return np.asarray(coeffs)


def scale_wavefunctions_like_rose_basis(wavefunctions: np.ndarray, mesh: np.ndarray):
    """Apply ROSE's scale=True snapshot convention to FOM wavefunctions."""
    scaled = []
    for phi in wavefunctions:
        norm = np.trapz(np.absolute(phi) ** 2, mesh)
        scaled.append(phi / norm)
    return np.asarray(scaled)


def train_one_parameter_rbe(
    name: str,
    param_index: int,
    train_values: np.ndarray,
    test_values: np.ndarray,
    n_phi: int = 4,
    n_U: int = 8,
    l_max: int = 1,
    n_mesh: int = 1000,
    real_ws_only: bool = False,
):
    """Train ROSE RBM for one-parameter variation and collect l=0 data.

    We keep ROSE's snapshot scaling disabled here. The reduced coordinates
    are therefore tied to the raw FOM wavefunctions rather than to ROSE's
    optional normalization convention.
    """
    if real_ws_only:
        mu, e_com, k, _eta, _r_c, alpha0 = central_real_ws_parameters()
        potential = real_volume_woods_saxon
        spin_orbit_term = None
        is_complex = False
    else:
        mu, e_com, k, _eta, _r_c, alpha0 = central_kd_parameters()
        potential = rose.koning_delaroche.KD_simple
        spin_orbit_term = rose.koning_delaroche.KD_simple_so
        is_complex = True
    train_alphas = make_alphas(alpha0, param_index, train_values)
    test_alphas = make_alphas(alpha0, param_index, test_values)

    bounds = np.column_stack(
        [
            np.minimum(train_alphas.min(axis=0), alpha0),
            np.maximum(train_alphas.max(axis=0), alpha0),
        ]
    )
    rho_mesh = np.linspace(1e-8, 8 * np.pi, n_mesh)
    angles = np.linspace(1, 179, 120) * np.pi / 180

    eim_kwargs = dict(
        l_max=l_max,
        coordinate_space_potential=potential,
        n_theta=alpha0.size,
        mu=mu,
        energy=e_com,
        is_complex=is_complex,
        training_info=bounds,
        n_basis=n_U,
        rho_mesh=rho_mesh,
    )
    if spin_orbit_term is not None:
        eim_kwargs["spin_orbit_term"] = spin_orbit_term
    interactions = rose.InteractionEIMSpace(**eim_kwargs)
    base_solver = rose.SchroedingerEquation.make_base_solver(
        s_0=6 * np.pi,
        rk_tols=[1e-9, 1e-9],
        domain=np.array([1e-8, 8 * np.pi]),
    )
    emulator = rose.ScatteringAmplitudeEmulator.from_train(
        interactions,
        train_alphas,
        base_solver=base_solver,
        l_max=l_max,
        angles=angles,
        n_basis=n_phi,
        use_svd=True,
        scale=False,
        s_mesh=rho_mesh,
        Smatrix_abs_tol=1e-8,
    )
    rbe = emulator.rbes[0][0]

    coeff_rom_train = np.array([rbe.coefficients(alpha) for alpha in train_alphas])
    coeff_rom_test = np.array([rbe.coefficients(alpha) for alpha in test_alphas])

    phi_hf_train = np.array([rbe.basis.solver.phi(alpha, rho_mesh) for alpha in train_alphas])
    phi_hf_test = np.array([rbe.basis.solver.phi(alpha, rho_mesh) for alpha in test_alphas])
    phi_rbm_train = np.array([rbe.emulate_wave_function(alpha) for alpha in train_alphas])
    phi_rbm_test = np.array([rbe.emulate_wave_function(alpha) for alpha in test_alphas])
    phi_basis_train = phi_hf_train.copy()
    phi_basis_test = phi_hf_test.copy()
    coeff_train = least_squares_basis_coefficients(rbe.basis, phi_basis_train, rho_mesh)
    coeff_test = least_squares_basis_coefficients(rbe.basis, phi_basis_test, rho_mesh)

    return OneParameterDataset(
        name=name,
        param_index=param_index,
        central_alpha=alpha0,
        train_values=train_values,
        test_values=test_values,
        train_alphas=train_alphas,
        test_alphas=test_alphas,
        emulator=emulator,
        rbe=rbe,
        rho_mesh=rho_mesh,
        r_mesh=rho_mesh / k,
        coeff_train=coeff_train,
        coeff_test=coeff_test,
        coeff_rom_train=coeff_rom_train,
        coeff_rom_test=coeff_rom_test,
        phi_hf_train=phi_hf_train,
        phi_hf_test=phi_hf_test,
        phi_basis_train=phi_basis_train,
        phi_basis_test=phi_basis_test,
        phi_rbm_train=phi_rbm_train,
        phi_rbm_test=phi_rbm_test,
    )


def train_one_parameter_central_rbe(
    name: str,
    param_index: int,
    train_values: np.ndarray,
    test_values: np.ndarray,
    n_phi: int = 4,
    n_U: int = 8,
    l_max: int = 1,
    n_mesh: int = 1000,
    central_value: float | None = None,
    real_ws_only: bool = False,
    scale_snapshots: bool = False,
):
    """Train a one-parameter ROSE RBE using the central FOM solution as phi0.

    Set ``real_ws_only=True`` for the teaching examples where the interaction is
    just ``-Vv f_WS(r; Rv, av)``.  The full KD optical model is kept as the
    default for the more realistic notebooks.

    ``scale_snapshots=False`` keeps the reduced coordinates in raw wavefunction
    units and is the default used in the clean demo. The optional
    ``scale_snapshots=True`` branch is retained only for reproducing old
    exploratory tests.
    """
    if real_ws_only:
        mu, e_com, k, _eta, _r_c, alpha0 = central_real_ws_parameters()
        potential = real_volume_woods_saxon
        spin_orbit_term = None
        is_complex = False
    else:
        mu, e_com, k, _eta, _r_c, alpha0 = central_kd_parameters()
        potential = rose.koning_delaroche.KD_simple
        spin_orbit_term = rose.koning_delaroche.KD_simple_so
        is_complex = True
    if central_value is None:
        central_value = alpha0[param_index]

    train_alphas = make_alphas(alpha0, param_index, train_values)
    test_alphas = make_alphas(alpha0, param_index, test_values)
    central_alpha = alpha0.copy()
    central_alpha[param_index] = central_value

    bounds = np.column_stack(
        [
            np.minimum(train_alphas.min(axis=0), alpha0),
            np.maximum(train_alphas.max(axis=0), alpha0),
        ]
    )
    rho_mesh = np.linspace(1e-8, 8 * np.pi, n_mesh)

    eim_kwargs = dict(
        l_max=l_max,
        coordinate_space_potential=potential,
        n_theta=alpha0.size,
        mu=mu,
        energy=e_com,
        is_complex=is_complex,
        training_info=bounds,
        n_basis=n_U,
        rho_mesh=rho_mesh,
    )
    if spin_orbit_term is not None:
        eim_kwargs["spin_orbit_term"] = spin_orbit_term
    interactions = rose.InteractionEIMSpace(**eim_kwargs)
    base_solver = rose.SchroedingerEquation.make_base_solver(
        s_0=6 * np.pi,
        rk_tols=[1e-9, 1e-9],
        domain=np.array([1e-8, 8 * np.pi]),
    )

    interaction = interactions.interactions[0][0]
    solver = base_solver.clone_for_new_interaction(interaction)
    phi_hf_train = np.array([solver.phi(alpha, rho_mesh) for alpha in train_alphas])
    phi_hf_test = np.array([solver.phi(alpha, rho_mesh) for alpha in test_alphas])
    phi_hf_central = solver.phi(central_alpha, rho_mesh)

    if scale_snapshots:
        phi_basis_train = scale_wavefunctions_like_rose_basis(phi_hf_train, rho_mesh)
        phi_basis_test = scale_wavefunctions_like_rose_basis(phi_hf_test, rho_mesh)
        phi0 = scale_wavefunctions_like_rose_basis(phi_hf_central[np.newaxis, :], rho_mesh)[0]
    else:
        phi_basis_train = phi_hf_train.copy()
        phi_basis_test = phi_hf_test.copy()
        phi0 = phi_hf_central.copy()

    custom_basis = CustomBasis(
        solutions=phi_basis_train.T.copy(),
        phi_0=phi0.copy(),
        rho_mesh=rho_mesh,
        n_basis=n_phi,
        solver=solver,
        subtract_phi0=True,
        use_svd=True,
        center=False,
        scale=False,
    )
    rbe = ReducedBasisEmulator(
        interaction,
        custom_basis,
        s_0=base_solver.s_0,
        initialize_emulator=True,
    )

    coeff_rom_train = np.array([rbe.coefficients(alpha) for alpha in train_alphas])
    coeff_rom_test = np.array([rbe.coefficients(alpha) for alpha in test_alphas])
    phi_rbm_train = np.array([rbe.emulate_wave_function(alpha) for alpha in train_alphas])
    phi_rbm_test = np.array([rbe.emulate_wave_function(alpha) for alpha in test_alphas])
    coeff_train = least_squares_basis_coefficients(rbe.basis, phi_basis_train, rho_mesh)
    coeff_test = least_squares_basis_coefficients(rbe.basis, phi_basis_test, rho_mesh)

    return OneParameterDataset(
        name=name,
        param_index=param_index,
        central_alpha=central_alpha,
        train_values=train_values,
        test_values=test_values,
        train_alphas=train_alphas,
        test_alphas=test_alphas,
        emulator=None,
        rbe=rbe,
        rho_mesh=rho_mesh,
        r_mesh=rho_mesh / k,
        coeff_train=coeff_train,
        coeff_test=coeff_test,
        coeff_rom_train=coeff_rom_train,
        coeff_rom_test=coeff_rom_test,
        phi_hf_train=phi_hf_train,
        phi_hf_test=phi_hf_test,
        phi_basis_train=phi_basis_train,
        phi_basis_test=phi_basis_test,
        phi_rbm_train=phi_rbm_train,
        phi_rbm_test=phi_rbm_test,
    )


def train_multiparameter_rbe(
    name: str,
    train_samples: np.ndarray,
    test_samples: np.ndarray,
    central_sample: np.ndarray,
    feature_scales: np.ndarray,
    n_phi: int = 4,
    n_U: int = 8,
    l_max: int = 1,
    n_mesh: int = 900,
    real_ws_only: bool = False,
):
    """Train the standard ROSE/free-reference ROM for multiparameter samples.

    This is the multiparameter analogue of :func:`train_one_parameter_rbe`.
    It lets ROSE construct its usual reduced basis from snapshots relative to
    the free/reference solution, with snapshot scaling disabled. Use this when
    comparing against the actual ROSE ROM rather than a central-reference
    Galerkin diagnostic.
    """
    rho_mesh = np.linspace(1e-8, 8 * np.pi, n_mesh)
    if real_ws_only:
        mu, e_com, k, _eta, _r_c, _alpha0 = central_real_ws_parameters()
        train_alphas = np.asarray(train_samples[:, :3], dtype=float)
        test_alphas = np.asarray(test_samples[:, :3], dtype=float)
        central_alpha = np.asarray(central_sample[:3], dtype=float)
        eim_training = np.vstack([train_alphas, central_alpha])
        interactions = rose.InteractionEIMSpace(
            l_max=l_max,
            coordinate_space_potential=real_volume_woods_saxon,
            n_theta=3,
            mu=mu,
            energy=e_com,
            is_complex=False,
            training_info=eim_training,
            explicit_training=True,
            n_basis=n_U,
            rho_mesh=rho_mesh,
        )
    else:
        _mu, _e_com, _k, _eta, _r_c, optical_alpha0 = central_kd_parameters()
        train_alphas = energized_alphas_from_samples(train_samples, optical_alpha0)
        test_alphas = energized_alphas_from_samples(test_samples, optical_alpha0)
        central_alpha = energized_alpha_from_sample(central_sample, optical_alpha0)
        eim_training = np.vstack([train_alphas, central_alpha])
        interactions = rose.koning_delaroche.EnergizedKoningDelaroche(
            training_info=eim_training,
            explicit_training=True,
            l_max=l_max,
            n_basis=n_U,
            rho_mesh=rho_mesh,
        )

    base_solver = rose.SchroedingerEquation.make_base_solver(
        s_0=6 * np.pi,
        rk_tols=[1e-9, 1e-9],
        domain=np.array([1e-8, 8 * np.pi]),
    )
    angles = np.linspace(1, 179, 120) * np.pi / 180
    emulator = rose.ScatteringAmplitudeEmulator.from_train(
        interactions,
        train_alphas,
        base_solver=base_solver,
        l_max=l_max,
        angles=angles,
        n_basis=n_phi,
        use_svd=True,
        scale=False,
        s_mesh=rho_mesh,
        Smatrix_abs_tol=1e-8,
    )
    rbe = emulator.rbes[0][0]

    coeff_rom_train = np.array([rbe.coefficients(alpha) for alpha in train_alphas])
    coeff_rom_test = np.array([rbe.coefficients(alpha) for alpha in test_alphas])
    phi_hf_train = np.array([rbe.basis.solver.phi(alpha, rho_mesh) for alpha in train_alphas])
    phi_hf_test = np.array([rbe.basis.solver.phi(alpha, rho_mesh) for alpha in test_alphas])
    phi_rbm_train = np.array([rbe.emulate_wave_function(alpha) for alpha in train_alphas])
    phi_rbm_test = np.array([rbe.emulate_wave_function(alpha) for alpha in test_alphas])
    coeff_train = least_squares_basis_coefficients(rbe.basis, phi_hf_train, rho_mesh)
    coeff_test = least_squares_basis_coefficients(rbe.basis, phi_hf_test, rho_mesh)

    return MultiParameterDataset(
        name=name,
        feature_names=["Vv", "Rv", "av", "E_lab"],
        feature_scales=np.asarray(feature_scales, dtype=float),
        central_sample=np.asarray(central_sample, dtype=float),
        central_alpha=central_alpha,
        train_samples=np.asarray(train_samples, dtype=float),
        test_samples=np.asarray(test_samples, dtype=float),
        train_alphas=train_alphas,
        test_alphas=test_alphas,
        rbe=rbe,
        rho_mesh=rho_mesh,
        coeff_train=coeff_train,
        coeff_test=coeff_test,
        coeff_rom_train=coeff_rom_train,
        coeff_rom_test=coeff_rom_test,
        phi_hf_train=phi_hf_train,
        phi_hf_test=phi_hf_test,
        phi_basis_train=phi_hf_train.copy(),
        phi_basis_test=phi_hf_test.copy(),
        phi_rbm_train=phi_rbm_train,
        phi_rbm_test=phi_rbm_test,
    )


def train_multiparameter_central_rbe(
    name: str,
    train_samples: np.ndarray,
    test_samples: np.ndarray,
    central_sample: np.ndarray,
    feature_scales: np.ndarray,
    n_phi: int = 4,
    n_U: int = 8,
    l_max: int = 1,
    n_mesh: int = 900,
    real_ws_only: bool = False,
    scale_snapshots: bool = False,
):
    """Train central-basis ROSE ROM data for [Vv, Rv, av, E_lab] variation.

    The optical parameters not listed in the sample are held at the central
    KD values. Energy variation enters through ROSE's energized interaction
    vector [E_com, mu, k, optical_alpha...].  If ``real_ws_only=True``, only
    [Vv, Rv, av] are used and the energy entry is ignored.
    """
    rho_mesh = np.linspace(1e-8, 8 * np.pi, n_mesh)
    if real_ws_only:
        mu, e_com, k, _eta, _r_c, _alpha0 = central_real_ws_parameters()
        train_alphas = np.asarray(train_samples[:, :3], dtype=float)
        test_alphas = np.asarray(test_samples[:, :3], dtype=float)
        central_alpha = np.asarray(central_sample[:3], dtype=float)
        eim_training = np.vstack([train_alphas, central_alpha])
        interactions = rose.InteractionEIMSpace(
            l_max=l_max,
            coordinate_space_potential=real_volume_woods_saxon,
            n_theta=3,
            mu=mu,
            energy=e_com,
            is_complex=False,
            training_info=eim_training,
            explicit_training=True,
            n_basis=n_U,
            rho_mesh=rho_mesh,
        )
    else:
        _mu, _e_com, _k, _eta, _r_c, optical_alpha0 = central_kd_parameters()
        train_alphas = energized_alphas_from_samples(train_samples, optical_alpha0)
        test_alphas = energized_alphas_from_samples(test_samples, optical_alpha0)
        central_alpha = energized_alpha_from_sample(central_sample, optical_alpha0)
        eim_training = np.vstack([train_alphas, central_alpha])
        interactions = rose.koning_delaroche.EnergizedKoningDelaroche(
            training_info=eim_training,
            explicit_training=True,
            l_max=l_max,
            n_basis=n_U,
            rho_mesh=rho_mesh,
        )
    base_solver = rose.SchroedingerEquation.make_base_solver(
        s_0=6 * np.pi,
        rk_tols=[1e-9, 1e-9],
        domain=np.array([1e-8, 8 * np.pi]),
    )

    interaction = interactions.interactions[0][0]
    solver = base_solver.clone_for_new_interaction(interaction)
    phi_hf_train = np.array([solver.phi(alpha, rho_mesh) for alpha in train_alphas])
    phi_hf_test = np.array([solver.phi(alpha, rho_mesh) for alpha in test_alphas])
    phi_hf_central = solver.phi(central_alpha, rho_mesh)

    if scale_snapshots:
        phi_basis_train = scale_wavefunctions_like_rose_basis(phi_hf_train, rho_mesh)
        phi_basis_test = scale_wavefunctions_like_rose_basis(phi_hf_test, rho_mesh)
        phi0 = scale_wavefunctions_like_rose_basis(phi_hf_central[np.newaxis, :], rho_mesh)[0]
    else:
        phi_basis_train = phi_hf_train.copy()
        phi_basis_test = phi_hf_test.copy()
        phi0 = phi_hf_central.copy()

    custom_basis = CustomBasis(
        solutions=phi_basis_train.T.copy(),
        phi_0=phi0.copy(),
        rho_mesh=rho_mesh,
        n_basis=n_phi,
        solver=solver,
        subtract_phi0=True,
        use_svd=True,
        center=False,
        scale=False,
    )
    rbe = ReducedBasisEmulator(
        interaction,
        custom_basis,
        s_0=base_solver.s_0,
        initialize_emulator=True,
    )

    coeff_rom_train = np.array([rbe.coefficients(alpha) for alpha in train_alphas])
    coeff_rom_test = np.array([rbe.coefficients(alpha) for alpha in test_alphas])
    phi_rbm_train = np.array([rbe.emulate_wave_function(alpha) for alpha in train_alphas])
    phi_rbm_test = np.array([rbe.emulate_wave_function(alpha) for alpha in test_alphas])
    coeff_train = least_squares_basis_coefficients(rbe.basis, phi_basis_train, rho_mesh)
    coeff_test = least_squares_basis_coefficients(rbe.basis, phi_basis_test, rho_mesh)

    return MultiParameterDataset(
        name=name,
        feature_names=["Vv", "Rv", "av", "E_lab"],
        feature_scales=np.asarray(feature_scales, dtype=float),
        central_sample=np.asarray(central_sample, dtype=float),
        central_alpha=central_alpha,
        train_samples=np.asarray(train_samples, dtype=float),
        test_samples=np.asarray(test_samples, dtype=float),
        train_alphas=train_alphas,
        test_alphas=test_alphas,
        rbe=rbe,
        rho_mesh=rho_mesh,
        coeff_train=coeff_train,
        coeff_test=coeff_test,
        coeff_rom_train=coeff_rom_train,
        coeff_rom_test=coeff_rom_test,
        phi_hf_train=phi_hf_train,
        phi_hf_test=phi_hf_test,
        phi_basis_train=phi_basis_train,
        phi_basis_test=phi_basis_test,
        phi_rbm_train=phi_rbm_train,
        phi_rbm_test=phi_rbm_test,
    )


def train_multiparameter_central_sae(
    name: str,
    train_samples: np.ndarray,
    test_samples: np.ndarray,
    central_sample: np.ndarray,
    feature_scales: np.ndarray,
    n_phi: int = 4,
    n_U: int = 8,
    l_max: int = 10,
    n_mesh: int = 900,
    n_angles: int = 160,
    scale_snapshots: bool = False,
):
    """Train a central-basis scattering-amplitude emulator over 4 parameters.

    This constructs one custom central basis for every ROSE partial-wave channel
    and stores LS coefficient targets for each channel. The ROSE object itself
    provides the traditional ROM and FOM cross sections.
    """
    _mu, _e_com, _k, _eta, _r_c, optical_alpha0 = central_kd_parameters()
    train_alphas = energized_alphas_from_samples(train_samples, optical_alpha0)
    test_alphas = energized_alphas_from_samples(test_samples, optical_alpha0)
    central_alpha = energized_alpha_from_sample(central_sample, optical_alpha0)

    rho_mesh = np.linspace(1e-8, 8 * np.pi, n_mesh)
    angles = np.linspace(1, 179, n_angles) * np.pi / 180

    eim_training = np.vstack([train_alphas, central_alpha])
    interactions = rose.koning_delaroche.EnergizedKoningDelaroche(
        training_info=eim_training,
        explicit_training=True,
        l_max=l_max,
        n_basis=n_U,
        rho_mesh=rho_mesh,
    )
    base_solver = rose.SchroedingerEquation.make_base_solver(
        s_0=6 * np.pi,
        rk_tols=[1e-9, 1e-9],
        domain=np.array([1e-8, 8 * np.pi]),
    )

    bases = []
    coeff_train = []
    coeff_test = []
    for interaction_list in interactions.interactions:
        basis_list = []
        coeff_train_list = []
        coeff_test_list = []
        for interaction in interaction_list:
            solver = base_solver.clone_for_new_interaction(interaction)
            phi_hf_train = np.array([solver.phi(alpha, rho_mesh) for alpha in train_alphas])
            phi_hf_test = np.array([solver.phi(alpha, rho_mesh) for alpha in test_alphas])
            phi_hf_central = solver.phi(central_alpha, rho_mesh)

            if scale_snapshots:
                phi_basis_train = scale_wavefunctions_like_rose_basis(phi_hf_train, rho_mesh)
                phi_basis_test = scale_wavefunctions_like_rose_basis(phi_hf_test, rho_mesh)
                phi0 = scale_wavefunctions_like_rose_basis(phi_hf_central[np.newaxis, :], rho_mesh)[0]
            else:
                phi_basis_train = phi_hf_train.copy()
                phi_basis_test = phi_hf_test.copy()
                phi0 = phi_hf_central.copy()

            custom_basis = CustomBasis(
                solutions=phi_basis_train.T.copy(),
                phi_0=phi0.copy(),
                rho_mesh=rho_mesh,
                n_basis=n_phi,
                solver=solver,
                subtract_phi0=True,
                use_svd=True,
                center=False,
                scale=False,
            )
            basis_list.append(custom_basis)
            coeff_train_list.append(
                least_squares_basis_coefficients(custom_basis, phi_basis_train, rho_mesh)
            )
            coeff_test_list.append(
                least_squares_basis_coefficients(custom_basis, phi_basis_test, rho_mesh)
            )
        bases.append(basis_list)
        coeff_train.append(coeff_train_list)
        coeff_test.append(coeff_test_list)

    sae = rose.ScatteringAmplitudeEmulator(
        interactions,
        bases,
        l_max=l_max,
        angles=angles,
        s_0=base_solver.s_0,
        Smatrix_abs_tol=1e-8,
        initialize_emulator=True,
    )

    return MultiPartialWaveDataset(
        name=name,
        feature_names=["Vv", "Rv", "av", "E_lab"],
        feature_scales=np.asarray(feature_scales, dtype=float),
        central_sample=np.asarray(central_sample, dtype=float),
        central_alpha=central_alpha,
        train_samples=np.asarray(train_samples, dtype=float),
        test_samples=np.asarray(test_samples, dtype=float),
        train_alphas=train_alphas,
        test_alphas=test_alphas,
        sae=sae,
        rho_mesh=rho_mesh,
        angles=angles,
        coeff_train=coeff_train,
        coeff_test=coeff_test,
    )


def train_generic_central_sae(
    name: str,
    interaction_space,
    train_samples: np.ndarray,
    test_samples: np.ndarray,
    central_sample: np.ndarray,
    feature_scales: np.ndarray,
    feature_names: list[str],
    base_solver,
    rho_mesh: np.ndarray,
    angles: np.ndarray,
    n_phi: int = 4,
    scale_snapshots: bool = False,
):
    """Train central-basis SAE and LS targets for an arbitrary interaction space."""
    bases = []
    coeff_train = []
    coeff_test = []
    for interaction_list in interaction_space.interactions:
        basis_list = []
        coeff_train_list = []
        coeff_test_list = []
        for interaction in interaction_list:
            solver = base_solver.clone_for_new_interaction(interaction)
            phi_hf_train = np.array([solver.phi(alpha, rho_mesh) for alpha in train_samples])
            phi_hf_test = np.array([solver.phi(alpha, rho_mesh) for alpha in test_samples])
            phi_hf_central = solver.phi(central_sample, rho_mesh)

            if scale_snapshots:
                phi_basis_train = scale_wavefunctions_like_rose_basis(phi_hf_train, rho_mesh)
                phi_basis_test = scale_wavefunctions_like_rose_basis(phi_hf_test, rho_mesh)
                phi0 = scale_wavefunctions_like_rose_basis(phi_hf_central[np.newaxis, :], rho_mesh)[0]
            else:
                phi_basis_train = phi_hf_train.copy()
                phi_basis_test = phi_hf_test.copy()
                phi0 = phi_hf_central.copy()

            custom_basis = CustomBasis(
                solutions=phi_basis_train.T.copy(),
                phi_0=phi0.copy(),
                rho_mesh=rho_mesh,
                n_basis=n_phi,
                solver=solver,
                subtract_phi0=True,
                use_svd=True,
                center=False,
                scale=False,
            )
            basis_list.append(custom_basis)
            coeff_train_list.append(
                least_squares_basis_coefficients(custom_basis, phi_basis_train, rho_mesh)
            )
            coeff_test_list.append(
                least_squares_basis_coefficients(custom_basis, phi_basis_test, rho_mesh)
            )
        bases.append(basis_list)
        coeff_train.append(coeff_train_list)
        coeff_test.append(coeff_test_list)

    sae = rose.ScatteringAmplitudeEmulator(
        interaction_space,
        bases,
        l_max=interaction_space.l_max,
        angles=angles,
        s_0=base_solver.s_0,
        Smatrix_abs_tol=1e-8,
        initialize_emulator=True,
    )

    return GenericCentralSAEDataset(
        name=name,
        feature_names=list(feature_names),
        feature_scales=np.asarray(feature_scales, dtype=float),
        central_sample=np.asarray(central_sample, dtype=float),
        train_samples=np.asarray(train_samples, dtype=float),
        test_samples=np.asarray(test_samples, dtype=float),
        sae=sae,
        rho_mesh=rho_mesh,
        angles=angles,
        coeff_train=coeff_train,
        coeff_test=coeff_test,
    )


def reduced_matrix_and_rhs(rbe, alpha: np.ndarray):
    """Reconstruct the reduced linear system used by ROSE: M(alpha) a = b(alpha)."""
    invk, beta = rbe.interaction.coefficients(alpha)
    matrix = rbe.A_13 + np.einsum("i,ijk", beta, rbe.A_2) + invk * rbe.A_3_coulomb
    rhs = rbe.b_13 + beta @ rbe.b_2 + invk * rbe.b_3_coulomb
    return matrix, rhs


def affine_galerkin_pieces(rbe, alpha0: np.ndarray, param_index: int):
    """Extract exact affine pieces for changing one EIM coefficient linearly.

    This is exact for parameters that are affine in ROSE's interaction
    coefficients. For geometric parameters such as Rv, use finite differences
    on reduced_matrix_and_rhs instead.
    """
    invk, beta0 = rbe.interaction.coefficients(alpha0)
    matrix0 = rbe.A_13 + invk * rbe.A_3_coulomb
    rhs0 = rbe.b_13 + invk * rbe.b_3_coulomb
    for j, beta_j in enumerate(beta0):
        if j == param_index:
            continue
        matrix0 = matrix0 + beta_j * rbe.A_2[j]
        rhs0 = rhs0 + beta_j * rbe.b_2[j]
    return matrix0, rbe.A_2[param_index].copy(), rhs0, rbe.b_2[param_index].copy()


def finite_difference_pieces(rbe, alpha0: np.ndarray, param_index: int, h: float):
    """Linearize M and rhs with respect to a physical parameter."""
    alpha_p = alpha0.copy()
    alpha_m = alpha0.copy()
    alpha_p[param_index] += h
    alpha_m[param_index] -= h

    m0, b0 = reduced_matrix_and_rhs(rbe, alpha0)
    mp, bp = reduced_matrix_and_rhs(rbe, alpha_p)
    mm, bm = reduced_matrix_and_rhs(rbe, alpha_m)
    return m0, (mp - mm) / (2 * h), b0, (bp - bm) / (2 * h)


def complex_pack(arrays: list[np.ndarray]):
    flat = np.concatenate([a.ravel() for a in arrays])
    return np.concatenate([flat.real, flat.imag])


def complex_unpack(vector: np.ndarray, shapes: list[tuple[int, ...]]):
    n_complex = vector.size // 2
    flat = vector[:n_complex] + 1j * vector[n_complex:]
    arrays = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        arrays.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return arrays


def model_predict(feature_values, matrices, vectors):
    preds = []
    for feats in feature_values:
        matrix = sum(f * m for f, m in zip(feats, matrices))
        rhs = sum(f * v for f, v in zip(feats, vectors))
        preds.append(np.linalg.solve(matrix, rhs))
    return np.array(preds)


def coefficient_residual_vector(pred, ref):
    diff = (pred - ref).ravel()
    return np.concatenate([diff.real, diff.imag])


def relative_rows(pred, ref):
    return np.linalg.norm(pred - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)


def reconstruct_wavefunctions(dataset: OneParameterDataset, coeffs: np.ndarray):
    basis = dataset.rbe.basis
    return np.array([basis.phi_hat(c[np.newaxis, :]) for c in coeffs])


def align_wavefunctions(reference: np.ndarray, candidate: np.ndarray):
    """Align candidate wavefunctions to reference by matching max-amplitude point."""
    aligned = []
    for ref, cand in zip(reference, candidate):
        idx = np.argmax(np.abs(ref))
        if np.abs(cand[idx]) > 0:
            aligned.append(cand * (ref[idx] / cand[idx]))
        else:
            aligned.append(cand)
    return np.array(aligned)


def train_implicit_model(
    name: str,
    feature_values: np.ndarray,
    coeff_ref: np.ndarray,
    init_matrices: list[np.ndarray],
    init_vectors: list[np.ndarray],
    gauge: tuple[str, int, int, complex] | None = None,
    prior_strength: float = 0.0,
    max_nfev: int = 2000,
):
    """Fit M(features) a = rhs(features) to reference coefficients."""
    shapes = [a.shape for a in init_matrices + init_vectors]
    init_arrays = init_matrices + init_vectors
    p0_full = complex_pack(init_arrays)
    p_prior = p0_full.copy()

    fixed_index = None
    fixed_value = None
    if gauge is not None:
        kind, matrix_i, row, value = gauge
        if kind != "matrix_entry":
            raise ValueError("Only matrix_entry gauge is implemented.")
        n = init_matrices[matrix_i].shape[0]
        complex_offset = sum(m.size for m in init_matrices[:matrix_i]) + row * n
        fixed_index = complex_offset
        fixed_value = value

        arrays = complex_unpack(p0_full, shapes)
        arrays[matrix_i][row, row] = fixed_value
        p0_full = complex_pack(arrays)
        p_prior = p0_full.copy()

        mask = np.ones(p0_full.size, dtype=bool)
        n_complex = p0_full.size // 2
        mask[fixed_index] = False
        mask[fixed_index + n_complex] = False
        p0 = p0_full[mask]
    else:
        mask = None
        p0 = p0_full

    losses = []

    def inflate(p):
        if mask is None:
            full = p
        else:
            full = p0_full.copy()
            full[mask] = p
            n_complex = full.size // 2
            full[fixed_index] = np.real(fixed_value)
            full[fixed_index + n_complex] = np.imag(fixed_value)
        return full

    def residuals(p):
        full = inflate(p)
        arrays = complex_unpack(full, shapes)
        matrices = arrays[: len(init_matrices)]
        vectors = arrays[len(init_matrices) :]
        try:
            pred = model_predict(feature_values, matrices, vectors)
            res = coefficient_residual_vector(pred, coeff_ref) / np.sqrt(coeff_ref.size)
        except np.linalg.LinAlgError:
            res = np.full(coeff_ref.size * 2, 1e6)

        if prior_strength > 0:
            prior = np.sqrt(prior_strength) * (full - p_prior)
            res = np.concatenate([res, prior])
        losses.append(float(np.mean(res[: coeff_ref.size * 2] ** 2)))
        return res

    start = time.perf_counter()
    result = least_squares(
        residuals,
        p0,
        max_nfev=max_nfev,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    train_seconds = time.perf_counter() - start

    full = inflate(result.x)
    arrays = complex_unpack(full, shapes)
    matrices = arrays[: len(init_matrices)]
    vectors = arrays[len(init_matrices) :]
    return FitResult(
        name=name,
        params=full,
        matrices=matrices,
        vectors=vectors,
        losses=np.array(losses),
        train_seconds=train_seconds,
        nfev=result.nfev,
        success=result.success,
        message=result.message,
    )


def operator_residual_vector(feature_values, coeff_ref, matrices, vectors):
    """Evaluate residuals M(x) a_ref - b(x) without solving for a."""
    residuals = []
    for feats, coeff in zip(feature_values, coeff_ref):
        matrix = sum(f * m for f, m in zip(feats, matrices))
        rhs = sum(f * v for f, v in zip(feats, vectors))
        residuals.append(matrix @ coeff - rhs)
    residuals = np.asarray(residuals).ravel()
    return np.concatenate([residuals.real, residuals.imag])


def operator_unknown_shapes(n_features: int, n_basis: int):
    matrix_shapes = [(n_basis, n_basis) for _ in range(n_features)]
    vector_shapes = [(n_basis,) for _ in range(n_features)]
    return matrix_shapes + vector_shapes


def build_operator_design(
    feature_values: np.ndarray,
    coeff_ref: np.ndarray,
    init_matrices: list[np.ndarray],
    init_vectors: list[np.ndarray],
    gauge: tuple[str, int, int, complex],
):
    """Build the complex linear system D theta = y for RF-LROM.

    The unknown order is all matrix-feature blocks followed by all vector
    feature blocks: M0, M1, ..., b0, b1, ...
    """
    if gauge is None:
        raise ValueError("Residual fitting needs a gauge to avoid the zero solution.")
    kind, matrix_i, row, fixed_value = gauge
    if kind != "matrix_entry":
        raise ValueError("Only matrix_entry gauge is implemented.")

    n_features = len(init_matrices)
    n = init_matrices[0].shape[0]
    n_matrix_unknowns = n_features * n * n
    n_vector_unknowns = n_features * n
    n_unknowns = n_matrix_unknowns + n_vector_unknowns
    shapes = operator_unknown_shapes(n_features, n)

    fixed_complex_index = matrix_i * n * n + row * n + row
    keep = np.ones(n_unknowns, dtype=bool)
    keep[fixed_complex_index] = False

    design = np.zeros((feature_values.shape[0] * n, n_unknowns), dtype=np.complex128)

    eq = 0
    for feats, coeff in zip(feature_values, coeff_ref):
        for r in range(n):
            for j, feat in enumerate(feats):
                matrix_offset = j * n * n
                for c in range(n):
                    design[eq, matrix_offset + r * n + c] = feat * coeff[c]

                vector_offset = n_matrix_unknowns + j * n
                design[eq, vector_offset + r] = -feat
            eq += 1

    target = -design[:, fixed_complex_index] * fixed_value
    return OperatorDesign(
        design=design,
        target=target,
        keep=keep,
        fixed_complex_index=fixed_complex_index,
        fixed_value=fixed_value,
        shapes=shapes,
        n_features=n_features,
        n_basis=n,
        n_unknowns=n_unknowns,
    )


def build_operator_action_design(
    feature_values: np.ndarray,
    coeff_inputs: np.ndarray,
    residual_targets: np.ndarray,
    init_matrices: list[np.ndarray],
    init_vectors: list[np.ndarray],
    gauge: tuple[str, int, int, complex],
):
    """Build D theta = y for target residuals M(x) a_input - b(x).

    Unlike build_operator_design, this accepts nonzero residual targets. It can
    therefore train RF-LROM on off-solution-manifold operator action.
    """
    design_info = build_operator_design(
        feature_values,
        coeff_inputs,
        init_matrices,
        init_vectors,
        gauge,
    )
    targets = np.asarray(residual_targets, dtype=np.complex128).reshape(-1)
    if targets.shape[0] != design_info.design.shape[0]:
        raise ValueError(
            "residual_targets must have one complex vector per feature row."
        )
    design_info.target = targets - design_info.design[:, design_info.fixed_complex_index] * design_info.fixed_value
    return design_info


def fit_operator_action_linear(
    name: str,
    feature_values: np.ndarray,
    coeff_inputs: np.ndarray,
    residual_targets: np.ndarray,
    init_matrices: list[np.ndarray],
    init_vectors: list[np.ndarray],
    gauge: tuple[str, int, int, complex],
    prior_strength: float = 0.0,
    prior_matrices: list[np.ndarray] | None = None,
    prior_vectors: list[np.ndarray] | None = None,
):
    """Fit M(features) a_input - b(features) to target residuals."""
    init_arrays = init_matrices + init_vectors
    if prior_matrices is None:
        prior_matrices = init_matrices
    if prior_vectors is None:
        prior_vectors = init_vectors
    prior_arrays = prior_matrices + prior_vectors
    design_info = build_operator_action_design(
        feature_values,
        coeff_inputs,
        residual_targets,
        init_matrices,
        init_vectors,
        gauge,
    )
    reduced_design = design_info.design[:, design_info.keep]
    reduced_target = design_info.target
    reduced_design, reduced_target = augment_design_with_prior(
        reduced_design,
        reduced_target,
        design_info.keep,
        prior_arrays,
        prior_strength,
    )

    start = time.perf_counter()
    solution, _residual_sum, rank, singular_values = np.linalg.lstsq(
        reduced_design,
        reduced_target,
        rcond=None,
    )
    train_seconds = time.perf_counter() - start

    full_complex = np.zeros(design_info.n_unknowns, dtype=np.complex128)
    full_complex[design_info.fixed_complex_index] = design_info.fixed_value
    full_complex[design_info.keep] = solution

    full_packed = np.concatenate([full_complex.real, full_complex.imag])
    arrays = complex_unpack(full_packed, design_info.shapes)
    matrices = arrays[: len(init_matrices)]
    vectors = arrays[len(init_matrices) :]

    residual = operator_residual_vector(feature_values, coeff_inputs, matrices, vectors)
    target = np.concatenate([residual_targets.reshape(-1).real, residual_targets.reshape(-1).imag])
    residual_mse = float(np.mean((residual - target) ** 2))
    return ResidualFitResult(
        name=name,
        params=full_packed,
        matrices=matrices,
        vectors=vectors,
        train_seconds=train_seconds,
        residual_mse=residual_mse,
        rank=int(rank),
        singular_values=singular_values,
        success=np.isfinite(residual_mse),
        message="linear operator-action least squares completed",
    )


def fit_vv_identity_m0_rf(
    name: str,
    vv_values: np.ndarray,
    coeff_ref: np.ndarray,
    init_m1: np.ndarray,
    init_b0: np.ndarray,
    init_b1: np.ndarray,
    prior_strength: float = 0.0,
):
    """Fit the Vv RF-LROM gauge with M0 fixed to the identity.

    Equation:
        (I + Vv M1) a = b0 + Vv b1

    Unknowns are M1, b0, b1. This removes the left-multiplication gauge
    freedom more strongly than fixing one matrix element.
    """
    n = coeff_ref.shape[1]
    n_unknowns = n * n + 2 * n
    design = np.zeros((len(vv_values) * n, n_unknowns), dtype=np.complex128)
    target = np.zeros(len(vv_values) * n, dtype=np.complex128)

    eq = 0
    for vv, coeff in zip(vv_values, coeff_ref):
        for r in range(n):
            # M1 row r contributes vv * M1[r, :] @ a.
            for c in range(n):
                design[eq, r * n + c] = vv * coeff[c]
            # b0 and b1 are moved to the left as -b0 - vv*b1.
            b0_offset = n * n
            b1_offset = n * n + n
            design[eq, b0_offset + r] = -1.0
            design[eq, b1_offset + r] = -vv
            # Target is -I @ a.
            target[eq] = -coeff[r]
            eq += 1

    if prior_strength > 0:
        prior = np.concatenate([init_m1.ravel(), init_b0.ravel(), init_b1.ravel()])
        scale = np.sqrt(prior_strength)
        design = np.vstack([design, scale * np.eye(n_unknowns, dtype=np.complex128)])
        target = np.concatenate([target, scale * prior])

    start = time.perf_counter()
    solution, _residual_sum, rank, singular_values = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )
    train_seconds = time.perf_counter() - start

    m1 = solution[: n * n].reshape(n, n)
    b0 = solution[n * n : n * n + n]
    b1 = solution[n * n + n :]
    identity = np.eye(n, dtype=np.complex128)

    residual = operator_residual_vector(
        np.column_stack([np.ones_like(vv_values), vv_values]),
        coeff_ref,
        [identity, m1],
        [b0, b1],
    )

    return ResidualFitResult(
        name=name,
        params=np.concatenate([solution.real, solution.imag]),
        matrices=[identity, m1],
        vectors=[b0, b1],
        train_seconds=train_seconds,
        residual_mse=float(np.mean(residual**2)),
        rank=int(rank),
        singular_values=singular_values,
        success=True,
        message="Vv RF-LROM with M0=I completed",
    )


def fit_identity_m0_rf(
    name: str,
    feature_values: np.ndarray,
    coeff_ref: np.ndarray,
):
    """Fit central RF-LROM with M0 fixed to identity and no prior.

    Equation:
        (I + x1 M1 + ... + xp Mp) a = b0 + x1 b1 + ... + xp bp

    feature_values must contain the leading constant column [1, x1, ..., xp].
    Unknowns are M1..Mp and b0..bp.
    """
    features = np.asarray(feature_values)
    if not np.allclose(features[:, 0], 1.0):
        raise ValueError("feature_values must include a leading constant column.")
    n_samples = features.shape[0]
    n_features = features.shape[1]
    n_param_features = n_features - 1
    n = coeff_ref.shape[1]
    n_matrix_unknowns = n_param_features * n * n
    n_vector_unknowns = n_features * n
    n_unknowns = n_matrix_unknowns + n_vector_unknowns
    design = np.zeros((n_samples * n, n_unknowns), dtype=np.complex128)
    target = np.zeros(n_samples * n, dtype=np.complex128)

    eq = 0
    for feats, coeff in zip(features, coeff_ref):
        param_feats = feats[1:]
        for r in range(n):
            for j, feat in enumerate(param_feats):
                matrix_offset = j * n * n
                for c in range(n):
                    design[eq, matrix_offset + r * n + c] = feat * coeff[c]
            vector_offset = n_matrix_unknowns
            for j, feat in enumerate(feats):
                design[eq, vector_offset + j * n + r] = -feat
            target[eq] = -coeff[r]
            eq += 1

    start = time.perf_counter()
    solution, _residual_sum, rank, singular_values = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )
    train_seconds = time.perf_counter() - start

    matrices = [np.eye(n, dtype=np.complex128)]
    offset = 0
    for _j in range(n_param_features):
        matrices.append(solution[offset : offset + n * n].reshape(n, n))
        offset += n * n

    vectors = []
    vector_coeffs = solution[n_matrix_unknowns:]
    for j in range(n_features):
        vectors.append(vector_coeffs[j * n : (j + 1) * n])

    residual = operator_residual_vector(features, coeff_ref, matrices, vectors)

    return ResidualFitResult(
        name=name,
        params=np.concatenate([solution.real, solution.imag]),
        matrices=matrices,
        vectors=vectors,
        train_seconds=train_seconds,
        residual_mse=float(np.mean(residual**2)),
        rank=int(rank),
        singular_values=singular_values,
        success=True,
        message="central RF-LROM with M0=I completed",
    )


def fit_identity_m0_rf_real(
    name: str,
    feature_values: np.ndarray,
    coeff_ref: np.ndarray,
):
    """Fit central RF-LROM with M0=I and real operator blocks.

    Features may be complex. Unknown matrix/vector entries are constrained to
    be real, so complex dependence can enter through the predictors while the
    learned operator blocks remain real.
    """
    features = np.asarray(feature_values)
    if not np.allclose(features[:, 0], 1.0):
        raise ValueError("feature_values must include a leading constant column.")
    n_samples = features.shape[0]
    n_features = features.shape[1]
    n_param_features = n_features - 1
    n = coeff_ref.shape[1]
    n_matrix_unknowns = n_param_features * n * n
    n_vector_unknowns = n_features * n
    n_unknowns = n_matrix_unknowns + n_vector_unknowns
    design_complex = np.zeros((n_samples * n, n_unknowns), dtype=np.complex128)
    target_complex = np.zeros(n_samples * n, dtype=np.complex128)

    eq = 0
    for feats, coeff in zip(features, coeff_ref):
        param_feats = feats[1:]
        for r in range(n):
            for j, feat in enumerate(param_feats):
                matrix_offset = j * n * n
                for c in range(n):
                    design_complex[eq, matrix_offset + r * n + c] = feat * coeff[c]
            vector_offset = n_matrix_unknowns
            for j, feat in enumerate(feats):
                design_complex[eq, vector_offset + j * n + r] = -feat
            target_complex[eq] = -coeff[r]
            eq += 1

    design = np.vstack([design_complex.real, design_complex.imag])
    target = np.concatenate([target_complex.real, target_complex.imag])

    start = time.perf_counter()
    solution, _residual_sum, rank, singular_values = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )
    train_seconds = time.perf_counter() - start

    matrices = [np.eye(n, dtype=np.complex128)]
    offset = 0
    for _j in range(n_param_features):
        matrices.append(solution[offset : offset + n * n].reshape(n, n).astype(np.complex128))
        offset += n * n

    vectors = []
    vector_coeffs = solution[n_matrix_unknowns:]
    for j in range(n_features):
        vectors.append(vector_coeffs[j * n : (j + 1) * n].astype(np.complex128))

    residual = operator_residual_vector(features, coeff_ref, matrices, vectors)

    return ResidualFitResult(
        name=name,
        params=solution,
        matrices=matrices,
        vectors=vectors,
        train_seconds=train_seconds,
        residual_mse=float(np.mean(residual**2)),
        rank=int(rank),
        singular_values=singular_values,
        success=True,
        message="central RF-LROM with M0=I and real operator blocks completed",
    )


def s_matrix_from_coefficients(rbe, coeff: np.ndarray):
    """Convert reduced coefficients into the channel S-matrix element."""
    x = np.hstack((1, coeff))
    phi = np.dot(x, rbe.asymptotic_vals)
    phi_prime = np.dot(x, rbe.asymptotic_ders)
    r_matrix = 1 / rbe.s_0 * phi / phi_prime
    return (rbe.Hm - rbe.s_0 * r_matrix * rbe.Hmp) / (
        rbe.Hp - rbe.s_0 * r_matrix * rbe.Hpp
    )


def lrom_smatrix_elements(sae, lrom_fits: list[list[ResidualFitResult]], features):
    """Assemble spin-up/down S-matrix arrays from per-channel LROM fits."""
    n_l = len(sae.rbes)
    splus = np.zeros(n_l, dtype=np.complex128)
    sminus = np.zeros(n_l, dtype=np.complex128)
    feature_row = np.asarray(features)[np.newaxis, :]
    for ell in range(n_l):
        rbe_plus = sae.rbes[ell][0]
        coeff_plus = model_predict(
            feature_row,
            lrom_fits[ell][0].matrices,
            lrom_fits[ell][0].vectors,
        )[0]
        splus[ell] = s_matrix_from_coefficients(rbe_plus, coeff_plus)
        if ell == 0:
            sminus[ell] = splus[ell]
        else:
            rbe_minus = sae.rbes[ell][1]
            coeff_minus = model_predict(
                feature_row,
                lrom_fits[ell][1].matrices,
                lrom_fits[ell][1].vectors,
            )[0]
            sminus[ell] = s_matrix_from_coefficients(rbe_minus, coeff_minus)
    return splus, sminus


def rom_smatrix_elements_fixed(sae, alpha: np.ndarray):
    """ROSE ROM S-matrix elements using all constructed partial-wave channels."""
    n_l = len(sae.rbes)
    splus = np.zeros(n_l, dtype=np.complex128)
    sminus = np.zeros(n_l, dtype=np.complex128)
    splus[0] = sae.rbes[0][0].S_matrix_element(alpha)
    sminus[0] = splus[0]
    for ell in range(1, n_l):
        splus[ell] = sae.rbes[ell][0].S_matrix_element(alpha)
        sminus[ell] = sae.rbes[ell][1].S_matrix_element(alpha)
    return splus, sminus


def exact_smatrix_elements_fixed(sae, alpha: np.ndarray):
    """FOM S-matrix elements using all constructed partial-wave channels."""
    n_l = len(sae.rbes)
    splus = np.zeros(n_l, dtype=np.complex128)
    sminus = np.zeros(n_l, dtype=np.complex128)
    splus[0] = sae.rbes[0][0].basis.solver.smatrix(alpha)
    sminus[0] = splus[0]
    for ell in range(1, n_l):
        splus[ell] = sae.rbes[ell][0].basis.solver.smatrix(alpha)
        sminus[ell] = sae.rbes[ell][1].basis.solver.smatrix(alpha)
    return splus, sminus


def dsdo_from_smatrix(sae, alpha: np.ndarray, splus: np.ndarray, sminus: np.ndarray):
    return sae.calculate_xs(splus, sminus, alpha).dsdo


def complex_unknowns_from_arrays(arrays: list[np.ndarray]):
    packed = complex_pack(arrays)
    n_complex = packed.size // 2
    return packed[:n_complex] + 1j * packed[n_complex:]


def augment_design_with_prior(
    design: np.ndarray,
    target: np.ndarray,
    keep: np.ndarray,
    prior_arrays: list[np.ndarray],
    prior_strength: float,
):
    """Append sqrt(lambda) * theta = sqrt(lambda) * theta_prior rows."""
    if prior_strength <= 0:
        return design, target
    prior_complex = complex_unknowns_from_arrays(prior_arrays)
    scale = np.sqrt(prior_strength)
    prior_design = scale * np.eye(design.shape[1], dtype=np.complex128)
    prior_target = scale * prior_complex[keep]
    return np.vstack([design, prior_design]), np.concatenate([target, prior_target])


def design_svd_summary(design: np.ndarray):
    """Return singular values, rank, and numerical nullity for a design matrix."""
    singular_values = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    return {
        "singular_values": singular_values,
        "rank": rank,
        "nullity": int(design.shape[1] - rank),
        "shape": design.shape,
    }


def unpack_operator_complex_vector(theta: np.ndarray, n_features: int, n_basis: int):
    shapes = operator_unknown_shapes(n_features, n_basis)
    arrays = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        arrays.append(theta[offset : offset + size].reshape(shape))
        offset += size
    matrices = arrays[:n_features]
    vectors = arrays[n_features:]
    return matrices, vectors


def null_vector_block_norms(design_info: OperatorDesign, singular_cutoff=None, max_vectors: int = 6):
    """Describe dominant blocks in the numerical nullspace of an RF-LROM design."""
    reduced_design = design_info.design[:, design_info.keep]
    _u, singular_values, vh = np.linalg.svd(reduced_design, full_matrices=True)
    if singular_cutoff is None:
        singular_cutoff = np.finfo(float).eps * max(reduced_design.shape) * singular_values[0]
    rank = int(np.sum(singular_values > singular_cutoff))
    rows = []
    for local_i, reduced_vec in enumerate(vh[rank : rank + max_vectors]):
        full_vec = np.zeros(design_info.n_unknowns, dtype=np.complex128)
        full_vec[design_info.keep] = reduced_vec
        matrices, vectors = unpack_operator_complex_vector(
            full_vec,
            design_info.n_features,
            design_info.n_basis,
        )
        row = {
            "null_vector": local_i,
            "singular_value_cutoff": float(singular_cutoff),
        }
        for j, matrix in enumerate(matrices):
            row[f"M{j}_norm"] = float(np.linalg.norm(matrix))
        for j, vector in enumerate(vectors):
            row[f"b{j}_norm"] = float(np.linalg.norm(vector))
        rows.append(row)
    return rows


def null_vector_blocks(design_info: OperatorDesign, vector_index: int = 0, singular_cutoff=None):
    """Return matrix/vector blocks for one numerical null vector."""
    reduced_design = design_info.design[:, design_info.keep]
    _u, singular_values, vh = np.linalg.svd(reduced_design, full_matrices=True)
    if singular_cutoff is None:
        singular_cutoff = np.finfo(float).eps * max(reduced_design.shape) * singular_values[0]
    rank = int(np.sum(singular_values > singular_cutoff))
    null_vectors = vh[rank:]
    if vector_index >= len(null_vectors):
        raise IndexError(
            f"Requested null vector {vector_index}, but only {len(null_vectors)} are available."
        )
    full_vec = np.zeros(design_info.n_unknowns, dtype=np.complex128)
    full_vec[design_info.keep] = null_vectors[vector_index]
    matrices, vectors = unpack_operator_complex_vector(
        full_vec,
        design_info.n_features,
        design_info.n_basis,
    )
    return {
        "rank": rank,
        "singular_value_cutoff": float(singular_cutoff),
        "matrices": matrices,
        "vectors": vectors,
        "singular_values": singular_values,
    }


def fit_operator_residual_linear(
    name: str,
    feature_values: np.ndarray,
    coeff_ref: np.ndarray,
    init_matrices: list[np.ndarray],
    init_vectors: list[np.ndarray],
    gauge: tuple[str, int, int, complex],
    prior_strength: float = 0.0,
    prior_matrices: list[np.ndarray] | None = None,
    prior_vectors: list[np.ndarray] | None = None,
):
    """Fit M(features) a_ref = rhs(features) by one linear least-squares solve.

    The residual objective is homogeneous in all matrix/vector entries, so a
    gauge is required; otherwise the all-zero operator is a perfect solution.
    Currently the gauge fixes one diagonal matrix entry, matching the gauge used
    by train_implicit_model.
    """
    init_arrays = init_matrices + init_vectors
    if prior_matrices is None:
        prior_matrices = init_matrices
    if prior_vectors is None:
        prior_vectors = init_vectors
    prior_arrays = prior_matrices + prior_vectors
    design_info = build_operator_design(
        feature_values,
        coeff_ref,
        init_matrices,
        init_vectors,
        gauge,
    )
    reduced_design = design_info.design[:, design_info.keep]
    reduced_target = design_info.target
    reduced_design, reduced_target = augment_design_with_prior(
        reduced_design,
        reduced_target,
        design_info.keep,
        prior_arrays,
        prior_strength,
    )

    start = time.perf_counter()
    solution, residual_sum, rank, singular_values = np.linalg.lstsq(
        reduced_design,
        reduced_target,
        rcond=None,
    )
    train_seconds = time.perf_counter() - start

    full_complex = np.zeros(design_info.n_unknowns, dtype=np.complex128)
    full_complex[design_info.fixed_complex_index] = design_info.fixed_value
    full_complex[design_info.keep] = solution

    full_packed = np.concatenate([full_complex.real, full_complex.imag])
    arrays = complex_unpack(full_packed, design_info.shapes)
    matrices = arrays[: len(init_matrices)]
    vectors = arrays[len(init_matrices) :]

    residual = operator_residual_vector(feature_values, coeff_ref, matrices, vectors)
    residual_mse = float(np.mean(residual**2))
    success = np.isfinite(residual_mse)
    message = "linear operator-residual least squares completed"

    return ResidualFitResult(
        name=name,
        params=full_packed,
        matrices=matrices,
        vectors=vectors,
        train_seconds=train_seconds,
        residual_mse=residual_mse,
        rank=int(rank),
        singular_values=singular_values,
        success=success,
        message=message,
    )


def solve_strategy_benchmark(matrices: np.ndarray, rhs: np.ndarray, repeats: int = 200):
    """Compare small-system solve strategies for Phase 3."""
    import scipy.linalg as sla

    trusted = np.array([np.linalg.solve(m, b) for m, b in zip(matrices, rhs)])
    rows = []

    def add(name, elapsed, solutions):
        residuals = np.linalg.norm(
            np.einsum("nij,nj->ni", matrices, solutions) - rhs,
            axis=1,
        )
        diffs = np.linalg.norm(solutions - trusted, axis=1)
        rows.append(
            {
                "method": name,
                "seconds": elapsed,
                "max_residual": float(residuals.max()),
                "max_diff_vs_numpy": float(diffs.max()),
            }
        )

    start = time.perf_counter()
    for _ in range(repeats):
        sol = np.array([np.linalg.solve(m, b) for m, b in zip(matrices, rhs)])
    add("numpy.linalg.solve loop", time.perf_counter() - start, sol)

    start = time.perf_counter()
    for _ in range(repeats):
        sol = np.array([sla.solve(m, b, assume_a="gen") for m, b in zip(matrices, rhs)])
    add("scipy.linalg.solve loop", time.perf_counter() - start, sol)

    start = time.perf_counter()
    for _ in range(repeats):
        sol = np.array([np.linalg.inv(m) @ b for m, b in zip(matrices, rhs)])
    add("explicit inverse loop", time.perf_counter() - start, sol)

    return rows
