from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numba import njit
import numpy as np


TARGET = (40, 20)
PROJECTILE = (1, 0)
E_LAB = 14.1


def ensure_scipy_spherical_harmonic_compat() -> None:
    import scipy.special

    if hasattr(scipy.special, "sph_harm") or not hasattr(scipy.special, "sph_harm_y"):
        return

    def sph_harm(m: Any, n: Any, theta: Any, phi: Any) -> Any:
        # legacy sph_harm took (theta=azimuthal, phi=polar);
        # sph_harm_y takes angles in (polar, azimuthal) order
        return scipy.special.sph_harm_y(n, m, phi, theta)

    scipy.special.sph_harm = sph_harm


def import_rose() -> Any:
    ensure_scipy_spherical_harmonic_compat()
    try:
        import rose
    except ImportError as exc:
        raise ImportError(
            "The LROM benchmark requires nuclear-rose, imported as rose. "
            "Install it with `python -m pip install nuclear-rose`."
        ) from exc
    return rose


@dataclass(frozen=True)
class KDParameters:
    mu: float
    e_com: float
    k: float
    eta: float
    r_c: float
    alpha: np.ndarray


@dataclass(frozen=True)
class RealWSParameters:
    mu: float
    e_com: float
    k: float
    eta: float
    r_c: float
    alpha: np.ndarray


@dataclass(frozen=True)
class RealWSProblem:
    params: RealWSParameters
    rho_mesh: np.ndarray
    r_mesh: np.ndarray
    interaction: Any
    solver: Any
    base_solver: Any

    def solve_phi(self, alpha: np.ndarray) -> np.ndarray:
        return self.solver.phi(np.asarray(alpha, dtype=float), self.rho_mesh)

    def solve_wavefunctions(self, alphas: np.ndarray) -> np.ndarray:
        return np.asarray([self.solve_phi(alpha) for alpha in np.asarray(alphas, dtype=float)])

    def potential(self, r: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return real_woods_saxon_potential(r, alpha)


def central_kd_parameters(
    target: tuple[int, int] = TARGET,
    projectile: tuple[int, int] = PROJECTILE,
    e_lab: float = E_LAB,
) -> KDParameters:
    rose = import_rose()
    mu, e_com, k, eta = rose.kinematics(
        target=target,
        projectile=projectile,
        E_lab=e_lab,
    )
    kd = rose.koning_delaroche.KDGlobal(rose.Projectile.neutron)
    r_c, alpha = kd.get_params(target[0], target[1], mu, e_lab, k)
    return KDParameters(
        mu=float(mu),
        e_com=float(e_com),
        k=float(k),
        eta=float(eta),
        r_c=float(r_c),
        alpha=np.asarray(alpha, dtype=float),
    )


def central_real_ws_parameters(
    target: tuple[int, int] = TARGET,
    projectile: tuple[int, int] = PROJECTILE,
    e_lab: float = E_LAB,
) -> RealWSParameters:
    params = central_kd_parameters(target=target, projectile=projectile, e_lab=e_lab)
    return RealWSParameters(
        mu=params.mu,
        e_com=params.e_com,
        k=params.k,
        eta=params.eta,
        r_c=params.r_c,
        alpha=params.alpha[:3].copy(),
    )


@njit
def real_woods_saxon_interaction(r: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    vv = alpha[0]
    rv = alpha[1]
    av = alpha[2]
    return -vv / (1.0 + np.exp((r - rv) / av))


def real_woods_saxon_potential(r: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    vv, rv, av = np.asarray(alpha, dtype=float)
    exponent = np.clip((r - rv) / av, -700.0, 700.0)
    return -vv / (1.0 + np.exp(exponent))


def make_real_ws_problem(
    params: RealWSParameters,
    train_alphas: np.ndarray,
    n_u: int,
    l_max: int,
    n_mesh: int,
    domain: tuple[float, float] = (1e-8, 8 * np.pi),
    rk_tols: tuple[float, float] = (1e-9, 1e-9),
    s_0_factor: float = 6 * np.pi,
) -> RealWSProblem:
    rose = import_rose()
    train_alphas = np.asarray(train_alphas, dtype=float)
    if train_alphas.ndim != 2 or train_alphas.shape[1] != params.alpha.size:
        raise ValueError("train_alphas must have shape (n_samples, 3)")
    rho_mesh = np.linspace(domain[0], domain[1], n_mesh)
    bounds = np.column_stack(
        [
            np.minimum(train_alphas.min(axis=0), params.alpha),
            np.maximum(train_alphas.max(axis=0), params.alpha),
        ]
    )
    interactions = rose.InteractionEIMSpace(
        l_max=l_max,
        coordinate_space_potential=real_woods_saxon_interaction,
        n_theta=params.alpha.size,
        mu=params.mu,
        energy=params.e_com,
        is_complex=False,
        training_info=bounds,
        n_basis=n_u,
        rho_mesh=rho_mesh,
    )
    base_solver = rose.SchroedingerEquation.make_base_solver(
        s_0=s_0_factor,
        rk_tols=list(rk_tols),
        domain=np.asarray(domain, dtype=float),
    )
    interaction = interactions.interactions[0][0]
    solver = base_solver.clone_for_new_interaction(interaction)
    return RealWSProblem(
        params=params,
        rho_mesh=rho_mesh,
        r_mesh=rho_mesh / params.k,
        interaction=interaction,
        solver=solver,
        base_solver=base_solver,
    )


def make_real_ws_custom_basis(
    problem: RealWSProblem,
    phi0: np.ndarray,
    wavefunctions: np.ndarray,
    n_basis: int,
) -> Any:
    rose = import_rose()
    return rose.basis.CustomBasis(
        solutions=np.asarray(wavefunctions, dtype=np.complex128).T.copy(),
        phi_0=np.asarray(phi0, dtype=np.complex128).copy(),
        rho_mesh=problem.rho_mesh,
        n_basis=n_basis,
        solver=problem.solver,
        subtract_phi0=True,
        use_svd=True,
        center=False,
        scale=False,
    )


def make_real_ws_rbe(problem: RealWSProblem, custom_basis: Any) -> Any:
    rose = import_rose()
    return rose.reduced_basis_emulator.ReducedBasisEmulator(
        problem.interaction,
        custom_basis,
        s_0=problem.base_solver.s_0,
        initialize_emulator=True,
    )


def make_alphas(alpha0: np.ndarray, param_index: int, values: np.ndarray) -> np.ndarray:
    alphas = np.tile(np.asarray(alpha0, dtype=float), (len(values), 1))
    alphas[:, param_index] = values
    return alphas
