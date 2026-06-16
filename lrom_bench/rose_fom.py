from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


TARGET = (40, 20)
PROJECTILE = (1, 0)
E_LAB = 14.1


def ensure_scipy_spherical_harmonic_compat() -> None:
    import scipy.special

    if hasattr(scipy.special, "sph_harm") or not hasattr(scipy.special, "sph_harm_y"):
        return

    def sph_harm(m: Any, n: Any, theta: Any, phi: Any) -> Any:
        return scipy.special.sph_harm_y(n, m, theta, phi)

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


def real_woods_saxon_potential(r: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    rose = import_rose()
    r = np.asarray(r, dtype=float)
    vv, rv, av = np.asarray(alpha, dtype=float)
    return -vv * rose.koning_delaroche.woods_saxon_safe(r, rv, av)


def make_alphas(alpha0: np.ndarray, param_index: int, values: np.ndarray) -> np.ndarray:
    alphas = np.tile(np.asarray(alpha0, dtype=float), (len(values), 1))
    alphas[:, param_index] = values
    return alphas
