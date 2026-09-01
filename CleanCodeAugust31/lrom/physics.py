"""Physics ingredients shared by the independent FOM and the LROM.

Lengths are in fm and energies/masses are in MeV (with c=1).  The parameter
ordering follows the reference 40Ca calculation and ROSE's ``KD_simple``
interaction. None of the functions in this file depends on ROSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

HBAR_C = 197.3269804  # MeV fm
AMU = 931.49410242  # MeV
MASS_NEUTRON = 1.008665 * AMU
MASS_PROTON = 1.007276 * AMU
MASS_PION_INVERSE_FM = np.sqrt(0.5)

FULL_PARAMETER_NAMES = (
    "Vv", "Wv", "Wd", "Vso", "Rv", "Rd", "Rso", "av", "ad", "aso"
)

KD_PARAMETER_NAMES = (
    "Vv", "Rv", "av", "Wv", "Rwv", "awv", "Wd", "Rd", "ad",
    "Vso", "Rso", "aso", "Wso", "Rwso", "awso",
)


@dataclass(frozen=True)
class PotentialModel:
    """Named optical-potential parameterization used by a scattering problem."""

    name: str
    parameter_names: tuple[str, ...]
    evaluator: Callable
    central_values: Callable
    global_parameter_map: Callable | None = None

    def evaluate(
        self, radius: np.ndarray, omega: np.ndarray, spin_orbit_factor: float = 0.0
    ) -> np.ndarray:
        """Evaluate the potential at the requested radii."""
        return self.evaluator(radius, omega, spin_orbit_factor)

    def reference_parameters(self, target_a: int = 40) -> np.ndarray:
        """Return the model's default parameter vector for a target mass."""
        return np.asarray(self.central_values(target_a), dtype=float)

    def parameters_for(self, target_a: int, target_z: int, lab_energy: float) -> np.ndarray:
        """Return parameters at one physical location, using a global map if present."""
        if self.global_parameter_map is None:
            return self.reference_parameters(target_a)
        return np.asarray(
            self.global_parameter_map(target_a, target_z, lab_energy), dtype=float
        )


def two_body_kinematics(
    target_a: int,
    target_z: int,
    projectile_a: int,
    projectile_z: int,
    lab_energy: float,
    *,
    target_binding_energy: float,
    projectile_binding_energy: float = 0.0,
) -> tuple[float, float, float]:
    """Return effective reduced mass, COM energy, and relativistic momentum.

    Nuclear rest masses are constructed from neutron/proton masses and supplied
    binding energies. The effective reduced mass follows the Ingemarsson
    fixed-energy prescription also used in ROSE. The LROM FOM remains
    independent: all inputs and formulas are evaluated locally.
    """
    target_n = target_a - target_z
    projectile_n = projectile_a - projectile_z
    target_mass = (
        target_z * MASS_PROTON + target_n * MASS_NEUTRON - target_binding_energy
    )
    projectile_mass = (
        projectile_z * MASS_PROTON
        + projectile_n * MASS_NEUTRON
        - projectile_binding_energy
    )
    com_energy = target_mass / (target_mass + projectile_mass) * lab_energy
    projectile_com_energy = com_energy + projectile_mass
    momentum = (
        target_mass
        * np.sqrt(lab_energy * (lab_energy + 2.0 * projectile_mass))
        / np.sqrt(
            (target_mass + projectile_mass) ** 2
            + 2.0 * target_mass * lab_energy
        )
        / HBAR_C
    )
    effective_mu = (
        momentum**2
        * projectile_com_energy
        / (projectile_com_energy**2 - projectile_mass**2)
        * HBAR_C**2
    )
    return float(effective_mu), float(com_energy), float(momentum)


def full_woods_saxon_parameters(target_a: int = 40) -> np.ndarray:
    """Return the central values used in the reference 40Ca calculation."""
    values = {
        "Vv": 46.7238,
        "Wv": 1.72334,
        "Wd": 7.2357,
        "Vso": 6.1,
        "Rv": 4.0538,
        "Rd": 4.4055,
        "Rso": 1.01 * target_a ** (1.0 / 3.0),
        "av": 0.6718,
        "ad": 0.5379,
        "aso": 0.60,
    }
    return np.array([values[name] for name in FULL_PARAMETER_NAMES])


def real_woods_saxon(radius: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Real volume Woods–Saxon potential for ``omega=(Vv, Rv, av)``."""
    vv, rv, av = np.asarray(omega, dtype=float)
    x = np.clip((np.asarray(radius) - rv) / av, -700.0, 700.0)
    return -vv / (1.0 + np.exp(x))


def _real_woods_saxon_adapter(
    radius: np.ndarray, omega: np.ndarray, spin_orbit_factor: float = 0.0
) -> np.ndarray:
    """Adapt the real Woods--Saxon to the common potential-model signature."""
    del spin_orbit_factor
    return real_woods_saxon(radius, omega)


def full_woods_saxon(
    radius: np.ndarray,
    omega: np.ndarray,
    spin_orbit_factor: float = 0.0,
) -> np.ndarray:
    """Complex volume, surface, and spin-orbit Woods–Saxon potential.

    ``spin_orbit_factor`` is ``2 l.s``: it is ``l`` for ``j=l+1/2`` and
    ``-(l+1)`` for ``j=l-1/2``.
    """
    vv, wv, wd, vso, rv, rd, rso, av, ad, aso = np.asarray(omega, dtype=float)
    r = np.asarray(radius, dtype=float)

    xv = np.clip((r - rv) / av, -700.0, 700.0)
    xd = np.clip((r - rd) / ad, -700.0, 700.0)
    f_volume = 1.0 / (1.0 + np.exp(xv))
    exp_d = np.exp(xd)
    df_surface = -(exp_d / ad) / (1.0 + exp_d) ** 2
    central = -vv * f_volume - 1j * wv * f_volume + 4j * ad * wd * df_surface

    xs = np.clip((r - rso) / aso, -700.0, 700.0)
    exp_s = np.exp(xs)
    df_spin = -(exp_s / aso) / (1.0 + exp_s) ** 2
    spin = (
        spin_orbit_factor * vso / MASS_PION_INVERSE_FM**2 * df_spin / r
    )
    return central + spin


def kd_neutron_parameters(target_a: int, target_z: int, lab_energy: float) -> np.ndarray:
    """Return the 15 Koning--Delaroche neutron parameters for ``(A,Z,E)``.

    This is the published 2003 global map used by ROSE's ``KDGlobal`` class,
    reproduced locally so the LROM FOM remains independent of ROSE.
    """
    a = float(target_a)
    z = float(target_z)
    delta = (a - 2.0 * z) / a
    energy = float(lab_energy)
    ef = -11.2814 + 0.02646 * a

    v1 = 59.30 - 21.0 * delta - 0.024 * a
    v2 = 0.007228 - 1.48e-6 * a
    v3 = 1.994e-5 - 2.0e-8 * a
    vv = v1 * (
        1.0 - v2 * (energy - ef) + v3 * (energy - ef) ** 2
        - 7.0e-9 * (energy - ef) ** 3
    )
    rv_reduced = 1.3039 - 0.4054 * a ** (-1.0 / 3.0)
    rv = rv_reduced * a ** (1.0 / 3.0)
    av = 0.6778 - 1.487e-4 * a

    w1 = 12.195 + 0.0167 * a
    w2 = 73.55 + 0.0795 * a
    wv = w1 * (energy - ef) ** 2 / ((energy - ef) ** 2 + w2**2)

    d1 = 16.0 - 16.0 * delta
    d2 = 0.0180 + 0.003802 / (1.0 + np.exp((a - 156.0) / 8.0))
    wd = (
        d1 * (energy - ef) ** 2 / ((energy - ef) ** 2 + 11.5**2)
        * np.exp(-d2 * (energy - ef))
    )
    rd_reduced = 1.3424 - 0.01585 * a ** (1.0 / 3.0)
    rd = rd_reduced * a ** (1.0 / 3.0)
    ad = 0.5446 - 1.656e-4 * a

    vso1 = 5.922 + 0.0030 * a
    vso = vso1 * np.exp(-0.0040 * (energy - ef))
    rso_reduced = 1.1854 - 0.647 * a ** (-1.0 / 3.0)
    rso = rso_reduced * a ** (1.0 / 3.0)
    aso = 0.59
    wso = -3.1 * (energy - ef) ** 2 / ((energy - ef) ** 2 + 160.0**2)

    return np.asarray([
        vv, rv, av, wv, rv, av, wd, rd, ad,
        vso, rso, aso, wso, rso, aso,
    ])


def kd_neutron_potential(
    radius: np.ndarray, omega: np.ndarray, spin_orbit_factor: float = 0.0
) -> np.ndarray:
    """Evaluate the 15-parameter Koning--Delaroche neutron potential."""
    (vv, rv, av, wv, rwv, awv, wd, rd, ad,
     vso, rso, aso, wso, rwso, awso) = np.asarray(omega, dtype=float)
    r = np.asarray(radius, dtype=float)

    def form(radial_values, radius_value, diffuseness):
        x = np.clip((radial_values - radius_value) / diffuseness, -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(x))

    def derivative(radial_values, radius_value, diffuseness):
        x = np.clip((radial_values - radius_value) / diffuseness, -700.0, 700.0)
        exponential = np.exp(x)
        return -(exponential / diffuseness) / (1.0 + exponential) ** 2

    central = (
        -vv * form(r, rv, av)
        - 1j * wv * form(r, rwv, awv)
        + 4j * ad * wd * derivative(r, rd, ad)
    )
    l_dot_s = 0.5 * spin_orbit_factor
    spin = l_dot_s / MASS_PION_INVERSE_FM**2 * (
        vso * derivative(r, rso, aso) / r
        + 1j * wso * derivative(r, rwso, awso) / r
    )
    return central + spin


def _real_woods_saxon_reference(target_a: int) -> np.ndarray:
    """Return the fixed three-parameter reference in a pickle-safe function."""
    del target_a
    return np.array([46.7238, 4.0538, 0.6718])


def _kd_neutron_reference(target_a: int) -> np.ndarray:
    """Return the conventional stable-line KD reference for one mass."""
    return kd_neutron_parameters(target_a, target_a // 2, 14.1)


def potential_model(name: str) -> PotentialModel:
    """Return a built-in potential model by its short, user-facing name."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key in {"full", "full_optical", "full_woods_saxon"}:
        return PotentialModel(
            "full_woods_saxon", FULL_PARAMETER_NAMES, full_woods_saxon,
            full_woods_saxon_parameters,
        )
    if key in {"real", "real_woods_saxon"}:
        return PotentialModel(
            "real_woods_saxon", ("Vv", "Rv", "av"),
            _real_woods_saxon_adapter,
            _real_woods_saxon_reference,
        )
    if key in {"kd", "kd_neutron", "koning_delaroche"}:
        return PotentialModel(
            "kd_neutron", KD_PARAMETER_NAMES, kd_neutron_potential,
            _kd_neutron_reference,
            kd_neutron_parameters,
        )
    raise ValueError(f"unknown potential model: {name!r}")
