"""Physics ingredients shared by the independent FOM and the LROM.

Lengths are in fm and energies/masses are in MeV (with c=1).  The parameter
ordering follows the student's cross-section notebook and ROSE's ``KD_simple``
interaction.  None of the functions in this file depends on ROSE.
"""

from __future__ import annotations

import numpy as np

HBAR_C = 197.3269804  # MeV fm
AMU = 931.49410242  # MeV
MASS_NEUTRON = 1.008665 * AMU
MASS_PROTON = 1.007276 * AMU
MASS_PION_INVERSE_FM = np.sqrt(0.5)

FULL_PARAMETER_NAMES = (
    "Vv", "Wv", "Wd", "Vso", "Rv", "Rd", "Rso", "av", "ad", "aso"
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
    """Return the central values used in the student's 40Ca calculations."""
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
