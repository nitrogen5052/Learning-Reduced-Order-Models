"""Elastic-scattering observables assembled from partial-wave S matrices."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.special import eval_legendre, lpmv


Channel = tuple[int, float]


@dataclass(frozen=True)
class ElasticCrossSectionKernel:
    """Cached angular factors for fast repeated elastic cross sections.

    Constructing Legendre functions is relatively expensive compared with
    solving the small reduced systems.  This object evaluates them once for a
    chosen angular grid and then accepts complete arrays of partial-wave
    S matrices, including batches of parameter selections.
    """

    angles_degrees: np.ndarray
    wave_number: float
    legendre: np.ndarray
    associated_legendre: np.ndarray

    @classmethod
    def build(
        cls, angles_degrees: np.ndarray, wave_number: float, l_max: int
    ) -> "ElasticCrossSectionKernel":
        """Precompute angular functions through ``l_max``."""
        angles = np.asarray(angles_degrees, dtype=float).reshape(-1)
        if wave_number <= 0:
            raise ValueError("wave_number must be positive")
        x = np.cos(np.deg2rad(angles))
        ells = np.arange(l_max + 1)
        legendre = np.asarray([eval_legendre(ell, x) for ell in ells])
        associated = np.zeros_like(legendre)
        if l_max >= 1:
            associated[1:] = np.asarray([
                lpmv(1, ell, x) for ell in ells[1:]
            ])
        return cls(angles.copy(), float(wave_number), legendre, associated)

    def evaluate(self, s_plus: np.ndarray, s_minus: np.ndarray) -> np.ndarray:
        """Return cross sections for arrays shaped ``(..., l_max + 1)``."""
        plus = np.asarray(s_plus, dtype=complex)
        minus = np.asarray(s_minus, dtype=complex)
        if plus.shape != minus.shape or plus.shape[-1] != len(self.legendre):
            raise ValueError("S-matrix arrays do not match the cached partial waves")
        ells = np.arange(plus.shape[-1], dtype=float)
        coefficient_a = (
            (ells + 1.0) * (plus - 1.0) + ells * (minus - 1.0)
        )
        coefficient_b = plus - minus
        coefficient_b[..., 0] = 0.0
        amplitude_a = coefficient_a @ self.legendre / (2j * self.wave_number)
        amplitude_b = (
            coefficient_b @ self.associated_legendre / (2j * self.wave_number)
        )
        return 10.0 * (np.abs(amplitude_a) ** 2 + np.abs(amplitude_b) ** 2)


def differential_cross_section(
    angles_degrees: np.ndarray,
    wave_number: float,
    s_plus: np.ndarray,
    s_minus: np.ndarray,
) -> np.ndarray:
    """Calculate one spin-1/2-on-spin-0 elastic differential cross section.

    Parameters
    ----------
    angles_degrees
        Center-of-mass scattering angles in degrees.
    wave_number
        Center-of-mass wave number in fm^-1.
    s_plus, s_minus
        Partial-wave S matrices for j=ell+1/2 and j=ell-1/2. Both arrays
        contain ell=0, where the two entries must be identical.

    Returns
    -------
    np.ndarray
        Differential cross section at each angle, in mb/sr.
    """
    s_plus = np.asarray(s_plus, dtype=complex)
    s_minus = np.asarray(s_minus, dtype=complex)
    if s_plus.ndim != 1 or s_minus.ndim != 1 or s_plus.shape != s_minus.shape:
        raise ValueError("s_plus and s_minus must be equal-length one-dimensional arrays")
    if wave_number <= 0:
        raise ValueError("wave_number must be positive")

    minus = s_minus.copy()
    minus[0] = s_plus[0]
    kernel = ElasticCrossSectionKernel.build(
        angles_degrees, wave_number, len(s_plus) - 1
    )
    return kernel.evaluate(s_plus, minus)


def assemble_elastic_cross_sections(
    s_by_channel: Mapping[Channel, np.ndarray],
    angles_degrees: np.ndarray,
    wave_number: float,
) -> np.ndarray:
    """Assemble cross sections for several cases from channel S matrices.

    Parameters
    ----------
    s_by_channel
        Mapping ``(ell, spin_factor) -> S``. Each value is a one-dimensional
        array with one S matrix per parameter case. The spin factors are 0 for
        ell=0, ell for j=ell+1/2, and ``-(ell+1)`` for j=ell-1/2.
    angles_degrees
        Center-of-mass scattering angles in degrees.
    wave_number
        Center-of-mass wave number in fm^-1.

    Returns
    -------
    np.ndarray
        Array with shape ``(number_of_cases, number_of_angles)`` in mb/sr.
    """
    if not s_by_channel:
        raise ValueError("s_by_channel cannot be empty")
    if (0, 0.0) not in s_by_channel:
        raise ValueError("s_by_channel must contain the ell=0 channel (0, 0.0)")

    arrays = {key: np.atleast_1d(np.asarray(value, dtype=complex))
              for key, value in s_by_channel.items()}
    case_count = len(arrays[(0, 0.0)])
    if case_count == 0 or any(len(value) != case_count for value in arrays.values()):
        raise ValueError("all channel arrays must contain the same positive case count")

    l_max = max(ell for ell, _ in arrays)
    required = {(0, 0.0)}
    for ell in range(1, l_max + 1):
        required.update({(ell, float(ell)), (ell, float(-(ell + 1)))})
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"missing partial-wave channels: {sorted(missing)}")

    s_plus = np.empty((case_count, l_max + 1), dtype=complex)
    s_minus = np.empty_like(s_plus)
    s_plus[:, 0] = s_minus[:, 0] = arrays[(0, 0.0)]
    for ell in range(1, l_max + 1):
        s_plus[:, ell] = arrays[(ell, float(ell))]
        s_minus[:, ell] = arrays[(ell, float(-(ell + 1)))]
    kernel = ElasticCrossSectionKernel.build(angles_degrees, wave_number, l_max)
    return kernel.evaluate(s_plus, s_minus)


def assemble_varying_energy_cross_sections(
    s_by_channel: Mapping[Channel, np.ndarray],
    angles_degrees: np.ndarray,
    wave_numbers: np.ndarray,
) -> np.ndarray:
    """Assemble cases having different center-of-mass wave numbers.

    The angular partial-wave sums are shared by the batch.  Only the final
    ``1/k`` amplitude factor varies, so the function avoids rebuilding one
    angular kernel for every energy.

    Parameters
    ----------
    s_by_channel
        Complete channel mapping with one S matrix per case.
    angles_degrees
        Common center-of-mass angle grid.
    wave_numbers
        One positive wave number in fm^-1 per case.

    Returns
    -------
    np.ndarray
        Differential cross sections shaped ``(cases, angles)`` in mb/sr.
    """
    wave_numbers = np.asarray(wave_numbers, dtype=float).reshape(-1)
    if wave_numbers.size == 0 or np.any(wave_numbers <= 0.0):
        raise ValueError("wave_numbers must contain positive values")
    unit_k = assemble_elastic_cross_sections(
        s_by_channel, angles_degrees, wave_number=1.0
    )
    if len(unit_k) != len(wave_numbers):
        raise ValueError("wave_numbers must contain one value per case")
    return unit_k / wave_numbers[:, None] ** 2
