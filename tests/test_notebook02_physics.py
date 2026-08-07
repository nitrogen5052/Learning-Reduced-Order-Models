"""Physics regressions for the full optical potential used by Notebook 02."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = REPOSITORY_ROOT / "notebooks"
for path in (REPOSITORY_ROOT, NOTEBOOK_DIRECTORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark_helper
import lrom_legacy.v2_0 as lrom_v2


FULL_PARAMETERS = np.array(
    [46.7238, 1.72334, 7.2357, 6.1, 4.0538, 4.4055, 4.112, 0.6718, 0.5379, 0.60]
)


def expected_spin_orbit(radius_fm: float, two_l_dot_s: float) -> float:
    """Evaluate the ROSE-paper form using (hbar / m_pi c)^2 = 2 fm^2."""
    vso = FULL_PARAMETERS[3]
    rso = FULL_PARAMETERS[6]
    aso = FULL_PARAMETERS[9]
    exponential = np.exp((radius_fm - rso) / aso)
    derivative = -(exponential / aso) / (1.0 + exponential) ** 2
    return vso * 2.0 * two_l_dot_s * derivative / radius_fm


def test_parked_v2_spin_orbit_uses_inverse_fm_pion_scale() -> None:
    radius_fm = 4.0
    expected = expected_spin_orbit(radius_fm, two_l_dot_s=2.0)

    form_factor = lrom_v2.full_woods_saxon_spin_orbit(
        np.array([radius_fm]),
        FULL_PARAMETERS,
    )[0]

    assert 2.0 * form_factor == pytest.approx(expected, rel=1e-12)


def test_notebook_helper_spin_orbit_uses_inverse_fm_pion_scale() -> None:
    radius_fm = 4.0
    expected = expected_spin_orbit(radius_fm, two_l_dot_s=-3.0)

    actual = benchmark_helper.full_woods_saxon_spin_orbit(
        radius_fm,
        FULL_PARAMETERS,
        -3.0,
    )

    assert actual.real == pytest.approx(expected, rel=1e-12)
    assert actual.imag == 0.0


def test_centrifugal_barrier_is_reported_in_mev_on_physical_radius() -> None:
    radius_fm = np.array([4.0])
    reduced_mass_mev = 929.3429910745323
    hbar_c_mev_fm = 197.3269804
    expected = (
        hbar_c_mev_fm**2
        * 2.0
        * 3.0
        / (2.0 * reduced_mass_mev * radius_fm**2)
    )

    actual = benchmark_helper.centrifugal_barrier_mev(
        radius_fm,
        ell=2,
        reduced_mass_mev=reduced_mass_mev,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    ("radius_fm", "ell", "reduced_mass_mev"),
    [([0.0], 2, 929.0), ([4.0], -1, 929.0), ([4.0], 2, 0.0)],
)
def test_centrifugal_barrier_rejects_nonphysical_inputs(
    radius_fm: list[float],
    ell: int,
    reduced_mass_mev: float,
) -> None:
    with pytest.raises(ValueError):
        benchmark_helper.centrifugal_barrier_mev(
            np.asarray(radius_fm),
            ell=ell,
            reduced_mass_mev=reduced_mass_mev,
        )
