from __future__ import annotations

import scipy.special


def test_rose_fom_module_imports_without_importing_rose_immediately() -> None:
    import lrom_bench.rose_fom as rose_fom

    assert rose_fom.TARGET == (40, 20)
    assert rose_fom.PROJECTILE == (1, 0)


def test_scipy_spherical_harmonic_compatibility_alias_is_installed() -> None:
    from lrom_bench.rose_fom import ensure_scipy_spherical_harmonic_compat

    if hasattr(scipy.special, "sph_harm"):
        delattr(scipy.special, "sph_harm")

    ensure_scipy_spherical_harmonic_compat()

    assert scipy.special.sph_harm(0, 0, 0.0, 0.0) == scipy.special.sph_harm_y(
        0,
        0,
        0.0,
        0.0,
    )
