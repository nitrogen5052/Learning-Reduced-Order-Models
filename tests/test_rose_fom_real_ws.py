from __future__ import annotations

import numpy as np

from lrom_bench.rose_fom import (
    RealWSParameters,
    central_real_ws_parameters,
    make_alphas,
    real_woods_saxon_potential,
)


def test_real_woods_saxon_potential_is_attractive_and_depth_scaled() -> None:
    r = np.array([0.5, 1.0, 2.0])
    alpha = np.array([50.0, 1.2, 0.7])
    deeper = np.array([60.0, 1.2, 0.7])

    values = real_woods_saxon_potential(r, alpha)
    deeper_values = real_woods_saxon_potential(r, deeper)

    assert values.shape == r.shape
    assert np.all(values < 0.0)
    assert np.all(np.abs(deeper_values) > np.abs(values))


def test_central_real_ws_parameters_exposes_three_parameter_alpha() -> None:
    params = central_real_ws_parameters()

    assert isinstance(params, RealWSParameters)
    assert params.alpha.shape == (3,)
    assert np.all(np.isfinite(params.alpha))


def test_make_alphas_changes_only_requested_column() -> None:
    alpha0 = np.array([50.0, 1.2, 0.7])
    values = np.array([48.0, 50.0, 52.0])

    alphas = make_alphas(alpha0, param_index=0, values=values)

    assert np.allclose(alphas[:, 0], values)
    assert np.allclose(alphas[:, 1:], alpha0[1:])
