from __future__ import annotations

import numpy as np

from lrom_bench.rose_fom import (
    RealWSParameters,
    central_real_ws_parameters,
    make_real_ws_custom_basis,
    make_real_ws_problem,
    make_real_ws_rbe,
    make_alphas,
    real_woods_saxon_interaction,
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


def test_real_woods_saxon_interaction_matches_numpy_potential() -> None:
    r = np.array([0.5, 1.0, 2.0])
    alpha = np.array([50.0, 1.2, 0.7])

    assert np.allclose(real_woods_saxon_interaction(r, alpha), real_woods_saxon_potential(r, alpha))


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


def test_make_real_ws_problem_solves_wavefunctions_and_builds_rbe() -> None:
    params = central_real_ws_parameters()
    values = np.linspace(params.alpha[0] - 1.0, params.alpha[0] + 1.0, 5)
    train_alphas = make_alphas(params.alpha, param_index=0, values=values)

    problem = make_real_ws_problem(
        params=params,
        train_alphas=train_alphas,
        n_u=3,
        l_max=0,
        n_mesh=80,
        rk_tols=(1e-8, 1e-8),
    )
    phi0 = problem.solve_phi(params.alpha)
    wavefunctions = problem.solve_wavefunctions(train_alphas)
    custom_basis = make_real_ws_custom_basis(
        problem=problem,
        phi0=phi0,
        wavefunctions=wavefunctions,
        n_basis=2,
    )
    rbe = make_real_ws_rbe(problem=problem, custom_basis=custom_basis)
    coeffs = np.array([rbe.coefficients(alpha) for alpha in train_alphas])
    rbm_wavefunctions = np.array([rbe.emulate_wave_function(alpha) for alpha in train_alphas])

    assert problem.rho_mesh.shape == (80,)
    assert problem.r_mesh.shape == (80,)
    assert wavefunctions.shape == (5, 80)
    assert custom_basis.vectors.shape == (80, 2)
    assert coeffs.shape == (5, 2)
    assert rbm_wavefunctions.shape == (5, 80)
    assert np.all(np.isfinite(wavefunctions))
