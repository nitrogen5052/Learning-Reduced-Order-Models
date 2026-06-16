from __future__ import annotations

import numpy as np

from lrom_bench.predictors import (
    PredictorPack,
    centered_parameter_predictors,
    centered_potential_predictors,
    greedy_maxvol_indices,
    make_potential_predictor_pack,
)


def test_centered_parameter_predictors() -> None:
    samples = np.array([[2.0, 5.0], [4.0, 9.0]])
    center = np.array([3.0, 7.0])
    scales = np.array([2.0, 4.0])

    got = centered_parameter_predictors(samples, center, scales)

    assert np.allclose(got, [[-0.5, -0.5], [0.5, 0.5]])


def test_greedy_maxvol_indices_selects_independent_rows() -> None:
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.1, 0.1],
        ]
    )

    got = greedy_maxvol_indices(basis)

    assert set(got.tolist()) == {0, 1}


def test_potential_predictor_pack_centers_and_scales_values() -> None:
    mesh = np.linspace(0.0, 1.0, 5)
    alphas = np.array([[1.0], [2.0], [3.0]])
    center = np.array([2.0])

    def potential(points: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return alpha[0] * (1.0 + points)

    pack = make_potential_predictor_pack(
        potential=potential,
        train_alphas=alphas,
        central_alpha=center,
        mesh=mesh,
        n_predictors=2,
        min_mesh_value=0.0,
    )
    features = centered_potential_predictors(potential, np.array([[2.0], [3.0]]), pack)

    assert isinstance(pack, PredictorPack)
    assert pack.s_points.shape == (2,)
    assert np.allclose(features[0], 0.0)
    assert np.all(np.isfinite(features[1]))
