from types import SimpleNamespace

import numpy as np
import pytest

import lrom_legacy.v2_0 as v2


class FakeInteraction:
    def __init__(self, channel_scale):
        self.channel_scale = channel_scale

    def tilde(self, rho, alpha):
        rho = np.asarray(rho)
        a0, a1 = np.asarray(alpha)
        return self.channel_scale * (
            a0 * np.exp(-rho) + 1j * a1 * rho * np.exp(-0.5 * rho)
        )


def test_refined_maxvol_is_unique_and_deterministic():
    basis = np.array(
        [
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.1],
            [0.1, 0.1, 1.0],
            [0.9, 0.8, 0.0],
            [0.2j, 0.1, 0.8],
        ],
        dtype=np.complex128,
    )

    selected_a = v2._refined_maxvol_indices(basis)
    selected_b = v2._refined_maxvol_indices(basis)

    assert np.array_equal(selected_a, selected_b)
    assert selected_a.shape == (basis.shape[1],)
    assert np.unique(selected_a).size == selected_a.size
    assert np.linalg.matrix_rank(basis[selected_a]) == basis.shape[1]


def test_refined_maxvol_rejects_invalid_shape():
    with pytest.raises(ValueError, match="points, modes"):
        v2._refined_maxvol_indices(np.ones((2, 3)))


def test_effective_interaction_predictors_are_channel_specific_and_physical():
    rho = np.linspace(0.01, 8.0, 96)
    radius = rho / 1.4
    models = {
        0: SimpleNamespace(interaction=FakeInteraction(1.0)),
        (1, 0): SimpleNamespace(interaction=FakeInteraction(2.0)),
        (1, 1): SimpleNamespace(interaction=FakeInteraction(-3.0)),
    }
    center = np.array([1.0, 2.0])
    train = np.array(
        [[0.8, 1.7], [1.2, 2.3], [1.1, 1.8], [0.9, 2.2]]
    )
    test = np.array([[1.05, 2.1]])

    states = v2.build_effective_interaction_predictors(
        full_order_models=models,
        rho=rho,
        radius=radius,
        central_values=center,
        training_values=train,
        testing_values=test,
        predictor_count=2,
    )

    assert tuple(states) == (0, (1, 0), (1, 1))
    assert all(
        np.all(state.selected_radii >= 0.5) for state in states.values()
    )
    assert not np.array_equal(
        states[(1, 0)].training_features,
        states[(1, 1)].training_features,
    )
    assert all(
        np.max(np.abs(state.training_features), axis=0).max()
        <= 1.0 + 1e-12
        for state in states.values()
    )


def test_intercept_fit_recovers_held_out_coordinates():
    rng = np.random.default_rng(18)
    features = rng.uniform(-0.8, 0.8, size=(80, 2))
    matrices = np.array(
        [
            [[0.05, 0.01], [-0.02, 0.03]],
            [[-0.01, 0.02], [0.01, 0.04]],
        ],
        dtype=np.complex128,
    )
    vectors = np.array(
        [[0.2, -0.1], [0.04, 0.06]], dtype=np.complex128
    )
    constant = np.array([0.35 + 0.1j, -0.2 + 0.05j])

    def exact(rows):
        result = []
        for row in rows:
            matrix = np.eye(2) + np.einsum("k,kij->ij", row, matrices)
            rhs = constant + np.einsum("k,kj->j", row, vectors)
            result.append(np.linalg.solve(matrix, rhs))
        return np.asarray(result)

    model = v2.fit_rf_lrom(
        predictors=features,
        coordinates=exact(features),
        include_intercept=True,
    )
    held_out = rng.uniform(-0.7, 0.7, size=(12, 2))

    assert np.allclose(
        v2.solve_rf_lrom(model=model, predictors=held_out),
        exact(held_out),
        rtol=1e-9,
        atol=1e-10,
    )
    assert np.linalg.norm(model.constant_vector) > 0.0


def test_zero_intercept_fit_preserves_zero_constant_vector():
    predictors = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    coordinates = np.column_stack(
        [0.2 * predictors[:, 0], -0.1 * predictors[:, 0]]
    )

    model = v2.fit_rf_lrom(
        predictors=predictors,
        coordinates=coordinates,
    )

    assert np.array_equal(model.constant_vector, np.zeros(2))
