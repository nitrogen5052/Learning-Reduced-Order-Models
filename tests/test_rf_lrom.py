from __future__ import annotations

import numpy as np
import pytest

from lrom_bench.metrics import relative_l2_rows
from lrom_bench.prediction import predict_coefficients, reconstruct_from_basis
from lrom_bench.rf_lrom import fit_central_lrom


def test_fit_central_lrom_recovers_linear_coefficients() -> None:
    predictors = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=float)
    coeff_targets = np.array([[-2.0], [0.0], [2.0], [4.0]], dtype=float)

    model = fit_central_lrom("linear", predictors, coeff_targets)
    pred = predict_coefficients(model, predictors)

    assert model.n_basis == 1
    assert model.n_predictors == 1
    assert model.residual_mse < 1e-24
    assert np.allclose(pred, coeff_targets)


def test_reconstruct_from_basis_and_relative_l2_rows() -> None:
    phi0 = np.array([1.0, 1.0, 1.0])
    vectors = np.array([[1.0], [0.0], [-1.0]])
    coeffs = np.array([[2.0], [3.0]])

    recon = reconstruct_from_basis(phi0, vectors, coeffs)

    assert np.allclose(recon[0], [3.0, 1.0, -1.0])
    assert np.allclose(relative_l2_rows(recon, recon), 0.0)


def test_fit_central_lrom_recovers_multi_predictor_matrix_blocks() -> None:
    known_matrices = np.array(
        [
            [[0.10, -0.04], [0.03, 0.05]],
            [[-0.02, 0.07], [0.06, -0.03]],
        ],
        dtype=float,
    )
    known_vectors = np.array([[0.30, -0.20], [-0.10, 0.40]], dtype=float)
    predictors = np.array(
        [
            [-1.2, 0.4],
            [-0.7, -1.1],
            [-0.2, 1.3],
            [0.3, -0.8],
            [0.8, 0.9],
            [1.1, -0.3],
            [1.6, 1.2],
        ],
        dtype=float,
    )

    identity = np.eye(2)
    coeff_targets = np.array(
        [
            np.linalg.solve(
                identity + np.einsum("k,kij->ij", p_row, known_matrices),
                np.einsum("k,kj->j", p_row, known_vectors),
            )
            for p_row in predictors
        ]
    )

    model = fit_central_lrom("multi", predictors, coeff_targets)
    pred = predict_coefficients(model, predictors)

    assert model.rank == model.n_complex_parameters
    assert model.residual_mse < 1e-24
    assert np.allclose(pred, coeff_targets)
    assert np.allclose(model.matrices, known_matrices)
    assert np.allclose(model.vectors, known_vectors)


def test_predict_coefficients_rejects_invalid_predictor_dimensions() -> None:
    predictors = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=float)
    coeff_targets = np.array([[-2.0], [0.0], [2.0], [4.0]], dtype=float)
    model = fit_central_lrom("linear", predictors, coeff_targets)

    with pytest.raises(ValueError, match="predictors must have shape"):
        predict_coefficients(model, np.array(1.0))

    with pytest.raises(ValueError, match="predictors must have shape"):
        predict_coefficients(model, np.ones((1, 1, 1)))


def test_reconstruct_from_basis_rejects_invalid_shapes() -> None:
    phi0 = np.array([1.0, 1.0, 1.0])
    vectors = np.array([[1.0], [0.0], [-1.0]])
    coeffs = np.array([[2.0], [3.0]])

    with pytest.raises(ValueError, match="phi0 must have shape"):
        reconstruct_from_basis(phi0[np.newaxis, :], vectors, coeffs)

    with pytest.raises(ValueError, match="vectors must have shape"):
        reconstruct_from_basis(phi0, vectors[:, 0], coeffs)

    with pytest.raises(ValueError, match="coeffs must have shape"):
        reconstruct_from_basis(phi0, vectors, coeffs[:, 0])

    with pytest.raises(ValueError, match="basis width"):
        reconstruct_from_basis(phi0, np.ones((3, 2)), coeffs)

    with pytest.raises(ValueError, match="mesh width"):
        reconstruct_from_basis(phi0, np.ones((4, 1)), coeffs)
