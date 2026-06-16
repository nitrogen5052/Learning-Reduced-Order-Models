from __future__ import annotations

import numpy as np

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
