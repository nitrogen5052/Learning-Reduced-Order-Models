from __future__ import annotations

import numpy as np

from lrom_bench.numerics import least_squares_basis_coefficients, trapezoid_integral


def test_trapezoid_integral_matches_linear_function_area() -> None:
    x = np.array([0.0, 0.5, 1.0])
    y = 2.0 * x

    assert trapezoid_integral(y, x=x) == 1.0


def test_least_squares_basis_coefficients_recovers_known_coefficients() -> None:
    mesh = np.array([0.0, 1.0, 2.0])
    phi0 = np.array([1.0, 1.0, 1.0])
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    coeff_true = np.array([[2.0, -3.0], [0.5, 4.0]])
    wavefunctions = phi0[np.newaxis, :] + coeff_true @ vectors.T

    coeff = least_squares_basis_coefficients(vectors, phi0, wavefunctions, mesh)

    assert np.allclose(coeff, coeff_true)
