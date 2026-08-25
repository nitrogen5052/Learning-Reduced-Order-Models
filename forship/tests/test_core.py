"""Small numerical checks for the independent scattering core."""

import unittest

import numpy as np

from lrom.fom import ScatteringSystem, s_matrix_from_wavefunction, solve_channel


class FullOrderModelTests(unittest.TestCase):
    def test_free_particle_has_unit_s_matrix(self):
        system = ScatteringSystem(mesh_points=600, r_max=30.0)

        def zero_potential(radius, omega, spin_factor=0.0):
            return np.zeros_like(radius, dtype=complex)

        wavefunction, s_matrix = solve_channel(
            system, np.array([0.0]), ell=0, potential=zero_potential
        )
        reconstructed_s = s_matrix_from_wavefunction(
            system.radius, wavefunction, ell=0, wave_number=system.wave_number
        )
        self.assertLess(abs(s_matrix - 1.0), 2.0e-6)
        self.assertLess(abs(reconstructed_s - s_matrix), 2.0e-6)


if __name__ == "__main__":
    unittest.main()
