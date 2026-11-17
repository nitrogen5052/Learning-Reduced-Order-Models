"""Small numerical checks for the independent scattering core."""

import unittest
import tempfile
import warnings
from pathlib import Path

import numpy as np

from lrom.diagnostics import pointwise_relative_error
from lrom.fom import (
    ScatteringSystem, regular_start_s, s_matrices_from_scaled_wavefunctions,
    s_matrix_from_scaled_values, solve_channel,
)
from lrom.observables import (
    assemble_elastic_cross_sections, assemble_varying_energy_cross_sections,
    differential_cross_section,
)
from lrom import (
    BasisConfig, HardRoutedScatteringLROM, InputSpace, MaxVol,
    ScatteringLROM, ScatteringProblem,
)
from lrom.data import TrainingData
from lrom.physics import kd_neutron_parameters, kd_neutron_potential
from lrom.reduced import LearnedROM


class FullOrderModelTests(unittest.TestCase):
    def test_free_particle_has_unit_s_matrix(self):
        system = ScatteringSystem(mesh_points=800)

        def zero_potential(radius, omega, spin_factor=0.0):
            return np.zeros_like(radius, dtype=complex)

        wavefunction, s_matrix = solve_channel(
            system, np.array([0.0]), ell=0, potential=zero_potential
        )
        reconstructed_s = s_matrix_from_scaled_values(
            system.s_mesh[-12:], wavefunction[-12:], ell=0
        )
        self.assertLess(abs(s_matrix - 1.0), 2.0e-6)
        self.assertLess(abs(reconstructed_s - s_matrix), 2.0e-6)

        batch_s = s_matrices_from_scaled_wavefunctions(
            system.s_mesh, wavefunction, ell=0,
            matching_s=system.s_max,
        )
        self.assertEqual(batch_s.shape, (1,))
        self.assertLess(abs(batch_s[0] - s_matrix), 2.0e-6)

    def test_higher_partial_waves_use_l_dependent_safe_starts(self):
        self.assertLess(regular_start_s(0), regular_start_s(4))
        self.assertLess(regular_start_s(4), regular_start_s(8))

    def test_free_s_matrices_give_zero_cross_section(self):
        angles = np.array([1.0, 45.0, 90.0, 179.0])
        s_plus = np.ones(4, dtype=complex)
        s_minus = np.ones(4, dtype=complex)
        direct = differential_cross_section(angles, 0.8, s_plus, s_minus)
        self.assertTrue(np.allclose(direct, 0.0))

        channels = {(0, 0.0): np.ones(2, dtype=complex)}
        for ell in range(1, 4):
            channels[(ell, float(ell))] = np.ones(2, dtype=complex)
            channels[(ell, float(-(ell + 1)))] = np.ones(2, dtype=complex)
        assembled = assemble_elastic_cross_sections(channels, angles, 0.8)
        self.assertEqual(assembled.shape, (2, len(angles)))
        self.assertTrue(np.allclose(assembled, 0.0))

    def test_varying_energy_batch_has_inverse_k_squared_scaling(self):
        angles = np.array([30.0, 90.0, 150.0])
        channels = {(0, 0.0): np.array([0.8 + 0.1j, 0.8 + 0.1j])}
        result = assemble_varying_energy_cross_sections(
            channels, angles, np.array([1.0, 2.0])
        )
        np.testing.assert_allclose(result[1], result[0] / 4.0)

    def test_pointwise_relative_error_uses_explicit_floor(self):
        prediction = np.array([2.0, 1.0e-4])
        reference = np.array([1.0, 0.0])
        error = pointwise_relative_error(
            prediction, reference, denominator_floor=1.0e-3
        )
        np.testing.assert_allclose(error, [1.0, 0.1])


class PublicApiTests(unittest.TestCase):
    def test_local_kd_map_matches_reference_neutron_values(self):
        parameters = kd_neutron_parameters(48, 21, 17.6775625)
        expected = np.array([
            45.3197959955, 4.3332872220, 0.6706623998,
            1.4756965268, 4.3332872220, 0.6706623998,
            6.5289602675, 4.6692632103, 0.5366512102,
            5.4300256835, 3.6610296110, 0.59,
            -0.0901399217, 3.6610296110, 0.59,
        ])
        np.testing.assert_allclose(parameters, expected, rtol=2.0e-7, atol=2.0e-7)
        potential = kd_neutron_potential(np.array([0.5]), parameters, 2.0)[0]
        self.assertAlmostEqual(potential.real, -45.3426737, places=3)
        self.assertAlmostEqual(potential.imag, -1.4790297, places=3)

    def test_problem_data_prediction_and_serialization(self):
        problem = ScatteringProblem(
            potential="real_woods_saxon", l_max=1,
            mesh_points=180, target_binding_energy=None,
        )
        center = problem.reference_sample
        space = InputSpace(continuous={
            "Vv": (0.9 * center["Vv"], 1.1 * center["Vv"]),
            "Rv": (0.95 * center["Rv"], 1.05 * center["Rv"]),
            "av": (0.95 * center["av"], 1.05 * center["av"]),
        }, fixed={"E": 14.1, "A": 40, "Z": 20})
        samples = space.sample(12, seed=5)
        space.validate(samples)
        data = problem.generate_training_data(samples)
        self.assertEqual(data.potentials[(0, 0.0)].shape, (12, 180))
        local = data.subset([2, 3, 4], reference_index=3)
        self.assertEqual(len(local.samples), 3)
        self.assertEqual(local.reference_sample, samples[3])
        np.testing.assert_allclose(
            local.reference_wavefunctions[(0, 0.0)],
            data.wavefunctions[(0, 0.0)][3],
        )

        emulator = ScatteringLROM(
            problem, BasisConfig(size=3), MaxVol(count=3)
        ).train(data)
        prediction = emulator.predict(samples[0])
        angles = np.array([30.0, 90.0])
        cross_section = prediction.cross_section(angles)
        self.assertEqual(cross_section.shape, (2,))
        batch_cross_sections = emulator.cross_sections(samples[:4], angles)
        scalar_cross_sections = np.asarray([
            emulator.cross_section(sample, angles) for sample in samples[:4]
        ])
        np.testing.assert_allclose(
            batch_cross_sections, scalar_cross_sections,
            rtol=2.0e-12, atol=1.0e-12,
        )
        deployment = HardRoutedScatteringLROM(
            [emulator, emulator, emulator], [5.0, 10.0, 20.0, 30.0], angles
        )
        routed = deployment.cross_sections(samples[:4])
        np.testing.assert_allclose(
            routed, batch_cross_sections, rtol=2.0e-12, atol=1.0e-12
        )
        np.testing.assert_allclose(
            deployment.cross_section(samples[0]), cross_section,
            rtol=2.0e-12, atol=1.0e-12,
        )
        plus, minus, wave_numbers = deployment.partial_wave_s_matrices(
            samples[:4]
        )
        from lrom.observables import ElasticCrossSectionKernel
        rebuilt = ElasticCrossSectionKernel.build(
            angles, 1.0, problem.l_max
        ).evaluate(plus, minus) / wave_numbers[:, None] ** 2
        np.testing.assert_allclose(
            rebuilt, routed, rtol=2.0e-12, atol=1.0e-12
        )
        # The packed online route must reproduce channel-by-channel assembly.
        slow_channels = {
            channel: np.asarray([prediction.s_matrix(channel)])
            for channel in problem.channels
        }
        slow_cross_section = assemble_elastic_cross_sections(
            slow_channels, angles, problem.system(samples[0]).wave_number
        )[0]
        np.testing.assert_allclose(cross_section, slow_cross_section, rtol=2.0e-13)
        full_wave = prediction.wavefunction((0, 0.0))
        self.assertEqual(full_wave.shape, (180,))
        self.assertEqual(prediction.coordinates((0, 0.0)).shape, (3,))
        self.assertTrue(np.isfinite(prediction.condition_number((0, 0.0))))
        channel_model = emulator.channel_models[(0, 0.0)]
        self.assertTrue(
            np.all(channel_model.learned.selected_coordinates >= 0.1)
        )
        value, derivative = channel_model.boundary_values(
            prediction.coordinates((0, 0.0))
        )
        from lrom.fom import s_matrix_from_scaled_boundary
        full_grid_s = s_matrix_from_scaled_boundary(
            channel_model.matching_s, value, derivative, 0
        )
        self.assertLess(abs(prediction.s_matrix((0, 0.0)) - full_grid_s), 1.0e-12)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.npz"
            data.save(path)
            restored = TrainingData.load(path)
            np.testing.assert_allclose(
                restored.wavefunctions[(0, 0.0)], data.wavefunctions[(0, 0.0)]
            )

    def test_energy_can_be_an_explicit_learned_control(self):
        problem = ScatteringProblem(
            potential="real_woods_saxon", l_max=0,
            mesh_points=140, target_binding_energy=None,
        )
        center = problem.reference_sample
        space = InputSpace(
            continuous={
                "Vv": (0.95 * center["Vv"], 1.05 * center["Vv"]),
                "Rv": (0.98 * center["Rv"], 1.02 * center["Rv"]),
                "av": (0.98 * center["av"], 1.02 * center["av"]),
                "E": (12.0, 16.0),
            }, fixed={"A": 40, "Z": 20},
        )
        samples = space.sample(14, seed=11)
        data = problem.generate_training_data(samples)
        emulator = ScatteringLROM(
            problem, BasisConfig(size=3), MaxVol(count=3),
            control_inputs=("E",),
        ).train(data)
        learned = emulator.channel_models[(0, 0.0)].learned
        self.assertEqual(len(learned.matrices), 4)
        self.assertEqual(emulator.predict(samples[0]).coordinates((0, 0.0)).shape, (3,))

    def test_variable_channel_bases_use_grouped_packed_route(self):
        problem = ScatteringProblem(
            potential="real_woods_saxon", l_max=4,
            mesh_points=180, target_binding_energy=None,
        )
        center = problem.reference_sample
        space = InputSpace(continuous={
            "Vv": (0.8 * center["Vv"], 1.2 * center["Vv"]),
            "Rv": (0.9 * center["Rv"], 1.1 * center["Rv"]),
            "av": (0.9 * center["av"], 1.1 * center["av"]),
        }, fixed={"E": 14.1, "A": 40, "Z": 20})
        samples = space.sample(50, seed=5)
        data = problem.generate_training_data(samples)
        emulator = ScatteringLROM(
            problem,
            BasisConfig(relative_svd_tolerance=1.0e-2, max_size=8),
            MaxVol(count=6),
        ).train(data)
        sizes = {
            model.learned.basis.vectors.shape[1]
            for model in emulator.channel_models.values()
        }
        self.assertGreater(len(sizes), 1)

        angles = np.array([30.0, 90.0])
        prediction = emulator.predict(samples[0])
        packed = prediction.cross_section(angles)
        channel_values = {
            channel: np.asarray([prediction.s_matrix(channel)])
            for channel in problem.channels
        }
        direct = assemble_elastic_cross_sections(
            channel_values, angles, problem.system(samples[0]).wave_number
        )[0]
        np.testing.assert_allclose(packed, direct, rtol=2.0e-13, atol=1.0e-13)

        emulator.repack("padded")
        padded = emulator.cross_section(samples[0], angles)
        np.testing.assert_allclose(padded, packed, rtol=2.0e-13, atol=1.0e-13)
        self.assertEqual(
            emulator._packed_online.matrices.shape[-1], max(sizes)
        )
        emulator.repack("grouped")
        np.testing.assert_allclose(
            emulator.cross_section(samples[0], angles), packed,
            rtol=2.0e-13, atol=1.0e-13,
        )

    def test_tolerance_basis_can_round_up_to_packing_buckets(self):
        snapshots = np.diag([10.0, 1.0, 0.1, 0.01])
        reference = np.zeros(4)
        exact = BasisConfig(relative_svd_tolerance=2.0e-2).choose_size(
            snapshots, reference
        )
        bucketed = BasisConfig(
            relative_svd_tolerance=2.0e-2,
            allowed_sizes=(2, 4, 8),
        ).choose_size(snapshots, reference)
        self.assertEqual(exact, 2)
        self.assertEqual(bucketed, 2)

        rounded = BasisConfig(
            relative_svd_tolerance=5.0e-3,
            allowed_sizes=(2, 4, 8),
        ).choose_size(snapshots, reference)
        self.assertEqual(rounded, 4)

    def test_relative_l2_rule_can_select_zero_basis_vectors(self):
        coordinate = np.linspace(0.0, 4.0, 80)
        reference = 1.0 + 0.2 * np.sin(coordinate)
        snapshots = np.asarray([
            reference * (1.0 + amplitude * np.sin(2.0 * coordinate))
            for amplitude in (-1.0e-5, 0.0, 1.0e-5)
        ])
        _, _, vh = np.linalg.svd(
            snapshots - reference[None, :], full_matrices=False
        )
        frozen = BasisConfig(
            relative_l2_tolerance=1.0e-4, max_size=3
        ).choose_size_from_reconstruction(
            snapshots, reference, coordinate, vh
        )
        active = BasisConfig(
            relative_l2_tolerance=1.0e-6, max_size=3
        ).choose_size_from_reconstruction(
            snapshots, reference, coordinate, vh
        )
        self.assertEqual(frozen, 0)
        self.assertEqual(active, 1)

    def test_zero_dimensional_channel_uses_reference_without_a_solve(self):
        problem = ScatteringProblem(
            potential="real_woods_saxon", l_max=0,
            mesh_points=140, target_binding_energy=None,
        )
        center = problem.reference_sample
        space = InputSpace(continuous={
            "Vv": (0.95 * center["Vv"], 1.05 * center["Vv"]),
            "Rv": (0.98 * center["Rv"], 1.02 * center["Rv"]),
            "av": (0.98 * center["av"], 1.02 * center["av"]),
        }, fixed={"E": 14.1, "A": 40, "Z": 20})
        samples = space.sample(20, seed=17)
        data = problem.generate_training_data(samples)
        channel = (0, 0.0)
        data.wavefunctions[channel][:] = data.reference_wavefunctions[channel]
        emulator = ScatteringLROM(
            problem,
            BasisConfig(relative_l2_tolerance=1e-8, max_size=4),
            MaxVol(count=3), packing_strategy="padded",
        ).train(data)
        self.assertEqual(
            emulator.channel_models[channel].learned.basis.vectors.shape[1], 0
        )
        cross_section = emulator.cross_section(samples[0], np.array([30., 90.]))
        self.assertTrue(np.all(np.isfinite(cross_section)))


class FitCapacityWarningTests(unittest.TestCase):
    @staticmethod
    def _fit(sample_count):
        """Fit a small synthetic model with 12 coefficients per row."""
        rng = np.random.default_rng(91)
        coordinate = np.linspace(0.1, 5.0, 30)
        reference_wave = np.zeros(30, dtype=complex)
        reference_potential = np.zeros(30, dtype=complex)
        return LearnedROM.fit(
            reference_wave,
            rng.normal(size=(sample_count, 30))
            + 1j * rng.normal(size=(sample_count, 30)),
            coordinate,
            reference_potential,
            rng.normal(size=(sample_count, 30))
            + 1j * rng.normal(size=(sample_count, 30)),
            basis_size=3,
            predictor_count=3,
        )

    def test_small_sampling_margin_uses_regular_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._fit(12)
        self.assertTrue(any(
            item.category is UserWarning and "small sampling margin" in str(item.message)
            for item in caught
        ))

    def test_underdetermined_fit_uses_runtime_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._fit(11)
        self.assertTrue(any(
            item.category is RuntimeWarning and "underdetermined" in str(item.message)
            for item in caught
        ))


if __name__ == "__main__":
    unittest.main()
