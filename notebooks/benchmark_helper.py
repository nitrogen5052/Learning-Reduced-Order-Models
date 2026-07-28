from collections.abc import Mapping
from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from numba import njit
import scipy.special

if (
    not hasattr(scipy.special, "sph_harm")
    and hasattr(scipy.special, "sph_harm_y")
):
    scipy.special.sph_harm = (
        lambda m, n, theta, phi: scipy.special.sph_harm_y(
            n,
            m,
            phi,
            theta,
        )
    )

import rose

import lrom as lrom_v1
import lrom_legacy.v2_0 as lrom_v2


@dataclass(frozen=True)
class RoseWavefunctionResult:
    emulator: Any
    basis: Any
    interaction: Any
    training_rows: np.ndarray
    testing_rows: np.ndarray
    training_coefficients: np.ndarray
    testing_coefficients: np.ndarray
    training_wavefunctions: np.ndarray | None
    testing_wavefunctions: np.ndarray


@dataclass(frozen=True)
class LSWavefunctionResult:
    training_coefficients: np.ndarray
    testing_coefficients: np.ndarray
    training_wavefunctions: np.ndarray
    testing_wavefunctions: np.ndarray


@dataclass(frozen=True)
class WavefunctionBenchmarkResult:
    rose: RoseWavefunctionResult
    ls: LSWavefunctionResult


@dataclass(frozen=True)
class RoseCrossSectionResult:
    emulators: dict[tuple[int, int], Any]
    fom_training_cross_sections: np.ndarray
    fom_testing_cross_sections: np.ndarray
    results: dict[tuple[int, int], dict[str, np.ndarray]]


@dataclass(frozen=True)
class LSCrossSectionResult:
    results: dict[int, dict[str, np.ndarray]]


@dataclass(frozen=True)
class CrossSectionBenchmarkResult:
    rose: RoseCrossSectionResult
    ls: LSCrossSectionResult


@njit
def rose_real_woods_saxon(
    radius: float,
    alpha: np.ndarray,
) -> complex:
    vv, rv, av = alpha
    return -vv / (1.0 + np.exp((radius - rv) / av))


@njit
def full_woods_saxon(
    radius: float,
    alpha: np.ndarray,
) -> complex:
    vv, wv, wd, _vso, rv, rd, _rso, av, ad, _aso = alpha
    volume = 1.0 / (1.0 + np.exp((radius - rv) / av))
    exponential = np.exp((radius - rd) / ad)
    derivative = -(exponential / ad) / (1.0 + exponential) ** 2
    return -vv * volume - 1j * wv * volume + 4j * ad * wd * derivative


@njit
def full_woods_saxon_spin_orbit(
    radius: float,
    alpha: np.ndarray,
    ldots: float,
) -> complex:
    _vv, _wv, _wd, vso, _rv, _rd, rso, _av, _ad, aso = alpha
    exponential = np.exp((radius - rso) / aso)
    derivative = -(exponential / aso) / (1.0 + exponential) ** 2
    return vso / 139.57039**2 * ldots * derivative / radius


def pointwise_relative_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    denominator_floor: float,
) -> np.ndarray:
    denominator = np.maximum(np.abs(reference), denominator_floor)
    return np.abs(np.asarray(predicted) - np.asarray(reference)) / denominator


def summarize_relative_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    denominator_floor: float,
) -> dict[str, np.ndarray]:
    pointwise = pointwise_relative_error(
        predicted,
        reference,
        denominator_floor,
    )
    return {
        "median_over_angle_error": np.median(pointwise, axis=1),
        "maximum_over_angle_error": np.max(pointwise, axis=1),
    }


class WavefunctionBenchmark:
    def __init__(
        self,
        emulator: Any,
        center: Mapping[str, float],
    ) -> None:
        self.emulator = emulator
        self.center = center

    def _interaction_space(self, eim_basis_size: int) -> Any:
        samples = self.emulator.samples
        training_rows = samples.design.training.values
        testing_rows = samples.design.testing.values
        center = [self.center[name] for name in self.emulator.parameter_names]
        rows = np.vstack([center, training_rows, testing_rows])
        bounds = np.column_stack([rows.min(axis=0), rows.max(axis=0)])
        return rose.InteractionEIMSpace(
            l_max=0,
            coordinate_space_potential=rose_real_woods_saxon,
            n_theta=len(self.emulator.parameter_names),
            mu=self.emulator.kinematics.mu,
            energy=self.emulator.kinematics.e_com,
            is_complex=False,
            training_info=bounds,
            n_basis=eim_basis_size,
            rho_mesh=samples.mesh.rho,
        ).interactions[0][0]

    def _free_reference(self) -> np.ndarray:
        rho = self.emulator.samples.mesh.rho
        return np.asarray(
            [
                rose.free_solutions.phi_free(
                    float(value),
                    0,
                    self.emulator.kinematics.eta,
                )
                for value in rho
            ],
            dtype=np.complex128,
        )

    def _custom_basis(self, basis_size: int) -> Any:
        samples = self.emulator.samples
        return rose.basis.CustomBasis(
            solutions=np.asarray(
                samples.training_wavefunctions[0],
                dtype=np.complex128,
            ).T.copy(),
            phi_0=self._free_reference(),
            rho_mesh=samples.mesh.rho,
            n_basis=basis_size,
            solver=self.emulator.full_order_model[0].solver,
            subtract_phi0=True,
            use_svd=True,
            center=False,
            scale=False,
        )

    def _rose_emulator(self, interaction: Any, basis: Any) -> Any:
        return rose.reduced_basis_emulator.ReducedBasisEmulator(
            interaction,
            basis,
            s_0=self.emulator.full_order_model[0].base_solver.s_0,
            initialize_emulator=True,
        )

    def run_rose(
        self,
        basis_size: int,
        eim_basis_size: int = 8,
        include_training_wavefunctions: bool = True,
    ) -> RoseWavefunctionResult:
        samples = self.emulator.samples
        training_rows = samples.design.training.values
        testing_rows = samples.design.testing.values
        interaction = self._interaction_space(eim_basis_size)
        basis = self._custom_basis(basis_size)
        emulator = self._rose_emulator(interaction, basis)
        training_coefficients = np.asarray(
            [emulator.coefficients(row) for row in training_rows]
        )
        testing_coefficients = np.asarray(
            [emulator.coefficients(row) for row in testing_rows]
        )
        training_wavefunctions = (
            np.asarray(
                [
                    emulator.emulate_wave_function(row)
                    for row in training_rows
                ]
            )
            if include_training_wavefunctions
            else None
        )
        testing_wavefunctions = np.asarray(
            [
                emulator.emulate_wave_function(row)
                for row in testing_rows
            ]
        )
        return RoseWavefunctionResult(
            emulator=emulator,
            basis=basis,
            interaction=interaction,
            training_rows=training_rows,
            testing_rows=testing_rows,
            training_coefficients=training_coefficients,
            testing_coefficients=testing_coefficients,
            training_wavefunctions=training_wavefunctions,
            testing_wavefunctions=testing_wavefunctions,
        )

    def run_ls(self) -> LSWavefunctionResult:
        samples = self.emulator.samples
        training_coefficients, training_wavefunctions = (
            lrom_v1.least_squares_baseline(
                basis=self.emulator.basis[0],
                wavefunctions=samples.training_wavefunctions[0],
            )
        )
        testing_coefficients, testing_wavefunctions = (
            lrom_v1.least_squares_baseline(
                basis=self.emulator.basis[0],
                wavefunctions=samples.testing_wavefunctions[0],
            )
        )
        return LSWavefunctionResult(
            training_coefficients=training_coefficients,
            testing_coefficients=testing_coefficients,
            training_wavefunctions=training_wavefunctions,
            testing_wavefunctions=testing_wavefunctions,
        )

    def run(
        self,
        basis_size: int,
        eim_basis_size: int = 8,
        include_training_wavefunctions: bool = True,
    ) -> WavefunctionBenchmarkResult:
        rose_result = self.run_rose(
            basis_size=basis_size,
            eim_basis_size=eim_basis_size,
            include_training_wavefunctions=include_training_wavefunctions,
        )
        return WavefunctionBenchmarkResult(
            rose=rose_result,
            ls=self.run_ls(),
        )


class CrossSectionBenchmark:
    def __init__(
        self,
        emulator: Any,
        training_rows: np.ndarray,
        testing_rows: np.ndarray,
        angles_degrees: np.ndarray,
        denominator_floor: float,
        timing_repeats: int,
        timing_inner_loops: int,
    ) -> None:
        self.emulator = emulator
        self.training_rows = np.asarray(training_rows, dtype=float)
        self.testing_rows = np.asarray(testing_rows, dtype=float)
        self.angles_degrees = np.asarray(angles_degrees, dtype=float)
        self.angles_radians = np.deg2rad(self.angles_degrees)
        self.denominator_floor = denominator_floor
        self.timing_repeats = timing_repeats
        self.timing_inner_loops = timing_inner_loops

    @property
    def _rho_mesh(self) -> np.ndarray:
        return self.emulator.samples.mesh.rho

    @property
    def _l_max(self) -> int:
        return max(self.emulator.partial_waves)

    def _interaction_space(self, eim_size: int) -> Any:
        return rose.InteractionEIMSpace(
            l_max=self._l_max,
            coordinate_space_potential=full_woods_saxon,
            spin_orbit_term=full_woods_saxon_spin_orbit,
            n_theta=len(self.emulator.parameter_names),
            mu=self.emulator.kinematics.mu,
            energy=self.emulator.kinematics.e_com,
            is_complex=True,
            training_info=self.training_rows,
            explicit_training=True,
            n_basis=eim_size,
            rho_mesh=self._rho_mesh,
        )

    def _base_solver(self) -> Any:
        return rose.SchroedingerEquation.make_base_solver(
            s_0=6 * np.pi,
            rk_tols=[1e-9, 1e-9],
            domain=np.array([self._rho_mesh[0], self._rho_mesh[-1]]),
        )

    def _free_reference(self, ell: int) -> np.ndarray:
        return np.asarray(
            [
                rose.free_solutions.phi_free(
                    float(value),
                    ell,
                    self.emulator.kinematics.eta,
                )
                for value in self._rho_mesh
            ],
            dtype=np.complex128,
        )

    def _custom_bases(
        self,
        interaction: Any,
        basis_size: int,
    ) -> list[list[Any]]:
        bases = []
        for ell, interaction_row in enumerate(interaction.interactions):
            row_bases = []
            for spin_index, _ in enumerate(interaction_row):
                key = (
                    ell
                    if len(interaction_row) == 1
                    else (ell, spin_index)
                )
                model = self.emulator.samples.full_order_models[key]
                row_bases.append(
                    rose.basis.CustomBasis(
                        solutions=np.asarray(
                            self.emulator.samples.training_wavefunctions[key],
                            dtype=np.complex128,
                        ).T.copy(),
                        phi_0=self._free_reference(ell),
                        rho_mesh=self._rho_mesh,
                        n_basis=basis_size,
                        solver=model.solver,
                        subtract_phi0=True,
                        use_svd=True,
                        center=False,
                        scale=False,
                    )
                )
            bases.append(row_bases)
        return bases

    def _scattering_emulator(
        self,
        interaction: Any,
        bases: list[list[Any]],
    ) -> Any:
        return rose.ScatteringAmplitudeEmulator(
            interaction,
            bases,
            l_max=self._l_max,
            angles=self.angles_radians,
            s_0=self._base_solver().s_0,
            Smatrix_abs_tol=1e-8,
            initialize_emulator=True,
        )

    def _exact_smatrix(
        self,
        emulator: Any,
        row: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        splus = np.empty(len(emulator.rbes), dtype=np.complex128)
        sminus = np.empty_like(splus)
        for ell, rbe_row in enumerate(emulator.rbes):
            splus[ell] = rbe_row[0].basis.solver.smatrix(row)
            sminus[ell] = (
                splus[ell]
                if ell == 0
                else rbe_row[1].basis.solver.smatrix(row)
            )
        return splus, sminus

    def _emulated_smatrix(
        self,
        emulator: Any,
        row: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        splus = np.empty(len(emulator.rbes), dtype=np.complex128)
        sminus = np.empty_like(splus)
        for ell, rbe_row in enumerate(emulator.rbes):
            splus[ell] = rbe_row[0].S_matrix_element(row)
            sminus[ell] = (
                splus[ell]
                if ell == 0
                else rbe_row[1].S_matrix_element(row)
            )
        return splus, sminus

    def _cross_section(
        self,
        emulator: Any,
        row: np.ndarray,
        splus: np.ndarray,
        sminus: np.ndarray,
    ) -> np.ndarray:
        if not np.array_equal(emulator.angles, self.angles_radians):
            raise ValueError("ROSE angle cache does not match the study grid.")
        return emulator.calculate_xs(splus, sminus, row).dsdo

    def _cross_sections(
        self,
        emulator: Any,
        rows: np.ndarray,
        *,
        exact: bool,
    ) -> np.ndarray:
        smatrix = self._exact_smatrix if exact else self._emulated_smatrix
        return np.asarray(
            [
                self._cross_section(
                    emulator,
                    row,
                    *smatrix(emulator, row),
                )
                for row in rows
            ]
        )

    def _rose_times(
        self,
        emulator: Any,
    ) -> np.ndarray:
        row = self.testing_rows[0]
        self._cross_section(
            emulator,
            row,
            *self._emulated_smatrix(emulator, row),
        )
        seconds = []
        for row in self.testing_rows:
            repeats = []
            for _ in range(self.timing_repeats):
                start = time.perf_counter_ns()
                for _ in range(self.timing_inner_loops):
                    self._cross_section(
                        emulator,
                        row,
                        *self._emulated_smatrix(emulator, row),
                    )
                repeats.append(
                    (time.perf_counter_ns() - start)
                    / (1e9 * self.timing_inner_loops)
                )
            seconds.append(min(repeats))
        return np.asarray(seconds)

    def _result(
        self,
        training_cross_sections: np.ndarray,
        testing_cross_sections: np.ndarray,
        fom_training_cross_sections: np.ndarray,
        fom_testing_cross_sections: np.ndarray,
        test_seconds: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        training_error = summarize_relative_error(
            training_cross_sections,
            fom_training_cross_sections,
            self.denominator_floor,
        )
        testing_error = summarize_relative_error(
            testing_cross_sections,
            fom_testing_cross_sections,
            self.denominator_floor,
        )
        result = {
            "train_xs": training_cross_sections,
            "test_xs": testing_cross_sections,
            "train_median_over_angle_error": training_error[
                "median_over_angle_error"
            ],
            "test_median_over_angle_error": testing_error[
                "median_over_angle_error"
            ],
            "train_maximum_over_angle_error": training_error[
                "maximum_over_angle_error"
            ],
            "test_maximum_over_angle_error": testing_error[
                "maximum_over_angle_error"
            ],
        }
        if test_seconds is not None:
            result["test_seconds"] = test_seconds
        return result

    def run_rose(
        self,
        basis_sizes: tuple[int, ...],
        eim_sizes: tuple[int, ...],
    ) -> RoseCrossSectionResult:
        emulators = {}
        for basis_size in basis_sizes:
            for eim_size in eim_sizes:
                interaction = self._interaction_space(eim_size)
                bases = self._custom_bases(interaction, basis_size)
                emulators[(basis_size, eim_size)] = (
                    self._scattering_emulator(interaction, bases)
                )
        reference = emulators[(basis_sizes[0], eim_sizes[0])]
        fom_training = self._cross_sections(
            reference,
            self.training_rows,
            exact=True,
        )
        fom_testing = self._cross_sections(
            reference,
            self.testing_rows,
            exact=True,
        )
        results = {}
        for config, emulator in emulators.items():
            training = self._cross_sections(
                emulator,
                self.training_rows,
                exact=False,
            )
            testing = self._cross_sections(
                emulator,
                self.testing_rows,
                exact=False,
            )
            results[config] = self._result(
                training,
                testing,
                fom_training,
                fom_testing,
                self._rose_times(emulator),
            )
        return RoseCrossSectionResult(
            emulators=emulators,
            fom_training_cross_sections=fom_training,
            fom_testing_cross_sections=fom_testing,
            results=results,
        )

    def _run_ls_for_basis(
        self,
        basis_size: int,
        predictor_count: int,
        fom_training_cross_sections: np.ndarray,
        fom_testing_cross_sections: np.ndarray,
    ) -> dict[str, np.ndarray]:
        self.emulator.train(
            basis_size=basis_size,
            predictor="effective-interaction",
            predictor_count=predictor_count,
            observable="cross_section",
            angles_degrees=self.angles_degrees,
        )
        training_coordinates = {
            channel: lrom_v2.project_coordinates(
                basis=self.emulator.basis[channel],
                wavefunctions=(
                    self.emulator.samples.training_wavefunctions[channel]
                ),
            )
            for channel in self.emulator.basis
        }
        testing_coordinates = {
            channel: lrom_v2.project_coordinates(
                basis=self.emulator.basis[channel],
                wavefunctions=(
                    self.emulator.samples.testing_wavefunctions[channel]
                ),
            )
            for channel in self.emulator.basis
        }
        _, training_state = lrom_v2._cross_section_prediction(
            emulator=self.emulator,
            values=self.training_rows,
            coefficients=training_coordinates,
        )
        _, testing_state = lrom_v2._cross_section_prediction(
            emulator=self.emulator,
            values=self.testing_rows,
            coefficients=testing_coordinates,
        )
        return self._result(
            training_state.values.copy(),
            testing_state.values.copy(),
            fom_training_cross_sections,
            fom_testing_cross_sections,
        )

    def run_ls(
        self,
        basis_sizes: tuple[int, ...],
        fom_training_cross_sections: np.ndarray,
        fom_testing_cross_sections: np.ndarray,
        predictor_count: int,
    ) -> LSCrossSectionResult:
        return LSCrossSectionResult(
            results={
                basis_size: self._run_ls_for_basis(
                    basis_size,
                    predictor_count,
                    fom_training_cross_sections,
                    fom_testing_cross_sections,
                )
                for basis_size in basis_sizes
            }
        )

    def run(
        self,
        basis_sizes: tuple[int, ...],
        eim_sizes: tuple[int, ...],
        ls_predictor_count: int,
    ) -> CrossSectionBenchmarkResult:
        rose_result = self.run_rose(
            basis_sizes=basis_sizes,
            eim_sizes=eim_sizes,
        )
        ls_result = self.run_ls(
            basis_sizes=basis_sizes,
            fom_training_cross_sections=(
                rose_result.fom_training_cross_sections
            ),
            fom_testing_cross_sections=(
                rose_result.fom_testing_cross_sections
            ),
            predictor_count=ls_predictor_count,
        )
        return CrossSectionBenchmarkResult(
            rose=rose_result,
            ls=ls_result,
        )
