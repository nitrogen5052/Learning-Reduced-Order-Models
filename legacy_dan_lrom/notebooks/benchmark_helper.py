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


HBAR_C_MEV_FM = 197.3269804
MASS_PION_INVERSE_FM = np.sqrt(0.5)


@dataclass(frozen=True)
class RoseWavefunctionResult:
    """ROSE wavefunction benchmark state on shared training/testing rows."""

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
    """Exact least-squares coordinates and reconstructions in the LROM basis."""

    training_coefficients: np.ndarray
    testing_coefficients: np.ndarray
    training_wavefunctions: np.ndarray
    testing_wavefunctions: np.ndarray


@dataclass(frozen=True)
class WavefunctionBenchmarkResult:
    """Matched ROSE and least-squares wavefunction results."""

    rose: RoseWavefunctionResult
    ls: LSWavefunctionResult


@dataclass(frozen=True)
class RoseCrossSectionResult:
    """ROSE emulator grid, exact references, and per-configuration summaries."""

    emulators: dict[tuple[int, int], Any]
    fom_training_cross_sections: np.ndarray
    fom_testing_cross_sections: np.ndarray
    results: dict[tuple[int, int], dict[str, np.ndarray]]


@dataclass(frozen=True)
class LSCrossSectionResult:
    """Least-squares cross-section summaries keyed by wavefunction basis size."""

    results: dict[int, dict[str, np.ndarray]]


@dataclass(frozen=True)
class LromCrossSectionResult:
    """LROM summaries and the optionally captured predictor configuration."""

    results: dict[tuple[int, int], dict[str, np.ndarray]]
    predictors: Mapping[Any, Any] | None


@njit
def rose_real_woods_saxon(
    radius: float,
    alpha: np.ndarray,
) -> complex:
    """Return the real volume Woods-Saxon potential used by Notebook 01."""
    vv, rv, av = alpha
    return -vv / (1.0 + np.exp((radius - rv) / av))


@njit
def full_woods_saxon(
    radius: float,
    alpha: np.ndarray,
) -> complex:
    """Return the complex central full Woods-Saxon interaction in MeV."""
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
    """Return the ROSE spin-orbit term in MeV for radius in fm.

    ``ldots`` is ROSE's channel coefficient ``2 l.s``. The inverse-fm pion
    scale gives ``(hbar / m_pi c)^2 = 2 fm^2`` before the two radial inverse
    lengths from ``(1/r) d/dr`` are applied.
    """
    _vv, _wv, _wd, vso, _rv, _rd, rso, _av, _ad, aso = alpha
    exponential = np.exp((radius - rso) / aso)
    derivative = -(exponential / aso) / (1.0 + exponential) ** 2
    return vso / MASS_PION_INVERSE_FM**2 * ldots * derivative / radius


def centrifugal_barrier_mev(
    radius_fm: np.ndarray,
    *,
    ell: int,
    reduced_mass_mev: float,
) -> np.ndarray:
    """Return ``hbar^2 ell(ell+1) / (2 mu r^2)`` in MeV.

    Parameters use physical radius in fm and reduced rest energy ``mu c^2``
    in MeV, matching the kinematic convention used by the LROM emulator.
    """
    radius = np.asarray(radius_fm, dtype=float)
    if np.any(radius <= 0.0):
        raise ValueError("radius_fm must contain only positive radii")
    if ell < 0:
        raise ValueError("ell must be nonnegative")
    if reduced_mass_mev <= 0.0:
        raise ValueError("reduced_mass_mev must be positive")
    return (
        HBAR_C_MEV_FM**2
        * ell
        * (ell + 1)
        / (2.0 * reduced_mass_mev * radius**2)
    )


def pointwise_relative_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    denominator_floor: float,
) -> np.ndarray:
    """Return raw absolute relative differences without a logarithmic transform."""
    denominator = np.maximum(np.abs(reference), denominator_floor)
    return np.abs(np.asarray(predicted) - np.asarray(reference)) / denominator


def summarize_relative_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    denominator_floor: float,
) -> dict[str, np.ndarray]:
    """Reduce each case's raw pointwise errors over the angle dimension."""
    pointwise = pointwise_relative_error(
        predicted,
        reference,
        denominator_floor,
    )
    return {
        "median_over_angle_error": np.median(pointwise, axis=1),
        "maximum_over_angle_error": np.max(pointwise, axis=1),
    }


def cross_section_result(
    training_cross_sections: np.ndarray,
    testing_cross_sections: np.ndarray,
    fom_training_cross_sections: np.ndarray,
    fom_testing_cross_sections: np.ndarray,
    *,
    denominator_floor: float,
    test_seconds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build one raw cross-section and relative-error result record."""
    training = np.asarray(training_cross_sections)
    testing = np.asarray(testing_cross_sections)
    fom_training = np.asarray(fom_training_cross_sections)
    fom_testing = np.asarray(fom_testing_cross_sections)
    if training.shape != fom_training.shape:
        raise ValueError("training and FOM cross-section shapes differ")
    if testing.shape != fom_testing.shape:
        raise ValueError("testing and FOM cross-section shapes differ")
    if training.ndim != 2 or testing.ndim != 2:
        raise ValueError("cross-section arrays must have shape (cases, angles)")
    training_error = summarize_relative_error(
        training,
        fom_training,
        denominator_floor,
    )
    testing_error = summarize_relative_error(
        testing,
        fom_testing,
        denominator_floor,
    )
    result = {
        "train_xs": training,
        "test_xs": testing,
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
        seconds = np.asarray(test_seconds, dtype=float)
        if seconds.shape != (testing.shape[0],):
            raise ValueError("test_seconds must contain one value per testing case")
        result["test_seconds"] = seconds
    return result


def validate_configuration_grid(
    results: Mapping[tuple[int, int], Any],
    *,
    basis_sizes: tuple[int, ...],
    compression_sizes: tuple[int, ...],
    label: str,
) -> None:
    """Require a complete Cartesian configuration grid without extra keys."""
    expected = {
        (basis_size, compression_size)
        for basis_size in basis_sizes
        for compression_size in compression_sizes
    }
    if set(results) != expected:
        missing = sorted(expected - set(results))
        extra = sorted(set(results) - expected)
        raise ValueError(
            f"{label} configuration grid is incomplete: "
            f"missing={missing}, extra={extra}"
        )


def select_configurations(
    *,
    basis_sizes: tuple[int, ...],
    compression_sizes: tuple[int, ...],
    configurations: tuple[tuple[int, int], ...] | None,
) -> tuple[tuple[int, int], ...]:
    """Return an ordered full grid or a validated requested subset."""
    available = {
        (basis_size, compression_size)
        for basis_size in basis_sizes
        for compression_size in compression_sizes
    }
    selected = (
        tuple(
            (basis_size, compression_size)
            for basis_size in basis_sizes
            for compression_size in compression_sizes
        )
        if configurations is None
        else tuple(configurations)
    )
    if not selected:
        raise ValueError("at least one configuration is required")
    if len(set(selected)) != len(selected):
        raise ValueError("configurations must not contain duplicates")
    unavailable = set(selected) - available
    if unavailable:
        raise ValueError(f"configurations are outside the requested sizes: {sorted(unavailable)}")
    return selected


def validate_partial_wave_topology(
    channel_rows: list[list[Any]],
    *,
    l_max: int,
) -> tuple[int, ...]:
    """Validate inclusive partial-wave and spin-orbit channel coverage.

    ROSE stores one ``ell=0`` channel and two channels for each positive
    partial wave, corresponding to the two allowed total-angular-momentum
    branches.
    """
    if l_max < 0:
        raise ValueError("l_max must be nonnegative")
    counts = tuple(len(row) for row in channel_rows)
    if len(counts) != l_max + 1:
        raise ValueError(
            "partial waves must provide inclusive ell=0,...,l_max coverage"
        )
    if counts[0] != 1:
        raise ValueError("ell=0 must contain exactly one channel")
    for ell, count in enumerate(counts[1:], start=1):
        if count != 2:
            raise ValueError(f"ell={ell} must contain exactly two spin channels")
    return counts


def validate_predictor_radii(
    displayed: Mapping[Any, Any],
    trained: Mapping[Any, Any],
) -> dict[Any, np.ndarray]:
    """Validate exact physical-radius parity for displayed/trained predictors."""
    if set(displayed) != set(trained):
        raise ValueError("displayed and trained predictor channel keys differ")
    validated: dict[Any, np.ndarray] = {}
    for channel in displayed:
        shown = np.sort(np.asarray(displayed[channel].selected_radii, dtype=float))
        fitted = np.sort(np.asarray(trained[channel].selected_radii, dtype=float))
        if shown.ndim != 1 or fitted.ndim != 1:
            raise ValueError(f"channel {channel!r} selected radii must be one-dimensional")
        if not np.all(np.isfinite(shown)) or not np.all(np.isfinite(fitted)):
            raise ValueError(f"channel {channel!r} selected radii must be finite")
        if not np.array_equal(shown, fitted):
            raise ValueError(f"channel {channel!r} selected radii differ")
        validated[channel] = shown
    return validated


class WavefunctionBenchmark:
    """Construct matched Notebook 01 ROSE and least-squares comparisons."""

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
        """Build and evaluate one ROSE reduced wavefunction emulator."""
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
        """Project exact wavefunctions into the fixed LROM basis."""
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
        """Return the matched ROSE and least-squares wavefunction results."""
        rose_result = self.run_rose(
            basis_size=basis_size,
            eim_basis_size=eim_basis_size,
            include_training_wavefunctions=include_training_wavefunctions,
        )
        return WavefunctionBenchmarkResult(
            rose=rose_result,
            ls=self.run_ls(),
        )


class RoseCrossSectionPipeline:
    """Self-contained ROSE cross-section construction and evaluation."""

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

    def run(
        self,
        basis_sizes: tuple[int, ...],
        eim_sizes: tuple[int, ...],
        configurations: tuple[tuple[int, int], ...] | None = None,
    ) -> RoseCrossSectionResult:
        """Build and evaluate the requested ROSE configurations."""
        selected = select_configurations(
            basis_sizes=basis_sizes,
            compression_sizes=eim_sizes,
            configurations=configurations,
        )
        emulators = {}
        for basis_size, eim_size in selected:
            interaction = self._interaction_space(eim_size)
            bases = self._custom_bases(interaction, basis_size)
            emulators[(basis_size, eim_size)] = self._scattering_emulator(
                interaction, bases
            )
        reference = emulators[selected[0]]
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
            results[config] = cross_section_result(
                training,
                testing,
                fom_training,
                fom_testing,
                denominator_floor=self.denominator_floor,
                test_seconds=self._rose_times(emulator),
            )
        if set(results) != set(selected):
            raise ValueError("ROSE results do not match requested configurations")
        return RoseCrossSectionResult(
            emulators=emulators,
            fom_training_cross_sections=fom_training,
            fom_testing_cross_sections=fom_testing,
            results=results,
        )


class LromCrossSectionStudy:
    """Train and evaluate parked-v2 LROM and least-squares cross sections."""

    def __init__(
        self,
        *,
        emulator: Any,
        training_cases: list[dict[str, float]],
        testing_cases: list[dict[str, float]],
        training_rows: np.ndarray,
        testing_rows: np.ndarray,
        fom_training_cross_sections: np.ndarray,
        fom_testing_cross_sections: np.ndarray,
        angles_degrees: np.ndarray,
        denominator_floor: float,
        timing_repeats: int,
        timing_inner_loops: int,
    ) -> None:
        self.emulator = emulator
        self.training_cases = training_cases
        self.testing_cases = testing_cases
        self.training_rows = np.asarray(training_rows, dtype=float)
        self.testing_rows = np.asarray(testing_rows, dtype=float)
        self.fom_training_cross_sections = np.asarray(
            fom_training_cross_sections
        )
        self.fom_testing_cross_sections = np.asarray(fom_testing_cross_sections)
        self.angles_degrees = np.asarray(angles_degrees, dtype=float)
        self.denominator_floor = denominator_floor
        self.timing_repeats = timing_repeats
        self.timing_inner_loops = timing_inner_loops
        if len(self.training_cases) != self.training_rows.shape[0]:
            raise ValueError("training cases and rows require equal lengths")
        if len(self.testing_cases) != self.testing_rows.shape[0]:
            raise ValueError("testing cases and rows require equal lengths")
        if timing_repeats < 1 or timing_inner_loops < 1:
            raise ValueError("timing controls must be positive")

    def _predict_cross_sections(
        self,
        cases: list[dict[str, float]],
    ) -> np.ndarray:
        self.emulator.predict(
            parameters=cases,
            reconstruct_wavefunctions=False,
        )
        return self.emulator.predictions.cross_sections.values.copy()

    def _prediction_times(self) -> np.ndarray:
        self.emulator.predict(
            parameters=self.testing_cases[:1],
            reconstruct_wavefunctions=False,
        )
        seconds = []
        for case in self.testing_cases:
            measurements = []
            for _ in range(self.timing_repeats):
                start = time.perf_counter_ns()
                for _ in range(self.timing_inner_loops):
                    self.emulator.predict(
                        parameters=case,
                        reconstruct_wavefunctions=False,
                    )
                measurements.append(
                    (time.perf_counter_ns() - start)
                    / (1e9 * self.timing_inner_loops)
                )
            seconds.append(min(measurements))
        return np.asarray(seconds)

    def _train_and_evaluate(
        self,
        *,
        basis_size: int,
        predictor: str,
        operator_count: int,
    ) -> dict[str, np.ndarray]:
        self.emulator.train(
            basis_size=basis_size,
            predictor=predictor,
            predictor_count=operator_count,
            observable="cross_section",
            angles_degrees=self.angles_degrees,
        )
        test_seconds = self._prediction_times()
        training = self._predict_cross_sections(self.training_cases)
        testing = self._predict_cross_sections(self.testing_cases)
        return cross_section_result(
            training,
            testing,
            self.fom_training_cross_sections,
            self.fom_testing_cross_sections,
            denominator_floor=self.denominator_floor,
            test_seconds=test_seconds,
        )

    def run_lrom(
        self,
        *,
        basis_sizes: tuple[int, ...],
        operator_counts: tuple[int, ...],
        capture_predictors_at: tuple[int, int],
        configurations: tuple[tuple[int, int], ...] | None = None,
    ) -> LromCrossSectionResult:
        """Evaluate the requested learned effective-interaction configurations."""
        selected = select_configurations(
            basis_sizes=basis_sizes,
            compression_sizes=operator_counts,
            configurations=configurations,
        )
        results = {}
        captured_predictors = None
        for basis_size, operator_count in selected:
            key = (basis_size, operator_count)
            results[key] = self._train_and_evaluate(
                basis_size=basis_size,
                predictor="effective-interaction",
                operator_count=operator_count,
            )
            if key == capture_predictors_at:
                captured_predictors = self.emulator.predictors
        if set(results) != set(selected):
            raise ValueError("LROM results do not match requested configurations")
        if captured_predictors is None:
            raise ValueError("capture_predictors_at is outside the LROM grid")
        return LromCrossSectionResult(
            results=results,
            predictors=captured_predictors,
        )

    def run_parked_v2_reference(
        self,
        *,
        basis_size: int,
        operator_count: int,
    ) -> LromCrossSectionResult:
        """Evaluate the parked-v2 potential-predictor reference once."""
        key = (basis_size, operator_count)
        return LromCrossSectionResult(
            results={
                key: self._train_and_evaluate(
                    basis_size=basis_size,
                    predictor="potential",
                    operator_count=operator_count,
                )
            },
            predictors=None,
        )

    def _run_ls_for_basis(
        self,
        *,
        basis_size: int,
        operator_count: int,
    ) -> dict[str, np.ndarray]:
        self.emulator.train(
            basis_size=basis_size,
            predictor="effective-interaction",
            predictor_count=operator_count,
            observable="cross_section",
            angles_degrees=self.angles_degrees,
        )
        training_coordinates = {
            channel: lrom_v2.project_coordinates(
                basis=self.emulator.basis[channel],
                wavefunctions=self.emulator.samples.training_wavefunctions[
                    channel
                ],
            )
            for channel in self.emulator.basis
        }
        testing_coordinates = {
            channel: lrom_v2.project_coordinates(
                basis=self.emulator.basis[channel],
                wavefunctions=self.emulator.samples.testing_wavefunctions[
                    channel
                ],
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
        return cross_section_result(
            training_state.values.copy(),
            testing_state.values.copy(),
            self.fom_training_cross_sections,
            self.fom_testing_cross_sections,
            denominator_floor=self.denominator_floor,
        )

    def run_ls(
        self,
        *,
        basis_sizes: tuple[int, ...],
        operator_count: int,
    ) -> LSCrossSectionResult:
        """Evaluate the held-out-wavefunction least-squares oracle."""
        return LSCrossSectionResult(
            results={
                basis_size: self._run_ls_for_basis(
                    basis_size=basis_size,
                    operator_count=operator_count,
                )
                for basis_size in basis_sizes
            }
        )
