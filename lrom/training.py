"""Private training orchestration for the stateful LROM object."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .basis import project_coordinates, reconstruct
from .diagnostics import pointwise_absolute, relative_l2
from .fom import _import_rose
from .predictors import (
    PredictorState,
    build_parameter_predictor,
    build_potential_predictor,
    features_for_values,
)
from .rf import fit as fit_rf_lrom
from .rf import solve as solve_rf_lrom
from .state import (
    BasisState,
    CrossSectionState,
    PredictionState,
    RoseRBMState,
    SmatrixState,
    TestingResults,
    TrainingState,
)


def _channel_sort_key(channel) -> tuple[int, int]:
    if isinstance(channel, tuple):
        return int(channel[0]), int(channel[1])
    return int(channel), 0


def _trained_channels(emulator) -> tuple:
    return tuple(sorted(emulator.samples.full_order_models, key=_channel_sort_key))


def _interaction_channel_key(interaction_row: Sequence, ell: int, spin_index: int):
    return ell if len(interaction_row) == 1 else (ell, spin_index)


def _shared_rose_basis(*, emulator, channel, basis_size: int) -> RoseRBMState:
    rose = _import_rose()
    samples = emulator.samples
    model = samples.full_order_models[channel]
    custom_basis = rose.basis.CustomBasis(
        solutions=np.asarray(samples.training_wavefunctions[channel], dtype=np.complex128).T.copy(),
        phi_0=np.asarray(samples.central_wavefunctions[channel], dtype=np.complex128).copy(),
        rho_mesh=samples.mesh.rho,
        n_basis=basis_size,
        solver=model.solver,
        subtract_phi0=True,
        use_svd=True,
        center=False,
        scale=False,
    )
    singular_values = np.asarray(custom_basis.singular_values, dtype=float)[:basis_size]
    basis = BasisState(
        phi0=custom_basis.phi_0,
        vectors=custom_basis.vectors,
        radius=samples.mesh.radius,
        singular_values=singular_values,
    )
    rose_emulator = rose.reduced_basis_emulator.ReducedBasisEmulator(
        model.interaction,
        custom_basis,
        s_0=model.base_solver.s_0,
        initialize_emulator=True,
    )
    return RoseRBMState(
        basis=basis,
        custom_basis=custom_basis,
        emulator=rose_emulator,
    )


def _predictor(*, emulator, kind: str, count: int) -> PredictorState:
    samples = emulator.samples
    central = np.asarray(
        [emulator.central_parameters[name] for name in emulator.parameter_names]
    )
    if kind == "parameters":
        varying = np.ptp(samples.design.training.values, axis=0) > 0.0
        varied_names = tuple(
            name for name, varied in zip(emulator.parameter_names, varying) if varied
        )
        return build_parameter_predictor(
            parameter_names=emulator.parameter_names,
            varied_names=varied_names,
            central=central,
            training_values=samples.design.training.values,
            testing_values=samples.design.testing.values,
        )
    if kind == "potential":
        return build_potential_predictor(
            radius=samples.mesh.radius,
            central_potential=samples.central_potential,
            training_potentials=samples.training_potentials,
            testing_potentials=samples.testing_potentials,
            predictor_count=count,
            minimum_radius=0.2,
        )
    raise ValueError("predictor must be 'parameters' or 'potential'")


def _evaluate(
    *,
    emulator,
    bases,
    rose_states,
    rf_models,
    wavefunctions,
    parameter_values,
    predictor_features,
) -> TestingResults:
    high_fidelity = dict(wavefunctions)
    rose_wavefunctions = {}
    lrom_wavefunctions = {}
    ls_wavefunctions = {}
    coefficient_sets: dict[str, dict[object, np.ndarray]] = {
        "ls": {},
        "rose": {},
        "lrom": {},
    }
    pointwise_metrics: dict[object, dict[str, np.ndarray]] = {}
    relative_metrics: dict[object, dict[str, np.ndarray]] = {}
    for channel in _trained_channels(emulator):
        basis = bases[channel]
        ls_coordinates = project_coordinates(
            basis=basis,
            wavefunctions=high_fidelity[channel],
        )
        lrom_coordinates = solve_rf_lrom(
            model=rf_models[channel],
            predictors=predictor_features,
        )
        rose_model = rose_states[channel].emulator
        rose_coordinates = np.asarray(
            [rose_model.coefficients(row) for row in parameter_values]
        )
        coefficient_sets["ls"][channel] = ls_coordinates
        coefficient_sets["lrom"][channel] = lrom_coordinates
        coefficient_sets["rose"][channel] = rose_coordinates
        ls_wavefunctions[channel] = reconstruct(
            basis=basis,
            coordinates=ls_coordinates,
        )
        lrom_wavefunctions[channel] = reconstruct(
            basis=basis,
            coordinates=lrom_coordinates,
        )
        rose_wavefunctions[channel] = np.asarray(
            [rose_model.emulate_wave_function(row) for row in parameter_values]
        )
        predictions = {
            "rose": rose_wavefunctions[channel],
            "lrom": lrom_wavefunctions[channel],
            "ls": ls_wavefunctions[channel],
        }
        pointwise_metrics[channel] = {
            method: pointwise_absolute(
                prediction=prediction,
                reference=high_fidelity[channel],
            )
            for method, prediction in predictions.items()
        }
        relative_metrics[channel] = {
            method: relative_l2(
                prediction=prediction,
                reference=high_fidelity[channel],
            )
            for method, prediction in predictions.items()
        }
    return TestingResults(
        high_fidelity=high_fidelity,
        rose=rose_wavefunctions,
        lrom=lrom_wavefunctions,
        ls=ls_wavefunctions,
        coefficients=coefficient_sets,
        metrics={
            "relative_l2": relative_metrics,
            "pointwise_absolute": pointwise_metrics,
        },
    )


class TrainingEngine:
    """Build shared ROSE/LROM bases, fits, and testing diagnostics."""

    def reduced_basis(self, *, emulator, basis_size: int) -> TrainingState:
        rose_states = {
            channel: _shared_rose_basis(
                emulator=emulator, channel=channel, basis_size=basis_size
            )
            for channel in _trained_channels(emulator)
        }
        return TrainingState(
            basis={channel: state.basis for channel, state in rose_states.items()},
            predictors=None,
            rf_lrom={},
            rose_rbm=rose_states,
            testing_results=None,
            testing_errors={channel: {} for channel in _trained_channels(emulator)},
            training_options={"basis_size": basis_size},
        )

    def train(
        self,
        *,
        emulator,
        basis_size: int,
        predictor: str,
        predictor_count: int,
        operator_basis_size: int | None,
        observable: str,
        angles_degrees: Sequence[float] | None,
    ) -> TrainingState:
        if (
            operator_basis_size is not None
            and (
                isinstance(operator_basis_size, bool)
                or not isinstance(operator_basis_size, int)
                or operator_basis_size < 1
            )
        ):
            raise ValueError("operator_basis_size must be a positive integer")
        if observable not in {"wavefunction", "cross_section"}:
            raise ValueError("observable must be 'wavefunction' or 'cross_section'")
        angle_array = None
        if angles_degrees is not None:
            angle_array = np.asarray(angles_degrees, dtype=float)
            if angle_array.ndim != 1 or angle_array.size == 0:
                raise ValueError(
                    "angles_degrees must be a non-empty one-dimensional array"
                )
            if not np.all(np.isfinite(angle_array)):
                raise ValueError("angles_degrees must contain finite values")
        if observable == "cross_section" and angle_array is None:
            raise ValueError("cross_section observable requires angles_degrees")
        basis_only = self.reduced_basis(emulator=emulator, basis_size=basis_size)
        predictor_state = _predictor(
            emulator=emulator, kind=predictor, count=predictor_count
        )
        samples = emulator.samples
        bases = basis_only.basis
        rose_states = basis_only.rose_rbm
        rf_models = {}
        for channel in _trained_channels(emulator):
            basis = bases[channel]
            train_coordinates = project_coordinates(
                basis=basis, wavefunctions=samples.training_wavefunctions[channel]
            )
            model = fit_rf_lrom(
                predictors=predictor_state.training_features,
                coordinates=train_coordinates,
            )
            rf_models[channel] = model
        training_results = _evaluate(
            emulator=emulator,
            bases=bases,
            rose_states=rose_states,
            rf_models=rf_models,
            wavefunctions=samples.training_wavefunctions,
            parameter_values=samples.design.training.values,
            predictor_features=predictor_state.training_features,
        )
        testing_results = _evaluate(
            emulator=emulator,
            bases=bases,
            rose_states=rose_states,
            rf_models=rf_models,
            wavefunctions=samples.testing_wavefunctions,
            parameter_values=samples.design.testing.values,
            predictor_features=predictor_state.testing_features,
        )
        return TrainingState(
            basis=bases,
            predictors=predictor_state,
            rf_lrom=rf_models,
            rose_rbm=rose_states,
            testing_results=testing_results,
            testing_errors=testing_results.metrics["pointwise_absolute"],
            training_results=training_results,
            training_options={
                "basis_size": basis_size,
                "predictor": predictor,
                "predictor_count": predictor_count,
                "operator_basis_size": operator_basis_size,
                "observable": observable,
                "angles_degrees": angle_array,
            },
        )


def _parameter_rows(*, emulator, parameters) -> np.ndarray:
    rows = [parameters] if isinstance(parameters, Mapping) else list(parameters)
    if not rows:
        raise ValueError("parameters must contain at least one case")
    central = dict(emulator.central_parameters)
    result = []
    for row in rows:
        unknown = sorted(set(row) - set(emulator.parameter_names))
        if unknown:
            raise ValueError(f"unknown parameter names: {unknown}")
        merged = central | {name: float(value) for name, value in row.items()}
        result.append([merged[name] for name in emulator.parameter_names])
    values = np.asarray(result, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("prediction parameters must be finite")
    return values


def _s_matrix_from_coefficients(rbe, coeff: np.ndarray):
    x = np.hstack((1, np.asarray(coeff, dtype=np.complex128)))
    phi = np.dot(x, rbe.asymptotic_vals)
    phi_prime = np.dot(x, rbe.asymptotic_ders)
    r_matrix = 1 / rbe.s_0 * phi / phi_prime
    return (rbe.Hm - rbe.s_0 * r_matrix * rbe.Hmp) / (
        rbe.Hp - rbe.s_0 * r_matrix * rbe.Hpp
    )


def _cross_section_prediction(
    *,
    emulator,
    values: np.ndarray,
    coefficients: Mapping[int, np.ndarray],
) -> tuple[SmatrixState, CrossSectionState]:
    samples = emulator.samples
    options = emulator.training_options or {}
    angles_degrees = np.asarray(options.get("angles_degrees"), dtype=float)
    angles = np.deg2rad(angles_degrees)
    partial_waves = tuple(emulator.partial_waves)
    if samples.interaction_space is None:
        raise RuntimeError("cross-section prediction requires sampled interaction state")
    if partial_waves != tuple(range(max(partial_waves) + 1)):
        raise ValueError(
            "cross-section prediction requires contiguous partial waves starting at l=0"
        )
    rose = _import_rose()
    bases = []
    for ell in partial_waves:
        interaction_row = samples.interaction_space.interactions[ell]
        bases.append(
            [
                emulator.rose_rbm[
                    _interaction_channel_key(interaction_row, ell, spin_index)
                ].custom_basis
                for spin_index in range(len(interaction_row))
            ]
        )
    sae = rose.ScatteringAmplitudeEmulator(
        samples.interaction_space,
        bases,
        l_max=max(partial_waves),
        angles=angles,
        s_0=samples.full_order_models[partial_waves[0]].base_solver.s_0,
        Smatrix_abs_tol=1e-8,
        initialize_emulator=True,
    )
    splus_rows = []
    sminus_rows = []
    cross_sections = []
    for case_index, row in enumerate(values):
        splus = np.zeros(len(partial_waves), dtype=np.complex128)
        sminus = np.zeros(len(partial_waves), dtype=np.complex128)
        for offset, channel in enumerate(partial_waves):
            interaction_row = samples.interaction_space.interactions[channel]
            plus_key = _interaction_channel_key(interaction_row, channel, 0)
            coeff = coefficients[plus_key][case_index]
            splus[offset] = _s_matrix_from_coefficients(
                sae.rbes[channel][0],
                coeff,
            )
            if len(interaction_row) == 1:
                sminus[offset] = splus[offset]
            else:
                minus_key = _interaction_channel_key(interaction_row, channel, 1)
                sminus[offset] = _s_matrix_from_coefficients(
                    sae.rbes[channel][1],
                    coefficients[minus_key][case_index],
                )
        splus_rows.append(splus)
        sminus_rows.append(sminus)
        cross_sections.append(sae.calculate_xs(splus, sminus, row, angles=angles).dsdo)
    return (
        SmatrixState(
            partial_waves=partial_waves,
            splus=np.asarray(splus_rows),
            sminus=np.asarray(sminus_rows),
        ),
        CrossSectionState(
            angles_degrees=angles_degrees,
            values=np.asarray(cross_sections, dtype=float),
        ),
    )


def predict(*, emulator, parameters) -> PredictionState:
    """Predict one or more named parameter cases from trained portable state."""
    values = _parameter_rows(emulator=emulator, parameters=parameters)
    predictor = emulator.predictors
    features = features_for_values(
        predictor=predictor,
        values=values,
        potential_function=emulator.config.potential.function,
    )
    coefficients = {
        channel: solve_rf_lrom(model=model, predictors=features)
        for channel, model in emulator.rf_lrom.items()
    }
    wavefunctions = {
        channel: reconstruct(
            basis=emulator.basis[channel], coordinates=coordinates
        )
        for channel, coordinates in coefficients.items()
    }
    smatrix = None
    cross_sections = None
    options = emulator.training_options or {}
    if options.get("observable") == "cross_section":
        smatrix, cross_sections = _cross_section_prediction(
            emulator=emulator,
            values=values,
            coefficients=coefficients,
        )
    return PredictionState(
        parameter_names=emulator.parameter_names,
        parameters=values,
        coefficients=coefficients,
        wavefunctions=wavefunctions,
        smatrix=smatrix,
        cross_sections=cross_sections,
    )
