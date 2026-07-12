"""Private training orchestration for the stateful LROM object."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .basis import build_basis, project_coordinates, reconstruct
from .diagnostics import pointwise_absolute, relative_l2
from .predictors import (
    PredictorState,
    build_parameter_predictor,
    build_potential_predictor,
    features_for_values,
)
from .rf import fit as fit_rf_lrom
from .rf import solve as solve_rf_lrom
from .state import (
    PredictionState,
    TestingResults,
    TrainingState,
)


def _centered_basis(*, emulator, channel: int, basis_size: int):
    samples = emulator.samples
    return build_basis(
        phi0=np.asarray(samples.central_wavefunctions[channel], dtype=np.complex128),
        snapshots=np.asarray(samples.training_wavefunctions[channel], dtype=np.complex128),
        radius=samples.mesh.radius,
        basis_size=basis_size,
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
    rf_models,
    wavefunctions,
    predictor_features,
) -> TestingResults:
    high_fidelity = dict(wavefunctions)
    lrom_wavefunctions = {}
    ls_wavefunctions = {}
    coefficient_sets: dict[str, dict[int, np.ndarray]] = {
        "ls": {},
        "lrom": {},
    }
    pointwise_metrics: dict[int, dict[str, np.ndarray]] = {}
    relative_metrics: dict[int, dict[str, np.ndarray]] = {}
    for channel in emulator.partial_waves:
        basis = bases[channel]
        ls_coordinates = project_coordinates(
            basis=basis,
            wavefunctions=high_fidelity[channel],
        )
        lrom_coordinates = solve_rf_lrom(
            model=rf_models[channel],
            predictors=predictor_features,
        )
        coefficient_sets["ls"][channel] = ls_coordinates
        coefficient_sets["lrom"][channel] = lrom_coordinates
        ls_wavefunctions[channel] = reconstruct(
            basis=basis,
            coordinates=ls_coordinates,
        )
        lrom_wavefunctions[channel] = reconstruct(
            basis=basis,
            coordinates=lrom_coordinates,
        )
        predictions = {
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
        lrom=lrom_wavefunctions,
        ls=ls_wavefunctions,
        coefficients=coefficient_sets,
        metrics={
            "relative_l2": relative_metrics,
            "pointwise_absolute": pointwise_metrics,
        },
    )


class TrainingEngine:
    """Build centered LROM bases, fits, and testing diagnostics."""

    def reduced_basis(self, *, emulator, basis_size: int) -> TrainingState:
        bases = {
            channel: _centered_basis(
                emulator=emulator, channel=channel, basis_size=basis_size
            )
            for channel in emulator.partial_waves
        }
        return TrainingState(
            basis=bases,
            predictors=None,
            rf_lrom={},
            testing_results=None,
            testing_errors={channel: {} for channel in emulator.partial_waves},
        )

    def train(
        self,
        *,
        emulator,
        basis_size: int,
        predictor: str,
        predictor_count: int,
    ) -> TrainingState:
        basis_only = self.reduced_basis(emulator=emulator, basis_size=basis_size)
        predictor_state = _predictor(
            emulator=emulator, kind=predictor, count=predictor_count
        )
        samples = emulator.samples
        bases = basis_only.basis
        rf_models = {}
        for channel in emulator.partial_waves:
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
            rf_models=rf_models,
            wavefunctions=samples.training_wavefunctions,
            predictor_features=predictor_state.training_features,
        )
        testing_results = _evaluate(
            emulator=emulator,
            bases=bases,
            rf_models=rf_models,
            wavefunctions=samples.testing_wavefunctions,
            predictor_features=predictor_state.testing_features,
        )
        return TrainingState(
            basis=bases,
            predictors=predictor_state,
            rf_lrom=rf_models,
            testing_results=testing_results,
            testing_errors=testing_results.metrics["pointwise_absolute"],
            training_results=training_results,
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
    return PredictionState(
        parameter_names=emulator.parameter_names,
        parameters=values,
        coefficients=coefficients,
        wavefunctions=wavefunctions,
    )
