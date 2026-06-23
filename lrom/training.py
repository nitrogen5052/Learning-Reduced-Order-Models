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
    PredictionState,
    RoseRBMState,
    TestingResults,
    TrainingState,
)


def _shared_rose_basis(*, emulator, channel: int, basis_size: int) -> RoseRBMState:
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


class TrainingEngine:
    """Build shared ROSE/LROM bases, fits, and testing diagnostics."""

    def reduced_basis(self, *, emulator, basis_size: int) -> TrainingState:
        rose_states = {
            channel: _shared_rose_basis(
                emulator=emulator, channel=channel, basis_size=basis_size
            )
            for channel in emulator.partial_waves
        }
        return TrainingState(
            basis={channel: state.basis for channel, state in rose_states.items()},
            predictors=None,
            rf_lrom={},
            rose_rbm=rose_states,
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
        rose_states = basis_only.rose_rbm
        rf_models = {}
        high_fidelity = dict(samples.testing_wavefunctions)
        rose_wavefunctions = {}
        lrom_wavefunctions = {}
        ls_wavefunctions = {}
        coefficient_sets: dict[str, dict[int, np.ndarray]] = {
            "ls": {},
            "rose": {},
            "lrom": {},
        }
        testing_errors: dict[int, dict[str, np.ndarray]] = {}
        relative_metrics: dict[int, dict[str, np.ndarray]] = {}
        for channel in emulator.partial_waves:
            basis = bases[channel]
            train_coordinates = project_coordinates(
                basis=basis, wavefunctions=samples.training_wavefunctions[channel]
            )
            test_coordinates = project_coordinates(
                basis=basis, wavefunctions=samples.testing_wavefunctions[channel]
            )
            model = fit_rf_lrom(
                predictors=predictor_state.training_features,
                coordinates=train_coordinates,
            )
            lrom_coordinates = solve_rf_lrom(
                model=model, predictors=predictor_state.testing_features
            )
            rf_models[channel] = model
            coefficient_sets["ls"][channel] = test_coordinates
            coefficient_sets["lrom"][channel] = lrom_coordinates
            ls_wavefunctions[channel] = reconstruct(
                basis=basis, coordinates=test_coordinates
            )
            lrom_wavefunctions[channel] = reconstruct(
                basis=basis, coordinates=lrom_coordinates
            )
            rose_model = rose_states[channel].emulator
            coefficient_sets["rose"][channel] = np.asarray(
                [rose_model.coefficients(row) for row in samples.design.testing.values]
            )
            rose_wavefunctions[channel] = np.asarray(
                [rose_model.emulate_wave_function(row) for row in samples.design.testing.values]
            )
            testing_errors[channel] = {
                "rose": pointwise_absolute(
                    prediction=rose_wavefunctions[channel],
                    reference=high_fidelity[channel],
                ),
                "lrom": pointwise_absolute(
                    prediction=lrom_wavefunctions[channel],
                    reference=high_fidelity[channel],
                ),
                "ls": pointwise_absolute(
                    prediction=ls_wavefunctions[channel],
                    reference=high_fidelity[channel],
                ),
            }
            relative_metrics[channel] = {
                "rose": relative_l2(
                    prediction=rose_wavefunctions[channel],
                    reference=high_fidelity[channel],
                ),
                "lrom": relative_l2(
                    prediction=lrom_wavefunctions[channel],
                    reference=high_fidelity[channel],
                ),
                "ls": relative_l2(
                    prediction=ls_wavefunctions[channel],
                    reference=high_fidelity[channel],
                ),
            }
        results = TestingResults(
            high_fidelity=high_fidelity,
            rose=rose_wavefunctions,
            lrom=lrom_wavefunctions,
            ls=ls_wavefunctions,
            coefficients=coefficient_sets,
            metrics={"relative_l2": relative_metrics},
        )
        return TrainingState(
            basis=bases,
            predictors=predictor_state,
            rf_lrom=rf_models,
            rose_rbm=rose_states,
            testing_results=results,
            testing_errors=testing_errors,
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
