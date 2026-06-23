"""Named parameter and maxvol-selected potential predictors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictorState:
    """Fitted transformation from physical parameters to RF predictors."""

    kind: str
    names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_indices: np.ndarray
    center: np.ndarray
    scales: np.ndarray
    training_features: np.ndarray
    testing_features: np.ndarray
    selected_indices: np.ndarray
    selected_radii: np.ndarray
    central_values: np.ndarray
    singular_values: np.ndarray


def build_parameter_predictor(
    *,
    parameter_names: tuple[str, ...],
    varied_names: tuple[str, ...],
    central: np.ndarray,
    training_values: np.ndarray,
    testing_values: np.ndarray,
) -> PredictorState:
    """Normalize named parameter deviations by the training-domain extent."""
    indices = np.asarray([parameter_names.index(name) for name in varied_names], dtype=int)
    central = np.asarray(central, dtype=float)
    training_values = np.asarray(training_values, dtype=float)
    testing_values = np.asarray(testing_values, dtype=float)
    delta_training = training_values[:, indices] - central[indices]
    scales = np.max(np.abs(delta_training), axis=0)
    if np.any(scales <= 0.0):
        raise ValueError("varied parameter scales must be nonzero")
    return PredictorState(
        kind="parameters",
        names=varied_names,
        parameter_names=parameter_names,
        parameter_indices=indices,
        center=central,
        scales=scales,
        training_features=delta_training / scales,
        testing_features=(testing_values[:, indices] - central[indices]) / scales,
        selected_indices=np.empty(0, dtype=int),
        selected_radii=np.empty(0),
        central_values=np.empty(0),
        singular_values=np.empty(0),
    )


def _greedy_maxvol_indices(basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis)
    if basis.ndim != 2 or basis.shape[1] > basis.shape[0]:
        raise ValueError("maxvol basis must have shape (points, modes)")
    selected: list[int] = []
    residual = basis.copy()
    for _ in range(basis.shape[1]):
        norms = np.linalg.norm(residual, axis=1)
        if selected:
            norms[selected] = -np.inf
        index = int(np.argmax(norms))
        selected.append(index)
        chosen = basis[selected]
        coefficients = np.linalg.lstsq(chosen.T, basis.T, rcond=None)[0]
        residual = basis - (chosen.T @ coefficients).T
    return np.asarray(selected, dtype=int)


def build_potential_predictor(
    *,
    radius: np.ndarray,
    central_potential: np.ndarray,
    training_potentials: np.ndarray,
    testing_potentials: np.ndarray,
    predictor_count: int,
    minimum_radius: float = 0.0,
) -> PredictorState:
    """Select informative potential locations with SVD and greedy maxvol."""
    radius = np.asarray(radius, dtype=float)
    central = np.asarray(central_potential)
    training = np.asarray(training_potentials)
    testing = np.asarray(testing_potentials)
    if training.ndim != 2 or testing.ndim != 2 or central.shape != radius.shape:
        raise ValueError("potential arrays must use (samples, radius) shapes")
    allowed = np.flatnonzero(radius >= minimum_radius)
    delta = (training - central[np.newaxis, :]).T
    u, singular_values, _vh = np.linalg.svd(delta[allowed], full_matrices=False)
    if predictor_count < 1 or predictor_count > min(u.shape):
        raise ValueError("predictor_count exceeds the available potential rank")
    local = _greedy_maxvol_indices(u[:, :predictor_count])
    selected = allowed[local]
    raw_training = training[:, selected] - central[selected][np.newaxis, :]
    scales = np.maximum(np.std(raw_training, axis=0), 1e-12)
    return PredictorState(
        kind="potential",
        names=tuple(f"U(r={radius[index]:.8g})" for index in selected),
        parameter_names=(),
        parameter_indices=np.empty(0, dtype=int),
        center=np.empty(0),
        scales=scales,
        training_features=raw_training / scales,
        testing_features=(testing[:, selected] - central[selected][np.newaxis, :]) / scales,
        selected_indices=selected,
        selected_radii=radius[selected],
        central_values=central[selected],
        singular_values=singular_values,
    )


def features_for_values(
    *,
    predictor: PredictorState,
    values: np.ndarray,
    potential_function=None,
) -> np.ndarray:
    """Apply a fitted predictor transformation to new ordered parameter rows."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if predictor.kind == "parameters":
        return (
            values[:, predictor.parameter_indices]
            - predictor.center[predictor.parameter_indices][np.newaxis, :]
        ) / predictor.scales[np.newaxis, :]
    if potential_function is None:
        raise ValueError("potential predictor requires a potential function")
    raw = np.asarray(
        [potential_function(predictor.selected_radii, row) for row in values]
    )
    return (raw - predictor.central_values[np.newaxis, :]) / predictor.scales[np.newaxis, :]
