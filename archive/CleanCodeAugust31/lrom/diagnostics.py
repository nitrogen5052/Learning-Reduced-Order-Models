"""Numerical diagnostics shared by the demonstration notebooks."""

from __future__ import annotations

import numpy as np

from .data import TrainingData
from .emulator import ScatteringLROM
from .fom import s_matrices_from_scaled_wavefunctions
from .observables import assemble_varying_energy_cross_sections


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return one relative L2 error for each leading sample."""
    prediction = np.atleast_2d(prediction)
    reference = np.atleast_2d(reference)
    return np.linalg.norm(prediction - reference, axis=1) / np.maximum(
        np.linalg.norm(reference, axis=1), 1.0e-30
    )


def absolute_l2(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return one absolute L2 error for each leading sample."""
    return np.linalg.norm(np.atleast_2d(prediction) - np.atleast_2d(reference), axis=1)


def pointwise_relative_error(
    prediction: np.ndarray,
    reference: np.ndarray,
    denominator_floor: float = 1.0e-12,
) -> np.ndarray:
    """Return absolute pointwise errors normalized by the reference magnitude.

    Parameters
    ----------
    prediction, reference
        Arrays of matching shape containing predicted and reference values.
    denominator_floor
        Positive lower bound applied to the reference magnitude.

    Returns
    -------
    np.ndarray
        Dimensionless pointwise relative errors with the input shape.
    """
    if denominator_floor <= 0:
        raise ValueError("denominator_floor must be positive")
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must have matching shapes")
    denominator = np.maximum(np.abs(reference), denominator_floor)
    return np.abs(prediction - reference) / denominator


def projected_cross_sections(
    emulator: ScatteringLROM,
    data: TrainingData,
    angles: np.ndarray,
) -> np.ndarray:
    """Return the least-squares reduced-basis cross-section floor.

    Every FOM wavefunction is projected onto the trained channel basis.  The
    projected boundary values are converted to S matrices and then to elastic
    cross sections using each sample's own wave number.  The result isolates
    basis truncation from the learned implicit-coordinate error.
    """
    s_by_channel = {}
    for channel in data.channels:
        model = emulator.channel_models[channel]
        scaled_waves = emulator.basis_config.scale(
            data.wavefunctions[channel], data.s_mesh
        )
        coordinates = model.learned.basis.project(scaled_waves)
        projected = model.learned.basis.reconstruct(coordinates)
        s_by_channel[channel] = s_matrices_from_scaled_wavefunctions(
            data.s_mesh,
            projected,
            channel[0],
            matching_s=emulator.problem.matching_s,
        )
    wave_numbers = np.asarray([
        emulator.problem.system(sample).wave_number for sample in data.samples
    ])
    return assemble_varying_energy_cross_sections(
        s_by_channel, angles, wave_numbers
    )
