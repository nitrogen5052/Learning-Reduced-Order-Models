"""Scientific-invariant tests for the Notebook 02 helper boundary."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


NOTEBOOK_DIRECTORY = Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOK_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIRECTORY))

import benchmark_helper


def test_partial_wave_topology_accepts_inclusive_spin_channels() -> None:
    rows = [[object()], [object(), object()], [object(), object()]]

    counts = benchmark_helper.validate_partial_wave_topology(rows, l_max=2)

    assert counts == (1, 2, 2)


@pytest.mark.parametrize(
    ("rows", "l_max", "message"),
    [
        ([[object()]], 1, "inclusive"),
        ([[object(), object()], [object(), object()]], 1, "ell=0"),
        ([[object()], [object()]], 1, "ell=1"),
    ],
)
def test_partial_wave_topology_rejects_missing_or_wrong_channels(
    rows: list[list[object]],
    l_max: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark_helper.validate_partial_wave_topology(rows, l_max=l_max)


def predictor_state(*radii: float) -> SimpleNamespace:
    return SimpleNamespace(selected_radii=np.asarray(radii, dtype=float))


def test_predictor_radii_accept_exact_channel_parity() -> None:
    displayed = {0: predictor_state(0.5, 1.25), (1, 0): predictor_state(0.75, 2.0)}
    trained = {0: predictor_state(0.5, 1.25), (1, 0): predictor_state(0.75, 2.0)}

    radii = benchmark_helper.validate_predictor_radii(displayed, trained)

    assert set(radii) == {0, (1, 0)}
    np.testing.assert_array_equal(radii[(1, 0)], np.array([0.75, 2.0]))


def test_predictor_radii_reject_channel_mismatch() -> None:
    with pytest.raises(ValueError, match="channel keys"):
        benchmark_helper.validate_predictor_radii(
            {0: predictor_state(0.5)},
            {1: predictor_state(0.5)},
        )


def test_predictor_radii_reject_radius_mismatch() -> None:
    with pytest.raises(ValueError, match="selected radii"):
        benchmark_helper.validate_predictor_radii(
            {0: predictor_state(0.5, 1.25)},
            {0: predictor_state(0.5, 1.2501)},
        )
