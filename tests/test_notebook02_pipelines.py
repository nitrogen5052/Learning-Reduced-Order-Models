"""Pipeline-boundary tests for Notebook 02 cross-section studies."""

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


class DeterministicEmulator:
    """Small emulator double that exercises the real study orchestration."""

    def __init__(self) -> None:
        self.predictions = SimpleNamespace(
            cross_sections=SimpleNamespace(values=np.empty((0, 2)))
        )
        self.predictors = {0: SimpleNamespace(selected_radii=np.array([0.5]))}
        self.train_calls: list[dict[str, object]] = []

    def train(self, **kwargs: object) -> None:
        self.train_calls.append(dict(kwargs))

    def predict(
        self,
        *,
        parameters: dict[str, float] | list[dict[str, float]],
        reconstruct_wavefunctions: bool,
    ) -> None:
        del reconstruct_wavefunctions
        cases = [parameters] if isinstance(parameters, dict) else parameters
        self.predictions.cross_sections.values = np.asarray(
            [[case["x"], 2.0 * case["x"]] for case in cases],
            dtype=float,
        )


def test_cross_section_summary_keeps_raw_errors_and_optional_times() -> None:
    predicted = np.array([[2.0, 3.0], [4.0, 10.0]])
    reference = np.array([[1.0, 3.0], [2.0, 5.0]])

    result = benchmark_helper.cross_section_result(
        predicted,
        predicted,
        reference,
        reference,
        denominator_floor=1e-12,
        test_seconds=np.array([1e-4, 2e-4]),
    )

    np.testing.assert_array_equal(result["test_xs"], predicted)
    np.testing.assert_allclose(
        result["test_median_over_angle_error"],
        np.array([0.5, 1.0]),
    )
    np.testing.assert_array_equal(result["test_seconds"], np.array([1e-4, 2e-4]))


def test_configuration_grid_validation_requires_every_pair() -> None:
    complete = {(4, 4): {}, (4, 8): {}, (6, 4): {}, (6, 8): {}}

    benchmark_helper.validate_configuration_grid(
        complete,
        basis_sizes=(4, 6),
        compression_sizes=(4, 8),
        label="LROM",
    )

    with pytest.raises(ValueError, match="LROM configuration grid"):
        benchmark_helper.validate_configuration_grid(
            {(4, 4): {}},
            basis_sizes=(4, 6),
            compression_sizes=(4, 8),
            label="LROM",
        )


def test_lrom_study_owns_training_prediction_and_error_summaries() -> None:
    emulator = DeterministicEmulator()
    training_cases = [{"x": 1.0}, {"x": 2.0}]
    testing_cases = [{"x": 3.0}, {"x": 4.0}]
    training_reference = np.array([[1.0, 2.0], [2.0, 4.0]])
    testing_reference = np.array([[3.0, 6.0], [4.0, 8.0]])
    study = benchmark_helper.LromCrossSectionStudy(
        emulator=emulator,
        training_cases=training_cases,
        testing_cases=testing_cases,
        training_rows=np.array([[1.0], [2.0]]),
        testing_rows=np.array([[3.0], [4.0]]),
        fom_training_cross_sections=training_reference,
        fom_testing_cross_sections=testing_reference,
        angles_degrees=np.array([10.0, 20.0]),
        denominator_floor=1e-12,
        timing_repeats=1,
        timing_inner_loops=1,
    )

    output = study.run_lrom(
        basis_sizes=(4,),
        operator_counts=(2,),
        capture_predictors_at=(4, 2),
    )

    assert set(output.results) == {(4, 2)}
    assert output.predictors is emulator.predictors
    np.testing.assert_array_equal(output.results[(4, 2)]["test_xs"], testing_reference)
    np.testing.assert_array_equal(
        output.results[(4, 2)]["test_median_over_angle_error"],
        np.zeros(2),
    )
    assert output.results[(4, 2)]["test_seconds"].shape == (2,)
    assert emulator.train_calls == [
        {
            "basis_size": 4,
            "predictor": "effective-interaction",
            "predictor_count": 2,
            "observable": "cross_section",
            "angles_degrees": study.angles_degrees,
        }
    ]


def test_rose_pipeline_has_no_lrom_training_entrypoint() -> None:
    assert hasattr(benchmark_helper.RoseCrossSectionPipeline, "run")
    assert not hasattr(benchmark_helper.RoseCrossSectionPipeline, "run_ls")
    assert not hasattr(benchmark_helper.RoseCrossSectionPipeline, "run_lrom")
