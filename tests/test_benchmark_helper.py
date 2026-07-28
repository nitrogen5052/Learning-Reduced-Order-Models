import ast
from dataclasses import fields
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
HELPER = NOTEBOOKS / "benchmark_helper.py"
NOTEBOOK_01 = NOTEBOOKS / "01_rbm_vs_lrom_single_wavefunction.ipynb"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))


def helper_functions_and_classes() -> tuple[set[str], set[str]]:
    tree = ast.parse(HELPER.read_text())
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    classes = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    return functions, classes


def notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_benchmark_helper_public_surface() -> None:
    functions, classes = helper_functions_and_classes()
    assert {
        "pointwise_relative_error",
        "summarize_relative_error",
    } <= functions
    assert {
        "RoseWavefunctionResult",
        "LSWavefunctionResult",
        "WavefunctionBenchmarkResult",
        "RoseCrossSectionResult",
        "LSCrossSectionResult",
        "CrossSectionBenchmarkResult",
        "WavefunctionBenchmark",
        "CrossSectionBenchmark",
    } <= classes


def test_result_types_are_flat() -> None:
    import benchmark_helper

    assert [
        field.name
        for field in fields(benchmark_helper.CrossSectionBenchmarkResult)
    ] == ["rose", "ls"]
    assert [
        field.name
        for field in fields(benchmark_helper.WavefunctionBenchmarkResult)
    ] == ["rose", "ls"]


def test_wavefunction_combined_run_delegates(monkeypatch) -> None:
    import benchmark_helper

    benchmark = benchmark_helper.WavefunctionBenchmark(
        emulator=Mock(),
        center={"Vv": 1.0},
    )
    rose_result = Mock()
    ls_result = Mock()
    run_rose = Mock(return_value=rose_result)
    run_ls = Mock(return_value=ls_result)
    monkeypatch.setattr(benchmark, "run_rose", run_rose)
    monkeypatch.setattr(benchmark, "run_ls", run_ls)

    result = benchmark.run(
        basis_size=4,
        eim_basis_size=8,
        include_training_wavefunctions=False,
    )

    run_rose.assert_called_once_with(
        basis_size=4,
        eim_basis_size=8,
        include_training_wavefunctions=False,
    )
    run_ls.assert_called_once_with()
    assert result.rose is rose_result
    assert result.ls is ls_result


def test_cross_section_combined_run_delegates(monkeypatch) -> None:
    import benchmark_helper

    benchmark = benchmark_helper.CrossSectionBenchmark(
        emulator=Mock(),
        training_rows=np.empty((2, 1)),
        testing_rows=np.empty((1, 1)),
        angles_degrees=np.array([30.0]),
        denominator_floor=1e-8,
        timing_repeats=3,
        timing_inner_loops=20,
    )
    rose_result = Mock(
        fom_training_cross_sections=np.ones((2, 1)),
        fom_testing_cross_sections=np.ones((1, 1)),
    )
    ls_result = Mock()
    run_rose = Mock(return_value=rose_result)
    run_ls = Mock(return_value=ls_result)
    monkeypatch.setattr(benchmark, "run_rose", run_rose)
    monkeypatch.setattr(benchmark, "run_ls", run_ls)

    result = benchmark.run(
        basis_sizes=(4, 6),
        eim_sizes=(4, 8),
        ls_predictor_count=4,
    )

    run_rose.assert_called_once_with(
        basis_sizes=(4, 6),
        eim_sizes=(4, 8),
    )
    run_ls.assert_called_once_with(
        basis_sizes=(4, 6),
        fom_training_cross_sections=rose_result.fom_training_cross_sections,
        fom_testing_cross_sections=rose_result.fom_testing_cross_sections,
        predictor_count=4,
    )
    assert result.rose is rose_result
    assert result.ls is ls_result


def test_relative_error_summary_has_one_median_metric() -> None:
    import benchmark_helper

    reference = np.array([[2.0, 4.0]])
    predicted = np.array([[1.0, 6.0]])
    summary = benchmark_helper.summarize_relative_error(
        predicted,
        reference,
        1e-8,
    )

    assert set(summary) == {
        "median_over_angle_error",
        "maximum_over_angle_error",
    }
    np.testing.assert_allclose(
        summary["median_over_angle_error"],
        [0.5],
    )


def test_wavefunction_ls_does_not_construct_rose(monkeypatch) -> None:
    import benchmark_helper

    emulator = Mock()
    emulator.basis = {0: np.eye(2)}
    emulator.samples.training_wavefunctions = {0: np.ones((2, 2))}
    emulator.samples.testing_wavefunctions = {0: np.ones((1, 2))}
    baseline = Mock(
        side_effect=[
            (np.ones((2, 2)), np.ones((2, 2))),
            (np.ones((1, 2)), np.ones((1, 2))),
        ]
    )
    monkeypatch.setattr(
        benchmark_helper.lrom_v1,
        "least_squares_baseline",
        baseline,
    )
    monkeypatch.setattr(
        benchmark_helper.rose,
        "InteractionEIMSpace",
        Mock(side_effect=AssertionError("ROSE constructed during LS")),
    )

    result = benchmark_helper.WavefunctionBenchmark(
        emulator,
        {"Vv": 1.0},
    ).run_ls()

    assert result.testing_wavefunctions.shape == (1, 2)
    assert baseline.call_count == 2


def test_cross_section_ls_does_not_construct_rose(monkeypatch) -> None:
    import benchmark_helper

    benchmark = benchmark_helper.CrossSectionBenchmark(
        emulator=Mock(),
        training_rows=np.empty((2, 1)),
        testing_rows=np.empty((1, 1)),
        angles_degrees=np.array([30.0]),
        denominator_floor=1e-8,
        timing_repeats=3,
        timing_inner_loops=20,
    )
    monkeypatch.setattr(
        benchmark_helper.rose,
        "InteractionEIMSpace",
        Mock(side_effect=AssertionError("ROSE constructed during LS")),
    )
    monkeypatch.setattr(
        benchmark,
        "_run_ls_for_basis",
        Mock(
            return_value={
                "train_xs": np.ones((2, 1)),
                "test_xs": np.ones((1, 1)),
                "train_median_over_angle_error": np.zeros(2),
                "test_median_over_angle_error": np.zeros(1),
                "train_maximum_over_angle_error": np.zeros(2),
                "test_maximum_over_angle_error": np.zeros(1),
            }
        ),
    )

    result = benchmark.run_ls(
        basis_sizes=(4,),
        fom_training_cross_sections=np.ones((2, 1)),
        fom_testing_cross_sections=np.ones((1, 1)),
        predictor_count=4,
    )

    assert set(result.results) == {4}


def test_helper_follows_rose_guide_stage_order() -> None:
    text = HELPER.read_text()
    class_start = text.index("class CrossSectionBenchmark:")
    positions = [
        text.index("def _interaction_space", class_start),
        text.index("def _base_solver", class_start),
        text.index("def _free_reference", class_start),
        text.index("def _custom_bases", class_start),
        text.index("def _scattering_emulator", class_start),
        text.index("def _exact_smatrix", class_start),
        text.index("def _cross_section", class_start),
    ]
    assert positions == sorted(positions)
