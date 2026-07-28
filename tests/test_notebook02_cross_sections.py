import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_02 = ROOT / "notebooks" / "02_rose_vs_lrom_cross_sections.ipynb"
BENCHMARK_03 = (
    ROOT / "notebooks" / "benchmark_notebooks" / "2.0" / "benchmark_03.ipynb"
)

HEADINGS = (
    "# 02. ROSE and LROM Cross-Section Comparison",
    "## 1. Physical Problem and Optical-Potential Parameters",
    "## 2. Shared Training and Testing Samples",
    "## 3. Potential Variation and LROM Predictor Locations",
    "## 4. Equal-Basis ROSE and LROM Emulators",
    "## 5. Representative Cross-Section Predictions",
    "## 6. Cross-Section Error Distributions",
    "## 7. Basis and Operator-Size Comparison",
    "## 8. Accuracy Versus Online Time",
    "## 9. Validation Summary",
)


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def notebook_text(path: Path) -> str:
    notebook = load_notebook(path)
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def code_sources(path: Path) -> list[str]:
    notebook = load_notebook(path)
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def notebook_functions(path: Path) -> dict[str, ast.FunctionDef]:
    functions = {}
    for source in code_sources(path):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node
    return functions


def test_notebook02_clean_shell_contract() -> None:
    sources = code_sources(NOTEBOOK_02)
    for source in sources[1:]:
        assert not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            for node in ast.walk(ast.parse(source))
        )

    calls = [
        node
        for source in sources
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
    ]
    assert not any(
        isinstance(call.func, ast.Name) and call.func.id == "print"
        for call in calls
    )
    assert not any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "head"
        for call in calls
    )

    text = notebook_text(NOTEBOOK_02)
    assert "rose_train_rows" not in text
    assert "rose_test_rows" not in text

    expected = {
        "parameter_dicts",
        "pointwise_relative_error",
        "summarize_relative_error",
        "time_lrom_predictions",
        "time_rose_predictions",
        "build_sampled_study",
    }
    functions = notebook_functions(NOTEBOOK_02)
    assert expected <= functions.keys()
    for name in expected:
        function = functions[name]
        assert function.returns is not None
        assert all(
            argument.annotation is not None
            for argument in function.args.args
        )


def test_notebook02_experiment_functions_are_extracted() -> None:
    expected = {
        "build_rose_bases",
        "build_rose_emulator",
        "build_rose_emulators",
        "exact_smatrix_all_channels",
        "emulated_smatrix_all_channels",
        "cross_section_from_smatrix",
        "evaluate_fom_cross_sections",
        "evaluate_old_lrom",
        "evaluate_ls_oracle",
        "evaluate_lrom_grid",
        "evaluate_rose_grid",
    }
    functions = notebook_functions(NOTEBOOK_02)
    assert expected <= functions.keys()
    for name in expected:
        function = functions[name]
        assert function.returns is not None
        assert all(
            argument.annotation is not None
            for argument in function.args.args
        )


def test_notebook02_presentation_functions_are_extracted() -> None:
    expected = {
        "predictor_radius_figure",
        "select_alpha_cases",
        "representative_cross_section_figure",
        "cross_section_error_figure",
        "error_distribution_figure",
        "summary_results_table",
        "accuracy_time_figure",
        "validation_results_table",
    }
    functions = notebook_functions(NOTEBOOK_02)
    assert expected <= functions.keys()
    for name in expected:
        function = functions[name]
        assert function.returns is not None
        assert all(
            argument.annotation is not None
            for argument in function.args.args
        )


def test_notebook02_shell_contract() -> None:
    assert NOTEBOOK_02.exists()
    notebook = load_notebook(NOTEBOOK_02)
    markdown = tuple(
        "".join(cell["source"]).strip()
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert markdown == HEADINGS
    text = notebook_text(NOTEBOOK_02)
    for heading in HEADINGS:
        assert heading in text
    assert "import lrom_legacy.v2_0 as lrom" in text
    assert "TARGET = (40, 20)" in text
    assert "PROJECTILE = (1, 0)" in text
    assert "LAB_ENERGY = 14.1" in text
    assert "L_MAX = 3" in text
    assert "MESH_SIZE = 600" in text
    assert "N_TRAIN = 200" in text
    assert "N_TEST = 100" in text
    assert "HALF_WIDTH = 0.20" in text
    assert "BASIS_SIZES = (4, 6, 8)" in text
    assert "ROSE_EIM_SIZES = (4, 8, 12)" in text
    assert "LROM_PREDICTOR_COUNTS = (4, 8, 12)" in text
    assert "SEED = 1204" in text


def test_notebook02_code_cells_compile() -> None:
    for index, source in enumerate(code_sources(NOTEBOOK_02)):
        compile(source, f"{NOTEBOOK_02.name} code cell {index}", "exec")


def test_notebook02_scientific_core_contract() -> None:
    text = notebook_text(NOTEBOOK_02)
    assert 'potential="full_woods-saxon"' in text
    assert "l=tuple(range(l_max + 1))" in text
    assert 'strategy="latin_hypercube"' in text
    assert "seed=seed" in text
    assert "rose.InteractionEIMSpace(" in text
    assert "training_info=training_rows" in text
    assert "explicit_training=True" in text
    assert "rose.basis.CustomBasis(" in text
    assert "solutions=np.asarray(" in text
    assert "free_reference = np.asarray(" in text
    assert "def exact_smatrix_all_channels" in text
    assert "def emulated_smatrix_all_channels" in text
    assert "def cross_section_from_smatrix" in text
    assert "exact_dsdo" not in text
    assert "emulate_dsdo" not in text
    assert "lrom.project_coordinates(" in text
    assert "lrom._cross_section_prediction(" in text
    assert 'predictor="effective-interaction"' in text
    assert "reconstruct_wavefunctions=False" in text
    assert "summarize_relative_error" in text
    assert "median_over_angle_error" in text
    assert "maximum_over_angle_error" in text
    assert "old_v2_results" in text
    assert "archive_lrom_results" in text
    assert "LS-projected cross section" in text
    assert "linear LROM" not in text
    assert "np.median(pointwise, axis=1)" in text
    assert "np.max(pointwise, axis=1)" in text


def test_notebook02_results_contract() -> None:
    text = notebook_text(NOTEBOOK_02)
    for marker in (
        "potential-predictor-rainbows",
        "representative-cross-sections",
        "cross-section-errors",
        "error-violins",
        "cat-plot",
        "validation-summary",
    ):
        assert f"# FIGURE: {marker}" in text or f"# TABLE: {marker}" in text
    assert "selected_radii" in text
    assert "radius_mesh" in text
    assert "selected_radii >= 0.5" in text
    assert "def select_alpha_cases" in text
    assert "alpha selection A" in text
    assert "alpha selection B" in text
    assert "alpha selection C" in text
    assert "percentile" not in text.lower()
    assert "alpha_selection_table" in text
    assert "display(alpha_cases)" in text
    assert "one million evaluations/hour" in text
    assert 'label=f"{error_reference:.2f} median pointwise error"' in text
    assert "axes[0].set_ylim(bottom=plotting_floor)" in text
    assert "marker_sizes = dict(zip(compression_values, (16, 28, 44)))" in text
    assert "bbox_to_anchor=(1.02, 1.0)" in text
    for function_name in (
        "predictor_radius_figure",
        "representative_cross_section_figure",
        "cross_section_error_figure",
        "error_distribution_figure",
        "accuracy_time_figure",
    ):
        assert f"def {function_name}" in text


def test_notebook_timing_reuses_initialized_online_paths() -> None:
    for path in (NOTEBOOK_02, BENCHMARK_03):
        text = notebook_text(path)
        assert "TIMING_REPEATS = 3" in text
        assert "TIMING_INNER_LOOPS = 20" in text
        assert "time.perf_counter_ns()" in text
        assert "sae.calculate_xs(splus, sminus, parameters)" in text
        assert (
            "sae.calculate_xs(\n"
            "        splus,\n"
            "        sminus,\n"
            "        row,\n"
            "        angles=ANGLES_RAD"
        ) not in text

    notebook_text_02 = notebook_text(NOTEBOOK_02)
    assert "1e9 * inner_loops" in notebook_text_02
    assert "L_MAX = 3" in notebook_text_02
    assert "LROM_BENCHMARK_L_MAX" not in notebook_text_02
    assert "1e9 * TIMING_INNER_LOOPS" in notebook_text(BENCHMARK_03)


def test_benchmark03_notebook02_profile_contract() -> None:
    assert BENCHMARK_03.exists()
    text = notebook_text(BENCHMARK_03)
    assert 'os.environ.get("LROM_BENCHMARK_PROFILE", "reduced")' in text
    assert '"reduced": (120, 30)' in text
    assert '"full": (200, 100)' in text
    assert "HALF_WIDTH = 0.20" in text
    assert "LROM_BENCHMARK_L_MAX" in text
    assert 'L_MAX = int(os.environ.get("LROM_BENCHMARK_L_MAX", "3"))' in text
    assert "if L_MAX not in (3, 10):" in text
    assert "BASIS_SIZES = (4, 6, 8)" in text
    assert "ROSE_EIM_SIZES = (4, 8, 12)" in text
    assert "LROM_PREDICTOR_COUNTS = (4, 8, 12)" in text
    assert "training_info=rose_train_rows" in text
    assert "explicit_training=True" in text
    assert "rose.basis.CustomBasis(" in text
    assert "def exact_smatrix_all_channels" in text
    assert "def emulated_smatrix_all_channels" in text
    assert "def cross_section_from_smatrix" in text
    assert "exact_dsdo" not in text
    assert "emulate_dsdo" not in text
    assert 'predictor="effective-interaction"' in text
    assert "reconstruct_wavefunctions=False" in text
    assert "median_pointwise_relative_error" in text
    assert "maximum_over_angle_relative_error" in text
    assert "old_v2_results" in text
    assert "archive_lrom_results" in text
    assert "selected_radii >= 0.5" in text
    assert "axes[0].set_ylim(bottom=PLOTTING_FLOOR)" in text
    assert "compression_sizes = {4: 16, 8: 28, 12: 44}" in text
    assert "np.max(pointwise_relative_error" in text
    assert "test_seconds" in text
    assert "LS-projected cross section" in text
    assert "alpha selection A" in text
    assert "alpha selection B" in text
    assert "alpha selection C" in text
    assert "percentile" not in text.lower()
    assert "display(alpha_selection_table)" in text
    assert "linear LROM" not in text


def test_benchmark03_code_cells_compile() -> None:
    for index, source in enumerate(code_sources(BENCHMARK_03)):
        compile(source, f"{BENCHMARK_03.name} code cell {index}", "exec")
