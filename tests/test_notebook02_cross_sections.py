import ast
import io
import json
from pathlib import Path
import tokenize


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_02 = ROOT / "notebooks" / "02_rose_vs_lrom_cross_sections.ipynb"
BENCHMARK_HELPER = ROOT / "notebooks" / "benchmark_helper.py"
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


def cat_source(path: Path) -> str:
    matches = [
        source
        for source in code_sources(path)
        if "Computational Accuracy versus Time" in source
    ]
    assert len(matches) == 1
    return matches[0]


def test_notebook02_clean_shell_contract() -> None:
    sources = code_sources(NOTEBOOK_02)
    for source in sources[1:]:
        assert not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            for node in ast.walk(ast.parse(source))
        )

    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for source in sources
        for node in ast.walk(ast.parse(source))
    )
    assert not any(
        token.type == tokenize.COMMENT
        for source in sources
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
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
    for removed_import in (
        "from typing import Any",
        "import platform",
        "from matplotlib.figure import Figure",
        "from matplotlib.lines import Line2D",
        "from matplotlib.patches import Patch",
        "from scipy.special import",
    ):
        assert removed_import not in text
    for retained_import in (
        "from pathlib import Path",
        "import sys",
        "import time",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import pandas as pd",
        "import benchmark_helper",
        "import lrom_legacy.v2_0 as lrom",
    ):
        assert retained_import in sources[0]


def test_notebook02_uses_benchmark_helper_boundary() -> None:
    text = notebook_text(NOTEBOOK_02)
    assert "import benchmark_helper" in text
    assert "CrossSectionBenchmark(" in text
    assert ".run_rose(" in text
    assert ".run_ls(" in text
    for implementation_detail in (
        "rose.InteractionEIMSpace(",
        "rose.basis.CustomBasis(",
        "def build_rose_bases",
        "def build_rose_emulator",
        "def build_rose_emulators",
        "def exact_smatrix_all_channels",
        "def emulated_smatrix_all_channels",
        "def cross_section_from_smatrix",
        "def evaluate_fom_cross_sections",
        "def evaluate_ls_oracle",
        "def evaluate_rose_grid",
        "lrom.project_coordinates(",
        "lrom._cross_section_prediction(",
    ):
        assert implementation_detail not in text


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
    helper = BENCHMARK_HELPER.read_text()
    assert 'potential="full_woods-saxon"' in text
    assert "l=tuple(range(L_MAX + 1))" in text
    assert 'strategy="latin_hypercube"' in text
    assert "seed=SEED" in text
    assert "rose.InteractionEIMSpace(" in helper
    assert "training_info=self.training_rows" in helper
    assert "explicit_training=True" in helper
    assert "rose.basis.CustomBasis(" in helper
    assert "solutions=np.asarray(" in helper
    assert "def _free_reference" in helper
    assert "def _exact_smatrix" in helper
    assert "def _emulated_smatrix" in helper
    assert "def _cross_section" in helper
    assert "exact_dsdo" not in text
    assert "emulate_dsdo" not in text
    assert "lrom_v2.project_coordinates(" in helper
    assert "lrom_v2._cross_section_prediction(" in helper
    assert 'predictor="effective-interaction"' in text
    assert "reconstruct_wavefunctions=False" in text
    assert "summarize_relative_error" in text
    assert "median_over_angle_error" in text
    assert "maximum_over_angle_error" in text
    assert "old_v2_results" in text
    assert "archive_lrom_results" in text
    assert "LS-projected cross section" in text
    assert "linear LROM" not in text
    assert "np.median(pointwise, axis=1)" in helper
    assert "np.max(pointwise, axis=1)" in helper


def test_notebook02_results_contract() -> None:
    text = notebook_text(NOTEBOOK_02)
    assert "selected_radii" in text
    assert "radius_mesh" in text
    assert "selected_radii >= 0.5" in text
    assert "alpha selection A" in text
    assert "alpha selection B" in text
    assert "alpha selection C" in text
    assert "percentile" not in text.lower()
    assert "display(alpha_cases)" in text
    assert "one million evaluations/hour" in text
    assert "median pointwise error" in text
    assert "axes[0].set_ylim(bottom=PLOTTING_FLOOR)" in text
    assert "compression_sizes = dict(zip(compression_values, (16, 28, 44)))" in text
    assert "bbox_to_anchor=(1.02, 1.0)" in text
    assert 'set_title("Computational Accuracy versus Time")' in text
    assert 'set_ylabel("maximum relative error over angle")' in text


def test_notebook02_cat_uses_per_alpha_paper_encoding() -> None:
    source = cat_source(NOTEBOOK_02)
    assert 'result["test_seconds"]' in source
    assert 'result["test_maximum_over_angle_error"]' in source
    assert 'marker="s"' in source
    assert 'marker="o"' in source
    assert "color=colors[n_phi]" in source
    assert 'facecolors="none"' not in source
    assert "median_seconds" not in source
    assert "median_error" not in source
    assert 'set_ylabel("maximum relative error over angle")' in source
    assert 'label="ROSE"' in source
    assert 'label="LROM"' in source


def test_notebook_timing_reuses_initialized_online_paths() -> None:
    for path in (NOTEBOOK_02, BENCHMARK_03):
        text = notebook_text(path)
        assert "TIMING_REPEATS = 3" in text
        assert "TIMING_INNER_LOOPS = 20" in text
        assert "time.perf_counter_ns()" in text
        assert (
            "sae.calculate_xs(\n"
            "        splus,\n"
            "        sminus,\n"
            "        row,\n"
            "        angles=ANGLES_RAD"
        ) not in text

    notebook_text_02 = notebook_text(NOTEBOOK_02)
    assert "emulator.calculate_xs(splus, sminus, row)" in (
        BENCHMARK_HELPER.read_text()
    )
    assert "1e9 * TIMING_INNER_LOOPS" in notebook_text_02
    assert "L_MAX = 3" in notebook_text_02
    assert "LROM_BENCHMARK_L_MAX" not in notebook_text_02
    assert "1e9 * inner_loops" in notebook_text(BENCHMARK_03)


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
    assert "exact_dsdo" not in text
    assert "emulate_dsdo" not in text
    assert 'predictor="effective-interaction"' in text
    assert "reconstruct_wavefunctions=False" in text
    assert "summarize_relative_error" in text
    assert "median_over_angle_error" in text
    assert "maximum_over_angle_error" in text
    assert "old_v2_results" in text
    assert "archive_lrom_results" in text
    assert "selected_radii >= 0.5" in text
    assert 'f"l={L_MAX}, j=l-1/2: real"' in text
    assert 'f"l={L_MAX}, j=l+1/2: real"' in text
    assert "axes[0].set_ylim(bottom=PLOTTING_FLOOR)" in text
    assert "compression_sizes = {4: 16, 8: 28, 12: 44}" in text
    assert "test_seconds" in text
    assert "LS-projected cross section" in text
    assert "alpha selection A" in text
    assert "alpha selection B" in text
    assert "alpha selection C" in text
    assert "percentile" not in text.lower()
    assert "display(alpha_cases)" in text
    assert "linear LROM" not in text


def test_benchmark03_uses_benchmark_helper_boundary() -> None:
    text = notebook_text(BENCHMARK_03)
    assert 'NOTEBOOKS = ROOT / "notebooks"' in text
    assert "import benchmark_helper" in text
    assert "CrossSectionBenchmark(" in text
    assert ".run_rose(" in text
    assert ".run_ls(" in text
    for implementation_detail in (
        "rose.InteractionEIMSpace(",
        "rose.basis.CustomBasis(",
        "def build_rose_bases",
        "def build_rose_emulator",
        "def exact_smatrix_all_channels",
        "def emulated_smatrix_all_channels",
        "def cross_section_from_smatrix",
        "lrom.project_coordinates(",
        "lrom._cross_section_prediction(",
    ):
        assert implementation_detail not in text


def test_benchmark03_code_cells_compile() -> None:
    for index, source in enumerate(code_sources(BENCHMARK_03)):
        compile(source, f"{BENCHMARK_03.name} code cell {index}", "exec")
