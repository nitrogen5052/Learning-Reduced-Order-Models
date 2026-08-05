import ast
import json
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_02 = ROOT / "notebooks" / "02_rose_vs_lrom_cross_sections.ipynb"
BENCHMARK_HELPER = ROOT / "notebooks" / "benchmark_helper.py"
BENCHMARK_03 = (
    ROOT / "notebooks" / "benchmark_notebooks" / "2.0" / "benchmark_03.ipynb"
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
    markdown = notebook_text(NOTEBOOK_02)
    required_headings = (
        "# 02. ROSE and LROM Cross Section Comparison",
        "## 1. System Setup",
        "## 2. Initializaton: Training and Testing Samples",
        "## 3. LROM Predictor Locations",
        "## 4. ROSE, LS, and LROM Emulator pipeline",
        "## 5. Cross Section Test Cases: examples/performance",
        "## 6. Cross-Section Error Distributions",
        "## 7. CAT Plot",
    )
    for heading in required_headings:
        assert heading in markdown
    text = markdown
    assert "import lrom_legacy.v2_0 as lrom" in text
    assert "TARGET = (40, 20)" in text
    assert "PROJECTILE = (1, 0)" in text
    assert "LAB_ENERGY = 14.1" in text
    assert "MESH_SIZE = 600" in text
    assert "N_TRAIN = 200" in text
    assert "N_TEST = 100" in text
    assert "HALF_WIDTH = 0.20" in text
    assert "BASIS_SIZES = (4, 6, 8)" in text
    assert "ROSE_EIM_SIZES = (4, 8, 12)" in text
    assert "LROM_PREDICTOR_COUNTS = (4, 8, 12)" in text
    assert "SEED = 1204" in text


def test_notebook02_exfor_testing_envelope_contract() -> None:
    text = notebook_text(NOTEBOOK_02)
    sources = code_sources(NOTEBOOK_02)
    envelope_sources = [
        source
        for source in sources
        if "exfor_cos_cm" in source and "fill_between" in source
    ]

    assert len(envelope_sources) == 1
    source = envelope_sources[0]
    assert "L_MAX = 10" in text
    assert "exfor_cos_cm" in source
    assert "exfor_data_cm" in source
    assert "exfor_data_error" in source
    assert "np.degrees(np.arccos(exfor_cos_cm))" in source
    assert source.count("fill_between(") == 4
    assert "fom_test_xs" in source
    assert 'ls_results[DEFAULT_BASIS_SIZE]["test_xs"]' in source
    assert 'rose_results[default_key]["test_xs"]' in source
    assert 'lrom_results[default_key]["test_xs"]' in source
    assert "np.min(" in source
    assert "np.max(" in source
    assert "np.median(" in source
    assert source.count("errorbar(") == 2
    assert 'set_yscale("log")' in source
    assert r"$\theta$ [deg]" in source
    assert "mb/sr" in source
    assert "testing-design envelope" in text
    assert "11-degree angular resolution" in text
    assert "possible correlated 10% normalization" in text
    assert "0.88 b" in text
    assert "requests." not in source
    assert "urllib" not in source

    module = ast.parse(source)
    literal_arrays = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in {
            "exfor_cos_cm",
            "exfor_data_cm",
            "exfor_data_error",
        }:
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "np"
            and value.func.attr == "array"
            and value.args
        ):
            value = value.args[0]
        assert isinstance(value, (ast.List, ast.Tuple))
        literal_arrays[target.id] = value.elts
    assert set(literal_arrays) == {
        "exfor_cos_cm",
        "exfor_data_cm",
        "exfor_data_error",
    }
    assert all(len(values) == 14 for values in literal_arrays.values())
    assert "0.93420" in source
    assert "-0.9342" in source
    assert "5.34e2" in source
    assert "7.4" in source
    assert "2.3e1" in source
    assert "1.9" in source
    assert "11611003" in text
    assert "www-nds.iaea.org/exfor" in text

    representative_index = next(
        index
        for index, source in enumerate(sources)
        if 'label="FOM"' in source and "selected_indices" in source
    )
    envelope_index = sources.index(source)
    error_index = next(
        index for index, item in enumerate(sources) if "error_methods =" in item
    )
    assert representative_index < envelope_index < error_index


def test_notebook02_exfor_figure_presentation(monkeypatch) -> None:
    source = next(
        source
        for source in code_sources(NOTEBOOK_02)
        if "exfor_cos_cm" in source and "fill_between" in source
    )
    angles = np.arange(1.0, 180.0)
    scale = np.linspace(0.8, 1.2, 100)[:, None]
    cross_sections = scale * np.linspace(1.0, 100.0, angles.size)[None, :]
    namespace = {
        "np": np,
        "plt": plt,
        "ANGLES_DEG": angles,
        "N_TEST": 100,
        "fom_test_xs": cross_sections,
        "DEFAULT_BASIS_SIZE": 6,
        "default_key": (6, 8),
        "LS_LABEL": "LS-projected cross section",
        "ls_results": {6: {"test_xs": cross_sections}},
        "rose_results": {(6, 8): {"test_xs": 1.01 * cross_sections}},
        "lrom_results": {(6, 8): {"test_xs": 0.99 * cross_sections}},
    }

    monkeypatch.setattr(plt, "show", lambda: None)
    exec(compile(source, NOTEBOOK_02.name, "exec"), namespace)
    figure = namespace["fig"]
    axes = namespace["axes"]
    try:
        assert [axis.get_title() for axis in axes] == [
            "FOM versus data",
            "Emulators versus data",
        ]
        assert [axis.get_xlabel() for axis in axes] == [
            r"$\theta$ [deg]",
            r"$\theta$ [deg]",
        ]
        assert [text.get_text() for text in axes[0].get_legend().get_texts()] == [
            "FOM",
            "FOM median",
            "EXFOR data",
        ]
        assert [text.get_text() for text in axes[1].get_legend().get_texts()] == [
            "LS",
            "LS median",
            "ROSE",
            "ROSE median",
            "LROM",
            "LROM median",
            "EXFOR data",
        ]
        fill_hatches = {
            artist.get_label(): artist.get_hatch()
            for artist in axes[1].collections
            if artist.get_label() in {"LS", "ROSE", "LROM"}
        }
        assert fill_hatches == {"LS": "/", "ROSE": "\\", "LROM": "."}
    finally:
        plt.close(figure)


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
    for method in ("old v2", "LS", "ROSE", "archive LROM"):
        assert f"{method} default median pointwise error" in text
        assert f"{method} default median maximum-over-angle error" in text
    for method in ("old v2", "ROSE", "archive LROM"):
        assert f"{method} default median online time [s]" in text
    assert "LS default online time" in text
    assert "not applicable: held-out-wavefunction oracle" in text


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
