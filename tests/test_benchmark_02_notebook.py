from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "benchmark_notebooks" / "1.0" / "benchmark_02.ipynb"
SOURCE_NOTEBOOK = (
    "scientific_archive/legacy_code/Legacy_benchmark/notebooks/"
    "02_lrom_method_walkthrough.ipynb"
)
FIGURES = {
    "vv-rainbow",
    "vv-coefficients",
    "vv-wavefunction-errors",
    "rv-rainbow",
    "rv-coefficients",
    "rv-wavefunction-errors",
    "broad-coefficients",
    "broad-coefficient-errors",
    "broad-wavefunction-errors",
    "predictor-points",
    "predictor-rainbows",
    "predictor-values",
    "multiparameter-coefficients",
    "multiparameter-coefficient-errors",
    "multiparameter-wavefunction-errors",
    "multiparameter-violins",
}


def notebook_text() -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


def test_benchmark_02_notebook_contract() -> None:
    assert NOTEBOOK.exists()
    text = notebook_text()
    assert SOURCE_NOTEBOOK in text
    assert "import rose" in text
    assert "import lrom" in text
    assert "import lrom_legacy" not in text
    assert "eim_basis_size" not in text
    assert 'high_fidelity_solver="runge_kutta"' in text
    assert "import lrom_bench" not in text
    assert "lrom_demo" not in text
    assert "def plot_" not in text
    assert "def split_violin" not in text
    assert "def build_rose_comparison" not in text
    assert "rose.ScatteringAmplitudeEmulator.from_train" in text
    assert "ROSE free-basis LS" in text
    assert "LROM central-basis LS" in text
    for figure in FIGURES:
        assert f"# FIGURE: {figure}" in text


def test_rose_benchmark_follows_the_heldout_tutorial_workflow() -> None:
    text = notebook_text()
    assert "from scipy.stats import qmc" in text
    assert "ROSE_EIM_CANDIDATES = (4, 8, 12)" in text
    assert "ROSE_VALIDATION_SIZE = 20" in text
    assert "ROSE_TIMING_REPEATS = 3" in text
    assert text.count("qmc.LatinHypercube") == 3
    assert text.count("for n_u in ROSE_EIM_CANDIDATES:") == 3
    assert text.count(".exact_wave_functions(") == 3
    assert text.count(".emulate_wave_functions(") >= 6
    assert text.count("time.perf_counter()") >= 6
    assert text.count("n_basis=BASIS_SIZE") >= 3
    assert text.count("selected ROSE configuration") == 3
    assert "def build_rose" not in text
    assert "def validate_rose" not in text
    assert "CATPerformance" not in text


def test_benchmark_02_code_cells_compile() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"benchmark_02 cell {index}", "exec")


def test_difference_figures_use_the_1e_minus_5_display_floor() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join(cell.source for cell in notebook.cells)
    assert "DIFFERENCE_FLOOR = 1e-5" in text
    markers = {
        "vv-coefficients",
        "vv-wavefunction-errors",
        "rv-coefficients",
        "rv-wavefunction-errors",
        "broad-coefficient-errors",
        "broad-wavefunction-errors",
        "multiparameter-coefficient-errors",
        "multiparameter-wavefunction-errors",
    }
    for marker in markers:
        cell = next(
            cell
            for cell in notebook.cells
            if f"# FIGURE: {marker}" in cell.source
        )
        assert "DIFFERENCE_FLOOR" in cell.source
        assert "bottom=DIFFERENCE_FLOOR" in cell.source
