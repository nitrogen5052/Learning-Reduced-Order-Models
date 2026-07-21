from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_01 = ROOT / "notebooks" / "benchmark_notebooks" / "2.0" / "benchmark_01.ipynb"
BENCHMARK_03 = ROOT / "notebooks" / "benchmark_notebooks" / "2.0" / "benchmark_03.ipynb"
LEGACY_03 = (
    "scientific_archive/legacy_code/Legacy_benchmark/notebooks/"
    "03_cross_section_cat_comparison.ipynb"
)


def text_of(path: Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


def test_benchmark_01_version_comparison_contract() -> None:
    assert BENCHMARK_01.exists()
    text = text_of(BENCHMARK_01)
    assert "import lrom_legacy.v2_0 as v2_0" in text
    assert "import lrom as v1_2" in text
    assert "eim_basis_size" not in text
    assert 'high_fidelity_solver="runge_kutta"' in text
    assert "notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb" in text
    assert 'assert (agreement["max |2.0 - 1.2|"] < 1e-10).all()' in text
    assert "# FIGURE: version-comparison-errors" in text
    assert "def plot_" not in text


def test_benchmark_03_cat_contract() -> None:
    assert BENCHMARK_03.exists()
    text = text_of(BENCHMARK_03)
    assert LEGACY_03 in text
    assert "import rose" in text
    assert "import lrom_legacy.v2_0 as lrom" in text
    assert "eim_basis_size" not in text
    assert 'high_fidelity_solver="runge_kutta"' in text
    assert 'observable="cross_section"' in text
    assert "rose.ScatteringAmplitudeEmulator.from_train" in text
    assert "def plot_" not in text
    assert "def split_violin" not in text
    for figure in (
        "representative-cross-sections",
        "cross-section-errors",
        "error-violins",
        "cat-plot",
    ):
        assert f"# FIGURE: {figure}" in text


def test_benchmark_2_0_code_cells_compile() -> None:
    for path in (BENCHMARK_01, BENCHMARK_03):
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name} cell {index}", "exec")
