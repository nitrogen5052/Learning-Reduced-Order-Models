from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Benchmark_2.0.ipynb"
SOURCE_NOTEBOOK = (
    "scientific_archive/legacy_code/Legacy_benchmark/notebooks/"
    "03_cross_section_cat_comparison.ipynb"
)
FIGURES = {
    "representative-cross-sections",
    "cross-section-errors",
    "error-violins",
    "cat-plot",
}


def notebook_text() -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


def test_benchmark_2_0_notebook_contract() -> None:
    assert NOTEBOOK.exists()
    text = notebook_text()
    assert SOURCE_NOTEBOOK in text
    assert "import rose" in text
    assert "import lrom" in text
    assert 'observable="cross_section"' in text
    assert "import lrom_legacy" not in text
    assert "import lrom_bench" not in text
    assert "lrom_demo" not in text
    assert "def plot_" not in text
    assert "def split_violin" not in text
    assert "rose.ScatteringAmplitudeEmulator.from_train" in text
    for figure in FIGURES:
        assert f"# FIGURE: {figure}" in text


def test_benchmark_2_0_code_cells_compile() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"Benchmark_2.0 cell {index}", "exec")


def test_error_figures_use_the_display_floor() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for marker in ("cross-section-errors",):
        cell = next(
            cell
            for cell in notebook.cells
            if f"# FIGURE: {marker}" in cell.source
        )
        assert "DIFFERENCE_FLOOR" in cell.source
        assert "bottom=DIFFERENCE_FLOOR" in cell.source
