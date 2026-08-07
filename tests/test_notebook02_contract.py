"""Executable-artifact contracts for the two maintained top-level notebooks."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_DIRECTORY = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = (
    NOTEBOOK_DIRECTORY / "01_rbm_vs_lrom_single_wavefunction.ipynb",
    NOTEBOOK_DIRECTORY / "02_rose_vs_lrom_cross_sections.ipynb",
)


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def code_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_all_code_cells_compile_and_stored_outputs_have_no_errors() -> None:
    for path in NOTEBOOKS:
        notebook = load_notebook(path)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), f"{path.name}:{index}", "exec")
                assert all(
                    output.get("output_type") != "error"
                    for output in cell.get("outputs", [])
                )


def test_notebook02_markdown_contains_only_section_headings() -> None:
    notebook = load_notebook(NOTEBOOKS[1])
    markdown = [
        "".join(cell.get("source", [])).strip()
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown" and "".join(cell.get("source", [])).strip()
    ]

    assert all(text.startswith("#") and "\n" not in text for text in markdown)


def test_notebook02_displays_raw_errors_on_logarithmic_axes() -> None:
    source = code_source(load_notebook(NOTEBOOKS[1]))

    assert "np.log10" not in source
    assert 'set_yscale("log")' in source
    assert "pointwise_relative_error" in source


def test_notebook02_calls_separate_rose_and_lrom_pipelines() -> None:
    source = code_source(load_notebook(NOTEBOOKS[1]))

    assert "RoseCrossSectionPipeline" in source
    assert "LromCrossSectionStudy" in source
    assert "CrossSectionBenchmark" not in source
