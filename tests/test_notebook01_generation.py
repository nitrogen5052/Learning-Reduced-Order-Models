from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools import generate_notebook01


ROOT = Path(__file__).resolve().parents[1]


def notebook_source() -> str:
    return "\n\n".join(
        cell["source"] for cell in generate_notebook01.notebook_cells()
    )


def test_notebook01_bootstrap_finds_new_package() -> None:
    script = f"""
from pathlib import Path
import os
import sys

os.chdir({str(ROOT / "notebooks")!r})
sys.path = [p for p in sys.path if p not in {{{str(ROOT)!r}, {str(ROOT / "notebooks")!r}}}]

{generate_notebook01.repo_bootstrap_source()}

import lrom

assert ROOT == Path({str(ROOT)!r})
assert sys.path[0] == {str(ROOT)!r}
assert lrom.LROM.__name__ == "LROM"
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_setup_cell_uses_lrom_and_clears_stale_modules() -> None:
    setup = generate_notebook01.notebook_cells()[2]["source"]

    assert generate_notebook01.repo_bootstrap_source() in setup
    assert "import lrom" in setup
    assert 'name == "lrom"' in setup
    assert 'name.startswith("lrom.")' in setup
    assert "lrom_bench" not in setup
    compile(setup, "notebook01 setup", "exec")


def test_notebook01_keeps_the_three_scientific_sections() -> None:
    source = notebook_source()

    headings = (
        "## Section 1. Parameter Varying Vv",
        "## Section 2. Three-Parameter LROM Equation And Predictor Selection",
        "## Section 3. Three-Parameter Wavefunction Emulation Results",
    )
    assert all(heading in source for heading in headings)
    assert source.index(headings[0]) < source.index(headings[1]) < source.index(headings[2])


def test_notebook01_contains_no_cross_section_workflow() -> None:
    source = notebook_source().lower()

    assert "cross_section" not in source
    assert "cross section" not in source
    assert "differential_cross" not in source
