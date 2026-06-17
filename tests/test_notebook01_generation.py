from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import generate_notebook01


ROOT = Path(__file__).resolve().parents[1]


def test_notebook01_bootstrap_finds_repo_root_from_notebooks_dir() -> None:
    script = f"""
from pathlib import Path
import os
import sys

os.chdir({str(ROOT / "notebooks")!r})
sys.path = [p for p in sys.path if p not in {{{str(ROOT)!r}, {str(ROOT / "notebooks")!r}}}]

{generate_notebook01.repo_bootstrap_source()}

from lrom_bench.config import Notebook01Config

assert ROOT == Path({str(ROOT)!r})
assert sys.path[0] == {str(ROOT)!r}
assert Notebook01Config().parameter_names == ("Vv", "Rv", "av")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_notebook01_setup_cell_uses_repo_bootstrap_source() -> None:
    setup_cell = generate_notebook01.notebook_cells()[2]

    assert generate_notebook01.repo_bootstrap_source() in setup_cell["source"]
    compile(setup_cell["source"], "notebook01 setup cell", "exec")
