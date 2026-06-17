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


def test_notebook01_setup_cell_clears_stale_lrom_bench_modules() -> None:
    setup_cell = generate_notebook01.notebook_cells()[2]

    assert "list(sys.modules)" in setup_cell["source"]
    assert 'name == "lrom_bench"' in setup_cell["source"]
    assert 'name.startswith("lrom_bench.")' in setup_cell["source"]
    assert "del sys.modules[name]" in setup_cell["source"]


def test_notebook01_source_has_requested_three_section_spine() -> None:
    source = "\n\n".join(cell["source"] for cell in generate_notebook01.notebook_cells())

    assert "## Section 1. Parameter Varying Vv" in source
    assert "## Section 2. Three-Parameter LROM Equation And Predictor Selection" in source
    assert "## Section 3. Three-Parameter Wavefunction Emulation Results" in source
    assert "Why this section matters" in source
    assert source.index("## Section 1. Parameter Varying Vv") < source.index(
        "## Section 2. Three-Parameter LROM Equation And Predictor Selection"
    )
    assert source.index("## Section 2. Three-Parameter LROM Equation And Predictor Selection") < source.index(
        "## Section 3. Three-Parameter Wavefunction Emulation Results"
    )


def test_notebook01_source_includes_inputs_and_legacy_comparison() -> None:
    source = "\n\n".join(cell["source"] for cell in generate_notebook01.notebook_cells())

    assert "Notebook Inputs And Legacy Comparison" in source
    assert "Legacy Notebook 1" in source
    assert "Legacy Notebook 2" in source
    assert "40Ca(n,n)" in source
    assert "Vv/Rv/av" in source
    assert "n_basis" in source
    assert "n_U" in source
    assert "n_mesh" in source
    assert "n_predictors" in source


def test_notebook01_source_keeps_vv_scan_and_train_test_coefficients_visible() -> None:
    source = "\n\n".join(cell["source"] for cell in generate_notebook01.notebook_cells())

    assert "(1.0 - cfg.vv_train_fraction) * alpha_c[0]" in source
    assert "(1.0 + cfg.vv_test_fraction) * alpha_c[0]" in source
    assert "ax.scatter(vv_train" in source
    assert "coeff_ls_train_vv" in source
    assert "coeff_rose_train_vv" in source
    assert "training LS" in source
    assert "testing LS" in source


def test_notebook01_source_uses_vv_rv_av_after_vv_only_section() -> None:
    source = "\n\n".join(cell["source"] for cell in generate_notebook01.notebook_cells())

    assert "cfg.vv_3d_fraction * V0" in source
    assert "cfg.rv_3d_fraction * R0" in source
    assert "cfg.av_3d_fraction * a0" in source
    assert "train_samples_3d[:, 2]" in source
    assert "test_samples_3d[:, 2]" in source
    assert 'name="notebook01_raw_vv_rv_av"' in source
    assert "Vv/Rv/av wavefunction reproduction" in source


def test_notebook01_source_shows_transformed_lrom_equation_and_maxvol() -> None:
    source = "\n\n".join(cell["source"] for cell in generate_notebook01.notebook_cells())

    assert r"(I + p_1 M_1 + \cdots + p_K M_K)a" in source
    assert "maxvol" in source.lower()
    assert "make_potential_predictor_pack" in source
