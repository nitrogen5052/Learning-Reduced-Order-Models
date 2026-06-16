from __future__ import annotations

from pathlib import Path

from lrom_bench.config import BenchmarkPaths, Notebook01Config, Notebook02Config


def test_package_imports() -> None:
    import lrom_bench

    assert lrom_bench.__version__ == "0.1.0"


def test_notebook02_config_hash_is_stable() -> None:
    cfg = Notebook02Config(n_mesh=64, n_phi=3, n_u=5)

    assert cfg.config_hash() == Notebook02Config(n_mesh=64, n_phi=3, n_u=5).config_hash()
    assert cfg.config_hash() != Notebook02Config(n_mesh=65, n_phi=3, n_u=5).config_hash()
    assert len(cfg.config_hash()) == 16


def test_notebook01_config_hash_is_stable() -> None:
    cfg = Notebook01Config(n_mesh=64, n_basis=3, n_u=5)

    assert cfg.config_hash() == Notebook01Config(n_mesh=64, n_basis=3, n_u=5).config_hash()
    assert cfg.config_hash() != Notebook01Config(n_mesh=65, n_basis=3, n_u=5).config_hash()
    assert cfg.parameter_names == ("Vv", "Rv", "av")
    assert len(cfg.config_hash()) == 16


def test_benchmark_paths_are_stable(tmp_path: Path) -> None:
    paths = BenchmarkPaths(root=tmp_path)

    assert paths.legacy_npz("notebook02").as_posix().endswith(
        "outputs/benchmarks/legacy/notebook02.npz"
    )
    assert paths.new_npz("notebook02").as_posix().endswith(
        "outputs/benchmarks/new/notebook02.npz"
    )
    assert paths.report_json("notebook02").as_posix().endswith(
        "outputs/benchmarks/reports/notebook02.json"
    )
