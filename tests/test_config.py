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


def test_notebook01_defaults_match_legacy_real_ws_benchmark() -> None:
    cfg = Notebook01Config()

    assert (cfg.target_a, cfg.target_z) == (40, 20)
    assert (cfg.projectile_a, cfg.projectile_z) == (1, 0)
    assert cfg.e_lab == 14.1
    assert cfg.l == 1
    assert cfg.n_mesh == 800
    assert cfg.n_basis == 4
    assert cfg.n_u == 8
    assert cfg.n_vv_train == 35
    assert cfg.n_vv_test == 41
    assert cfg.n_box_train == 70
    assert cfg.n_box_test == 81
    assert cfg.n_predictors == 6
    assert cfg.seed_train == 1204
    assert cfg.vv_train_fraction == 0.10
    assert cfg.vv_test_fraction == 0.35
    assert cfg.rv_train_fraction == 0.06
    assert cfg.rv_test_fraction == 0.30
    assert cfg.vv_3d_fraction == 0.22
    assert cfg.rv_3d_fraction == 0.20
    assert cfg.av_3d_fraction == 0.20


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
