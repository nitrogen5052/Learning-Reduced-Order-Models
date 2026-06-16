from __future__ import annotations


def test_package_imports() -> None:
    import lrom_bench

    assert lrom_bench.__version__ == "0.1.0"
