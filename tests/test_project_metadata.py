from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_version_window_is_310_to_311() -> None:
    text = (ROOT / "pyproject.toml").read_text()

    assert 'requires-python = ">=3.10,<3.12"' in text


def test_runtime_dependencies_are_declared() -> None:
    text = (ROOT / "pyproject.toml").read_text()

    for dependency in [
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "nbformat",
        "nbclient",
        "nuclear-rose>=1.1.7",
        "numba",
    ]:
        assert f'"{dependency}"' in text
