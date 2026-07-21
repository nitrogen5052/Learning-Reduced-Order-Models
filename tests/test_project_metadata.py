from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_version_window_is_311_to_313() -> None:
    text = (ROOT / "pyproject.toml").read_text()

    assert 'requires-python = ">=3.11,<3.14"' in text


def test_project_is_named_lrom() -> None:
    text = (ROOT / "pyproject.toml").read_text()

    assert 'name = "lrom"' in text


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


def test_package_discovery_includes_new_and_legacy_namespaces() -> None:
    text = (ROOT / "pyproject.toml").read_text()

    assert 'include = ["lrom*", "lrom_legacy*"]' in text


def test_readme_documents_canonical_portable_workflow() -> None:
    text = (ROOT / "README.md").read_text()

    for snippet in (
        "import lrom",
        "emulator = lrom.LROM(",
        "emulator.sampling(",
        "emulator.train(",
        "emulator.save(",
        "portable = lrom.load(",
        "portable.predict(",
    ):
        assert snippet in text


def test_ci_covers_supported_python_versions() -> None:
    text = (ROOT / ".github" / "workflows" / "test.yml").read_text()

    assert 'python-version: ["3.11", "3.12", "3.13"]' in text
    assert "pip install -e .[test]" in text
    assert "pytest -q" in text


def test_architecture_understanding_guide_covers_user_ownership_model() -> None:
    text = (ROOT / "docs" / "LROM_ARCHITECTURE_UNDERSTANDING.md").read_text()

    for phrase in (
        "LROMConfig",
        "SamplingState",
        "TrainingState",
        "PredictionState",
        "NuclearScatteringFOM",
        "TrainingEngine",
        "lrom_legacy.v1_2",
        "central-reference",
        "free-reference",
        "Methodology",
        "Before",
        "After",
        "Execution",
        "What did not change",
        "Final prose ownership",
    ):
        assert phrase in text
