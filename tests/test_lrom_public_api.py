from __future__ import annotations

import inspect
from pathlib import Path

import lrom
import lrom_legacy.v2_0 as lrom_v2_0


ROOT = Path(__file__).resolve().parents[1]


def test_public_package_exports() -> None:
    assert lrom.__version__ == "2.0.0"
    assert lrom.LROM.__name__ == "LROM"
    assert callable(lrom.load)


def test_notebook1_snapshot_imports_as_legacy_namespace() -> None:
    assert lrom_v2_0.__version__ == "2.0.0"
    assert lrom_v2_0.LROM.__name__ == "LROM"
    assert callable(lrom_v2_0.load)
    assert lrom_v2_0 is not lrom


def test_constructor_is_keyword_only() -> None:
    parameters = inspect.signature(lrom.LROM).parameters

    assert parameters
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )


def test_supported_python_window_and_package_name() -> None:
    text = (ROOT / "pyproject.toml").read_text()

    assert 'name = "lrom"' in text
    assert 'requires-python = ">=3.11,<3.14"' in text
