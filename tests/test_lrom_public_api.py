from __future__ import annotations

import inspect
from pathlib import Path

import lrom
import lrom_legacy.v1_2 as lrom_v1_2
import lrom_legacy.v2_0 as lrom_v2_0


ROOT = Path(__file__).resolve().parents[1]


def test_public_package_exports() -> None:
    assert lrom.__version__ == "1.2.0"
    assert lrom.LROM is lrom_v1_2.LROM
    assert callable(lrom.load)


def test_v2_remains_an_explicit_future_shell() -> None:
    assert lrom_v2_0.__version__ == "2.0.0"
    assert lrom_v2_0.LROM.__name__ == "LROM"
    assert callable(lrom_v2_0.load)
    assert lrom_v2_0 is not lrom


def test_parked_v2_matches_the_verified_single_file_shell() -> None:
    train_parameters = inspect.signature(lrom_v2_0.LROM.train).parameters
    assert "observable" in train_parameters
    assert "angles_degrees" in train_parameters
    assert lrom_v2_0.LROM is not lrom.LROM


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
