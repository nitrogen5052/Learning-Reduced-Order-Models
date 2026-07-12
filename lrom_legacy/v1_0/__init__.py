"""Object-oriented learned reduced-operator models."""

from __future__ import annotations

from pathlib import Path

from .emulator import LROM
from .errors import (
    LROMArtifactError,
    LROMConfigurationError,
    LROMError,
    LROMSamplingError,
    LROMStateError,
)

__version__ = "1.0.0"


def load(*, path: str | Path) -> LROM:
    """Load a portable trained emulator."""
    from .artifacts import load as load_artifact

    return load_artifact(path=path)


__all__ = [
    "LROM",
    "LROMArtifactError",
    "LROMConfigurationError",
    "LROMError",
    "LROMSamplingError",
    "LROMStateError",
    "load",
]
