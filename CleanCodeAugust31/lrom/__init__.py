"""Self-contained learned reduced-order modeling for neutron scattering.

Implementation remains organized by responsibility across focused submodules.
This package root exports only the compact user-facing workflow.
"""

from .channels import Channel, elastic_channels
from .backends import Runtime, get_runtime
from .data import InputSpace, TrainingData
from .deployment import HardRoutedScatteringLROM
from .emulator import BasisConfig, MaxVol, Prediction, ScatteringLROM
from .fom import NumerovFOM
from .problem import ScatteringProblem

__all__ = [
    "BasisConfig", "Channel", "HardRoutedScatteringLROM", "InputSpace",
    "MaxVol", "Prediction", "NumerovFOM", "ScatteringLROM",
    "ScatteringProblem", "TrainingData",
    "elastic_channels", "Runtime", "get_runtime",
]

__version__ = "0.3.0"
