"""Load the trusted pretrained deployment distributed with the notebooks."""

from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np

from .deployment import HardRoutedScatteringLROM


THREE_WINDOW_EDGES = np.asarray([5.0, 70.0, 135.0, 200.0])


def load_three_window_lrom(directory, angles, *, runtime=None, maximum_batch_size=1536):
    """Load the three trusted ridge models and construct their hard router."""
    directory = Path(directory)
    paths = [
        directory / f"energy_window_{index}_b14_p20_ridge1e-14.pkl"
        for index in range(3)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing pretrained model(s): " + ", ".join(missing))
    models = []
    for path in paths:
        with path.open("rb") as stream:
            model = pickle.load(stream)  # trusted repository artifact only
        model.repack("padded")
        models.append(model)
    return HardRoutedScatteringLROM(
        models, THREE_WINDOW_EDGES, angles,
        maximum_batch_size=maximum_batch_size,
        linear_solver=runtime,
    )

