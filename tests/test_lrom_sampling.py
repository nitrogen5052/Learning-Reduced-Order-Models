from __future__ import annotations

import numpy as np
import pytest

from lrom.errors import LROMSamplingError
from lrom.sampling import create_sampling_design


CENTRAL = {"Vv": 50.0, "Rv": 4.0, "av": 0.65}
NAMES = ("Vv", "Rv", "av")


def test_linspace_uses_separate_training_and_testing_ranges() -> None:
    design = create_sampling_design(
        parameter_names=NAMES,
        sampleable_names=NAMES,
        central=CENTRAL,
        training_ranges={"Vv": (45.0, 55.0)},
        testing_ranges={"Vv": (32.5, 67.5)},
        training_size=35,
        testing_size=41,
        strategy="linspace",
        seed=1204,
    )

    assert design.training.values.shape == (35, 3)
    assert design.testing.values.shape == (41, 3)
    assert design.training.case_ids[0] == "train-0000"
    assert design.testing.case_ids[-1] == "test-0040"
    assert np.all(design.training.values[:, 1] == 4.0)
    assert np.all(design.training.values[:, 2] == 0.65)
    assert design.training.named(index=0) == {
        "Vv": 45.0,
        "Rv": 4.0,
        "av": 0.65,
    }


def test_lhs_is_deterministic_named_and_independent_between_splits() -> None:
    kwargs = dict(
        parameter_names=NAMES,
        sampleable_names=NAMES,
        central=CENTRAL,
        training_ranges={"Vv": (45, 55), "Rv": (3.6, 4.4), "av": (0.585, 0.715)},
        testing_ranges={"Vv": (39, 61), "Rv": (3.2, 4.8), "av": (0.52, 0.78)},
        training_size=70,
        testing_size=81,
        strategy="latin_hypercube",
        seed=1204,
    )

    first = create_sampling_design(**kwargs)
    second = create_sampling_design(**kwargs)

    assert np.allclose(first.training.values, second.training.values)
    assert np.allclose(first.testing.values, second.testing.values)
    assert not np.allclose(first.training.values[:10], first.testing.values[:10])
    assert np.all((first.training.values[:, 0] >= 45) & (first.training.values[:, 0] <= 55))
    assert np.all((first.testing.values[:, 1] >= 3.2) & (first.testing.values[:, 1] <= 4.8))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"training_ranges": {"depth": (1.0, 2.0)}, "testing_ranges": {"depth": (1.0, 2.0)}}, "unknown"),
        ({"training_ranges": {"Rv": (3.0, 4.0)}, "testing_ranges": {"Rv": (3.0, 4.0)}}, "sampleable"),
        ({"training_ranges": {"Vv": (55.0, 45.0)}}, "increasing"),
        ({"training_size": 0}, "training_size"),
        ({"strategy": "grid"}, "strategy"),
        (
            {
                "strategy": "linspace",
                "sampleable_names": NAMES,
                "training_ranges": {"Vv": (45.0, 55.0), "Rv": (3.6, 4.4)},
                "testing_ranges": {"Vv": (40.0, 60.0), "Rv": (3.2, 4.8)},
            },
            "exactly one",
        ),
        (
            {"sampleable_names": NAMES, "testing_ranges": {"Rv": (3.2, 4.8)}},
            "same parameter names",
        ),
    ],
)
def test_invalid_sampling_design_is_rejected(
    changes: dict[str, object], message: str
) -> None:
    kwargs: dict[str, object] = {
        "parameter_names": NAMES,
        "sampleable_names": ("Vv",),
        "central": CENTRAL,
        "training_ranges": {"Vv": (45.0, 55.0)},
        "testing_ranges": {"Vv": (32.5, 67.5)},
        "training_size": 35,
        "testing_size": 41,
        "strategy": "latin_hypercube",
        "seed": 1204,
    }
    kwargs.update(changes)

    with pytest.raises(LROMSamplingError, match=message):
        create_sampling_design(**kwargs)
