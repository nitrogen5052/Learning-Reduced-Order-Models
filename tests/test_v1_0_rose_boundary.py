from lrom_legacy.v1_0.emulator import LROM
from lrom_legacy.v1_0.state import (
    TestingCase as N1TestingCase,
    TestingResults as N1TestingResults,
    TrainingState,
)


def test_n1_exposes_no_rose_rom_state() -> None:
    assert not hasattr(LROM, "rose_rbm")
    assert "rose_rbm" not in TrainingState.__dataclass_fields__
    assert "rose" not in N1TestingResults.__dataclass_fields__
    assert "rose" not in N1TestingCase.__dataclass_fields__


def test_n1_result_schema_contains_only_lrom_and_references() -> None:
    assert set(N1TestingResults.__dataclass_fields__) == {
        "high_fidelity",
        "lrom",
        "ls",
        "coefficients",
        "metrics",
    }
