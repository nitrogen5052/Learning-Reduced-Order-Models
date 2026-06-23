from __future__ import annotations

from scripts import generate_notebook01


def source() -> str:
    return "\n\n".join(
        cell["source"] for cell in generate_notebook01.notebook_cells()
    )


def test_notebook01_uses_two_stateful_emulators() -> None:
    text = source()

    assert "vv_emulator = lrom.LROM(" in text
    assert 'potential="ws_1"' in text
    assert "ws3_emulator = lrom.LROM(" in text
    assert 'potential="ws_3"' in text
    assert "vv_emulator.sampling(" in text
    assert "ws3_emulator.sampling(" in text
    assert "vv_emulator.train(" in text
    assert "ws3_emulator.train(" in text
    assert 'predictor="parameters"' in text
    assert 'predictor="potential"' in text


def test_notebook01_names_separate_training_and_testing_ranges() -> None:
    text = source()

    assert "training_ranges=vv_training_ranges" in text
    assert "testing_ranges=vv_testing_ranges" in text
    assert "training_ranges=ws3_training_ranges" in text
    assert "testing_ranges=ws3_testing_ranges" in text
    assert 'strategy="linspace"' in text
    assert 'strategy="latin_hypercube"' in text


def test_notebook01_plots_required_physical_diagnostics_explicitly() -> None:
    text = source()

    assert "selected_radii" in text
    assert "selected potential predictor points" in text.lower()
    assert "np.real(case.high_fidelity[0])" in text
    assert "np.real(case.rose[0])" in text
    assert "np.real(case.lrom[0])" in text
    assert "vv_emulator.testing_errors[0]" in text
    assert "ws3_emulator.testing_errors[0]" in text
    assert 'ax.set_yscale("log")' in text
    assert 'ax.set_xlabel("r [fm]")' in text


def test_notebook01_uses_approved_sample_and_model_sizes() -> None:
    text = source()

    assert "training_size=35" in text
    assert "testing_size=41" in text
    assert "training_size=70" in text
    assert "testing_size=81" in text
    assert "basis_size=4" in text
    assert "predictor_count=6" in text
    assert "eim_basis_size=8" in text
