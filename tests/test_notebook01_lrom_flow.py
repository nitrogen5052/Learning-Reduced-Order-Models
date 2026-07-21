from __future__ import annotations

from tools import generate_notebook01


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
    assert "np.real(ws3_rose_wf_test[representative_index])" in text
    assert "np.real(case.lrom[0])" in text
    assert "vv_emulator.testing_errors[0]" in text
    assert 'ax.set_yscale("log")' in text
    assert 'ax.set_xlabel("r [fm]")' in text


def test_notebook01_uses_approved_sample_and_model_sizes() -> None:
    text = source()

    assert "training_size=35" in text
    assert "testing_size=41" in text
    assert "training_size=70" in text
    assert "testing_size=81" in text
    assert "BASIS_SIZE = 4" in text
    assert text.count("basis_size=BASIS_SIZE") == 2
    assert text.count("n_basis=BASIS_SIZE") == 2
    assert "predictor_count=6" in text
    assert "eim_basis_size=8" in text


def test_vv_coefficients_use_separate_lrom_and_rose_coordinate_figures() -> None:
    text = source()

    assert "for coefficient_index in range(2)" in text
    assert 'for method, color in (("ls", "blue"), ("lrom", "orange"))' in text
    assert "coefficients[method][0][:, coefficient_index]" in text
    assert "vv_rose_coefficients[:, coefficient_index]" in text
    assert ".scatter(" in text
    assert "ax.axvspan(vv_test.min(), vv_train_low" in text
    assert "ax.axvspan(vv_train_high, vv_test.max()" in text
    assert "np.abs(ls_coefficients - lrom_coefficients)" in text
    assert "|LS - ROSE|" not in text


def test_vv_central_testing_wavefunction_compares_all_methods() -> None:
    text = source()

    assert "vv_representative_index = len(vv_test) // 2" in text
    assert "vv_emulator.testing_case(case_id=vv_representative_id)" in text
    for method in ("high_fidelity", "ls", "lrom"):
        assert f"vv_case.{method}[0]" in text
    assert "vv_rose_wavefunctions[vv_representative_index]" in text
    for method in ("ls", "lrom"):
        assert f"np.abs(vv_case.high_fidelity[0] - vv_case.{method}[0])" in text
    assert "np.abs(vv_case.high_fidelity[0] - vv_rose_wavefunctions[vv_representative_index])" in text


def test_ws3_coefficients_are_separate_and_wavefunctions_keep_absolute_differences() -> None:
    text = source()

    assert "np.abs(ws3_ls_coefficients - ws3_lrom_coefficients)" in text
    assert "np.abs(ws3_ls_coefficients - ws3_rose_coefficients)" not in text
    for method in ("ls", "lrom"):
        assert f"np.abs(case.high_fidelity[0] - case.{method}[0])" in text
    assert "np.abs(case.high_fidelity[0] - ws3_rose_wf_test[representative_index])" in text


def test_notebook01_rose_uses_free_reference_without_lrom_basis_overwrites() -> None:
    text = source()

    assert text.count("rose.free_solutions.phi_free(") == 2
    assert text.count("rose.basis.CustomBasis(") == 2
    assert text.count("subtract_phi0=True") == 2
    assert text.count("use_svd=True") >= 2
    assert "phi_0=np.asarray(vv_emulator.samples.central_wavefunctions" not in text
    assert "phi_0=np.asarray(ws3_emulator.samples.central_wavefunctions" not in text
    assert "vv_rose_basis.vectors =" not in text
    assert "vv_rose_basis.phi_0 =" not in text
    assert "ws3_rose_basis.vectors =" not in text
    assert "ws3_rose_basis.phi_0 =" not in text


def test_notebook01_declares_equal_rank_but_separate_coordinate_conventions() -> None:
    text = source()

    assert "BASIS_SIZE = 4" in text
    assert "same high-fidelity training snapshots and retained rank" in text
    assert "different reference functions" in text
    assert "|LS - ROSE|" not in text
    assert "shared-basis" not in text.lower()


def test_ws3_final_figure_is_relative_l2_violin_comparison() -> None:
    text = source()

    assert 'training_metrics = dict(ws3_emulator.training_results.metrics["relative_l2"][0])' in text
    assert 'training_metrics["rose"] = ws3_rose_rel_train' in text
    assert 'testing_metrics = dict(ws3_emulator.testing_results.metrics["relative_l2"][0])' in text
    assert 'testing_metrics["rose"] = ws3_rose_rel_test' in text
    assert "training_violin = ax.violinplot(" in text
    assert "testing_violin = ax.violinplot(" in text
    assert 'training_violin["bodies"]' in text
    assert 'testing_violin["bodies"]' in text
