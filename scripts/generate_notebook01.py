from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_rbm_vs_lrom_single_wavefunction.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def write_notebook(path: Path, cells: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    for idx, cell in enumerate(cells):
        cell["id"] = f"notebook01-{idx:02d}"
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, path)


def notebook_cells() -> list:
    setup = r"""
    from pathlib import Path
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ROOT = Path.cwd()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from lrom_bench.config import Notebook01Config
    from lrom_bench import metrics, prediction, predictors, reduced_basis, rf_lrom, rose_fom, sampling

    cfg = Notebook01Config()
    params = rose_fom.central_real_ws_parameters()
    alpha_c = params.alpha

    print("config hash:", cfg.config_hash())
    print("central [Vv, Rv, av]:", alpha_c)
    """
    vv_samples = r"""
    vv_train = sampling.centered_1d_values(alpha_c[0], cfg.vv_width, cfg.n_vv_train)
    vv_test = sampling.centered_1d_values(alpha_c[0], cfg.vv_width, cfg.n_vv_test)

    train_alphas = rose_fom.make_alphas(alpha_c, param_index=0, values=vv_train)
    test_alphas = rose_fom.make_alphas(alpha_c, param_index=0, values=vv_test)

    problem_vv = rose_fom.make_real_ws_problem(
        params=params,
        train_alphas=train_alphas,
        n_u=cfg.n_u,
        l_max=cfg.l,
        n_mesh=cfg.n_mesh,
    )
    phi0_vv = problem_vv.solve_phi(alpha_c)
    phi_train_vv = problem_vv.solve_wavefunctions(train_alphas)
    phi_test_vv = problem_vv.solve_wavefunctions(test_alphas)
    potentials_train = np.array(
        [problem_vv.potential(problem_vv.r_mesh, alpha) for alpha in train_alphas]
    )

    print("Vv train wavefunctions:", phi_train_vv.shape)
    """
    vv_plot = r"""
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    for value, potential in zip(vv_train, potentials_train):
        label = f"{value:.1f}" if value in vv_train[[0, -1]] else None
        ax.plot(problem_vv.r_mesh, potential, alpha=0.55, label=label)
    ax.set_xlabel("r [fm]")
    ax.set_ylabel("V(r)")
    ax.set_title("Real Woods-Saxon potential rainbow: Vv-only scan")
    ax.grid(alpha=0.25)
    ax.legend(title="Vv edge values")
    fig.tight_layout()
    """
    vv_wave_plot = r"""
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    for value, phi in zip(vv_train, phi_train_vv):
        label = f"{value:.1f}" if value in vv_train[[0, -1]] else None
        ax.plot(problem_vv.r_mesh, phi.real, alpha=0.55, label=label)
    ax.set_xlabel("r [fm]")
    ax.set_ylabel("Re phi(r)")
    ax.set_title("FOM wavefunction rainbow: Vv-only scan")
    ax.grid(alpha=0.25)
    ax.legend(title="Vv edge values")
    fig.tight_layout()
    """
    vv_basis = r"""
    custom_basis_vv = rose_fom.make_real_ws_custom_basis(
        problem=problem_vv,
        phi0=phi0_vv,
        wavefunctions=phi_train_vv,
        n_basis=cfg.n_basis,
    )
    rbe_vv = rose_fom.make_real_ws_rbe(problem=problem_vv, custom_basis=custom_basis_vv)
    basis_vv = reduced_basis.CentralBasisData(
        phi0=custom_basis_vv.phi_0,
        vectors=custom_basis_vv.vectors,
        mesh=problem_vv.rho_mesh,
    )

    coeff_ls_train_vv = reduced_basis.project_ls_coordinates(basis_vv, phi_train_vv)
    coeff_ls_test_vv = reduced_basis.project_ls_coordinates(basis_vv, phi_test_vv)
    phi_ls_test_vv = prediction.reconstruct_from_basis(
        basis_vv.phi0,
        basis_vv.vectors,
        coeff_ls_test_vv,
    )
    ls_error_vv = metrics.relative_l2_rows(phi_ls_test_vv, phi_test_vv)

    print("basis shape:", basis_vv.vectors.shape)
    print("median LS-floor error:", np.median(ls_error_vv))
    """
    vv_basis_plot = r"""
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    ax.plot(problem_vv.r_mesh, basis_vv.phi0.real, color="black", alpha=0.35, label="phi0")
    display_height = 0.65 * np.max(np.abs(basis_vv.phi0.real))
    for idx in range(min(4, basis_vv.n_basis)):
        vec = basis_vv.vectors[:, idx].real
        scale = display_height / max(np.max(np.abs(vec)), 1e-14)
        ax.plot(problem_vv.r_mesh, scale * vec, label=f"basis {idx + 1} x {scale:.1f}")
    ax.set_xlabel("r [fm]")
    ax.set_ylabel("display-scaled real component")
    ax.set_title("Central-reference basis for the Vv-only case")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    """
    vv_lrom = r"""
    p_train_vv = predictors.centered_parameter_predictors(
        samples=vv_train[:, None],
        center=np.array([alpha_c[0]]),
        scales=np.array([cfg.vv_width]),
    )
    p_test_vv = predictors.centered_parameter_predictors(
        samples=vv_test[:, None],
        center=np.array([alpha_c[0]]),
        scales=np.array([cfg.vv_width]),
    )

    coeff_rose_train_vv = np.array([rbe_vv.coefficients(alpha) for alpha in train_alphas])
    coeff_rose_test_vv = np.array([rbe_vv.coefficients(alpha) for alpha in test_alphas])

    lrom_vv = rf_lrom.fit_central_lrom(
        name="notebook01_vv_only",
        predictors=p_train_vv,
        coeff_targets=coeff_ls_train_vv,
    )
    coeff_lrom_test_vv = prediction.predict_coefficients(lrom_vv, p_test_vv)

    print("LROM parameters:", lrom_vv.n_complex_parameters)
    print("residual MSE:", lrom_vv.residual_mse)
    """
    vv_coeff_plot = r"""
    fig, axes = plt.subplots(min(2, cfg.n_basis), 1, figsize=(7.0, 4.8), dpi=140, sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.plot(vv_test, coeff_ls_test_vv[:, idx].real, label="LS floor")
        ax.plot(vv_test, coeff_rose_test_vv[:, idx].real, "--", label="ROSE/RBM")
        ax.plot(vv_test, coeff_lrom_test_vv[:, idx].real, ":", linewidth=2.0, label="LROM")
        ax.set_ylabel(f"Re a{idx + 1}")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Vv")
    axes[0].set_title("Vv-only reduced coefficients")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    """
    vv_wave_errors = r"""
    phi_rose_test_vv = np.array([rbe_vv.emulate_wave_function(alpha) for alpha in test_alphas])
    phi_lrom_test_vv = prediction.reconstruct_from_basis(
        basis_vv.phi0,
        basis_vv.vectors,
        coeff_lrom_test_vv,
    )

    errors_vv = {
        "LS floor": metrics.relative_l2_rows(phi_ls_test_vv, phi_test_vv),
        "ROSE/RBM": metrics.relative_l2_rows(phi_rose_test_vv, phi_test_vv),
        "LROM": metrics.relative_l2_rows(phi_lrom_test_vv, phi_test_vv),
    }
    vv_error_summary = pd.DataFrame(
        {
            name: {
                "median": np.median(values),
                "p95": np.percentile(values, 95),
                "max": np.max(values),
            }
            for name, values in errors_vv.items()
        }
    ).T
    vv_error_summary
    """
    vv_wave_error_plot = r"""
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=140)
    ax.boxplot(list(errors_vv.values()), labels=list(errors_vv.keys()), showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("relative L2 wavefunction error")
    ax.set_title("Vv-only wavefunction reproduction")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    """
    box_samples = r"""
    widths = np.array([cfg.vv_width, cfg.rv_width, cfg.av_width])
    train_samples_3d = sampling.centered_box_samples(
        center=alpha_c,
        widths=widths,
        n_samples=cfg.n_box_train,
        seed=cfg.seed_train,
        include_center=True,
    )
    test_samples_3d = sampling.centered_box_samples(
        center=alpha_c,
        widths=widths,
        n_samples=cfg.n_box_test,
        seed=cfg.seed_test,
        include_center=True,
    )
    problem_3d = rose_fom.make_real_ws_problem(
        params=params,
        train_alphas=train_samples_3d,
        n_u=cfg.n_u,
        l_max=cfg.l,
        n_mesh=cfg.n_mesh,
    )
    phi0_3d = problem_3d.solve_phi(alpha_c)
    phi_train_3d = problem_3d.solve_wavefunctions(train_samples_3d)
    phi_test_3d = problem_3d.solve_wavefunctions(test_samples_3d)
    potentials_3d = np.array(
        [problem_3d.potential(problem_3d.r_mesh, alpha) for alpha in train_samples_3d]
    )

    print("three-parameter train wavefunctions:", phi_train_3d.shape)
    """
    box_plot = r"""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=140)
    for alpha, potential in zip(train_samples_3d, potentials_3d):
        axes[0].plot(problem_3d.r_mesh, potential, alpha=0.22)
    axes[0].set_xlabel("r [fm]")
    axes[0].set_ylabel("V(r)")
    axes[0].set_title("Vv/Rv/av potential variation")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(train_samples_3d[:, 1], train_samples_3d[:, 2], c=train_samples_3d[:, 0], s=24)
    axes[1].set_xlabel("Rv")
    axes[1].set_ylabel("av")
    axes[1].set_title("Training samples colored by Vv")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    """
    raw_lrom = r"""
    custom_basis_3d = rose_fom.make_real_ws_custom_basis(
        problem=problem_3d,
        phi0=phi0_3d,
        wavefunctions=phi_train_3d,
        n_basis=cfg.n_basis,
    )
    rbe_3d = rose_fom.make_real_ws_rbe(problem=problem_3d, custom_basis=custom_basis_3d)
    basis_3d = reduced_basis.CentralBasisData(
        phi0=custom_basis_3d.phi_0,
        vectors=custom_basis_3d.vectors,
        mesh=problem_3d.rho_mesh,
    )
    coeff_ls_train_3d = reduced_basis.project_ls_coordinates(basis_3d, phi_train_3d)
    coeff_ls_test_3d = reduced_basis.project_ls_coordinates(basis_3d, phi_test_3d)

    raw_p_train = predictors.centered_parameter_predictors(train_samples_3d, alpha_c, widths)
    raw_p_test = predictors.centered_parameter_predictors(test_samples_3d, alpha_c, widths)
    raw_lrom = rf_lrom.fit_central_lrom(
        name="notebook01_raw_vv_rv_av",
        predictors=raw_p_train,
        coeff_targets=coeff_ls_train_3d,
    )
    raw_coeff_test = prediction.predict_coefficients(raw_lrom, raw_p_test)
    phi_raw_lrom_test = prediction.reconstruct_from_basis(
        basis_3d.phi0,
        basis_3d.vectors,
        raw_coeff_test,
    )

    print("raw-parameter LROM parameters:", raw_lrom.n_complex_parameters)
    print("raw-parameter residual MSE:", raw_lrom.residual_mse)
    """
    potential_predictors = r"""
    pack = predictors.make_potential_predictor_pack(
        potential=problem_3d.potential,
        train_alphas=train_samples_3d,
        central_alpha=alpha_c,
        mesh=problem_3d.r_mesh,
        n_predictors=cfg.n_predictors,
        min_mesh_value=cfg.min_predictor_radius,
    )
    pot_p_train = predictors.centered_potential_predictors(
        potential=problem_3d.potential,
        alphas=train_samples_3d,
        pack=pack,
    )
    pot_p_test = predictors.centered_potential_predictors(
        potential=problem_3d.potential,
        alphas=test_samples_3d,
        pack=pack,
    )
    pot_lrom = rf_lrom.fit_central_lrom(
        name="notebook01_potential_predictors",
        predictors=pot_p_train,
        coeff_targets=coeff_ls_train_3d,
    )
    pot_coeff_test = prediction.predict_coefficients(pot_lrom, pot_p_test)
    phi_pot_lrom_test = prediction.reconstruct_from_basis(
        basis_3d.phi0,
        basis_3d.vectors,
        pot_coeff_test,
    )

    print("potential-predictor points:", pack.s_points)
    print("potential-predictor LROM parameters:", pot_lrom.n_complex_parameters)
    """
    predictor_plot = r"""
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    central_potential = problem_3d.potential(problem_3d.r_mesh, alpha_c)
    ax.plot(problem_3d.r_mesh, central_potential, color="black", linewidth=1.6, label="central potential")
    ax.scatter(pack.s_points, pack.center_values.real, color="tab:red", zorder=5, label="maxvol predictors")
    ax.set_xlabel("r [fm]")
    ax.set_ylabel("V(r)")
    ax.set_title("Operator-informed predictor locations")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    """
    performance = r"""
    phi_ls_test_3d = prediction.reconstruct_from_basis(
        basis_3d.phi0,
        basis_3d.vectors,
        coeff_ls_test_3d,
    )
    phi_rose_test_3d = np.array([rbe_3d.emulate_wave_function(alpha) for alpha in test_samples_3d])

    comparison_3d = {
        "LS floor": metrics.relative_l2_rows(phi_ls_test_3d, phi_test_3d),
        "ROSE/RBM": metrics.relative_l2_rows(phi_rose_test_3d, phi_test_3d),
        "raw-param LROM": metrics.relative_l2_rows(phi_raw_lrom_test, phi_test_3d),
        "potential-predictor LROM": metrics.relative_l2_rows(phi_pot_lrom_test, phi_test_3d),
    }
    comparison_summary_3d = pd.DataFrame(
        {
            name: {
                "median": np.median(values),
                "p95": np.percentile(values, 95),
                "max": np.max(values),
            }
            for name, values in comparison_3d.items()
        }
    ).T
    comparison_summary_3d
    """
    performance_plot = r"""
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=140)
    ax.boxplot(list(comparison_3d.values()), labels=list(comparison_3d.keys()), showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("relative L2 wavefunction error")
    ax.set_title("Three-parameter wavefunction reproduction")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    """
    takeaway = r"""
    pd.concat(
        {
            "Vv-only": vv_error_summary,
            "Vv/Rv/av": comparison_summary_3d,
        },
        axis=0,
    )
    """
    return [
        md(
            """
            # 01. RBM vs LROM for a Single Scattering Wavefunction

            This notebook compares a traditional reduced-basis view and the RF-LROM view for one `l = 0`
            real Woods-Saxon scattering wavefunction. The first act varies only `Vv`; the second act expands
            to `Vv`, `Rv`, and `av` to motivate operator-informed predictors.
            """
        ),
        md("## 1. Scientific Setup"),
        code(setup),
        md("## 2. Vv-Only Samples"),
        code(vv_samples),
        code(vv_plot),
        code(vv_wave_plot),
        md("## 3. Reduced Basis And LS Floor"),
        code(vv_basis),
        code(vv_basis_plot),
        md("## 4. RBM/ROSE vs LROM Coordinates"),
        code(vv_lrom),
        code(vv_coeff_plot),
        md("## 5. Wavefunction Reproduction"),
        code(vv_wave_errors),
        code(vv_wave_error_plot),
        md("## 6. Three-Parameter Samples"),
        code(box_samples),
        code(box_plot),
        md("## 7. Why Raw Parameters Are Not Enough"),
        code(raw_lrom),
        md("## 8. Operator-Informed Potential Predictors"),
        code(potential_predictors),
        code(predictor_plot),
        md("## 9. Three-Parameter Performance"),
        code(performance),
        code(performance_plot),
        md("## 10. Notebook 1 Takeaways"),
        code(takeaway),
    ]


def main() -> None:
    write_notebook(NOTEBOOK_PATH, notebook_cells())


if __name__ == "__main__":
    main()
