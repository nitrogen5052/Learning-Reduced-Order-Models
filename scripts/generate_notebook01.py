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


def repo_bootstrap_source() -> str:
    return textwrap.dedent(
        """
        ROOT = next(
            candidate
            for candidate in (Path.cwd(), *Path.cwd().parents)
            if (candidate / "lrom_bench").is_dir()
        )
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        """
    ).strip()


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
    setup = "\n".join(
        [
            "from pathlib import Path",
            "import sys",
            "",
            "import matplotlib.pyplot as plt",
            "import numpy as np",
            "import pandas as pd",
            "",
            repo_bootstrap_source(),
            "",
            "from lrom_bench.config import Notebook01Config",
            "from lrom_bench import metrics, prediction, predictors, reduced_basis, rf_lrom, rose_fom, sampling",
            "",
            "cfg = Notebook01Config()",
            "params = rose_fom.central_real_ws_parameters()",
            "alpha_c = params.alpha",
            "",
            'print("config hash:", cfg.config_hash())',
            'print("central [Vv, Rv, av]:", alpha_c)',
        ]
    )
    input_table = r"""
    input_rows = [
        {
            "source": "This Notebook 1",
            "physics": "40Ca(n,n), real Woods-Saxon Vv/Rv/av",
            "projectile / target": f"({cfg.projectile_a}, {cfg.projectile_z}) on ({cfg.target_a}, {cfg.target_z})",
            "E_lab [MeV]": cfg.e_lab,
            "channel shown": "l=0 wavefunction",
            "ROSE interaction": f"EIM, l_max={cfg.l}, n_U={cfg.n_u}",
            "basis": f"central CustomBasis, n_basis={cfg.n_basis}",
            "mesh": f"n_mesh={cfg.n_mesh}, rho in [1e-8, 8*pi]",
            "training ranges": (
                f"Vv +/- {100 * cfg.vv_3d_fraction:.0f}%, "
                f"Rv +/- {100 * cfg.rv_3d_fraction:.0f}%, "
                f"av +/- {100 * cfg.av_3d_fraction:.0f}%"
            ),
            "train / test": f"{cfg.n_box_train} / {cfg.n_box_test}",
            "predictors": f"K={cfg.n_predictors} maxvol potential samples",
        },
        {
            "source": "Legacy Notebook 1",
            "physics": "40Ca(n,n), full KD optical-potential ROSE emulator",
            "projectile / target": "(1, 0) on (40, 20)",
            "E_lab [MeV]": 14.1,
            "channel shown": "l=0 diagnostics inside a larger scattering emulator",
            "ROSE interaction": "EIM, larger l_max for cross-section context",
            "basis": "ROSE ScatteringAmplitudeEmulator basis",
            "mesh": "publication-quality examples may increase n_mesh",
            "training ranges": "Latin-hypercube KD parameter variation",
            "train / test": "larger ROSE benchmark sample counts",
            "predictors": "not an RF-LROM predictor-selection notebook",
        },
        {
            "source": "Legacy Notebook 2",
            "physics": "40Ca(n,n), real Woods-Saxon teaching benchmark",
            "projectile / target": "(1, 0) on (40, 20)",
            "E_lab [MeV]": 14.1,
            "channel shown": "l=0 wavefunction",
            "ROSE interaction": "EIM, l_max=1, n_U=8",
            "basis": "n_phi=4 central and ROSE reduced bases",
            "mesh": "n_mesh=800",
            "training ranges": "Vv-only, Rv-only, then broad Vv/Rv predictor stress test",
            "train / test": "35/41 for one-parameter scans; 70 train for broad box",
            "predictors": "maxvol-style potential predictors",
        },
    ]
    input_comparison = pd.DataFrame(input_rows)
    input_comparison
    """
    vv_samples = r"""
    vv_train = np.linspace(
        (1.0 - cfg.vv_train_fraction) * alpha_c[0],
        (1.0 + cfg.vv_train_fraction) * alpha_c[0],
        cfg.n_vv_train,
    )
    vv_test = np.linspace(
        (1.0 - cfg.vv_test_fraction) * alpha_c[0],
        (1.0 + cfg.vv_test_fraction) * alpha_c[0],
        cfg.n_vv_test,
    )

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
        scales=np.array([cfg.vv_train_fraction * alpha_c[0]]),
    )
    p_test_vv = predictors.centered_parameter_predictors(
        samples=vv_test[:, None],
        center=np.array([alpha_c[0]]),
        scales=np.array([cfg.vv_train_fraction * alpha_c[0]]),
    )

    coeff_rose_train_vv = np.array([rbe_vv.coefficients(alpha) for alpha in train_alphas])
    coeff_rose_test_vv = np.array([rbe_vv.coefficients(alpha) for alpha in test_alphas])

    lrom_vv = rf_lrom.fit_central_lrom(
        name="notebook01_vv_only",
        predictors=p_train_vv,
        coeff_targets=coeff_ls_train_vv,
    )
    coeff_lrom_train_vv = prediction.predict_coefficients(lrom_vv, p_train_vv)
    coeff_lrom_test_vv = prediction.predict_coefficients(lrom_vv, p_test_vv)

    print("LROM parameters:", lrom_vv.n_complex_parameters)
    print("residual MSE:", lrom_vv.residual_mse)
    """
    vv_coeff_plot = r"""
    fig, axes = plt.subplots(min(2, cfg.n_basis), 1, figsize=(7.0, 4.8), dpi=140, sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.scatter(vv_train, coeff_ls_train_vv[:, idx].real, s=18, alpha=0.75, label="training LS")
        ax.scatter(vv_train, coeff_rose_train_vv[:, idx].real, s=14, marker="x", alpha=0.75, label="training ROSE/RBM")
        ax.plot(vv_train, coeff_lrom_train_vv[:, idx].real, ".", alpha=0.75, label="training LROM")
        ax.plot(vv_test, coeff_ls_test_vv[:, idx].real, label="testing LS")
        ax.plot(vv_test, coeff_rose_test_vv[:, idx].real, "--", label="testing ROSE/RBM")
        ax.plot(vv_test, coeff_lrom_test_vv[:, idx].real, ":", linewidth=2.0, label="testing LROM")
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
    V0, R0, a0 = alpha_c
    center_3d = np.array([V0, R0, a0])
    widths_3d = np.array([
        cfg.vv_3d_fraction * V0,
        cfg.rv_3d_fraction * R0,
        cfg.av_3d_fraction * a0,
    ])

    train_samples_3d = sampling.centered_box_samples(
        center=center_3d,
        widths=widths_3d,
        n_samples=cfg.n_box_train,
        seed=cfg.seed_train,
        include_center=True,
    )
    test_samples_3d = sampling.centered_box_samples(
        center=center_3d,
        widths=widths_3d,
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

    print("Vv/Rv/av train wavefunctions:", phi_train_3d.shape)
    """
    box_plot = r"""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=140)
    for alpha, potential in zip(train_samples_3d, potentials_3d):
        axes[0].plot(problem_3d.r_mesh, potential, alpha=0.22)
    axes[0].set_xlabel("r [fm]")
    axes[0].set_ylabel("V(r)")
    axes[0].set_title("Vv/Rv/av potential variation")
    axes[0].grid(alpha=0.25)

    scatter = axes[1].scatter(
        train_samples_3d[:, 0],
        train_samples_3d[:, 1],
        c=train_samples_3d[:, 2],
        s=24,
    )
    axes[1].set_xlabel("Vv")
    axes[1].set_ylabel("Rv")
    axes[1].set_title("Training samples colored by av")
    axes[1].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[1], label="av")
    fig.tight_layout()

    pd.DataFrame(
        {
            "split": ["train", "test"],
            "Vv min": [train_samples_3d[:, 0].min(), test_samples_3d[:, 0].min()],
            "Vv max": [train_samples_3d[:, 0].max(), test_samples_3d[:, 0].max()],
            "Rv min": [train_samples_3d[:, 1].min(), test_samples_3d[:, 1].min()],
            "Rv max": [train_samples_3d[:, 1].max(), test_samples_3d[:, 1].max()],
            "av min": [train_samples_3d[:, 2].min(), test_samples_3d[:, 2].min()],
            "av max": [train_samples_3d[:, 2].max(), test_samples_3d[:, 2].max()],
        }
    )
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

    raw_p_train = predictors.centered_parameter_predictors(
        train_samples_3d,
        center_3d,
        widths_3d,
    )
    raw_p_test = predictors.centered_parameter_predictors(
        test_samples_3d,
        center_3d,
        widths_3d,
    )
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

    print("raw Vv/Rv/av LROM parameters:", raw_lrom.n_complex_parameters)
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
    ax.set_title("Vv/Rv/av wavefunction reproduction")
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

            This notebook compares a traditional reduced-basis view and the RF-LROM view for the `l = 0`
            real Woods-Saxon scattering wavefunction from the legacy `40Ca(n,n)` benchmark. The ROSE
            interaction space is initialized with EIM and `l_max = 1`, as in the legacy teaching
            benchmark. The first section varies only `Vv`; every section after that varies all three
            real Woods-Saxon parameters: `Vv`, `Rv`, and `av`.
            """
        ),
        md(
            """
            ## Notebook Inputs And Legacy Comparison

            Why this section matters: benchmark notebooks are only scientifically useful if the
            initialization is visible. This table makes the isotope, projectile, mesh, reduced-basis
            sizes, training ranges, and legacy reference points explicit before any emulator is built.
            """
        ),
        code(setup),
        code(input_table),
        md(
            """
            ## Section 1. Parameter Varying Vv

            Why this section matters: changing only `Vv` mostly rescales the real Woods-Saxon depth,
            so it is the cleanest first check that ROSE/RBM coordinates, central least-squares
            coordinates, and the RF-LROM equation are talking about the same reduced space.

            ### 1.1 Samples And Full-Order Wavefunctions
            """
        ),
        code(vv_samples),
        code(vv_plot),
        code(vv_wave_plot),
        md(
            """
            ### 1.2 Reduced Basis And LS Floor

            Why this subsection matters: the LS projection is the best result this basis can produce.
            It separates basis truncation error from emulator error.
            """
        ),
        code(vv_basis),
        code(vv_basis_plot),
        md(
            """
            ### 1.3 RBM/ROSE vs LROM Coordinates

            Why this subsection matters: plotting both training and testing coefficients against `Vv`
            shows what the LROM actually learns and where it is being asked to predict.
            """
        ),
        code(vv_lrom),
        code(vv_coeff_plot),
        md(
            """
            ### 1.4 Wavefunction Reproduction

            Why this subsection matters: coefficient agreement is not enough by itself. The benchmark
            ultimately cares whether the reconstructed scattering wavefunction is accurate.
            """
        ),
        code(vv_wave_errors),
        code(vv_wave_error_plot),
        md(
            r"""
            ## Section 2. Three-Parameter LROM Equation And Predictor Selection

            Why this section matters: once `Vv`, `Rv`, and `av` all vary, the reduced coordinates are
            no longer described well by a single depth-like scalar. The central RF-LROM equation is
            written in transformed predictor coordinates,

            \[
            (I + p_1 M_1 + \cdots + p_K M_K)a =
            p_1 b_1 + \cdots + p_K b_K ,
            \]

            where the predictors may be raw parameters or operator-informed potential samples. ROSE/RBM
            still uses EIM for the reduced interaction, while LROM uses maxvol-selected potential
            predictors to choose informative radial locations.

            ### 2.1 Vv/Rv/av Training And Testing Samples
            """
        ),
        code(box_samples),
        code(box_plot),
        md(
            """
            ### 2.2 Raw-Parameter Transformed Equation

            Why this subsection matters: raw `Vv/Rv/av` predictors are the simplest transformed
            equation. They are a useful baseline before we ask whether potential samples carry better
            operator information.
            """
        ),
        code(raw_lrom),
        md(
            """
            ### 2.3 Maxvol Potential Predictors

            Why this subsection matters: radius and diffuseness move the Woods-Saxon surface, so
            selecting predictor locations from the potential variation gives LROM a physics-aware
            coordinate system instead of only raw parameter coordinates.
            """
        ),
        code(potential_predictors),
        code(predictor_plot),
        md(
            """
            ## Section 3. Three-Parameter Wavefunction Emulation Results

            Why this section matters: this is the actual benchmark question. Given held-out
            `Vv/Rv/av` samples, we compare the full-order wavefunction against the LS floor, ROSE/RBM,
            raw-parameter LROM, and maxvol-predictor LROM.

            ### 3.1 Wavefunction Error Summary
            """
        ),
        code(performance),
        code(performance_plot),
        md(
            """
            ### 3.2 Notebook 1 Takeaways

            Why this subsection matters: the final table keeps the one-parameter baseline and the
            three-parameter benchmark visible in one place, so the package result can be compared to
            the legacy notebook story.
            """
        ),
        code(takeaway),
    ]


def main() -> None:
    write_notebook(NOTEBOOK_PATH, notebook_cells())


if __name__ == "__main__":
    main()
