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
    rho_mesh = np.linspace(1e-8, 8 * np.pi, cfg.n_mesh)
    r_mesh = rho_mesh / params.k

    print("config hash:", cfg.config_hash())
    print("central [Vv, Rv, av]:", alpha_c)
    """
    vv_samples = r"""
    vv_train = sampling.centered_1d_values(alpha_c[0], cfg.vv_width, cfg.n_vv_train)
    vv_test = sampling.centered_1d_values(alpha_c[0], cfg.vv_width, cfg.n_vv_test)

    train_alphas = rose_fom.make_alphas(alpha_c, param_index=0, values=vv_train)
    test_alphas = rose_fom.make_alphas(alpha_c, param_index=0, values=vv_test)

    potentials_train = np.array(
        [rose_fom.real_woods_saxon_potential(r_mesh, alpha) for alpha in train_alphas]
    )
    """
    vv_plot = r"""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    for value, potential in zip(vv_train, potentials_train):
        label = f"{value:.1f}" if value in vv_train[[0, -1]] else None
        ax.plot(r_mesh, potential, alpha=0.55, label=label)
    ax.set_xlabel("r [fm]")
    ax.set_ylabel("V(r)")
    ax.set_title("Real Woods-Saxon potential rainbow: Vv-only scan")
    ax.grid(alpha=0.25)
    ax.legend(title="Vv edge values")
    fig.tight_layout()
    """
    basis_placeholder = r"""
    # The first executable implementation slice wires ROSE wavefunctions here.
    # Once phi_train is available:
    #
    # phi0 = problem.solve_phi(alpha_c)
    # basis = reduced_basis.build_centered_svd_basis(phi0, phi_train, rho_mesh, cfg.n_basis)
    # coeff_ls_train = reduced_basis.project_ls_coordinates(basis, phi_train)
    # phi_ls_train = prediction.reconstruct_from_basis(basis.phi0, basis.vectors, coeff_ls_train)
    # ls_error = metrics.relative_l2_rows(phi_ls_train, phi_train)
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
        md("## 3. Reduced Basis And LS Floor"),
        code(basis_placeholder),
        md("## 4. RBM/ROSE vs LROM Coordinates"),
        code("# Fit RF-LROM after ROSE/RBM coefficients and LS coordinates are available."),
        md("## 5. Wavefunction Reproduction"),
        code("# Compare LS floor, ROSE/RBM, and LROM wavefunction errors here."),
        md("## 6. Three-Parameter Samples"),
        code("# Build centered-box samples for [Vv, Rv, av] and add Vv/Rv/av rainbow summaries."),
        md("## 7. Why Raw Parameters Are Not Enough"),
        code("# Fit and inspect a raw-parameter LROM diagnostic without hiding the flow."),
        md("## 8. Operator-Informed Potential Predictors"),
        code("# Build maxvol-selected potential predictors and visualize selected points inline."),
        md("## 9. Three-Parameter Performance"),
        code("# Compare LS floor, ROSE/RBM, raw-parameter LROM, and potential-predictor LROM."),
        md("## 10. Notebook 1 Takeaways"),
        code("pd.DataFrame({'next': ['Notebook 2 moves to cross-section-level comparisons.']})"),
    ]


def main() -> None:
    write_notebook(NOTEBOOK_PATH, notebook_cells())


if __name__ == "__main__":
    main()
