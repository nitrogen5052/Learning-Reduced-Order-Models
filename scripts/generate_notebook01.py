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
            if (candidate / "lrom").is_dir()
        )
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        """
    ).strip()


def write_notebook(path: Path, cells: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    for index, cell in enumerate(cells):
        cell["id"] = f"notebook01-{index:02d}"
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(notebook, path)


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
            "for name in list(sys.modules):",
            '    if name == "lrom" or name.startswith("lrom."):',
            "        del sys.modules[name]",
            "",
            "import lrom",
            "",
            'plt.rcParams["figure.dpi"] = 130',
            'plt.rcParams["axes.grid"] = True',
            'plt.rcParams["grid.alpha"] = 0.25',
        ]
    )

    return [
        md(
            """
            # 01. ROSE and LROM for a Single Scattering Wavefunction

            This notebook is the first end-to-end demonstration of the stateful
            `lrom.LROM` API. It compares high-fidelity solutions of the nuclear
            scattering equation with ROSE/RBM and LROM approximations for the
            exact $l=0$ channel of $^{40}$Ca(n,n) at 14.1 MeV.

            All horizontal axes use physical radius $r$ in fm. The package owns
            numerical state; the short plotting cells remain visible here.
            """
        ),
        md(
            """
            ## Notebook inputs

            Two independent objects make the scientific distinction explicit:

            - `vv_emulator`: `ws_1`, a Vv-only linspace study with raw parameter predictors.
            - `ws3_emulator`: `ws_3`, a Vv/Rv/av Latin-hypercube study with six selected potential predictors.

            Both use a four-vector wavefunction basis and an eight-vector EIM basis.
            Training and testing domains are deliberately separate so interpolation
            and extrapolation behavior can be examined.
            """
        ),
        code(setup),
        md(
            """
            ## Section 1. Parameter Varying Vv

            The first object isolates the effect of the real Woods-Saxon depth.
            Its testing interval is wider than its training interval.
            """
        ),
        code(
            """
            vv_emulator = lrom.LROM(
                target=(40, 20),
                projectile=(1, 0),
                lab_energy=14.1,
                l=0,
                fom="nucl-scatter-eq",
                potential="ws_1",
            )
            vv_center = dict(vv_emulator.central_parameters)
            Vv0 = vv_center["Vv"]
            vv_training_ranges = {"Vv": (0.90 * Vv0, 1.10 * Vv0)}
            vv_testing_ranges = {"Vv": (0.65 * Vv0, 1.35 * Vv0)}

            vv_emulator.sampling(
                training_ranges=vv_training_ranges,
                testing_ranges=vv_testing_ranges,
                training_size=35,
                testing_size=41,
                mesh_size=800,
                strategy="linspace",
                seed=1204,
                eim_basis_size=8,
            )
            vv_emulator.train(
                basis_size=4,
                predictor="parameters",
                predictor_count=1,
            )

            print("central parameters:", dict(vv_emulator.central_parameters))
            print("training wavefunctions:", vv_emulator.samples.training_wavefunctions[0].shape)
            print("testing wavefunctions:", vv_emulator.samples.testing_wavefunctions[0].shape)
            """
        ),
        code(
            """
            r = vv_emulator.mesh.radius
            vv_values = vv_emulator.samples.design.training.values[:, 0]
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(vv_values)))
            for color, value, potential, phi in zip(
                colors,
                vv_values,
                vv_emulator.samples.training_potentials,
                vv_emulator.samples.training_wavefunctions[0],
            ):
                axes[0].plot(r, np.real(potential), color=color, alpha=0.55)
                axes[1].plot(r, np.real(phi), color=color, alpha=0.55)
            axes[0].set(xlabel="r [fm]", ylabel="V(r) [MeV]", title="Vv training potentials")
            axes[1].set(xlabel="r [fm]", ylabel="Re(phi)", title="High-fidelity training solutions")
            fig.colorbar(
                plt.cm.ScalarMappable(
                    norm=plt.Normalize(vv_values.min(), vv_values.max()), cmap="viridis"
                ),
                ax=axes,
                label="Vv [MeV]",
            )
            plt.show()
            """
        ),
        code(
            """
            basis = vv_emulator.basis[0]
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            for index in range(basis.basis_size):
                axes[0].plot(r, np.real(basis.vectors[:, index]), label=f"basis {index + 1}")

            vv_test = vv_emulator.samples.design.testing.values[:, 0]
            coefficients = vv_emulator.testing_results.coefficients
            for method, style in (("ls", "-"), ("rose", "--"), ("lrom", ":")):
                axes[1].plot(
                    vv_test,
                    np.real(coefficients[method][0][:, 0]),
                    style,
                    label=method.upper(),
                )
            axes[0].set(xlabel="r [fm]", ylabel="Re(basis vector)", title="Shared wavefunction basis")
            axes[1].set(xlabel="Vv [MeV]", ylabel="Re(first coordinate)", title="Shared-basis coordinates")
            axes[0].legend()
            axes[1].legend()
            plt.show()
            """
        ),
        code(
            """
            vv_errors = vv_emulator.testing_errors[0]
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            for method, color in (("rose", "red"), ("lrom", "orange"), ("ls", "blue")):
                for error in vv_errors[method]:
                    ax.plot(r, np.maximum(error, 1e-14), color=color, alpha=0.16)
            ax.set_yscale("log")
            ax.set_xlabel("r [fm]")
            ax.set_ylabel("absolute difference")
            ax.set_title("Vv-only testing errors: ROSE (red), LROM (orange), LS (blue)")
            plt.show()
            """
        ),
        md(
            """
            ## Section 2. Three-Parameter LROM Equation And Predictor Selection

            The second object varies Vv, Rv, and av together. Its default potential
            predictors are values of the Woods-Saxon potential at six physical radii
            selected from the training ensemble by SVD and maxvol-style selection.

            The transformed equation has the form
            $(I + p_1M_1 + \\cdots + p_KM_K)a = b_0 + \\sum_k p_kb_k$.
            """
        ),
        code(
            """
            ws3_emulator = lrom.LROM(
                target=(40, 20),
                projectile=(1, 0),
                lab_energy=14.1,
                l=0,
                fom="nucl-scatter-eq",
                potential="ws_3",
            )
            ws3_center = dict(ws3_emulator.central_parameters)
            ws3_training_ranges = {
                name: (0.90 * ws3_center[name], 1.10 * ws3_center[name])
                for name in ("Vv", "Rv", "av")
            }
            ws3_testing_ranges = {
                "Vv": (0.78 * ws3_center["Vv"], 1.22 * ws3_center["Vv"]),
                "Rv": (0.80 * ws3_center["Rv"], 1.20 * ws3_center["Rv"]),
                "av": (0.80 * ws3_center["av"], 1.20 * ws3_center["av"]),
            }

            ws3_emulator.sampling(
                training_ranges=ws3_training_ranges,
                testing_ranges=ws3_testing_ranges,
                training_size=70,
                testing_size=81,
                mesh_size=900,
                strategy="latin_hypercube",
                seed=1204,
                eim_basis_size=8,
            )
            ws3_emulator.train(
                basis_size=4,
                predictor="potential",
                predictor_count=6,
            )
            print("potential predictor radii [fm]:", ws3_emulator.predictors.selected_radii)
            """
        ),
        code(
            """
            r3 = ws3_emulator.mesh.radius
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            for potential in ws3_emulator.samples.training_potentials:
                axes[0].plot(r3, np.real(potential), color="slateblue", alpha=0.12)
            axes[0].plot(r3, np.real(ws3_emulator.samples.central_potential), color="black", label="central")
            selected_radii = ws3_emulator.predictors.selected_radii
            selected_values = ws3_emulator.samples.central_potential[
                ws3_emulator.predictors.selected_indices
            ]
            axes[0].scatter(
                selected_radii,
                np.real(selected_values),
                color="crimson",
                zorder=5,
                label="selected potential predictor points",
            )
            axes[0].set(xlabel="r [fm]", ylabel="V(r) [MeV]", title="ws_3 potential predictors")
            axes[0].legend()

            for index, singular_value in enumerate(ws3_emulator.predictors.singular_values[:10], start=1):
                axes[1].semilogy(index, singular_value, "o", color="slateblue")
            axes[1].set(xlabel="potential mode", ylabel="singular value", title="Potential-ensemble spectrum")
            plt.show()
            """
        ),
        code(
            """
            ws3_basis = ws3_emulator.basis[0]
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            for index in range(ws3_basis.basis_size):
                axes[0].plot(r3, np.real(ws3_basis.vectors[:, index]), label=f"basis {index + 1}")
            coefficients = ws3_emulator.testing_results.coefficients
            case_number = np.arange(ws3_emulator.samples.design.testing.values.shape[0])
            for method, style in (("ls", "-"), ("rose", "--"), ("lrom", ":")):
                axes[1].plot(
                    case_number,
                    np.real(coefficients[method][0][:, 0]),
                    style,
                    alpha=0.8,
                    label=method.upper(),
                )
            axes[0].set(xlabel="r [fm]", ylabel="Re(basis vector)", title="ws_3 shared basis")
            axes[1].set(xlabel="testing case", ylabel="Re(first coordinate)", title="ws_3 shared-basis coordinates")
            axes[0].legend()
            axes[1].legend()
            plt.show()
            """
        ),
        md(
            """
            ## Section 3. Three-Parameter Wavefunction Emulation Results

            We now inspect one representative testing solution and the pointwise
            absolute differences over all 81 testing cases. The least-squares curve
            is the attainable floor for this fixed basis.
            """
        ),
        code(
            """
            lrom_relative = ws3_emulator.testing_results.metrics["relative_l2"][0]["lrom"]
            representative_index = int(np.argsort(lrom_relative)[len(lrom_relative) // 2])
            representative_id = ws3_emulator.samples.design.testing.case_ids[representative_index]
            case = ws3_emulator.testing_case(case_id=representative_id)

            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            ax.plot(case.radius, np.real(case.high_fidelity[0]), color="black", label="high fidelity")
            ax.plot(case.radius, np.real(case.rose[0]), "--", color="red", label="ROSE")
            ax.plot(case.radius, np.real(case.lrom[0]), ":", color="orange", linewidth=2, label="LROM")
            ax.set_xlabel("r [fm]")
            ax.set_ylabel("Re(phi)")
            ax.set_title(f"Representative l=0 testing solution: {representative_id}")
            ax.legend()
            plt.show()
            """
        ),
        code(
            """
            ws3_errors = ws3_emulator.testing_errors[0]
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for method, color in (("rose", "red"), ("lrom", "orange"), ("ls", "blue")):
                for error in ws3_errors[method]:
                    ax.plot(r3, np.maximum(error, 1e-14), color=color, alpha=0.13)
            ax.set_yscale("log")
            ax.set_xlabel("r [fm]")
            ax.set_ylabel("absolute difference")
            ax.set_title("All ws_3 testing cases: ROSE (red), LROM (orange), LS (blue)")
            plt.show()
            """
        ),
        code(
            """
            test_values = ws3_emulator.samples.design.testing.values
            names = ws3_emulator.parameter_names
            interpolation = np.ones(len(test_values), dtype=bool)
            for column, name in enumerate(names):
                low, high = ws3_training_ranges[name]
                interpolation &= (test_values[:, column] >= low) & (test_values[:, column] <= high)

            rows = []
            metrics = ws3_emulator.testing_results.metrics["relative_l2"][0]
            for region, mask in (("interpolation", interpolation), ("extrapolation", ~interpolation)):
                for method in ("rose", "lrom", "ls"):
                    values = metrics[method][mask]
                    rows.append(
                        {
                            "region": region,
                            "method": method.upper(),
                            "cases": int(mask.sum()),
                            "median relative L2": float(np.median(values)),
                            "maximum relative L2": float(np.max(values)),
                        }
                    )
            pd.DataFrame(rows).set_index(["region", "method"])
            """
        ),
        code(
            """
            artifact_path = ROOT / "outputs" / "notebook01_ws3_model.lrom"
            ws3_emulator.save(path=artifact_path)
            portable_emulator = lrom.load(path=artifact_path)
            portable_emulator.predict(parameters=case.parameters)
            print("portable model:", artifact_path)
            print("prediction shape:", portable_emulator.predictions.wavefunctions[0].shape)
            """
        ),
    ]


def main() -> None:
    write_notebook(NOTEBOOK_PATH, notebook_cells())
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
