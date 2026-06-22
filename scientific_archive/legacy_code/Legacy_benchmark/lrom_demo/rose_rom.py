"""Phase 2: traditional ROSE reduced-basis emulator benchmarks.

This script follows the local ROSE tutorials:

* docs/tutorials/ROSE_tutorial_1_building_an_emulator.ipynb
* docs/tutorials/ROSE_tutorial_2_optical_potential_surmise_UQ.ipynb

It trains ROSE's standard reduced-basis/Galerkin/EIM emulator for
40Ca(n,n) near 14 MeV, then compares high-fidelity and emulated
wavefunctions and differential cross sections.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import sys
import time

from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
# Portable demo version: use the public PyPI package nuclear-rose, which imports as rose.
# Install with: pip install nuclear-rose

import rose  # noqa: E402
from rose.training import sample_params_LHC  # noqa: E402


TARGET = (40, 20)
PROJECTILE = (1, 0)
E_LAB = 14.1
L_MAX = 12
N_TRAIN = 50
N_TEST = 12
PARAMETER_SCALE = 0.20
N_PHI = 12
N_U = 12
SEED_TRAIN = 20260515
SEED_TEST = 20260516
KD_PARAMETER_NAMES = [
    "Vv",
    "Rv",
    "av",
    "Wv",
    "Rwv",
    "awv",
    "Wd",
    "Rd",
    "ad",
    "Vso",
    "Rso",
    "aso",
    "Wso",
    "Rwso",
    "awso",
]

OUT_DIR = ROOT / "outputs" / "phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Phase2Config:
    target: tuple[int, int] = TARGET
    projectile: tuple[int, int] = PROJECTILE
    e_lab: float = E_LAB
    l_max: int = L_MAX
    n_train: int = N_TRAIN
    n_test: int = N_TEST
    parameter_scale: float = PARAMETER_SCALE
    n_phi: int = N_PHI
    n_U: int = N_U
    seed_train: int = SEED_TRAIN
    seed_test: int = SEED_TEST
    rk_tols: tuple[float, float] = (1e-9, 1e-9)
    s_0_factor: float = 6 * np.pi
    domain_min: float = 1e-8
    domain_max: float = 8 * np.pi
    n_mesh: int = 1800
    n_angles: int = 240


def central_kd_parameters(config: Phase2Config):
    """Return kinematics and ROSE's 15-component KD parameter vector."""
    mu, e_com, k, eta = rose.kinematics(
        target=config.target,
        projectile=config.projectile,
        E_lab=config.e_lab,
    )
    kd = rose.koning_delaroche.KDGlobal(rose.Projectile.neutron)
    r_c, alpha = kd.get_params(config.target[0], config.target[1], mu, config.e_lab, k)
    return mu, e_com, k, eta, r_c, alpha


def build_training_problem(config: Phase2Config):
    """Construct interactions, samples, solver, and angular/s meshes."""
    mu, e_com, k, eta, r_c, alpha_central = central_kd_parameters(config)
    bounds = np.array(
        [
            alpha_central - np.abs(alpha_central * config.parameter_scale),
            alpha_central + np.abs(alpha_central * config.parameter_scale),
        ]
    ).T

    training_samples = sample_params_LHC(
        config.n_train,
        alpha_central,
        scale=config.parameter_scale,
        seed=config.seed_train,
    )
    test_samples = sample_params_LHC(
        config.n_test,
        alpha_central,
        scale=config.parameter_scale,
        seed=config.seed_test,
    )

    rho_mesh = np.linspace(config.domain_min, config.domain_max, config.n_mesh)
    angles = np.linspace(1, 179, config.n_angles) * np.pi / 180

    interactions = rose.InteractionEIMSpace(
        l_max=config.l_max,
        coordinate_space_potential=rose.koning_delaroche.KD_simple,
        n_theta=alpha_central.size,
        mu=mu,
        energy=e_com,
        is_complex=True,
        spin_orbit_term=rose.koning_delaroche.KD_simple_so,
        training_info=bounds,
        n_basis=config.n_U,
        rho_mesh=rho_mesh,
    )
    base_solver = rose.SchroedingerEquation.make_base_solver(
        s_0=config.s_0_factor,
        rk_tols=list(config.rk_tols),
        domain=np.array([config.domain_min, config.domain_max]),
    )
    return {
        "mu": mu,
        "e_com": e_com,
        "k": k,
        "eta": eta,
        "r_c": r_c,
        "alpha_central": alpha_central,
        "bounds": bounds,
        "training_samples": training_samples,
        "test_samples": test_samples,
        "rho_mesh": rho_mesh,
        "angles": angles,
        "interactions": interactions,
        "base_solver": base_solver,
    }


def train_emulator(config: Phase2Config, problem: dict):
    """Train ROSE's standard scattering amplitude emulator."""
    emulator = rose.ScatteringAmplitudeEmulator.from_train(
        problem["interactions"],
        problem["training_samples"],
        base_solver=problem["base_solver"],
        l_max=config.l_max,
        angles=problem["angles"],
        n_basis=config.n_phi,
        use_svd=True,
        scale=False,
        s_mesh=problem["rho_mesh"],
        Smatrix_abs_tol=1e-8,
    )
    return emulator


def align_phase(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Match the arbitrary complex normalization/phase of a wavefunction."""
    idx = np.argmax(np.abs(reference))
    if np.abs(candidate[idx]) > 0:
        return candidate * (reference[idx] / candidate[idx])
    return candidate


def wavefunction_comparison(emulator, problem: dict, config: Phase2Config | None = None):
    """ROSE Fig. 1-style S-wave HF vs RBM comparison."""
    rho_mesh = problem["rho_mesh"]
    k = problem["k"]
    r_mesh = rho_mesh / k
    test_samples = problem["test_samples"][:4]
    rbe_l0 = emulator.rbes[0][0]
    basis = rbe_l0.basis

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    errors = []
    for i, alpha in enumerate(test_samples):
        phi_hf = rbe_l0.basis.solver.phi(alpha, rho_mesh)
        phi_rbm = align_phase(phi_hf, rbe_l0.emulate_wave_function(alpha))
        rel_err = np.linalg.norm(phi_hf - phi_rbm) / np.linalg.norm(phi_hf)
        errors.append(rel_err)

        line = ax.plot(r_mesh, phi_hf.real, lw=2.0, label=f"HF {i + 1}")[0]
        ax.plot(r_mesh, phi_rbm.real, "--", color="black", lw=1.5, alpha=0.9)

    ax.set_xlabel(r"$r$ [fm]")
    ax.set_ylabel(r"Re $\phi_{l=0}(r)$")
    ax.set_title(
        rf"ROSE RBM wavefunctions: $^{{40}}$Ca$(n,n)$, "
        rf"$n_\phi={getattr(config, 'n_phi', basis.vectors.shape[1])}$"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    n_basis_plot = min(4, basis.vectors.shape[1])
    phi0_amp = float(np.max(np.abs(basis.phi_0.real)))
    basis_display_height = 0.65 * phi0_amp
    ax2.plot(r_mesh, basis.phi_0.real, color="black", lw=1.5, alpha=0.35, label=r"$\phi_0$")
    for i in range(n_basis_plot):
        vec = basis.vectors[:, i].real
        display_scale = basis_display_height / max(float(np.max(np.abs(vec))), 1e-14)
        ax2.plot(
            r_mesh,
            display_scale * vec,
            lw=1.7,
            label=rf"PCA {i + 1} $\times {display_scale:.1f}$",
        )
    ax2.set_xlabel(r"$r$ [fm]")
    ax2.set_ylabel(r"display-scaled real component")
    ax2.set_title("S-wave reduced basis components")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)
    fig2.tight_layout()

    return {
        "figure": fig,
        "basis_figure": fig2,
        "relative_errors": np.array(errors),
    }


def select_cross_section_cases(emulator, problem: dict):
    """Pick three test cases with visibly different HF cross sections."""
    xs_all = []
    for alpha in problem["test_samples"]:
        xs_all.append(emulator.exact_xs(alpha).dsdo)
    xs_all = np.array(xs_all)

    # Greedy selection: lowest integrated strength, highest, and one in the middle.
    strengths = np.trapezoid(np.log10(xs_all + 1e-30), problem["angles"], axis=1)
    order = np.argsort(strengths)
    if len(order) >= 3:
        chosen = np.array([order[0], order[len(order) // 2], order[-1]])
    else:
        chosen = order
    return chosen, xs_all


def cross_section_comparison(emulator, problem: dict):
    """ROSE Fig. 4(a)-style HF vs RBM differential cross section comparison."""
    angles = problem["angles"]
    chosen, xs_hf_all = select_cross_section_cases(emulator, problem)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    rel_errors = []
    phase_errors = []
    hf_times = []
    rbm_times = []
    chosen_alphas = []

    for i, idx in enumerate(chosen):
        alpha = problem["test_samples"][idx]
        chosen_alphas.append(alpha)

        t0 = time.perf_counter()
        xs_hf_obj = emulator.exact_xs(alpha)
        hf_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        xs_rbm_obj = emulator.emulate_xs(alpha)
        rbm_times.append(time.perf_counter() - t0)

        xs_hf = xs_hf_obj.dsdo
        xs_rbm = xs_rbm_obj.dsdo
        rel_errors.append(np.max(np.abs(xs_hf - xs_rbm) / np.maximum(xs_hf, 1e-30)))

        deltas_hf = emulator.exact_phase_shifts(alpha)
        deltas_rbm = emulator.emulate_phase_shifts(alpha)
        phase_errors.append(abs(deltas_hf[0][0] - deltas_rbm[0][0]))

        line = ax.plot(angles * 180 / np.pi, xs_hf, lw=2.0, label=f"case {i + 1}")[0]
        case_color = np.array(to_rgb(line.get_color()))
        rose_color = tuple(0.40 * case_color)
        ax.plot(
            angles * 180 / np.pi,
            xs_rbm,
            "--",
            color=rose_color,
            lw=1.6,
            alpha=0.95,
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta_{\rm cm}$ [deg]")
    ax.set_ylabel(r"$d\sigma/d\Omega$ [mb/sr]")
    ax.set_title(r"ROSE RBM vs HF: $^{40}$Ca$(n,n)$ at 14.1 MeV")
    ax.grid(True, which="both", alpha=0.25)
    case_legend = ax.legend(fontsize=8, title="parameter case", loc="upper right")
    ax.add_artist(case_legend)
    style_handles = [
        Line2D([0], [0], color="0.25", lw=2.0, label="HF"),
        Line2D([0], [0], color="0.25", lw=1.7, ls="--", label="ROSE"),
    ]
    ax.legend(handles=style_handles, fontsize=8, loc="lower left")
    fig.tight_layout()

    chosen_alphas = np.asarray(chosen_alphas)
    central = problem["alpha_central"]
    return {
        "figure": fig,
        "chosen_indices": chosen,
        "xs_hf_all": xs_hf_all,
        "parameter_names": KD_PARAMETER_NAMES,
        "case_parameter_values": chosen_alphas,
        "case_parameter_percent_changes": 100 * (chosen_alphas - central) / np.maximum(np.abs(central), 1e-30),
        "cross_section_max_relative_errors": np.array(rel_errors),
        "phase_l0_abs_errors": np.array(phase_errors),
        "hf_times": np.array(hf_times),
        "rbm_times": np.array(rbm_times),
    }


def diagnostic_plots(emulator, problem: dict):
    """Singular-value and conditioning diagnostics."""
    rbe_l0 = emulator.rbes[0][0]
    basis = rbe_l0.basis
    eim_l0 = emulator.rbes[0][0].interaction

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=180)
    ax.semilogy(np.arange(1, len(basis.singular_values) + 1), basis.singular_values, "o-")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.set_title("Wavefunction snapshot singular values, l=0")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "singular_values_wavefunction_l0.png")

    fig2, ax2 = plt.subplots(figsize=(6.2, 4.0), dpi=180)
    ax2.semilogy(np.arange(1, len(eim_l0.singular_values) + 1), eim_l0.singular_values, "o-")
    ax2.set_xlabel("index")
    ax2.set_ylabel("singular value")
    ax2.set_title("EIM potential singular values, l=0")
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "singular_values_eim_potential_l0.png")

    conds = []
    for alpha in problem["test_samples"]:
        invk, beta = rbe_l0.interaction.coefficients(alpha)
        a_utilde = np.einsum("i,ijk", beta, rbe_l0.A_2)
        a_matrix = rbe_l0.A_13 + a_utilde + invk * rbe_l0.A_3_coulomb
        conds.append(np.linalg.cond(a_matrix))
    return np.array(conds)


def save_summary(config, problem, train_seconds, wf_errors, xs_diag, conds):
    summary = {
        "config": asdict(config),
        "central_alpha_15": problem["alpha_central"].tolist(),
        "training_shape": list(problem["training_samples"].shape),
        "test_shape": list(problem["test_samples"].shape),
        "train_seconds": train_seconds,
        "wavefunction_l2_relative_errors": wf_errors.tolist(),
        "cross_section_max_relative_errors": xs_diag["cross_section_max_relative_errors"].tolist(),
        "phase_l0_abs_errors": [complex(x).__repr__() for x in xs_diag["phase_l0_abs_errors"]],
        "hf_times_seconds": xs_diag["hf_times"].tolist(),
        "rbm_times_seconds": xs_diag["rbm_times"].tolist(),
        "mean_hf_time_seconds": float(np.mean(xs_diag["hf_times"])),
        "mean_rbm_time_seconds": float(np.mean(xs_diag["rbm_times"])),
        "speedup_factor": float(np.mean(xs_diag["hf_times"]) / np.mean(xs_diag["rbm_times"])),
        "l0_reduced_matrix_condition_numbers": conds.tolist(),
    }
    with open(OUT_DIR / "phase2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        OUT_DIR / "phase2_diagnostics.npz",
        central_alpha=problem["alpha_central"],
        training_samples=problem["training_samples"],
        test_samples=problem["test_samples"],
        wavefunction_l2_relative_errors=wf_errors,
        cross_section_max_relative_errors=xs_diag["cross_section_max_relative_errors"],
        phase_l0_abs_errors=xs_diag["phase_l0_abs_errors"],
        hf_times=xs_diag["hf_times"],
        rbm_times=xs_diag["rbm_times"],
        l0_condition_numbers=conds,
    )
    return summary


def main():
    config = Phase2Config()
    print("Building Phase 2 ROSE RBM problem")
    problem = build_training_problem(config)
    print("central alpha:")
    print(problem["alpha_central"])
    print(f"training samples: {problem['training_samples'].shape}")
    print(f"test samples: {problem['test_samples'].shape}")
    print(f"n_phi={config.n_phi}, n_U={config.n_U}")

    start = time.perf_counter()
    emulator = train_emulator(config, problem)
    train_seconds = time.perf_counter() - start
    print(f"training seconds: {train_seconds:.3f}")

    wf_diag = wavefunction_comparison(emulator, problem, config)
    wf_errors = wf_diag["relative_errors"]
    print("wavefunction relative errors:", wf_errors)

    xs_diag = cross_section_comparison(emulator, problem)
    print("cross-section max relative errors:", xs_diag["cross_section_max_relative_errors"])
    print("l=0 phase abs errors:", xs_diag["phase_l0_abs_errors"])
    print("HF times:", xs_diag["hf_times"])
    print("RBM times:", xs_diag["rbm_times"])

    conds = diagnostic_plots(emulator, problem)
    print("l=0 reduced matrix condition numbers:", conds)

    summary = save_summary(config, problem, train_seconds, wf_errors, xs_diag, conds)
    print(f"speedup factor: {summary['speedup_factor']:.1f}")
    print(f"outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
