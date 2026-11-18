"""Plot cross-section, residual, outlier, and posterior comparisons."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SCRATCH = ROOT / "Scratch/CodeRestructure"
SOURCE = SCRATCH / "global_diagnostics/curated_neutron_map"
DATA = HERE / "data/likelihood_model_comparison"
OUT = HERE / "figures/likelihood_model_comparison"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRATCH))
from kd_all_coefficients import NAMES

MODELS = {
    "fixed10_data_gaussian": ("Gaussian, fixed 10%", "#4C78A8"),
    "inferred_data_gaussian": ("Gaussian, inferred data scale", "#59A14F"),
    "inferred_prediction_gaussian": ("Gaussian, inferred prediction scale", "#9C755F"),
    "inferred_kduq_gaussian": ("Gaussian, KDUQ scale", "#F28E2B"),
    "inferred_data_student_t4": (r"Student-$t_4$, inferred data scale", "#B04A6F"),
}
CASES = [
    (40, 20, "Ca", 13.90, 8, "#2864B7"),
    (54, 26, "Fe", 16.93, 6, "#2A9D8F"),
    (60, 28, "Ni", 13.94, 4, "#7A5195"),
    (120, 50, "Sn", 13.92, 2, "#E09F3E"),
    (208, 82, "Pb", 16.90, 0, "#C44536"),
]


def profile_normalizations(prediction, observed, sigma, record, tau, student=False):
    residual = prediction - observed
    inverse = 1.0 / sigma**2
    if student:
        eta = np.zeros(len(tau))
        for _ in range(30):
            adjusted = residual + eta[record] * prediction
            z2 = adjusted**2 * inverse
            weight = 5.0 / (4.0 + z2)
            numerator = np.bincount(
                record, weights=weight * prediction * residual * inverse,
                minlength=len(tau),
            )
            denominator = 1.0 / np.maximum(tau, 1e-8) ** 2
            denominator += np.bincount(
                record, weights=weight * prediction**2 * inverse,
                minlength=len(tau),
            )
            updated = -numerator / denominator
            if np.max(np.abs(updated - eta)) < 1e-10:
                eta = updated; break
            eta = updated
    else:
        projection = np.bincount(
            record, weights=prediction * inverse * residual, minlength=len(tau)
        )
        norm = np.bincount(
            record, weights=prediction**2 * inverse, minlength=len(tau)
        )
        eta = -tau**2 * projection / (1.0 + tau**2 * norm)
    adjusted = residual + eta[record] * prediction
    return eta, adjusted / sigma


archive = np.load(SOURCE / "curated_kduq_sampling_payload.npz")
points = pd.read_csv(SOURCE / "map_fit_points.csv")
verified = np.load(SOURCE / "refined_fom_verification.npz")
canonical = verified["canonical"]
observed = np.asarray(archive["observed"], float)
experimental = np.asarray(archive["sigma"], float)
record = np.asarray(archive["record_index"], int)
tau = np.asarray(archive["normalization_fraction"], float)
obs_case = np.asarray(archive["observation_case"], int)
case_a = np.asarray(archive["case_a"], int)
case_z = np.asarray(archive["case_z"], int)
case_e = np.asarray(archive["case_energy"], float)
angles = points.angle_cm_deg.to_numpy(float)
errors = points.point_error_mb_sr.to_numpy(float)

residual_rows = []
record_tables = {}
summaries = {}
posteriors = {}

for model, (label, method_color) in MODELS.items():
    model_dir = DATA / model
    predictive = np.load(model_dir / "posterior_predictive.npz")
    posterior_file = np.load(model_dir / "posterior.npz")
    summaries[model] = json.loads((model_dir / "summary.json").read_text())
    posterior = posterior_file["posterior"]
    posteriors[model] = posterior
    parameter = np.asarray(predictive["parameter_predictions"], float)
    fractions = np.asarray(predictive["fractions"], float)
    median = np.median(parameter, axis=0)
    fraction = float(np.median(fractions))
    if model == "inferred_kduq_gaussian":
        discrepancy_scale = 0.5 * (np.abs(observed) + median)
    elif model == "inferred_prediction_gaussian":
        discrepancy_scale = median
    else:
        discrepancy_scale = np.abs(observed)
    sigma = np.sqrt(experimental**2 + (fraction * discrepancy_scale) ** 2)
    _, z = profile_normalizations(
        median, observed, sigma, record, tau,
        student=(model == "inferred_data_student_t4"),
    )
    for value in z:
        residual_rows.append({"model": model, "z": value})
    local = points[["record_index", "record_id", "target", "energy_MeV"]].copy()
    local["z"] = z
    table = local.groupby("record_index", sort=False).agg(
        record_id=("record_id", "first"), target=("target", "first"),
        energy_MeV=("energy_MeV", "first"), points=("z", "size"),
        rms_z=("z", lambda x: float(np.sqrt(np.mean(x**2)))),
        max_abs_z=("z", lambda x: float(np.max(np.abs(x)))),
        tail_points=("z", lambda x: int(np.sum(np.abs(x) > 3))),
    ).reset_index().sort_values("rms_z", ascending=False)
    table.to_csv(DATA / f"records_{model}.csv", index=False)
    record_tables[model] = table

    # Generate observation-level predictive realizations consistently with
    # each fitted likelihood, including point errors and record normalization.
    rng = np.random.default_rng(3100 + list(MODELS).index(model))
    if model == "inferred_kduq_gaussian":
        scale_draws = 0.5 * (np.abs(observed)[None] + parameter)
    elif model == "inferred_prediction_gaussian":
        scale_draws = parameter
    else:
        scale_draws = np.broadcast_to(np.abs(observed)[None], parameter.shape)
    sigma_draws = np.sqrt(
        experimental[None] ** 2 + (fractions[:, None] * scale_draws) ** 2
    )
    normalization_draws = rng.normal(size=(len(parameter), len(tau))) * tau[None]
    if model == "inferred_data_student_t4":
        independent = rng.standard_t(4.0, size=parameter.shape) * sigma_draws
    else:
        independent = rng.normal(size=parameter.shape) * sigma_draws
    total = np.maximum(
        parameter * (1.0 + normalization_draws[:, record]) + independent,
        1e-30,
    )
    q025, q16, q84, q975 = np.quantile(total, [.025, .16, .84, .975], axis=0)

    fig, ax = plt.subplots(figsize=(6.6, 8.2), layout="constrained")
    for a, zed, symbol, wanted, exponent, color in CASES:
        candidates = np.flatnonzero((case_a == a) & (case_z == zed))
        case = candidates[np.argmin(np.abs(case_e[candidates] - wanted))]
        idx = np.flatnonzero(obs_case == case)
        idx = idx[np.argsort(angles[idx])]
        theta = angles[idx]; scale = 10.0**exponent
        ax.fill_between(theta, scale*q025[idx], scale*q975[idx], color=color, alpha=.18, lw=0)
        ax.fill_between(theta, scale*q16[idx], scale*q84[idx], color=color, alpha=.36, lw=0)
        ax.plot(theta, scale*median[idx], color=color, lw=2.0)
        ax.plot(theta, scale*canonical[idx], color=".20", lw=1.15, ls="--")
        ax.errorbar(theta, scale*observed[idx], yerr=scale*errors[idx], fmt="o", ms=3.3,
                    mfc="black", mec="black", ecolor=".35", elinewidth=.65, capsize=0, zorder=5)
        left_height = scale*q975[idx][0]*1.6
        right_height = scale*q975[idx][-1]*1.6
        ax.text(2.0, left_height, rf"$\times 10^{{{exponent}}}$", color=color,
                fontsize=13, ha="left", va="center")
        ax.text(177.0, right_height, rf"$^{{{a}}}\mathrm{{{symbol}}}$, {case_e[case]:g} MeV",
                color=color, fontsize=13, ha="right", va="center")
    ax.set_yscale("log"); ax.set_xlim(0, 180); ax.set_ylim(0.3, 3e11)
    ax.set_xlabel(r"$\theta_{\rm cm}$ [deg]", fontsize=16)
    ax.set_ylabel(r"scaled $d\sigma/d\Omega$ [mb/sr]", fontsize=16)
    ax.tick_params(axis="both", which="both", labelsize=13); ax.grid(False)
    ax.text(.50, .985, label, transform=ax.transAxes, ha="center", va="top",
            fontsize=10.5, color=method_color,
            bbox={"facecolor":"white", "edgecolor":method_color, "alpha":.92, "pad":2})
    handles = [
        Line2D([], [], color=".20", ls="--", lw=1.4, label="canonical KD (FOM)"),
        Line2D([], [], color=".30", lw=2, label="posterior median (LROM)"),
        Line2D([], [], marker="o", ls="none", color="black", markersize=4, label="curated data"),
        Patch(facecolor=".35", alpha=.36, label="68% posterior predictive"),
        Patch(facecolor=".35", alpha=.18, label="95% posterior predictive"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"cross_sections_{model}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Residual distributions and record-level tail concentration.
residual_frame = pd.DataFrame(residual_rows)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")
bins = np.linspace(-8, 8, 97)
for model, (label, color) in MODELS.items():
    z = residual_frame.loc[residual_frame.model == model, "z"].to_numpy()
    axes[0].hist(z, bins=bins, density=True, histtype="step", lw=1.8,
                 color=color, label=label)
normal_x = np.linspace(-5, 5, 401)
axes[0].plot(normal_x, np.exp(-normal_x**2/2)/np.sqrt(2*np.pi), "k--", lw=1,
             label=r"$\mathcal{N}(0,1)$")
axes[0].set(xlabel="profiled standardized residual", ylabel="density", yscale="log", ylim=(1e-4, 1))
axes[0].legend(fontsize=8); axes[0].grid(False)

top_ids = record_tables["fixed10_data_gaussian"].head(12).record_id.tolist()[::-1]
y = np.arange(len(top_ids)); width = .18
for offset, (model, (label, color)) in enumerate(MODELS.items()):
    lookup = record_tables[model].set_index("record_id").rms_z
    axes[1].barh(y + (offset-1.5)*width, [lookup.get(item, np.nan) for item in top_ids],
                 height=width, color=color, label=label)
axes[1].axvline(3, color=".25", ls="--", lw=1)
axes[1].set_yticks(y, [item.split(":")[2] + ", " + item.split(":")[3] + " MeV" for item in top_ids], fontsize=7)
axes[1].set(xlabel="record RMS standardized residual", ylabel="")
axes[1].grid(False)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"residual_and_outlier_comparison.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

# The inferred discrepancy itself is an important posterior diagnostic.
fig, ax = plt.subplots(figsize=(6.2, 3.6), layout="constrained")
for model in ("inferred_data_gaussian", "inferred_prediction_gaussian", "inferred_kduq_gaussian", "inferred_data_student_t4"):
    label, color = MODELS[model]
    fraction = 100.0 * np.exp(posteriors[model][:, -1])
    ax.hist(fraction, bins=50, density=True, histtype="step", lw=1.8,
            color=color, label=label)
x = np.linspace(0.01, 60, 500)
prior = np.exp(-0.5*((x/100-.20)/.20)**2)
prior /= np.trapezoid(prior, x)
ax.plot(x, prior, "k--", lw=1.2, label="KDUQ differential-error prior")
ax.set(xlabel="fractional unaccounted-for uncertainty [%]", ylabel="density", xlim=(0, 40))
ax.legend(fontsize=8); ax.grid(False)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"discrepancy_posterior.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

# Compact posterior summary: marginal shifts and widths in log-coefficient space.
fig, axes = plt.subplots(1, 2, figsize=(11.0, 7.8), sharey=True, layout="constrained")
y = np.arange(37)
offsets = np.linspace(-.27, .27, len(MODELS))
for offset, (model, (label, color)) in zip(offsets, MODELS.items()):
    posterior = posteriors[model][:, :37]
    q16, q50, q84 = np.quantile(posterior, [.16, .50, .84], axis=0)
    axes[0].errorbar(100*q50, y+offset, xerr=[100*(q50-q16), 100*(q84-q50)],
                     fmt="o", ms=3, capsize=0, color=color, label=label)
    axes[1].plot(100*(q84-q16)/2, y+offset, "o", ms=3, color=color)
axes[0].axvline(0, color=".35", lw=1)
axes[0].set(xlabel="posterior shift from canonical coefficient [%]", yticks=y,
            yticklabels=NAMES)
axes[1].set(xlabel="marginal posterior 1-sigma width [%]")
axes[0].invert_yaxis(); axes[0].legend(fontsize=8, loc="lower right")
for axis in axes: axis.grid(False); axis.tick_params(labelsize=8)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"coefficient_posterior_summary.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

comparison = {
    model: {
        **summaries[model],
        "residual_rms": float(np.sqrt(np.mean(
            residual_frame.loc[residual_frame.model == model, "z"].to_numpy()**2
        ))),
        "residual_fraction_abs_gt_3": float(np.mean(np.abs(
            residual_frame.loc[residual_frame.model == model, "z"].to_numpy()
        ) > 3)),
        "worst_record_rms": float(record_tables[model].rms_z.iloc[0]),
        "worst_record": str(record_tables[model].record_id.iloc[0]),
    }
    for model in MODELS
}
(DATA / "comparison_summary.json").write_text(json.dumps(comparison, indent=2))
print(OUT)
