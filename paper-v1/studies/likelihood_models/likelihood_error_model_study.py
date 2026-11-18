"""Compare Gaussian and robust discrepancy models at the refined KD MAP.

This is a diagnostic study, separate from the paper notebooks.  It asks how
much extra fractional scatter is supported once the normalization nuisances
and the likelihood normalization are treated consistently, and whether the
large residuals are diffuse or concentrated in a few experimental records.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "Scratch/CodeRestructure"
SOURCE_DIR = SCRATCH / "global_diagnostics/curated_neutron_map"
OUT = ROOT / "PaperNotebooks/data/likelihood_error_model_study"
FIGURES = ROOT / "PaperNotebooks/figures/likelihood_error_model_study"
sys.path.insert(0, str(SCRATCH))

from run_kd_coefficient_calibration_jax import build_model


def gaussian_profile(prediction, observed, sigma, record, tau):
    """Gaussian normalization-nuisance mode and marginalized NLL pieces."""
    residual = prediction - observed
    inverse = 1.0 / sigma**2
    count = len(tau)
    projection = np.bincount(
        record, weights=prediction * inverse * residual, minlength=count
    )
    norm = np.bincount(
        record, weights=prediction**2 * inverse, minlength=count
    )
    denominator = 1.0 + tau**2 * norm
    eta = -tau**2 * projection / denominator
    adjusted = residual + eta[record] * prediction
    quadratic = np.sum(residual**2 * inverse) - np.sum(
        tau**2 * projection**2 / denominator
    )
    logdet = np.sum(np.log(sigma**2)) + np.sum(np.log(denominator))
    return eta, adjusted, float(quadratic), float(logdet)


def robust_profile(prediction, observed, sigma, record, tau, nu, iterations=30):
    """IRLS profile of record normalizations for a Student-t likelihood."""
    residual = prediction - observed
    eta = np.zeros(len(tau))
    inverse = 1.0 / sigma**2
    for _ in range(iterations):
        adjusted = residual + eta[record] * prediction
        z2 = adjusted**2 * inverse
        weight = (nu + 1.0) / (nu + z2)
        numerator = np.bincount(
            record, weights=weight * prediction * residual * inverse,
            minlength=len(tau),
        )
        denominator = 1.0 / tau**2 + np.bincount(
            record, weights=weight * prediction**2 * inverse,
            minlength=len(tau),
        )
        updated = -numerator / denominator
        if np.max(np.abs(updated - eta)) < 1e-10:
            eta = updated
            break
        eta = updated
    adjusted = residual + eta[record] * prediction
    z2 = adjusted**2 / sigma**2
    # Keep the normalization constant so different values of nu can be
    # compared on the same likelihood scale.
    normalization = (
        0.5 * np.log(nu * np.pi)
        + gammaln(0.5 * nu)
        - gammaln(0.5 * (nu + 1.0))
    )
    nll = np.sum(
        np.log(sigma)
        + normalization
        + 0.5 * (nu + 1.0) * np.log1p(z2 / nu)
    )
    nll += 0.5 * np.sum((eta / tau) ** 2)
    return eta, adjusted, float(nll)


def record_table(points, record, standardized):
    frame = points.copy()
    frame["standardized_residual"] = standardized
    grouped = frame.groupby("record_index", sort=False)
    table = grouped.agg(
        record_id=("record_id", "first"),
        target=("target", "first"),
        energy_MeV=("energy_MeV", "first"),
        points=("record_index", "size"),
        rms_z=("standardized_residual", lambda x: float(np.sqrt(np.mean(x**2)))),
        max_abs_z=("standardized_residual", lambda x: float(np.max(np.abs(x)))),
        chi2=("standardized_residual", lambda x: float(np.sum(x**2))),
    ).reset_index()
    return table.sort_values("chi2", ascending=False)


def main():
    jax.config.update("jax_enable_x64", True)
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    source = np.load(SOURCE_DIR / "curated_kduq_sampling_payload.npz")
    center = np.asarray(np.load(SOURCE_DIR / "refined_map.npz")["theta"], float)
    predict, _ = build_model(source, parameterization="all", optical_box="hard")
    map_prediction = np.asarray(jax.jit(predict)(jnp.asarray(center[None])))[0]
    observed = np.asarray(source["observed"], float)
    experimental = np.asarray(source["sigma"], float)
    record = np.asarray(source["record_index"], int)
    tau = np.asarray(source["normalization_fraction"], float)
    points = pd.read_csv(SOURCE_DIR / "map_fit_points.csv")

    fractions = np.linspace(0.0, 0.60, 121)
    gaussian_scans = {}
    for scale_name, scale in {
        "data": np.abs(observed),
        "prediction": np.abs(map_prediction),
    }.items():
        nll = []
        chi2 = []
        for fraction in fractions:
            sigma = np.sqrt(experimental**2 + (fraction * scale) ** 2)
            _, _, quadratic, logdet = gaussian_profile(
                map_prediction, observed, sigma, record, tau
            )
            nll.append(0.5 * (quadratic + logdet + len(observed) * np.log(2 * np.pi)))
            chi2.append(quadratic / len(observed))
        gaussian_scans[scale_name] = {
            "nll": np.asarray(nll),
            "chi2": np.asarray(chi2),
            "fraction": float(fractions[np.argmin(nll)]),
        }
    gaussian_fraction = gaussian_scans["data"]["fraction"]

    robust_results = {}
    for nu in (3.0, 4.0, 6.0, 10.0, 30.0):
        values = []
        for fraction in fractions:
            sigma = np.sqrt(experimental**2 + (fraction * observed) ** 2)
            _, _, nll = robust_profile(
                map_prediction, observed, sigma, record, tau, nu
            )
            values.append(nll)
        values = np.asarray(values)
        robust_results[nu] = {
            "nll": values,
            "fraction": float(fractions[np.argmin(values)]),
        }

    # Compare residual concentration for the reported-error model, the
    # likelihood-optimal Gaussian discrepancy, and the best Student-t model.
    _, baseline_adjusted, baseline_q, _ = gaussian_profile(
        map_prediction, observed, experimental, record, tau
    )
    sigma_fixed_ten = np.sqrt(experimental**2 + (0.10 * observed) ** 2)
    _, fixed_ten_adjusted, fixed_ten_q, _ = gaussian_profile(
        map_prediction, observed, sigma_fixed_ten, record, tau
    )
    sigma_gaussian = np.sqrt(experimental**2 + (gaussian_fraction * observed) ** 2)
    _, gaussian_adjusted, gaussian_q, _ = gaussian_profile(
        map_prediction, observed, sigma_gaussian, record, tau
    )
    best_nu = min(
        robust_results,
        key=lambda nu: np.min(robust_results[nu]["nll"]),
    )
    robust_fraction = robust_results[best_nu]["fraction"]
    sigma_robust = np.sqrt(experimental**2 + (robust_fraction * observed) ** 2)
    _, robust_adjusted, robust_nll = robust_profile(
        map_prediction, observed, sigma_robust, record, tau, best_nu
    )

    z_baseline = baseline_adjusted / experimental
    z_fixed_ten = fixed_ten_adjusted / sigma_fixed_ten
    z_gaussian = gaussian_adjusted / sigma_gaussian
    z_robust = robust_adjusted / sigma_robust
    baseline_records = record_table(points, record, z_baseline)
    fixed_ten_records = record_table(points, record, z_fixed_ten)
    gaussian_records = record_table(points, record, z_gaussian)
    robust_records = record_table(points, record, z_robust)
    baseline_records.to_csv(OUT / "records_baseline.csv", index=False)
    fixed_ten_records.to_csv(OUT / "records_fixed_10pct_data.csv", index=False)
    gaussian_records.to_csv(OUT / "records_gaussian_optimal.csv", index=False)
    robust_records.to_csv(OUT / "records_student_t_optimal.csv", index=False)

    def residual_summary(z):
        return {
            "rms": float(np.sqrt(np.mean(z**2))),
            "median_abs": float(np.median(np.abs(z))),
            "fraction_above_3sigma": float(np.mean(np.abs(z) > 3)),
            "fraction_above_5sigma": float(np.mean(np.abs(z) > 5)),
            "maximum_abs": float(np.max(np.abs(z))),
        }

    summary = {
        "observations": len(observed),
        "records": len(tau),
        "baseline": {
            "chi2_per_observation": float(baseline_q / len(observed)),
            "residuals": residual_summary(z_baseline),
        },
        "gaussian_fixed_10pct_data_scaled": {
            "chi2_per_observation": float(fixed_ten_q / len(observed)),
            "residuals": residual_summary(z_fixed_ten),
        },
        "gaussian_fractional": {
            "scale": "observed data",
            "likelihood_optimal_fraction": gaussian_fraction,
            "chi2_per_observation": float(gaussian_q / len(observed)),
            "residuals": residual_summary(z_gaussian),
        },
        "student_t_fractional": {
            "scale": "observed data",
            "degrees_of_freedom": best_nu,
            "likelihood_optimal_fraction": robust_fraction,
            "profile_nll": robust_nll,
            "residuals": residual_summary(z_robust),
        },
        "prediction_scaled_gaussian_diagnostic": {
            "likelihood_optimal_fraction": gaussian_scans["prediction"]["fraction"],
            "minimum_profile_nll_difference_from_data_scaled": float(
                np.min(gaussian_scans["prediction"]["nll"])
                - np.min(gaussian_scans["data"]["nll"])
            ),
        },
        "top_baseline_records": baseline_records.head(12).to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), layout="constrained")
    for scale_name, result in gaussian_scans.items():
        values = result["nll"]
        axes[0].plot(
            fractions,
            values - values.min(),
            lw=2,
            label=f"Gaussian, {scale_name}-scaled",
        )
    for nu, result in robust_results.items():
        values = result["nll"]
        axes[0].plot(fractions, values - values.min(), lw=1.4, label=rf"Student-$t$, $\nu={nu:g}$")
    axes[0].set(xlabel="fractional discrepancy", ylabel=r"$\Delta$ profile NLL", ylim=(0, 250))
    axes[0].legend(fontsize=8)
    axes[0].grid(False)

    bins = np.linspace(-10, 10, 101)
    axes[1].hist(z_baseline, bins=bins, density=True, histtype="step", lw=1.8, label="reported errors")
    axes[1].hist(z_fixed_ten, bins=bins, density=True, histtype="step", lw=1.8, label="10% of data")
    axes[1].hist(z_gaussian, bins=bins, density=True, histtype="step", lw=1.8, label="Gaussian discrepancy")
    axes[1].hist(z_robust, bins=bins, density=True, histtype="step", lw=1.8, label="Student-$t$")
    axes[1].set(xlabel="profiled standardized residual", ylabel="density", yscale="log")
    axes[1].legend(fontsize=8)
    axes[1].grid(False)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"likelihood_comparison.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
