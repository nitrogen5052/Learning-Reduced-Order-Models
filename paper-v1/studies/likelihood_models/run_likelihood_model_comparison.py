"""Refit and sample simple likelihood alternatives for the curated KD study.

This is deliberately isolated from the paper notebooks.  It compares a fixed
10% data-scaled Gaussian discrepancy, two inferred Gaussian discrepancies,
and a robust Student-t likelihood.  All cases retain the reported point errors
and the per-angular-distribution normalization uncertainties already present
in the curated payload.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "Scratch/CodeRestructure"
SOURCE_DIR = SCRATCH / "global_diagnostics/curated_neutron_map"
OUT = ROOT / "PaperNotebooks/data/likelihood_model_comparison"
sys.path.insert(0, str(SCRATCH))

from run_curated_kd_sampling_jax import fisher_covariance
from run_kd_coefficient_calibration_jax import build_model, diagnostics

MODELS = (
    "fixed10_data_gaussian",
    "inferred_data_gaussian",
    "inferred_prediction_gaussian",
    "inferred_kduq_gaussian",
    "inferred_data_student_t4",
)


def gaussian_marginal_likelihood(prediction, sigma, observed, record, tau):
    """Normalized Gaussian likelihood with normalization nuisances marginalized."""
    residual = prediction - observed[None]
    inverse = 1.0 / sigma**2
    weighted_prediction = prediction * inverse
    record_count = len(tau)
    projection = jnp.zeros((prediction.shape[0], record_count), dtype=jnp.float64)
    projection = projection.at[:, record].add(weighted_prediction * residual)
    norm = jnp.zeros_like(projection).at[:, record].add(weighted_prediction * prediction)
    denominator = 1.0 + tau[None] ** 2 * norm
    quadratic = jnp.sum(residual**2 * inverse, axis=1)
    quadratic -= jnp.sum(tau[None] ** 2 * projection**2 / denominator, axis=1)
    logdet = jnp.sum(jnp.log(sigma**2), axis=1) + jnp.sum(jnp.log(denominator), axis=1)
    return -0.5 * (quadratic + logdet + prediction.shape[1] * jnp.log(2.0 * jnp.pi))


def gaussian_base_likelihood(prediction, sigma, observed, record, tau):
    """Exact likelihood part used by build_model (diagonal constants omitted)."""
    residual = prediction - observed[None]
    inverse = 1.0 / sigma[None] ** 2
    weighted_prediction = prediction * inverse
    projection = jnp.zeros((prediction.shape[0], len(tau)), dtype=jnp.float64)
    projection = projection.at[:, record].add(weighted_prediction * residual)
    norm = jnp.zeros_like(projection).at[:, record].add(weighted_prediction * prediction)
    denominator = 1.0 + tau[None] ** 2 * norm
    quadratic = jnp.sum(residual**2 * inverse, axis=1)
    quadratic -= jnp.sum(tau[None] ** 2 * projection**2 / denominator, axis=1)
    return -0.5 * (quadratic + jnp.sum(jnp.log(denominator), axis=1))


def student_profile_likelihood(prediction, sigma, observed, record, tau, nu=4.0):
    """Student-t likelihood with Gaussian record normalizations profiled by IRLS."""
    residual = prediction - observed[None]
    inverse = 1.0 / sigma**2
    eta = jnp.zeros((prediction.shape[0], len(tau)), dtype=jnp.float64)
    safe_tau = jnp.maximum(tau, 1e-8)
    for _ in range(6):
        adjusted = residual + eta[:, record] * prediction
        z2 = adjusted**2 * inverse
        weight = (nu + 1.0) / (nu + z2)
        numerator = jnp.zeros_like(eta).at[:, record].add(
            weight * prediction * residual * inverse
        )
        denominator = 1.0 / safe_tau[None] ** 2
        denominator += jnp.zeros_like(eta).at[:, record].add(
            weight * prediction**2 * inverse
        )
        eta = -numerator / denominator
    adjusted = residual + eta[:, record] * prediction
    z2 = adjusted**2 * inverse
    constant = (
        0.5 * jnp.log(nu * jnp.pi)
        + jax.scipy.special.gammaln(0.5 * nu)
        - jax.scipy.special.gammaln(0.5 * (nu + 1.0))
    )
    likelihood = -jnp.sum(
        jnp.log(sigma) + constant + 0.5 * (nu + 1.0) * jnp.log1p(z2 / nu),
        axis=1,
    )
    likelihood -= 0.5 * jnp.sum((eta / safe_tau[None]) ** 2, axis=1)
    return likelihood


def build_comparison_model(source, model):
    predict, raw_logp = build_model(source, parameterization="all", optical_box="hard")
    observed = jax.device_put(np.asarray(source["observed"], np.float64))
    experimental = jax.device_put(np.asarray(source["sigma"], np.float64))
    record = jax.device_put(np.asarray(source["record_index"], np.int32))
    tau = jax.device_put(np.asarray(source["normalization_fraction"], np.float64))
    bounds = jax.device_put(np.asarray(source["bounds"], np.float64))

    def prior_and_constraints(theta, prediction):
        base = gaussian_base_likelihood(prediction, experimental, observed, record, tau)
        return raw_logp(theta) - base

    inferred = model != "fixed10_data_gaussian"

    def logp(parameters):
        theta = parameters[:, :37]
        prediction = predict(theta)
        if inferred:
            log_fraction = parameters[:, 37]
            fraction = jnp.exp(log_fraction)
        else:
            log_fraction = jnp.full(parameters.shape[0], jnp.log(0.10))
            fraction = jnp.full(parameters.shape[0], 0.10)
        if model == "inferred_kduq_gaussian":
            scale = 0.5 * (jnp.abs(observed)[None] + prediction)
        elif model == "inferred_prediction_gaussian":
            scale = prediction
        else:
            scale = jnp.broadcast_to(jnp.abs(observed)[None], prediction.shape)
        sigma = jnp.sqrt(experimental[None] ** 2 + (fraction[:, None] * scale) ** 2)
        if model == "inferred_data_student_t4":
            likelihood = student_profile_likelihood(
                prediction, sigma, observed, record, tau, nu=4.0
            )
        else:
            likelihood = gaussian_marginal_likelihood(
                prediction, sigma, observed, record, tau
            )
        result = likelihood + prior_and_constraints(theta, prediction)
        if inferred:
            # Match the KDUQ prior for differential observables:
            # delta ~ N(0.20, 0.20), truncated at delta > 0.  Sampling is in
            # log(delta), hence the Jacobian term +log(delta).
            discrepancy_prior = -0.5 * ((fraction - 0.20) / 0.20) ** 2 + log_fraction
            valid_fraction = (fraction >= 0.005) & (fraction <= 0.80)
            result = result + discrepancy_prior
            result = jnp.where(valid_fraction, result, -jnp.inf)
        return result

    return predict, logp, inferred, np.asarray(bounds)


def optimize(logp, initial, bounds):
    scalar = lambda x: logp(x[None])[0]
    value_gradient = jax.jit(jax.value_and_grad(scalar))
    value_gradient(jnp.asarray(initial))[0].block_until_ready()
    evaluations = 0

    def objective(x):
        nonlocal evaluations
        value, gradient = value_gradient(jnp.asarray(x))
        evaluations += 1
        return -float(value), -np.asarray(gradient, float)

    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 120, "ftol": 1e-10, "maxls": 30},
    )
    return np.asarray(result.x), result, evaluations


def sample(model, chains=16, steps=2600, burn=700, seed=20261118):
    jax.config.update("jax_enable_x64", True)
    OUT.mkdir(parents=True, exist_ok=True)
    source = np.load(SOURCE_DIR / "curated_kduq_sampling_payload.npz", allow_pickle=False)
    old_map = np.asarray(np.load(SOURCE_DIR / "refined_map.npz")["theta"], float)
    predict, raw_logp, inferred, theta_bounds = build_comparison_model(source, model)
    compiled_predict = jax.jit(predict)
    compiled_logp = jax.jit(raw_logp)
    fraction_start = {
        "fixed10_data_gaussian": 0.10,
        "inferred_data_gaussian": 0.215,
        "inferred_prediction_gaussian": 0.255,
        "inferred_kduq_gaussian": 0.20,
        "inferred_data_student_t4": 0.13,
    }[model]
    initial = old_map if not inferred else np.r_[old_map, np.log(fraction_start)]
    parameter_bounds = list(zip(-theta_bounds, theta_bounds))
    if inferred:
        parameter_bounds.append((np.log(0.005), np.log(0.80)))
    started = time.perf_counter()
    map_estimate, optimization, evaluations = optimize(
        compiled_logp, initial, parameter_bounds
    )
    theta_map = map_estimate[:37]
    fraction_map = 0.10 if not inferred else float(np.exp(map_estimate[-1]))

    values = {key: source[key] for key in source.files}
    map_prediction = np.asarray(compiled_predict(jnp.asarray(theta_map[None])))[0]
    if model == "inferred_kduq_gaussian":
        scale_reference = 0.5 * (np.abs(source["observed"]) + map_prediction)
    elif model == "inferred_prediction_gaussian":
        scale_reference = map_prediction
    else:
        scale_reference = np.abs(source["observed"])
    values["sigma"] = np.sqrt(
        np.asarray(source["sigma"]) ** 2 + (fraction_map * scale_reference) ** 2
    )
    fisher_path = OUT / f"{model}_fisher_payload.npz"
    np.savez_compressed(fisher_path, **values)
    fisher_archive = np.load(fisher_path, allow_pickle=False)
    covariance, eigenvalues, floor = fisher_covariance(
        compiled_predict, fisher_archive, theta_map
    )
    theta_chol = np.linalg.cholesky(covariance + 1e-14 * np.eye(37))

    if inferred:
        q0 = map_estimate[-1]
        step = 1e-3
        left = map_estimate.copy(); left[-1] -= step
        right = map_estimate.copy(); right[-1] += step
        center_value = float(compiled_logp(jnp.asarray(map_estimate[None]))[0])
        q_curvature = -(
            float(compiled_logp(jnp.asarray(right[None]))[0])
            - 2.0 * center_value
            + float(compiled_logp(jnp.asarray(left[None]))[0])
        ) / step**2
        q_sd = 1.0 / np.sqrt(max(q_curvature, 1.0))
    else:
        q_sd = 0.0

    rng = np.random.default_rng(seed)
    current = np.repeat(map_estimate[None], chains, axis=0)
    current[:, :37] += 0.01 * rng.normal(size=(chains, 37)) @ theta_chol.T
    if inferred:
        current[:, -1] += 0.01 * q_sd * rng.normal(size=chains)
    current_logp = np.array(compiled_logp(jnp.asarray(current)), copy=True)
    invalid = ~np.isfinite(current_logp)
    current[invalid] = map_estimate
    current_logp[invalid] = float(compiled_logp(jnp.asarray(map_estimate[None]))[0])

    samples = np.empty((steps, chains, len(map_estimate)), np.float32)
    accepted = np.zeros(chains, int)
    proposal_scale = 0.10
    recent_accept = recent_trials = 0
    sampling_started = time.perf_counter()
    for index in range(steps):
        proposed = current.copy()
        proposed[:, :37] += (
            proposal_scale * rng.normal(size=(chains, 37)) @ theta_chol.T
        )
        if inferred:
            proposed[:, -1] += proposal_scale * q_sd * rng.normal(size=chains)
        proposed_logp = np.asarray(compiled_logp(jnp.asarray(proposed)))
        accept = np.log(rng.random(chains)) < proposed_logp - current_logp
        current[accept] = proposed[accept]
        current_logp[accept] = proposed_logp[accept]
        accepted += accept
        recent_accept += int(accept.sum()); recent_trials += chains
        samples[index] = current
        if index < burn and (index + 1) % 50 == 0:
            proposal_scale *= np.exp(
                np.clip(recent_accept / recent_trials - 0.20, -0.18, 0.18)
            )
            recent_accept = recent_trials = 0
        if (index + 1) % 250 == 0:
            print(
                f"{model}: {index+1}/{steps}, acceptance="
                f"{accepted.sum()/((index+1)*chains):.3f}, scale={proposal_scale:.4g}",
                flush=True,
            )

    kept = samples[burn:].astype(float)
    rhat, ess = diagnostics(kept)
    posterior = kept.reshape(-1, kept.shape[-1])
    draw_count = min(480, len(posterior))
    selected = rng.choice(len(posterior), draw_count, replace=False)
    theta_draws = posterior[selected, :37]
    prediction_blocks = []
    for start in range(0, draw_count, 24):
        prediction_blocks.append(
            np.asarray(compiled_predict(jnp.asarray(theta_draws[start:start+24])))
        )
    parameter_predictions = np.concatenate(prediction_blocks)
    if inferred:
        fractions = np.exp(posterior[selected, -1])
    else:
        fractions = np.full(draw_count, 0.10)
    if model == "inferred_kduq_gaussian":
        noise_scale = 0.5 * (
            np.abs(source["observed"])[None] + parameter_predictions
        )
    elif model == "inferred_prediction_gaussian":
        noise_scale = parameter_predictions
    else:
        noise_scale = np.abs(source["observed"])[None]
    if model == "inferred_data_student_t4":
        random_noise = rng.standard_t(4.0, size=parameter_predictions.shape)
    else:
        random_noise = rng.normal(size=parameter_predictions.shape)
    total_predictions = np.maximum(
        parameter_predictions + random_noise * fractions[:, None] * noise_scale,
        1e-30,
    )

    model_out = OUT / model
    model_out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_out / "posterior.npz",
        samples=samples,
        posterior=posterior,
        map_estimate=map_estimate,
        rhat=rhat,
        ess=ess,
        acceptance=accepted / steps,
    )
    np.savez_compressed(
        model_out / "posterior_predictive.npz",
        draws=theta_draws,
        fractions=fractions,
        parameter_predictions=parameter_predictions.astype(np.float32),
        total_predictions=total_predictions.astype(np.float32),
    )
    summary = {
        "model": model,
        "chains": chains,
        "steps": steps,
        "burn": burn,
        "map_fraction": fraction_map,
        "posterior_fraction_2.5_50_97.5": (
            np.quantile(np.exp(posterior[:, -1]), [0.025, 0.5, 0.975]).tolist()
            if inferred else [0.10, 0.10, 0.10]
        ),
        "optimization_success": bool(optimization.success),
        "optimization_message": str(optimization.message),
        "optimization_evaluations": evaluations,
        "map_log_posterior": float(compiled_logp(jnp.asarray(map_estimate[None]))[0]),
        "acceptance": float(np.mean(accepted / steps)),
        "proposal_scale": float(proposal_scale),
        "maximum_rhat": float(np.max(rhat)),
        "minimum_ess": float(np.min(ess)),
        "fisher_condition": float(eigenvalues[-1] / max(eigenvalues[0], 1e-300)),
        "fisher_floor": float(floor),
        "sampling_seconds": time.perf_counter() - sampling_started,
        "total_seconds": time.perf_counter() - started,
    }
    (model_out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--chains", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2600)
    parser.add_argument("--burn", type=int, default=700)
    parser.add_argument("--seed", type=int, default=20261118)
    arguments = parser.parse_args()
    sample(arguments.model, arguments.chains, arguments.steps, arguments.burn, arguments.seed)
