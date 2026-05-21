"""Recompute clean cross-section comparison data for notebook 03.

This module keeps the expensive observable-level recomputation out of the
notebook generator.  It mirrors the Phase 3 online-optimization setup, but
only keeps the methods used in the clean demo:

* FOM
* LS projection floor
* ROSE ROM (n=4, n_U=10)
* linear RF-LROM (n=4, K=10 raw optical-parameter predictors)
* predictor RF-LROM (n=4, K=10 delta-maxvol potential predictors)
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
from numba import njit
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase3_implicit as p3  # noqa: E402
from rose.utility import woods_saxon, woods_saxon_prime  # noqa: E402


FEATURE_NAMES = ["Vv", "Wv", "Wd", "Vso", "Rv", "Rd", "Rso", "av", "ad", "aso"]
CENTRAL_SAMPLE = np.array(
    [
        46.7238,
        1.72334,
        -7.2357,
        6.1,
        4.0538,
        4.4055,
        1.01 * 40 ** (1.0 / 3.0),
        0.6718,
        0.5379,
        0.60,
    ],
    dtype=float,
)
WSO_FIXED = 0.0
SCALE_TRAINING = 0.20
N_TRAIN = 100
N_TEST = 80
N_PHI = 4
N_U = 10
K_PREDICTOR = 10
R_MIN_PREDICTOR = 0.5
L_MAX = 10
N_MESH = 700
N_ANGLES = 120


def _build_samples():
    feature_scales = np.abs(SCALE_TRAINING * CENTRAL_SAMPLE)
    lower = CENTRAL_SAMPLE - feature_scales
    upper = CENTRAL_SAMPLE + feature_scales
    sampler_train = qmc.LatinHypercube(d=CENTRAL_SAMPLE.size, seed=20260524)
    sampler_test = qmc.LatinHypercube(d=CENTRAL_SAMPLE.size, seed=20260525)
    train_samples = qmc.scale(sampler_train.random(N_TRAIN), lower, upper)
    test_samples = qmc.scale(sampler_test.random(N_TEST), lower, upper)
    return feature_scales, lower, upper, train_samples, test_samples


@njit
def _optical_potential_10(r, theta):
    vv, wv, wd, vso, rv, rd, rso, av, ad, aso = theta
    return (1j * wv - vv) * woods_saxon(r, rv, av) - (4j * ad * wd) * woods_saxon_prime(r, rd, ad)


@njit
def _spin_orbit_10(r, theta, ldots):
    vv, wv, wd, vso, rv, rd, rso, av, ad, aso = theta
    mass_pion = p3.rose.constants.MASS_PION
    return (vso + 1j * WSO_FIXED) / mass_pion**2 * ldots * woods_saxon_prime(r, rso, aso) / r


def _greedy_maxvol_indices(basis):
    n_rows, n_cols = basis.shape
    selected = [int(np.argmax(np.linalg.norm(basis, axis=1)))]
    for _ in range(1, n_cols):
        q, *_ = np.linalg.qr(basis[selected, :].T, mode="reduced")
        residual = basis - (basis @ q) @ q.T.conj()
        scores = np.linalg.norm(residual, axis=1)
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
    for _ in range(50):
        sub = basis[selected, :]
        try:
            coeff = basis @ np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            break
        abs_coeff = np.abs(coeff)
        abs_coeff[selected, :] = 0.0
        row, col = np.unravel_index(np.argmax(abs_coeff), abs_coeff.shape)
        if abs_coeff[row, col] <= 1.0 + 1e-10:
            break
        selected[col] = int(row)
    return np.array(sorted(selected), dtype=int)


def _woods_saxon_np(r, R, a):
    x = np.clip((r - R) / a, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(x))


def _woods_saxon_prime_np(r, R, a):
    x = np.clip((r - R) / a, -700.0, 700.0)
    ex = np.exp(x)
    return -(ex / a) / (1.0 + ex) ** 2


def _thomas_np(r, R, a):
    return _woods_saxon_prime_np(r, R, a) / np.maximum(r, 1e-12)


def recompute(cache_path: str | Path, force: bool = False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return dict(np.load(cache_path, allow_pickle=True))

    t_all = time.perf_counter()
    feature_scales, lower, upper, train_samples, test_samples = _build_samples()
    rho_mesh = np.linspace(1e-8, 8 * np.pi, N_MESH)
    angles = np.linspace(1, 179, N_ANGLES) * np.pi / 180
    angles_deg = angles * 180 / np.pi

    mu, e_com, _k_helper, _eta = p3.rose.kinematics(
        target=p3.TARGET,
        projectile=p3.PROJECTILE,
        E_lab=p3.E_LAB,
    )
    k_fixed = np.sqrt(2.0 * mu * e_com) / p3.rose.constants.HBARC
    central_alpha = CENTRAL_SAMPLE.copy()
    base_solver = p3.rose.SchroedingerEquation.make_base_solver(
        s_0=6 * np.pi,
        rk_tols=[1e-9, 1e-9],
        domain=np.array([1e-8, 8 * np.pi]),
    )

    def make_native_basis(interaction, theta_train):
        solver = base_solver.clone_for_new_interaction(interaction)
        return p3.rose.basis.RelativeBasis(
            solver=solver,
            theta_train=theta_train,
            rho_mesh=rho_mesh,
            n_basis=N_PHI,
            use_svd=True,
            center=False,
            scale=False,
        )

    def make_central_basis(interaction, theta_train):
        solver = base_solver.clone_for_new_interaction(interaction)
        phi_train = np.array([solver.phi(alpha, rho_mesh) for alpha in theta_train])
        phi_central = solver.phi(central_alpha, rho_mesh)
        return p3.CustomBasis(
            solutions=phi_train.T.copy(),
            phi_0=phi_central.copy(),
            rho_mesh=rho_mesh,
            n_basis=N_PHI,
            solver=solver,
            subtract_phi0=True,
            use_svd=True,
            center=False,
            scale=False,
        )

    interactions = p3.rose.InteractionEIMSpace(
        coordinate_space_potential=_optical_potential_10,
        n_theta=CENTRAL_SAMPLE.size,
        mu=mu,
        energy=e_com,
        is_complex=True,
        spin_orbit_term=_spin_orbit_10,
        training_info=np.vstack([train_samples, central_alpha]),
        explicit_training=True,
        l_max=L_MAX,
        n_basis=N_U,
        rho_mesh=rho_mesh,
    )

    native_bases = []
    central_bases = []
    coeff_train = []
    coeff_test = []
    for interaction_list in interactions.interactions:
        native_row = []
        central_row = []
        coeff_train_row = []
        coeff_test_row = []
        for interaction in interaction_list:
            native_basis = make_native_basis(interaction, train_samples)
            central_basis = make_central_basis(interaction, train_samples)
            native_row.append(native_basis)
            central_row.append(central_basis)

            solver = central_basis.solver
            phi_train = np.array([solver.phi(alpha, rho_mesh) for alpha in train_samples])
            phi_test = np.array([solver.phi(alpha, rho_mesh) for alpha in test_samples])
            coeff_train_row.append(p3.least_squares_basis_coefficients(central_basis, phi_train, rho_mesh))
            coeff_test_row.append(p3.least_squares_basis_coefficients(central_basis, phi_test, rho_mesh))
        native_bases.append(native_row)
        central_bases.append(central_row)
        coeff_train.append(coeff_train_row)
        coeff_test.append(coeff_test_row)

    native_sae = p3.rose.ScatteringAmplitudeEmulator(
        interactions,
        native_bases,
        l_max=L_MAX,
        angles=angles,
        s_0=base_solver.s_0,
        Smatrix_abs_tol=1e-8,
        initialize_emulator=True,
    )
    central_sae = p3.rose.ScatteringAmplitudeEmulator(
        interactions,
        central_bases,
        l_max=L_MAX,
        angles=angles,
        s_0=base_solver.s_0,
        Smatrix_abs_tol=1e-8,
        initialize_emulator=True,
    )

    features_train_linear = p3.multiparameter_feature_values(train_samples, CENTRAL_SAMPLE, feature_scales)
    features_test_linear = p3.multiparameter_feature_values(test_samples, CENTRAL_SAMPLE, feature_scales)

    mass_pion = p3.rose.koning_delaroche.MASS_PION

    def direct_tilde(alphas, s_points, ell, ch):
        alphas = np.atleast_2d(np.asarray(alphas, dtype=float))
        s_points = np.asarray(s_points, dtype=float)
        r = s_points[np.newaxis, :] / k_fixed

        vv = alphas[:, 0:1]
        wv = alphas[:, 1:2]
        wd = alphas[:, 2:3]
        vso = alphas[:, 3:4]
        rv = alphas[:, 4:5]
        rd = alphas[:, 5:6]
        rso = alphas[:, 6:7]
        av = alphas[:, 7:8]
        ad = alphas[:, 8:9]
        aso = alphas[:, 9:10]

        central = (
            -vv * _woods_saxon_np(r, rv, av)
            + 1j * wv * _woods_saxon_np(r, rv, av)
            - 4j * ad * wd * _woods_saxon_prime_np(r, rd, ad)
        )
        lds = 0.0 if ell == 0 else (ell if ch == 0 else -(ell + 1.0))
        spin_orbit = lds * (
            vso / mass_pion**2 * _thomas_np(r, rso, aso)
            + 1j * WSO_FIXED / mass_pion**2 * _thomas_np(r, rso, aso)
        )
        return (central + spin_orbit) / e_com

    def delta_maxvol_points(interaction):
        center = interaction.tilde(rho_mesh, central_alpha)
        delta = np.array([interaction.tilde(rho_mesh, alpha) - center for alpha in train_samples]).T
        r_mesh_central = rho_mesh / k_fixed
        allowed = np.flatnonzero(r_mesh_central >= R_MIN_PREDICTOR)
        U, S, _ = np.linalg.svd(delta[allowed], full_matrices=False)
        local = _greedy_maxvol_indices(U[:, :K_PREDICTOR])
        idx = allowed[local]
        return {"s_points": rho_mesh[idx], "r_points": r_mesh_central[idx]}

    def make_predictor_spec(ell, ch, points):
        p_train = direct_tilde(train_samples, points["s_points"], ell, ch)
        p_center = direct_tilde(central_alpha[np.newaxis, :], points["s_points"], ell, ch)[0]
        scale = np.maximum(np.max(np.abs(p_train - p_center), axis=0), 1e-14)
        return {"s_points": points["s_points"], "r_points": points["r_points"], "center": p_center, "scale": scale}

    def predictor_features(alphas, ell, ch, spec):
        p = direct_tilde(alphas, spec["s_points"], ell, ch)
        return np.column_stack([np.ones(len(p)), (p - spec["center"]) / spec["scale"]])

    predictor_specs = []
    for ell, interaction_list in enumerate(interactions.interactions):
        spec_row = []
        for ch, interaction in enumerate(interaction_list):
            points = delta_maxvol_points(interaction)
            spec_row.append(make_predictor_spec(ell, ch, points))
        predictor_specs.append(spec_row)

    linear_fits = []
    predictor_fits = []
    for ell, coeff_row in enumerate(coeff_train):
        linear_row = []
        predictor_row = []
        for ch, coeff_ref in enumerate(coeff_row):
            linear_row.append(
                p3.fit_identity_m0_rf(
                    name=f"linear ell={ell}, ch={ch}",
                    feature_values=features_train_linear,
                    coeff_ref=coeff_ref,
                )
            )
            predictor_row.append(
                p3.fit_identity_m0_rf(
                    name=f"delta-maxvol K={K_PREDICTOR}, ell={ell}, ch={ch}",
                    feature_values=predictor_features(train_samples, ell, ch, predictor_specs[ell][ch]),
                    coeff_ref=coeff_ref,
                )
            )
        linear_fits.append(linear_row)
        predictor_fits.append(predictor_row)

    def ls_smatrix_elements(sample_index, coeffs):
        n_l = len(central_sae.rbes)
        splus = np.zeros(n_l, dtype=np.complex128)
        sminus = np.zeros(n_l, dtype=np.complex128)
        for ell in range(n_l):
            splus[ell] = p3.s_matrix_from_coefficients(central_sae.rbes[ell][0], coeffs[ell][0][sample_index])
            if ell == 0:
                sminus[ell] = splus[ell]
            else:
                sminus[ell] = p3.s_matrix_from_coefficients(central_sae.rbes[ell][1], coeffs[ell][1][sample_index])
        return splus, sminus

    def predictor_smatrix(alpha):
        n_l = len(central_sae.rbes)
        splus = np.zeros(n_l, dtype=np.complex128)
        sminus = np.zeros(n_l, dtype=np.complex128)
        alpha_row = np.asarray(alpha, dtype=float)[np.newaxis, :]
        for ell in range(n_l):
            feats = predictor_features(alpha_row, ell, 0, predictor_specs[ell][0])[0]
            coeff = p3.model_predict(feats[np.newaxis, :], predictor_fits[ell][0].matrices, predictor_fits[ell][0].vectors)[0]
            splus[ell] = p3.s_matrix_from_coefficients(central_sae.rbes[ell][0], coeff)
            if ell == 0:
                sminus[ell] = splus[ell]
            else:
                feats = predictor_features(alpha_row, ell, 1, predictor_specs[ell][1])[0]
                coeff = p3.model_predict(feats[np.newaxis, :], predictor_fits[ell][1].matrices, predictor_fits[ell][1].vectors)[0]
                sminus[ell] = p3.s_matrix_from_coefficients(central_sae.rbes[ell][1], coeff)
        return splus, sminus


    def flatten_channels(sae, fits, specs):
        channels = []
        for ell, rbe_row in enumerate(sae.rbes):
            for ch, rbe in enumerate(rbe_row):
                channels.append({
                    "ell": ell,
                    "ch": ch,
                    "rbe": rbe,
                    "fit": fits[ell][ch],
                    "spec": specs[ell][ch],
                })
        return channels

    def compile_predictor_lrom(sae, fits, specs):
        channels = flatten_channels(sae, fits, specs)
        return {
            "matrices": np.asarray([[np.asarray(m) for m in c["fit"].matrices] for c in channels], dtype=np.complex128),
            "vectors": np.asarray([[np.asarray(v) for v in c["fit"].vectors] for c in channels], dtype=np.complex128),
            "asym_vals": np.asarray([c["rbe"].asymptotic_vals for c in channels], dtype=np.complex128),
            "asym_ders": np.asarray([c["rbe"].asymptotic_ders for c in channels], dtype=np.complex128),
            "Hm": np.asarray([c["rbe"].Hm for c in channels], dtype=np.complex128),
            "Hp": np.asarray([c["rbe"].Hp for c in channels], dtype=np.complex128),
            "Hmp": np.asarray([c["rbe"].Hmp for c in channels], dtype=np.complex128),
            "Hpp": np.asarray([c["rbe"].Hpp for c in channels], dtype=np.complex128),
            "s0": np.asarray([c["rbe"].s_0 for c in channels], dtype=float),
            "ell": np.asarray([c["ell"] for c in channels], dtype=int),
            "ch": np.asarray([c["ch"] for c in channels], dtype=int),
            "s_points": np.asarray([c["spec"]["s_points"] for c in channels], dtype=float),
            "center": np.asarray([c["spec"]["center"] for c in channels], dtype=np.complex128),
            "scale": np.asarray([c["spec"]["scale"] for c in channels], dtype=float),
            "n_l": len(sae.rbes),
        }

    def direct_tilde_flat(alpha, s_flat, ell_flat, ch_flat):
        alpha = np.asarray(alpha, dtype=float)
        r = np.asarray(s_flat, dtype=float) / k_fixed
        ell_flat = np.asarray(ell_flat, dtype=float)
        ch_flat = np.asarray(ch_flat, dtype=float)

        vv, wv, wd, vso, rv, rd, rso, av, ad, aso = alpha
        f_volume = _woods_saxon_np(r, rv, av)
        fp_surface = _woods_saxon_prime_np(r, rd, ad)
        thomas_so = _thomas_np(r, rso, aso)

        central = -vv * f_volume + 1j * wv * f_volume - 4j * ad * wd * fp_surface
        lds = np.where(ell_flat == 0, 0.0, np.where(ch_flat == 0, ell_flat, -(ell_flat + 1.0)))
        spin_orbit = lds * (vso + 1j * WSO_FIXED) / mass_pion**2 * thomas_so
        return (central + spin_orbit) / e_com

    def compile_fast_predictor_arrays(packed):
        n_channels, n_pred = packed["s_points"].shape
        return {
            "s_flat": packed["s_points"].reshape(-1),
            "ell_flat": np.repeat(packed["ell"], n_pred),
            "ch_flat": np.repeat(packed["ch"], n_pred),
            "shape": (n_channels, n_pred),
        }

    def packed_predictor_features_fast(alpha, packed, fast_arrays):
        raw = direct_tilde_flat(
            alpha,
            fast_arrays["s_flat"],
            fast_arrays["ell_flat"],
            fast_arrays["ch_flat"],
        ).reshape(fast_arrays["shape"])
        feats = np.empty((fast_arrays["shape"][0], fast_arrays["shape"][1] + 1), dtype=np.complex128)
        feats[:, 0] = 1.0
        feats[:, 1:] = (raw - packed["center"]) / packed["scale"]
        return feats

    def packed_coefficients_from_features(features, packed):
        mats = np.einsum("cf,cfij->cij", features, packed["matrices"], optimize=True)
        rhs = np.einsum("cf,cfi->ci", features, packed["vectors"], optimize=True)
        return np.linalg.solve(mats, rhs[..., np.newaxis])[..., 0]

    def packed_smatrix_from_coefficients(coeffs, packed):
        ones = np.ones((coeffs.shape[0], 1), dtype=np.complex128)
        x = np.concatenate([ones, coeffs], axis=1)
        phi = np.sum(x * packed["asym_vals"], axis=1)
        phi_prime = np.sum(x * packed["asym_ders"], axis=1)
        r_matrix = phi / (packed["s0"] * phi_prime)
        s_flat = (packed["Hm"] - packed["s0"] * r_matrix * packed["Hmp"]) / (
            packed["Hp"] - packed["s0"] * r_matrix * packed["Hpp"]
        )
        splus = np.zeros(packed["n_l"], dtype=np.complex128)
        sminus = np.zeros(packed["n_l"], dtype=np.complex128)
        for value, ell, ch in zip(s_flat, packed["ell"], packed["ch"]):
            if ch == 0:
                splus[ell] = value
                if ell == 0:
                    sminus[ell] = value
            else:
                sminus[ell] = value
        return splus, sminus

    predictor_packed = compile_predictor_lrom(central_sae, predictor_fits, predictor_specs)
    predictor_fast_arrays = compile_fast_predictor_arrays(predictor_packed)

    def predictor_smatrix_fast(alpha):
        feats = packed_predictor_features_fast(alpha, predictor_packed, predictor_fast_arrays)
        coeffs = packed_coefficients_from_features(feats, predictor_packed)
        return packed_smatrix_from_coefficients(coeffs, predictor_packed)

    fast_check_diffs = []
    for alpha in test_samples[: min(5, len(test_samples))]:
        slow = predictor_smatrix(alpha)
        fast = predictor_smatrix_fast(alpha)
        fast_check_diffs.append(max(np.max(np.abs(slow[0] - fast[0])), np.max(np.abs(slow[1] - fast[1]))))
    fast_check_max_smatrix_diff = float(np.max(fast_check_diffs))

    def compute_split(samples, linear_features, coeffs):
        method_names = ["FOM", "LS floor", "ROSE ROM", "linear LROM", "predictor LROM"]
        xs = {name: [] for name in method_names}
        times = {name: [] for name in method_names}
        for i, alpha in enumerate(samples):
            t0 = time.perf_counter()
            splus, sminus = p3.exact_smatrix_elements_fixed(central_sae, alpha)
            xs["FOM"].append(p3.dsdo_from_smatrix(central_sae, alpha, splus, sminus))
            times["FOM"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            splus, sminus = ls_smatrix_elements(i, coeffs)
            xs["LS floor"].append(p3.dsdo_from_smatrix(central_sae, alpha, splus, sminus))
            times["LS floor"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            splus, sminus = p3.rom_smatrix_elements_fixed(native_sae, alpha)
            xs["ROSE ROM"].append(p3.dsdo_from_smatrix(central_sae, alpha, splus, sminus))
            times["ROSE ROM"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            splus, sminus = p3.lrom_smatrix_elements(central_sae, linear_fits, linear_features[i])
            xs["linear LROM"].append(p3.dsdo_from_smatrix(central_sae, alpha, splus, sminus))
            times["linear LROM"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            splus, sminus = predictor_smatrix_fast(alpha)
            xs["predictor LROM"].append(p3.dsdo_from_smatrix(central_sae, alpha, splus, sminus))
            times["predictor LROM"].append(time.perf_counter() - t0)
        return (
            {key: np.asarray(val) for key, val in xs.items()},
            {key: np.asarray(val) for key, val in times.items()},
        )

    xs_train, times_train = compute_split(train_samples, features_train_linear, coeff_train)
    xs_test, times_test = compute_split(test_samples, features_test_linear, coeff_test)

    def metric_arrays(xs):
        methods = ["LS floor", "ROSE ROM", "linear LROM", "predictor LROM"]
        median_pointwise = []
        l2 = []
        max_pointwise = []
        for method in methods:
            med = []
            l2_vals = []
            mx = []
            for pred, ref in zip(xs[method], xs["FOM"]):
                rel = np.abs(pred - ref) / np.maximum(np.abs(ref), 1e-30)
                med.append(np.median(rel))
                mx.append(np.max(rel))
                l2_vals.append(np.sqrt(np.trapz((pred - ref) ** 2, angles) / np.maximum(np.trapz(ref**2, angles), 1e-30)))
            median_pointwise.append(med)
            l2.append(l2_vals)
            max_pointwise.append(mx)
        return methods, np.asarray(median_pointwise), np.asarray(l2), np.asarray(max_pointwise)

    methods, train_med, train_l2, train_max = metric_arrays(xs_train)
    _, test_med, test_l2, test_max = metric_arrays(xs_test)
    chosen = np.array([51, 63, 79], dtype=int)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        feature_names=np.asarray(FEATURE_NAMES),
        central_sample=CENTRAL_SAMPLE,
        feature_scales=feature_scales,
        lower=lower,
        upper=upper,
        train_samples=train_samples,
        test_samples=test_samples,
        angles_deg=angles_deg,
        methods=np.asarray(methods),
        chosen_test_indices=chosen,
        xs_train_fom=xs_train["FOM"],
        xs_train_ls=xs_train["LS floor"],
        xs_train_rose=xs_train["ROSE ROM"],
        xs_train_linear=xs_train["linear LROM"],
        xs_train_predictor=xs_train["predictor LROM"],
        xs_test_fom=xs_test["FOM"],
        xs_test_ls=xs_test["LS floor"],
        xs_test_rose=xs_test["ROSE ROM"],
        xs_test_linear=xs_test["linear LROM"],
        xs_test_predictor=xs_test["predictor LROM"],
        train_median_pointwise=train_med,
        train_l2=train_l2,
        train_max_pointwise=train_max,
        test_median_pointwise=test_med,
        test_l2=test_l2,
        test_max_pointwise=test_max,
        train_times=np.asarray([times_train[m] for m in methods]),
        test_times=np.asarray([times_test[m] for m in methods]),
        n_phi=N_PHI,
        n_U=N_U,
        k_linear=CENTRAL_SAMPLE.size,
        k_predictor=K_PREDICTOR,
        l_max=L_MAX,
        e_lab=p3.E_LAB,
        target_A=p3.TARGET[0],
        target_Z=p3.TARGET[1],
        fast_check_max_smatrix_diff=fast_check_max_smatrix_diff,
        recompute_seconds=time.perf_counter() - t_all,
    )
    return dict(np.load(cache_path, allow_pickle=True))
