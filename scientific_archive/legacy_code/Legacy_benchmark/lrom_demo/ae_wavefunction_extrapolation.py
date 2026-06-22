"""Wavefunction-level isotope/energy extrapolation data for notebook 04.

The notebook should stay readable, so the heavier ROSE/KD and LROM plumbing
lives here.  The calculation is deliberately wavefunction-first:

* generate FOM wavefunctions from the Koning-Delaroche global potential,
* build a central-reference reduced basis,
* learn predictor RF-LROM operator blocks from LS coordinates,
* evaluate interpolation and extrapolation errors against FOM wavefunctions.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys
import time

import numpy as np
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase3_implicit as p3  # noqa: E402


Z_TARGET = 20
A_CENTRAL = 40.0
E_CENTRAL = 14.1
PROJECTILE = p3.PROJECTILE

# Train on a central box, then test on the wider range requested in the
# notebook.  This makes the extrapolation question visible instead of only
# testing interpolation.
TRAIN_A_RANGE = (32.0, 48.0)
TRAIN_E_RANGE = (10.0, 22.0)
TEST_A_RANGE = (30.0, 60.0)
TEST_E_RANGE = (8.0, 30.0)

N_TRAIN = 200
N_TEST = 110
N_PHI = 4
K_PREDICTORS = 12
N_U = 9
N_MESH = 650
N_ANGLES = 120
L_MAX = 15
SELECTED_CHANNELS = ((0, 0), (4, 0), (4, 1))
R_MIN_PREDICTOR = 0.5


def alpha_from_AE(sample: np.ndarray) -> np.ndarray:
    """Return energized ROSE alpha for a calcium isotope and beam energy.

    `A` is rounded to the nearest mass number before calling KD/kinematics.
    This lets the notebook draw smooth-looking scans while still using a
    physically meaningful integer isotope in the optical-potential call.
    """

    A_float, E_lab = map(float, sample)
    A_int = int(round(A_float))
    kd = p3.rose.koning_delaroche.KDGlobal(p3.rose.Projectile.neutron)
    mu, e_com, k, _eta = p3.rose.kinematics(
        target=(A_int, Z_TARGET),
        projectile=PROJECTILE,
        E_lab=E_lab,
    )
    _rc, optical_alpha = kd.get_params(A_int, Z_TARGET, mu, E_lab, k)
    return np.concatenate([[e_com, mu, k], np.asarray(optical_alpha, dtype=float)])


def alphas_from_AE(samples: np.ndarray) -> np.ndarray:
    return np.asarray([alpha_from_AE(sample) for sample in np.asarray(samples)])


def _sample_box(n: int, a_range: tuple[float, float], e_range: tuple[float, float], seed: int):
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    lower = np.asarray([a_range[0], e_range[0]], dtype=float)
    upper = np.asarray([a_range[1], e_range[1]], dtype=float)
    samples = qmc.scale(sampler.random(n), lower, upper)
    samples[:, 0] = np.round(samples[:, 0])
    return samples


def _greedy_maxvol_indices(basis: np.ndarray) -> np.ndarray:
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
    return np.asarray(selected, dtype=int)


def _wavefunction_rel_errors(basis, coeffs: np.ndarray, phi_ref: np.ndarray, rho_mesh: np.ndarray):
    phi_pred = np.asarray([basis.phi_hat(c[np.newaxis, :]) for c in coeffs])
    num = np.sqrt(np.trapz(np.abs(phi_pred - phi_ref) ** 2, rho_mesh, axis=1))
    den = np.sqrt(np.trapz(np.abs(phi_ref) ** 2, rho_mesh, axis=1))
    return num / np.maximum(den, 1e-30), phi_pred


def _delta_maxvol_points(interaction, rho_mesh: np.ndarray, train_alphas: np.ndarray, central_alpha: np.ndarray):
    center = interaction.tilde(rho_mesh, central_alpha)
    delta = np.asarray([interaction.tilde(rho_mesh, alpha) - center for alpha in train_alphas]).T
    r_mesh_central = rho_mesh / central_alpha[2]
    allowed = np.flatnonzero(r_mesh_central >= R_MIN_PREDICTOR)
    U, singular_values, _ = np.linalg.svd(delta[allowed], full_matrices=False)
    local = _greedy_maxvol_indices(U[:, :K_PREDICTORS])
    idx = allowed[local]
    return {
        "indices": idx,
        "s_points": rho_mesh[idx],
        "r_points": r_mesh_central[idx],
        "singular_values": singular_values,
    }


def _make_predictor_pack(interaction, rho_mesh, train_alphas, central_alpha):
    points = _delta_maxvol_points(interaction, rho_mesh, train_alphas, central_alpha)
    p_train_raw = np.asarray([interaction.tilde(points["s_points"], alpha) for alpha in train_alphas])
    p_center = interaction.tilde(points["s_points"], central_alpha)
    p_scale = np.maximum(np.max(np.abs(p_train_raw - p_center), axis=0), 1e-14)
    return {
        "s_points": points["s_points"],
        "r_points": points["r_points"],
        "singular_values": points["singular_values"],
        "p_center": p_center,
        "p_scale": p_scale,
        "p_train_raw": p_train_raw,
    }


def _predictor_features(interaction, alphas: np.ndarray, pack: dict):
    p_raw = np.asarray([interaction.tilde(pack["s_points"], alpha) for alpha in alphas])
    return np.column_stack([np.ones(len(p_raw)), (p_raw - pack["p_center"]) / pack["p_scale"]])


def _make_channel_data(interaction, base_solver, rho_mesh, train_alphas, test_alphas, central_alpha):
    solver = base_solver.clone_for_new_interaction(interaction)
    phi_train_raw = np.asarray([solver.phi(alpha, rho_mesh) for alpha in train_alphas])
    phi_test_raw = np.asarray([solver.phi(alpha, rho_mesh) for alpha in test_alphas])
    phi_central_raw = solver.phi(central_alpha, rho_mesh)

    # Keep the same wavefunction scaling convention used in the earlier clean
    # notebook diagnostics so errors are comparable across channels.
    phi_train = p3.scale_wavefunctions_like_rose_basis(phi_train_raw, rho_mesh)
    phi_test = p3.scale_wavefunctions_like_rose_basis(phi_test_raw, rho_mesh)
    phi0 = p3.scale_wavefunctions_like_rose_basis(phi_central_raw[np.newaxis, :], rho_mesh)[0]

    basis = p3.CustomBasis(
        solutions=phi_train.T.copy(),
        phi_0=phi0.copy(),
        rho_mesh=rho_mesh,
        n_basis=N_PHI,
        solver=solver,
        subtract_phi0=True,
        use_svd=True,
        center=False,
        scale=False,
    )
    coeff_train_ls = p3.least_squares_basis_coefficients(basis, phi_train, rho_mesh)
    coeff_test_ls = p3.least_squares_basis_coefficients(basis, phi_test, rho_mesh)
    wf_train_ls, _ = _wavefunction_rel_errors(basis, coeff_train_ls, phi_train, rho_mesh)
    wf_test_ls, _ = _wavefunction_rel_errors(basis, coeff_test_ls, phi_test, rho_mesh)
    return {
        "basis": basis,
        "solver": solver,
        "phi_train": phi_train,
        "phi_test": phi_test,
        "coeff_train_ls": coeff_train_ls,
        "coeff_test_ls": coeff_test_ls,
        "wf_train_ls": wf_train_ls,
        "wf_test_ls": wf_test_ls,
    }


def _evaluate_lrom(data: dict, fit, features: np.ndarray, split: str, rho_mesh: np.ndarray):
    coeff = p3.model_predict(features, fit.matrices, fit.vectors)
    wf_err, phi_pred = _wavefunction_rel_errors(data["basis"], coeff, data[f"phi_{split}"], rho_mesh)
    coeff_err = np.linalg.norm(coeff - data[f"coeff_{split}_ls"], axis=1)
    cond = np.asarray([
        np.linalg.cond(sum(f * m for f, m in zip(row, fit.matrices)))
        for row in features
    ])
    return {"coeff": coeff, "wf_err": wf_err, "phi": phi_pred, "coeff_err": coeff_err, "cond": cond}


def recompute(cache_path: str | Path, force: bool = False) -> dict:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            return pickle.load(f)

    t_all = time.perf_counter()
    rho_mesh = np.linspace(1e-8, 8 * np.pi, N_MESH)
    train_samples = _sample_box(N_TRAIN, TRAIN_A_RANGE, TRAIN_E_RANGE, seed=20260530)
    test_samples = _sample_box(N_TEST, TEST_A_RANGE, TEST_E_RANGE, seed=20260531)
    central_sample = np.asarray([A_CENTRAL, E_CENTRAL], dtype=float)
    central_alpha = alpha_from_AE(central_sample)
    train_alphas = alphas_from_AE(train_samples)
    test_alphas = alphas_from_AE(test_samples)

    base_solver = p3.rose.SchroedingerEquation.make_base_solver(
        s_0=6 * np.pi,
        rk_tols=[1e-9, 1e-9],
        domain=np.asarray([1e-8, 8 * np.pi]),
    )
    interactions = p3.rose.koning_delaroche.EnergizedKoningDelaroche(
        training_info=np.vstack([train_alphas, central_alpha]),
        explicit_training=True,
        l_max=L_MAX,
        n_basis=N_U,
        rho_mesh=rho_mesh,
    )

    channels = {}
    for key in SELECTED_CHANNELS:
        ell, ch = key
        interaction = interactions.interactions[ell][ch]
        data = _make_channel_data(
            interaction, base_solver, rho_mesh, train_alphas, test_alphas, central_alpha
        )
        pack = _make_predictor_pack(interaction, rho_mesh, train_alphas, central_alpha)
        features_train = _predictor_features(interaction, train_alphas, pack)
        features_test = _predictor_features(interaction, test_alphas, pack)
        fit = p3.fit_identity_m0_rf(
            name=f"KD A/E predictor RF-LROM l={ell}, ch={ch}",
            feature_values=features_train,
            coeff_ref=data["coeff_train_ls"],
        )
        train_eval = _evaluate_lrom(data, fit, features_train, "train", rho_mesh)
        test_eval = _evaluate_lrom(data, fit, features_test, "test", rho_mesh)
        channels[key] = {
            "ell": ell,
            "channel": ch,
            "interaction": interaction,
            "data": data,
            "pack": pack,
            "fit": fit,
            "features_train": features_train,
            "features_test": features_test,
            "train": train_eval,
            "test": test_eval,
        }

    def evaluate_scan(key, param: str, n: int = 72):
        if param == "A":
            x = np.arange(int(np.ceil(TEST_A_RANGE[0])), int(np.floor(TEST_A_RANGE[1])) + 1, dtype=float)
            samples = np.column_stack([x, np.full(len(x), E_CENTRAL)])
            train_span = TRAIN_A_RANGE
        elif param == "E":
            x = np.linspace(TEST_E_RANGE[0], TEST_E_RANGE[1], n)
            samples = np.column_stack([np.full(n, A_CENTRAL), x])
            train_span = TRAIN_E_RANGE
        else:
            raise ValueError(param)

        alphas = alphas_from_AE(samples)
        ell, ch = key
        interaction = interactions.interactions[ell][ch]
        ch_data = channels[key]
        solver = ch_data["data"]["solver"]
        phi_raw = np.asarray([solver.phi(alpha, rho_mesh) for alpha in alphas])
        phi = p3.scale_wavefunctions_like_rose_basis(phi_raw, rho_mesh)
        coeff_ls = p3.least_squares_basis_coefficients(ch_data["data"]["basis"], phi, rho_mesh)
        features = _predictor_features(interaction, alphas, ch_data["pack"])
        coeff_lrom = p3.model_predict(features, ch_data["fit"].matrices, ch_data["fit"].vectors)
        wf_ls, _ = _wavefunction_rel_errors(ch_data["data"]["basis"], coeff_ls, phi, rho_mesh)
        wf_lrom, _ = _wavefunction_rel_errors(ch_data["data"]["basis"], coeff_lrom, phi, rho_mesh)
        coeff_err = np.linalg.norm(coeff_lrom - coeff_ls, axis=1)
        return {
            "x": x,
            "samples": samples,
            "train_span": train_span,
            "coeff_ls": coeff_ls,
            "coeff_lrom": coeff_lrom,
            "wf_ls": wf_ls,
            "wf_lrom": wf_lrom,
            "coeff_err": coeff_err,
        }

    scans = {
        key: {"A": evaluate_scan(key, "A"), "E": evaluate_scan(key, "E")}
        for key in SELECTED_CHANNELS
    }

    def build_cross_sections():
        angles = np.linspace(1.0, 179.0, N_ANGLES) * np.pi / 180.0
        angles_deg = angles * 180.0 / np.pi
        central_bases = []
        coeff_train_all = []
        coeff_test_all = []
        fit_rows = []
        pack_rows = []
        interaction_rows = []

        for ell, interaction_list in enumerate(interactions.interactions):
            basis_row = []
            coeff_train_row = []
            coeff_test_row = []
            fit_row = []
            pack_row = []
            interaction_row = []
            for ch, interaction in enumerate(interaction_list):
                key = (ell, ch)
                if key in channels:
                    ch_data = channels[key]
                    data = ch_data["data"]
                    pack = ch_data["pack"]
                    fit = ch_data["fit"]
                else:
                    data = _make_channel_data(
                        interaction, base_solver, rho_mesh, train_alphas, test_alphas, central_alpha
                    )
                    pack = _make_predictor_pack(interaction, rho_mesh, train_alphas, central_alpha)
                    features_train = _predictor_features(interaction, train_alphas, pack)
                    fit = p3.fit_identity_m0_rf(
                        name=f"KD A/E predictor RF-LROM l={ell}, ch={ch}",
                        feature_values=features_train,
                        coeff_ref=data["coeff_train_ls"],
                    )
                basis_row.append(data["basis"])
                coeff_train_row.append(data["coeff_train_ls"])
                coeff_test_row.append(data["coeff_test_ls"])
                fit_row.append(fit)
                pack_row.append(pack)
                interaction_row.append(interaction)
            central_bases.append(basis_row)
            coeff_train_all.append(coeff_train_row)
            coeff_test_all.append(coeff_test_row)
            fit_rows.append(fit_row)
            pack_rows.append(pack_row)
            interaction_rows.append(interaction_row)

        sae = p3.rose.ScatteringAmplitudeEmulator(
            interactions,
            central_bases,
            l_max=L_MAX,
            angles=angles,
            s_0=base_solver.s_0,
            Smatrix_abs_tol=1e-8,
            initialize_emulator=True,
        )

        def ls_smatrix_elements(sample_index: int, coeffs):
            n_l = len(sae.rbes)
            splus = np.zeros(n_l, dtype=np.complex128)
            sminus = np.zeros(n_l, dtype=np.complex128)
            for ell in range(n_l):
                splus[ell] = p3.s_matrix_from_coefficients(sae.rbes[ell][0], coeffs[ell][0][sample_index])
                if ell == 0:
                    sminus[ell] = splus[ell]
                else:
                    sminus[ell] = p3.s_matrix_from_coefficients(sae.rbes[ell][1], coeffs[ell][1][sample_index])
            return splus, sminus

        def lrom_smatrix(alpha: np.ndarray):
            n_l = len(sae.rbes)
            alpha_row = np.asarray(alpha, dtype=float)[np.newaxis, :]
            splus = np.zeros(n_l, dtype=np.complex128)
            sminus = np.zeros(n_l, dtype=np.complex128)
            for ell in range(n_l):
                features = _predictor_features(interaction_rows[ell][0], alpha_row, pack_rows[ell][0])
                coeff = p3.model_predict(features, fit_rows[ell][0].matrices, fit_rows[ell][0].vectors)[0]
                splus[ell] = p3.s_matrix_from_coefficients(sae.rbes[ell][0], coeff)
                if ell == 0:
                    sminus[ell] = splus[ell]
                else:
                    features = _predictor_features(interaction_rows[ell][1], alpha_row, pack_rows[ell][1])
                    coeff = p3.model_predict(features, fit_rows[ell][1].matrices, fit_rows[ell][1].vectors)[0]
                    sminus[ell] = p3.s_matrix_from_coefficients(sae.rbes[ell][1], coeff)
            return splus, sminus

        def compute_split(samples: np.ndarray, coeffs):
            xs = {"FOM": [], "LS floor": [], "LROM": []}
            for i, alpha in enumerate(alphas_from_AE(samples)):
                splus, sminus = p3.exact_smatrix_elements_fixed(sae, alpha)
                xs["FOM"].append(p3.dsdo_from_smatrix(sae, alpha, splus, sminus))

                splus, sminus = ls_smatrix_elements(i, coeffs)
                xs["LS floor"].append(p3.dsdo_from_smatrix(sae, alpha, splus, sminus))

                splus, sminus = lrom_smatrix(alpha)
                xs["LROM"].append(p3.dsdo_from_smatrix(sae, alpha, splus, sminus))
            return {key: np.asarray(val) for key, val in xs.items()}

        xs_train = compute_split(train_samples, coeff_train_all)
        xs_test = compute_split(test_samples, coeff_test_all)

        def metric_arrays(xs):
            methods = ["LS floor", "LROM"]
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
                    l2_vals.append(np.sqrt(
                        np.trapz((pred - ref) ** 2, angles)
                        / np.maximum(np.trapz(ref ** 2, angles), 1e-30)
                    ))
                median_pointwise.append(med)
                l2.append(l2_vals)
                max_pointwise.append(mx)
            return np.asarray(median_pointwise), np.asarray(l2), np.asarray(max_pointwise)

        train_med, train_l2, train_max = metric_arrays(xs_train)
        test_med, test_l2, test_max = metric_arrays(xs_test)
        return {
            "angles_deg": angles_deg,
            "methods": np.asarray(["LS floor", "LROM"]),
            "xs_train_fom": xs_train["FOM"],
            "xs_train_ls": xs_train["LS floor"],
            "xs_train_lrom": xs_train["LROM"],
            "xs_test_fom": xs_test["FOM"],
            "xs_test_ls": xs_test["LS floor"],
            "xs_test_lrom": xs_test["LROM"],
            "train_median_pointwise": train_med,
            "train_l2": train_l2,
            "train_max_pointwise": train_max,
            "test_median_pointwise": test_med,
            "test_l2": test_l2,
            "test_max_pointwise": test_max,
            "l_max": L_MAX,
            "n_angles": N_ANGLES,
        }

    cross_sections = build_cross_sections()

    result = {
        "config": {
            "Z_TARGET": Z_TARGET,
            "A_CENTRAL": A_CENTRAL,
            "E_CENTRAL": E_CENTRAL,
            "TRAIN_A_RANGE": TRAIN_A_RANGE,
            "TRAIN_E_RANGE": TRAIN_E_RANGE,
            "TEST_A_RANGE": TEST_A_RANGE,
            "TEST_E_RANGE": TEST_E_RANGE,
            "N_TRAIN": N_TRAIN,
            "N_TEST": N_TEST,
            "N_PHI": N_PHI,
            "K_PREDICTORS": K_PREDICTORS,
            "N_U": N_U,
            "N_MESH": N_MESH,
            "N_ANGLES": N_ANGLES,
            "L_MAX": L_MAX,
            "SELECTED_CHANNELS": SELECTED_CHANNELS,
        },
        "rho_mesh": rho_mesh,
        "central_sample": central_sample,
        "central_alpha": central_alpha,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "train_alphas": train_alphas,
        "test_alphas": test_alphas,
        "channels": channels,
        "scans": scans,
        "cross_sections": cross_sections,
        "recompute_seconds": time.perf_counter() - t_all,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result



def representative_wavefunctions(ae_data: dict, samples: np.ndarray) -> dict:
    """Evaluate FOM, LS-floor, and predictor-LROM wavefunctions at A/E samples."""

    samples = np.asarray(samples, dtype=float)
    rho_mesh = ae_data["rho_mesh"]
    alphas = alphas_from_AE(samples)
    out = {
        "samples": samples,
        "alphas": alphas,
        "rho_mesh": rho_mesh,
        "channels": {},
    }
    for key, channel in ae_data["channels"].items():
        data = channel["data"]
        solver = data["solver"]
        phi_raw = np.asarray([solver.phi(alpha, rho_mesh) for alpha in alphas])
        phi_fom = p3.scale_wavefunctions_like_rose_basis(phi_raw, rho_mesh)

        coeff_ls = p3.least_squares_basis_coefficients(data["basis"], phi_fom, rho_mesh)
        wf_ls, phi_ls = _wavefunction_rel_errors(data["basis"], coeff_ls, phi_fom, rho_mesh)

        interaction = channel.get("interaction")
        if interaction is None:
            ell, ch = key
            interactions = p3.rose.koning_delaroche.EnergizedKoningDelaroche(
                training_info=np.vstack([ae_data["train_alphas"], ae_data["central_alpha"]]),
                explicit_training=True,
                l_max=ae_data["config"]["L_MAX"],
                n_basis=ae_data["config"]["N_U"],
                rho_mesh=rho_mesh,
            )
            interaction = interactions.interactions[ell][ch]
        features = _predictor_features(interaction, alphas, channel["pack"])
        coeff_lrom = p3.model_predict(features, channel["fit"].matrices, channel["fit"].vectors)
        wf_lrom, phi_lrom = _wavefunction_rel_errors(data["basis"], coeff_lrom, phi_fom, rho_mesh)

        out["channels"][key] = {
            "phi_fom": phi_fom,
            "phi_ls": phi_ls,
            "phi_lrom": phi_lrom,
            "coeff_ls": coeff_ls,
            "coeff_lrom": coeff_lrom,
            "wf_ls": wf_ls,
            "wf_lrom": wf_lrom,
        }
    return out
