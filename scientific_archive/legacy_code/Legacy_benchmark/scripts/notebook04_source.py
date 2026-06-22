"""Source cells for clean demo notebook 04."""

from __future__ import annotations


def make_cells_04(md, code, setup_code):
    return [
        md(
            r"""
            # 04. Predictor LROM Across Isotopes And Beam Energies

            This notebook asks what happens when the control variables are the
            target isotope and neutron beam energy.

            We use the Koning-Delaroche global optical potential through ROSE.
            Each sample is a calcium isotope, fixed `Z=20`, with parameters
            `(A, E_lab)`.  For every sample we recompute the KD optical
            potential and the scattering kinematics before solving the FOM
            wavefunction.

            The model trained here is only the central-reference predictor
            RF-LROM.  There is no ROSE ROM comparison in this notebook.  The
            target is the least-squares coordinate vector in the central reduced
            basis, so the main questions are whether the predictor LROM learns
            the optimal reduced coordinates and where extrapolation in `(A,E)`
            becomes fragile.
            """
        ),
        code(setup_code),
        code(
            r"""
            from lrom_demo import ae_wavefunction_extrapolation as ae
            ae = importlib.reload(ae)
            """
        ),
        md(
            r"""
            ## 1. Generate Or Load The A/E Wavefunction Dataset

            The training box is deliberately smaller than the full diagnostic
            box.  This makes extrapolation visible: the gray rectangle is where
            the RF-LROM sees training wavefunctions, while the orange points
            probe the wider requested range `30 <= A <= 60`,
            `8 <= E_lab <= 30 MeV`.
            """
        ),
        code(
            r"""
            CACHE = ROOT / "outputs" / "cached_ae" / "wavefunction_predictor_lrom_a30_60_emin8_integerA_lmax15_k12_ntrain200.pkl"
            FORCE_RECOMPUTE_AE = False

            t0 = time.perf_counter()
            ae_data = ae.recompute(CACHE, force=FORCE_RECOMPUTE_AE)
            print(f"loaded/recomputed in {time.perf_counter() - t0:.2f} s")
            print(f"cached recompute seconds: {ae_data['recompute_seconds']:.2f}")

            cfg = ae_data["config"]
            n = cfg["N_PHI"]
            K = cfg["K_PREDICTORS"]
            n_features = K + 1
            n_equations_per_channel = cfg["N_TRAIN"] * n
            n_unknowns_per_channel = K * n * n + n_features * n
            n_selected_channels = len(cfg["SELECTED_CHANNELS"])
            n_all_scattering_channels = 1 + 2 * cfg["L_MAX"]
            config_rows = list(cfg.items()) + [
                ("RF_LROM_FEATURE_COLUMNS", n_features),
                ("RF_LROM_EQUATIONS_PER_CHANNEL", n_equations_per_channel),
                ("RF_LROM_UNKNOWNS_PER_CHANNEL", n_unknowns_per_channel),
                ("RF_LROM_EQS_PER_UNKNOWN_PER_CHANNEL", n_equations_per_channel / n_unknowns_per_channel),
                ("RF_LROM_SELECTED_CHANNELS_TOTAL_EQUATIONS", n_selected_channels * n_equations_per_channel),
                ("RF_LROM_SELECTED_CHANNELS_TOTAL_UNKNOWNS", n_selected_channels * n_unknowns_per_channel),
                ("RF_LROM_ALL_LMAX_CHANNELS", n_all_scattering_channels),
                ("RF_LROM_ALL_LMAX_TOTAL_EQUATIONS", n_all_scattering_channels * n_equations_per_channel),
                ("RF_LROM_ALL_LMAX_TOTAL_UNKNOWNS", n_all_scattering_channels * n_unknowns_per_channel),
            ]
            pd.DataFrame(config_rows, columns=["quantity", "value"])
            """
        ),
        code(
            r"""
            cfg = ae_data["config"]
            train = ae_data["train_samples"]
            test = ae_data["test_samples"]
            central = ae_data["central_sample"]

            fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=160)
            ax.scatter(test[:, 0], test[:, 1], s=22, alpha=0.45, color="tab:orange", label="wide test")
            ax.scatter(train[:, 0], train[:, 1], s=28, alpha=0.70, color="tab:blue", label="train")
            ax.scatter([central[0]], [central[1]], s=90, marker="*", color="black", label="central")
            ax.add_patch(plt.Rectangle(
                (cfg["TRAIN_A_RANGE"][0], cfg["TRAIN_E_RANGE"][0]),
                cfg["TRAIN_A_RANGE"][1] - cfg["TRAIN_A_RANGE"][0],
                cfg["TRAIN_E_RANGE"][1] - cfg["TRAIN_E_RANGE"][0],
                fill=False, lw=2.0, ec="0.25", label="training box",
            ))
            ax.set_xlabel("A for calcium isotopes (Z=20)")
            ax.set_ylabel(r"$E_{\rm lab}$ [MeV]")
            ax.set_title("A/E samples for KD wavefunction extrapolation")
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(OUT / "ae_sample_design.png", dpi=260, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        md(
            r"""
            ## 2. What KD Parameters Are Being Used?

            For a few representative points, the helper calls KD directly and
            prepends the kinematic entries `(E_com, mu, k)` expected by ROSE's
            energized interaction.
            """
        ),
        code(
            r"""
            representative_samples = np.array([[32.0, 8.5], [40.0, 14.1], [58.0, 28.0]])
            rows = []
            for A, E in representative_samples:
                alpha = ae.alpha_from_AE([A, E])
                rows.append({
                    "A": int(A),
                    "Z": ae.Z_TARGET,
                    "E_lab [MeV]": E,
                    "E_com [MeV]": alpha[0],
                    "mu [MeV]": alpha[1],
                    "k [1/fm]": alpha[2],
                    "KD optical alpha length": len(alpha) - 3,
                    "first KD optical entry": alpha[3],
                    "second KD optical entry": alpha[4],
                })
            pd.DataFrame(rows)
            """
        ),
        code(
            r"""
            KD_ALPHA_LABELS = [
                "Vv", "Rv", "av",
                "Wv", "Rwv", "awv",
                "Wd", "Rd", "ad",
                "Vso", "Rso", "aso",
                "Wso", "Rwso", "awso",
            ]

            E_grid = np.linspace(cfg["TEST_E_RANGE"][0], cfg["TEST_E_RANGE"][1], 120)
            kd_rows = []
            for E in E_grid:
                alpha = ae.alpha_from_AE([40.0, E])
                row = {
                    "A": 40,
                    "Z": ae.Z_TARGET,
                    "E_lab [MeV]": E,
                    "E_com [MeV]": alpha[0],
                    "mu [MeV]": alpha[1],
                    "k [1/fm]": alpha[2],
                }
                row.update({name: value for name, value in zip(KD_ALPHA_LABELS, alpha[3:])})
                kd_rows.append(row)
            kd_param_table = pd.DataFrame(kd_rows)

            fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), dpi=170, sharex=True)
            groups = [
                (["E_com [MeV]", "k [1/fm]"], "kinematics"),
                (["Vv", "Wv", "Wd", "Vso", "Wso"], "depth strengths [MeV]"),
                (["Rv", "Rwv", "Rd", "Rso", "Rwso"], "radii [fm]"),
                (["av", "awv", "ad", "aso", "awso"], "diffuseness [fm]"),
            ]
            for ax, (names, ylabel) in zip(axes.flat, groups):
                for name in names:
                    ax.plot(kd_param_table["E_lab [MeV]"], kd_param_table[name], lw=2.0, label=name)
                ax.axvspan(cfg["TRAIN_E_RANGE"][0], cfg["TRAIN_E_RANGE"][1], color="0.90", zorder=-5)
                ax.axvline(cfg["E_CENTRAL"], color="0.25", lw=1.2)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=8, ncols=2)
            for ax in axes[-1]:
                ax.set_xlabel(r"$E_{\rm lab}$ [MeV]")
            fig.suptitle(r"Koning-Delaroche parameters for $^{40}$Ca$(n,n)$")
            fig.tight_layout()
            fig.savefig(OUT / "kd_ca40_parameters_vs_energy.png", dpi=300, bbox_inches="tight")
            display(fig)
            plt.close(fig)

            kd_param_table.iloc[::20].round(5)
            """
        ),
        code(
            r"""
            A_grid = np.arange(int(cfg["TEST_A_RANGE"][0]), int(cfg["TEST_A_RANGE"][1]) + 1)
            kd_a_rows = []
            for A in A_grid:
                alpha = ae.alpha_from_AE([float(A), cfg["E_CENTRAL"]])
                row = {
                    "A": int(A),
                    "Z": ae.Z_TARGET,
                    "E_lab [MeV]": cfg["E_CENTRAL"],
                    "E_com [MeV]": alpha[0],
                    "mu [MeV]": alpha[1],
                    "k [1/fm]": alpha[2],
                }
                row.update({name: value for name, value in zip(KD_ALPHA_LABELS, alpha[3:])})
                kd_a_rows.append(row)
            kd_a_param_table = pd.DataFrame(kd_a_rows)

            fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), dpi=170, sharex=True)
            groups = [
                (["E_com [MeV]", "mu [MeV]", "k [1/fm]"], "kinematics"),
                (["Vv", "Wv", "Wd", "Vso", "Wso"], "depth strengths [MeV]"),
                (["Rv", "Rwv", "Rd", "Rso", "Rwso"], "radii [fm]"),
                (["av", "awv", "ad", "aso", "awso"], "diffuseness [fm]"),
            ]
            for ax, (names, ylabel) in zip(axes.flat, groups):
                for name in names:
                    ax.plot(kd_a_param_table["A"], kd_a_param_table[name], marker="o", ms=3.2, lw=1.7, label=name)
                ax.axvspan(cfg["TRAIN_A_RANGE"][0], cfg["TRAIN_A_RANGE"][1], color="0.90", zorder=-5)
                ax.axvline(cfg["A_CENTRAL"], color="0.25", lw=1.2)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=8, ncols=2)
            for ax in axes[-1]:
                ax.set_xlabel("A for calcium isotopes (Z=20)")
            fig.suptitle(rf"Koning-Delaroche parameters for calcium at $E_{{\rm lab}}={cfg['E_CENTRAL']}$ MeV")
            fig.tight_layout()
            fig.savefig(OUT / "kd_calcium_parameters_vs_A.png", dpi=300, bbox_inches="tight")
            display(fig)
            plt.close(fig)

            kd_a_param_table.round(5)
            """
        ),
        md(
            r"""
            ## 3. Delta-Maxvol Predictor Points

            For each selected partial-wave channel, the predictor features are
            KD potential values at delta-maxvol points.  The features are
            centered and scaled so they vanish at `(A,E)=(40,14.1 MeV)`.

            These points are selected by the local helper in this demo, not by
            a ROSE EIM/maxvol routine.  For each channel we build the training
            matrix of variations

            `Delta U(s; A,E) = U(s; A,E) - U(s; A_c,E_c)`

            using the integer calcium-isotope training samples.  We then take
            an SVD of those variations and run a small greedy/maxvol-style row
            selector on the leading left singular vectors.  So A-dependence is
            included directly through the KD-generated potential variations.
            """
        ),
        code(
            r"""
            rows = []
            for key, channel in ae_data["channels"].items():
                pack = channel["pack"]
                rows.append({
                    "ell": key[0],
                    "channel": key[1],
                    "K": len(pack["s_points"]),
                    "s points": np.array2string(pack["s_points"], precision=3),
                    "r points at central k [fm]": np.array2string(pack["r_points"], precision=3),
                    "fit rank": channel["fit"].rank,
                    "residual MSE": channel["fit"].residual_mse,
                })
            pd.DataFrame(rows)
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(len(ae_data["channels"]), 1, figsize=(8.2, 6.3), dpi=160, sharex=True)
            if len(ae_data["channels"]) == 1:
                axes = [axes]
            for ax, (key, channel) in zip(axes, ae_data["channels"].items()):
                pack = channel["pack"]
                ax.semilogy(pack["singular_values"][:24] / pack["singular_values"][0], marker="o", ms=3.8, lw=1.5)
                ax.axvline(cfg["K_PREDICTORS"] - 1, color="tab:purple", ls="--", lw=1.4, label=f"K={cfg['K_PREDICTORS']}")
                ax.set_ylabel(f"l={key[0]}, ch={key[1]}\nrelative SV")
                ax.grid(True, which="both", alpha=0.25)
                ax.legend(fontsize=8)
            axes[-1].set_xlabel("singular-value index")
            fig.suptitle("Low-rank structure of KD potential variations")
            fig.tight_layout()
            fig.savefig(OUT / "ae_predictor_singular_values.png", dpi=260, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        md(
            r"""
            ## 3b. KD Potential Rainbows And Predictor Points

            The LROM predictors are samples of the scaled operator potential
            `U(s; A,E) / E_com` at the red points.  The rainbows below split
            the KD potential into central and spin-orbit pieces while varying
            one physical input at a time through the central point.
            """
        ),
        code(
            r"""
            def _ws(r, R, a):
                x = np.clip((r - R) / a, -700, 700)
                return 1.0 / (1.0 + np.exp(x))

            def _ws_prime(r, R, a):
                x = np.clip((r - R) / a, -700, 700)
                ex = np.exp(x)
                return -ex / (a * (1.0 + ex) ** 2)

            def _thomas(r, R, a):
                return _ws_prime(r, R, a) / np.maximum(r, 1e-12)

            def kd_scaled_components(s_mesh, alpha, l_dot_s):
                E_com = alpha[0]
                k = alpha[2]
                r = np.asarray(s_mesh) / k
                vv, rv, av, wv, rwv, awv, wd, rd, ad, vso, rso, aso, wso, rwso, awso = alpha[3:]
                pieces = {
                    "real volume": (-vv * _ws(r, rv, av)) / E_com,
                    "imag volume": (-wv * _ws(r, rwv, awv)) / E_com,
                    "imag surface": ((4.0 * ad) * wd * _ws_prime(r, rd, ad)) / E_com,
                    "real spin-orbit": (l_dot_s * vso * _thomas(r, rso, aso) / ae.p3.rose.koning_delaroche.MASS_PION**2) / E_com,
                    "imag spin-orbit": (l_dot_s * wso * _thomas(r, rwso, awso) / ae.p3.rose.koning_delaroche.MASS_PION**2) / E_com,
                }
                return pieces

            def _variation_grid(param):
                if param == "A":
                    x = np.arange(int(cfg["TEST_A_RANGE"][0]), int(cfg["TEST_A_RANGE"][1]) + 1, dtype=float)
                    samples = np.column_stack([x, np.full(len(x), cfg["E_CENTRAL"])])
                    train_span = cfg["TRAIN_A_RANGE"]
                    xlabel = "A for calcium isotopes (Z=20)"
                elif param == "E":
                    x = np.linspace(cfg["TEST_E_RANGE"][0], cfg["TEST_E_RANGE"][1], 55)
                    samples = np.column_stack([np.full(len(x), cfg["A_CENTRAL"]), x])
                    train_span = cfg["TRAIN_E_RANGE"]
                    xlabel = r"$E_{\rm lab}$ [MeV]"
                else:
                    raise ValueError(param)
                return x, samples, train_span, xlabel

            def plot_kd_component_rainbows(key):
                channel = ae_data["channels"][key]
                pack = channel["pack"]
                interaction = channel["interaction"]
                l_dot_s = interaction.spin_orbit_term.l_dot_s
                s_mesh = ae_data["rho_mesh"]
                terms = ["real volume", "imag volume", "imag surface", "real spin-orbit", "imag spin-orbit"]

                fig, axes = plt.subplots(len(terms), 2, figsize=(12.4, 10.6), dpi=170, sharex="col")
                for col, param in enumerate(["E", "A"]):
                    x, samples, train_span, xlabel = _variation_grid(param)
                    alphas = ae.alphas_from_AE(samples)
                    cmap = plt.cm.viridis
                    colors = cmap(np.linspace(0.05, 0.95, len(samples)))
                    center_components = kd_scaled_components(pack["s_points"], ae_data["central_alpha"], l_dot_s)
                    for row, term in enumerate(terms):
                        ax = axes[row, col]
                        for alpha, color in zip(alphas, colors):
                            y = kd_scaled_components(s_mesh, alpha, l_dot_s)[term]
                            ax.plot(s_mesh, y.real, color=color, lw=1.0, alpha=0.74)
                        ax.scatter(
                            pack["s_points"],
                            center_components[term].real,
                            s=34,
                            color="crimson",
                            edgecolor="white",
                            linewidth=0.45,
                            zorder=5,
                            label="predictor points" if row == 0 else None,
                        )
                        for s_point in pack["s_points"]:
                            ax.axvline(s_point, color="crimson", lw=0.7, alpha=0.16, zorder=-4)
                        ax.set_ylabel(term)
                        ax.grid(True, alpha=0.22)
                        if row == 0:
                            ax.set_title(f"vary {param}")
                            ax.legend(loc="best", fontsize=8)
                        if row == len(terms) - 1:
                            ax.set_xlabel("operator coordinate s")
                        vals = []
                        for alpha in alphas:
                            vals.append(kd_scaled_components(s_mesh, alpha, l_dot_s)[term].real)
                        vals = np.asarray(vals)
                        lo, hi = np.quantile(vals, [0.01, 0.99])
                        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                            pad = 0.10 * (hi - lo)
                            ax.set_ylim(lo - pad, hi + pad)
                fig.suptitle(f"KD scaled potential-term rainbows with predictor points, l={key[0]}, ch={key[1]}", y=0.995)
                fig.tight_layout()
                return fig

            for key in ae_data["channels"]:
                fig = plot_kd_component_rainbows(key)
                fig.savefig(OUT / f"ae_kd_potential_rainbows_l{key[0]}_ch{key[1]}.png", dpi=300, bbox_inches="tight")
                display(fig)
                plt.close(fig)
            """

        ),
        md(
            r"""
            ## 4. Coefficient Trajectories

            These panels compare the LS target coordinates with the predictor
            LROM coordinates along one-parameter A and E scans through the
            central point.  The gray band marks the training interval.
            """
        ),
        code(
            r"""
            def plot_coeff_scans(key):
                scans = ae_data["scans"][key]
                fig, axes = plt.subplots(2, cfg["N_PHI"], figsize=(13.5, 5.8), dpi=160, sharex="row")
                for row, param in enumerate(["A", "E"]):
                    scan = scans[param]
                    x = scan["x"]
                    for j in range(cfg["N_PHI"]):
                        ax = axes[row, j]
                        ax.plot(x, scan["coeff_ls"][:, j].real, color="black", lw=2.4, label="LS target")
                        ax.plot(x, scan["coeff_lrom"][:, j].real, color="tab:purple", ls="-.", lw=2.2, label=f"predictor LROM (n={cfg['N_PHI']}, K={cfg['K_PREDICTORS']})")
                        ax.axvspan(scan["train_span"][0], scan["train_span"][1], color="0.88", zorder=-5)
                        ax.axvline(central[0 if param == "A" else 1], color="0.35", lw=1.2)
                        ax.set_title(rf"$a_{{{j+1}}}$")
                        ax.grid(True, alpha=0.25)
                        if j == 0:
                            ax.set_ylabel(r"Re $a_j$")
                        ax.set_xlabel("A" if param == "A" else r"$E_{\rm lab}$ [MeV]")
                        if row == 0 and j == 0:
                            ax.legend(fontsize=8)
                fig.suptitle(f"KD A/E coefficient scans, l={key[0]}, ch={key[1]}")
                fig.tight_layout()
                return fig

            for key in ae_data["channels"]:
                fig = plot_coeff_scans(key)
                fig.savefig(OUT / f"ae_coeff_scans_l{key[0]}_ch{key[1]}.png", dpi=260, bbox_inches="tight")
                display(fig)
                plt.close(fig)
            """
        ),
        md(
            r"""
            ## 5. Wavefunction Error Scans

            The LS floor is the best reconstruction possible in the central
            basis.  The predictor LROM error includes both basis error and
            reduced-equation error.
            """
        ),
        code(
            r"""
            WF_ERROR_YMIN = 1e-5
            COEFF_ERROR_YMIN = 1e-5

            def plot_error_scans(key):
                scans = ae_data["scans"][key]
                fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), dpi=160, sharex="row")
                for row, param in enumerate(["A", "E"]):
                    scan = scans[param]
                    x = scan["x"]
                    xlabel = "A" if param == "A" else r"$E_{\rm lab}$ [MeV]"
                    ax = axes[row, 0]
                    ax.plot(x, scan["wf_ls"], color="0.35", ls="--", lw=2.2, label=f"LS floor (n={cfg['N_PHI']})")
                    ax.plot(x, scan["wf_lrom"], color="tab:purple", lw=2.4, label=f"predictor LROM (n={cfg['N_PHI']}, K={cfg['K_PREDICTORS']})")
                    ax.set_yscale("log")
                    ax.set_ylim(bottom=WF_ERROR_YMIN)
                    ax.set_ylabel("WF relative L2 error")
                    ax.set_xlabel(xlabel)
                    ax.axvspan(scan["train_span"][0], scan["train_span"][1], color="0.88", zorder=-5)
                    ax.axvline(central[0 if param == "A" else 1], color="0.35", lw=1.2)
                    ax.grid(True, which="both", alpha=0.25)
                    if row == 0:
                        ax.legend(fontsize=8)
                    ax = axes[row, 1]
                    ax.plot(x, scan["coeff_err"], color="tab:purple", lw=2.2)
                    ax.set_yscale("log")
                    ax.set_ylim(bottom=COEFF_ERROR_YMIN)
                    ax.set_ylabel(r"$||a_{\rm LROM}-a_{\rm LS}||_2$")
                    ax.set_xlabel(xlabel)
                    ax.axvspan(scan["train_span"][0], scan["train_span"][1], color="0.88", zorder=-5)
                    ax.axvline(central[0 if param == "A" else 1], color="0.35", lw=1.2)
                    ax.grid(True, which="both", alpha=0.25)
                fig.suptitle(f"KD A/E extrapolation errors, l={key[0]}, ch={key[1]}")
                fig.tight_layout()
                return fig

            for key in ae_data["channels"]:
                fig = plot_error_scans(key)
                fig.savefig(OUT / f"ae_error_scans_l{key[0]}_ch{key[1]}.png", dpi=260, bbox_inches="tight")
                display(fig)
                plt.close(fig)
            """
        ),
        md(
            r"""
            ## 6. Representative Wavefunctions

            These are direct wavefunction checks at a few representative
            isotope/energy points.  Solid colored curves are FOM wavefunctions.  The predictor LROM
            reconstructions are always black dashed curves so the emulator
            deviations are easy to pick out against the colored references.
            """
        ),
        code(
            r"""
            REPRESENTATIVE_AE_SAMPLES = np.array([
                [32.0, 8.5],
                [32.0, 14.1],
                [40.0, 14.1],
                [48.0, 22.0],
                [58.0, 28.0],
            ])

            wf_reps = ae.representative_wavefunctions(ae_data, REPRESENTATIVE_AE_SAMPLES)
            pd.DataFrame([
                {"sample": i, "A": int(A), "E_lab [MeV]": E}
                for i, (A, E) in enumerate(REPRESENTATIVE_AE_SAMPLES, start=1)
            ])
            """
        ),
        code(
            r"""
            from matplotlib.lines import Line2D

            colors = plt.cm.tab10(np.linspace(0, 1, len(REPRESENTATIVE_AE_SAMPLES)))
            rho = wf_reps["rho_mesh"]
            for key, reps in wf_reps["channels"].items():
                fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=170)
                sample_handles = []
                for i, ((A, E), color) in enumerate(zip(REPRESENTATIVE_AE_SAMPLES, colors), start=1):
                    label = rf"$A={int(A)}$, $E_{{lab}}={E:g}$ MeV"
                    ax.plot(rho, reps["phi_fom"][i-1].real, color=color, lw=2.5, label=label)
                    ax.plot(rho, reps["phi_lrom"][i-1].real, color="black", ls="--", lw=2.0, alpha=0.80)
                    sample_handles.append(Line2D([], [], color=color, lw=2.5, label=label))
                style_handles = [
                    Line2D([], [], color="0.15", lw=2.6, label="FOM"),
                    Line2D([], [], color="0.15", lw=2.2, ls="--", label=f"LROM (n={cfg['N_PHI']}, K={cfg['K_PREDICTORS']})"),
                ]
                ax.set_xlabel("operator coordinate s")
                ax.set_ylabel("real wavefunction")
                ax.set_title(f"Representative KD wavefunctions, l={key[0]}, ch={key[1]}")
                ax.grid(True, alpha=0.25)
                leg1 = ax.legend(handles=sample_handles, loc="upper right", fontsize=8, title="samples")
                ax.add_artist(leg1)
                ax.legend(handles=style_handles, loc="lower right", fontsize=9)
                fig.tight_layout()
                fig.savefig(OUT / f"ae_representative_wavefunctions_l{key[0]}_ch{key[1]}.png", dpi=280, bbox_inches="tight")
                display(fig)
                plt.close(fig)
            """
        ),
        md(
            r"""
            ## 7. Train vs Extrapolation Error Distributions

            These violins summarize the wavefunction errors over all training
            points and all wide-box test points.
            """
        ),
        code(
            r"""
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            def split_violin(ax, values, x, side, color, width=0.34, alpha=0.70, floor=1e-8):
                values = np.asarray(values, dtype=float)
                values = values[np.isfinite(values) & (values > 0)]
                values = np.clip(values, floor, None)
                y = np.log10(values)
                if len(np.unique(np.round(y, 12))) < 2:
                    grid = np.linspace(y[0] - 0.25, y[0] + 0.25, 160)
                    dens = np.exp(-0.5 * ((grid - y[0]) / 0.06) ** 2)
                else:
                    lo, hi = np.quantile(y, [0.01, 0.99])
                    pad = max(0.12, 0.08 * (hi - lo))
                    grid = np.linspace(lo - pad, hi + pad, 180)
                    bw = max(0.06, 0.25 * np.std(y), 1e-3)
                    dens = np.mean(np.exp(-0.5 * ((grid[:, None] - y[None, :]) / bw) ** 2), axis=1)
                dens = dens / max(np.max(dens), 1e-30) * width
                yy = 10 ** grid
                xx = x - dens if side == "left" else x + dens
                ax.fill_betweenx(yy, x, xx, color=color, alpha=alpha, lw=0)
                dx = -0.18 if side == "left" else 0.18
                ax.scatter(x + dx, np.median(values), marker="D", s=34, color=color, edgecolor="black", linewidth=0.7, zorder=6)

            fig, axes = plt.subplots(1, len(ae_data["channels"]), figsize=(13.0, 4.4), dpi=170, sharey=True)
            if len(ae_data["channels"]) == 1:
                axes = [axes]
            train_color = "#4C78A8"
            test_color = "#F58518"
            method_labels = [f"LS floor\n(n={cfg['N_PHI']})", f"predictor LROM\n(n={cfg['N_PHI']}, K={cfg['K_PREDICTORS']})"]
            for ax, (key, channel) in zip(axes, ae_data["channels"].items()):
                clouds = [
                    (channel["data"]["wf_train_ls"], channel["data"]["wf_test_ls"]),
                    (channel["train"]["wf_err"], channel["test"]["wf_err"]),
                ]
                for xpos, (train_vals, test_vals) in enumerate(clouds, start=1):
                    split_violin(ax, train_vals, xpos, "left", train_color)
                    split_violin(ax, test_vals, xpos, "right", test_color)
                ax.set_yscale("log")
                ax.set_xticks([1, 2], method_labels)
                ax.set_title(f"l={key[0]}, ch={key[1]}")
                ax.grid(True, which="both", alpha=0.25)
            axes[0].set_ylabel("WF relative L2 error")
            axes[-1].legend(
                handles=[
                    Patch(facecolor=train_color, alpha=0.70, label="train"),
                    Patch(facecolor=test_color, alpha=0.70, label="wide test"),
                    Line2D([], [], marker="D", linestyle="None", color="white", markerfacecolor="0.75", markeredgecolor="black", label="median"),
                ],
                loc="upper right",
                fontsize=8,
            )
            fig.suptitle(r"KD calcium isotope/energy wavefunction errors")
            fig.tight_layout()
            fig.savefig(OUT / "ae_wf_error_violins.png", dpi=280, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        md(
            r"""
            ## 8. Where Is Extrapolation Hardest?

            This table lists the wide-box test samples with the largest
            predictor LROM wavefunction errors for each channel.
            """
        ),
        code(
            r"""
            rows = []
            for key, channel in ae_data["channels"].items():
                err = channel["test"]["wf_err"]
                order = np.argsort(err)[-6:][::-1]
                for idx in order:
                    rows.append({
                        "ell": key[0],
                        "channel": key[1],
                        "test index": int(idx),
                        "A": int(test[idx, 0]),
                        "E_lab [MeV]": float(test[idx, 1]),
                        "LROM WF error": float(err[idx]),
                        "LS floor": float(channel["data"]["wf_test_ls"][idx]),
                        "coefficient error": float(channel["test"]["coeff_err"][idx]),
                        "condition number": float(channel["test"]["cond"][idx]),
                    })
            hardest_df = pd.DataFrame(rows)
            hardest_df
            """
        ),
        md(
            r"""
            ## 9. Inspect The Worst `l=4, ch=1` Wavefunction

            The long tail in the wide-test violin is driven by individual
            extrapolation samples.  This cell pulls out the largest
            `l=4, ch=1` predictor-LROM wavefunction error and compares the FOM,
            the LS-floor reconstruction, and the LROM reconstruction directly.
            """
        ),
        code(
            r"""
            key = (4, 1)
            channel = ae_data["channels"][key]
            worst_idx = int(np.argmax(channel["test"]["wf_err"]))
            worst_sample = test[worst_idx]
            worst_summary = pd.DataFrame([{
                "test index": worst_idx,
                "A": int(worst_sample[0]),
                "E_lab [MeV]": float(worst_sample[1]),
                "LROM WF error": float(channel["test"]["wf_err"][worst_idx]),
                "LS floor": float(channel["data"]["wf_test_ls"][worst_idx]),
                "coefficient error": float(channel["test"]["coeff_err"][worst_idx]),
                "condition number": float(channel["test"]["cond"][worst_idx]),
            }])
            worst_summary
            """
        ),
        code(
            r"""
            worst_reps = ae.representative_wavefunctions(ae_data, worst_sample[np.newaxis, :])
            rho = worst_reps["rho_mesh"]
            reps = worst_reps["channels"][key]

            fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), dpi=170, sharex=True)
            axes[0].plot(rho, reps["phi_fom"][0].real, color="tab:blue", lw=2.8, label="FOM")
            axes[0].plot(rho, reps["phi_ls"][0].real, color="0.55", lw=2.1, ls="--", label=f"LS floor (n={cfg['N_PHI']})")
            axes[0].plot(rho, reps["phi_lrom"][0].real, color="black", lw=2.2, ls="--", label=f"LROM (n={cfg['N_PHI']}, K={cfg['K_PREDICTORS']})")
            axes[0].set_ylabel("real wavefunction")
            axes[0].set_title(rf"Worst wide-test sample for $l=4$, ch=1: $A={int(worst_sample[0])}$, $E_{{lab}}={worst_sample[1]:.2f}$ MeV")
            axes[0].legend(fontsize=9)
            axes[0].grid(True, alpha=0.25)

            axes[1].plot(rho, (reps["phi_lrom"][0] - reps["phi_fom"][0]).real, color="black", lw=2.1, label="LROM - FOM")
            axes[1].plot(rho, (reps["phi_ls"][0] - reps["phi_fom"][0]).real, color="0.55", lw=2.0, ls="--", label="LS - FOM")
            axes[1].axhline(0.0, color="0.25", lw=1.0)
            axes[1].set_xlabel("operator coordinate s")
            axes[1].set_ylabel("real residual")
            axes[1].legend(fontsize=9)
            axes[1].grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(OUT / "ae_worst_l4_ch1_wavefunction.png", dpi=300, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        md(
            r"""
            ## 10. Cross-Section Checks

            We now combine the learned channel-wise S-matrix elements into
            differential cross sections.  This is still a predictor-LROM-only
            check: the curves below compare FOM, the LS floor in the same
            central basis, and the predictor LROM.
            """
        ),
        code(
            r"""
            xs = ae_data["cross_sections"]
            xs_method_legend = pd.DataFrame(
                [
                    ("FOM", "full ROSE/KD solve for each A/E sample"),
                    ("LS floor", f"best central-basis cross-section reconstruction (n={cfg['N_PHI']}, l_max={xs['l_max']})"),
                    ("LROM", f"predictor RF-LROM (n={cfg['N_PHI']}, K={cfg['K_PREDICTORS']}, l_max={xs['l_max']})"),
                ],
                columns=["curve", "configuration"],
            )
            display(xs_method_legend)
            """
        ),
        code(
            r"""
            lrom_xs_l2 = np.asarray(xs["test_l2"])[1]
            order = np.argsort(lrom_xs_l2)
            candidate_indices = [int(order[-1]), int(order[int(0.90 * (len(order) - 1))]), int(order[len(order) // 2])]
            REPRESENTATIVE_XS_TEST_INDICES = []
            for idx in candidate_indices:
                if idx not in REPRESENTATIVE_XS_TEST_INDICES:
                    REPRESENTATIVE_XS_TEST_INDICES.append(idx)
            while len(REPRESENTATIVE_XS_TEST_INDICES) < 3:
                idx = int(order[-len(REPRESENTATIVE_XS_TEST_INDICES) - 1])
                if idx not in REPRESENTATIVE_XS_TEST_INDICES:
                    REPRESENTATIVE_XS_TEST_INDICES.append(idx)

            chosen = np.asarray(REPRESENTATIVE_XS_TEST_INDICES[:3], dtype=int)
            rows = []
            for idx in chosen:
                rows.append({
                    "test index": int(idx),
                    "A": int(test[idx, 0]),
                    "E_lab [MeV]": float(test[idx, 1]),
                    "LROM relative L2 XS error": float(xs["test_l2"][1, idx]),
                    "LS floor relative L2 XS error": float(xs["test_l2"][0, idx]),
                    "LROM median pointwise XS error": float(xs["test_median_pointwise"][1, idx]),
                })
            pd.DataFrame(rows)
            """
        ),
        code(
            r"""
            angles_deg = xs["angles_deg"]
            curves = [
                ("FOM", "xs_test_fom", "tab:blue", "-", 3.0),
                (f"LS floor (n={cfg['N_PHI']})", "xs_test_ls", "0.55", ":", 2.2),
                (f"LROM (n={cfg['N_PHI']}, K={cfg['K_PREDICTORS']})", "xs_test_lrom", "black", "--", 2.2),
            ]
            fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=190, sharey=True)
            for ax, idx in zip(axes, chosen):
                for label, key_name, color, style, lw in curves:
                    ax.plot(angles_deg, xs[key_name][idx], style, color=color, lw=lw, label=label)
                ax.set_yscale("log")
                ax.grid(True, which="both", alpha=0.25)
                ax.set_xlabel(r"$\theta_{\rm cm}$ [deg]")
                ax.set_title(rf"test {idx}: $A={int(test[idx, 0])}$, $E_{{lab}}={test[idx, 1]:.2f}$ MeV")
            axes[0].set_ylabel(r"$d\sigma/d\Omega$ [mb/sr]")
            axes[0].legend(fontsize=9)
            fig.suptitle(r"Representative KD calcium cross sections", y=1.03)
            fig.tight_layout()
            fig.savefig(OUT / "ae_representative_cross_sections.png", dpi=320, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        md(
            r"""
            ## 11. Cross-Section Error Distributions

            These violins summarize observable-level errors over the same
            training and wide-test A/E samples.
            """
        ),
        code(
            r"""
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            def split_violin_standard(ax, values, x0, side, color, width=0.72):
                values = np.asarray(values, dtype=float)
                values = values[np.isfinite(values) & (values > 0)]
                if values.size == 0:
                    return
                parts = ax.violinplot(
                    [values],
                    positions=[x0],
                    widths=width,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in parts["bodies"]:
                    verts = body.get_paths()[0].vertices
                    if side == "left":
                        verts[:, 0] = np.minimum(verts[:, 0], x0)
                    else:
                        verts[:, 0] = np.maximum(verts[:, 0], x0)
                    body.set_facecolor(color)
                    body.set_edgecolor("0.25")
                    body.set_alpha(0.68)

            method_labels = [f"LS floor\n(n={cfg['N_PHI']})", f"LROM\n(n={cfg['N_PHI']}, K={cfg['K_PREDICTORS']})"]
            metric_specs = [
                ("median_pointwise", "median pointwise relative XS error"),
                ("l2", "relative L2 XS error"),
            ]
            metric_arrays = {
                "median_pointwise": ("train_median_pointwise", "test_median_pointwise"),
                "l2": ("train_l2", "test_l2"),
            }
            fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), dpi=190)
            train_color = "#4C78A8"
            test_color = "#F58518"
            display_floor = 1e-5
            for ax, (metric, ylabel) in zip(axes, metric_specs):
                train_key, test_key = metric_arrays[metric]
                all_values = []
                for xpos, label in enumerate(method_labels, start=1):
                    train_vals = np.clip(xs[train_key][xpos - 1], display_floor, None)
                    test_vals = np.clip(xs[test_key][xpos - 1], display_floor, None)
                    all_values.extend([train_vals, test_vals])
                    split_violin_standard(ax, train_vals, xpos, "left", train_color)
                    split_violin_standard(ax, test_vals, xpos, "right", test_color)
                    ax.scatter(xpos - 0.18, np.median(train_vals), marker="D", s=34, color=train_color, edgecolor="black", linewidth=0.7, zorder=6)
                    ax.scatter(xpos + 0.18, np.median(test_vals), marker="D", s=34, color=test_color, edgecolor="black", linewidth=0.7, zorder=6)
                ax.set_yscale("log")
                ax.set_xticks(range(1, len(method_labels) + 1), method_labels)
                ax.set_ylabel(ylabel)
                ax.grid(True, which="both", alpha=0.25)
                top = max(float(np.quantile(v, 0.98)) for v in all_values)
                ax.set_ylim(display_floor, max(1.0, 1.5 * top))
            axes[1].legend(
                handles=[
                    Patch(facecolor=train_color, alpha=0.68, label="train"),
                    Patch(facecolor=test_color, alpha=0.68, label="wide test"),
                    Line2D([], [], marker="D", linestyle="None", color="white", markerfacecolor="0.75", markeredgecolor="black", label="median"),
                ],
                loc="upper right",
                fontsize=8,
            )
            fig.suptitle("KD calcium cross-section error distributions")
            fig.tight_layout()
            fig.savefig(OUT / "ae_cross_section_error_violins.png", dpi=320, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        md(
            r"""
            ## 11b. Predictor Count Sweep

            The predictor count controls how many potential-collocation
            features enter the RF-LROM.  For fixed `n=4` and `N_train=200`,
            the residual-fit system has

            `N_train * n`

            scalar equations per channel and

            `K * n^2 + (K+1) * n`

            unknowns per channel.  The table below keeps this bookkeeping next
            to the error distributions so we can see whether extra predictors
            are helping without quietly moving into an underdetermined fit.
            """
        ),
        code(
            r"""
            K_SWEEP_VALUES = [6, 9, 12, 15]
            K_SWEEP_FORCE_RECOMPUTE = False

            original_K = ae.K_PREDICTORS
            original_n_train = ae.N_TRAIN
            sweep_data = {}
            try:
                ae.N_TRAIN = 200
                for K_value in K_SWEEP_VALUES:
                    ae.K_PREDICTORS = K_value
                    cache = ROOT / "outputs" / "cached_ae" / f"wavefunction_predictor_lrom_a30_60_emin8_integerA_lmax15_k{K_value}_ntrain200.pkl"
                    t_start = time.perf_counter()
                    sweep_data[K_value] = ae.recompute(cache, force=K_SWEEP_FORCE_RECOMPUTE)
                    elapsed = time.perf_counter() - t_start
                    print(f"K={K_value}: loaded/recomputed in {elapsed:.2f} s; cache recompute seconds = {sweep_data[K_value]['recompute_seconds']:.2f}")
            finally:
                ae.K_PREDICTORS = original_K
                ae.N_TRAIN = original_n_train

            sweep_rows = []
            for K_value, data in sweep_data.items():
                c = data["config"]
                n = c["N_PHI"]
                equations = c["N_TRAIN"] * n
                unknowns = K_value * n * n + (K_value + 1) * n
                all_channels = 1 + 2 * c["L_MAX"]
                sweep_rows.append({
                    "K": K_value,
                    "N_train": c["N_TRAIN"],
                    "n": n,
                    "equations / channel": equations,
                    "unknowns / channel": unknowns,
                    "eqs / unknown": equations / unknowns,
                    "all lmax channels": all_channels,
                    "total equations": all_channels * equations,
                    "total unknowns": all_channels * unknowns,
                    "cache recompute seconds": data["recompute_seconds"],
                })
            k_sweep_summary = pd.DataFrame(sweep_rows)
            k_sweep_summary
            """
        ),
        code(
            r"""
            def split_violin_standard(ax, values, x0, side, color, width=0.72):
                values = np.asarray(values, dtype=float)
                values = values[np.isfinite(values) & (values > 0)]
                if values.size == 0:
                    return
                parts = ax.violinplot(
                    [values],
                    positions=[x0],
                    widths=width,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in parts["bodies"]:
                    verts = body.get_paths()[0].vertices
                    if side == "left":
                        verts[:, 0] = np.minimum(verts[:, 0], x0)
                    else:
                        verts[:, 0] = np.maximum(verts[:, 0], x0)
                    body.set_facecolor(color)
                    body.set_edgecolor("0.25")
                    body.set_alpha(0.68)

            def lrom_wf_clouds(data):
                train_vals = []
                test_vals = []
                for channel in data["channels"].values():
                    train_vals.append(channel["train"]["wf_err"])
                    test_vals.append(channel["test"]["wf_err"])
                return np.concatenate(train_vals), np.concatenate(test_vals)

            train_color = "#4C78A8"
            test_color = "#F58518"
            fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=190)
            metric_specs = [
                ("wavefunction", "LROM WF relative L2 error"),
                ("cross section", "LROM relative L2 XS error"),
            ]
            display_floor = 1e-5
            for ax, (metric, ylabel) in zip(axes, metric_specs):
                all_vals = []
                for xpos, K_value in enumerate(K_SWEEP_VALUES, start=1):
                    data = sweep_data[K_value]
                    if metric == "wavefunction":
                        train_vals, test_vals = lrom_wf_clouds(data)
                    else:
                        xs_k = data["cross_sections"]
                        train_vals = xs_k["train_l2"][1]
                        test_vals = xs_k["test_l2"][1]
                    train_vals = np.clip(train_vals, display_floor, None)
                    test_vals = np.clip(test_vals, display_floor, None)
                    all_vals.extend([train_vals, test_vals])
                    split_violin_standard(ax, train_vals, xpos, "left", train_color)
                    split_violin_standard(ax, test_vals, xpos, "right", test_color)
                    ax.scatter(xpos - 0.18, np.median(train_vals), marker="D", s=34, color=train_color, edgecolor="black", linewidth=0.7, zorder=6)
                    ax.scatter(xpos + 0.18, np.median(test_vals), marker="D", s=34, color=test_color, edgecolor="black", linewidth=0.7, zorder=6)
                ax.set_yscale("log")
                ax.set_xticks(range(1, len(K_SWEEP_VALUES) + 1), [f"K={K}" for K in K_SWEEP_VALUES])
                ax.set_ylabel(ylabel)
                ax.grid(True, which="both", alpha=0.25)
                top = max(float(np.quantile(v, 0.985)) for v in all_vals)
                ax.set_ylim(display_floor, max(0.2, 1.6 * top))
            axes[1].legend(
                handles=[
                    Patch(facecolor=train_color, alpha=0.68, label="train"),
                    Patch(facecolor=test_color, alpha=0.68, label="wide test"),
                    Line2D([], [], marker="D", linestyle="None", color="white", markerfacecolor="0.75", markeredgecolor="black", label="median"),
                ],
                loc="upper right",
                fontsize=9,
            )
            fig.suptitle("Predictor-count sweep for KD calcium A/E extrapolation")
            fig.tight_layout()
            fig.savefig(OUT / "ae_k_sweep_error_violins.png", dpi=320, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        code(
            r"""
            k_summary_rows = []
            for K_value, data in sweep_data.items():
                wf_train, wf_test = lrom_wf_clouds(data)
                xs_k = data["cross_sections"]
                k_summary_rows.append({
                    "K": K_value,
                    "median WF train": np.median(wf_train),
                    "median WF wide test": np.median(wf_test),
                    "90% WF wide test": np.quantile(wf_test, 0.90),
                    "median XS train": np.median(xs_k["train_l2"][1]),
                    "median XS wide test": np.median(xs_k["test_l2"][1]),
                    "90% XS wide test": np.quantile(xs_k["test_l2"][1], 0.90),
                })
            pd.DataFrame(k_summary_rows)
            """

        ),
        md(
            r"""
            ## 11c. Basis-Size Sweep At Fixed Predictor Count

            Now we hold the predictor count fixed at `K=9` and compare
            `n=4` against `n=6`.  This is a slightly different question than
            the LS floor alone: increasing `n` also increases the size of each
            learned matrix block, so the implicit reduced equation has more
            internal structure.

            The bookkeeping is important because the unknown count grows like
            `K n^2 + (K+1)n`, while the number of residual equations grows
            only like `N_train n`.
            """
        ),
        code(
            r"""
            N_SWEEP_VALUES = [4, 6]
            N_SWEEP_FIXED_K = 9
            N_SWEEP_FORCE_RECOMPUTE = False

            original_K = ae.K_PREDICTORS
            original_n_phi = ae.N_PHI
            original_n_train = ae.N_TRAIN
            n_sweep_data = {}
            try:
                ae.K_PREDICTORS = N_SWEEP_FIXED_K
                ae.N_TRAIN = 200
                for n_value in N_SWEEP_VALUES:
                    ae.N_PHI = n_value
                    cache = ROOT / "outputs" / "cached_ae" / f"wavefunction_predictor_lrom_a30_60_emin8_integerA_lmax15_k{N_SWEEP_FIXED_K}_n{n_value}_ntrain200.pkl"
                    t_start = time.perf_counter()
                    n_sweep_data[n_value] = ae.recompute(cache, force=N_SWEEP_FORCE_RECOMPUTE)
                    elapsed = time.perf_counter() - t_start
                    print(f"n={n_value}, K={N_SWEEP_FIXED_K}: loaded/recomputed in {elapsed:.2f} s; cache recompute seconds = {n_sweep_data[n_value]['recompute_seconds']:.2f}")
            finally:
                ae.K_PREDICTORS = original_K
                ae.N_PHI = original_n_phi
                ae.N_TRAIN = original_n_train

            n_sweep_rows = []
            for n_value, data in n_sweep_data.items():
                c = data["config"]
                equations = c["N_TRAIN"] * n_value
                unknowns = N_SWEEP_FIXED_K * n_value * n_value + (N_SWEEP_FIXED_K + 1) * n_value
                all_channels = 1 + 2 * c["L_MAX"]
                n_sweep_rows.append({
                    "n": n_value,
                    "K": N_SWEEP_FIXED_K,
                    "N_train": c["N_TRAIN"],
                    "equations / channel": equations,
                    "unknowns / channel": unknowns,
                    "eqs / unknown": equations / unknowns,
                    "all lmax channels": all_channels,
                    "total equations": all_channels * equations,
                    "total unknowns": all_channels * unknowns,
                    "cache recompute seconds": data["recompute_seconds"],
                })
            n_sweep_summary = pd.DataFrame(n_sweep_rows)
            n_sweep_summary
            """
        ),
        code(
            r"""
            def lrom_wf_clouds_from_data(data):
                train_vals = []
                test_vals = []
                for channel in data["channels"].values():
                    train_vals.append(channel["train"]["wf_err"])
                    test_vals.append(channel["test"]["wf_err"])
                return np.concatenate(train_vals), np.concatenate(test_vals)

            train_color = "#4C78A8"
            test_color = "#F58518"
            display_floor = 1e-5

            fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), dpi=190)
            metric_specs = [
                ("wavefunction", "LROM WF relative L2 error"),
                ("cross section", "LROM relative L2 XS error"),
            ]
            for ax, (metric, ylabel) in zip(axes, metric_specs):
                all_vals = []
                for xpos, n_value in enumerate(N_SWEEP_VALUES, start=1):
                    data = n_sweep_data[n_value]
                    if metric == "wavefunction":
                        train_vals, test_vals = lrom_wf_clouds_from_data(data)
                    else:
                        xs_n = data["cross_sections"]
                        train_vals = xs_n["train_l2"][1]
                        test_vals = xs_n["test_l2"][1]
                    train_vals = np.clip(train_vals, display_floor, None)
                    test_vals = np.clip(test_vals, display_floor, None)
                    all_vals.extend([train_vals, test_vals])
                    split_violin_standard(ax, train_vals, xpos, "left", train_color)
                    split_violin_standard(ax, test_vals, xpos, "right", test_color)
                    ax.scatter(xpos - 0.18, np.median(train_vals), marker="D", s=34, color=train_color, edgecolor="black", linewidth=0.7, zorder=6)
                    ax.scatter(xpos + 0.18, np.median(test_vals), marker="D", s=34, color=test_color, edgecolor="black", linewidth=0.7, zorder=6)
                ax.set_yscale("log")
                ax.set_xticks(range(1, len(N_SWEEP_VALUES) + 1), [f"n={n}, K={N_SWEEP_FIXED_K}" for n in N_SWEEP_VALUES])
                ax.set_ylabel(ylabel)
                ax.grid(True, which="both", alpha=0.25)
                top = max(float(np.quantile(v, 0.985)) for v in all_vals)
                ax.set_ylim(display_floor, max(0.2, 1.6 * top))
            axes[1].legend(
                handles=[
                    Patch(facecolor=train_color, alpha=0.68, label="train"),
                    Patch(facecolor=test_color, alpha=0.68, label="wide test"),
                    Line2D([], [], marker="D", linestyle="None", color="white", markerfacecolor="0.75", markeredgecolor="black", label="median"),
                ],
                loc="upper right",
                fontsize=9,
            )
            fig.suptitle("Basis-size sweep at fixed predictor count")
            fig.tight_layout()
            fig.savefig(OUT / "ae_n_sweep_error_violins.png", dpi=320, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        code(
            r"""
            n_summary_rows = []
            for n_value, data in n_sweep_data.items():
                wf_train, wf_test = lrom_wf_clouds_from_data(data)
                xs_n = data["cross_sections"]
                n_summary_rows.append({
                    "n": n_value,
                    "K": N_SWEEP_FIXED_K,
                    "median WF train": np.median(wf_train),
                    "median WF wide test": np.median(wf_test),
                    "90% WF wide test": np.quantile(wf_test, 0.90),
                    "median XS train": np.median(xs_n["train_l2"][1]),
                    "median XS wide test": np.median(xs_n["test_l2"][1]),
                    "90% XS wide test": np.quantile(xs_n["test_l2"][1], 0.90),
                })
            pd.DataFrame(n_summary_rows)
            """
        ),
        md(
            r"""
            ## 12. Cross-Section Error-Colored A/E Map

            This map uses the same geometry as the wavefunction maps, but the
            color now shows the predictor-LROM relative L2 error in the
            differential cross section.
            """
        ),
        code(
            r"""
            from matplotlib.colors import LogNorm
            from matplotlib.patches import Rectangle

            xs_train_err = np.asarray(xs["train_l2"])[1]
            xs_test_err = np.asarray(xs["test_l2"])[1]
            xs_all_err = np.concatenate([xs_train_err, xs_test_err])
            xs_finite = xs_all_err[np.isfinite(xs_all_err) & (xs_all_err > 0)]
            xs_vmin = max(1e-5, np.quantile(xs_finite, 0.03))
            xs_vmax = max(xs_vmin * 10, np.quantile(xs_finite, 0.995))
            xs_norm = LogNorm(vmin=xs_vmin, vmax=xs_vmax)

            fig, ax = plt.subplots(figsize=(7.2, 5.3), dpi=180)
            scatter = ax.scatter(
                test[:, 0], test[:, 1],
                c=xs_test_err,
                norm=xs_norm,
                cmap="magma",
                s=42,
                marker="o",
                edgecolor="0.25",
                linewidth=0.35,
                alpha=0.85,
                label="wide test",
            )
            ax.scatter(
                train[:, 0], train[:, 1],
                c=xs_train_err,
                norm=xs_norm,
                cmap="magma",
                s=40,
                marker="s",
                edgecolor="0.15",
                linewidth=0.35,
                alpha=0.92,
                label="train",
            )
            ax.scatter([central[0]], [central[1]], s=125, marker="*", color="white", edgecolor="black", linewidth=1.2, label="central", zorder=5)
            ax.add_patch(Rectangle(
                (cfg["TRAIN_A_RANGE"][0], cfg["TRAIN_E_RANGE"][0]),
                cfg["TRAIN_A_RANGE"][1] - cfg["TRAIN_A_RANGE"][0],
                cfg["TRAIN_E_RANGE"][1] - cfg["TRAIN_E_RANGE"][0],
                fill=False,
                lw=2.0,
                ec="0.20",
            ))
            ax.set_xlabel("A for calcium isotopes (Z=20)")
            ax.set_ylabel(r"$E_{\rm lab}$ [MeV]")
            ax.set_title("A/E samples colored by LROM cross-section error")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=9)
            cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label("LROM relative L2 XS error")
            fig.tight_layout()
            fig.savefig(OUT / "ae_cross_section_error_colored_map.png", dpi=300, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        md(
            r"""
            ## 13. Error-Colored A/E Maps

            These maps show the same train/test geometry as the sample-design
            plot, but now colored by predictor-LROM wavefunction error.  The
            color scale is logarithmic so isolated extrapolation failures are
            visible without hiding the well-behaved region.
            """
        ),
        code(
            r"""
            from matplotlib.colors import LogNorm
            from matplotlib.patches import Rectangle

            all_err = np.concatenate([
                channel["train"]["wf_err"]
                for channel in ae_data["channels"].values()
            ] + [
                channel["test"]["wf_err"]
                for channel in ae_data["channels"].values()
            ])
            finite_err = all_err[np.isfinite(all_err) & (all_err > 0)]
            vmin = max(1e-4, np.quantile(finite_err, 0.03))
            vmax = max(vmin * 10, np.quantile(finite_err, 0.995))
            norm = LogNorm(vmin=vmin, vmax=vmax)

            fig, axes = plt.subplots(1, len(ae_data["channels"]), figsize=(15.2, 4.7), dpi=170, sharex=True, sharey=True)
            fig.subplots_adjust(right=0.86, wspace=0.12)
            if len(ae_data["channels"]) == 1:
                axes = [axes]
            scatter_for_cbar = None
            for ax, (key, channel) in zip(axes, ae_data["channels"].items()):
                scatter_for_cbar = ax.scatter(
                    test[:, 0], test[:, 1],
                    c=channel["test"]["wf_err"],
                    norm=norm,
                    cmap="magma",
                    s=34,
                    marker="o",
                    edgecolor="0.25",
                    linewidth=0.35,
                    alpha=0.82,
                    label="wide test",
                )
                ax.scatter(
                    train[:, 0], train[:, 1],
                    c=channel["train"]["wf_err"],
                    norm=norm,
                    cmap="magma",
                    s=34,
                    marker="s",
                    edgecolor="0.15",
                    linewidth=0.35,
                    alpha=0.92,
                    label="train",
                )
                ax.scatter([central[0]], [central[1]], s=110, marker="*", color="white", edgecolor="black", linewidth=1.2, label="central", zorder=5)
                ax.add_patch(Rectangle(
                    (cfg["TRAIN_A_RANGE"][0], cfg["TRAIN_E_RANGE"][0]),
                    cfg["TRAIN_A_RANGE"][1] - cfg["TRAIN_A_RANGE"][0],
                    cfg["TRAIN_E_RANGE"][1] - cfg["TRAIN_E_RANGE"][0],
                    fill=False,
                    lw=2.0,
                    ec="0.20",
                ))
                ax.set_title(f"l={key[0]}, ch={key[1]}")
                ax.set_xlabel("A for calcium isotopes (Z=20)")
                ax.grid(True, alpha=0.25)
            axes[0].set_ylabel(r"$E_{\rm lab}$ [MeV]")
            axes[-1].legend(loc="upper right", fontsize=8)
            cax = fig.add_axes([0.885, 0.18, 0.018, 0.64])
            cbar = fig.colorbar(scatter_for_cbar, cax=cax)
            cbar.set_label("LROM WF relative L2 error", labelpad=10)
            fig.suptitle("A/E samples colored by predictor-LROM wavefunction error", y=0.99)
            fig.savefig(OUT / "ae_error_colored_sample_maps.png", dpi=300, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        md(
            r"""
            ## Notes

            This notebook is a stress test, not the final global model.  If the
            predictor LROM tracks the LS floor inside the training rectangle and
            then peels away at low energy or isotope endpoints, that tells us
            the reduced basis and predictor map need to become more local or
            more explicitly energy-aware.
            """
        ),
    ]
