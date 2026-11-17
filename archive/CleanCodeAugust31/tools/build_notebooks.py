"""Generate the three small, reviewable notebooks shipped with this repository."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def write(name, title, cells):
    notebook = nbf.v4.new_notebook(
        cells=[md(f"# {title}")] + cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    nbf.write(notebook, OUT / name)


COMMON = r'''
import os, sys
from pathlib import Path
ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lrom import BasisConfig, InputSpace, MaxVol, ScatteringLROM, ScatteringProblem, get_runtime
from lrom.diagnostics import relative_l2

plt.style.use("seaborn-v0_8-whitegrid")
FAST = os.getenv("LROM_FULL_RUN", "0") != "1"
BACKEND = os.getenv("LROM_BACKEND", "cpu")       # cpu, gpu, or auto
PRECISION = os.getenv("LROM_PRECISION", "double") # double or single (GPU)
runtime = get_runtime(BACKEND, PRECISION)
print(f"mode={'quick' if FAST else 'full'}, runtime={runtime.name}, device={runtime.device}")
'''


write(
    "01_single_wavefunction.ipynb",
    "A single scattering wavefunction",
    [
        md(r'''
        This notebook introduces the complete workflow on one $s$ wave:

        1. generate independent FOM snapshots with the regular-origin shooting convention;
        2. learn a centered reduced basis and implicit coordinate equation;
        3. compare least-squares projection (the basis limit) with the online LROM.

        `LROM_FULL_RUN=1` increases the mesh and sample counts. The accelerator
        setting is shared with the later notebooks, although this small scalar
        problem is normally faster on the CPU.
        '''),
        code(COMMON),
        code(r'''
        problem = ScatteringProblem(
            potential="real_woods_saxon", target_a=40, target_z=20,
            lab_energy=14.1, l_max=0, mesh_points=400 if FAST else 800,
            target_binding_energy=342.052184,
        )
        center = problem.reference_omega

        def sample(vv):
            row = problem.reference_sample.copy()
            row["Vv"] = float(vv)
            return row

        n_train = 21 if FAST else 41
        train_vv = np.linspace(.90 * center[0], 1.10 * center[0], n_train)
        test_vv = np.linspace(.82 * center[0], 1.18 * center[0], 51)
        train_samples = [sample(value) for value in train_vv]
        test_samples = [sample(value) for value in test_vv]
        training = problem.generate_training_data(train_samples)
        testing = problem.generate_training_data(test_samples)
        '''),
        code(r'''
        emulator = ScatteringLROM(
            problem, BasisConfig(size=4), MaxVol(count=1, ridge=1e-14)
        ).train(training)
        channel = (0, 0.0)
        learned = emulator.channel_models[channel].learned

        exact = testing.wavefunctions[channel]
        predicted = np.asarray([
            emulator.predict(row).wavefunction(channel) for row in test_samples
        ])
        projected_coordinates = learned.basis.project(exact)
        projected = learned.basis.reconstruct(projected_coordinates)

        table = pd.DataFrame({
            "method": ["basis projection", "online LROM"],
            "median relative L2": [
                np.median(relative_l2(projected, exact)),
                np.median(relative_l2(predicted, exact)),
            ],
            "maximum relative L2": [
                np.max(relative_l2(projected, exact)),
                np.max(relative_l2(predicted, exact)),
            ],
        })
        table
        '''),
        code(r'''
        radius = problem.system().radius
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for index in np.linspace(0, len(test_vv)-1, 7, dtype=int):
            color = plt.cm.viridis(index / (len(test_vv)-1))
            axes[0].plot(radius, exact[index].real, color=color, lw=1.7)
            axes[0].plot(radius, predicted[index].real, color=color, ls="--", lw=1)
        axes[0].set(xlabel="r [fm]", ylabel=r"Re $\phi$", title="FOM (solid), LROM (dashed)")
        axes[1].semilogy(test_vv, relative_l2(projected, exact), label="basis projection")
        axes[1].semilogy(test_vv, relative_l2(predicted, exact), label="online LROM")
        axes[1].axvspan(train_vv.min(), train_vv.max(), alpha=.12, color="black", label="training interval")
        axes[1].set(xlabel=r"$V_v$ [MeV]", ylabel="relative L2 error", title="Interpolation and mild extrapolation")
        axes[1].legend()
        plt.show()
        '''),
        md(r'''
        The online error contains both basis truncation and learned-coordinate
        error. The ridge value is dimensionless: $\alpha=10^{-14}\sigma_{\max}(D)^2$.
        Set it to zero in `MaxVol` for ordinary least squares.
        '''),
    ],
)


write(
    "02_elastic_cross_sections.ipynb",
    "Elastic cross sections from partial-wave LROMs",
    [
        md(r'''
        We now vary all ten local Woods--Saxon parameters, train one reduced
        model per partial-wave channel, and assemble the observable. OLS and the
        weak ridge retained by the research study use identical snapshots,
        bases, and predictors.
        '''),
        code(COMMON + r'''
from lrom.physics import FULL_PARAMETER_NAMES
from lrom.diagnostics import pointwise_relative_error
'''),
        code(r'''
        l_max = 6 if FAST else 10
        n_train, n_test = ((100, 20) if FAST else (500, 100))
        problem = ScatteringProblem(
            potential="full_woods_saxon", target_a=40, target_z=20,
            lab_energy=14.1, l_max=l_max, mesh_points=400 if FAST else 800,
            target_binding_energy=342.052184,
        )
        center = problem.reference_omega
        space = InputSpace(
            continuous={name: (.82*v, 1.18*v) for name, v in zip(FULL_PARAMETER_NAMES, center)},
            fixed={"E": 14.1, "A": 40, "Z": 20},
        )
        train_samples = space.sample(n_train, seed=1204)
        test_samples = space.sample(n_test, seed=1205)
        training = problem.generate_training_data(train_samples)
        testing = problem.generate_training_data(test_samples)
        angles = np.arange(2.0, 179.0, 2.0)
        reference = problem.cross_sections_from_data(testing, angles)
        '''),
        code(r'''
        methods = {}
        for name, ridge in {"OLS": 0.0, "ridge-14": 1e-14}.items():
            model = ScatteringLROM(
                problem, BasisConfig(size=5 if FAST else 6),
                MaxVol(count=6 if FAST else 12, ridge=ridge),
                packing_strategy="padded",
            ).train(training)
            prediction = model.cross_sections(
                test_samples, angles, linear_solver=runtime.solve
            )
            error = pointwise_relative_error(prediction, reference, denominator_floor=1e-10)
            methods[name] = {"model": model, "prediction": prediction, "error": error}

        pd.DataFrame([
            {
                "method": name,
                "median case error": np.median(value["error"], axis=1).mean(),
                "95th-percentile case error": np.quantile(np.median(value["error"], axis=1), .95),
                "worst matrix condition": max(
                    value["model"].predict(row).condition_number(channel)
                    for row in test_samples
                    for channel in value["model"].problem.channels
                ),
            }
            for name, value in methods.items()
        ])
        '''),
        code(r'''
        ranking = np.argsort(np.median(methods["ridge-14"]["error"], axis=1))
        selected = ranking[[0, len(ranking)//2, -1]]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        for axis, index in zip(axes, selected):
            axis.semilogy(angles, reference[index], color="black", lw=2.2, label="FOM")
            for name, style in (("OLS", "--"), ("ridge-14", "-")):
                axis.semilogy(angles, methods[name]["prediction"][index], style, lw=1.5, label=name)
            axis.set(xlabel=r"$\theta_{cm}$ [deg]", ylabel=r"$d\sigma/d\Omega$ [mb/sr]",
                     title=f"held-out case {index}")
        axes[0].legend()
        plt.show()
        '''),
        md(r'''
        GPU mode changes only the batched reduced solves; training and the FOM
        remain CPU operations. This is intentional: one fitted equation and one
        observable implementation are tested regardless of deployment device.
        '''),
    ],
)


write(
    "03_real_data_pilot.ipynb",
    "A cautious pilot with curated real scattering data",
    [
        md(r'''
        This is a deliberately small first contact with the curated corpus—not
        the final global calibration. It audits provenance, selects three
        $^{40}$Ca curves, profiles each experiment's correlated normalization,
        optimizes two global KD coefficients with the FOM, and trains a local
        ridge LROM around that point. The narrow problem makes every assumption
        inspectable before scaling to all 37 coefficients and all isotopes.
        '''),
        code(COMMON + r'''
from scipy.optimize import least_squares
from lrom.curated_data import load_neutron_elastic, strict_lrom_selection
from lrom.global_kd import COEFFICIENT_NAMES, kd_parameters_from_coefficients
from lrom.physics import KD_PARAMETER_NAMES
from lrom.reduced import latin_hypercube
'''),
        code(r'''
        candidates = [
            os.getenv("SCATTERING_DATA_ROOT", ""),
            ROOT / "external" / "nucleon-nucleus-data",
        ]
        data_root = next((Path(p) for p in candidates if p and (Path(p) / "data").is_dir()), None)
        if data_root is None:
            raise FileNotFoundError(
                "Run `python tools/fetch_data.py` from the repository root, "
                "or set SCATTERING_DATA_ROOT as described in data/README.md"
            )

        all_points, records = load_neutron_elastic(data_root)
        points = strict_lrom_selection(all_points)
        audit = pd.DataFrame({
            "records": points.groupby("corpus").record_id.nunique(),
            "points": points.groupby("corpus").size(),
            "isotopes": points.groupby("corpus").target.nunique(),
        })
        audit
        '''),
        code(r'''
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        for corpus, group in points.drop_duplicates("record_id").groupby("corpus"):
            ax.scatter(group.A, group.energy_MeV, s=22, alpha=.65, label=corpus)
        ax.set(xlabel="mass number A", ylabel=r"$E_{lab}$ [MeV]", title="Curated strict-elastic coverage")
        ax.legend()
        plt.show()
        '''),
        code(r'''
        calcium = points[(points.corpus == "kduq") & (points.target == "Ca-40")]
        record_order = (
            calcium[["record_id", "energy_MeV"]].drop_duplicates()
            .sort_values("energy_MeV").reset_index(drop=True)
        )
        take = np.unique(np.linspace(0, len(record_order)-1, 3, dtype=int))
        chosen_ids = record_order.iloc[take].record_id
        pilot = calcium[calcium.record_id.isin(chosen_ids)].copy()
        pilot.groupby("record_id").agg(energy=("energy_MeV", "first"), points=("angle_cm_deg", "size"))
        '''),
        code(r'''
        active = np.asarray([2, 18])  # V1 constant and D1 constant
        l_max = 10 if FAST else 16
        mesh_points = 400 if FAST else 800
        problem = ScatteringProblem(
            potential="kd_neutron", target_a=40, target_z=20,
            lab_energy=float(pilot.energy_MeV.median()), l_max=l_max,
            mesh_points=mesh_points, target_binding_energy=None,
        )

        def sample_for(a, z, energy, active_values):
            shifts = np.zeros(37); shifts[active] = active_values
            omega = kd_parameters_from_coefficients(a, z, energy, shifts)
            row = dict(zip(KD_PARAMETER_NAMES, np.asarray(omega, float)))
            row.update(A=int(a), Z=int(z), E=float(energy))
            return row

        grouped = [group for _, group in pilot.groupby("record_id", sort=False)]

        def residuals(active_values, predictor=None):
            residual = []
            for group in grouped:
                row = group.iloc[0]
                sample = sample_for(row.A, row.Z, row.energy_MeV, active_values)
                angles = group.angle_cm_deg.to_numpy(float)
                theory = (problem.cross_section(sample, angles) if predictor is None
                          else predictor.cross_section(sample, angles))
                observed = group.cross_section_mb_sr.to_numpy(float)
                reported = group.point_error_mb_sr.to_numpy(float)
                sigma = np.sqrt(reported**2 + (.02*observed)**2 + (.001*observed.max())**2)
                tau = float(group.normalization_fraction.iloc[0])
                weight = 1.0 / sigma**2
                delta = np.sum(theory*(observed-theory)*weight) / (1/tau**2 + np.sum(theory**2*weight))
                residual.extend((observed-(1+delta)*theory)/sigma)
                residual.append(delta/tau)
            return np.asarray(residual)

        fit = least_squares(residuals, np.zeros(len(active)), bounds=(-.08, .08), max_nfev=30 if FAST else 80)
        pd.DataFrame({
            "coefficient": [COEFFICIENT_NAMES[i] for i in active],
            "log shift": fit.x,
            "multiplicative change": np.exp(fit.x),
        })
        '''),
        code(r'''
        # Train a small patch around the FOM optimum. The training interval is
        # slightly wider than the selected experimental energies so evaluation
        # at the boundary is interpolation, not accidental extrapolation.
        e_lo, e_hi = pilot.energy_MeV.min()-1.0, pilot.energy_MeV.max()+1.0
        n_train = 120 if FAST else 240
        lower = np.r_[e_lo, fit.x-.035]
        upper = np.r_[e_hi, fit.x+.035]
        design = latin_hypercube(lower, upper, n_train, seed=1204)
        train_samples = [sample_for(40, 20, row[0], row[1:]) for row in design]
        reference = sample_for(40, 20, float(pilot.energy_MeV.median()), fit.x)
        local_problem = ScatteringProblem(
            potential="kd_neutron", target_a=40, target_z=20,
            lab_energy=reference["E"], l_max=l_max, mesh_points=mesh_points,
            target_binding_energy=None,
            reference_omega=np.asarray([reference[name] for name in KD_PARAMETER_NAMES]),
        )
        training = local_problem.generate_training_data(train_samples)
        local_lrom = ScatteringLROM(
            local_problem, BasisConfig(size=6 if FAST else 9),
            MaxVol(count=8 if FAST else 12, ridge=1e-14),
            packing_strategy="padded",
        ).train(training)
        print(f"local LROM trained on {n_train} snapshots across {len(local_problem.channels)} channels")
        '''),
        code(r'''
        fig, axes = plt.subplots(1, len(grouped), figsize=(14, 4), layout="constrained")
        for axis, group in zip(np.atleast_1d(axes), grouped):
            row = group.iloc[0]
            angles = group.angle_cm_deg.to_numpy(float)
            baseline = sample_for(row.A, row.Z, row.energy_MeV, np.zeros(len(active)))
            at_map = sample_for(row.A, row.Z, row.energy_MeV, fit.x)
            axis.errorbar(angles, group.cross_section_mb_sr, yerr=group.point_error_mb_sr,
                          fmt="o", ms=3, color="black", label="curated data")
            axis.semilogy(angles, problem.cross_section(baseline, angles), label="published KD")
            axis.semilogy(angles, problem.cross_section(at_map, angles), label="pilot FOM fit")
            axis.semilogy(angles, local_lrom.cross_section(at_map, angles), "--", label="local ridge LROM")
            axis.set(xlabel=r"$\theta_{cm}$ [deg]", ylabel=r"$d\sigma/d\Omega$ [mb/sr]",
                     title=f"40Ca, {row.energy_MeV:g} MeV")
        axes[0].legend(fontsize=8)
        plt.show()
        '''),
        md(r'''
        **Interpretation.** This pilot is useful only if the local LROM follows
        the FOM at the optimized point and across a held-out neighborhood. It
        is not evidence that two coefficients identify the global potential.
        The production study should restore the full isotope/energy corpus,
        independent held-out records, hard-routed energy patches, and posterior
        predictive checks before sampling all KD coefficients.
        '''),
    ],
)

print(f"Wrote notebooks to {OUT}")
