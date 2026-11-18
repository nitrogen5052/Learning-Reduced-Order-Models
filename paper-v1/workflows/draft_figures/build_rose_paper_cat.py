"""Recompute and cache the clean ROSE/LROM benchmark, then draw a compact CAT plot."""
from __future__ import annotations

import sys, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT
sys.path.insert(0, str(CLEAN))
from lrom import BasisConfig, InputSpace, MaxVol, ScatteringLROM, ScatteringProblem
from lrom.diagnostics import pointwise_relative_error
from lrom.physics import FULL_PARAMETER_NAMES

SEED = 1204
N_TRAIN, N_TEST = 500, 100
CONFIGS = ((4, 8), (6, 12), (6, 16), (8, 16), (12, 16))
ANGLES = np.arange(1.0, 180.0)
FLOOR = 1e-12
DATA = HERE / "rose_lrom_paper_cat_metrics.npz"
OUT = ROOT / "figures/draft_paper"
ROSE = CLEAN / "benchmarks/cache/rose_notebook02_c15714dd84e007f6.npz"

def timed(evaluator, samples):
    evaluator(samples[0]); result = np.empty(len(samples))
    for i, sample in enumerate(samples):
        trials = []
        for _ in range(3):
            tic = time.perf_counter_ns()
            for _ in range(20): evaluator(sample)
            trials.append((time.perf_counter_ns() - tic) / 20e9)
        result[i] = min(trials)
    return result

def build():
    problem = ScatteringProblem(
        potential="full_woods_saxon", target_a=40, target_z=20,
        lab_energy=14.1, l_max=10, mesh_points=800,
        target_binding_energy=342.052184,
    )
    omega = problem.reference_omega
    space = InputSpace(
        continuous={n: (.8*v, 1.2*v) for n, v in zip(FULL_PARAMETER_NAMES, omega)},
        fixed={"E": 14.1, "A": 40, "Z": 20},
    )
    train, test = space.sample(N_TRAIN, SEED), space.sample(N_TEST, SEED + 1)
    print("Generating the independent FOM database ...", flush=True)
    train_data = problem.generate_training_data(train)
    fom = np.asarray([problem.cross_section(s, ANGLES) for s in test])
    values = {}
    for nphi, nop in CONFIGS:
        print(f"Training LROM ({nphi}, {nop}) ...", flush=True)
        model = ScatteringLROM(
            problem, BasisConfig(size=nphi),
            MaxVol(count=nop, minimum_s=.1),
        ).train(train_data)
        pred = np.asarray([model.cross_section(s, ANGLES) for s in test])
        tag = f"{nphi}_{nop}"
        values[f"lrom_error_{tag}"] = pointwise_relative_error(
            pred, fom, denominator_floor=FLOOR).max(axis=1)
        values[f"lrom_seconds_{tag}"] = timed(
            lambda s: model.cross_section(s, ANGLES), test)
    values["lrom_fom_seconds"] = timed(
        lambda s: problem.cross_section(s, ANGLES), test)
    rose = np.load(ROSE)
    for nphi, nop in CONFIGS:
        tag = f"{nphi}_{nop}"
        values[f"rose_error_{tag}"] = pointwise_relative_error(
            rose[f"test_xs_{tag}"], rose["fom_test_xs"],
            denominator_floor=FLOOR).max(axis=1)
        values[f"rose_seconds_{tag}"] = rose[f"online_seconds_{tag}"]
    values["rose_fom_seconds"] = rose["fom_seconds"]
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA, **values)
    return values

def plot(v):
    selected = ((4, 8), (6, 12), (8, 16), (12, 16))
    markers = ("o", "^", "D", "P")
    lrom_colors = plt.cm.Blues(np.linspace(.48, .88, len(selected)))
    rose_colors = plt.cm.OrRd(np.linspace(.46, .86, len(selected)))
    fig, axes = plt.subplots(
        1, 3, figsize=(10.3, 5.9), sharey=True,
        gridspec_kw={"width_ratios": [6.2, 1.0, 1.0], "wspace": .06},
    )
    ax, fom_axis, rose_fom_axis = axes
    for marker, lrom_color, rose_color, (nphi, nop) in zip(
            markers, lrom_colors, rose_colors, selected):
        tag = f"{nphi}_{nop}"
        ax.scatter(v[f"lrom_seconds_{tag}"], np.maximum(v[f"lrom_error_{tag}"], FLOOR),
                   s=31, marker=marker, color=lrom_color, alpha=.55, linewidths=0)
        ax.scatter(v[f"rose_seconds_{tag}"], np.maximum(v[f"rose_error_{tag}"], FLOOR),
                   s=35, marker=marker, facecolors="none", edgecolors=rose_color,
                   alpha=.62, linewidths=1.05)
    ax.axhline(.1, color=".25", ls="--", lw=1.4)
    ax.text(.985, .1, "10% error", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=10)
    lrom_fom = float(np.median(v["lrom_fom_seconds"]))
    rose_fom = float(np.median(v["rose_fom_seconds"]))
    fom_axis.axvline(lrom_fom, color=lrom_colors[-1], ls=":", lw=2)
    rose_fom_axis.axvline(rose_fom, color=rose_colors[-1], ls=":", lw=2)
    fom_axis.text(lrom_fom, 1.2e-3, "Numerov\nFOM", rotation=90,
                  color=lrom_colors[-1], ha="center", va="bottom", fontsize=8)
    rose_fom_axis.text(rose_fom, 1.2e-3, "ROSE\nFOM", rotation=90,
                       color=rose_colors[-1], ha="center", va="bottom", fontsize=8)
    rom_times = np.concatenate([
        v[f"lrom_seconds_{a}_{b}"] for a, b in selected
    ] + [v[f"rose_seconds_{a}_{b}"] for a, b in selected])
    ax.set_xlim(.82*rom_times.min(), 1.22*rom_times.max())
    fom_axis.set_xlim(.75*lrom_fom, 1.25*lrom_fom)
    rose_fom_axis.set_xlim(.75*rose_fom, 1.25*rose_fom)
    for axis in axes:
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.grid(True, which="major", alpha=.35)
    fom_axis.set_xticks([lrom_fom], [rf"${lrom_fom:.1e}$"])
    rose_fom_axis.set_xticks([rose_fom], [rf"${rose_fom:.1e}$"])
    fom_axis.minorticks_off(); rose_fom_axis.minorticks_off()
    ax.set_ylabel("maximum pointwise relative cross-section error")
    fig.supxlabel("online time per cross section [s]", y=.035)
    fig.suptitle(r"Computational accuracy versus time: $^{40}$Ca(n,n), 14.1 MeV", y=.98)
    for left, right in zip(axes[:-1], axes[1:]):
        left.spines["right"].set_visible(False); right.spines["left"].set_visible(False)
        right.tick_params(left=False, labelleft=False)
        for y in (-.012, .988):
            left.plot((.985, 1.015), (y, y+.024), transform=left.transAxes,
                      color="black", clip_on=False, lw=1)
            right.plot((-.015, .015), (y, y+.024), transform=right.transAxes,
                       color="black", clip_on=False, lw=1)
    methods = [
        Line2D([], [], marker="o", ls="none", markersize=7,
               markerfacecolor=lrom_colors[-2], markeredgewidth=0, label="LROM"),
        Line2D([], [], marker="o", ls="none", markersize=7,
               markerfacecolor="none", markeredgecolor=rose_colors[-2], label="ROSE"),
    ]
    first = ax.legend(handles=methods, title="method", loc="upper right",
                      borderpad=.6, labelspacing=.35)
    ax.add_artist(first)
    configs = [Line2D([], [], marker=m, ls="none", color=".35", markersize=6,
                      label=rf"$({a},{b})$") for m, (a, b) in zip(markers, selected)]
    ax.legend(handles=configs, title=r"$(n_\phi,n_{\rm op})$", loc="lower left",
              ncol=2, borderpad=.6, columnspacing=.9, handletextpad=.4, labelspacing=.3)
    fig.subplots_adjust(left=.09, right=.98, bottom=.13, top=.90)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rose_lrom_cat.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

def plot_one_column(v):
    """Paper-column CAT plot with the remote FOM timings in a lower strip."""
    selected = ((4, 8), (6, 12), (8, 16), (12, 16))
    markers = ("o", "^", "D", "P")
    lrom_colors = plt.cm.Blues(np.linspace(.48, .88, len(selected)))
    rose_colors = plt.cm.OrRd(np.linspace(.46, .86, len(selected)))
    fig, (ax, reference_axis) = plt.subplots(
        2, 1, figsize=(3.45, 4.25),
        gridspec_kw={"height_ratios": [5.0, 1.15], "hspace": .34},
    )

    for marker, lrom_color, rose_color, (nphi, nop) in zip(
            markers, lrom_colors, rose_colors, selected):
        tag = f"{nphi}_{nop}"
        ax.scatter(
            v[f"lrom_seconds_{tag}"],
            np.maximum(v[f"lrom_error_{tag}"], FLOOR),
            s=15, marker=marker, color=lrom_color,
            alpha=.58, linewidths=0,
        )
        ax.scatter(
            v[f"rose_seconds_{tag}"],
            np.maximum(v[f"rose_error_{tag}"], FLOOR),
            s=18, marker=marker, facecolors="none",
            edgecolors=rose_color, alpha=.66, linewidths=.75,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    rom_times = np.concatenate([
        v[f"lrom_seconds_{a}_{b}"] for a, b in selected
    ] + [v[f"rose_seconds_{a}_{b}"] for a, b in selected])
    ax.set_xlim(.80*rom_times.min(), 1.28*rom_times.max())
    ax.axhline(.1, color=".25", ls="--", lw=1.0)
    ax.text(.98, .1, "10%", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7.5)
    ax.set_ylabel("maximum relative error", fontsize=9)
    ax.set_xlabel("online time [s]", fontsize=9)
    ax.set_xticks([6e-5, 2e-4, 6e-4])
    ax.set_xticklabels([r"$6\!\times\!10^{-5}$", r"$2\!\times\!10^{-4}$", r"$6\!\times\!10^{-4}$"])
    ax.minorticks_off()
    ax.tick_params(axis="both", which="both", labelsize=7.5)
    ax.grid(False)

    methods = [
        Line2D([], [], marker="o", ls="none", markersize=5,
               markerfacecolor=lrom_colors[-2], markeredgewidth=0, label="LROM"),
        Line2D([], [], marker="o", ls="none", markersize=5,
               markerfacecolor="none", markeredgecolor=rose_colors[-2], label="ROSE"),
    ]
    method_legend = ax.legend(
        handles=methods, loc="upper right", frameon=True,
        borderpad=.35, handletextpad=.35, labelspacing=.2, fontsize=7.5,
    )
    ax.add_artist(method_legend)
    configs = [
        Line2D([], [], marker=m, ls="none", color=".35", markersize=4.5,
               label=rf"$({a},{b})$")
        for m, (a, b) in zip(markers, selected)
    ]
    ax.legend(
        handles=configs, title=r"$(n_\phi,n_{\rm op})$",
        loc="lower left", ncol=2, frameon=True,
        borderpad=.35, columnspacing=.6, handletextpad=.25,
        labelspacing=.15, fontsize=7, title_fontsize=7.5,
    )

    lrom_fom = float(np.median(v["lrom_fom_seconds"]))
    rose_fom = float(np.median(v["rose_fom_seconds"]))
    reference_axis.set_xscale("log")
    reference_axis.set_xlim(.006, .8)
    reference_axis.set_ylim(0, 1)
    reference_axis.axvline(lrom_fom, color=lrom_colors[-1], lw=1.6)
    reference_axis.axvline(rose_fom, color=rose_colors[-1], lw=1.6)
    reference_axis.text(
        lrom_fom*1.08, .52, "Numerov\nFOM",
        ha="left", va="center", color=lrom_colors[-1], fontsize=7,
    )
    reference_axis.text(
        rose_fom/1.08, .52, "ROSE\nFOM",
        ha="right", va="center", color=rose_colors[-1], fontsize=7,
    )
    reference_axis.set_xlabel("full-order time [s]", fontsize=9)
    reference_axis.set_yticks([])
    reference_axis.tick_params(axis="x", which="both", labelsize=7.5)
    reference_axis.spines["left"].set_visible(False)
    reference_axis.spines["right"].set_visible(False)
    reference_axis.grid(False)

    fig.subplots_adjust(left=.19, right=.98, bottom=.12, top=.98)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT / f"rose_lrom_cat_one_column.{ext}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)

if __name__ == "__main__":
    if DATA.exists():
        with np.load(DATA) as f: metrics = {k: f[k] for k in f.files}
    else: metrics = build()
    plot(metrics)
    plot_one_column(metrics)
    print(OUT / "rose_lrom_cat_one_column.png")
