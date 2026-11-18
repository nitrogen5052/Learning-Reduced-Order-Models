"""Generate candidate layouts for the introductory LROM paper figures.

This is deliberately a paper-design companion to notebook 01.  All FOM,
basis, predictor-selection, and LROM operations use the maintained ``lrom``
package in this repository.  The only local calculation is the conventional Galerkin projection
of the same scaled radial equation onto the learned wavefunction basis.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_ROOT = ROOT
sys.path.insert(0, str(SOURCE_ROOT))
sys.dont_write_bytecode = True
_MPL_CONFIG = tempfile.TemporaryDirectory(prefix="lrom-paper-variants-")
os.environ["MPLCONFIGDIR"] = _MPL_CONFIG.name

import lrom
from lrom import BasisConfig, MaxVol, ScatteringLROM, ScatteringProblem
from lrom.physics import real_woods_saxon
from lrom.reduced import latin_hypercube


FIGURES = ROOT / "figures" / "design_variants"
FIGURES.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Match the SMLR plotting style: retain Inferno's perceptual ordering while
# avoiding the nearly black and nearly white endpoints.
_INFERNO = plt.get_cmap("inferno")
CMAP = ListedColormap(_INFERNO(np.linspace(0.10, 0.88, 256)), name="inferno_trimmed")
SEED = 1204
BASIS_SIZE = 4
PREDICTOR_COUNT = 2
ERROR_FLOOR = 1.0e-11


def samples_from_rows(rows):
    return [
        {
            "Vv": float(row[0]), "Rv": float(row[1]), "av": float(row[2]),
            "E": 14.1, "A": 40, "Z": 20,
        }
        for row in np.asarray(rows)
    ]


def save(fig, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


@dataclass
class Study:
    problem: object
    system: object
    omega0: np.ndarray
    train_lower: np.ndarray
    train_upper: np.ndarray
    emulator: object
    model: object
    train_rows: np.ndarray


def build_study() -> Study:
    problem = ScatteringProblem(
        potential="real_woods_saxon",
        target_a=40,
        target_z=20,
        lab_energy=14.1,
        l_max=0,
        mesh_points=800,
        target_binding_energy=342.052184,
    )
    system = problem.system()
    omega0 = np.array([46.7238, 4.0538, 0.6718])
    train_lower = omega0 * np.array([0.88, 0.92, 1.00])
    train_upper = omega0 * np.array([1.12, 1.08, 1.00])
    train_rows = latin_hypercube(train_lower, train_upper, 96, SEED)
    train_rows[:, 2] = omega0[2]
    train_rows = np.vstack([omega0, train_rows])
    data = problem.generate_training_data(samples_from_rows(train_rows))
    emulator = ScatteringLROM(
        problem,
        BasisConfig(size=BASIS_SIZE),
        MaxVol(count=PREDICTOR_COUNT, ridge=1.0e-14),
    ).train(data)
    model = emulator.channel_models[(0, 0.0)].learned
    return Study(
        problem, system, omega0, train_lower, train_upper,
        emulator, model, train_rows,
    )


def evaluate(study: Study, rows):
    rows = np.asarray(rows)
    data = study.problem.generate_training_data(samples_from_rows(rows))
    waves = data.wavefunctions[(0, 0.0)]
    ls = study.model.basis.project(waves)
    predictions = [study.emulator.predict(s) for s in samples_from_rows(rows)]
    lrom = np.asarray([p.coordinates((0, 0.0)) for p in predictions])
    return waves, ls, lrom


def second_derivative(values, spacing):
    result = np.empty_like(values)
    result[1:-1] = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / spacing**2
    result[0] = result[1]
    result[-1] = result[-2]
    return result


def galerkin_coordinates(study: Study, rows):
    """Project the scaled radial operator onto the centered LROM basis."""
    rows = np.asarray(rows)
    basis = study.model.basis
    modes = basis.vectors
    phi0 = basis.reference
    s = study.system.s_mesh
    radius = study.system.radius
    ds = s[1] - s[0]
    interior = slice(2, -2)
    d2_modes = np.column_stack([
        second_derivative(modes[:, j], ds) for j in range(modes.shape[1])
    ])
    d2_phi0 = second_derivative(phi0, ds)
    left = np.conjugate(modes[interior])
    free_action = d2_modes[interior] + modes[interior]
    free_matrix = left.T @ free_action * ds
    free_reference = left.T @ (d2_phi0[interior] + phi0[interior]) * ds
    results = []
    for depth, radial_scale, diffuseness in rows:
        shape = 1.0 / (1.0 + np.exp((radius - radial_scale) / diffuseness))
        weighted_modes = shape[interior, None] * modes[interior]
        interaction_matrix = (
            left.T @ weighted_modes * ds / study.system.com_energy
        )
        reference_interaction = (
            left.T @ (shape[interior] * phi0[interior])
            * ds / study.system.com_energy
        )
        matrix = free_matrix + depth * interaction_matrix
        # The maintained basis is centered on the central-potential solution,
        # not on a free wave.  The forcing is therefore the complete projected
        # residual O(omega) phi_ref; it vanishes at the reference parameters
        # up to the finite-difference/quadrature error of this diagnostic.
        rhs = -(free_reference + depth * reference_interaction)
        try:
            coordinates = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            coordinates = np.linalg.lstsq(matrix, rhs, rcond=1.0e-12)[0]
        results.append(coordinates)
    return np.asarray(results)


def physical_rows(study: Study, color_by: str):
    """Return rows and labels for the two-parameter physical-family panel."""
    vv = study.omega0[0] * np.linspace(0.90, 1.10, 7)
    rr = study.omega0[1] * np.array([0.94, 1.00, 1.06])
    if color_by == "depth":
        color_values, style_values = vv, rr
        rows = np.array([[v, r, study.omega0[2]] for r in rr for v in vv])
        color_label = r"depth $V_0$ [MeV]"
        style_label = r"$R/R_0$"
    else:
        color_values, style_values = rr, vv[[0, 3, 6]]
        rows = np.array([[v, r, study.omega0[2]] for v in style_values for r in rr])
        color_label = r"radius $R$ [fm]"
        style_label = r"$V_0/V_{0,\rm ref}$"
    return rows, color_values, style_values, color_label, style_label


def figure_one(study: Study, color_by: str):
    rows, color_values, style_values, color_label, style_label = physical_rows(
        study, color_by
    )
    waves, _, _ = evaluate(study, rows)
    potentials = np.asarray([
        real_woods_saxon(study.system.radius, row).real for row in rows
    ])
    norm = plt.Normalize(color_values.min(), color_values.max())
    fig, axes = plt.subplots(
        3, 1, figsize=(7.5, 9.2), sharex=True,
        gridspec_kw={"height_ratios": [0.85, 1.15, 1.0]},
        layout="constrained",
    )
    potential_axis, wave_axis, basis_axis = axes
    index = 0
    if color_by == "depth":
        for radius_value in style_values:
            for depth in color_values:
                color = CMAP(norm(depth))
                wave_axis.plot(
                    study.system.radius, waves[index].real,
                    color=color, lw=1.35, alpha=0.88,
                )
                potential_axis.plot(
                    study.system.radius, potentials[index],
                    color=color, lw=1.35, alpha=0.88,
                )
                index += 1
        family_note = r"three radii per color: $R/R_0=0.94,1.00,1.06$"
    else:
        for depth in style_values:
            for radius_value in color_values:
                color = CMAP(norm(radius_value))
                wave_axis.plot(
                    study.system.radius, waves[index].real,
                    color=color, lw=1.35, alpha=0.88,
                )
                potential_axis.plot(
                    study.system.radius, potentials[index],
                    color=color, lw=1.35, alpha=0.88,
                )
                index += 1
        family_note = r"three depths per color: $V_0/V_{0,\rm ref}=0.90,1.00,1.10$"

    potential_axis.set(
        xlim=(0, 10), ylabel=r"$V(r)$ [MeV]",
        title="Woods--Saxon potential family",
    )
    potential_axis.text(
        0.985, 0.07, family_note, transform=potential_axis.transAxes,
        ha="right", va="bottom", fontsize=8, color="0.3",
    )
    wave_axis.axhline(0.0, color="0.45", lw=0.7)
    wave_axis.set(
        xlim=(0, 10), ylabel=r"$\mathrm{Re}\,\phi(r)$",
        title="Matching wavefunction snapshots",
    )
    predictor_r = study.model.selected_coordinates / study.system.wave_number
    central_v = real_woods_saxon(study.system.radius, study.omega0).real
    potential_axis.scatter(
        predictor_r,
        np.interp(predictor_r, study.system.radius, central_v),
        marker="D", s=34, facecolor="white", edgecolor="black", zorder=8,
    )
    # The basis belongs to the complete manifold, not to particular parameter
    # values.  A separate qualitative palette avoids implying that connection.
    mode_colors = ["#2864B7", "#2A9D8F", "#E09F3E", "#A23B72"]
    for j in range(BASIS_SIZE):
        basis_axis.plot(
            study.system.radius, study.model.basis.vectors[:, j].real,
            color=mode_colors[j], lw=1.8, label=rf"$\phi_{j+1}$",
        )
    basis_axis.set(
        xlim=(0, 10), xlabel=r"$r$ [fm]", ylabel="basis function",
        title="Reduced wavefunction basis",
    )
    basis_axis.legend(loc="lower right", ncol=2)
    inset = inset_axes(basis_axis, width="46%", height="42%", loc="upper right", borderpad=1.0)
    singular = study.model.basis.singular_values
    relative = singular / singular[0]
    inset.semilogy(np.arange(1, len(relative) + 1), relative, "o-", ms=2.8, color="0.2")
    inset.axvline(BASIS_SIZE + 0.5, color=CMAP(0.65), ls="--", lw=1.0)
    inset.set(xlim=(0.5, 14.5), ylim=(1e-9, 2))
    inset.set_title("snapshot spectrum", fontsize=8, pad=2)
    inset.tick_params(labelsize=7)

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes,
        location="right", shrink=0.94, pad=0.025, aspect=34,
    )
    colorbar.set_label(color_label)
    fig.suptitle(
        "From interaction family to reduced wavefunction basis",
        fontsize=13,
    )
    save(fig, f"figure_01_two_panel_color_by_{color_by}")


def _plot_physical_pair(axis_potential, axis_wave, study, depth_factors, radius_factors):
    """Plot one compact potential/wavefunction family for layout comparison."""
    depths = study.omega0[0] * np.asarray(depth_factors)
    radii = study.omega0[1] * np.asarray(radius_factors)
    rows = np.asarray([
        [depth, radius, study.omega0[2]]
        for radius in radii for depth in depths
    ])
    waves, _, _ = evaluate(study, rows)
    potentials = np.asarray([
        real_woods_saxon(study.system.radius, row).real for row in rows
    ])
    norm = plt.Normalize(depths.min(), depths.max())
    index = 0
    for _radius in radii:
        for depth in depths:
            color = CMAP(norm(depth))
            axis_potential.plot(
                study.system.radius, potentials[index], color=color,
                lw=1.25, alpha=0.90,
            )
            axis_wave.plot(
                study.system.radius, waves[index].real, color=color,
                lw=1.25, alpha=0.90,
            )
            index += 1
    axis_potential.set(xlim=(0, 10), ylabel=r"$V(r)$ [MeV]")
    axis_wave.axhline(0.0, color="0.45", lw=0.65)
    axis_wave.set(xlim=(0, 10), ylabel=r"$\mathrm{Re}\,\phi(r)$")
    return norm


def figure_one_family_sweeps(study: Study):
    """Focused design sweeps without repeatedly plotting the reduced basis."""
    radius_factors = np.array([0.94, 1.00, 1.06])

    # First isolate curve count at the same physical depth interval.
    count_configs = [
        (7, np.linspace(0.90, 1.10, 7)),
        (5, np.linspace(0.90, 1.10, 5)),
        (4, np.linspace(0.90, 1.10, 4)),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 9.0), sharex=True, layout="constrained")
    for row_index, (count, depth_factors) in enumerate(count_configs):
        norm = _plot_physical_pair(
            axes[row_index, 0], axes[row_index, 1], study,
            depth_factors, radius_factors,
        )
        axes[row_index, 0].set_title(
            rf"{count} depths $\times$ 3 radii: potential"
        )
        axes[row_index, 1].set_title(
            rf"{count} depths $\times$ 3 radii: wavefunction"
        )
    axes[-1, 0].set_xlabel(r"$r$ [fm]")
    axes[-1, 1].set_xlabel(r"$r$ [fm]")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes,
        location="right", shrink=0.92, pad=0.02,
    )
    colorbar.set_label(r"depth $V_0$ [MeV]")
    fig.suptitle(
        r"Family-density comparison at fixed depth span ($\pm10\%$)", fontsize=13
    )
    save(fig, "figure_01_exploration_depth_count")

    # Then isolate physical spacing with a fixed five depths.
    span_configs = [
        (10, np.linspace(0.90, 1.10, 5)),
        (15, np.linspace(0.85, 1.15, 5)),
        (20, np.linspace(0.80, 1.20, 5)),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 9.0), sharex=True, layout="constrained")
    norms = []
    for row_index, (percent, depth_factors) in enumerate(span_configs):
        norms.append(_plot_physical_pair(
            axes[row_index, 0], axes[row_index, 1], study,
            depth_factors, radius_factors,
        ))
        axes[row_index, 0].set_title(
            rf"5 depths over $\pm{percent}\%$: potential"
        )
        axes[row_index, 1].set_title(
            rf"5 depths over $\pm{percent}\%$: wavefunction"
        )
    axes[-1, 0].set_xlabel(r"$r$ [fm]")
    axes[-1, 1].set_xlabel(r"$r$ [fm]")
    # Use the widest span for a common legend covering every displayed depth.
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norms[-1], cmap=CMAP), ax=axes,
        location="right", shrink=0.92, pad=0.02,
    )
    colorbar.set_label(r"depth $V_0$ [MeV]")
    fig.suptitle("Depth-spacing comparison with three radius groups", fontsize=13)
    save(fig, "figure_01_exploration_depth_span")


def figure_one_selected_candidate(study: Study):
    """Polished vertical candidate selected after the family-design sweep."""
    depth_factors = np.linspace(0.80, 1.20, 5)
    radius_factors = np.array([0.94, 1.00, 1.06])
    depths = study.omega0[0] * depth_factors
    radii = study.omega0[1] * radius_factors
    rows = np.asarray([
        [depth, radius, study.omega0[2]]
        for radius in radii for depth in depths
    ])
    waves, _, _ = evaluate(study, rows)
    potentials = np.asarray([
        real_woods_saxon(study.system.radius, row).real for row in rows
    ])
    norm = plt.Normalize(depths.min(), depths.max())

    fig, axes = plt.subplots(
        3, 1, figsize=(7.4, 8.9), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [0.78, 1.10, 0.92]},
    )
    potential_axis, wave_axis, basis_axis = axes

    index = 0
    for _radius in radii:
        for depth in depths:
            color = CMAP(norm(depth))
            potential_axis.plot(
                study.system.radius, potentials[index], color=color,
                lw=1.35, alpha=0.92,
            )
            wave_axis.plot(
                study.system.radius, waves[index].real, color=color,
                lw=1.35, alpha=0.92,
            )
            index += 1

    radial_limit = 10.0
    potential_axis.set(
        xlim=(0, radial_limit), ylabel=r"$V(r)$ [MeV]",
    )
    wave_axis.axhline(0.0, color="0.45", lw=0.7)
    wave_axis.set(
        xlim=(0, radial_limit), ylabel=r"$\phi(r)$",
    )
    predictor_r = study.model.selected_coordinates / study.system.wave_number
    for predictor_index, location in enumerate(predictor_r):
        potential_axis.axvline(
            location, color="#C62828", ls="--", lw=1.4, alpha=0.9,
            label="Predictor locations" if predictor_index == 0 else None,
        )
    potential_axis.legend(loc="lower right", frameon=True, fontsize=12)

    mode_colors = ["#2864B7", "#2A9D8F", "#E09F3E", "#A23B72"]
    phi0 = study.model.basis.reference.real
    phi0_amplitude = np.max(np.abs(phi0))
    basis_axis.plot(
        study.system.radius, phi0, color="0.20", lw=2.2,
        label=r"$\phi_0$",
    )
    for mode in range(BASIS_SIZE):
        raw_mode = study.model.basis.vectors[:, mode].real
        display_mode = raw_mode * phi0_amplitude / np.max(np.abs(raw_mode))
        basis_axis.plot(
            study.system.radius, display_mode,
            color=mode_colors[mode], lw=1.8, label=rf"$\phi_{mode+1}$",
        )
    basis_axis.set(
        xlim=(0, radial_limit), xlabel=r"$r$ [fm]", ylabel="Reduced Basis",
        ylim=(-0.82, 0.82),
    )
    basis_axis.legend(loc="lower right", ncol=3, frameon=True, fontsize=12)

    for axis, panel_label in zip(axes, ("(a)", "(b)", "(c)")):
        axis.text(
            0.015, 0.94, panel_label, transform=axis.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold",
        )
        axis.set_ylabel(axis.get_ylabel(), fontsize=14)
        axis.tick_params(axis="both", labelsize=11)
        axis.set_xticks(np.arange(0, radial_limit + 0.1, 2.0))
    potential_axis.tick_params(axis="x", bottom=True, labelbottom=False)
    wave_axis.tick_params(axis="x", bottom=True, labelbottom=False)
    basis_axis.tick_params(axis="x", bottom=True, labelbottom=True)
    basis_axis.set_xlabel(r"$r$ [fm]", fontsize=14)

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes,
        orientation="vertical", location="right", shrink=0.91,
        fraction=0.028, pad=0.018, aspect=42,
    )
    colorbar.set_label(r"depth $V_0$ [MeV]")
    colorbar.ax.tick_params(labelsize=11)
    for axis in axes:
        axis.grid(False)
    save(fig, "figure_01_selected_five_depths_pm20_vertical")


def coordinate_grid(study: Study, color_by: str):
    if color_by == "depth":
        x_values = np.linspace(0.82 * study.omega0[1], 1.18 * study.omega0[1], 81)
        color_values = study.omega0[0] * np.linspace(0.80, 1.20, 5)
        rows = np.array([[v, x, study.omega0[2]] for v in color_values for x in x_values])
        x_index, color_index = 1, 0
        x_label, color_label = r"radius $R$ [fm]", r"depth $V_0$ [MeV]"
        train_x = study.train_lower[1], study.train_upper[1]
    else:
        x_values = np.linspace(0.78 * study.omega0[0], 1.22 * study.omega0[0], 81)
        color_values = study.omega0[1] * np.array([0.94, 1.00, 1.06])
        rows = np.array([[x, r, study.omega0[2]] for r in color_values for x in x_values])
        x_index, color_index = 0, 1
        x_label, color_label = r"depth $V_0$ [MeV]", r"radius $R$ [fm]"
        train_x = study.train_lower[0], study.train_upper[0]
    waves, ls, lrom = evaluate(study, rows)
    galerkin = galerkin_coordinates(study, rows)
    shape = (len(color_values), len(x_values), BASIS_SIZE)
    return {
        "x": x_values,
        "colors": color_values,
        "rows": rows,
        "x_index": x_index,
        "color_index": color_index,
        "x_label": x_label,
        "color_label": color_label,
        "train_x": train_x,
        "waves": waves.reshape(len(color_values), len(x_values), -1),
        "ls": ls.reshape(shape),
        "lrom": lrom.reshape(shape),
        "galerkin": galerkin.reshape(shape),
    }


def shade_extrapolation(axis, grid):
    axis.axvspan(grid["x"].min(), grid["train_x"][0], color="0.88", alpha=0.8, zorder=0)
    axis.axvspan(grid["train_x"][1], grid["x"].max(), color="0.88", alpha=0.8, zorder=0)
    for boundary in grid["train_x"]:
        axis.axvline(boundary, color="0.5", ls=":", lw=0.8)


def method_handles():
    return [
        Line2D([], [], marker="o", ls="none", mfc="white", mec="0.25", label="LS projection"),
        Line2D([], [], color="0.25", lw=2.0, label="LROM"),
        Line2D([], [], color="0.25", lw=1.4, ls="--", label="Galerkin"),
    ]


def figure_two_paired(study: Study, color_by: str):
    grid = coordinate_grid(study, color_by)
    norm = plt.Normalize(grid["colors"].min(), grid["colors"].max())
    fig, axes = plt.subplots(
        2, 2, figsize=(10.8, 6.5), sharex="col",
        gridspec_kw={"height_ratios": [3.2, 1.25]}, layout="constrained",
    )
    error_values_by_column = []
    for column, coordinate in enumerate([0, 3]):
        main, error = axes[0, column], axes[1, column]
        column_errors = []
        for color_index, value in enumerate(grid["colors"]):
            color = CMAP(norm(value))
            main.plot(grid["x"], grid["lrom"][color_index, :, coordinate].real, color=color, lw=2.0)
            main.plot(grid["x"], grid["galerkin"][color_index, :, coordinate].real, color=color, lw=1.35, ls="--")
            main.scatter(
                grid["x"][::8], grid["ls"][color_index, ::8, coordinate].real,
                s=20, facecolor="white", edgecolor=color, linewidth=1.0, zorder=5,
            )
            lrom_error = np.abs(grid["lrom"][color_index, :, coordinate] - grid["ls"][color_index, :, coordinate])
            galerkin_error = np.abs(grid["galerkin"][color_index, :, coordinate] - grid["ls"][color_index, :, coordinate])
            if color_index == len(grid["colors"]) // 2:
                reference_index = int(np.argmin(np.abs(grid["x"] - study.omega0[grid["x_index"]])))
                lrom_error[reference_index] = np.nan
                galerkin_error[reference_index] = np.nan
            column_errors.extend([lrom_error[np.isfinite(lrom_error)], galerkin_error[np.isfinite(galerkin_error)]])
            error.semilogy(grid["x"], np.maximum(lrom_error, ERROR_FLOOR), color=color, lw=1.6)
            error.semilogy(grid["x"], np.maximum(galerkin_error, ERROR_FLOOR), color=color, lw=1.1, ls="--")
        shade_extrapolation(main, grid)
        shade_extrapolation(error, grid)
        main.set(ylabel=rf"$a_{coordinate+1}$")
        error.set(xlabel=grid["x_label"], ylabel=rf"$|\Delta a_{coordinate+1}|$")
        main.grid(False)
        error.grid(False)
        main.text(
            0.018, 0.94, "(a)" if column == 0 else "(b)",
            transform=main.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold",
        )
        error.text(
            0.018, 0.90, "(c)" if column == 0 else "(d)",
            transform=error.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold",
        )
        main.set_ylabel(main.get_ylabel(), fontsize=14)
        error.set_ylabel(error.get_ylabel(), fontsize=13)
        error.set_xlabel(error.get_xlabel(), fontsize=14)
        main.tick_params(axis="both", labelsize=11)
        error.tick_params(axis="both", labelsize=11)
        if column == 0:
            main.legend(
                handles=method_handles(), loc="lower center", ncol=3,
                frameon=True, fontsize=12,
            )
        error_values_by_column.append(np.concatenate(column_errors))
    for column, values in enumerate(error_values_by_column):
        positive = values[values > 0]
        lower = 10.0 ** np.floor(np.log10(np.percentile(positive, 2)))
        upper = 10.0 ** np.ceil(np.log10(np.percentile(positive, 99.5)))
        axes[1, column].set_ylim(lower, upper)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes,
        location="bottom", shrink=0.48, pad=0.04, aspect=35,
    )
    colorbar.set_label(grid["color_label"])
    colorbar.ax.tick_params(labelsize=11)
    colorbar.set_label(grid["color_label"], fontsize=14)
    save(fig, f"figure_02a_a1_a4_errors_color_by_{color_by}")


def figure_two_four_coordinates(study: Study, color_by: str):
    grid = coordinate_grid(study, color_by)
    norm = plt.Normalize(grid["colors"].min(), grid["colors"].max())
    fig, axes = plt.subplots(
        5, 1, figsize=(7.2, 10.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.72]},
        layout="constrained",
    )
    for coordinate, axis in enumerate(axes[:4]):
        for color_index, value in enumerate(grid["colors"]):
            color = CMAP(norm(value))
            axis.plot(grid["x"], grid["lrom"][color_index, :, coordinate].real, color=color, lw=1.8)
            axis.plot(grid["x"], grid["galerkin"][color_index, :, coordinate].real, color=color, lw=1.15, ls="--")
            axis.scatter(
                grid["x"][::10], grid["ls"][color_index, ::10, coordinate].real,
                s=16, facecolor="white", edgecolor=color, linewidth=0.9, zorder=5,
            )
        shade_extrapolation(axis, grid)
        axis.set_ylabel(rf"$\mathrm{{Re}}\,a_{coordinate+1}$")
        axis.text(0.015, 0.86, rf"$a_{coordinate+1}$", transform=axis.transAxes, weight="bold")
    axes[0].legend(handles=method_handles(), loc="best", ncol=3)

    error_axis = axes[4]
    aggregate_errors = []
    for color_index, value in enumerate(grid["colors"]):
        color = CMAP(norm(value))
        lrom_error = np.linalg.norm(grid["lrom"][color_index] - grid["ls"][color_index], axis=1)
        galerkin_error = np.linalg.norm(grid["galerkin"][color_index] - grid["ls"][color_index], axis=1)
        if color_index == len(grid["colors"]) // 2:
            reference_index = int(np.argmin(np.abs(grid["x"] - study.omega0[grid["x_index"]])))
            lrom_error[reference_index] = np.nan
            galerkin_error[reference_index] = np.nan
        aggregate_errors.extend([lrom_error[np.isfinite(lrom_error)], galerkin_error[np.isfinite(galerkin_error)]])
        error_axis.semilogy(grid["x"], np.maximum(lrom_error, ERROR_FLOOR), color=color, lw=1.6)
        error_axis.semilogy(grid["x"], np.maximum(galerkin_error, ERROR_FLOOR), color=color, lw=1.1, ls="--")
    shade_extrapolation(error_axis, grid)
    error_axis.set(xlabel=grid["x_label"], ylabel=r"$\|\Delta\mathbf{a}\|_2$")
    error_axis.grid(True, which="both", alpha=0.25)
    positive = np.concatenate(aggregate_errors)
    positive = positive[positive > 0]
    error_axis.set_ylim(
        10.0 ** np.floor(np.log10(np.percentile(positive, 2))),
        10.0 ** np.ceil(np.log10(np.percentile(positive, 99.5))),
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes,
        location="right", shrink=0.70, pad=0.02,
    )
    colorbar.set_label(grid["color_label"])
    fig.suptitle("Complete four-coordinate response and aggregate error", fontsize=13)
    save(fig, f"figure_02b_four_coordinates_color_by_{color_by}")


def main():
    assert Path(lrom.__file__).resolve().parent == SOURCE_ROOT / "lrom"
    study = build_study()
    if os.environ.get("LROM_FIGURE_ONLY") == "selected":
        figure_one_selected_candidate(study)
        print(f"saved selected candidate to {FIGURES}", flush=True)
        return
    if os.environ.get("LROM_FIGURE_ONLY") == "coefficients":
        figure_two_paired(study, "depth")
        print(f"saved coefficient comparison to {FIGURES}", flush=True)
        return
    figure_one_family_sweeps(study)
    figure_one_selected_candidate(study)
    for color_by in ("depth", "radius"):
        print(f"generating color-by-{color_by} variants", flush=True)
        figure_one(study, color_by)
        figure_two_paired(study, color_by)
        figure_two_four_coordinates(study, color_by)
    print(f"saved design variants to {FIGURES}", flush=True)


if __name__ == "__main__":
    main()
