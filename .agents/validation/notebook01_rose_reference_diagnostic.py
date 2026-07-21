from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scipy.special  # noqa: E402 - repository bootstrap must run first

if not hasattr(scipy.special, "sph_harm") and hasattr(scipy.special, "sph_harm_y"):
    scipy.special.sph_harm = lambda m, n, theta, phi: scipy.special.sph_harm_y(
        n, m, phi, theta
    )

import numpy as np  # noqa: E402
import rose  # noqa: E402
from numba import njit  # noqa: E402

import lrom  # noqa: E402


CASE_IDS = ("test-0021", "test-0039", "test-0065")
BASIS_SIZE = 4


@njit
def rose_real_woods_saxon(radius, alpha):
    vv, rv, av = alpha
    return -vv / (1.0 + np.exp((radius - rv) / av))


def reduced_matrix(rbe, theta: np.ndarray) -> np.ndarray:
    invk, beta = rbe.interaction.coefficients(theta)
    return (
        rbe.A_13
        + np.einsum("i,ijk", beta, rbe.A_2)
        + invk * rbe.A_3_coulomb
    )


def build_rbe(
    emulator, interaction, phi0: np.ndarray, *, overwrite_with_lrom: bool
):
    basis = rose.basis.CustomBasis(
        solutions=np.asarray(
            emulator.samples.training_wavefunctions[0], dtype=np.complex128
        ).T.copy(),
        phi_0=np.asarray(phi0, dtype=np.complex128).copy(),
        rho_mesh=emulator.samples.mesh.rho,
        n_basis=BASIS_SIZE,
        solver=emulator.full_order_model[0].solver,
        subtract_phi0=True,
        use_svd=True,
        center=False,
        scale=False,
    )
    if overwrite_with_lrom:
        basis.vectors = np.asarray(
            emulator.basis[0].vectors, dtype=np.complex128
        )
        basis.phi_0 = np.asarray(emulator.basis[0].phi0, dtype=np.complex128)
    return rose.reduced_basis_emulator.ReducedBasisEmulator(
        interaction,
        basis,
        s_0=emulator.full_order_model[0].base_solver.s_0,
        initialize_emulator=True,
    )


def main() -> None:
    emulator = lrom.LROM(
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        l=0,
        fom="nucl-scatter-eq",
        potential="ws_3",
    )
    center = dict(emulator.central_parameters)
    training_ranges = {
        name: (0.90 * center[name], 1.10 * center[name])
        for name in ("Vv", "Rv", "av")
    }
    testing_ranges = {
        "Vv": (0.78 * center["Vv"], 1.22 * center["Vv"]),
        "Rv": (0.80 * center["Rv"], 1.20 * center["Rv"]),
        "av": (0.80 * center["av"], 1.20 * center["av"]),
    }
    emulator.sampling(
        training_ranges=training_ranges,
        testing_ranges=testing_ranges,
        training_size=70,
        testing_size=81,
        mesh_size=800,
        strategy="latin_hypercube",
        seed=1204,
        high_fidelity_solver="runge_kutta",
    )
    emulator.train(
        basis_size=BASIS_SIZE,
        predictor="potential",
        predictor_count=6,
    )

    central_row = np.asarray([center[name] for name in emulator.parameter_names])
    rose_rows = np.vstack([
        central_row,
        emulator.samples.design.training.values,
        emulator.samples.design.testing.values,
    ])
    rose_bounds = np.column_stack([rose_rows.min(axis=0), rose_rows.max(axis=0)])
    interactions = rose.InteractionEIMSpace(
        l_max=0,
        coordinate_space_potential=rose_real_woods_saxon,
        n_theta=len(emulator.parameter_names),
        mu=emulator.kinematics.mu,
        energy=emulator.kinematics.e_com,
        is_complex=False,
        training_info=rose_bounds,
        n_basis=8,
        rho_mesh=emulator.samples.mesh.rho,
    )
    interaction = interactions.interactions[0][0]

    central_rbe = build_rbe(
        emulator,
        interaction,
        emulator.samples.central_wavefunctions[0],
        overwrite_with_lrom=True,
    )
    free_phi0 = np.asarray(
        [
            rose.free_solutions.phi_free(
                float(rho), 0, emulator.kinematics.eta
            )
            for rho in emulator.samples.mesh.rho
        ],
        dtype=np.complex128,
    )
    free_rbe = build_rbe(
        emulator, interaction, free_phi0, overwrite_with_lrom=False
    )

    print(
        "| reference | case | interpolation | Vv [MeV] | Rv [fm] | av [fm] "
        "| coefficient infinity norm | matrix condition | relative L2 error |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for reference, rbe in (
        ("invalid central", central_rbe),
        ("native free", free_rbe),
    ):
        for case_id in CASE_IDS:
            index = emulator.samples.design.testing.case_ids.index(case_id)
            theta = emulator.samples.design.testing.values[index]
            exact = emulator.samples.testing_wavefunctions[0][index]
            coefficient = rbe.coefficients(theta)
            wavefunction = rbe.emulate_wave_function(theta)
            coefficient_norm = np.linalg.norm(coefficient, ord=np.inf)
            condition = np.linalg.cond(reduced_matrix(rbe, theta))
            relative_l2 = np.linalg.norm(wavefunction - exact) / np.linalg.norm(exact)
            interpolation = all(
                training_ranges[name][0] <= theta[column] <= training_ranges[name][1]
                for column, name in enumerate(emulator.parameter_names)
            )
            print(
                f"| {reference} | `{case_id}` | {interpolation} | "
                f"{theta[0]:.12g} | {theta[1]:.12g} | {theta[2]:.12g} | "
                f"{coefficient_norm:.12e} | {condition:.12e} | {relative_l2:.12e} |"
            )


if __name__ == "__main__":
    main()
