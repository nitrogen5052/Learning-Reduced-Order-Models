from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scipy.special

if not hasattr(scipy.special, "sph_harm") and hasattr(scipy.special, "sph_harm_y"):
    scipy.special.sph_harm = lambda m, n, theta, phi: scipy.special.sph_harm_y(
        n, m, phi, theta
    )

import numpy as np
import rose

import lrom_legacy.v1_2 as lrom


CASE_IDS = ("test-0021", "test-0039", "test-0065")
BASIS_SIZE = 4


def reduced_matrix(rbe, theta: np.ndarray) -> np.ndarray:
    invk, beta = rbe.interaction.coefficients(theta)
    return (
        rbe.A_13
        + np.einsum("i,ijk", beta, rbe.A_2)
        + invk * rbe.A_3_coulomb
    )


def build_rbe(emulator, phi0: np.ndarray, *, overwrite_with_lrom: bool):
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
        emulator.full_order_model[0].interaction,
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
        mesh_size=900,
        strategy="latin_hypercube",
        seed=1204,
        eim_basis_size=8,
    )
    emulator.train(
        basis_size=BASIS_SIZE,
        predictor="potential",
        predictor_count=6,
    )

    central_rbe = build_rbe(
        emulator,
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
    free_rbe = build_rbe(emulator, free_phi0, overwrite_with_lrom=False)

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
