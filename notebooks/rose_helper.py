# HELPER FUNCTIONS

from types import SimpleNamespace
import rose
import numpy as np
from numba import njit
@njit
def rose_real_woods_saxon(radius, alpha):
    """ROSE evaluates this exact callback for HF solves and builds its EIM from it."""
    vv, rv, av = alpha
    return -vv / (1.0 + np.exp((radius - rv) / av))


def build_rose(emulator, center, n_basis, eim_basis=8, emulate_train_wf=True):#
    """Build a ROSE reduced-basis emulator mirroring an LROM emulator's samples."""
    s, kin = emulator.samples, emulator.kinematics
    rho = s.mesh.rho
    train_rows, test_rows = s.design.training.values, s.design.testing.values

    rows = np.vstack([[center[n] for n in emulator.parameter_names], train_rows, test_rows])
    bounds = np.column_stack([rows.min(0), rows.max(0)])

    interaction = rose.InteractionEIMSpace(
        l_max=0,
        coordinate_space_potential=rose_real_woods_saxon,
        n_theta=len(emulator.parameter_names),
        mu=kin.mu,
        energy=kin.e_com,
        is_complex=False,
        training_info=bounds,
        n_basis=eim_basis,
        rho_mesh=rho,
    ).interactions[0][0]

    basis = rose.basis.CustomBasis(
        solutions=np.asarray(s.training_wavefunctions[0], dtype=np.complex128).T.copy(),
        phi_0=np.array([rose.free_solutions.phi_free(float(x), 0, kin.eta) for x in rho],
                       dtype=np.complex128),
        rho_mesh=rho,
        n_basis=n_basis,
        solver=emulator.full_order_model[0].solver,
        subtract_phi0=True, use_svd=True, center=False, scale=False,
    )
    rbe = rose.reduced_basis_emulator.ReducedBasisEmulator(
        interaction, basis, s_0=emulator.full_order_model[0].base_solver.s_0,
        initialize_emulator=True,
    )

    coef = lambda rows_: np.array([rbe.coefficients(r_) for r_ in rows_])
    wf = lambda rows_: np.array([rbe.emulate_wave_function(r_) for r_ in rows_])
    return SimpleNamespace(
        rbe=rbe, basis=basis, interaction=interaction,
        train_rows=train_rows, test_rows=test_rows,
        train_coefficients=coef(train_rows), coefficients=coef(test_rows),
        wf_train=wf(train_rows) if emulate_train_wf else None,
        wf_test=wf(test_rows),
    )

