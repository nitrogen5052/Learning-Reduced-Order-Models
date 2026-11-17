import numpy as np

from lrom import BasisConfig, MaxVol, ScatteringLROM, ScatteringProblem, get_runtime
from lrom.global_kd import KD_COEFFICIENTS, kd_parameters_from_coefficients


def test_cpu_runtime_solves_complex_batch():
    runtime = get_runtime("cpu")
    matrices = np.asarray([np.eye(2), 2 * np.eye(2)], complex)
    rhs = np.ones((2, 2), complex)
    np.testing.assert_allclose(runtime.solve(matrices, rhs), [[1, 1], [.5, .5]])


def test_global_kd_zero_shift_is_vectorized():
    values = kd_parameters_from_coefficients(
        np.asarray([40, 208]), np.asarray([20, 82]), np.asarray([14.1, 50.0])
    )
    assert values.shape == (2, 15)
    assert len(KD_COEFFICIENTS) == 37
    assert np.all(np.isfinite(values))


def test_small_lrom_cpu_smoke():
    problem = ScatteringProblem(
        potential="real_woods_saxon", target_a=40, target_z=20,
        lab_energy=14.1, l_max=0, mesh_points=180,
        target_binding_energy=342.052184,
    )
    center = problem.reference_omega
    samples = []
    for scale in np.linspace(.92, 1.08, 10):
        sample = problem.reference_sample.copy()
        sample.update(dict(zip(problem.potential.parameter_names, center * scale)))
        samples.append(sample)
    model = ScatteringLROM(
        problem, BasisConfig(size=3), MaxVol(count=2, ridge=1e-14)
    ).train(
        problem.generate_training_data(samples)
    )
    prediction = model.predict(samples[4]).wavefunction((0, 0.0))
    assert prediction.shape == (180,)
    assert np.all(np.isfinite(prediction))
