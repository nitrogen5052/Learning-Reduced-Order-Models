from __future__ import annotations

import numpy as np

from lrom_bench.sampling import centered_box_samples, one_at_a_time_scan_samples


def test_one_at_a_time_scan_samples_preserves_center_except_active_coordinate() -> None:
    center = np.array([10.0, 2.0])
    widths = np.array([1.0, 0.5])

    samples, info = one_at_a_time_scan_samples(center, widths, n_scan=3, names=("Vv", "Rv"))

    assert samples.shape == (6, 2)
    assert info[0].name == "Vv"
    assert info[0].values.tolist() == [9.0, 10.0, 11.0]
    assert np.allclose(samples[:3, 1], 2.0)
    assert info[1].name == "Rv"
    assert info[1].values.tolist() == [1.5, 2.0, 2.5]
    assert np.allclose(samples[3:, 0], 10.0)


def test_centered_box_samples_are_deterministic_and_include_center() -> None:
    center = np.array([10.0, 2.0])
    widths = np.array([1.0, 0.5])

    samples_a = centered_box_samples(center, widths, n_samples=5, seed=123, include_center=True)
    samples_b = centered_box_samples(center, widths, n_samples=5, seed=123, include_center=True)

    assert samples_a.shape == (6, 2)
    assert np.allclose(samples_a, samples_b)
    assert np.allclose(samples_a[0], center)
    assert np.all(samples_a[:, 0] >= 9.0)
    assert np.all(samples_a[:, 0] <= 11.0)
    assert np.all(samples_a[:, 1] >= 1.5)
    assert np.all(samples_a[:, 1] <= 2.5)
