from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class ScanInfo:
    name: str
    values: np.ndarray
    slc: slice


def centered_1d_values(center: float, width: float, n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2")
    if width <= 0.0:
        raise ValueError("width must be positive")
    return np.linspace(float(center) - float(width), float(center) + float(width), int(n))


def one_at_a_time_scan_samples(
    center: np.ndarray,
    widths: np.ndarray,
    n_scan: int,
    names: tuple[str, ...],
) -> tuple[np.ndarray, list[ScanInfo]]:
    center = np.asarray(center, dtype=float)
    widths = np.asarray(widths, dtype=float)
    if center.shape != widths.shape:
        raise ValueError("center and widths must have the same shape")
    if len(names) != center.size:
        raise ValueError("names must have one entry per parameter")
    if n_scan < 2:
        raise ValueError("n_scan must be at least 2")

    samples = []
    info: list[ScanInfo] = []
    start = 0
    for j, name in enumerate(names):
        values = np.linspace(center[j] - widths[j], center[j] + widths[j], n_scan)
        block = np.tile(center, (n_scan, 1))
        block[:, j] = values
        stop = start + n_scan
        samples.append(block)
        info.append(ScanInfo(name=name, values=values, slc=slice(start, stop)))
        start = stop
    return np.vstack(samples), info


def centered_box_samples(
    center: np.ndarray,
    widths: np.ndarray,
    n_samples: int,
    seed: int,
    include_center: bool = False,
) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    widths = np.asarray(widths, dtype=float)
    if center.shape != widths.shape:
        raise ValueError("center and widths must have the same shape")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    lower = center - widths
    upper = center + widths
    sampler = qmc.LatinHypercube(d=center.size, seed=seed)
    samples = qmc.scale(sampler.random(n_samples), lower, upper)
    if include_center:
        samples = np.vstack([center, samples])
    return samples
