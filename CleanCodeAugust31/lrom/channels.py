"""Partial-wave channel definitions for spin-1/2 on spin-zero scattering."""

from __future__ import annotations

Channel = tuple[int, float]


def elastic_channels(l_max: int) -> tuple[Channel, ...]:
    """Return ``(ell, 2 l.s)`` channels through ``l_max``."""
    if l_max < 0:
        raise ValueError("l_max must be non-negative")
    channels: list[Channel] = [(0, 0.0)]
    for ell in range(1, l_max + 1):
        channels.extend(((ell, float(ell)), (ell, float(-(ell + 1)))))
    return tuple(channels)

