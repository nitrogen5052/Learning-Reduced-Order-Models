"""Central public LROM object."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np


PotentialFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


class LROM:
    """Stateful learned reduced-operator model workflow."""

    def __init__(
        self,
        *,
        target: tuple[int, int],
        projectile: tuple[int, int],
        lab_energy: float,
        l: int | tuple[int, ...] = 0,
        fom: str = "nucl-scatter-eq",
        potential: str | PotentialFunction = "ws_3",
        central_parameters: Mapping[str, float] | None = None,
    ) -> None:
        self.target = target
        self.projectile = projectile
        self.lab_energy = lab_energy
        self.l = l
        self.fom = fom
        self.potential = potential
        self.central_parameters = central_parameters
