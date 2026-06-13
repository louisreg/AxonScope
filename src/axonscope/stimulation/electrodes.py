from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from axonscope.stimulation import ExtracellularContext
from axonscope.stimulus import ArrayLike, Stimulus


@dataclass(frozen=True)
class Electrode(ABC):
    """Spatial extracellular electrode description."""

    @abstractmethod
    def footprint(self, x_positions_m: ArrayLike) -> np.ndarray:
        """Return extracellular potential per unit current in V/A."""
        raise NotImplementedError

    def attach_stimulus(self, stimulus: Stimulus) -> ExtracellularContext:
        """Pair this electrode with a temporal stimulus description."""
        return ExtracellularContext(electrode=self, stimulus=stimulus)


@dataclass(frozen=True)
class PointSourceElectrode(Electrode):
    """
    Point-source electrode in an infinite homogeneous conductive medium.

    Formula
    -------
        Vext(x, t) = I(t) / (4πσr)
    """

    x0_m: float
    y0_m: float = 0.0
    z0_m: float = 1e-3
    sigma_S_m: float = 0.3
    min_distance_m: float = 1e-9

    def footprint(self, x_positions_m: ArrayLike) -> np.ndarray:
        x = np.asarray(x_positions_m, dtype=float)
        r = np.sqrt((x - self.x0_m) ** 2 + self.y0_m**2 + self.z0_m**2)
        r = np.maximum(r, self.min_distance_m)
        return 1.0 / (4.0 * np.pi * self.sigma_S_m * r)
