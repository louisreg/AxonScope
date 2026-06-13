"""Extracellular electrode source descriptions.

Electrodes describe physical sources in the global simulation frame: position,
attached temporal current stimulus, and source-specific geometry. They do not
own the extracellular medium. Analytical media such as homogeneous point-source
conductivity live on `AnalyticalExtracellularContext`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from axonscope.stimulation.stimuli import ArrayLike, Stimulus
from axonscope.utils import units


def _normalize_stimulus(stimulus: Stimulus) -> Stimulus:
    """Validate an electrode stimulus and express amplitudes in amperes."""

    if not isinstance(stimulus, Stimulus):
        raise TypeError("stimulus must be an axonscope.stimulation.Stimulus.")
    return stimulus.as_unit("ampere")


@dataclass(frozen=True)
class Electrode(ABC):
    """Base class for positioned extracellular current sources.

    An electrode can own an attached stimulus, but it does not know how the
    surrounding medium maps its current to Vext. Context objects provide that
    medium-specific mapping.
    """

    stimulus: Stimulus | None = field(default=None, init=False, compare=False)
    """Attached current waveform normalized to amperes, or `None`."""

    def with_stimulus(self, stimulus: Stimulus) -> "Electrode":
        """Return a copy of this electrode with `stimulus` attached.

        Plain stimulus amplitudes are interpreted as amperes for extracellular
        stimulation. This is the preferred method when reusing the same
        electrode geometry across multiple simulations with different stimuli.
        """

        electrode = copy.copy(self)
        object.__setattr__(electrode, "stimulus", _normalize_stimulus(stimulus))
        return electrode

    def with_scaled_stimulus(self, scale: float) -> "Electrode":
        """Return a copy whose attached stimulus amplitude is scaled."""

        stimulus = getattr(self, "stimulus", None)
        if stimulus is None:
            raise ValueError("Attach a stimulus before scaling an electrode.")
        return self.with_stimulus(stimulus.scaled(scale))


@dataclass(frozen=True)
class AnalyticalElectrode(Electrode, ABC):
    """Base class for electrodes with an analytic footprint formula."""

    @abstractmethod
    def footprint(
        self,
        x_positions_m: ArrayLike,
        *,
        sigma_S_m: float,
    ) -> np.ndarray:
        """Return V/A footprint samples for an axon at y=z=0.

        Parameters
        ----------
        x_positions_m:
            Axial sample positions in meters.
        sigma_S_m:
            Conductivity of the homogeneous medium in S/m.
        """
        raise NotImplementedError

    def footprint_for_axon(
        self,
        x_positions_m: ArrayLike,
        *,
        sigma_S_m: float,
        axon_y_um: Any = 0.0,
        axon_z_um: Any = 0.0,
    ) -> np.ndarray:
        """Return V/A footprint samples for one globally positioned axon."""

        return self.footprint(x_positions_m, sigma_S_m=sigma_S_m)


@dataclass(frozen=True, init=False)
class PointSourceElectrode(AnalyticalElectrode):
    """Point-source electrode in a homogeneous infinite medium.

    The electrode stores only source geometry and stimulus. Conductivity is
    supplied by `AnalyticalExtracellularContext`, which evaluates:

        Vext(x, t) = I(t) / (4*pi*sigma*r)
    """

    x_um: float
    y_um: float
    z_um: float
    min_distance_um: float = 1e-3

    def __init__(
        self,
        *,
        x_um: Any | None = None,
        y_um: Any = 0.0,
        z_um: Any | None = None,
        min_distance_um: Any | None = None,
        stimulus: Stimulus | None = None,
        x0_m: Any | None = None,
        y0_m: Any | None = None,
        z0_m: Any | None = None,
        min_distance_m: Any | None = None,
    ) -> None:
        """Create a point-source electrode.

        Parameters
        ----------
        x_um, y_um, z_um:
            Global electrode coordinates. Plain numbers are interpreted as
            micrometers. `x_um` is required unless `x0_m` is provided.
        min_distance_um:
            Lower bound on source distance to avoid singular footprints.
        stimulus:
            Optional temporal current waveform attached at construction.
        x0_m, y0_m, z0_m, min_distance_m:
            SI aliases for diagnostics and low-level tests. Do not combine
            them with the corresponding micrometer arguments.
        """

        if x_um is not None and x0_m is not None:
            raise ValueError("Provide either x_um or x0_m, not both.")
        if z_um is not None and z0_m is not None:
            raise ValueError("Provide either z_um or z0_m, not both.")
        if y0_m is not None:
            if y_um != 0.0:
                raise ValueError("Provide either y_um or y0_m, not both.")
            y_um = units.to_m(y0_m) * 1e6
        if x_um is None:
            if x0_m is None:
                raise ValueError("x_um is required.")
            x_um = units.to_m(x0_m) * 1e6
        if z_um is None:
            z_um = units.to_m(z0_m) * 1e6 if z0_m is not None else 1000.0
        if min_distance_um is not None and min_distance_m is not None:
            raise ValueError("Provide either min_distance_um or min_distance_m, not both.")
        if min_distance_um is None:
            min_distance_um = (
                units.to_m(min_distance_m) * 1e6 if min_distance_m is not None else 1e-3
            )

        object.__setattr__(self, "x_um", units.to_um(x_um))
        object.__setattr__(self, "y_um", units.to_um(y_um))
        object.__setattr__(self, "z_um", units.to_um(z_um))
        object.__setattr__(self, "min_distance_um", units.to_um(min_distance_um))
        if stimulus is not None:
            stimulus = _normalize_stimulus(stimulus)
        object.__setattr__(self, "stimulus", stimulus)

    @property
    def x0_m(self) -> float:
        """Electrode x position in meters."""

        return self.x_um * 1e-6

    @property
    def y0_m(self) -> float:
        """Electrode y position in meters."""

        return self.y_um * 1e-6

    @property
    def z0_m(self) -> float:
        """Electrode z position in meters."""

        return self.z_um * 1e-6

    @property
    def min_distance_m(self) -> float:
        """Minimum source distance in meters."""

        return self.min_distance_um * 1e-6

    def footprint(
        self,
        x_positions_m: ArrayLike,
        *,
        sigma_S_m: float,
    ) -> np.ndarray:
        """Return the point-source V/A footprint for an axon at y=z=0."""

        x = units.to_m_array(x_positions_m, dtype=float)
        r = np.sqrt((x - self.x0_m) ** 2 + self.y0_m**2 + self.z0_m**2)
        r = np.maximum(r, self.min_distance_m)
        return 1.0 / (4.0 * np.pi * float(sigma_S_m) * r)

    def footprint_for_axon(
        self,
        x_positions_m: ArrayLike,
        *,
        sigma_S_m: float,
        axon_y_um: Any = 0.0,
        axon_z_um: Any = 0.0,
    ) -> np.ndarray:
        """Return the point-source footprint for an axon in global coordinates.

        `x_positions_m` should include the axon's global x offset. `axon_y_um`
        and `axon_z_um` are subtracted from the electrode's global transverse
        coordinates before computing the source distance.
        """

        x = units.to_m_array(x_positions_m, dtype=float)
        y_rel_m = (self.y_um - units.to_um(axon_y_um)) * 1e-6
        z_rel_m = (self.z_um - units.to_um(axon_z_um)) * 1e-6
        r = np.sqrt((x - self.x0_m) ** 2 + y_rel_m**2 + z_rel_m**2)
        r = np.maximum(r, self.min_distance_m)
        return 1.0 / (4.0 * np.pi * float(sigma_S_m) * r)
