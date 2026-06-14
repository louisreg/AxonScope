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

from axonscope.identifiers import AxonId
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

    def set_stimulus(self, stimulus: Stimulus) -> None:
        """Attach `stimulus` to this electrode in place.

        Use this when a simulation keeps the same electrode geometry and only
        changes the driven current between runs, for example during threshold
        searches. Plain stimulus amplitudes are interpreted as amperes for
        extracellular stimulation.
        """

        object.__setattr__(self, "stimulus", _normalize_stimulus(stimulus))

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
        x: Any,
        z: Any,
        y: Any | None = None,
        min_distance: Any | None = None,
        stimulus: Stimulus | None = None,
    ) -> None:
        """Create a point-source electrode.

        Parameters
        ----------
        x, y, z:
            Global electrode coordinates. Values must carry length units. `y`
            defaults to 0 um when omitted.
        min_distance:
            Lower bound on source distance to avoid singular footprints.
        stimulus:
            Optional temporal current waveform attached at construction.
        """

        object.__setattr__(self, "x_um", units.require_length_um(x, name="x"))
        object.__setattr__(
            self,
            "y_um",
            0.0 if y is None else units.require_length_um(y, name="y"),
        )
        object.__setattr__(self, "z_um", units.require_length_um(z, name="z"))
        object.__setattr__(
            self,
            "min_distance_um",
            1e-3
            if min_distance is None
            else units.require_length_um(min_distance, name="min_distance"),
        )
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

    def build_footprint(
        self,
        positions: ArrayLike,
        *,
        sigma: Any,
        axon_y: Any | None = None,
        axon_z: Any | None = None,
        source_id: str | None = None,
        axon_id: AxonId | None = None,
    ):
        """Build a static `ExtracellularFootprint` from this point source."""

        from axonscope.stimulation.extracellular import ExtracellularFootprint

        positions_um = units.require_length_array_um(
            positions,
            name="positions",
            dtype=float,
        )
        x_m = positions_um * 1e-6
        sigma_S_m = units.require_conductivity_S_per_m(sigma, name="sigma")
        values = self.footprint_for_axon(
            x_m,
            sigma_S_m=sigma_S_m,
            axon_y_um=0.0 if axon_y is None else axon_y,
            axon_z_um=0.0 if axon_z is None else axon_z,
        )
        if axon_id is not None and not isinstance(axon_id, AxonId):
            raise TypeError("axon_id must be an AxonId.")
        axon_ids = None if axon_id is None else (axon_id,)
        return ExtracellularFootprint(
            values=values,
            positions=units.Q_(positions_um, "micrometer"),
            axon_ids=axon_ids,
            source_id=source_id,
            metadata={
                "builder": "PointSourceElectrode.build_footprint",
                "electrode_x_um": self.x_um,
                "electrode_y_um": self.y_um,
                "electrode_z_um": self.z_um,
                "sigma_S_m": sigma_S_m,
            },
        )
