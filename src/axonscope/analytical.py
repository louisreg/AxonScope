"""Lightweight analytical helpers for examples and quick starts.

This module does not make AxonScope own nerve geometry. It provides small
closed-form helpers that convert didactic analytical setups into the typed
footprint/drive objects consumed by normal solver execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from axonscope.identifiers import AxonId, DriveId
from axonscope.stimulation import (
    ExtracellularDrive,
    ExtracellularFootprint,
    ExtracellularStimulation,
    Stimulus,
)
from axonscope.stimulation.stimuli import ArrayLike
from axonscope.utils import units


@dataclass(frozen=True, init=False)
class PointSourceElectrode:
    """Analytical point source in a homogeneous infinite medium.

    This is a convenience helper, not a solver/runtime concept. Use
    `point_source_footprint(...)`, `point_source_drive(...)`, or
    `point_source_stimulation(...)` to convert it to the typed extracellular
    objects used by simulations.
    """

    x_um: float
    y_um: float
    z_um: float
    min_distance_um: float = 1e-3
    stimulus: Stimulus | None = None

    def __init__(
        self,
        *,
        x: Any,
        z: Any,
        y: Any | None = None,
        min_distance: Any | None = None,
        stimulus: Stimulus | None = None,
    ) -> None:
        """Create a point-source helper.

        Coordinates belong to the caller's external analytical frame. They are
        converted to an axon-local sampled footprint before solver execution.
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
            if not isinstance(stimulus, Stimulus):
                raise TypeError("stimulus must be an axonscope.stimulation.Stimulus.")
            stimulus = stimulus.as_unit("ampere")
        object.__setattr__(self, "stimulus", stimulus)

    def with_stimulus(self, stimulus: Stimulus) -> "PointSourceElectrode":
        """Return a copy with `stimulus` attached as an ampere waveform."""

        if not isinstance(stimulus, Stimulus):
            raise TypeError("stimulus must be an axonscope.stimulation.Stimulus.")
        return PointSourceElectrode(
            x=units.Q_(self.x_um, "micrometer"),
            y=units.Q_(self.y_um, "micrometer"),
            z=units.Q_(self.z_um, "micrometer"),
            min_distance=units.Q_(self.min_distance_um, "micrometer"),
            stimulus=stimulus,
        )

    def with_scaled_stimulus(self, scale: float) -> "PointSourceElectrode":
        """Return a copy whose attached stimulus is multiplied by `scale`."""

        if self.stimulus is None:
            raise ValueError("PointSourceElectrode has no attached stimulus to scale.")
        return self.with_stimulus(self.stimulus.scaled(float(scale)))

    def set_stimulus(self, stimulus: Stimulus) -> None:
        """Attach `stimulus` in place for legacy notebook convenience."""

        if not isinstance(stimulus, Stimulus):
            raise TypeError("stimulus must be an axonscope.stimulation.Stimulus.")
        object.__setattr__(self, "stimulus", stimulus.as_unit("ampere"))

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
        """Return the V/A footprint for local axon positions at y=z=0."""

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
        """Return the V/A footprint after optional external transverse offsets."""

        x = units.to_m_array(x_positions_m, dtype=float)
        y_rel_m = (self.y_um - units.to_um(axon_y_um)) * 1e-6
        z_rel_m = (self.z_um - units.to_um(axon_z_um)) * 1e-6
        r = np.sqrt((x - self.x0_m) ** 2 + y_rel_m**2 + z_rel_m**2)
        r = np.maximum(r, self.min_distance_m)
        return 1.0 / (4.0 * np.pi * float(sigma_S_m) * r)


def point_source_footprint(
    electrode: PointSourceElectrode,
    positions: ArrayLike,
    *,
    sigma: Any,
    axon_x_offset: Any | None = None,
    axon_y: Any | None = None,
    axon_z: Any | None = None,
    source_id: str | None = None,
    axon_id: AxonId | None = None,
) -> ExtracellularFootprint:
    """Sample a point-source helper into an `ExtracellularFootprint`.

    External placement offsets are used only while sampling. The returned
    footprint is expressed on intrinsic axon positions.
    """

    if not isinstance(electrode, PointSourceElectrode):
        raise TypeError("electrode must be a PointSourceElectrode.")
    positions_um = units.require_length_array_um(
        positions,
        name="positions",
        dtype=float,
    )
    sigma_S_m = units.require_conductivity_S_per_m(sigma, name="sigma")
    axon_x_um = (
        0.0
        if axon_x_offset is None
        else units.require_length_um(axon_x_offset, name="axon_x_offset")
    )
    axon_y_um = (
        0.0 if axon_y is None else units.require_length_um(axon_y, name="axon_y")
    )
    axon_z_um = (
        0.0 if axon_z is None else units.require_length_um(axon_z, name="axon_z")
    )

    local_electrode = PointSourceElectrode(
        x=units.Q_(electrode.x_um - axon_x_um, "micrometer"),
        y=units.Q_(electrode.y_um, "micrometer"),
        z=units.Q_(electrode.z_um, "micrometer"),
        min_distance=units.Q_(electrode.min_distance_um, "micrometer"),
    )
    values = local_electrode.footprint_for_axon(
        positions_um * 1e-6,
        sigma_S_m=sigma_S_m,
        axon_y_um=axon_y_um,
        axon_z_um=axon_z_um,
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
            "builder": "axonscope.analytical.point_source_footprint",
            "source": "point_source_helper",
            "electrode_x_um": electrode.x_um,
            "electrode_y_um": electrode.y_um,
            "electrode_z_um": electrode.z_um,
            "axon_x_offset_um": axon_x_um,
            "axon_y_um": axon_y_um,
            "axon_z_um": axon_z_um,
            "sigma_S_m": sigma_S_m,
        },
    )


def point_source_drive(
    electrode: PointSourceElectrode,
    positions: ArrayLike,
    *,
    sigma: Any,
    stimulus: Stimulus | None = None,
    drive_id: DriveId | str = "point_source",
    axon_x_offset: Any | None = None,
    axon_y: Any | None = None,
    axon_z: Any | None = None,
    source_id: str | None = None,
    axon_id: AxonId | None = None,
) -> ExtracellularDrive:
    """Build one typed drive from a point-source helper."""

    attached_stimulus = stimulus if stimulus is not None else electrode.stimulus
    if attached_stimulus is None:
        raise ValueError("Provide a stimulus or pass an electrode that already has one.")
    drive = drive_id if isinstance(drive_id, DriveId) else DriveId(str(drive_id))
    return ExtracellularDrive(
        id=drive,
        footprint=point_source_footprint(
            electrode,
            positions,
            sigma=sigma,
            axon_x_offset=axon_x_offset,
            axon_y=axon_y,
            axon_z=axon_z,
            source_id=source_id,
            axon_id=axon_id,
        ),
        stimulus=attached_stimulus,
        metadata={"source": "point_source_helper"},
    )


def point_source_stimulation(
    electrode: PointSourceElectrode,
    positions: ArrayLike,
    *,
    sigma: Any,
    stimulus: Stimulus | None = None,
    drive_id: DriveId | str = "point_source",
    axon_x_offset: Any | None = None,
    axon_y: Any | None = None,
    axon_z: Any | None = None,
    source_id: str | None = None,
    axon_id: AxonId | None = None,
) -> ExtracellularStimulation:
    """Build a one-drive `ExtracellularStimulation` from a point-source helper."""

    return ExtracellularStimulation(
        [
            point_source_drive(
                electrode,
                positions,
                sigma=sigma,
                stimulus=stimulus,
                drive_id=drive_id,
                axon_x_offset=axon_x_offset,
                axon_y=axon_y,
                axon_z=axon_z,
                source_id=source_id,
                axon_id=axon_id,
            )
        ]
    )


__all__ = [
    "PointSourceElectrode",
    "point_source_drive",
    "point_source_footprint",
    "point_source_stimulation",
]
