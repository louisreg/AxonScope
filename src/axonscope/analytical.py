"""Lightweight analytical helpers for examples and tests.

This module does not make AxonScope own nerve geometry. It only converts simple
analytical point-source setups into the local axon frame expected by the current
extracellular context runtime.
"""

from __future__ import annotations

from typing import Any

from axonscope.stimulation import (
    AnalyticalExtracellularContext,
    PointSourceElectrode,
    Stimulus,
)
from axonscope.utils import units


def local_point_source_context(
    electrode: PointSourceElectrode,
    *,
    stimulus: Stimulus | None = None,
    sigma: Any | None = None,
    axon_x_offset: Any | None = None,
    axon_y: Any | None = None,
    axon_z: Any | None = None,
) -> AnalyticalExtracellularContext:
    """Build a point-source context expressed in one axon's local frame.

    `axon_x_offset`, `axon_y`, and `axon_z` are external placement offsets used
    only to shift the analytical electrode into the axon-local frame. The
    returned context can be attached to an `AxonInstance` with no
    world-coordinate placement.
    """

    if not isinstance(electrode, PointSourceElectrode):
        raise TypeError("electrode must be a PointSourceElectrode.")

    attached_stimulus = stimulus if stimulus is not None else electrode.stimulus
    if attached_stimulus is None:
        raise ValueError("Provide a stimulus or pass an electrode that already has one.")

    axon_x_um = (
        0.0
        if axon_x_offset is None
        else units.require_length_um(axon_x_offset, name="axon_x_offset")
    )
    axon_y_um = 0.0 if axon_y is None else units.require_length_um(axon_y, name="axon_y")
    axon_z_um = 0.0 if axon_z is None else units.require_length_um(axon_z, name="axon_z")

    local_electrode = PointSourceElectrode(
        x=units.Q_(electrode.x_um - axon_x_um, "micrometer"),
        y=units.Q_(electrode.y_um - axon_y_um, "micrometer"),
        z=units.Q_(electrode.z_um - axon_z_um, "micrometer"),
        min_distance=units.Q_(electrode.min_distance_um, "micrometer"),
    ).with_stimulus(attached_stimulus)

    return AnalyticalExtracellularContext(
        electrodes=[local_electrode],
        sigma=sigma,
    )


__all__ = ["local_point_source_context"]
