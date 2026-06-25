"""Small NRV-to-AxonScope mapping helpers.

This module intentionally does not import NRV. It only converts NRV table
metadata into AxonScope public model parameters.
"""

from __future__ import annotations

from typing import Literal

from axonscope.axons.templates import mrg_like_node_spacing
from axonscope.utils.units import Q_


FiberKind = Literal["hh", "rattay", "mrg"]


def fiber_kind_from_nrv(nrv_type: int, *, include_mrg: bool) -> FiberKind:
    """Map NRV's fiber type code to the AxonScope template used by examples."""

    if int(nrv_type) == 1:
        return "mrg" if include_mrg else "rattay"
    return "rattay"


def nrv_node_shift_to_x_shift_um(
    node_shift: float,
    diameter_um: float,
    *,
    kind: FiberKind,
) -> float:
    """Convert NRV's fractional MRG node shift to AxonScope MRG phase."""

    if kind != "mrg":
        return 0.0
    node_spacing_um = mrg_like_node_spacing(Q_(float(diameter_um), "micrometer"))
    return float(node_shift) * node_spacing_um


__all__ = [
    "FiberKind",
    "fiber_kind_from_nrv",
    "nrv_node_shift_to_x_shift_um",
]
