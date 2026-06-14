"""Plotting helpers for AxonScope results and model descriptions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from axonscope.analysis.posthoc import rasterize
from axonscope.results.single import SimResult
from axonscope.utils import units

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


def plot_raster(
    result: SimResult,
    ax: Axes | None = None,
    *,
    threshold_mV: Any = -10.0,
    min_distance_ms: Any = 1.0,
    line_half_height_um: Any = 0.5,
) -> Axes:
    """Plot detected spikes as time-position raster lines.

    Plain numeric values are interpreted as millivolts, milliseconds, and
    micrometers. Pint-like quantities are accepted for all unit-bearing
    parameters.
    """

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()

    line_half_height = units.to_um(line_half_height_um)
    spike_times_ms, spike_positions_um = rasterize(
        result,
        threshold_mV=threshold_mV,
        min_distance_ms=min_distance_ms,
    )
    if spike_times_ms.size:
        ax.vlines(
            spike_times_ms,
            spike_positions_um - line_half_height,
            spike_positions_um + line_half_height,
            color="black",
            linewidth=1,
        )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Axon position (um)")
    return ax


rasterplot = plot_raster


__all__ = ["plot_raster", "rasterplot"]
