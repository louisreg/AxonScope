"""Post-hoc analysis helpers for simulation results."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks

from axonscope.results.single import SimResult
from axonscope.utils import units


def _vm_matrix(result: SimResult) -> np.ndarray:
    vm = np.asarray(result.Vm, dtype=float)
    if vm.ndim != 2:
        raise ValueError(f"result.Vm must be 2D (time, position), got shape {vm.shape}.")
    return vm


def _time_vector_ms(result: SimResult) -> np.ndarray:
    time_ms = np.asarray(result.t, dtype=float)
    if time_ms.ndim != 1:
        raise ValueError(f"result.t must be 1D, got shape {time_ms.shape}.")
    return time_ms


def recorded_positions_um(result: SimResult) -> np.ndarray:
    """Return the axon positions represented by ``result.Vm`` columns.

    Full recordings map directly to the axon layout positions. Filtered recordings must
    carry ``record_indices`` so analysis code can map recorded columns back to
    physical positions instead of assuming that ``Vm.shape[1] == axon.n_compartments``.
    """

    vm = _vm_matrix(result)
    if not hasattr(result.axon, "layout"):
        raise ValueError("result.axon must expose a layout for spatial analysis.")
    positions = np.asarray(result.axon.layout.position_values(unit="micrometer"), dtype=float)
    if positions.ndim != 1:
        raise ValueError(f"result axon positions must be 1D, got shape {positions.shape}.")
    if result.record_indices is not None:
        indices = np.asarray(result.record_indices, dtype=int)
        if indices.shape != (vm.shape[1],):
            raise ValueError(
                "record_indices must contain one entry per Vm column; "
                f"got {indices.shape[0]} indices for {vm.shape[1]} columns."
            )
        if np.any(indices < 0) or np.any(indices >= positions.shape[0]):
            raise ValueError("record_indices contains values outside axon positions.")
        return positions[indices]

    if vm.shape[1] == positions.shape[0]:
        return positions

    raise ValueError(
        "result.Vm is spatially filtered but result.record_indices is missing; "
        "cannot infer physical positions for analysis."
    )


def _peak_height_mV(value: Any | None, *, threshold_mV: Any) -> float | tuple[float, float] | None:
    if value is None:
        return units.to_mV(threshold_mV)
    converted = units.to_mV_array(value, dtype=float)
    if converted.ndim == 0:
        return float(converted)
    flat = converted.reshape(-1)
    if flat.shape[0] != 2:
        raise ValueError("peak_height_mV must be a scalar or a (min, max) pair.")
    lower = float(flat[0])
    upper = float(flat[1])
    if upper < lower:
        raise ValueError("peak_height_mV upper bound must be >= lower bound.")
    return (lower, upper)


def _recorded_indices(result: SimResult, vm: np.ndarray) -> np.ndarray:
    if result.record_indices is None:
        return np.arange(vm.shape[1], dtype=int)
    indices = np.asarray(result.record_indices, dtype=int)
    if indices.shape != (vm.shape[1],):
        raise ValueError(
            "record_indices must contain one entry per Vm column; "
            f"got {indices.shape[0]} indices for {vm.shape[1]} columns."
        )
    return indices


def _spatially_filter_recording(
    result: SimResult,
    vm: np.ndarray,
    positions_um: np.ndarray,
    *,
    spatial_filter: str,
) -> tuple[np.ndarray, np.ndarray]:
    if spatial_filter == "recorded":
        return vm, positions_um
    if spatial_filter not in {"nodes", "nodes_if_available"}:
        raise ValueError(
            "spatial_filter must be one of 'recorded', 'nodes', or 'nodes_if_available'."
        )

    node_indices = getattr(result.axon, "node_indices", None)
    if node_indices is None:
        if spatial_filter == "nodes_if_available":
            return vm, positions_um
        raise ValueError("spatial_filter='nodes' requires result.axon.node_indices.")

    node_set = {int(index) for index in node_indices}
    original_indices = _recorded_indices(result, vm)
    keep_columns = np.asarray(
        [column for column, index in enumerate(original_indices) if int(index) in node_set],
        dtype=int,
    )
    if keep_columns.size == 0:
        return vm[:, :0], positions_um[:0]
    return vm[:, keep_columns], positions_um[keep_columns]


def rasterize(
    result: SimResult,
    *,
    threshold_mV: Any = -10.0,
    min_distance_ms: Any = 1.0,
    peak_height_mV: Any | None = None,
    min_width_ms: Any | None = None,
    spatial_filter: str = "recorded",
) -> tuple[np.ndarray, np.ndarray]:
    """Detect action potentials and return spike times in ms and positions in um.

    ``peak_height_mV`` can be a scalar lower bound or a ``(min, max)`` pair.
    ``spatial_filter="nodes_if_available"`` restricts detection to node
    compartments when the axon exposes ``node_indices``.
    """

    peak_height = _peak_height_mV(peak_height_mV, threshold_mV=threshold_mV)
    min_distance_ms = units.to_ms(min_distance_ms)
    if min_distance_ms < 0.0:
        raise ValueError("min_distance_ms must be >= 0.")
    min_width_points: int | None = None
    min_width = None if min_width_ms is None else units.to_ms(min_width_ms)
    if min_width is not None and min_width < 0.0:
        raise ValueError("min_width_ms must be >= 0.")

    vm = _vm_matrix(result)
    time_ms = _time_vector_ms(result)
    positions_um = recorded_positions_um(result)
    if vm.shape[0] != time_ms.shape[0]:
        raise ValueError(
            "result.Vm time dimension must match result.t; "
            f"got {vm.shape[0]} and {time_ms.shape[0]}."
        )
    if vm.shape[1] != positions_um.shape[0]:
        raise ValueError(
            "result.Vm position dimension must match recorded positions; "
            f"got {vm.shape[1]} and {positions_um.shape[0]}."
        )

    distance_points = 1
    if time_ms.shape[0] >= 2:
        dt_ms = float(np.median(np.diff(time_ms)))
        if dt_ms > 0.0:
            distance_points = max(1, int(np.ceil(min_distance_ms / dt_ms)))
            if min_width is not None:
                min_width_points = max(1, int(np.ceil(min_width / dt_ms)))

    vm, positions_um = _spatially_filter_recording(
        result,
        vm,
        positions_um,
        spatial_filter=spatial_filter,
    )

    spike_times: list[float] = []
    spike_positions: list[float] = []
    for column, position_um in enumerate(positions_um):
        peaks, _ = find_peaks(
            vm[:, column],
            height=peak_height,
            distance=distance_points,
            width=min_width_points,
        )
        spike_times.extend(time_ms[peaks])
        spike_positions.extend([float(position_um)] * len(peaks))

    return np.asarray(spike_times), np.asarray(spike_positions)


def conduction_velocity(
    result: SimResult,
    *,
    threshold_mV: Any | None = None,
    min_distance_ms: Any = 0.5,
    peak_height_mV: Any | None = (-20.0, 70.0),
    min_width_ms: Any | None = 0.1,
    spatial_filter: str = "nodes_if_available",
) -> float:
    """Estimate action-potential conduction velocity in meters/second.

    The detector finds clean action-potential peaks, restricts myelinated
    recordings to node compartments when available, then measures the distance
    and delay between the first detected spike and the farthest propagated
    spike.
    """

    if threshold_mV is not None and peak_height_mV == (-20.0, 70.0):
        peak_height_mV = None
    if threshold_mV is None:
        threshold_mV = -20.0

    spike_times_ms, spike_positions_um = rasterize(
        result,
        threshold_mV=threshold_mV,
        min_distance_ms=min_distance_ms,
        peak_height_mV=peak_height_mV,
        min_width_ms=min_width_ms,
        spatial_filter=spatial_filter,
    )
    if spike_times_ms.size < 2:
        return 0.0

    return _distance_delay_velocity(spike_times_ms, spike_positions_um)


def average_velocity(
    result: SimResult,
    *,
    threshold_mV: Any | None = None,
    min_distance_ms: Any = 0.5,
    peak_height_mV: Any | None = (-20.0, 70.0),
    min_width_ms: Any | None = 0.1,
    spatial_filter: str = "nodes_if_available",
) -> float:
    """Alias for ``conduction_velocity``."""

    return conduction_velocity(
        result,
        threshold_mV=threshold_mV,
        min_distance_ms=min_distance_ms,
        peak_height_mV=peak_height_mV,
        min_width_ms=min_width_ms,
        spatial_filter=spatial_filter,
    )


def peak_voltage(result: SimResult) -> np.ndarray:
    """Return the peak membrane voltage in mV for each recorded position."""

    return np.max(_vm_matrix(result), axis=0)


def _distance_delay_velocity(spike_times_ms: np.ndarray, spike_positions_um: np.ndarray) -> float:
    order = np.argsort(spike_times_ms)
    times_ms = np.asarray(spike_times_ms, dtype=float)[order]
    positions_um = np.asarray(spike_positions_um, dtype=float)[order]
    x_start = float(positions_um[0])
    t_start = float(times_ms[0])
    stop_index = int(np.argmax(np.abs(positions_um - x_start)))
    delay_ms = float(times_ms[stop_index]) - t_start
    if delay_ms <= 0.0:
        return 0.0
    distance_um = abs(float(positions_um[stop_index]) - x_start)
    return float(distance_um / delay_ms * 1e-3)


__all__ = [
    "average_velocity",
    "conduction_velocity",
    "peak_voltage",
    "rasterize",
    "recorded_positions_um",
]
