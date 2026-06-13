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


def rasterize(
    result: SimResult,
    *,
    threshold_mV: Any = -10.0,
    min_distance_ms: Any = 1.0,
    threshold: Any | None = None,
    min_distance: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect action potentials and return spike times in ms and positions in um.

    ``threshold`` and ``min_distance`` are accepted as short aliases for older
    call sites; public examples should prefer the unit-explicit names. Plain
    numeric thresholds are interpreted as millivolts and distances as
    milliseconds; Pint-like quantities are converted at the API boundary.
    """

    if threshold is not None:
        threshold_mV = threshold
    if min_distance is not None:
        min_distance_ms = min_distance
    threshold_mV = units.to_mV(threshold_mV)
    min_distance_ms = units.to_ms(min_distance_ms)
    if min_distance_ms < 0.0:
        raise ValueError("min_distance_ms must be >= 0.")

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

    spike_times: list[float] = []
    spike_positions: list[float] = []
    for column, position_um in enumerate(positions_um):
        peaks, _ = find_peaks(
            vm[:, column],
            height=threshold_mV,
            distance=distance_points,
        )
        spike_times.extend(time_ms[peaks])
        spike_positions.extend([float(position_um)] * len(peaks))

    return np.asarray(spike_times), np.asarray(spike_positions)


def conduction_velocity(
    result: SimResult,
    *,
    threshold_mV: Any = -10.0,
    min_distance_ms: Any = 1.0,
    threshold: Any | None = None,
    min_distance: Any | None = None,
) -> float:
    """Estimate average action-potential conduction velocity in meters/second."""

    spike_times_ms, spike_positions_um = rasterize(
        result,
        threshold_mV=threshold_mV,
        min_distance_ms=min_distance_ms,
        threshold=threshold,
        min_distance=min_distance,
    )
    if spike_times_ms.size == 0:
        return 0.0

    time_s = np.asarray(spike_times_ms, dtype=float) * 1e-3
    position_m = np.asarray(spike_positions_um, dtype=float) * 1e-6

    sort_idx = np.argsort(time_s)
    time_s = time_s[sort_idx]
    position_m = position_m[sort_idx]

    x0 = position_m[0]
    recorded_positions_m = recorded_positions_um(result) * 1e-6
    x_min = float(np.min(recorded_positions_m))
    x_max = float(np.max(recorded_positions_m))

    mask_forward = (position_m >= x0) & (position_m <= x_max)
    v_forward = _fit_velocity(time_s[mask_forward], position_m[mask_forward])

    mask_backward = (position_m <= x0) & (position_m >= x_min)
    backward_t = time_s[mask_backward]
    backward_x = position_m[mask_backward]
    v_backward = 0.0
    if backward_t.shape[0] >= 2:
        order = np.argsort(backward_t)
        v_backward = _fit_velocity(backward_t[order], backward_x[order][::-1])

    velocities = [
        velocity for velocity in (v_forward, v_backward) if velocity != 0.0
    ]
    if not velocities:
        return 0.0
    return float(np.mean(velocities))


def average_velocity(
    result: SimResult,
    *,
    threshold_mV: Any = -10.0,
    min_distance_ms: Any = 1.0,
    threshold: Any | None = None,
    min_distance: Any | None = None,
) -> float:
    """Alias for ``conduction_velocity``."""

    return conduction_velocity(
        result,
        threshold_mV=threshold_mV,
        min_distance_ms=min_distance_ms,
        threshold=threshold,
        min_distance=min_distance,
    )


def peak_voltage(result: SimResult) -> np.ndarray:
    """Return the peak membrane voltage in mV for each recorded position."""

    return np.max(_vm_matrix(result), axis=0)


def _fit_velocity(time_s: np.ndarray, position_m: np.ndarray) -> float:
    if time_s.shape[0] < 2:
        return 0.0
    order = np.argsort(time_s)
    coefficient = np.polyfit(time_s[order], position_m[order], 1)
    return float(coefficient[0])


__all__ = [
    "average_velocity",
    "conduction_velocity",
    "peak_voltage",
    "rasterize",
    "recorded_positions_um",
]
