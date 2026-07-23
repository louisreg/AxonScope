"""Activation event detection shared by analyses and protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np

from axonfleet.positions import ALL, PositionSelector
from axonfleet.utils import units


class _PeakEvent(TypedDict):
    peak_mV: float | None
    peak_time_ms: float | None
    peak_index: int | None


@dataclass(frozen=True)
class ActivationEvent:
    """Compact post-hoc activation result."""

    activated: bool
    first_time_ms: float | None = None
    first_position_um: float | None = None
    first_index: int | None = None
    peak_mV: float | None = None
    peak_time_ms: float | None = None
    peak_index: int | None = None


def _detect_activation(
    result: Any,
    *,
    threshold: Any = -20.0,
    blanking: Any = 0.0,
    target: PositionSelector = ALL,
) -> ActivationEvent:
    """Detect one threshold-crossing event in a recorded axon result."""

    threshold_mV = units.to_mV(threshold)
    blanking_ms = units.to_ms(blanking)
    if blanking_ms < 0.0:
        raise ValueError("blanking must be non-negative.")

    vm = result.voltage_values(unit="millivolt")
    if vm.ndim != 2:
        raise ValueError(f"result.Vm must be 2D (time, position), got {vm.shape}.")
    time_ms = result.time_values(unit="millisecond")
    if time_ms.ndim != 1:
        raise ValueError(f"result.t must be 1D, got {time_ms.shape}.")
    if time_ms.shape[0] != vm.shape[0]:
        raise ValueError("result.t length must match result.Vm time dimension.")

    columns, original_indices, positions_um = _selected_columns(result, target=target)
    selected_vm = vm[:, columns]
    eligible_times = time_ms >= blanking_ms
    if not np.any(eligible_times):
        peak = _peak_event(selected_vm, time_ms, original_indices)
        return ActivationEvent(activated=False, **peak)

    eligible_vm = selected_vm[eligible_times]
    eligible_time_ms = time_ms[eligible_times]
    crossing = eligible_vm >= threshold_mV
    peak = _peak_event(eligible_vm, eligible_time_ms, original_indices)
    if not np.any(crossing):
        return ActivationEvent(activated=False, **peak)

    time_row, local_col = np.argwhere(crossing)[0]
    return ActivationEvent(
        activated=True,
        first_time_ms=float(eligible_time_ms[time_row]),
        first_position_um=float(positions_um[local_col]),
        first_index=int(original_indices[local_col]),
        **peak,
    )


def _selected_columns(
    result: Any,
    *,
    target: PositionSelector,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vm = result.voltage_values(unit="millivolt")
    axis = result.recorded_axis
    positions_um = axis.position_values(unit="micrometer")
    if positions_um.shape != (vm.shape[1],):
        raise ValueError("recorded positions must match result.Vm columns.")

    original_indices = axis.index_values()
    if not isinstance(target, PositionSelector):
        raise TypeError("target must be an axonfleet.positions.PositionSelector.")
    selected_columns = target.columns(
        positions_um=positions_um,
        original_indices=original_indices,
    )
    return (
        selected_columns,
        original_indices[selected_columns],
        positions_um[selected_columns],
    )


def _peak_event(
    vm: np.ndarray,
    time_ms: np.ndarray,
    original_indices: np.ndarray,
) -> _PeakEvent:
    if vm.size == 0:
        return {
            "peak_mV": None,
            "peak_time_ms": None,
            "peak_index": None,
        }
    time_row, local_col = np.unravel_index(int(np.argmax(vm)), vm.shape)
    return {
        "peak_mV": float(vm[time_row, local_col]),
        "peak_time_ms": float(time_ms[time_row]),
        "peak_index": int(original_indices[local_col]),
    }


__all__ = [
    "ActivationEvent",
]
