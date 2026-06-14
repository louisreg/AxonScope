"""Post-hoc activation criteria for simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from axonscope.positions import ALL, PositionSelector
from axonscope.results.single import SimResult
from axonscope.utils import units


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


@dataclass(frozen=True)
class ActivationCriterion:
    """Detect whether a voltage trace satisfies an activation criterion.

    Parameters
    ----------
    threshold
        Membrane voltage threshold. Plain numbers are interpreted as millivolts.
    blanking
        Initial time interval ignored by the detector. Plain numbers are
        interpreted as milliseconds.
    target
        Typed recorded-position selector from `axs.positions`, such as
        `axs.positions.ALL`, `axs.positions.DISTAL`,
        `axs.positions.At(...)`, or `axs.positions.Indices(...)`.
    require_propagation
        Reserved semantic flag for the future observer implementation. In this
        post-hoc implementation, pass distal positions or indices explicitly to
        enforce a propagation-like check.
    """

    threshold: Any = -20.0
    blanking: Any = 0.0
    target: PositionSelector = ALL
    require_propagation: bool = False

    def evaluate(self, result: SimResult) -> ActivationEvent:
        """Evaluate the criterion on a simulation result."""

        threshold_mV = units.to_mV(self.threshold)
        blanking_ms = units.to_ms(self.blanking)
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

        columns, original_indices, positions_um = self._selected_columns(result)
        selected_vm = vm[:, columns]
        eligible_times = time_ms >= blanking_ms
        if not np.any(eligible_times):
            peak = self._peak_event(selected_vm, time_ms, original_indices, positions_um)
            return ActivationEvent(activated=False, **peak)

        eligible_vm = selected_vm[eligible_times]
        eligible_time_ms = time_ms[eligible_times]
        crossing = eligible_vm >= threshold_mV
        peak = self._peak_event(
            eligible_vm,
            eligible_time_ms,
            original_indices,
            positions_um,
        )
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
        self,
        result: SimResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vm = result.voltage_values(unit="millivolt")
        positions_um = result.position_values(unit="micrometer")
        if positions_um.shape != (vm.shape[1],):
            raise ValueError("recorded positions must match result.Vm columns.")

        if result.record_indices is None:
            original_indices = np.arange(vm.shape[1], dtype=int)
        else:
            original_indices = np.asarray(result.record_indices, dtype=int)

        if not isinstance(self.target, PositionSelector):
            raise TypeError("target must be an axonscope.positions.PositionSelector.")
        selected_columns = self.target.columns(
            positions_um=positions_um,
            original_indices=original_indices,
        )

        return (
            selected_columns,
            original_indices[selected_columns],
            positions_um[selected_columns],
        )

    @staticmethod
    def _peak_event(
        vm: np.ndarray,
        time_ms: np.ndarray,
        original_indices: np.ndarray,
        positions_um: np.ndarray,
    ) -> dict[str, float | int | None]:
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


def detect_activation(result: SimResult, **kwargs: Any) -> ActivationEvent:
    """Convenience wrapper around ``ActivationCriterion(...).evaluate(result)``."""

    return ActivationCriterion(**kwargs).evaluate(result)


__all__ = [
    "ActivationCriterion",
    "ActivationEvent",
    "detect_activation",
]
