"""Post-hoc activation criteria for simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypeAlias

import numpy as np

from axonscope.results.single import SimResult
from axonscope.utils import units


ActivationPositions: TypeAlias = Literal["all", "center", "distal"] | Sequence[Any]


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
    positions
        Recorded positions to inspect. Use ``"all"``, ``"center"``,
        ``"distal"``, or a sequence of physical positions.
    indices
        Original axon compartment indices to inspect. If the result was
        spatially filtered, these indices are mapped through
        ``result.record_indices``.
    require_propagation
        Reserved semantic flag for the future observer implementation. In this
        post-hoc implementation, pass distal positions or indices explicitly to
        enforce a propagation-like check.
    """

    threshold: Any = -20.0
    blanking: Any = 0.0
    positions: ActivationPositions = "all"
    indices: Sequence[int] | None = None
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

        if self.indices is not None:
            if not isinstance(self.positions, str) or self.positions != "all":
                raise ValueError("Provide either positions or indices, not both.")
            requested = tuple(int(index) for index in self.indices)
            if not requested:
                raise ValueError("indices must contain at least one entry.")
            if any(index < 0 for index in requested):
                raise ValueError("indices must be non-negative.")
            columns = [
                int(np.flatnonzero(original_indices == index)[0])
                for index in requested
                if np.any(original_indices == index)
            ]
            if len(columns) != len(requested):
                missing = sorted(set(requested).difference(set(original_indices)))
                raise ValueError(f"indices are not present in this result: {missing}.")
            selected_columns = np.asarray(columns, dtype=int)
            return (
                selected_columns,
                original_indices[selected_columns],
                positions_um[selected_columns],
            )

        if isinstance(self.positions, str):
            selected_columns = self._columns_from_position_mode(
                self.positions,
                positions_um,
            )
        else:
            target_um = units.to_um_array(self.positions)
            selected_columns = np.asarray(
                [int(np.argmin(np.abs(positions_um - target))) for target in target_um],
                dtype=int,
            )
            selected_columns = np.unique(selected_columns)

        return (
            selected_columns,
            original_indices[selected_columns],
            positions_um[selected_columns],
        )

    def _columns_from_position_mode(
        self,
        mode: str,
        positions_um: np.ndarray,
    ) -> np.ndarray:
        if mode == "all":
            return np.arange(positions_um.shape[0], dtype=int)
        if mode == "center":
            center_um = 0.5 * (float(np.min(positions_um)) + float(np.max(positions_um)))
            return np.asarray([int(np.argmin(np.abs(positions_um - center_um)))], dtype=int)
        if mode == "distal":
            return np.asarray([int(np.argmax(positions_um))], dtype=int)
        raise ValueError(
            "positions must be 'all', 'center', 'distal', or a sequence of positions."
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
    "ActivationPositions",
    "detect_activation",
]
