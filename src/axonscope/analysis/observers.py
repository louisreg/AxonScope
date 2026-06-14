"""Online analysis observers for streamed membrane-voltage traces."""

from __future__ import annotations

from typing import Any

import numpy as np

from axonscope.analysis.activation import ActivationEvent
from axonscope.analysis.core import AnalysisResult, AnalysisStatus
from axonscope.positions import PositionSelector
from axonscope.utils import units


def _positions_um(values: Any) -> np.ndarray:
    positions = units.to_um_array(values, dtype=float)
    if positions.ndim != 1:
        raise ValueError(f"observer positions must be 1D, got shape {positions.shape}.")
    if positions.size == 0:
        raise ValueError("observer positions must contain at least one entry.")
    return positions


def _original_indices(values: Any | None, *, size: int) -> np.ndarray:
    if values is None:
        return np.arange(size, dtype=int)
    indices = np.asarray(values, dtype=int)
    if indices.shape != (size,):
        raise ValueError(
            "observer original_indices must contain one entry per recorded position."
        )
    return indices


def _chunk(time: Any, values: Any, *, width: int) -> tuple[np.ndarray, np.ndarray]:
    time_ms = units.to_ms_array(time, dtype=float)
    vm_mV = units.to_mV_array(values, dtype=float)
    if time_ms.ndim == 0:
        time_ms = time_ms.reshape(1)
    if vm_mV.ndim == 1:
        vm_mV = vm_mV.reshape(1, -1)
    if time_ms.ndim != 1:
        raise ValueError(f"observer time chunk must be 1D, got shape {time_ms.shape}.")
    if vm_mV.ndim != 2:
        raise ValueError(f"observer Vm chunk must be 2D, got shape {vm_mV.shape}.")
    if vm_mV.shape != (time_ms.shape[0], width):
        raise ValueError(
            "observer Vm chunk must have shape (time, recorded_position); "
            f"got {vm_mV.shape} for {time_ms.shape[0]} samples and {width} positions."
        )
    return time_ms, vm_mV


class _SelectedVmObserver:
    def __init__(
        self,
        *,
        target: PositionSelector,
        positions: Any,
        original_indices: Any | None = None,
    ) -> None:
        self.positions_um = _positions_um(positions)
        self.original_indices = _original_indices(
            original_indices,
            size=self.positions_um.shape[0],
        )
        self.columns = target.columns(
            positions_um=self.positions_um,
            original_indices=self.original_indices,
        )
        if self.columns.size == 0:
            raise ValueError("observer target selects no recorded positions.")


class ActivationObserver(_SelectedVmObserver):
    """Online observer equivalent of `axs.analysis.Activation`."""

    def __init__(
        self,
        definition: Any,
        *,
        positions: Any,
        original_indices: Any | None = None,
    ) -> None:
        super().__init__(
            target=definition.target,
            positions=positions,
            original_indices=original_indices,
        )
        self.definition = definition
        self.threshold_mV = units.to_mV(definition.threshold)
        self.blanking_ms = units.to_ms(definition.blanking)
        if self.blanking_ms < 0.0:
            raise ValueError("blanking must be non-negative.")
        self._activated = False
        self._first_time_ms: float | None = None
        self._first_position_um: float | None = None
        self._first_index: int | None = None
        self._peak_mV: float | None = None
        self._peak_time_ms: float | None = None
        self._peak_index: int | None = None
        self._saw_eligible_sample = False

    @property
    def requirements(self) -> Any:
        """Input requirements inherited from the analysis definition."""

        return self.definition.requirements

    def update(self, time: Any, values: Any) -> "ActivationObserver":
        """Consume one time chunk of membrane-voltage samples."""

        time_ms, vm_mV = _chunk(time, values, width=self.positions_um.shape[0])
        eligible = time_ms >= self.blanking_ms
        if not np.any(eligible):
            return self

        self._saw_eligible_sample = True
        selected = vm_mV[eligible][:, self.columns]
        selected_time = time_ms[eligible]

        peak_row, peak_col = np.unravel_index(int(np.argmax(selected)), selected.shape)
        peak_mV = float(selected[peak_row, peak_col])
        if self._peak_mV is None or peak_mV > self._peak_mV:
            column = int(self.columns[peak_col])
            self._peak_mV = peak_mV
            self._peak_time_ms = float(selected_time[peak_row])
            self._peak_index = int(self.original_indices[column])

        if not self._activated:
            crossing = selected >= self.threshold_mV
            if np.any(crossing):
                row, local_col = np.argwhere(crossing)[0]
                column = int(self.columns[local_col])
                self._activated = True
                self._first_time_ms = float(selected_time[row])
                self._first_position_um = float(self.positions_um[column])
                self._first_index = int(self.original_indices[column])

        return self

    def finalize(self) -> AnalysisResult:
        """Return the observer result."""

        status = (
            AnalysisStatus.VALID
            if self._saw_eligible_sample
            else AnalysisStatus.UNDETERMINED
        )
        message = "" if status is AnalysisStatus.VALID else "observer received no eligible samples."
        event = ActivationEvent(
            activated=self._activated,
            first_time_ms=self._first_time_ms,
            first_position_um=self._first_position_um,
            first_index=self._first_index,
            peak_mV=self._peak_mV,
            peak_time_ms=self._peak_time_ms,
            peak_index=self._peak_index,
        )
        return AnalysisResult(
            name=self.definition.name,
            values=np.asarray([self._activated]),
            statuses=(status,),
            messages=(message,),
            definition=self.definition,
            events=(event,),
        )


class PeakVoltageObserver(_SelectedVmObserver):
    """Online observer equivalent of `axs.analysis.PeakVoltage`."""

    def __init__(
        self,
        definition: Any,
        *,
        positions: Any,
        original_indices: Any | None = None,
    ) -> None:
        super().__init__(
            target=definition.target,
            positions=positions,
            original_indices=original_indices,
        )
        self.definition = definition
        self._peak_mV: float | None = None

    @property
    def requirements(self) -> Any:
        """Input requirements inherited from the analysis definition."""

        return self.definition.requirements

    def update(self, time: Any, values: Any) -> "PeakVoltageObserver":
        """Consume one time chunk of membrane-voltage samples."""

        _, vm_mV = _chunk(time, values, width=self.positions_um.shape[0])
        peak_mV = float(np.max(vm_mV[:, self.columns]))
        if self._peak_mV is None or peak_mV > self._peak_mV:
            self._peak_mV = peak_mV
        return self

    def finalize(self) -> AnalysisResult:
        """Return the observer result."""

        if self._peak_mV is None:
            return AnalysisResult(
                name=self.definition.name,
                values=np.asarray([np.nan]),
                statuses=(AnalysisStatus.UNDETERMINED,),
                messages=("observer received no samples.",),
                unit="millivolt",
                definition=self.definition,
            )
        return AnalysisResult(
            name=self.definition.name,
            values=np.asarray([self._peak_mV]),
            statuses=(AnalysisStatus.VALID,),
            unit="millivolt",
            definition=self.definition,
        )


__all__ = [
    "ActivationObserver",
    "PeakVoltageObserver",
]
