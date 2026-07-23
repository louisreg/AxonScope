"""Typed, reusable protocol updates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from axonfleet.axon_instance import AxonInstance
from axonfleet.dispatcher.numeric_axis import ExtracellularWaveformAxisInput
from axonfleet.identifiers import DriveId
from axonfleet.stimulation import Stimulus


@dataclass(frozen=True)
class _PreparedExtracellularWaveformUpdate:
    waveform: Callable[[Any], Stimulus]
    source_drive_waveforms: tuple[tuple[Stimulus, ...], ...]
    selected_drive_indices: tuple[int, ...]

    def numeric_axis_input(
        self,
        values: tuple[Any, ...],
    ) -> ExtracellularWaveformAxisInput:
        return ExtracellularWaveformAxisInput(
            waveforms=tuple(self._waveform_for(value) for value in values),
            source_drive_waveforms=self.source_drive_waveforms,
            selected_drive_indices=self.selected_drive_indices,
        )

    def _waveform_for(self, value: Any) -> Stimulus:
        stimulus = self.waveform(value)
        if not isinstance(stimulus, Stimulus):
            raise TypeError("waveform(value) must return an axonfleet.Stimulus.")
        return stimulus.as_unit("ampere")


class ExtracellularWaveformUpdate:
    """Update one extracellular drive with one waveform per sweep value.

    The waveform factory is evaluated once per sampled value and materialized
    as typed numeric-axis input. Source axons, drives, stimulations, and stimuli
    remain immutable across executions.
    """

    def __init__(
        self,
        waveform: Callable[[Any], Stimulus],
        *,
        drive_id: DriveId | None = None,
    ) -> None:
        if not callable(waveform):
            raise TypeError("waveform must be callable.")
        if drive_id is not None and not isinstance(drive_id, DriveId):
            raise TypeError("drive_id must be an axonfleet.DriveId or None.")
        self.waveform = waveform
        self.drive_id = drive_id

    def prepare_numeric_axis(
        self,
        pool: tuple[Any, ...],
    ) -> _PreparedExtracellularWaveformUpdate:
        """Prepare immutable source-drive rows once for all value chunks."""

        source_drive_waveforms: list[tuple[Stimulus, ...]] = []
        selected_drive_indices: list[int] = []
        drive_count: int | None = None
        for row in pool:
            stimulation = row.extracellular_stimulation
            selected_drive = self._drive_for(row)
            row_drive_count = len(stimulation.drives)
            if drive_count is None:
                drive_count = row_drive_count
            elif row_drive_count != drive_count:
                raise ValueError(
                    "compact numeric-axis execution requires one stable drive "
                    "count across source rows."
                )
            source_drive_waveforms.append(
                tuple(drive.stimulus.as_unit("ampere") for drive in stimulation.drives)
            )
            selected_drive_indices.append(
                next(
                    index
                    for index, drive in enumerate(stimulation.drives)
                    if drive is selected_drive
                )
            )
        return _PreparedExtracellularWaveformUpdate(
            waveform=self.waveform,
            source_drive_waveforms=tuple(source_drive_waveforms),
            selected_drive_indices=tuple(selected_drive_indices),
        )

    def __call__(self, row: Any, value: Any) -> Any:
        """Apply through the generic row-update contract when pooling is unavailable."""

        drive = self._drive_for(row)
        stimulation = row.extracellular_stimulation
        updated = stimulation.replace_drive(
            drive.id,
            stimulus=self._waveform_for(value),
        )
        row.add_extracellular_stimulation(stimulation=updated, replace=True)
        return row

    def _waveform_for(self, value: Any) -> Stimulus:
        stimulus = self.waveform(value)
        if not isinstance(stimulus, Stimulus):
            raise TypeError("waveform(value) must return an axonfleet.Stimulus.")
        return stimulus.as_unit("ampere")

    def _drive_for(self, row: Any):
        if not isinstance(row, AxonInstance):
            raise TypeError(
                "ExtracellularWaveformUpdate requires AxonInstance population rows."
            )
        stimulation = row.extracellular_stimulation
        if stimulation is None:
            raise ValueError("population row has no extracellular stimulation.")
        if self.drive_id is not None:
            return stimulation[self.drive_id]
        if len(stimulation.drives) != 1:
            raise ValueError(
                "drive_id is required when a population row has multiple drives."
            )
        return stimulation.drives[0]


__all__ = ["ExtracellularWaveformUpdate"]
