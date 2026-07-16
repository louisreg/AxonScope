"""Typed dynamic inputs for one numeric execution axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from axonscope.stimulation import Stimulus


class NumericAxisInput(Protocol):
    """Backend-neutral dynamic input carried by a numeric execution axis."""

    @property
    def size(self) -> int:
        """Number of ordered values represented by this input."""

    @property
    def dispatch_signature(self) -> tuple[object, ...]:
        """Static signature contribution used for dispatch grouping."""

    def for_source_indices(
        self,
        source_indices: tuple[int, ...],
    ) -> "NumericAxisInput":
        """Select source rows for one dispatch compatibility group."""


@dataclass(frozen=True)
class ExtracellularWaveformAxisInput:
    """One selected waveform plus static drive rows for each axis value."""

    waveforms: tuple[Stimulus, ...]
    source_drive_waveforms: tuple[tuple[Stimulus, ...], ...]
    selected_drive_indices: tuple[int, ...]
    kind: Literal["extracellular_waveform"] = "extracellular_waveform"

    def __post_init__(self) -> None:
        if not self.waveforms:
            raise ValueError("numeric axis must contain at least one waveform.")
        if not all(isinstance(waveform, Stimulus) for waveform in self.waveforms):
            raise TypeError("waveforms must contain only axonscope.Stimulus values.")
        if not self.source_drive_waveforms:
            raise ValueError("numeric axis must describe at least one source row.")
        drive_count = len(self.source_drive_waveforms[0])
        if drive_count < 1:
            raise ValueError("each numeric-axis source row must contain at least one drive.")
        if any(len(row) != drive_count for row in self.source_drive_waveforms):
            raise ValueError("numeric-axis source rows must have one stable drive count.")
        if not all(
            isinstance(waveform, Stimulus)
            for row in self.source_drive_waveforms
            for waveform in row
        ):
            raise TypeError("source drive waveforms must be axonscope.Stimulus values.")
        if len(self.selected_drive_indices) != len(self.source_drive_waveforms):
            raise ValueError("selected drive indices must contain one index per source row.")
        if any(index < 0 or index >= drive_count for index in self.selected_drive_indices):
            raise ValueError("selected drive index is outside the source drive table.")

    @property
    def size(self) -> int:
        return len(self.waveforms)

    @property
    def source_size(self) -> int:
        return len(self.source_drive_waveforms)

    @property
    def drive_count(self) -> int:
        return len(self.source_drive_waveforms[0])

    @property
    def dispatch_signature(self) -> tuple[object, ...]:
        return (self.kind, self.size, self.drive_count)

    def for_source_indices(
        self,
        source_indices: tuple[int, ...],
    ) -> "ExtracellularWaveformAxisInput":
        if source_indices == tuple(range(self.source_size)):
            return self
        return ExtracellularWaveformAxisInput(
            waveforms=self.waveforms,
            source_drive_waveforms=tuple(
                self.source_drive_waveforms[index] for index in source_indices
            ),
            selected_drive_indices=tuple(
                self.selected_drive_indices[index] for index in source_indices
            ),
        )


__all__ = [
    "ExtracellularWaveformAxisInput",
    "NumericAxisInput",
]
