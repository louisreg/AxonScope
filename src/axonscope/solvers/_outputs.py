"""Internal solver output payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from axonscope.results.axes import RecordedAxis
from axonscope.signals import Signal
from axonscope.utils import units

if TYPE_CHECKING:
    from axonscope.axon_instance import AxonInstance
    from axonscope.axons.axon import Axon
    from axonscope.recording import Recording


ResultArray: TypeAlias = Any
RecordingValue: TypeAlias = ResultArray | dict[str, ResultArray]
RecordingDict: TypeAlias = dict[str, RecordingValue]
ObservationDict: TypeAlias = dict[str, Any]


def _normalize_recordings(
    recordings: RecordingDict | None,
    Vm: ResultArray | None,
) -> RecordingDict | None:
    normalized: RecordingDict = {}
    if recordings is not None:
        normalized.update(recordings)
    if Vm is not None:
        normalized["Vm"] = Vm
    return normalized or None


@dataclass(init=False)
class SolverOutput:
    """Internal output returned by scalar solver calls.

    This is a solver/backend payload, not the public result model. Public
    execution converts solver outputs through dispatcher records into
    ``AxonSimulationResult``.
    """

    axon: Axon
    t: ResultArray
    diagnostics: dict[str, Any] | None = None
    recordings: RecordingDict | None = None
    observations: ObservationDict | None = None
    recording: Recording | None = None
    record_indices: tuple[int, ...] | None = None
    final_state: Any | None = None
    simulation: AxonInstance | None = None

    def __init__(
        self,
        axon: Axon,
        Vm: ResultArray | None = None,
        t: ResultArray | None = None,
        *,
        diagnostics: dict[str, Any] | None = None,
        recordings: RecordingDict | None = None,
        observations: ObservationDict | None = None,
        recording: Recording | None = None,
        record_indices: tuple[int, ...] | None = None,
        final_state: Any | None = None,
        simulation: AxonInstance | None = None,
    ) -> None:
        if t is None:
            raise TypeError("SolverOutput requires a time vector `t`.")
        self.axon = axon
        self.t = t
        self.diagnostics = diagnostics
        self.recordings = _normalize_recordings(recordings, Vm)
        self.observations = observations
        self.recording = recording
        self.record_indices = record_indices
        self.final_state = final_state
        self.simulation = simulation

    @property
    def Vm(self) -> ResultArray:
        """Membrane voltage recording, equivalent to ``recordings["Vm"]``."""

        from axonscope.signals import MEMBRANE_VOLTAGE

        try:
            return self.signal(MEMBRANE_VOLTAGE)
        except KeyError as exc:
            raise AttributeError("this SolverOutput does not contain a Vm recording.") from exc

    @Vm.setter
    def Vm(self, value: ResultArray) -> None:
        """Update the membrane-voltage recording in place."""

        recordings = {} if self.recordings is None else dict(self.recordings)
        recordings["Vm"] = value
        self.recordings = recordings

    def signal(self, signal: Any) -> ResultArray:
        """Return one recorded signal by public signal descriptor."""

        if isinstance(signal, str):
            raise TypeError("signal must use axonscope.signals values, not strings.")
        if not isinstance(signal, Signal):
            raise TypeError("signal must be an axonscope.Signal descriptor.")
        if self.recordings is None or signal.result_key not in self.recordings:
            raise KeyError(f"signal {signal.id!s} is not available in this output.")
        value = self.recordings[signal.result_key]
        if isinstance(value, dict):
            raise TypeError(
                f"recordings[{signal.result_key!r}] is a grouped recording, not an array."
            )
        return value

    @property
    def recorded_axis(self) -> RecordedAxis:
        """Recorded intrinsic spatial axis for ``Vm`` columns."""

        return RecordedAxis.from_result(self)

    def time_values(self, *, unit: Any = "millisecond") -> np.ndarray:
        """Return solver output times as plain values in `unit`."""

        unit_label = units.unit_label(unit) or "millisecond"
        return units.to_array(
            units.Q_(np.asarray(self.t, dtype=float), "millisecond"),
            unit_label,
            dtype=float,
        )

    def position_values(self, *, unit: Any = "micrometer") -> np.ndarray:
        """Return recorded axon positions as plain values in `unit`."""

        return self.recorded_axis.position_values(unit=unit)

    def voltage_values(self, *, unit: Any = "millivolt") -> np.ndarray:
        """Return membrane voltages as plain values in `unit`."""

        unit_label = units.unit_label(unit) or "millivolt"
        vm = np.asarray(self.Vm, dtype=float)
        return units.to_array(units.Q_(vm, "millivolt"), unit_label, dtype=float)


__all__ = [
    "ObservationDict",
    "RecordingDict",
    "RecordingValue",
    "ResultArray",
    "SolverOutput",
]
