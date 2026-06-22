"""Single-axon simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

from axonscope.axons.axon import Axon
from axonscope.results.axes import RecordedAxis
from axonscope.results.common import SingleAxonResultMixin
from axonscope.signals import Signal

if TYPE_CHECKING:
    from axonscope.axon_instance import AxonInstance
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
class SimResult(SingleAxonResultMixin):
    """Internal scalar solver result.

    Public execution wrappers convert this object to ``AxonSimulationResult``.
    Recorded traces live in ``recordings``. ``recordings["Vm"]`` is the
    membrane-voltage matrix indexed as ``(time, compartment)``. Solver-side
    payloads such as packed ``VmRaster`` output live in ``observations``.
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
            raise TypeError("SimResult requires a time vector `t`.")
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
            raise AttributeError("this SimResult does not contain a Vm recording.") from exc

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
            raise KeyError(f"signal {signal.id!s} is not available in this result.")
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


__all__ = ["RecordingDict", "ResultArray", "SimResult"]
