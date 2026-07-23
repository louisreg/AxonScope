"""Shared test-only result fixtures."""

from __future__ import annotations

from typing import Any

from axonfleet.results.axes import RecordedAxis
from axonfleet.results.common import SingleAxonResultMixin
from axonfleet.signals import Signal


class FakeSingleAxonResult(SingleAxonResultMixin):
    """Small one-axon result object used by tests."""

    axon: Any
    t: Any
    recordings: dict[str, Any] | None = None
    record_indices: tuple[int, ...] | None = None

    def __init__(
        self,
        axon: Any,
        Vm: Any | None = None,
        t: Any | None = None,
        *,
        recordings: dict[str, Any] | None = None,
        record_indices: tuple[int, ...] | None = None,
    ) -> None:
        if t is None:
            raise TypeError("FakeSingleAxonResult requires a time vector `t`.")
        payload = {} if recordings is None else dict(recordings)
        if Vm is not None:
            payload["Vm"] = Vm
        self.axon = axon
        self.t = t
        self.recordings = payload or None
        self.record_indices = record_indices

    @property
    def Vm(self) -> Any:
        from axonfleet.signals import MEMBRANE_VOLTAGE

        try:
            return self.signal(MEMBRANE_VOLTAGE)
        except KeyError as exc:
            raise AttributeError("this fake result does not contain Vm.") from exc

    def signal(self, signal: Any) -> Any:
        if isinstance(signal, str):
            raise TypeError("signal must use axonfleet.signals values, not strings.")
        if not isinstance(signal, Signal):
            raise TypeError("signal must be an axonfleet.Signal descriptor.")
        if self.recordings is None or signal.result_key not in self.recordings:
            raise KeyError(signal.result_key)
        value = self.recordings[signal.result_key]
        if isinstance(value, dict):
            raise TypeError(
                f"recordings[{signal.result_key!r}] is a grouped recording, not an array."
            )
        return value

    @property
    def recorded_axis(self) -> RecordedAxis:
        return RecordedAxis.from_result(self)
