"""Internal runtime output-plan descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from axonscope.analysis.definitions import (
    Activation,
    ConductionBlock,
    ConductionVelocity,
    Latency,
)
from axonscope.solvers.options import BatchOptions, BatchRecording

OutputSink = Literal["vm", "none", "activation", "vm_raster"]


@dataclass(frozen=True)
class OutputPlan:
    """Runtime-neutral output and chunking plan.

    Public recording requests are lowered to ``BatchOptions`` at the orchestration
    boundary. Concrete runtimes then normalize those options into this internal
    plan so full Vm, sampled Vm, and observer-only VmRaster routes share one
    output descriptor without coupling input lowering to observer details.
    """

    recording: BatchRecording
    time_chunk_steps: int | None
    sink: OutputSink
    row_record_indices: Any | None = None

    @classmethod
    def from_batch_options(
        cls,
        options: BatchOptions,
        *,
        observers: tuple[object, ...] | None,
        row_record_indices: Any | None = None,
    ) -> "OutputPlan":
        if options.recording.mode == "none":
            sink: OutputSink = observer_output_label(observers, recording_mode="none")
        else:
            sink = "vm"
        return cls(
            recording=options.recording,
            time_chunk_steps=options.time_chunk_steps,
            sink=sink,
            row_record_indices=row_record_indices,
        )

    def to_batch_options(self) -> BatchOptions:
        """Return the structural batch options expected by result assembly."""

        return BatchOptions(
            recording=self.recording,
            time_chunk_steps=self.time_chunk_steps,
        )


def observer_output_label(
    observers: tuple[Any, ...] | None,
    *,
    recording_mode: str,
) -> str:
    """Return the selected observer output route."""

    if observers is None:
        return "none"
    if recording_mode == "none" and observers_are_compact_activation_compatible(observers):
        return "activation"
    if recording_mode == "none" and observers_are_vm_raster_compatible(observers):
        return "vm_raster"
    if recording_mode == "none":
        return "unsupported_observer_only"
    return "posthoc_from_recorded_vm"


def observers_are_vm_raster_compatible(observers: tuple[Any, ...] | None) -> bool:
    """Return whether all observer definitions can lower to VmRaster."""

    if observers is None:
        return False
    return bool(observers) and len(vm_raster_definitions(observers)) == len(observers)


def observers_are_compact_activation_compatible(
    observers: tuple[Any, ...] | None,
) -> bool:
    """Return whether observers need only one retained activation flag per row."""

    return bool(observers) and all(isinstance(observer, Activation) for observer in observers)


def vm_raster_definitions(observers: tuple[Any, ...] | None) -> tuple[Any, ...]:
    """Return observer definitions supported by solver-side VmRaster."""

    if observers is None:
        return ()
    return tuple(
        observer
        for observer in observers
        if isinstance(observer, (Activation, Latency, ConductionBlock, ConductionVelocity))
    )


def observer_definition_signature(observer: Any) -> tuple[Any, ...]:
    """Return a runtime-neutral cache signature for one observer definition."""

    signal = getattr(observer, "signal", None)
    signal_id = getattr(signal, "id", repr(signal))
    target = getattr(observer, "target", None)
    return (
        type(observer).__module__,
        type(observer).__qualname__,
        str(getattr(observer, "name", "")),
        str(signal_id),
        repr(target),
        _maybe_millivolt(getattr(observer, "threshold", None)),
        _maybe_millisecond(getattr(observer, "blanking", None)),
    )


def _maybe_millivolt(value: Any) -> float | None:
    if value is None:
        return None
    from axonscope.utils import units

    return float(units.to_mV(value))


def _maybe_millisecond(value: Any) -> float | None:
    if value is None:
        return None
    from axonscope.utils import units

    return float(units.to_ms(value))


__all__ = [
    "OutputPlan",
    "OutputSink",
    "observer_definition_signature",
    "observer_output_label",
    "observers_are_vm_raster_compatible",
    "vm_raster_definitions",
]
