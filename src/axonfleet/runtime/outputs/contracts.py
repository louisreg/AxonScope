"""Internal runtime output-plan descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from axonfleet.analysis.definitions import (
    Activation,
    ConductionVelocity,
    Latency,
    SpikeCount,
    VmRaster,
)
from axonfleet.solvers.options import BatchOptions, BatchRecording

OutputSink = Literal[
    "vm",
    "none",
    "activation",
    "first_crossing",
    "spike_summary",
    "spike_events",
    "vm_raster",
]


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
    if recording_mode == "none" and observers_are_compact_latency_compatible(observers):
        return "first_crossing"
    if recording_mode == "none" and observers_are_compact_spike_compatible(observers):
        return (
            "spike_events"
            if any(observer.max_spikes is not None for observer in observers)
            else "spike_summary"
        )
    if recording_mode == "none" and any(
        isinstance(observer, SpikeCount) and observer.max_spikes is not None
        for observer in observers
    ):
        return "unsupported_observer_only"
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


def observers_are_compact_latency_compatible(
    observers: tuple[Any, ...] | None,
) -> bool:
    """Return whether observers need only one first-crossing step per row."""

    return bool(observers) and all(isinstance(observer, Latency) for observer in observers)


def observers_are_compact_spike_compatible(
    observers: tuple[Any, ...] | None,
) -> bool:
    """Return whether observers need bounded per-probe spike summaries."""

    if not observers or not all(isinstance(observer, SpikeCount) for observer in observers):
        return False
    capacities = {observer.max_spikes for observer in observers}
    return len(capacities) == 1


def vm_raster_definitions(observers: tuple[Any, ...] | None) -> tuple[Any, ...]:
    """Return observer definitions supported by solver-side VmRaster."""

    if observers is None:
        return ()
    return tuple(
        observer
        for observer in observers
        if isinstance(
            observer,
            (Activation, Latency, SpikeCount, VmRaster, ConductionVelocity),
        )
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
        _maybe_millivolt(getattr(observer, "reset_threshold", None)),
        _maybe_millisecond(getattr(observer, "refractory", None)),
        getattr(observer, "max_spikes", None),
        bool(getattr(observer, "allow_all_compartments", False)),
        int(getattr(observer, "every_n_steps", 1)),
    )


def _maybe_millivolt(value: Any) -> float | None:
    if value is None:
        return None
    from axonfleet.utils import units

    return float(units.to_mV(value))


def _maybe_millisecond(value: Any) -> float | None:
    if value is None:
        return None
    from axonfleet.utils import units

    return float(units.to_ms(value))


__all__ = [
    "OutputPlan",
    "OutputSink",
    "observer_definition_signature",
    "observer_output_label",
    "observers_are_compact_activation_compatible",
    "observers_are_compact_latency_compatible",
    "observers_are_compact_spike_compatible",
    "observers_are_vm_raster_compatible",
    "vm_raster_definitions",
]
