"""JAX batch result synchronization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from axonscope.benchmarking import benchmark_span
from axonscope.dispatcher.plan import DispatchGroup
from axonscope.runtime.jax.cable_geometry import Array
from axonscope.runtime.jax.recording.observer import (
    PendingVmRasterObservation,
    finalize_vm_raster_state,
    trim_pending_vm_raster_observation,
)
from axonscope.runtime.result_assembly import (
    trim_observations_batch,
    trim_recordings_batch,
)


@dataclass(frozen=True)
class BatchKernelResult:
    """Raw batched solver-kernel output before packaging public simulations."""

    Vm: Array | None
    t: Array
    recordings: dict[str, Any] | None = None
    observations: dict[str, object] | None = None
    pending_observation: PendingVmRasterObservation | None = None


def trim_batch_kernel_result(
    out: BatchKernelResult,
    *,
    batch_size: int,
) -> BatchKernelResult:
    """Drop backend-only padded batch rows before public result assembly."""

    size = int(batch_size)
    with benchmark_span(
        "results.trim_padded_batch",
        batch_size=size,
        has_vm=out.Vm is not None,
        has_recordings=out.recordings is not None,
        has_observations=out.observations is not None,
        has_pending_observation=out.pending_observation is not None,
    ):
        Vm = None if out.Vm is None else out.Vm[:size]
        recordings = trim_recordings_batch(out.recordings, batch_size=size)
        observations = trim_observations_batch(out.observations, batch_size=size)
        pending = (
            None
            if out.pending_observation is None
            else trim_pending_vm_raster_observation(
                out.pending_observation,
                batch_size=size,
            )
        )
    return BatchKernelResult(
        Vm=Vm,
        t=out.t,
        recordings=recordings,
        observations=observations,
        pending_observation=pending,
    )


def batch_wait_target(out: BatchKernelResult) -> Any:
    """Return a JAX/NumPy object that synchronizes a batch kernel result."""

    if out.Vm is not None:
        return out.Vm
    if out.recordings:
        first_recording = next(iter(out.recordings.values()))
        if isinstance(first_recording, dict):
            return next(iter(first_recording.values()))
        return first_recording
    if out.pending_observation is not None:
        return out.pending_observation.state
    if not out.observations:
        raise RuntimeError("batch kernel produced neither Vm nor observations.")
    first = next(iter(out.observations.values()))
    if hasattr(first, "words"):
        return first.words
    return first.values


def finalize_pending_batch_observation(
    out: BatchKernelResult,
    *,
    group: DispatchGroup,
    mode: str,
) -> BatchKernelResult:
    """Finalize observer output after the explicit group-level device wait."""

    pending = out.pending_observation
    if pending is None:
        return out
    with benchmark_span(
        "kernel.finalize_observer",
        mode=mode,
        observer="vm_raster",
        group_id=group.group_id,
        group_size=group.size,
        synchronized_before_finalize=True,
        wait_span="kernel.wait",
    ):
        observations = cast(
            dict[str, object],
            finalize_vm_raster_state(
                pending.plan,
                pending.state,
                nt=pending.nt,
                dt_ms=pending.dt_ms,
                synchronize=False,
                materialize_words=False,
            ),
        )
    return BatchKernelResult(
        Vm=out.Vm,
        t=out.t,
        recordings=out.recordings,
        observations=observations,
        pending_observation=None,
    )


__all__ = [
    "BatchKernelResult",
    "batch_wait_target",
    "finalize_pending_batch_observation",
    "trim_batch_kernel_result",
]
