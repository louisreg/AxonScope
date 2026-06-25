"""JAX lowering for backend-neutral recording plans."""

from __future__ import annotations

from dataclasses import replace

from axonscope.recording import Recording, RecordingPlan, RecordingSpatial
from axonscope.solvers.options import BatchOptions, BatchRecording


def recording_plan_from_recording(recording: Recording | RecordingPlan) -> RecordingPlan:
    """Return a backend-neutral plan from a public recording-like value."""

    if isinstance(recording, RecordingPlan):
        return recording
    if isinstance(recording, Recording):
        return recording.to_plan()
    raise TypeError("recording must be an axonscope.Recording or RecordingPlan value.")


def batch_recording_from_recording_plan(plan: RecordingPlan) -> BatchRecording:
    """Lower a backend-neutral recording plan to JAX batch recording options."""

    if plan.wants_observables:
        raise NotImplementedError("pool recording currently supports Vm only.")
    if not plan.voltage:
        return BatchRecording.none()
    if plan.positions_um is not None:
        raise NotImplementedError(
            "position-based batch recording is not wired yet; "
            "use center/probes/indices/full."
        )
    if plan.sample_dt_ms is not None or plan.every_n_steps is not None:
        raise NotImplementedError("temporal recording subsampling is not wired yet.")
    if plan.spatial is RecordingSpatial.CENTER:
        return BatchRecording.center()
    if plan.spatial is RecordingSpatial.PROBES:
        return BatchRecording.probes(plan.probe_count)
    if plan.spatial is RecordingSpatial.INDICES:
        if plan.record_indices is None:
            raise ValueError("indices recording requires record_indices.")
        return BatchRecording.indices(plan.record_indices)
    return BatchRecording.full()


def batch_options_from_recording_plan(
    plan: RecordingPlan,
    *,
    batch_options: BatchOptions | None = None,
) -> BatchOptions:
    """Merge a recording plan into existing JAX batch execution options."""

    recording = batch_recording_from_recording_plan(plan)
    if batch_options is None:
        if recording.mode == "none":
            return BatchOptions.none()
        return BatchOptions(recording=recording)
    return replace(batch_options, recording=recording)


def batch_options_from_recording(
    recording: Recording | RecordingPlan | None,
    *,
    batch_options: BatchOptions | None = None,
) -> BatchOptions | None:
    """Return batch options after applying an optional public recording policy."""

    if recording is None:
        return batch_options
    plan = recording_plan_from_recording(recording)
    return batch_options_from_recording_plan(plan, batch_options=batch_options)


__all__ = [
    "batch_options_from_recording",
    "batch_options_from_recording_plan",
    "batch_recording_from_recording_plan",
    "recording_plan_from_recording",
]
