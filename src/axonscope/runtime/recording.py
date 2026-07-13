"""Runtime-neutral lowering from public recording plans to batch options."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from axonscope.recording import Recording, RecordingPlan, RecordingSpatial
from axonscope.solvers.options import BatchOptions, BatchRecording


def recording_plan_from_recording(recording: Recording | RecordingPlan) -> RecordingPlan:
    """Return a runtime-neutral plan from a public recording-like value."""

    if isinstance(recording, RecordingPlan):
        return recording
    if isinstance(recording, Recording):
        return recording.to_plan()
    raise TypeError("recording must be an axonscope.Recording or RecordingPlan value.")


def batch_recording_from_recording_plan(plan: RecordingPlan) -> BatchRecording:
    """Lower a runtime-neutral recording plan to batch recording options."""

    if not plan.voltage and not plan.wants_observables:
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
    """Merge a recording plan into existing batch execution options."""

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


def lower_batch_recording_options(
    group: Any,
    options: BatchOptions,
    *,
    observers: tuple[Any, ...] | None,
) -> BatchOptions:
    """Return kernel recording options after padding/observer lowering."""

    if not group.has_padding:
        return options
    if options.recording.mode == "none" and observers is not None:
        return options
    if row_recording_indices_for_group(group, options.recording) is not None:
        return options
    return options if options.recording.mode == "full" else _replace_full_recording(options)


def row_recording_indices_for_group(
    group: Any,
    recording: BatchRecording,
) -> np.ndarray | None:
    """Return row-aware retained Vm indices for padded batch groups.

    ``BatchRecording`` is shape-only and normally resolves indices against one
    ``Nx``. Padded mixed-diameter groups need the same number of retained
    columns per row, but those columns must be selected against each row's
    original compartment count.
    """

    if not getattr(group, "has_padding", False):
        return None
    if recording.mode not in {"center", "probes", "indices"}:
        return None

    rows: list[np.ndarray] = []
    width: int | None = None
    for item in group.items:
        row = np.asarray(
            recording.indices_for(int(item.solver_axon.n_compartments)),
            dtype=np.int32,
        )
        if row.ndim != 1:
            return None
        if width is None:
            width = int(row.shape[0])
        elif int(row.shape[0]) != width:
            return None
        rows.append(row)
    if not rows or width is None or width < 1:
        return None
    return np.stack(rows, axis=0)


def cohort_original_indices(cohort: Any) -> np.ndarray:
    """Return row-aware original compartment indices, with -1 for padding."""

    rows = np.full((cohort.size, cohort.nx), -1, dtype=np.int32)
    for row_index, solver_axon in enumerate(cohort.solver_axons):
        original_nx = int(solver_axon.n_compartments)
        rows[row_index, :original_nx] = np.arange(original_nx, dtype=np.int32)
    return rows


def _replace_full_recording(options: BatchOptions) -> BatchOptions:
    return replace(options, recording=BatchRecording.full())


__all__ = [
    "batch_options_from_recording",
    "batch_options_from_recording_plan",
    "batch_recording_from_recording_plan",
    "cohort_original_indices",
    "lower_batch_recording_options",
    "recording_plan_from_recording",
    "row_recording_indices_for_group",
]
