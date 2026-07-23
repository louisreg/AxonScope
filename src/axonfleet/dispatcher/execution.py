from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from axonfleet.axon_instance import AxonInstance
from axonfleet.axons.axon import Axon
from axonfleet.runtime.execution import (
    enqueue_batch_group,
    finalize_batch_group,
    run_batch_group,
)
from axonfleet.benchmarking import benchmark_span, record_benchmark_metadata
from axonfleet.recording import RecordingPlan
from axonfleet.dispatcher.plan import (
    DispatchGroup,
    DispatchPlan,
    build_dispatch_plan,
)
from axonfleet.dispatcher.progress import (
    DispatchProgress,
    ProgressOption,
    emit_initial_progress,
)
from axonfleet.dispatcher._records import (
    DispatchCohortRecord,
    DispatchRecord,
)
from axonfleet.solvers import BatchOptions
from axonfleet.utils import units


@dataclass(frozen=True)
class DispatchSchedulingOptions:
    """Internal dispatch scheduling knobs for evidence-gated benchmarking."""

    async_groups: bool = False
    max_pending_groups: int = 4

    def __post_init__(self) -> None:
        if int(self.max_pending_groups) < 1:
            raise ValueError("max_pending_groups must be >= 1.")


def run_pool(
    axons: Sequence[Axon | AxonInstance],
    *,
    tsim_ms: Any,
    dt_ms: Any,
    runtime_context: Any,
    batch_options: BatchOptions | None = None,
    observers: Sequence[Any] | None = None,
    recording_plan: RecordingPlan | None = None,
    progress: ProgressOption = False,
    dispatch_plan: DispatchPlan | None = None,
    scheduling_options: DispatchSchedulingOptions | None = None,
) -> tuple[DispatchRecord, ...]:
    """Run an axon pool and return raw dispatch records.

    Public code should generally call ``AxonSimulation(...).run()`` so these raw
    dispatch records are converted to public cohort results. Batched observer-
    only groups may remain a single compact record instead of one record per
    input axon. Plain numeric times are interpreted as milliseconds; Pint-like
    quantities are converted at this boundary. ``progress`` enables optional
    Rich/plain event reporting for dispatch planning, route choice, preparation,
    input lowering, cold JAX compilation points, kernel solving, and result
    assembly.
    """

    if not axons:
        raise ValueError("axons must contain at least one Axon.")
    tsim_ms = units.to_ms(tsim_ms)
    dt_ms = units.to_ms(dt_ms)
    if tsim_ms <= 0.0:
        raise ValueError("tsim_ms must be > 0.")
    if dt_ms <= 0.0:
        raise ValueError("dt_ms must be > 0.")

    with benchmark_span(
        "simulation.pool.total",
        pool_size=len(axons),
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
    ):
        return _run_pool_checked(
            axons,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            batch_options=batch_options,
            observers=tuple(observers) if observers is not None else None,
            recording_plan=recording_plan,
            progress=progress,
            runtime_context=runtime_context,
            dispatch_plan=dispatch_plan,
            scheduling_options=scheduling_options,
        )


def _run_pool_checked(
    axons: Sequence[Axon | AxonInstance],
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions | None,
    observers: tuple[Any, ...] | None,
    recording_plan: RecordingPlan | None,
    progress: ProgressOption,
    runtime_context: Any | None,
    dispatch_plan: DispatchPlan | None,
    scheduling_options: DispatchSchedulingOptions | None,
) -> tuple[DispatchRecord, ...]:
    resolved_batch_options = BatchOptions.full() if batch_options is None else batch_options
    resolved_scheduling = _resolve_dispatch_scheduling(
        runtime_context,
        scheduling_options=scheduling_options,
    )
    emit_initial_progress(progress, rows=len(axons), message="building dispatch plan")
    if dispatch_plan is None:
        plan = build_dispatch_plan(axons)
        record_benchmark_metadata(dispatch_plan_source="builder")
    else:
        plan = dispatch_plan
        record_benchmark_metadata(dispatch_plan_source="provided")
    record_benchmark_metadata(dispatch_group_count=len(plan.groups))

    results: list[DispatchRecord] = []
    seen_indices: set[int] = set()
    with DispatchProgress(progress, plan) as progress_reporter:
        if resolved_scheduling.async_groups:
            return _run_pool_async_groups(
                plan,
                tsim_ms=tsim_ms,
                dt_ms=dt_ms,
                batch_options=resolved_batch_options,
                observers=observers,
                recording_plan=recording_plan,
                progress_reporter=progress_reporter,
                runtime_context=runtime_context,
                scheduling_options=resolved_scheduling,
            )
        for group in plan.groups:
            with benchmark_span(
                "dispatch.group.total",
                group_id=group.group_id,
                group_size=group.size,
                mode=group.mode,
                batch_kind=group.batch_kind,
                nx=group.nx,
                geometry_shared=group.geometry_shared,
                has_padding=group.has_padding,
            ):
                progress_reporter.start_group(group)
                progress_reporter.route_group(
                    group,
                    route=group.dispatch_method,
                    reason="planned batch route",
                )
                group_results = _run_batch_group(
                    group,
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                    batch_options=resolved_batch_options,
                    observers=observers,
                    recording_plan=recording_plan,
                    progress_callback=progress_reporter.kernel_callback(group),
                    runtime_context=runtime_context,
                )
                progress_reporter.finish_group(group)
            if _is_complete_single_cohort_result(
                plan,
                group=group,
                group_results=group_results,
            ):
                record_benchmark_metadata(dispatch_result_validation="cohort-identity")
                return group_results
            _store_dispatch_results(
                group_results,
                results=results,
                seen_indices=seen_indices,
            )

    _validate_dispatch_results(results, seen_indices=seen_indices, plan=plan)
    return tuple(results)


def _is_complete_single_cohort_result(
    plan: DispatchPlan,
    *,
    group: DispatchGroup,
    group_results: tuple[DispatchRecord, ...],
) -> bool:
    """Recognize a complete internal cohort result without scanning its rows."""

    if len(plan.groups) != 1 or len(group_results) != 1:
        return False
    result = group_results[0]
    return (
        isinstance(result, DispatchCohortRecord)
        and result.group_id == group.group_id
        and result.group_size == group.size
        and result.indices is group.pool_indices
        and result.axons is group.axons
        and result.simulations is group.simulations
    )


def _resolve_dispatch_scheduling(
    runtime_context: Any | None,
    *,
    scheduling_options: DispatchSchedulingOptions | None,
) -> DispatchSchedulingOptions:
    if scheduling_options is not None:
        return scheduling_options
    context_options = (
        None
        if runtime_context is None
        else getattr(runtime_context, "dispatch_scheduling", None)
    )
    if context_options is None:
        return DispatchSchedulingOptions()
    if not isinstance(context_options, DispatchSchedulingOptions):
        raise TypeError(
            "runtime_context.dispatch_scheduling must be a DispatchSchedulingOptions value."
        )
    return context_options


def _run_pool_async_groups(
    plan: DispatchPlan,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None,
    recording_plan: RecordingPlan | None,
    progress_reporter: DispatchProgress,
    runtime_context: Any | None,
    scheduling_options: DispatchSchedulingOptions,
) -> tuple[DispatchRecord, ...]:
    """Run compatible groups by enqueueing several JAX calls before waiting."""

    results: list[DispatchRecord] = []
    seen_indices: set[int] = set()
    pending: list[Any] = []
    pending_groups: list[DispatchGroup] = []
    flush_count = 0
    pending_max = 0
    record_benchmark_metadata(
        dispatch_async_groups=True,
        dispatch_async_max_pending_groups=int(scheduling_options.max_pending_groups),
    )
    for group in plan.groups:
        with benchmark_span(
            "dispatch.group.total",
            group_id=group.group_id,
            group_size=group.size,
            mode=group.mode,
            batch_kind=group.batch_kind,
            nx=group.nx,
            geometry_shared=group.geometry_shared,
            has_padding=group.has_padding,
            dispatch_schedule="async_enqueue",
        ):
            progress_reporter.start_group(group)
            progress_reporter.route_group(
                group,
                route=group.dispatch_method,
                reason="planned batch route",
            )
            pending.append(
                enqueue_batch_group(
                    group,
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                    batch_options=batch_options,
                    observers=observers,
                    recording_plan=recording_plan,
                    progress_callback=progress_reporter.kernel_callback(group),
                    runtime_context=runtime_context,
                )
            )
            pending_groups.append(group)
            pending_max = max(pending_max, len(pending))
        if len(pending) >= int(scheduling_options.max_pending_groups):
            group_results, flush_count = _flush_pending_batch_groups(
                pending,
                pending_groups,
                flush_count=flush_count,
                progress_reporter=progress_reporter,
            )
            _store_dispatch_results(
                group_results,
                results=results,
                seen_indices=seen_indices,
            )

    group_results, flush_count = _flush_pending_batch_groups(
        pending,
        pending_groups,
        flush_count=flush_count,
        progress_reporter=progress_reporter,
    )
    _store_dispatch_results(
        group_results,
        results=results,
        seen_indices=seen_indices,
    )
    record_benchmark_metadata(
        dispatch_async_flush_count=int(flush_count),
        dispatch_async_pending_max=int(pending_max),
    )
    _validate_dispatch_results(results, seen_indices=seen_indices, plan=plan)
    return tuple(results)


def _flush_pending_batch_groups(
    pending: list[Any],
    pending_groups: list[DispatchGroup],
    *,
    flush_count: int,
    progress_reporter: DispatchProgress,
) -> tuple[tuple[DispatchRecord, ...], int]:
    if not pending:
        return (), flush_count
    group_count = len(pending)
    row_count = sum(group.size for group in pending_groups)
    out: list[DispatchRecord] = []
    with benchmark_span(
        "dispatch.async_flush",
        group_count=group_count,
        row_count=row_count,
        flush_index=flush_count,
    ):
        for pending_group, group in zip(pending, pending_groups, strict=True):
            out.extend(finalize_batch_group(pending_group))
            progress_reporter.finish_group(group)
    pending.clear()
    pending_groups.clear()
    return tuple(out), flush_count + 1


def _store_dispatch_results(
    group_results: tuple[DispatchRecord, ...],
    *,
    results: list[DispatchRecord],
    seen_indices: set[int],
) -> None:
    for result in group_results:
        indices = (
            result.indices
            if isinstance(result, DispatchCohortRecord)
            else (result.index,)
        )
        for index in indices:
            if index in seen_indices:
                raise RuntimeError(f"duplicate dispatch result for pool index {index}.")
            seen_indices.add(index)
        results.append(result)


def _validate_dispatch_results(
    results: list[DispatchRecord],
    *,
    seen_indices: set[int],
    plan: DispatchPlan,
) -> None:
    if len(seen_indices) != len(plan.items):
        missing = sorted(set(range(len(plan.items))) - seen_indices)
        raise RuntimeError(f"pool dispatch did not produce all axon results: {missing}.")
    if any(index < 0 or index >= len(plan.items) for index in seen_indices):
        raise RuntimeError("pool dispatch did not produce all axon results.")


def _run_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None,
    recording_plan: RecordingPlan | None,
    progress_callback: Any = None,
    runtime_context: Any | None = None,
) -> tuple[DispatchRecord, ...]:
    """Execute one compatible group through the active backend facade."""

    return run_batch_group(
        group,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        batch_options=batch_options,
        observers=observers,
        recording_plan=recording_plan,
        progress_callback=progress_callback,
        runtime_context=runtime_context,
    )


__all__ = [
    "DispatchSchedulingOptions",
    "run_pool",
]
