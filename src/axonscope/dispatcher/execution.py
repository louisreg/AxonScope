from __future__ import annotations

from typing import Any, Sequence

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.runtime.execution import run_batch_group
from axonscope.benchmarking import benchmark_span, record_benchmark_metadata
from axonscope.recording import RecordingPlan
from axonscope.dispatcher.plan import (
    DispatchGroup,
    DispatchPlan,
    build_dispatch_plan,
)
from axonscope.dispatcher.progress import (
    DispatchProgress,
    ProgressOption,
    emit_initial_progress,
)
from axonscope.dispatcher.routing import can_use_batch_route
from axonscope.dispatcher._records import (
    DispatchCohortRecord,
    DispatchRecord,
)
from axonscope.solvers import BatchOptions, SolverOptions
from axonscope.utils import units


def run_pool(
    axons: Sequence[Axon | AxonInstance],
    *,
    tsim_ms: Any,
    dt_ms: Any,
    solver_options: SolverOptions | None = None,
    batch_options: BatchOptions | None = None,
    observers: Sequence[Any] | None = None,
    record_observables: bool = False,
    recording_plan: RecordingPlan | None = None,
    progress: ProgressOption = False,
    runtime_context: Any | None = None,
    dispatch_plan: DispatchPlan | None = None,
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
            solver_options=solver_options,
            batch_options=batch_options,
            observers=tuple(observers) if observers is not None else None,
            record_observables=bool(record_observables),
            recording_plan=recording_plan,
            progress=progress,
            runtime_context=runtime_context,
            dispatch_plan=dispatch_plan,
        )


def _run_pool_checked(
    axons: Sequence[Axon | AxonInstance],
    *,
    tsim_ms: float,
    dt_ms: float,
    solver_options: SolverOptions | None,
    batch_options: BatchOptions | None,
    observers: tuple[Any, ...] | None,
    record_observables: bool,
    recording_plan: RecordingPlan | None,
    progress: ProgressOption,
    runtime_context: Any | None,
    dispatch_plan: DispatchPlan | None,
) -> tuple[DispatchRecord, ...]:
    resolved_batch_options = BatchOptions.full() if batch_options is None else batch_options
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
                can_batch = _can_run_batch_group(
                    group,
                    batch_options=resolved_batch_options,
                    observers=observers,
                    record_observables=record_observables,
                )
                if can_batch:
                    progress_reporter.route_group(
                        group,
                        route=_dispatch_method(group),
                        reason="compatible batch route",
                    )
                    group_results = _run_batch_group(
                        group,
                        tsim_ms=tsim_ms,
                        dt_ms=dt_ms,
                        batch_options=resolved_batch_options,
                        solver_options=solver_options,
                        observers=observers,
                        recording_plan=recording_plan,
                        progress_callback=progress_reporter.kernel_callback(group),
                        runtime_context=runtime_context,
                    )
                else:
                    reason = _batch_rejection_reason(
                        group,
                        record_observables=record_observables,
                    )
                    progress_reporter.route_group(
                        group,
                        route="unsupported",
                        reason=reason,
                    )
                    raise NotImplementedError(reason)
                progress_reporter.finish_group(group)
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

    if len(seen_indices) != len(plan.items):
        missing = sorted(set(range(len(plan.items))) - seen_indices)
        raise RuntimeError(f"pool dispatch did not produce all axon results: {missing}.")
    if any(index < 0 or index >= len(plan.items) for index in seen_indices):
        raise RuntimeError("pool dispatch did not produce all axon results.")
    return tuple(results)


def _can_run_batch_group(
    group: DispatchGroup,
    *,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None,
    record_observables: bool = False,
) -> bool:
    """Return whether a dispatch group can use the current batch backend."""

    return can_use_batch_route(
        group,
        batch_options=batch_options,
        observers=observers,
        record_observables=record_observables,
    )


def _dispatch_method(group: DispatchGroup) -> str:
    """Return the public diagnostic label for a dispatch group."""

    prefix = "batch" if group.geometry_shared else "parameter-batch"
    if group.mode == "double":
        return f"{prefix}-double-cable"
    return f"{prefix}-single-cable"


def _batch_rejection_reason(
    group: DispatchGroup,
    *,
    record_observables: bool,
) -> str:
    """Return a readable reason why a group cannot use batch execution."""

    if record_observables:
        return "dense observable recording is unavailable for this batch group."
    if group.mode not in {"single", "double"}:
        return f"unsupported batch mode {group.mode!r}"
    return "batch route unavailable"


def _run_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    solver_options: SolverOptions | None,
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
        solver_options=solver_options,
        observers=observers,
        recording_plan=recording_plan,
        progress_callback=progress_callback,
        runtime_context=runtime_context,
    )


__all__ = [
    "run_pool",
]
