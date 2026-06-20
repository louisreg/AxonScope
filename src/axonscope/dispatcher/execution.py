from __future__ import annotations

from typing import Any, Sequence

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.backends.jax.group_runner import run_jax_batch_group
from axonscope.benchmarking.hotpaths import benchmark_span, record_benchmark_metadata
from axonscope.dispatcher.plan import DispatchGroup, DispatchItem, build_dispatch_plan
from axonscope.dispatcher.progress import DispatchProgress, ProgressOption
from axonscope.dispatcher.results import DispatchCohortResult, DispatchRecord, DispatchResult
from axonscope.results import SimResult
from axonscope.solvers import BatchOptions, CrankNicholson, SolverOptions
from axonscope.utils import units


def run_pool(
    axons: Sequence[Axon | AxonInstance],
    *,
    tsim_ms: Any,
    dt_ms: Any,
    solver_options: SolverOptions | None = None,
    batch_options: BatchOptions | None = None,
    observers: Sequence[Any] | None = None,
    progress: ProgressOption = False,
) -> tuple[DispatchRecord, ...]:
    """Run an axon pool and return raw dispatch records.

    Public code should generally call ``axonscope.simulate_pool`` so these raw
    dispatch records are converted to public cohort results. Batched observer-
    only groups may remain a single compact record instead of one record per
    input axon. Plain numeric times are interpreted as milliseconds; Pint-like
    quantities are converted at this boundary. ``progress`` enables optional
    Rich/plain progress reporting at the dispatch-group level and, for chunked
    batch runs, at the kernel-chunk level.
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
            progress=progress,
        )


def _run_pool_checked(
    axons: Sequence[Axon | AxonInstance],
    *,
    tsim_ms: float,
    dt_ms: float,
    solver_options: SolverOptions | None,
    batch_options: BatchOptions | None,
    observers: tuple[Any, ...] | None,
    progress: ProgressOption,
) -> tuple[DispatchRecord, ...]:
    resolved_batch_options = BatchOptions.full() if batch_options is None else batch_options
    plan = build_dispatch_plan(axons)
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
                if _can_run_batch_group(group, observers=observers):
                    group_results = _run_batch_group(
                        group,
                        tsim_ms=tsim_ms,
                        dt_ms=dt_ms,
                        batch_options=resolved_batch_options,
                        solver_options=solver_options,
                        observers=observers,
                        progress_callback=progress_reporter.kernel_callback(group),
                    )
                else:
                    group_results = _run_scalar_group(
                        group,
                        tsim_ms=tsim_ms,
                        dt_ms=dt_ms,
                        solver_options=solver_options,
                        observers=observers,
                        record_voltage=resolved_batch_options.recording.mode != "none",
                    )
                    callback = progress_reporter.kernel_callback(group)
                    if callback is not None:
                        callback(1, 1)
                progress_reporter.finish_group(group)
            for result in group_results:
                indices = (
                    result.indices
                    if isinstance(result, DispatchCohortResult)
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


def _run_scalar_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    solver_options: SolverOptions | None,
    observers: tuple[Any, ...] | None,
    record_voltage: bool,
) -> tuple[DispatchResult, ...]:
    """Execute a dispatch group through scalar solves."""

    solver = CrankNicholson(solver_options=solver_options)
    return tuple(
        _dispatch_result_from_sim(
            item,
            solver.solve(
                item.simulation,
                tsim=tsim_ms,
                dt=dt_ms,
                record_voltage=record_voltage,
                observers=observers,
            ),
            group_id=group.group_id,
        )
        for item in group.items
    )


def _can_run_batch_group(
    group: DispatchGroup,
    *,
    observers: tuple[Any, ...] | None,
) -> bool:
    """Return whether a dispatch group can use the current batch backend."""

    if group.size < 2:
        return False
    return group.mode in {"single", "double"}


def _run_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    solver_options: SolverOptions | None,
    observers: tuple[Any, ...] | None,
    progress_callback: Any = None,
) -> tuple[DispatchRecord, ...]:
    """Execute one compatible group through the JAX batch backend."""

    return run_jax_batch_group(
        group,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        batch_options=batch_options,
        solver_options=solver_options,
        observers=observers,
        progress_callback=progress_callback,
    )


def _dispatch_result_from_sim(
    item: DispatchItem,
    sim: SimResult,
    *,
    group_id: int,
) -> DispatchResult:
    """Convert a scalar ``SimResult`` to a raw dispatch result."""

    return DispatchResult(
        index=item.index,
        axon=item.simulation.axon,
        simulation=item.simulation,
        Vm=sim.recordings["Vm"] if sim.recordings is not None and "Vm" in sim.recordings else None,
        t=sim.t,
        group_id=group_id,
        method="scalar",
        record_indices=None,
        observations=sim.observations,
        group_size=1,
        batch_kind="scalar",
        geometry_shared=True,
        has_padding=False,
    )


__all__ = [
    "DispatchCohortResult",
    "DispatchRecord",
    "DispatchResult",
    "run_pool",
]
