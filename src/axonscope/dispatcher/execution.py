from __future__ import annotations

from typing import Any, Sequence

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.backends.jax.group_runner import run_jax_batch_group
from axonscope.benchmarking.hotpaths import benchmark_span, record_benchmark_metadata
from axonscope.dispatcher.plan import DispatchGroup, DispatchItem, build_dispatch_plan
from axonscope.dispatcher.progress import DispatchProgress, ProgressOption
from axonscope.dispatcher.results import DispatchResult
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
    progress: ProgressOption = False,
) -> tuple[DispatchResult, ...]:
    """Run an axon pool and return one raw dispatch result per input simulation.

    Public code should generally call ``axonscope.simulate_pool`` so these raw
    dispatch results are converted to ``SimResult`` objects. Plain numeric times
    are interpreted as milliseconds; Pint-like quantities are converted at this
    boundary. ``progress`` enables optional Rich/plain progress reporting at the
    dispatch-group level and, for chunked batch runs, at the kernel-chunk level.
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
            progress=progress,
        )


def _run_pool_checked(
    axons: Sequence[Axon | AxonInstance],
    *,
    tsim_ms: float,
    dt_ms: float,
    solver_options: SolverOptions | None,
    batch_options: BatchOptions | None,
    progress: ProgressOption,
) -> tuple[DispatchResult, ...]:
    resolved_batch_options = BatchOptions.full() if batch_options is None else batch_options
    plan = build_dispatch_plan(axons)
    record_benchmark_metadata(dispatch_group_count=len(plan.groups))

    results: list[DispatchResult | None] = [None] * len(plan.items)
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
                if _can_run_batch_group(group):
                    group_results = _run_batch_group(
                        group,
                        tsim_ms=tsim_ms,
                        dt_ms=dt_ms,
                        batch_options=resolved_batch_options,
                        solver_options=solver_options,
                        progress_callback=progress_reporter.kernel_callback(group),
                    )
                else:
                    group_results = _run_scalar_group(
                        group,
                        tsim_ms=tsim_ms,
                        dt_ms=dt_ms,
                        solver_options=solver_options,
                    )
                    callback = progress_reporter.kernel_callback(group)
                    if callback is not None:
                        callback(1, 1)
                progress_reporter.finish_group(group)
            for result in group_results:
                results[result.index] = result

    if any(result is None for result in results):
        raise RuntimeError("pool dispatch did not produce all axon results.")
    return tuple(result for result in results if result is not None)


def _run_scalar_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    solver_options: SolverOptions | None,
) -> tuple[DispatchResult, ...]:
    """Execute a dispatch group through scalar solves."""

    solver = CrankNicholson(solver_options=solver_options)
    return tuple(
        _dispatch_result_from_sim(
            item,
            solver.solve(item.simulation, tsim=tsim_ms, dt=dt_ms),
            group_id=group.group_id,
        )
        for item in group.items
    )


def _can_run_batch_group(group: DispatchGroup) -> bool:
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
    progress_callback: Any = None,
) -> tuple[DispatchResult, ...]:
    """Execute one compatible group through the JAX batch backend."""

    return run_jax_batch_group(
        group,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        batch_options=batch_options,
        solver_options=solver_options,
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
        Vm=sim.Vm,
        t=sim.t,
        group_id=group_id,
        method="scalar",
        record_indices=None,
        group_size=1,
        batch_kind="scalar",
        geometry_shared=True,
        has_padding=False,
    )


__all__ = [
    "DispatchResult",
    "run_pool",
]
