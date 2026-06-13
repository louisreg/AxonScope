from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Sequence, TypeAlias

import jax.numpy as jnp

from axonscope.axon_simulation import AxonSimulation, as_axon_simulation
from axonscope.utils import units
from axonscope.axons.axon import Axon
from axonscope.dispatcher import run_pool
from axonscope.dispatcher.progress import ProgressOption
from axonscope.recording import Recording
from axonscope.results import SimResult
from axonscope.solvers import BatchOptions, CrankNicholson, Solver, SolverOptions

if TYPE_CHECKING:
    from axonscope.dispatcher.execution import DispatchResult


AxonPoolInput: TypeAlias = Sequence[Axon | AxonSimulation]

_RECORDING_GROUPS = (
    ("gates", "gates"),
    ("currents", "currents"),
    ("conductances", "conductances"),
    ("state_variables", "states"),
)


def _resolve_time(
    *,
    duration_ms: Any | None,
    dt_ms: Any | None,
    duration_alias: Any | None,
    duration_alias_name: str,
    dt: Any | None,
) -> tuple[float, float]:
    """Resolve public time aliases to canonical milliseconds."""

    if duration_ms is not None and duration_alias is not None:
        raise ValueError(
            f"Provide either duration_ms or {duration_alias_name}, not both."
        )
    if dt_ms is not None and dt is not None:
        raise ValueError("Provide either dt_ms or dt, not both.")

    duration = duration_ms if duration_ms is not None else duration_alias
    step = dt_ms if dt_ms is not None else dt
    if duration is None:
        raise ValueError(f"duration_ms or {duration_alias_name} is required.")
    if step is None:
        raise ValueError("dt_ms or dt is required.")
    resolved_duration = units.to_ms(duration)
    resolved_step = units.to_ms(step)
    if resolved_duration <= 0.0:
        raise ValueError("duration_ms must be > 0.")
    if resolved_step <= 0.0:
        raise ValueError("dt_ms must be > 0.")
    return resolved_duration, resolved_step


def _resolve_solver(
    solver: Solver | None,
    solver_options: SolverOptions | None,
) -> Solver:
    """Return the explicit solver or the default public solver."""

    if solver is not None and solver_options is not None:
        raise ValueError("Provide either solver or solver_options, not both.")
    return CrankNicholson(solver_options=solver_options) if solver is None else solver


def _resolve_recording(recording: Recording | None) -> Recording:
    """Return the explicit recording policy or the public default."""

    return Recording.voltage() if recording is None else recording


def _validate_single_recording(recording: Recording) -> None:
    """Validate recording features currently supported by scalar public runs."""

    if not recording.voltage:
        raise NotImplementedError("single-axon simulation currently always returns Vm.")
    if recording.positions_um is not None or recording.spatial_mode != "full":
        raise NotImplementedError("spatial single-axon recording filters are not wired yet.")
    if recording.sample_dt_ms is not None or recording.every_n_steps is not None:
        raise NotImplementedError("temporal single-axon recording filters are not wired yet.")


def _filter_observable_recordings(result: SimResult, recording: Recording) -> SimResult:
    """Keep only observable groups requested by a public ``Recording``."""

    if result.recordings is None:
        return result
    wanted = {}
    if recording.voltage and "Vm" in result.recordings:
        wanted["Vm"] = result.recordings["Vm"]
    for attr_name, result_key in _RECORDING_GROUPS:
        if getattr(recording, attr_name) and result_key in result.recordings:
            wanted[result_key] = result.recordings[result_key]
    return replace(result, recordings=wanted or None)


def _finalize_single_result(result: SimResult, recording: Recording) -> SimResult:
    """Apply public single-run recording metadata and observable filters."""

    result = _filter_observable_recordings(result, recording)
    return replace(result, recording=recording)


def _filter_pool_recording(
    results: Sequence[DispatchResult],
    recording: Recording,
) -> tuple[DispatchResult, ...]:
    """Apply spatial Vm filtering when a scalar fallback returned full traces."""

    batch_options = recording.to_batch_options()
    filtered = []
    for axon_result in results:
        if axon_result.record_indices is not None:
            filtered.append(axon_result)
            continue
        indices = batch_options.recording.indices_for(int(axon_result.axon.n_compartments))
        if indices is None:
            filtered.append(axon_result)
            continue
        index_tuple = tuple(int(value) for value in indices)
        filtered.append(
            replace(
                axon_result,
                Vm=jnp.take(axon_result.Vm, jnp.asarray(indices), axis=1),
                record_indices=index_tuple,
            )
        )
    return tuple(filtered)


def _pool_batch_options_for_recording(
    *,
    recording: Recording | None,
    batch_options: BatchOptions | None,
) -> BatchOptions | None:
    """Merge explicit public recording with lower-level batch execution knobs."""

    if recording is None:
        return batch_options
    recording_options = recording.to_batch_options()
    if batch_options is None:
        return recording_options
    return replace(batch_options, recording=recording_options.recording)


def _dispatch_result_to_sim_result(
    result: DispatchResult,
    recording: Recording | None,
) -> SimResult:
    """Convert the lower-level pool dispatch result to public ``SimResult``."""

    return SimResult(
        axon=result.axon,
        Vm=result.Vm,
        t=result.t,
        diagnostics={
            "pool_index": result.index,
            "dispatch_group_id": result.group_id,
            "dispatch_method": result.method,
            "dispatch_group_size": result.group_size,
            "dispatch_batch_kind": result.batch_kind,
            "dispatch_geometry_shared": result.geometry_shared,
            "dispatch_has_padding": result.has_padding,
        },
        recording=recording,
        record_indices=result.record_indices,
        simulation=result.simulation,
    )


def simulate(
    axon: Axon | AxonSimulation,
    *,
    duration_ms: Any | None = None,
    dt_ms: Any | None = None,
    tsim: Any | None = None,
    dt: Any | None = None,
    solver: Solver | None = None,
    solver_options: SolverOptions | None = None,
    recording: Recording | None = None,
    observers: Sequence[Any] | None = None,
) -> SimResult:
    """Run one axon simulation and return a ``SimResult``.

    Plain numeric durations are interpreted as milliseconds. Pint-like
    quantities are converted at the public boundary. Passing a pure `Axon`
    creates a no-stimulation protocol around it.
    """

    if observers:
        raise NotImplementedError("solver-side observers are not wired yet.")
    simulation = as_axon_simulation(axon)
    duration, step = _resolve_time(
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        duration_alias=tsim,
        duration_alias_name="tsim",
        dt=dt,
    )
    active_solver = _resolve_solver(solver, solver_options)
    rec = _resolve_recording(recording)
    _validate_single_recording(rec)
    result = active_solver.solve(
        simulation,
        tsim=duration,
        dt=step,
        record_observables=rec.wants_observables,
    )
    return _finalize_single_result(result, rec)


def simulate_pool(
    pool: AxonPoolInput,
    *,
    duration_ms: Any | None = None,
    dt_ms: Any | None = None,
    tsim_ms: Any | None = None,
    dt: Any | None = None,
    solver_options: SolverOptions | None = None,
    batch_options: BatchOptions | None = None,
    recording: Recording | None = None,
    progress: ProgressOption = False,
) -> list[SimResult]:
    """Run a pool and return one ``SimResult`` per simulation.

    Results are returned in pool order. The lower-level dispatch metadata
    is kept in each result's ``diagnostics`` dictionary.
    ``recording`` controls the public output contract. ``solver_options`` are
    forwarded unchanged to solver runtime preparation; ``batch_options`` only
    affects batch-kernel execution details such as chunking. Set ``progress``
    to True, ``"rich"``, or ``"plain"`` to display dispatch/solver progress.
    """

    duration, step = _resolve_time(
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        duration_alias=tsim_ms,
        duration_alias_name="tsim_ms",
        dt=dt,
    )
    resolved_batch_options = _pool_batch_options_for_recording(
        recording=recording,
        batch_options=batch_options,
    )
    results = run_pool(
        pool,
        tsim_ms=duration,
        dt_ms=step,
        solver_options=solver_options,
        batch_options=resolved_batch_options,
        progress=progress,
    )
    if recording is not None:
        results = _filter_pool_recording(results, recording)
    return [_dispatch_result_to_sim_result(result, recording) for result in results]


__all__ = ["simulate", "simulate_pool"]
