from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Sequence, TypeAlias

import jax.numpy as jnp

from axonscope.axon_instance import AxonInstance, as_axon_instance
from axonscope.utils import units
from axonscope.axons.axon import Axon
from axonscope.dispatcher import run_pool
from axonscope.dispatcher.progress import ProgressOption
from axonscope.population import AxonPopulation
from axonscope.recording import Recording, RecordingSpatial
from axonscope.results import SimResult
from axonscope.solvers import BatchOptions, CrankNicholson, Solver, SolverOptions

if TYPE_CHECKING:
    from axonscope.dispatcher.execution import DispatchResult


AxonInput: TypeAlias = Axon | AxonInstance
AxonPoolInput: TypeAlias = AxonPopulation | Sequence[Axon | AxonInstance]
AxonSimulationResult: TypeAlias = SimResult | list[SimResult]

_RECORDING_GROUPS = (
    ("gates", "gates"),
    ("currents", "currents"),
    ("conductances", "conductances"),
    ("state_variables", "states"),
)


def _normalize_axon_inputs(axons: AxonInput | Sequence[AxonInput] | AxonPopulation) -> AxonPopulation:
    """Normalize one or many public axon inputs for executable simulations."""

    if isinstance(axons, AxonPopulation):
        return axons
    try:
        return AxonPopulation(axons)
    except (TypeError, ValueError) as exc:
        message = str(exc).replace("AxonPopulation", "AxonSimulation")
        raise type(exc)(message) from exc


class AxonSimulation:
    """Executable simulation definition for one or more axon instances.

    `AxonInstance` describes one concrete axon occurrence and its local
    stimulation. `AxonSimulation` collects one or more axons/instances with
    execution parameters such as duration, step size, recording policy, and
    solver options. The current implementation delegates to the existing
    single-axon and pool wrappers; later architecture phases can replace that
    lowering behind this stable root object.
    """

    def __init__(
        self,
        axons: AxonInput | Sequence[AxonInput] | AxonPopulation,
        *,
        duration: Any,
        dt: Any,
        recording: Recording | None = None,
        solver: Solver | None = None,
        solver_options: SolverOptions | None = None,
        batch_options: BatchOptions | None = None,
        observers: Sequence[Any] | None = None,
        progress: ProgressOption = False,
    ) -> None:
        population_lifecycle = not isinstance(axons, (Axon, AxonInstance))
        self.population = _normalize_axon_inputs(axons)
        self.axons = self.population.instances
        self._population_lifecycle = population_lifecycle
        self.duration = duration
        self.dt = dt
        self.recording = recording
        self.solver = solver
        self.solver_options = solver_options
        self.batch_options = batch_options
        self.observers = tuple(observers) if observers is not None else None
        self.progress = progress

    @property
    def is_single(self) -> bool:
        """Return whether this executable definition uses scalar execution."""

        return not self._population_lifecycle

    @property
    def is_population(self) -> bool:
        """Return whether this executable definition uses population execution."""

        return self._population_lifecycle

    def run(self) -> AxonSimulationResult:
        """Execute this simulation definition and return public results."""

        if self.is_single:
            if self.batch_options is not None:
                raise ValueError("batch_options are only valid for multi-axon simulations.")
            if self.progress is not False:
                raise ValueError("progress is only valid for multi-axon simulations.")
            return simulate(
                self.axons[0],
                duration=self.duration,
                dt=self.dt,
                solver=self.solver,
                solver_options=self.solver_options,
                recording=self.recording,
                observers=self.observers,
            )

        if self.solver is not None:
            raise NotImplementedError(
                "explicit solver objects are currently supported only for single-axon "
                "AxonSimulation runs; use solver_options for pools."
            )
        if self.observers:
            raise NotImplementedError("solver-side observers are not wired for pool runs yet.")
        return simulate_pool(
            self.axons,
            duration=self.duration,
            dt=self.dt,
            solver_options=self.solver_options,
            batch_options=self.batch_options,
            recording=self.recording,
            progress=self.progress,
        )


def _resolve_time(
    *,
    duration: Any | None,
    dt: Any | None,
) -> tuple[float, float]:
    """Resolve public time values to canonical milliseconds."""

    if duration is None:
        raise ValueError("duration is required.")
    if dt is None:
        raise ValueError("dt is required.")
    resolved_duration = units.to_ms(duration)
    resolved_step = units.to_ms(dt)
    if resolved_duration <= 0.0:
        raise ValueError("duration must be > 0.")
    if resolved_step <= 0.0:
        raise ValueError("dt must be > 0.")
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

    return Recording() if recording is None else recording


def _validate_single_recording(recording: Recording) -> None:
    """Validate recording features currently supported by scalar public runs."""

    if not recording.voltage:
        raise NotImplementedError("single-axon simulation currently always returns Vm.")
    if recording.positions_um is not None or recording.spatial is not RecordingSpatial.FULL:
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
    axon: Axon | AxonInstance,
    *,
    duration: Any,
    dt: Any,
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
    simulation = as_axon_instance(axon)
    duration_ms, step_ms = _resolve_time(duration=duration, dt=dt)
    active_solver = _resolve_solver(solver, solver_options)
    rec = _resolve_recording(recording)
    _validate_single_recording(rec)
    result = active_solver.solve(
        simulation,
        tsim=duration_ms,
        dt=step_ms,
        record_observables=rec.wants_observables,
    )
    return _finalize_single_result(result, rec)


def simulate_pool(
    pool: AxonPoolInput,
    *,
    duration: Any,
    dt: Any,
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

    population = pool if isinstance(pool, AxonPopulation) else AxonPopulation(pool)
    duration_ms, step_ms = _resolve_time(duration=duration, dt=dt)
    resolved_batch_options = _pool_batch_options_for_recording(
        recording=recording,
        batch_options=batch_options,
    )
    results = run_pool(
        population,
        tsim_ms=duration_ms,
        dt_ms=step_ms,
        solver_options=solver_options,
        batch_options=resolved_batch_options,
        progress=progress,
    )
    if recording is not None:
        results = _filter_pool_recording(results, recording)
    return [_dispatch_result_to_sim_result(result, recording) for result in results]


__all__ = ["AxonSimulation", "simulate", "simulate_pool"]
