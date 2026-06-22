from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Sequence, TypeAlias

import numpy as np

from axonscope.axon_instance import AxonInstance, as_axon_instance
from axonscope.backends.jax.execution_policy import JaxExecutionContext, jax_execution_context
from axonscope.backends.jax.recording import batch_options_from_recording
from axonscope.benchmarking.hotpaths import benchmark_span
from axonscope.performance import ExecutionPolicy
from axonscope.utils import units
from axonscope.axons.axon import Axon
from axonscope.dispatcher import run_pool
from axonscope.dispatcher.progress import ProgressOption
from axonscope.population import AxonPopulation
from axonscope.recording import Recording, RecordingSpatial
from axonscope.results import AxonSimulationResult
from axonscope.results.pool import CohortResult
from axonscope.results.single import SimResult
from axonscope.solvers import (
    BatchOptions,
    CrankNicholson,
    Solver,
    SolverOptions,
    resolve_double_cable_block_solver,
)

if TYPE_CHECKING:
    from axonscope.dispatcher.results import DispatchRecord


AxonInput: TypeAlias = Axon | AxonInstance
AxonPoolInput: TypeAlias = AxonPopulation | Sequence[Axon | AxonInstance]
SimulationRunResult: TypeAlias = AxonSimulationResult

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
        execution_policy: ExecutionPolicy | None = None,
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
        self.execution_policy = execution_policy
        self.progress = progress

    @property
    def is_single(self) -> bool:
        """Return whether this executable definition uses scalar execution."""

        return not self._population_lifecycle

    @property
    def is_population(self) -> bool:
        """Return whether this executable definition uses population execution."""

        return self._population_lifecycle

    def run(self) -> SimulationRunResult:
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
                execution_policy=self.execution_policy,
            )

        if self.solver is not None:
            raise NotImplementedError(
                "explicit solver objects are currently supported only for single-axon "
                "AxonSimulation runs; use solver_options for pools."
            )
        return simulate_pool(
            self.axons,
            duration=self.duration,
            dt=self.dt,
            solver_options=self.solver_options,
            batch_options=self.batch_options,
            recording=self.recording,
            observers=self.observers,
            execution_policy=self.execution_policy,
            progress=self.progress,
        )

    def estimate(self, **kwargs: Any):
        """Estimate memory pressure for this simulation without running it."""

        from axonscope.performance import estimate_simulation

        policy_kwargs: dict[str, Any] = {}
        if self.execution_policy is not None:
            policy_kwargs["runtime"] = self.execution_policy.runtime
            policy_kwargs["device"] = self.execution_policy.device
            if self.execution_policy.precision is not None:
                policy_kwargs["precision"] = self.execution_policy.precision
        policy_kwargs.update(kwargs)

        return estimate_simulation(
            self.population,
            duration=self.duration,
            dt=self.dt,
            recording=self.recording,
            batch_options=self.batch_options,
            observers=self.observers,
            population_lifecycle=self.is_population,
            **policy_kwargs,
        )

    def inspect(self, *, print_summary: bool = False):
        """Inspect planning, dispatch/batch grouping, and preparation."""

        from axonscope.inspection import inspect_simulation

        return inspect_simulation(
            self.population,
            duration=self.duration,
            dt=self.dt,
            recording=self.recording,
            batch_options=self.batch_options,
            observers=self.observers,
            execution_policy=self.execution_policy,
            print_summary=print_summary,
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


def _validate_single_recording(
    recording: Recording,
    *,
    observers_present: bool = False,
) -> None:
    """Validate recording features currently supported by scalar public runs."""

    if not recording.voltage and recording.wants_observables:
        raise NotImplementedError(
            "single-axon observable-only recording is not supported; include Vm "
            "or use Recording.none() with solver-side observers."
        )
    if not recording.voltage and not observers_present:
        raise NotImplementedError("Recording.none() requires solver-side observers.")
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


def _single_result_to_public(result: SimResult, recording: Recording) -> AxonSimulationResult:
    """Convert one internal scalar result to the canonical public container."""

    simulation = result.simulation
    if simulation is None:
        simulation = as_axon_instance(result.axon)
    cohort = CohortResult(
        input_indices=(0,),
        axons=(result.axon,),
        simulations=(simulation,),
        Vm=(
            None
            if result.recordings is None or "Vm" not in result.recordings
            else np.asarray(result.Vm)[None, ...]
        ),
        t=np.asarray(result.t),
        diagnostics=(
            {
                **(result.diagnostics or {}),
                "pool_index": 0,
                "dispatch_method": "scalar",
                "dispatch_batch_kind": "scalar",
            },
        ),
        record_indices=(result.record_indices,),
        recording=recording,
        observations=result.observations,
        recordings=(result.recordings,),
        final_states=(result.final_state,),
    )
    return AxonSimulationResult((cohort,), size=1, recording=recording)


def _filter_pool_recording(
    results: Sequence[DispatchRecord],
    recording: Recording,
) -> tuple[DispatchRecord, ...]:
    """Apply spatial Vm filtering when a scalar fallback returned full traces."""

    if not recording.voltage:
        return tuple(results)
    plan = recording.to_plan()
    filtered = []
    for axon_result in results:
        if hasattr(axon_result, "indices"):
            filtered.append(axon_result)
            continue
        if axon_result.record_indices is not None:
            filtered.append(axon_result)
            continue
        indices = plan.indices_for(int(axon_result.axon.n_compartments))
        if indices is None:
            filtered.append(axon_result)
            continue
        index_tuple = tuple(int(value) for value in indices)
        filtered.append(
            replace(
                axon_result,
                Vm=np.take(np.asarray(axon_result.Vm), np.asarray(indices), axis=1),
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

    return batch_options_from_recording(recording, batch_options=batch_options)


def _pool_batch_options_for_execution_context(
    batch_options: BatchOptions | None,
    context: JaxExecutionContext,
) -> BatchOptions | None:
    """Apply execution-policy device routing to batch-only solver options."""

    if context.platform is None:
        return batch_options
    options = BatchOptions.full() if batch_options is None else batch_options
    if options.double_cable_block_solver != "auto":
        return options
    return replace(
        options,
        double_cable_block_solver=resolve_double_cable_block_solver(
            "auto",
            platform=context.platform,
        ),
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
    execution_policy: ExecutionPolicy | None = None,
) -> AxonSimulationResult:
    """Run one axon simulation and return an ``AxonSimulationResult``.

    Plain numeric durations are interpreted as milliseconds. Pint-like
    quantities are converted at the public boundary. Passing a pure `Axon`
    creates a no-stimulation protocol around it. Use ``result.single`` or
    ``result[0]`` for one-axon access.
    """

    observer_defs = tuple(observers) if observers is not None else None
    simulation = as_axon_instance(axon)
    duration_ms, step_ms = _resolve_time(duration=duration, dt=dt)
    active_solver = _resolve_solver(solver, solver_options)
    rec = _resolve_recording(recording)
    _validate_single_recording(rec, observers_present=bool(observer_defs))
    with benchmark_span(
        "simulation.total",
        pool_size=1,
        tsim_ms=duration_ms,
        dt_ms=step_ms,
    ):
        with jax_execution_context(execution_policy, instances=(simulation,)):
            result = active_solver.solve(
                simulation,
                tsim=duration_ms,
                dt=step_ms,
                record_observables=rec.wants_observables,
                record_voltage=rec.voltage,
                observers=observer_defs,
            )
        with benchmark_span("results.to_public", pool_size=1):
            return _single_result_to_public(_finalize_single_result(result, rec), rec)


def simulate_pool(
    pool: AxonPoolInput,
    *,
    duration: Any,
    dt: Any,
    solver_options: SolverOptions | None = None,
    batch_options: BatchOptions | None = None,
    recording: Recording | None = None,
    observers: Sequence[Any] | None = None,
    execution_policy: ExecutionPolicy | None = None,
    progress: ProgressOption = False,
) -> AxonSimulationResult:
    """Run a pool and return the canonical ``AxonSimulationResult``.

    Per-axon views are exposed in pool order through indexing and iteration.
    The lower-level dispatch metadata is kept in each view's ``diagnostics``
    dictionary.
    ``recording`` controls the public output contract. ``solver_options`` are
    forwarded unchanged to solver runtime preparation; ``batch_options`` only
    affects batch-kernel execution details such as chunking. Set ``progress``
    to True, ``"rich"``, or ``"plain"`` to display dispatch/solver progress.
    """

    population = pool if isinstance(pool, AxonPopulation) else AxonPopulation(pool)
    observer_defs = tuple(observers) if observers is not None else None
    if (
        recording is not None
        and not recording.voltage
        and not recording.wants_observables
        and not observer_defs
    ):
        raise NotImplementedError("Recording.none() requires solver-side observers.")
    duration_ms, step_ms = _resolve_time(duration=duration, dt=dt)
    resolved_batch_options = _pool_batch_options_for_recording(
        recording=recording,
        batch_options=batch_options,
    )
    with jax_execution_context(execution_policy, instances=population.instances) as context:
        effective_batch_options = _pool_batch_options_for_execution_context(
            resolved_batch_options,
            context,
        )
        results = run_pool(
            population,
            tsim_ms=duration_ms,
            dt_ms=step_ms,
            solver_options=solver_options,
            batch_options=effective_batch_options,
            observers=observer_defs,
            progress=progress,
        )
    with benchmark_span("results.to_public", pool_size=len(population.instances)):
        if recording is not None:
            results = _filter_pool_recording(results, recording)
        return AxonSimulationResult.from_dispatch_results(
            results,
            recording=recording,
        )


__all__ = ["AxonSimulation", "simulate", "simulate_pool"]
