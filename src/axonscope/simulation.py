from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Sequence, TypeAlias

import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.backends.execution import (
    batch_options_for_execution_context,
    batch_options_from_recording,
    execution_context,
)
from axonscope.benchmarking.hotpaths import benchmark_span
from axonscope.performance import ExecutionPolicy
from axonscope.signals import MEMBRANE_VOLTAGE
from axonscope.utils import units
from axonscope.axons.axon import Axon
from axonscope.dispatcher import run_pool
from axonscope.dispatcher.progress import ProgressOption
from axonscope.population import AxonPopulation
from axonscope.recording import Recording, RecordingPlan
from axonscope.results import AxonSimulationResult
from axonscope.solvers import BatchOptions, Solver, SolverOptions

if TYPE_CHECKING:
    from axonscope.dispatcher._records import DispatchRecord


AxonInput: TypeAlias = Axon | AxonInstance
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
    solver options. All runs are normalized through ``AxonPopulation`` so a
    one-axon run and a many-axon run share the same dispatcher/result lifecycle.
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
        self.population = _normalize_axon_inputs(axons)
        self.axons = self.population.instances
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
        """Return whether this executable definition contains one axon."""

        return self.population.is_single

    @property
    def is_population(self) -> bool:
        """Return whether this definition uses the population lifecycle."""

        return True

    def run(self) -> SimulationRunResult:
        """Execute this simulation definition and return public results."""

        if self.solver is not None:
            raise NotImplementedError(
                "explicit solver objects are not part of the unified AxonSimulation "
                "pipeline; use solver_options."
            )
        return _run_population_simulation(
            self.population,
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
            population_lifecycle=True,
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


def _recording_as_vm_only(recording: Recording) -> RecordingPlan:
    """Return the same output placement policy without non-Vm observable groups."""

    plan = recording.to_plan()
    return replace(
        plan,
        gates=False,
        currents=False,
        conductances=False,
        state_variables=False,
        signals=(MEMBRANE_VOLTAGE,) if recording.voltage else (),
    )


def _validate_single_row_observable_recording(recording: Recording) -> None:
    """Validate observable groups supported by the one-row scalar fallback."""

    if not recording.voltage:
        raise NotImplementedError(
            "single-row observable-only recording is not supported; include Vm "
            "or use Recording.none() with solver-side observers."
        )


def _filter_pool_recording(
    results: Sequence[DispatchRecord],
    recording: Recording,
) -> tuple[DispatchRecord, ...]:
    """Apply public recording selection to dispatcher scalar fallback rows."""

    if not recording.voltage and not recording.wants_observables:
        return tuple(results)
    plan = recording.to_plan()
    filtered = []
    for axon_result in results:
        if hasattr(axon_result, "indices"):
            filtered.append(axon_result)
            continue
        vm = axon_result.Vm
        record_indices = axon_result.record_indices
        if recording.voltage and vm is not None and record_indices is None:
            indices = plan.indices_for(int(axon_result.axon.n_compartments))
            if indices is not None:
                record_indices = tuple(int(value) for value in indices)
                vm = np.take(np.asarray(vm), np.asarray(indices), axis=1)
        recordings = _filter_recording_payload(
            axon_result.recordings,
            recording=recording,
            vm=vm,
        )
        filtered.append(
            replace(
                axon_result,
                Vm=vm,
                record_indices=record_indices,
                recordings=recordings,
            )
        )
    return tuple(filtered)


def _filter_recording_payload(
    recordings: dict[str, Any] | None,
    *,
    recording: Recording,
    vm: Any | None,
) -> dict[str, Any] | None:
    """Keep only the groups requested by the public recording policy."""

    if recordings is None:
        return None
    wanted: dict[str, Any] = {}
    if recording.voltage:
        if vm is not None:
            wanted["Vm"] = vm
        elif "Vm" in recordings:
            wanted["Vm"] = recordings["Vm"]
    for attr_name, result_key in _RECORDING_GROUPS:
        if getattr(recording, attr_name) and result_key in recordings:
            wanted[result_key] = recordings[result_key]
    return wanted or None


def _pool_batch_options_for_recording(
    *,
    population: AxonPopulation,
    recording: Recording | None,
    batch_options: BatchOptions | None,
) -> BatchOptions | None:
    """Merge explicit public recording with lower-level batch execution knobs."""

    if recording is not None and recording.wants_observables and population.is_single:
        _validate_single_row_observable_recording(recording)
        return batch_options_from_recording(
            _recording_as_vm_only(recording),
            batch_options=batch_options,
        )
    return batch_options_from_recording(recording, batch_options=batch_options)


def _run_population_simulation(
    pool: AxonPopulation | Sequence[Axon | AxonInstance],
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
    """Run a population and return the canonical ``AxonSimulationResult``.

    Per-axon views are exposed in pool order through indexing and iteration.
    The lower-level dispatch metadata is kept in each view's ``diagnostics``
    dictionary.
    ``recording`` controls the public output contract. ``solver_options`` are
    forwarded unchanged to solver runtime preparation; ``batch_options`` only
    affects batch-kernel execution details such as chunking. Set ``progress``
    to True, ``"rich"``, or ``"plain"`` to display dispatch planning,
    preparation, optional compilation points, kernel solving, and
    result-assembly progress.
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
        population=population,
        recording=recording,
        batch_options=batch_options,
    )
    record_observables = bool(recording is not None and recording.wants_observables)
    with execution_context(execution_policy, instances=population.instances) as context:
        effective_batch_options = batch_options_for_execution_context(
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
            record_observables=record_observables,
            progress=progress,
            backend_context=context,
        )
    with benchmark_span("results.to_public", pool_size=len(population.instances)):
        if recording is not None:
            results = _filter_pool_recording(results, recording)
        return AxonSimulationResult.from_dispatch_results(
            results,
            recording=recording,
        )


__all__ = ["AxonSimulation"]
