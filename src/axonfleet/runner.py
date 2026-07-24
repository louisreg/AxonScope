"""Execution owner for backend-neutral runnable plans."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from threading import Event
from typing import Any, Sequence

import numpy as np

from axonfleet.axon_instance import simulation_structure_revision
from axonfleet.benchmarking import benchmark_span, record_benchmark_metadata
from axonfleet.dispatcher._records import DispatchCohortRecord, DispatchRecord
from axonfleet.dispatcher.plan import (
    DispatchGroup,
    DispatchPlan,
    build_dispatch_plan,
    dispatch_plan_identity_key,
    expand_dispatch_plan_for_numeric_axis,
)
from axonfleet.dispatcher.progress import DispatchProgress, emit_initial_progress
from axonfleet.plans import (
    LeafPlan,
    NumericAxisPlan,
    Plan,
    PopulationPlan,
    SimulationPlan,
    SweepPlan,
    StudyPlan,
    ThresholdPlan,
)
from axonfleet.population import AxonPopulation
from axonfleet.recording import Recording
from axonfleet.results import AxonSimulationResult
from axonfleet.runtime.execution import (
    batch_options_from_recording,
    enqueue_batch_group,
    execution_context,
    finalize_batch_group,
    run_batch_group,
)
from axonfleet.runtime.group_preparation import PreparedCohortCache
from axonfleet.runtime.timebase import resolve_time
from axonfleet.solvers import BatchOptions


_RECORDING_GROUPS = (
    ("gates", "gates"),
    ("currents", "currents"),
    ("conductances", "conductances"),
    ("state_variables", "states"),
)


@dataclass(slots=True)
class _DispatchCacheEntry:
    population: AxonPopulation
    revision: int
    identity: tuple[Any, ...]
    dispatch_plan: DispatchPlan


@dataclass(slots=True)
class _PopulationCacheEntry:
    plan: PopulationPlan
    population: AxonPopulation


@dataclass(frozen=True, slots=True)
class _RunnerSchedulingOptions:
    """Runner-owned scheduling policy for independent dispatch groups."""

    async_groups: bool = False
    max_pending_groups: int = 4

    def __post_init__(self) -> None:
        if int(self.max_pending_groups) < 1:
            raise ValueError("max_pending_groups must be >= 1.")


class CancellationToken:
    """Thread-safe cooperative cancellation signal for runner work."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        """Request cancellation at the next safe runner boundary."""

        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""

        return self._event.is_set()


@dataclass(frozen=True, slots=True)
class StudyResult:
    """Ordered in-memory results from one named study plan."""

    name: str
    keys: tuple[str, ...]
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("study result name must be a non-empty string.")
        if len(self.keys) != len(self.values):
            raise ValueError("study result keys and values must have equal length.")

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, int):
            return self.values[key]
        try:
            index = self.keys.index(key)
        except ValueError as exc:
            raise KeyError(key) from exc
        return self.values[index]

    def __len__(self) -> int:
        return len(self.values)

    def items(self) -> tuple[tuple[str, Any], ...]:
        """Return ordered ``(task_key, result)`` pairs."""

        return tuple(zip(self.keys, self.values, strict=True))


class StudyExecutionError(RuntimeError):
    """Fail-fast study error retaining deterministic completed results."""

    def __init__(
        self,
        task_key: str,
        completed: StudyResult,
        pending_keys: tuple[str, ...],
        cause: Exception,
    ) -> None:
        super().__init__(f"study task {task_key!r} failed: {cause}")
        self.task_key = task_key
        self.completed = completed
        self.pending_keys = pending_keys
        self.cause = cause


class PlanCancelledError(RuntimeError):
    """Cooperative cancellation with completed and pending task metadata."""

    def __init__(
        self,
        completed: StudyResult | None,
        pending_keys: tuple[str, ...],
    ) -> None:
        super().__init__("runner execution was cancelled.")
        self.completed = completed
        self.pending_keys = pending_keys


class _CancellationRequested(Exception):
    pass


class Runner:
    """Execute one or many plans and own reusable execution state.

    The runner is the only public-workflow owner of dispatch materialization.
    Its caches contain populations, backend-neutral dispatch plans, and host
    prepared cohorts. Concrete runtime arrays and compiled executables remain
    owned by runtime caches.
    """

    def __init__(
        self,
        *,
        dispatch_cache_size: int = 64,
        _scheduling_options: _RunnerSchedulingOptions | None = None,
    ) -> None:
        cache_size = int(dispatch_cache_size)
        if cache_size < 0:
            raise ValueError("dispatch_cache_size must be >= 0.")
        self._dispatch_cache_size = cache_size
        self._populations: dict[int, _PopulationCacheEntry] = {}
        self._dispatch_plans: OrderedDict[int, _DispatchCacheEntry] = OrderedDict()
        self._prepared_cohorts = PreparedCohortCache()
        self._scheduling_options = _scheduling_options or _RunnerSchedulingOptions()

    def run(
        self,
        plan: Plan,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Any:
        """Execute one plan and return its canonical public result."""

        if isinstance(plan, StudyPlan):
            return self._run_study(plan, cancellation=cancellation)
        try:
            return self._run_leaf(plan, cancellation=cancellation)
        except _CancellationRequested as exc:
            raise PlanCancelledError(
                None,
                (plan.plan_kind,),
            ) from exc

    def _run_leaf(
        self,
        plan: LeafPlan,
        *,
        cancellation: CancellationToken | None,
    ) -> Any:
        _check_cancellation(cancellation)

        if isinstance(plan, ThresholdPlan):
            return self._run_threshold(plan, cancellation=cancellation)
        if isinstance(plan, SweepPlan):
            return self._run_sweep(plan, cancellation=cancellation)
        if isinstance(plan, NumericAxisPlan):
            dispatch_plan = expand_dispatch_plan_for_numeric_axis(
                self._dispatch_plan(plan.source),
                plan.axis_input,
            )
            return self._run_simulation(plan.source, dispatch_plan=dispatch_plan)
        if isinstance(plan, SimulationPlan):
            return self._run_simulation(
                plan,
                dispatch_plan=self._dispatch_plan(plan),
            )
        raise TypeError(f"Runner cannot execute {type(plan).__name__}.")

    def run_many(
        self,
        plans: Sequence[LeafPlan],
        *,
        cancellation: CancellationToken | None = None,
    ) -> tuple[Any, ...]:
        """Execute plans in order through shared runner-owned state."""

        plan_tuple = tuple(plans)
        if not plan_tuple:
            return ()
        result = self.run(
            StudyPlan.from_plans(plan_tuple),
            cancellation=cancellation,
        )
        return result.values

    def _run_study(
        self,
        study: StudyPlan,
        *,
        cancellation: CancellationToken | None,
    ) -> StudyResult:
        ordered = study.ordered_tasks()
        completed_keys: list[str] = []
        completed_values: list[Any] = []
        for index, task in enumerate(ordered):
            try:
                _check_cancellation(cancellation)
                with benchmark_span(
                    "runner.study.task",
                    study_name=study.name,
                    task_key=task.key,
                    task_index=index,
                    task_count=len(ordered),
                    depends_on=task.depends_on,
                ):
                    value = self._run_leaf(
                        task.plan,
                        cancellation=cancellation,
                    )
            except _CancellationRequested as exc:
                raise PlanCancelledError(
                    StudyResult(
                        study.name,
                        tuple(completed_keys),
                        tuple(completed_values),
                    ),
                    tuple(item.key for item in ordered[index:]),
                ) from exc
            except Exception as exc:
                raise StudyExecutionError(
                    task.key,
                    StudyResult(
                        study.name,
                        tuple(completed_keys),
                        tuple(completed_values),
                    ),
                    tuple(item.key for item in ordered[index + 1 :]),
                    exc,
                ) from exc
            completed_keys.append(task.key)
            completed_values.append(value)
        return StudyResult(
            study.name,
            tuple(completed_keys),
            tuple(completed_values),
        )

    def estimate(self, plan: Plan, **kwargs: Any):
        """Estimate peak memory and repeated work without executing kernels."""

        from axonfleet.performance import PlanEstimate, PlanEstimateComponent

        if isinstance(plan, SimulationPlan):
            return self._estimate_simulation(plan, **kwargs)

        if isinstance(plan, StudyPlan):
            components = []
            notes: list[str] = []
            for task in plan.ordered_tasks():
                task_estimate = self.estimate(task.plan, **kwargs)
                peak, minimum, maximum, task_notes = _estimate_summary(task_estimate)
                components.append(
                    PlanEstimateComponent(
                        key=task.key,
                        plan_kind=task.plan.plan_kind,
                        expected_rows=task.plan.expected_rows,
                        simulation_executions_min=minimum,
                        simulation_executions_max=maximum,
                        peak=peak,
                        depends_on=task.depends_on,
                    )
                )
                notes.extend(f"{task.key}: {note}" for note in task_notes)
            return PlanEstimate(
                plan_kind=plan.plan_kind,
                expected_rows=plan.expected_rows,
                simulation_executions_min=sum(
                    component.simulation_executions_min for component in components
                ),
                simulation_executions_max=sum(
                    component.simulation_executions_max for component in components
                ),
                peak=_largest_estimate(component.peak for component in components),
                components=tuple(components),
                name=plan.name,
                notes=(
                    "Study peak is the largest task peak because tasks are currently "
                    "scheduled sequentially.",
                    *notes,
                ),
            )

        peak, minimum, maximum, notes = self._estimate_composed_leaf(plan, **kwargs)
        return PlanEstimate(
            plan_kind=plan.plan_kind,
            expected_rows=plan.expected_rows,
            simulation_executions_min=minimum,
            simulation_executions_max=maximum,
            peak=peak,
            components=(
                PlanEstimateComponent(
                    key=plan.plan_kind,
                    plan_kind=plan.plan_kind,
                    expected_rows=plan.expected_rows,
                    simulation_executions_min=minimum,
                    simulation_executions_max=maximum,
                    peak=peak,
                ),
            ),
            notes=notes,
        )

    def _estimate_simulation(
        self,
        plan: SimulationPlan,
        *,
        numeric_axis: Any = None,
        **kwargs: Any,
    ):
        from axonfleet.performance import estimate_simulation

        policy_kwargs: dict[str, Any] = {}
        if plan.execution_policy is not None:
            policy_kwargs["runtime"] = plan.execution_policy.runtime
            policy_kwargs["device"] = plan.execution_policy.device
            if plan.execution_policy.precision is not None:
                policy_kwargs["precision"] = plan.execution_policy.precision
        policy_kwargs.update(kwargs)
        return estimate_simulation(
            self._population(plan.population),
            duration=plan.duration,
            dt=plan.dt,
            recording=plan.recording,
            batch_options=plan.batch_options,
            observers=plan.observers,
            population_lifecycle=True,
            numeric_axis=numeric_axis,
            **policy_kwargs,
        )

    def _estimate_composed_leaf(self, plan: LeafPlan, **kwargs: Any):
        from axonfleet.protocols.types import NumericAxisUpdate

        if isinstance(plan, NumericAxisPlan):
            return (
                self._estimate_simulation(
                    plan.source,
                    numeric_axis=plan.axis_input,
                    **kwargs,
                ),
                1,
                1,
                ("Numeric-axis values share one compact simulation execution.",),
            )
        if isinstance(plan, SweepPlan):
            execution_count = sum(1 for _ in plan.value_batches())
            if not plan.values:
                return None, 0, 0, ("Empty sweep performs no simulation work.",)
            if isinstance(plan.update, NumericAxisUpdate):
                builder = plan.update.prepare_numeric_axis(
                    self._population(plan.source.population).instances
                )
                estimates = tuple(
                    self._estimate_simulation(
                        plan.source,
                        numeric_axis=builder.numeric_axis_input(values),
                        **kwargs,
                    )
                    for _, values in plan.value_batches()
                )
                return (
                    _largest_estimate(estimates),
                    execution_count,
                    execution_count,
                    (
                        "Sweep peak includes the largest compact numeric-axis batch; "
                        "batches are repeated work, not concurrent memory.",
                    ),
                )
            return (
                self._estimate_simulation(plan.source, **kwargs),
                execution_count,
                execution_count,
                (
                    "Generic sweep updates are assumed to preserve source shapes; "
                    "use a typed numeric-axis update for exact batched peak analysis.",
                ),
            )
        if isinstance(plan, ThresholdPlan):
            return (
                self._estimate_simulation(plan.source, **kwargs),
                2,
                2 + plan.max_iterations,
                (
                    "Threshold work includes two bound evaluations and up to "
                    f"{plan.max_iterations} bisection evaluations.",
                    "Threshold updates are assumed to preserve source shapes.",
                ),
            )
        raise TypeError(f"Runner cannot estimate {type(plan).__name__}.")

    def inspect(self, plan: Plan, *, print_summary: bool = False):
        """Inspect planning and repeated work without executing runtime kernels."""

        from axonfleet.inspection import PlanInspection, PlanInspectionComponent

        if isinstance(plan, SimulationPlan):
            return self._inspect_simulation(plan, print_summary=print_summary)

        if isinstance(plan, StudyPlan):
            components = []
            notes: list[str] = []
            for task in plan.ordered_tasks():
                task_inspection = self.inspect(task.plan)
                simulation, minimum, maximum, task_notes = _inspection_summary(
                    task_inspection
                )
                components.append(
                    PlanInspectionComponent(
                        key=task.key,
                        plan_kind=task.plan.plan_kind,
                        expected_rows=task.plan.expected_rows,
                        simulation_executions_min=minimum,
                        simulation_executions_max=maximum,
                        simulation=simulation,
                        depends_on=task.depends_on,
                    )
                )
                notes.extend(f"{task.key}: {note}" for note in task_notes)
            inspection = PlanInspection(
                plan_kind=plan.plan_kind,
                expected_rows=plan.expected_rows,
                simulation_executions_min=sum(
                    component.simulation_executions_min for component in components
                ),
                simulation_executions_max=sum(
                    component.simulation_executions_max for component in components
                ),
                components=tuple(components),
                name=plan.name,
                notes=tuple(notes),
            )
        else:
            simulation, minimum, maximum, notes = self._inspect_composed_leaf(plan)
            inspection = PlanInspection(
                plan_kind=plan.plan_kind,
                expected_rows=plan.expected_rows,
                simulation_executions_min=minimum,
                simulation_executions_max=maximum,
                components=(
                    PlanInspectionComponent(
                        key=plan.plan_kind,
                        plan_kind=plan.plan_kind,
                        expected_rows=plan.expected_rows,
                        simulation_executions_min=minimum,
                        simulation_executions_max=maximum,
                        simulation=simulation,
                    ),
                ),
                notes=notes,
            )
        if print_summary:
            inspection.print()
        return inspection

    def _inspect_simulation(
        self,
        plan: SimulationPlan,
        *,
        numeric_axis: Any = None,
        print_summary: bool = False,
    ):
        from axonfleet.inspection import inspect_simulation

        return inspect_simulation(
            self._population(plan.population),
            duration=plan.duration,
            dt=plan.dt,
            recording=plan.recording,
            batch_options=plan.batch_options,
            observers=plan.observers,
            execution_policy=plan.execution_policy,
            print_summary=print_summary,
            numeric_axis=numeric_axis,
        )

    def _inspect_composed_leaf(self, plan: LeafPlan):
        from axonfleet.protocols.types import NumericAxisUpdate

        if isinstance(plan, NumericAxisPlan):
            return (
                self._inspect_simulation(plan.source, numeric_axis=plan.axis_input),
                1,
                1,
                ("Numeric-axis values share one compact simulation execution.",),
            )
        if isinstance(plan, SweepPlan):
            execution_count = sum(1 for _ in plan.value_batches())
            if not plan.values:
                return None, 0, 0, ("Empty sweep performs no simulation work.",)
            if isinstance(plan.update, NumericAxisUpdate):
                builder = plan.update.prepare_numeric_axis(
                    self._population(plan.source.population).instances
                )
                largest_values = max(
                    (values for _, values in plan.value_batches()),
                    key=len,
                )
                simulation = self._inspect_simulation(
                    plan.source,
                    numeric_axis=builder.numeric_axis_input(largest_values),
                )
                notes = ("Inspection uses the largest compact numeric-axis batch.",)
            else:
                simulation = self._inspect_simulation(plan.source)
                notes = (
                    "Generic sweep updates are assumed to preserve source shapes.",
                )
            return simulation, execution_count, execution_count, notes
        if isinstance(plan, ThresholdPlan):
            return (
                self._inspect_simulation(plan.source),
                2,
                2 + plan.max_iterations,
                ("Threshold updates are assumed to preserve source shapes.",),
            )
        raise TypeError(f"Runner cannot inspect {type(plan).__name__}.")

    def clear(self) -> None:
        """Drop runner-owned population, planning, and host preparation state."""

        self._dispatch_plans.clear()
        self._populations.clear()
        self._prepared_cohorts.clear()

    def _population(self, plan: PopulationPlan) -> AxonPopulation:
        """Materialize and retain one concrete population for a plan."""

        key = id(plan)
        cached = self._populations.get(key)
        if cached is not None and cached.plan is plan:
            record_benchmark_metadata(runner_population_cache="hit")
            return cached.population
        source = plan.source
        if isinstance(source, AxonPopulation):
            population = source
        else:
            try:
                population = AxonPopulation(source)
            except (TypeError, ValueError) as exc:
                message = str(exc).replace("AxonPopulation", "AxonSimulation")
                raise type(exc)(message) from exc
        if len(population) != plan.expected_rows:
            raise RuntimeError("population source changed after its plan was created.")
        record_benchmark_metadata(runner_population_cache="miss")
        self._populations[key] = _PopulationCacheEntry(plan, population)
        return population

    def _dispatch_plan(self, plan: SimulationPlan) -> DispatchPlan:
        population = self._population(plan.population)
        revision = simulation_structure_revision()
        population_key = id(population)
        cached = self._dispatch_plans.get(population_key)
        if cached is not None and cached.population is population:
            if cached.revision == revision:
                self._dispatch_plans.move_to_end(population_key)
                record_benchmark_metadata(runner_dispatch_plan_cache="identity-hit")
                return cached.dispatch_plan
            identity = dispatch_plan_identity_key(population.instances)
            if cached.identity == identity:
                cached.revision = revision
                self._dispatch_plans.move_to_end(population_key)
                record_benchmark_metadata(runner_dispatch_plan_cache="structural-hit")
                return cached.dispatch_plan
        else:
            identity = dispatch_plan_identity_key(population.instances)

        record_benchmark_metadata(runner_dispatch_plan_cache="miss")
        dispatch_plan = build_dispatch_plan(population.instances)
        if self._dispatch_cache_size:
            self._dispatch_plans[population_key] = _DispatchCacheEntry(
                population=population,
                revision=revision,
                identity=identity,
                dispatch_plan=dispatch_plan,
            )
            self._dispatch_plans.move_to_end(population_key)
            while len(self._dispatch_plans) > self._dispatch_cache_size:
                self._dispatch_plans.popitem(last=False)
        return dispatch_plan

    def _run_sweep(
        self,
        plan: SweepPlan,
        *,
        cancellation: CancellationToken | None,
    ):
        from axonfleet.protocols.progress import _OneShotProgress, _SweepProgress
        from axonfleet.protocols.results import PoolSweepResult
        from axonfleet.protocols.types import NumericAxisUpdate

        source_size = plan.source.expected_rows
        if not plan.values:
            return PoolSweepResult(
                values=plan.values,
                observations=np.zeros((0, source_size), dtype=object),
            )

        numeric_axis = isinstance(plan.update, NumericAxisUpdate)
        if not numeric_axis and plan.value_batch_size != 1:
            raise ValueError(
                "value batching requires a typed NumericAxisUpdate."
            )
        input_builder = (
            plan.update.prepare_numeric_axis(
                self._population(plan.source.population).instances
            )
            if numeric_axis
            else None
        )
        progress_display = _SweepProgress(plan.progress)
        solver_progress = _OneShotProgress(plan.source.progress)
        observation_rows: list[np.ndarray] = []
        try:
            for batch_index, (start_index, values) in enumerate(plan.value_batches()):
                _check_cancellation(cancellation)
                progress_display.begin(
                    label="Pool sweep",
                    current_index=start_index,
                    values=plan.values,
                    completed_rows=observation_rows,
                    progress_summary=plan.progress_summary,
                )
                with benchmark_span(
                    "protocol.sweep.value_batch",
                    batch_index=batch_index,
                    start_index=start_index,
                    value_count=len(values),
                    pool_size=source_size,
                    execution_representation=(
                        "numeric_axis" if numeric_axis else "updated_population"
                    ),
                ):
                    source = replace(
                        plan.source,
                        progress=solver_progress.consume(),
                    )
                    if input_builder is not None:
                        execution_plan: Plan = source.with_numeric_axis(
                            input_builder.numeric_axis_input(values)
                        )
                    else:
                        updated = tuple(
                            _apply_sweep_update(row, plan.update, values[0])
                            for row in self._population(source.population).instances
                        )
                        execution_plan = replace(
                            source,
                            population=PopulationPlan(updated),
                        )
                    result = self._run_leaf(
                        execution_plan,
                        cancellation=cancellation,
                    )
                    decoded = np.asarray(plan.decode(result))

                expected = len(values) * source_size
                if decoded.ndim == 0 or decoded.shape[0] != expected:
                    raise ValueError(
                        "sweep decoder must return one observation per represented "
                        f"row; got shape {decoded.shape}, expected leading size {expected}."
                    )
                decoded = decoded.reshape((len(values), source_size, *decoded.shape[1:]))
                for offset, observations in enumerate(decoded):
                    observation_rows.append(observations)
                    progress_display.update(
                        label="Pool sweep",
                        current_index=start_index + offset,
                        values=plan.values,
                        completed_rows=observation_rows,
                        progress_summary=plan.progress_summary,
                    )
        finally:
            progress_display.close()

        return PoolSweepResult(
            values=plan.values,
            observations=np.stack(observation_rows, axis=0),
        )

    def _run_threshold(
        self,
        plan: ThresholdPlan,
        *,
        cancellation: CancellationToken | None,
    ):
        from axonfleet.protocols.progress import _OneShotProgress, _ThresholdProgress
        from axonfleet.protocols.results import ThresholdCurve
        from axonfleet.protocols.values import (
            _resolve_threshold_bounds,
            _threshold_converged,
        )
        from axonfleet.utils import units

        row_count = plan.expected_rows
        low_vector, high_vector = _resolve_threshold_bounds(
            plan.bounds,
            plan.row_labels,
        )
        progress_display = _ThresholdProgress(plan.progress)
        solver_progress = _OneShotProgress(plan.source.progress)
        tested: list[np.ndarray] = []
        satisfied_history: list[np.ndarray] = []

        def evaluate(values_uA: np.ndarray) -> np.ndarray:
            _check_cancellation(cancellation)
            values = np.asarray(values_uA, dtype=float)
            if values.shape != (row_count,):
                raise ValueError(
                    "threshold values must contain one value per source row; "
                    f"got {values.shape}."
                )
            updated_rows = []
            for row, value_uA in zip(
                self._population(plan.source.population).instances,
                values,
                strict=True,
            ):
                replacement = plan.update(
                    row,
                    units.Q_(float(value_uA), "microampere"),
                )
                updated_rows.append(row if replacement is None else replacement)
            updated = tuple(updated_rows)
            source = replace(
                plan.source,
                population=PopulationPlan(updated),
                progress=solver_progress.consume(),
            )
            with benchmark_span(
                "protocol.threshold.evaluate",
                pool_size=row_count,
            ):
                decoded = np.asarray(
                    plan.decode(
                        self._run_leaf(source, cancellation=cancellation)
                    ),
                    dtype=bool,
                )
            if decoded.shape != (row_count,):
                raise ValueError(
                    "threshold decoder must return one boolean per source row; "
                    f"got {decoded.shape}."
                )
            tested.append(values.copy())
            satisfied_history.append(decoded.copy())
            return decoded

        try:
            low_events = evaluate(low_vector)
            status = np.full(row_count, "threshold", dtype=object)
            high_events = evaluate(high_vector)

            threshold_uA = np.full(row_count, np.nan, dtype=float)
            below_mask = low_events
            above_mask = ~high_events
            status[below_mask] = "below_range"
            status[above_mask] = "above_range"
            threshold_uA[below_mask] = low_vector[below_mask]

            inactive_uA = low_vector.copy()
            active_uA = high_vector.copy()
            active_uA[below_mask] = low_vector[below_mask]
            unresolved = status == "threshold"
            progress_display.update(
                iteration="bounds",
                rows=plan.row_labels,
                tested_uA=high_vector,
                satisfied=high_events,
                lower_bound_uA=inactive_uA,
                upper_bound_uA=active_uA,
                status=status,
            )

            for iteration in range(1, plan.max_iterations + 1):
                if not np.any(unresolved):
                    break
                if np.all(
                    _threshold_converged(
                        inactive_uA[unresolved],
                        active_uA[unresolved],
                        tolerance_uA=plan.tolerance_uA,
                        relative_tolerance=plan.relative_tolerance,
                    )
                ):
                    break

                midpoint_uA = 0.5 * (inactive_uA + active_uA)
                events = evaluate(midpoint_uA)
                active_uA[unresolved & events] = midpoint_uA[unresolved & events]
                inactive_uA[unresolved & ~events] = midpoint_uA[unresolved & ~events]
                unresolved = (status == "threshold") & ~_threshold_converged(
                    inactive_uA,
                    active_uA,
                    tolerance_uA=plan.tolerance_uA,
                    relative_tolerance=plan.relative_tolerance,
                )
                progress_display.update(
                    iteration=str(iteration),
                    rows=plan.row_labels,
                    tested_uA=midpoint_uA,
                    satisfied=events,
                    lower_bound_uA=inactive_uA,
                    upper_bound_uA=active_uA,
                    status=status,
                )
        finally:
            progress_display.close()

        threshold_mask = status == "threshold"
        threshold_uA[threshold_mask] = active_uA[threshold_mask]
        return ThresholdCurve(
            row_labels=plan.row_labels,
            threshold_uA=threshold_uA,
            lower_bound_uA=inactive_uA,
            upper_bound_uA=active_uA,
            status=tuple(str(value) for value in status.tolist()),
            tested_uA=tuple(tested),
            satisfied=tuple(satisfied_history),
        )

    def _run_simulation(
        self,
        plan: SimulationPlan,
        *,
        dispatch_plan: DispatchPlan,
    ) -> AxonSimulationResult:
        with benchmark_span("simulation.setup"):
            population = self._population(plan.population)
            observer_defs = plan.observers
            recording = plan.recording
            if (
                recording is not None
                and not recording.voltage
                and not recording.wants_observables
                and not observer_defs
            ):
                raise NotImplementedError(
                    "Recording.none() requires solver-side observers."
                )
            duration_ms, step_ms = resolve_time(duration=plan.duration, dt=plan.dt)
            resolved_batch_options = batch_options_from_recording(
                recording,
                batch_options=plan.batch_options,
            )
            recording_plan = None if recording is None else recording.to_plan()

        context_manager = execution_context(
            plan.execution_policy,
            instances=population.instances,
        )
        with benchmark_span("simulation.execution_context.enter"):
            context = context_manager.__enter__()
        try:
            with benchmark_span("simulation.run_pool"):
                results = self._execute_dispatch_plan(
                    dispatch_plan,
                    tsim_ms=duration_ms,
                    dt_ms=step_ms,
                    batch_options=resolved_batch_options,
                    observers=observer_defs,
                    recording_plan=recording_plan,
                    progress=plan.progress,
                    runtime_context=context,
                )
        except BaseException as exc:
            with benchmark_span("simulation.execution_context.exit"):
                suppress = context_manager.__exit__(type(exc), exc, exc.__traceback__)
            if not suppress:
                raise
        else:
            with benchmark_span("simulation.execution_context.exit"):
                context_manager.__exit__(None, None, None)

        with benchmark_span("results.to_public", pool_size=len(population.instances)):
            if recording is not None:
                results = _filter_pool_recording(results, recording)
            return AxonSimulationResult.from_dispatch_results(
                results,
                recording=recording,
            )

    def _execute_dispatch_plan(
        self,
        plan: DispatchPlan,
        *,
        tsim_ms: float,
        dt_ms: float,
        batch_options: BatchOptions | None = None,
        observers: tuple[Any, ...] | None = None,
        recording_plan: Any = None,
        progress: Any = False,
        runtime_context: Any = None,
    ) -> tuple[DispatchRecord, ...]:
        """Execute one lowered dispatch plan through runner-owned scheduling."""

        resolved_batch_options = (
            BatchOptions.full() if batch_options is None else batch_options
        )
        emit_initial_progress(
            progress,
            rows=len(plan.items),
            message="building dispatch plan",
        )
        record_benchmark_metadata(
            dispatch_plan_source="runner",
            dispatch_group_count=len(plan.groups),
        )
        with benchmark_span(
            "simulation.pool.total",
            pool_size=len(plan.items),
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
        ):
            with DispatchProgress(progress, plan) as progress_reporter:
                if self._scheduling_options.async_groups:
                    return self._execute_async_groups(
                        plan,
                        tsim_ms=tsim_ms,
                        dt_ms=dt_ms,
                        batch_options=resolved_batch_options,
                        observers=observers,
                        recording_plan=recording_plan,
                        progress_reporter=progress_reporter,
                        runtime_context=runtime_context,
                        preparation_cache=self._prepared_cohorts,
                    )
                return self._execute_sync_groups(
                    plan,
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                    batch_options=resolved_batch_options,
                    observers=observers,
                    recording_plan=recording_plan,
                    progress_reporter=progress_reporter,
                    runtime_context=runtime_context,
                    preparation_cache=self._prepared_cohorts,
                )

    def _execute_sync_groups(
        self,
        plan: DispatchPlan,
        **kwargs: Any,
    ) -> tuple[DispatchRecord, ...]:
        results: list[DispatchRecord] = []
        seen_indices: set[int] = set()
        progress_reporter = kwargs.pop("progress_reporter")
        for group in plan.groups:
            with _dispatch_group_span(group):
                progress_reporter.start_group(group)
                progress_reporter.route_group(
                    group,
                    route=group.dispatch_method,
                    reason="planned batch route",
                )
                group_results = run_batch_group(
                    group,
                    progress_callback=progress_reporter.kernel_callback(group),
                    **kwargs,
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

    def _execute_async_groups(
        self,
        plan: DispatchPlan,
        **kwargs: Any,
    ) -> tuple[DispatchRecord, ...]:
        results: list[DispatchRecord] = []
        seen_indices: set[int] = set()
        pending: list[Any] = []
        pending_groups: list[DispatchGroup] = []
        flush_count = 0
        pending_max = 0
        progress_reporter = kwargs.pop("progress_reporter")
        max_pending = int(self._scheduling_options.max_pending_groups)
        record_benchmark_metadata(
            dispatch_async_groups=True,
            dispatch_async_max_pending_groups=max_pending,
        )
        for group in plan.groups:
            with _dispatch_group_span(group, dispatch_schedule="async_enqueue"):
                progress_reporter.start_group(group)
                progress_reporter.route_group(
                    group,
                    route=group.dispatch_method,
                    reason="planned batch route",
                )
                pending.append(
                    enqueue_batch_group(
                        group,
                        progress_callback=progress_reporter.kernel_callback(group),
                        **kwargs,
                    )
                )
                pending_groups.append(group)
                pending_max = max(pending_max, len(pending))
            if len(pending) >= max_pending:
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
            dispatch_async_flush_count=flush_count,
            dispatch_async_pending_max=pending_max,
        )
        _validate_dispatch_results(results, seen_indices=seen_indices, plan=plan)
        return tuple(results)


def _check_cancellation(cancellation: CancellationToken | None) -> None:
    if cancellation is not None and cancellation.cancelled:
        raise _CancellationRequested


def _dispatch_group_span(group: DispatchGroup, **metadata: Any):
    return benchmark_span(
        "dispatch.group.total",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        batch_kind=group.batch_kind,
        nx=group.nx,
        geometry_shared=group.geometry_shared,
        has_padding=group.has_padding,
        **metadata,
    )


def _is_complete_single_cohort_result(
    plan: DispatchPlan,
    *,
    group: DispatchGroup,
    group_results: tuple[DispatchRecord, ...],
) -> bool:
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


def _flush_pending_batch_groups(
    pending: list[Any],
    pending_groups: list[DispatchGroup],
    *,
    flush_count: int,
    progress_reporter: DispatchProgress,
) -> tuple[tuple[DispatchRecord, ...], int]:
    if not pending:
        return (), flush_count
    out: list[DispatchRecord] = []
    with benchmark_span(
        "dispatch.async_flush",
        group_count=len(pending),
        row_count=sum(group.size for group in pending_groups),
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

def _filter_pool_recording(
    results: Sequence[DispatchRecord],
    recording: Recording,
) -> tuple[DispatchRecord, ...]:
    """Apply public recording selection to dispatcher rows."""

    if not recording.voltage and not recording.wants_observables:
        return tuple(results)
    recording_plan = recording.to_plan()
    filtered = []
    for axon_result in results:
        if hasattr(axon_result, "indices"):
            filtered.append(axon_result)
            continue
        vm = axon_result.Vm
        record_indices = axon_result.record_indices
        if recording.voltage and vm is not None and record_indices is None:
            indices = recording_plan.indices_for(int(axon_result.axon.n_compartments))
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


def _apply_sweep_update(row: Any, update: Any, value: Any) -> Any:
    updated = update(row, value)
    return row if updated is None else updated


def _largest_estimate(estimates: Any):
    candidates = tuple(estimate for estimate in estimates if estimate is not None)
    if not candidates:
        return None
    return max(candidates, key=lambda estimate: estimate.total_bytes)


def _estimate_summary(estimate: Any):
    from axonfleet.performance import PlanEstimate

    if isinstance(estimate, PlanEstimate):
        return (
            estimate.peak,
            estimate.simulation_executions_min,
            estimate.simulation_executions_max,
            estimate.notes,
        )
    return estimate, 1, 1, ()


def _inspection_summary(inspection: Any):
    from axonfleet.inspection import PlanInspection

    if isinstance(inspection, PlanInspection):
        simulations = tuple(
            component.simulation
            for component in inspection.components
            if component.simulation is not None
        )
        representative = max(
            simulations,
            key=lambda report: sum(
                memory.total_estimated_bytes for memory in report.memory
            ),
            default=None,
        )
        return (
            representative,
            inspection.simulation_executions_min,
            inspection.simulation_executions_max,
            inspection.notes,
        )
    return inspection, 1, 1, ()


def _filter_recording_payload(
    recordings: dict[str, Any] | None,
    *,
    recording: Recording,
    vm: Any | None,
) -> dict[str, Any] | None:
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


__all__ = ["Runner"]
