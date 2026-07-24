"""Backend-neutral immutable descriptions of runnable simulation work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Generic, Iterator, Sequence, TypeVar

from axonfleet.axon_instance import AxonInstance
from axonfleet.axons.axon import Axon
from axonfleet.dispatcher.numeric_axis import NumericAxisInput
from axonfleet.dispatcher.progress import ProgressOption
from axonfleet.population import AxonPopulation
from axonfleet.recording import Recording
from axonfleet.runtime import ExecutionPolicy
from axonfleet.solvers import BatchOptions


ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class RunnablePlan(Generic[ResultT]):
    """Immutable backend-neutral description accepted by :class:`Runner`.

    A runnable plan contains Python descriptions only. Dispatch grouping,
    numerical lowering, device selection, compilation, and result allocation
    happen when a runner executes it.
    """

    plan_kind: ClassVar[str] = "runnable"


@dataclass(frozen=True, slots=True)
class PopulationPlan:
    """Ordered population description with no concrete instance materialization."""

    source: Axon | AxonInstance | Sequence[Axon | AxonInstance] | AxonPopulation

    def __post_init__(self) -> None:
        source = self.source
        if isinstance(source, (Axon, AxonInstance, AxonPopulation)):
            size = 1 if isinstance(source, (Axon, AxonInstance)) else len(source)
        else:
            try:
                source = tuple(source)
            except TypeError as exc:
                raise TypeError(
                    "PopulationPlan expects an Axon, AxonInstance, AxonPopulation, "
                    "or an ordered iterable of axon inputs."
                ) from exc
            size = len(source)
            object.__setattr__(self, "source", source)
        if size < 1:
            raise ValueError("PopulationPlan requires at least one source row.")

    @property
    def expected_rows(self) -> int:
        """Number of ordered rows represented without creating instances."""

        source = self.source
        if isinstance(source, (Axon, AxonInstance)):
            return 1
        return len(source)


@dataclass(frozen=True, slots=True)
class SimulationPlan(RunnablePlan[Any]):
    """One population simulation with no materialized runtime state."""

    population: PopulationPlan
    duration: Any
    dt: Any
    recording: Recording | None = None
    batch_options: BatchOptions | None = None
    observers: tuple[Any, ...] | None = None
    execution_policy: ExecutionPolicy | None = None
    progress: ProgressOption = False

    plan_kind: ClassVar[str] = "simulation"

    @property
    def expected_rows(self) -> int:
        """Number of result rows described by this plan."""

        return self.population.expected_rows

    def with_numeric_axis(self, axis_input: NumericAxisInput) -> "NumericAxisPlan":
        """Compose this simulation with one typed dynamic numeric axis."""

        return NumericAxisPlan(source=self, axis_input=axis_input)


@dataclass(frozen=True, slots=True)
class NumericAxisPlan(RunnablePlan[Any]):
    """One simulation repeated over a compact runtime numeric axis."""

    source: SimulationPlan
    axis_input: NumericAxisInput

    plan_kind: ClassVar[str] = "numeric_axis"

    def __post_init__(self) -> None:
        if int(self.axis_input.size) < 1:
            raise ValueError("numeric axis must contain at least one value.")

    @property
    def expected_rows(self) -> int:
        """Number of ordered result rows described by this plan."""

        return self.source.expected_rows * int(self.axis_input.size)


@dataclass(frozen=True, slots=True)
class SweepPlan(RunnablePlan[Any]):
    """Generic ordered value sweep over one simulation description.

    ``update`` describes how one value changes a source row. Typed numeric-axis
    updates are lowered compactly by the runner; ordinary callables are applied
    lazily immediately before each value is executed. ``decode`` converts one
    public simulation result into one observation per represented row.
    """

    source: SimulationPlan
    values: tuple[Any, ...]
    update: Any
    decode: Callable[[Any], Any]
    value_batch_size: int = 1
    progress: bool | str = False
    progress_summary: Callable[[Any], str] | None = None
    result_factory: Callable[[Any], Any] | None = None

    plan_kind: ClassVar[str] = "sweep"

    def __post_init__(self) -> None:
        batch_size = int(self.value_batch_size)
        if batch_size < 1:
            raise ValueError("value_batch_size must be >= 1.")
        if not callable(self.update):
            raise TypeError("sweep update must be callable.")
        if not callable(self.decode):
            raise TypeError("sweep decode must be callable.")
        if self.result_factory is not None and not callable(self.result_factory):
            raise TypeError("sweep result_factory must be callable or None.")
        object.__setattr__(self, "value_batch_size", batch_size)

    @property
    def expected_rows(self) -> int:
        """Total number of value-major result rows represented by the sweep."""

        return len(self.values) * self.source.expected_rows

    def value_batches(self) -> Iterator[tuple[int, tuple[Any, ...]]]:
        """Yield ordered bounded value chunks without expanding source rows."""

        for start in range(0, len(self.values), self.value_batch_size):
            yield start, self.values[start : start + self.value_batch_size]


@dataclass(frozen=True, slots=True)
class ThresholdPlan(RunnablePlan[Any]):
    """Per-row bounded threshold search over one simulation description."""

    source: SimulationPlan
    update: Any
    decode: Callable[[Any], Any]
    bounds: Any
    row_labels: tuple[Any, ...]
    tolerance_uA: float | None = 1.0
    relative_tolerance: float | None = None
    max_iterations: int = 20
    progress: bool | str = False

    plan_kind: ClassVar[str] = "threshold"

    def __post_init__(self) -> None:
        if not callable(self.update):
            raise TypeError("threshold update must be callable.")
        if not callable(self.decode):
            raise TypeError("threshold decode must be callable.")
        if len(self.row_labels) != self.source.expected_rows:
            raise ValueError("row_labels must contain one entry per source row.")
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be >= 1.")
        if self.tolerance_uA is not None and float(self.tolerance_uA) <= 0.0:
            raise ValueError("tolerance must be positive.")
        if self.relative_tolerance is not None and float(self.relative_tolerance) <= 0.0:
            raise ValueError("relative_tolerance must be positive.")
        if self.tolerance_uA is None and self.relative_tolerance is None:
            raise ValueError("Provide tolerance, relative_tolerance, or both.")
        object.__setattr__(self, "max_iterations", int(self.max_iterations))

    @property
    def expected_rows(self) -> int:
        """Number of independent thresholds represented by the plan."""

        return self.source.expected_rows


LeafPlan = SimulationPlan | NumericAxisPlan | SweepPlan | ThresholdPlan


@dataclass(frozen=True, slots=True)
class StudyTask:
    """One named study task and its prerequisite task keys."""

    key: str
    plan: LeafPlan
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("study task key must be a non-empty string.")
        if not isinstance(
            self.plan,
            (SimulationPlan, NumericAxisPlan, SweepPlan, ThresholdPlan),
        ):
            raise TypeError("StudyTask.plan must be a runnable leaf plan.")
        dependencies = tuple(self.depends_on)
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f"study task {self.key!r} repeats a dependency.")
        object.__setattr__(self, "depends_on", dependencies)


@dataclass(frozen=True, slots=True)
class StudyPlan(RunnablePlan[Any]):
    """Named backend-neutral study composed from runnable leaf tasks."""

    name: str
    tasks: tuple[StudyTask, ...]

    plan_kind: ClassVar[str] = "study"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("study name must be a non-empty string.")
        tasks = tuple(self.tasks)
        if not tasks:
            raise ValueError("StudyPlan requires at least one task.")
        object.__setattr__(self, "tasks", tasks)
        _ordered_task_indices(tasks)

    @property
    def expected_rows(self) -> int:
        """Combined result-row count described by all tasks."""

        return sum(task.plan.expected_rows for task in self.tasks)

    def ordered_tasks(self) -> tuple[StudyTask, ...]:
        """Return a stable topological order without executing any task."""

        return tuple(self.tasks[index] for index in _ordered_task_indices(self.tasks))


def _ordered_task_indices(tasks: tuple[StudyTask, ...]) -> tuple[int, ...]:
    key_to_index: dict[str, int] = {}
    for index, task in enumerate(tasks):
        if task.key in key_to_index:
            raise ValueError(f"StudyPlan repeats task key {task.key!r}.")
        key_to_index[task.key] = index

    indegree = [0] * len(tasks)
    dependents: list[list[int]] = [[] for _ in tasks]
    for index, task in enumerate(tasks):
        for dependency in task.depends_on:
            dependency_index = key_to_index.get(dependency)
            if dependency_index is None:
                raise ValueError(
                    f"study task {task.key!r} depends on unknown task {dependency!r}."
                )
            indegree[index] += 1
            dependents[dependency_index].append(index)

    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    ordered: list[int] = []
    while ready:
        index = ready.pop(0)
        ordered.append(index)
        for dependent in dependents[index]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
        ready.sort()
    if len(ordered) != len(tasks):
        raise ValueError("StudyPlan dependencies contain a cycle.")
    return tuple(ordered)


Plan = LeafPlan | StudyPlan


__all__ = [
    "LeafPlan",
    "NumericAxisPlan",
    "Plan",
    "PopulationPlan",
    "RunnablePlan",
    "SimulationPlan",
    "SweepPlan",
    "StudyPlan",
    "StudyTask",
    "ThresholdPlan",
]
