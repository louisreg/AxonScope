from __future__ import annotations

from typing import Any, Sequence, TypeAlias

from axonfleet.axon_instance import AxonInstance
from axonfleet.runtime import ExecutionPolicy
from axonfleet.axons.axon import Axon
from axonfleet.dispatcher.progress import ProgressOption
from axonfleet.plans import PopulationPlan, SimulationPlan
from axonfleet.population import AxonPopulation
from axonfleet.recording import Recording
from axonfleet.results import AxonSimulationResult
from axonfleet.runner import Runner
from axonfleet.solvers import BatchOptions


AxonInput: TypeAlias = Axon | AxonInstance
SimulationRunResult: TypeAlias = AxonSimulationResult


class AxonSimulation:
    """Executable simulation definition for one or more axon instances.

    `AxonInstance` describes one concrete axon occurrence and its local
    stimulation. `AxonSimulation` collects one or more axons/instances with
    execution parameters such as duration, step size, recording policy, and
    solver options. Input rows remain descriptive until ``Runner`` materializes
    one ``AxonPopulation`` for execution, estimation, inspection, or explicit
    access through :attr:`population`.
    """

    def __init__(
        self,
        axons: AxonInput | Sequence[AxonInput] | AxonPopulation,
        *,
        duration: Any,
        dt: Any,
        recording: Recording | None = None,
        batch_options: BatchOptions | None = None,
        observers: Sequence[Any] | None = None,
        execution_policy: ExecutionPolicy | None = None,
        progress: ProgressOption = False,
    ) -> None:
        self._population_plan = PopulationPlan(axons)
        self.duration = duration
        self.dt = dt
        self.recording = recording
        self.batch_options = batch_options
        self.observers = tuple(observers) if observers is not None else None
        self.execution_policy = execution_policy
        self.progress = progress
        self._runner: Runner | None = None

    @property
    def population(self) -> AxonPopulation:
        """Materialize the concrete population through this simulation's runner."""

        return self._execution_runner()._population(self._population_plan)

    @property
    def axons(self) -> tuple[AxonInstance, ...]:
        """Return concrete population rows, materializing them on first access."""

        return self.population.instances

    @property
    def is_single(self) -> bool:
        """Return whether this executable definition contains one axon."""

        return self._population_plan.expected_rows == 1

    @property
    def is_population(self) -> bool:
        """Return whether this definition uses the population lifecycle."""

        return True

    def run(self) -> SimulationRunResult:
        """Execute this simulation definition and return public results."""

        return self._execution_runner().run(self.plan())

    def plan(self) -> SimulationPlan:
        """Return an immutable backend-neutral description of this run."""

        return SimulationPlan(
            population=self._population_plan,
            duration=self.duration,
            dt=self.dt,
            recording=self.recording,
            batch_options=self.batch_options,
            observers=self.observers,
            execution_policy=self.execution_policy,
            progress=self.progress,
        )

    def estimate(self, **kwargs: Any):
        """Estimate memory pressure for this simulation without running it."""

        return self._execution_runner().estimate(self.plan(), **kwargs)

    def inspect(self, *, print_summary: bool = False):
        """Inspect planning, dispatch/batch grouping, and preparation."""

        return self._execution_runner().inspect(
            self.plan(),
            print_summary=print_summary,
        )

    def _execution_runner(self) -> Runner:
        if self._runner is None:
            self._runner = Runner()
        return self._runner


__all__ = ["AxonSimulation"]
