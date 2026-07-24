"""Observer-only protocol execution paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from axonfleet.analysis import Activation
from axonfleet.plans import SweepPlan
from axonfleet.protocols.types import SimulationCandidate
from axonfleet.protocols.progress import _activation_progress_summary
from axonfleet.recording import Recording
from axonfleet.results import VM_RASTER_OBSERVATION_KEY
from axonfleet.results.vm_raster import activation_values_from_vm_raster
from axonfleet.runtime import ExecutionPolicy
from axonfleet.simulation import AxonSimulation
from axonfleet.solvers import BatchOptions


@dataclass(frozen=True)
class _ActivationResultDecoder:
    activation: Activation

    def __call__(self, pool_result: Any) -> np.ndarray:
        return _activation_observations_from_pool_result(
            pool_result,
            self.activation,
        )


def _build_activation_observer_simulation(
    pool: tuple[SimulationCandidate, ...],
    *,
    criterion: Activation,
    duration: Any,
    dt: Any,
    progress: bool | str,
    batch_options: BatchOptions | None,
    execution_policy: ExecutionPolicy | None = None,
) -> tuple[AxonSimulation, Activation]:
    """Build a reusable observer-only simulation for one stable pool shape."""

    activation = criterion
    simulation = AxonSimulation(
        axons=pool,  # type: ignore[arg-type]
        duration=duration,
        dt=dt,
        recording=Recording.none(),
        batch_options=batch_options,
        execution_policy=execution_policy,
        observers=(activation,),
        progress=progress,
    )
    return simulation, activation


def _activation_observer_sweep_plan(
    pool: tuple[SimulationCandidate, ...],
    *,
    update: Any,
    values: tuple[Any, ...],
    value_batch_size: int,
    criterion: Activation,
    duration: Any,
    dt: Any,
    progress: bool | str,
    batch_options: BatchOptions | None,
    execution_policy: ExecutionPolicy | None,
    solver_progress: bool | str,
) -> SweepPlan:
    """Build a lazy activation sweep with vectorized compact-result decoding."""

    simulation, activation = _build_activation_observer_simulation(
        pool,
        criterion=criterion,
        duration=duration,
        dt=dt,
        progress=solver_progress,
        batch_options=batch_options,
        execution_policy=execution_policy,
    )
    return SweepPlan(
        source=simulation.plan(),
        values=values,
        update=update,
        decode=_ActivationResultDecoder(activation),
        value_batch_size=value_batch_size,
        progress=progress,
        progress_summary=_activation_progress_summary,
    )


def _activation_observations_from_pool_result(
    pool_result: Any,
    activation: Activation,
) -> np.ndarray:
    """Return activation flags for all pool rows without slicing row-by-row."""

    size = len(pool_result)
    cohorts = getattr(pool_result, "_cohorts", None)
    if cohorts:
        values = np.zeros(size, dtype=bool)
        filled = np.zeros(size, dtype=bool)
        for cohort in cohorts:
            observations = getattr(cohort, "observations", None)
            if observations is None:
                break
            cohort_values = _activation_observation_values(observations, activation)
            indices = np.asarray(getattr(cohort, "input_indices", ()), dtype=int)
            if cohort_values.shape != (len(indices),):
                raise RuntimeError(
                    "activation observer result width does not match cohort size; "
                    f"got {cohort_values.shape} for {len(indices)} rows."
                )
            values[indices] = cohort_values
            filled[indices] = True
        else:
            if bool(np.all(filled)):
                return values

    observations = getattr(pool_result, "observations", None)
    if observations is not None:
        values = _activation_observation_values(observations, activation)
        if values.shape == (size,):
            return values

    return np.asarray(
        [
            _activation_observation_activated(view, activation)
            for view in pool_result
        ],
        dtype=bool,
    )


def _activation_observation_values(
    observations: Any,
    activation: Activation,
) -> np.ndarray:
    if observations is None:
        raise RuntimeError("activation observer result is missing from solver output.")
    if activation.name in observations:
        return np.asarray(observations[activation.name].values, dtype=bool).reshape(-1)
    raster = observations.get(VM_RASTER_OBSERVATION_KEY)
    if raster is not None:
        return _activation_from_vm_raster_batch(raster, activation)
    raise RuntimeError("activation observer result is missing from solver output.")


def _activation_observation_activated(result: Any, activation: Activation) -> bool:
    observations = getattr(result, "observations", None)
    if observations is None:
        raise RuntimeError("activation observer result is missing from solver output.")
    if activation.name in observations:
        return bool(np.asarray(observations[activation.name].values)[0])
    raster = observations.get(VM_RASTER_OBSERVATION_KEY)
    if raster is not None:
        return _activation_from_vm_raster(raster, activation)
    raise RuntimeError("activation observer result is missing from solver output.")


def _activation_from_vm_raster(raster: Any, activation: Activation) -> bool:
    return bool(_activation_from_vm_raster_batch(raster, activation)[0])


def _activation_from_vm_raster_batch(raster: Any, activation: Activation) -> np.ndarray:
    return activation_values_from_vm_raster(raster, activation)


def _can_use_activation_observer(
    recording: Recording | None,
) -> bool:
    return recording is None or (
        isinstance(recording, Recording)
        and not recording.voltage
        and not recording.wants_observables
    )


__all__ = [
    "_activation_observations_from_pool_result",
    "_activation_observer_sweep_plan",
    "_build_activation_observer_simulation",
    "_can_use_activation_observer",
]
