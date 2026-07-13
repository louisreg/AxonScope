"""Observer-only protocol execution paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from axonscope.analysis import ActivationCriterion
from axonscope.analysis.definitions import Activation
from axonscope.protocols.types import SimulationCandidate
from axonscope.recording import Recording
from axonscope.results import VM_RASTER_OBSERVATION_KEY, activation_values_from_vm_raster
from axonscope.runtime import ExecutionPolicy
from axonscope.simulation import AxonSimulation
from axonscope.solvers import BatchOptions


def _evaluate_activation_observer_pool(
    pool: tuple[SimulationCandidate, ...],
    *,
    criterion: ActivationCriterion,
    duration: Any,
    dt: Any,
    progress: bool | str,
    batch_options: BatchOptions | None,
    execution_policy: ExecutionPolicy | None = None,
) -> np.ndarray:
    """Evaluate activation through compact solver-side observers."""

    activation = _activation_observer_definition(criterion)
    pool_result = AxonSimulation(
        axons=pool,  # type: ignore[arg-type]
        duration=duration,
        dt=dt,
        recording=Recording.none(),
        batch_options=batch_options,
        execution_policy=execution_policy,
        observers=(activation,),
        progress=progress,
    ).run()
    return _activation_observations_from_pool_result(pool_result, activation)


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
    criterion: ActivationCriterion,
) -> bool:
    if bool(getattr(criterion, "require_propagation", False)):
        return False
    return recording is None or (
        isinstance(recording, Recording)
        and not recording.voltage
        and not recording.wants_observables
    )


def _can_use_threshold_observer(
    recording: Recording | None,
    criterion: ActivationCriterion,
) -> bool:
    return _can_use_activation_observer(recording, criterion)


def _activation_observer_definition(criterion: ActivationCriterion) -> Activation:
    return Activation(
        threshold=criterion.threshold,
        blanking=criterion.blanking,
        target=criterion.target,
    )


__all__ = [
    "_activation_observations_from_pool_result",
    "_can_use_activation_observer",
    "_can_use_threshold_observer",
    "_evaluate_activation_observer_pool",
]
