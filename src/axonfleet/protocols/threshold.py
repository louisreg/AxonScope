"""Threshold-search protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from axonfleet.analysis import Activation
from axonfleet.plans import ThresholdPlan
from axonfleet.protocols.observer_path import (
    _activation_observations_from_pool_result,
    _build_activation_observer_simulation,
    _can_use_activation_observer,
)
from axonfleet.protocols.results import ThresholdCurve
from axonfleet.protocols.types import PoolUpdate, SimulationCandidate
from axonfleet.protocols.values import _normalize_rows, _require_current_array_uA
from axonfleet.recording import Recording
from axonfleet.runtime import ExecutionPolicy
from axonfleet.runner import Runner
from axonfleet.simulation import AxonSimulation
from axonfleet.solvers import BatchOptions
from axonfleet.utils import units


@dataclass(frozen=True)
class _ThresholdActivationDecoder:
    activation: Activation
    observer_path: bool

    def __call__(self, pool_result: Any) -> np.ndarray:
        if self.observer_path:
            return _activation_observations_from_pool_result(
                pool_result,
                self.activation,
            )
        return np.asarray(
            [self.activation.detect(result).activated for result in pool_result],
            dtype=bool,
        )


def find_threshold(
    pool: Sequence[SimulationCandidate],
    *,
    update: PoolUpdate,
    bounds: tuple[Any, Any] | Callable[[Any], tuple[Any, Any]],
    duration: Any,
    dt: Any,
    criterion: Activation,
    rows: Sequence[Any] | None = None,
    tolerance: Any | None = 1.0,
    relative_tolerance: float | None = None,
    max_iterations: int = 20,
    recording: Recording | None = None,
    batch_options: BatchOptions | None = None,
    execution_policy: ExecutionPolicy | None = None,
    progress: bool | str = False,
    solver_progress: bool | str = False,
) -> ThresholdCurve:
    """Estimate one threshold per pool row with runner-owned bisection."""

    base_pool = tuple(pool)
    row_tuple = _normalize_rows(
        rows if rows is not None else tuple(range(len(base_pool)))
    )
    if len(row_tuple) != len(base_pool):
        raise ValueError(
            f"rows must contain one entry per pool row; got {len(row_tuple)} rows "
            f"for {len(base_pool)} pool entries."
        )
    if not base_pool:
        return ThresholdCurve(
            row_labels=row_tuple,
            threshold_uA=np.asarray([], dtype=float),
            lower_bound_uA=np.asarray([], dtype=float),
            upper_bound_uA=np.asarray([], dtype=float),
            status=(),
            tested_uA=(),
            satisfied=(),
        )
    return Runner().run(
        find_threshold_plan(
            base_pool,
            update=update,
            bounds=bounds,
            duration=duration,
            dt=dt,
            criterion=criterion,
            rows=row_tuple,
            tolerance=tolerance,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            recording=recording,
            batch_options=batch_options,
            execution_policy=execution_policy,
            progress=progress,
            solver_progress=solver_progress,
        )
    )


def find_threshold_plan(
    pool: Sequence[SimulationCandidate],
    *,
    update: PoolUpdate,
    bounds: tuple[Any, Any] | Callable[[Any], tuple[Any, Any]],
    duration: Any,
    dt: Any,
    criterion: Activation,
    rows: Sequence[Any] | None = None,
    tolerance: Any | None = 1.0,
    relative_tolerance: float | None = None,
    max_iterations: int = 20,
    recording: Recording | None = None,
    batch_options: BatchOptions | None = None,
    execution_policy: ExecutionPolicy | None = None,
    progress: bool | str = False,
    solver_progress: bool | str = False,
) -> ThresholdPlan:
    """Build a lazy per-row threshold plan without resolving its bounds."""

    base_pool = tuple(pool)
    if not base_pool:
        raise ValueError("find_threshold_plan requires at least one source row.")
    if not isinstance(criterion, Activation):
        raise TypeError("criterion must be axs.analysis.Activation.")
    row_tuple = _normalize_rows(
        rows if rows is not None else tuple(range(len(base_pool)))
    )
    if len(row_tuple) != len(base_pool):
        raise ValueError(
            f"rows must contain one entry per pool row; got {len(row_tuple)} rows "
            f"for {len(base_pool)} pool entries."
        )
    if not callable(bounds):
        _require_current_array_uA(bounds[0], name="bounds[0]")
        _require_current_array_uA(bounds[1], name="bounds[1]")
    tolerance_uA = (
        None
        if tolerance is None
        else units.require_current_uA(tolerance, name="tolerance")
    )

    observer_path = _can_use_activation_observer(recording)
    if observer_path:
        simulation, activation = _build_activation_observer_simulation(
            base_pool,
            criterion=criterion,
            duration=duration,
            dt=dt,
            progress=solver_progress,
            batch_options=batch_options,
            execution_policy=execution_policy,
        )
    else:
        activation = criterion
        simulation = AxonSimulation(
            axons=base_pool,
            duration=duration,
            dt=dt,
            recording=recording or Recording.voltage(),
            batch_options=batch_options,
            execution_policy=execution_policy,
            progress=solver_progress,
        )

    return ThresholdPlan(
        source=simulation.plan(),
        update=update,
        decode=_ThresholdActivationDecoder(activation, observer_path),
        bounds=bounds,
        row_labels=row_tuple,
        tolerance_uA=tolerance_uA,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
        progress=progress,
    )


__all__ = [
    "find_threshold",
    "find_threshold_plan",
]
