"""Threshold-search protocols."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from axonscope.analysis import ActivationCriterion, ActivationEvent
from axonscope.analysis.definitions import Activation
from axonscope.protocols.observer_path import (
    _can_use_threshold_observer,
    _evaluate_activation_observer_pool,
)
from axonscope.protocols.progress import _OneShotProgress, _ThresholdProgress
from axonscope.protocols.results import (
    ThresholdCurve,
    ThresholdHistoryEntry,
    ThresholdSearchResult,
)
from axonscope.protocols.types import (
    PoolUpdate,
    SimulationCandidate,
    SimulationFactory,
    ThresholdCriterion,
    ThresholdUpdate,
)
from axonscope.protocols.values import (
    _normalize_rows,
    _resolve_threshold_bounds,
    _threshold_converged,
)
from axonscope.recording import Recording
from axonscope.simulation import AxonSimulation
from axonscope.solvers import BatchOptions
from axonscope.utils import units


def find_activation_threshold(
    simulation_factory: SimulationFactory,
    *,
    bounds: tuple[Any, Any],
    duration: Any,
    dt: Any,
    criterion: ActivationCriterion,
    tolerance: Any = 1.0,
    max_iterations: int = 20,
    recording: Recording | None = None,
) -> ThresholdSearchResult:
    """Find the activation threshold by binary search.

    ``simulation_factory`` is the only place where stimulation changes between
    evaluations. The protocol passes each tested current amplitude as a Pint
    quantity; the factory should attach the desired stimulus, electrode,
    extracellular context, axon position, or any other parameter change and
    return a fresh simulation candidate.
    """

    low_uA = units.require_current_uA(bounds[0], name="bounds[0]")
    high_uA = units.require_current_uA(bounds[1], name="bounds[1]")
    tolerance_uA = units.require_current_uA(tolerance, name="tolerance")
    if high_uA <= low_uA:
        raise ValueError("bounds must be ordered as (low, high).")
    if tolerance_uA <= 0.0:
        raise ValueError("tolerance must be positive.")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be >= 1.")

    history: list[ThresholdHistoryEntry] = []

    low_event = _evaluate_activation(
        simulation_factory,
        low_uA,
        duration=duration,
        dt=dt,
        criterion=criterion,
        recording=recording,
    )
    history.append(
        ThresholdHistoryEntry(low_uA, low_event.activated, low_event)
    )
    if low_event.activated:
        return ThresholdSearchResult(
            amplitude_uA=low_uA,
            lower_bound_uA=low_uA,
            upper_bound_uA=low_uA,
            status="below_range",
            history=tuple(history),
        )

    high_event = _evaluate_activation(
        simulation_factory,
        high_uA,
        duration=duration,
        dt=dt,
        criterion=criterion,
        recording=recording,
    )
    history.append(
        ThresholdHistoryEntry(high_uA, high_event.activated, high_event)
    )
    if not high_event.activated:
        return ThresholdSearchResult(
            amplitude_uA=None,
            lower_bound_uA=low_uA,
            upper_bound_uA=high_uA,
            status="above_range",
            history=tuple(history),
        )

    inactive_uA = low_uA
    active_uA = high_uA
    for _ in range(int(max_iterations)):
        if active_uA - inactive_uA <= tolerance_uA:
            break
        midpoint_uA = 0.5 * (inactive_uA + active_uA)
        event = _evaluate_activation(
            simulation_factory,
            midpoint_uA,
            duration=duration,
            dt=dt,
            criterion=criterion,
            recording=recording,
        )
        history.append(
            ThresholdHistoryEntry(midpoint_uA, event.activated, event)
        )
        if event.activated:
            active_uA = midpoint_uA
        else:
            inactive_uA = midpoint_uA

    return ThresholdSearchResult(
        amplitude_uA=active_uA,
        lower_bound_uA=inactive_uA,
        upper_bound_uA=active_uA,
        status="threshold",
        history=tuple(history),
    )


def find_threshold(
    pool: Sequence[SimulationCandidate],
    *,
    update: ThresholdUpdate,
    bounds: tuple[Any, Any] | Callable[[Any], tuple[Any, Any]],
    duration: Any,
    dt: Any,
    criterion: ThresholdCriterion,
    rows: Sequence[Any] | None = None,
    tolerance: Any | None = 1.0,
    relative_tolerance: float | None = None,
    max_iterations: int = 20,
    recording: Recording | None = None,
    batch_options: BatchOptions | None = None,
    progress: bool | str = False,
    solver_progress: bool | str = False,
) -> ThresholdCurve:
    """Estimate one threshold per pool row with batched bisection.

    ``pool`` contains the simulations to evaluate. ``update(simulation,
    current)`` is called before each run to change the searched parameter,
    usually a stimulus amplitude. ``criterion`` may be an
    ``ActivationCriterion`` or ``Activation``. The update function may mutate
    the row and return ``None``, or return a replacement simulation. ``rows``
    optionally carries user-facing values such as
    diameters for plotting and callable bounds. ``solver_progress`` is
    forwarded only to the first solver call so cold-start compilation remains
    visible without logging every bisection run.
    """

    base_pool = tuple(pool)
    row_count = len(base_pool)
    row_tuple = _normalize_rows(rows if rows is not None else tuple(range(row_count)))
    if len(row_tuple) != row_count:
        raise ValueError(
            f"rows must contain one entry per pool row; got {len(row_tuple)} rows "
            f"for {row_count} pool entries."
        )
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be >= 1.")
    tolerance_uA = (
        None
        if tolerance is None
        else units.require_current_uA(tolerance, name="tolerance")
    )
    if tolerance_uA is not None and tolerance_uA <= 0.0:
        raise ValueError("tolerance must be positive.")
    if relative_tolerance is not None and float(relative_tolerance) <= 0.0:
        raise ValueError("relative_tolerance must be positive.")
    if tolerance_uA is None and relative_tolerance is None:
        raise ValueError("Provide tolerance, relative_tolerance, or both.")

    low_vector, high_vector = _resolve_threshold_bounds(bounds, row_tuple)
    if row_count == 0:
        return ThresholdCurve(
            row_labels=row_tuple,
            threshold_uA=np.asarray([], dtype=float),
            lower_bound_uA=np.asarray([], dtype=float),
            upper_bound_uA=np.asarray([], dtype=float),
            status=(),
            tested_uA=(),
            satisfied=(),
        )

    progress_display = _ThresholdProgress(progress)
    solver_progress_gate = _OneShotProgress(solver_progress)
    try:
        low_events = _evaluate_threshold_updated_pool(
            base_pool,
            update,
            low_vector,
            duration=duration,
            dt=dt,
            criterion=criterion,
            recording=recording,
            batch_options=batch_options,
            progress=solver_progress_gate.consume(),
        )
        tested: list[np.ndarray] = [low_vector.copy()]
        satisfied_history: list[np.ndarray] = [low_events.copy()]
        status = np.full(row_count, "threshold", dtype=object)

        high_events = _evaluate_threshold_updated_pool(
            base_pool,
            update,
            high_vector,
            duration=duration,
            dt=dt,
            criterion=criterion,
            recording=recording,
            batch_options=batch_options,
            progress=solver_progress_gate.consume(),
        )
        _validate_pool_width(high_events, row_count)
        tested.append(high_vector.copy())
        satisfied_history.append(high_events.copy())

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
            rows=row_tuple,
            tested_uA=high_vector,
            satisfied=high_events,
            lower_bound_uA=inactive_uA,
            upper_bound_uA=active_uA,
            status=status,
        )

        for iteration in range(1, int(max_iterations) + 1):
            if not np.any(unresolved):
                break
            if np.all(
                _threshold_converged(
                    inactive_uA[unresolved],
                    active_uA[unresolved],
                    tolerance_uA=tolerance_uA,
                    relative_tolerance=relative_tolerance,
                )
            ):
                break

            midpoint_uA = 0.5 * (inactive_uA + active_uA)
            events = _evaluate_threshold_updated_pool(
                base_pool,
                update,
                midpoint_uA,
                duration=duration,
                dt=dt,
                criterion=criterion,
                recording=recording,
                batch_options=batch_options,
                progress=solver_progress_gate.consume(),
            )
            _validate_pool_width(events, row_count)
            tested.append(midpoint_uA.copy())
            satisfied_history.append(events.copy())

            active_uA[unresolved & events] = midpoint_uA[unresolved & events]
            inactive_uA[unresolved & ~events] = midpoint_uA[unresolved & ~events]
            unresolved = (status == "threshold") & ~_threshold_converged(
                inactive_uA,
                active_uA,
                tolerance_uA=tolerance_uA,
                relative_tolerance=relative_tolerance,
            )
            progress_display.update(
                iteration=str(iteration),
                rows=row_tuple,
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
        row_labels=row_tuple,
        threshold_uA=threshold_uA,
        lower_bound_uA=inactive_uA,
        upper_bound_uA=active_uA,
        status=tuple(str(value) for value in status.tolist()),
        tested_uA=tuple(tested),
        satisfied=tuple(satisfied_history),
    )


def _evaluate_activation(
    simulation_factory: SimulationFactory,
    tested_current_uA: float,
    *,
    duration: Any,
    dt: Any,
    criterion: ActivationCriterion,
    recording: Recording | None,
) -> ActivationEvent:
    tested_current = units.Q_(tested_current_uA, "microampere")
    candidate = simulation_factory(tested_current)
    result = AxonSimulation(
        axons=candidate,
        duration=duration,
        dt=dt,
        recording=recording or Recording.voltage(),
    ).run()
    return criterion.evaluate(result.single)


def _evaluate_threshold_updated_pool(
    pool: tuple[SimulationCandidate, ...],
    update: PoolUpdate,
    tested_current_uA: np.ndarray,
    *,
    duration: Any,
    dt: Any,
    criterion: ThresholdCriterion,
    recording: Recording | None,
    batch_options: BatchOptions | None,
    progress: bool | str,
) -> np.ndarray:
    values_uA = np.asarray(tested_current_uA, dtype=float)
    if values_uA.shape != (len(pool),):
        raise ValueError(
            f"tested_current_uA must contain one value per row; got {values_uA.shape}."
        )
    updated_pool = tuple(
        _apply_threshold_update(row, update, float(current_uA))
        for row, current_uA in zip(pool, values_uA, strict=True)
    )
    activation_criterion = _threshold_activation_criterion(criterion)
    if _can_use_threshold_observer(recording, activation_criterion):
        activation = _evaluate_activation_observer_pool(
            updated_pool,
            criterion=activation_criterion,
            duration=duration,
            dt=dt,
            batch_options=batch_options,
            progress=progress,
        )
        return np.asarray(activation, dtype=bool)
    pool_result = AxonSimulation(
        axons=updated_pool,
        duration=duration,
        dt=dt,
        recording=recording or Recording.voltage(),
        batch_options=batch_options,
        progress=progress,
    ).run()
    return np.asarray(
        [activation_criterion.evaluate(result).activated for result in pool_result],
        dtype=bool,
    )


def _threshold_activation_criterion(
    criterion: ThresholdCriterion,
) -> ActivationCriterion:
    """Return the activation detector used by threshold search."""

    if isinstance(criterion, ActivationCriterion):
        return criterion
    if isinstance(criterion, Activation):
        return ActivationCriterion(
            threshold=criterion.threshold,
            blanking=criterion.blanking,
            target=criterion.target,
        )
    raise TypeError(
        "criterion must be ActivationCriterion or axs.analysis.Activation."
    )


def _apply_threshold_update(
    row: SimulationCandidate,
    update: PoolUpdate,
    current_uA: float,
) -> SimulationCandidate:
    current = units.Q_(current_uA, "microampere")
    updated = update(row, current)
    return row if updated is None else updated


def _validate_pool_width(values: np.ndarray, expected: int) -> None:
    if values.shape != (expected,):
        raise ValueError(
            "pool/update must return the same number of fibers for every "
            f"threshold evaluation; expected {expected}, got {values.shape[0]}."
        )


__all__ = [
    "find_activation_threshold",
    "find_threshold",
]
