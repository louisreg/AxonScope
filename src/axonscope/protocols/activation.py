"""Activation threshold and recruitment protocols.

Protocols choose tested stimulation values and evaluate activation outcomes.
Single-fiber searches use a user-provided factory. Pool searches keep a stable
simulation pool and call a user-provided update function before each run, so
callers can change only the parameter being searched, such as an electrode
stimulus amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence, TypeAlias

import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.recording import Recording
from axonscope.analysis import ActivationCriterion, ActivationEvent
from axonscope.analysis.definitions import Activation
from axonscope.results import SimResult
from axonscope.simulation import simulate, simulate_pool
from axonscope.utils import units


SimulationCandidate: TypeAlias = Axon | AxonInstance | SimResult
SimulationFactory: TypeAlias = Callable[[Any], SimulationCandidate]
"""Callable that builds one simulation candidate for a tested value."""

PoolUpdate: TypeAlias = Callable[[SimulationCandidate, Any], SimulationCandidate | None]
"""Callable that updates or replaces one pool row for a tested value."""

PoolObserver: TypeAlias = Callable[[SimResult], Any]
"""Callable that extracts one observed value from one simulation result."""

ProgressSummary: TypeAlias = Callable[[np.ndarray], str]
"""Callable that formats one completed sweep-observation row for progress."""

ThresholdUpdate: TypeAlias = PoolUpdate
"""Callable that updates or replaces one threshold-search row."""

ThresholdStatus: TypeAlias = Literal["threshold", "below_range", "above_range"]


@dataclass(frozen=True)
class ThresholdHistoryEntry:
    """One activation test performed during threshold search."""

    amplitude_uA: float
    activated: bool
    event: ActivationEvent

    @property
    def amplitude(self) -> Any:
        """Amplitude as a Pint quantity in microamperes."""

        return units.Q_(self.amplitude_uA, "microampere")


@dataclass(frozen=True)
class ThresholdSearchResult:
    """Result returned by ``find_activation_threshold``."""

    amplitude_uA: float | None
    lower_bound_uA: float
    upper_bound_uA: float
    status: ThresholdStatus
    history: tuple[ThresholdHistoryEntry, ...]

    @property
    def amplitude(self) -> Any | None:
        """Estimated threshold amplitude, or ``None`` if outside bounds."""

        if self.amplitude_uA is None:
            return None
        return units.Q_(self.amplitude_uA, "microampere")

    @property
    def lower_bound(self) -> Any:
        """Final inactive lower bound as a Pint quantity."""

        return units.Q_(self.lower_bound_uA, "microampere")

    @property
    def upper_bound(self) -> Any:
        """Final active upper bound as a Pint quantity."""

        return units.Q_(self.upper_bound_uA, "microampere")

    @property
    def n_iterations(self) -> int:
        """Number of activation tests performed."""

        return len(self.history)

    def plot(
        self,
        ax: Any | None = None,
        *,
        unit: Any = "microampere",
        **plot_kwargs: Any,
    ) -> Any:
        """Plot activation decisions over tested amplitudes."""

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        unit_label = units.unit_label(unit) or "microampere"
        unit_text = units.short_unit_label(unit_label) or unit_label
        amplitudes = units.to_array(
            units.Q_([entry.amplitude_uA for entry in self.history], "microampere"),
            unit_label,
            dtype=float,
        )
        activated = np.asarray([entry.activated for entry in self.history], dtype=float)
        plot_kwargs.setdefault("marker", "o")
        plot_kwargs.setdefault("linestyle", "none")
        ax.plot(amplitudes, activated, **plot_kwargs)
        ax.set_xlabel(f"Amplitude [{unit_text}]")
        ax.set_ylabel("Activated")
        ax.set_yticks([0, 1], ["no", "yes"])
        ax.grid(True, alpha=0.3)
        return ax


@dataclass(frozen=True)
class RecruitmentCurve:
    """Recruitment fraction versus stimulus amplitude."""

    amplitudes_uA: np.ndarray
    activated: np.ndarray

    @property
    def amplitudes(self) -> Any:
        """Sweep amplitudes as a Pint quantity array."""

        return units.Q_(self.amplitudes_uA, "microampere")

    @property
    def count(self) -> np.ndarray:
        """Number of activated fibers at each amplitude."""

        return np.sum(self.activated, axis=1)

    @property
    def fraction(self) -> np.ndarray:
        """Activated fraction at each amplitude."""

        if self.activated.shape[1] == 0:
            return np.zeros(self.activated.shape[0], dtype=float)
        return self.count / float(self.activated.shape[1])

    @property
    def threshold_like_uA(self) -> np.ndarray:
        """First sampled activating amplitude for each fiber, or NaN."""

        values = np.full(self.activated.shape[1], np.nan, dtype=float)
        for fiber_index in range(self.activated.shape[1]):
            active_rows = np.flatnonzero(self.activated[:, fiber_index])
            if active_rows.size:
                values[fiber_index] = float(self.amplitudes_uA[active_rows[0]])
        return values

    def plot(
        self,
        ax: Any | None = None,
        *,
        unit: Any = "microampere",
        **plot_kwargs: Any,
    ) -> Any:
        """Plot recruitment fraction versus amplitude."""

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        unit_label = units.unit_label(unit) or "microampere"
        unit_text = units.short_unit_label(unit_label) or unit_label
        amplitudes = units.to_array(
            units.Q_(self.amplitudes_uA, "microampere"),
            unit_label,
            dtype=float,
        )
        plot_kwargs.setdefault("marker", "o")
        plot_kwargs.setdefault("linewidth", 2.0)
        ax.plot(amplitudes, self.fraction, **plot_kwargs)
        ax.set_xlabel(f"Amplitude [{unit_text}]")
        ax.set_ylabel("Recruitment fraction")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        return ax


@dataclass(frozen=True)
class PoolSweepResult:
    """Generic per-row observations over a swept parameter.

    This result is intentionally agnostic to what was observed: activation,
    peak voltage, latency, charge, energy, or any other scalar/object returned
    by the user-provided observer.
    """

    values: tuple[Any, ...]
    observations: np.ndarray

    @property
    def n_values(self) -> int:
        """Number of sampled parameter values."""

        return len(self.values)

    @property
    def n_rows(self) -> int:
        """Number of simulated pool rows."""

        if self.observations.ndim < 2:
            return 0
        return int(self.observations.shape[1])

    def value_values(self, *, unit: Any | None = None) -> np.ndarray:
        """Return swept values as floats when they are unit-bearing scalars."""

        if unit is None:
            return np.arange(len(self.values), dtype=float)
        unit_label = units.unit_label(unit) or str(unit)
        return units.to_array(list(self.values), unit_label, dtype=float)


@dataclass(frozen=True)
class ThresholdCurve:
    """Per-row activation thresholds estimated by batched binary search."""

    rows: tuple[Any, ...]
    threshold_uA: np.ndarray
    lower_bound_uA: np.ndarray
    upper_bound_uA: np.ndarray
    status: tuple[ThresholdStatus, ...]
    tested_uA: tuple[np.ndarray, ...]
    activated: tuple[np.ndarray, ...]

    @property
    def threshold(self) -> Any:
        """Threshold amplitudes as a Pint quantity array."""

        return units.Q_(self.threshold_uA, "microampere")

    @property
    def lower_bound(self) -> Any:
        """Final inactive lower bounds as a Pint quantity array."""

        return units.Q_(self.lower_bound_uA, "microampere")

    @property
    def upper_bound(self) -> Any:
        """Final active upper bounds as a Pint quantity array."""

        return units.Q_(self.upper_bound_uA, "microampere")

    @property
    def n_iterations(self) -> int:
        """Number of batched activation evaluations."""

        return len(self.tested_uA)

    def row_values(self, *, unit: Any | None = None) -> np.ndarray:
        """Return row values as floats when rows are unit-bearing scalars."""

        if unit is None:
            return np.arange(len(self.rows), dtype=float)
        unit_label = units.unit_label(unit) or str(unit)
        return units.to_array(list(self.rows), unit_label, dtype=float)

    def plot(
        self,
        ax: Any | None = None,
        *,
        row_unit: Any | None = None,
        threshold_unit: Any = "microampere",
        **plot_kwargs: Any,
    ) -> Any:
        """Plot threshold versus row value."""

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        x = self.row_values(unit=row_unit)
        y_unit = units.unit_label(threshold_unit) or "microampere"
        y = units.to_array(self.threshold, y_unit, dtype=float)
        row_text = (
            "row"
            if row_unit is None
            else units.short_unit_label(row_unit) or units.unit_label(row_unit) or str(row_unit)
        )
        threshold_text = units.short_unit_label(y_unit) or y_unit
        plot_kwargs.setdefault("marker", "o")
        ax.plot(x, y, **plot_kwargs)
        ax.set_xlabel("row" if row_unit is None else f"row [{row_text}]")
        ax.set_ylabel(f"threshold [{threshold_text}]")
        ax.grid(True, alpha=0.3)
        return ax

    def to_dataframe(
        self,
        *,
        row_name: str = "row",
        row_unit: Any | None = None,
        threshold_unit: Any = "microampere",
    ) -> Any:
        """Return a pandas DataFrame summary when pandas is installed."""

        import pandas as pd

        y_unit = units.unit_label(threshold_unit) or "microampere"
        data = {
            row_name: self.row_values(unit=row_unit)
            if row_unit is not None
            else list(self.rows),
            "threshold": units.to_array(self.threshold, y_unit, dtype=float),
            "lower_bound": units.to_array(self.lower_bound, y_unit, dtype=float),
            "upper_bound": units.to_array(self.upper_bound, y_unit, dtype=float),
            "status": list(self.status),
        }
        return pd.DataFrame(data)


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
    return a fresh simulation candidate or precomputed ``SimResult``.
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


def find_activation_threshold_curve(
    pool: Sequence[SimulationCandidate],
    *,
    update: ThresholdUpdate,
    bounds: tuple[Any, Any] | Callable[[Any], tuple[Any, Any]],
    duration: Any,
    dt: Any,
    criterion: ActivationCriterion,
    rows: Sequence[Any] | None = None,
    tolerance: Any | None = 1.0,
    relative_tolerance: float | None = None,
    max_iterations: int = 20,
    recording: Recording | None = None,
    progress: bool | str = False,
    solver_progress: bool | str = False,
) -> ThresholdCurve:
    """Estimate one activation threshold per pool row with batched bisection.

    ``pool`` contains the simulations to evaluate. ``update(simulation,
    current)`` is called before each run to change the searched parameter,
    usually a stimulus amplitude. The update function may mutate the row and
    return ``None``, or return a replacement simulation. ``rows`` optionally
    carries user-facing values such as diameters for plotting and callable
    bounds.
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
            rows=row_tuple,
            threshold_uA=np.asarray([], dtype=float),
            lower_bound_uA=np.asarray([], dtype=float),
            upper_bound_uA=np.asarray([], dtype=float),
            status=(),
            tested_uA=(),
            activated=(),
        )

    progress_display = _ThresholdProgress(progress)
    try:
        low_events = _evaluate_activation_updated_pool(
            base_pool,
            update,
            low_vector,
            duration=duration,
            dt=dt,
            criterion=criterion,
            recording=recording,
            progress=solver_progress,
        )
        tested: list[np.ndarray] = [low_vector.copy()]
        activated_history: list[np.ndarray] = [low_events.copy()]
        status = np.full(row_count, "threshold", dtype=object)

        high_events = _evaluate_activation_updated_pool(
            base_pool,
            update,
            high_vector,
            duration=duration,
            dt=dt,
            criterion=criterion,
            recording=recording,
            progress=solver_progress,
        )
        _validate_pool_width(high_events, row_count)
        tested.append(high_vector.copy())
        activated_history.append(high_events.copy())

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
            activated=high_events,
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
            events = _evaluate_activation_updated_pool(
                base_pool,
                update,
                midpoint_uA,
                duration=duration,
                dt=dt,
                criterion=criterion,
                recording=recording,
                progress=solver_progress,
            )
            _validate_pool_width(events, row_count)
            tested.append(midpoint_uA.copy())
            activated_history.append(events.copy())

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
                activated=events,
                lower_bound_uA=inactive_uA,
                upper_bound_uA=active_uA,
                status=status,
            )
    finally:
        progress_display.close()

    threshold_mask = status == "threshold"
    threshold_uA[threshold_mask] = active_uA[threshold_mask]
    return ThresholdCurve(
        rows=row_tuple,
        threshold_uA=threshold_uA,
        lower_bound_uA=inactive_uA,
        upper_bound_uA=active_uA,
        status=tuple(str(value) for value in status.tolist()),
        tested_uA=tuple(tested),
        activated=tuple(activated_history),
    )


def recruitment_sweep(
    pool: Sequence[SimulationCandidate],
    *,
    update: PoolUpdate,
    amplitudes: Any,
    duration: Any,
    dt: Any,
    criterion: ActivationCriterion,
    recording: Recording | None = None,
    progress: bool | str = False,
    solver_progress: bool | str = False,
) -> RecruitmentCurve:
    """Evaluate pool recruitment over sampled amplitudes.

    ``pool`` contains the simulations to evaluate. ``update(simulation,
    amplitude)`` is called before each run to change the swept parameter,
    usually an electrode current. The update function may mutate the row and
    return ``None``, or return a replacement simulation.
    """

    amplitudes_uA = _require_current_array_uA(amplitudes, name="amplitudes")
    if amplitudes_uA.ndim != 1:
        raise ValueError("amplitudes must be a 1D array.")
    if _can_use_activation_observer(recording, criterion):
        sweep = _activation_pool_sweep(
            pool,
            update=update,
            values=units.Q_(amplitudes_uA, "microampere"),
            duration=duration,
            dt=dt,
            criterion=criterion,
            progress=progress,
            solver_progress=solver_progress,
        )
        return RecruitmentCurve(
            amplitudes_uA=np.asarray(amplitudes_uA, dtype=float),
            activated=np.asarray(sweep.observations, dtype=bool),
        )
    sweep = pool_sweep(
        pool,
        update=update,
        values=units.Q_(amplitudes_uA, "microampere"),
        observe=lambda result: criterion.evaluate(result).activated,
        duration=duration,
        dt=dt,
        recording=recording,
        progress=progress,
        progress_summary=_activation_progress_summary,
        solver_progress=solver_progress,
    )
    return RecruitmentCurve(
        amplitudes_uA=np.asarray(amplitudes_uA, dtype=float),
        activated=np.asarray(sweep.observations, dtype=bool),
    )


def _activation_pool_sweep(
    pool: Sequence[SimulationCandidate],
    *,
    update: PoolUpdate,
    values: Sequence[Any],
    duration: Any,
    dt: Any,
    criterion: ActivationCriterion,
    progress: bool | str = False,
    solver_progress: bool | str = False,
) -> PoolSweepResult:
    """Sweep activation with solver-side observers instead of stored Vm traces."""

    base_pool = tuple(pool)
    value_tuple = _normalize_sweep_values(values)
    if len(base_pool) == 0:
        return PoolSweepResult(
            values=value_tuple,
            observations=np.zeros((len(value_tuple), 0), dtype=bool),
        )

    progress_display = _SweepProgress(progress)
    observation_rows: list[np.ndarray] = []
    try:
        for index, value in enumerate(value_tuple):
            updated_pool = tuple(
                _apply_pool_update(row, update, value) for row in base_pool
            )
            if all(isinstance(item, SimResult) for item in updated_pool):
                observations = np.asarray(
                    [
                        criterion.evaluate(result).activated
                        for result in updated_pool  # type: ignore[arg-type]
                    ],
                    dtype=bool,
                )
            else:
                observations = _evaluate_activation_observer_pool(
                    updated_pool,  # type: ignore[arg-type]
                    criterion=criterion,
                    duration=duration,
                    dt=dt,
                    progress=solver_progress,
                )
            observation_rows.append(observations)
            progress_display.update(
                label="Pool sweep",
                current_index=index,
                values=value_tuple,
                completed_rows=observation_rows,
                progress_summary=_activation_progress_summary,
            )
    finally:
        progress_display.close()

    if not observation_rows:
        return PoolSweepResult(
            values=value_tuple,
            observations=np.zeros((0, len(base_pool)), dtype=bool),
        )
    width = observation_rows[0].shape[0]
    if any(row.shape[0] != width for row in observation_rows):
        raise ValueError("pool/update must keep the same number of rows each time.")
    return PoolSweepResult(
        values=value_tuple,
        observations=np.stack(observation_rows, axis=0),
    )


def pool_sweep(
    pool: Sequence[SimulationCandidate],
    *,
    update: PoolUpdate,
    values: Sequence[Any],
    observe: PoolObserver,
    duration: Any,
    dt: Any,
    recording: Recording | None = None,
    progress: bool | str = False,
    progress_summary: ProgressSummary | None = None,
    solver_progress: bool | str = False,
) -> PoolSweepResult:
    """Sweep a parameter over a stable simulation pool.

    Parameters
    ----------
    pool:
        Stable sequence of simulations, axons, or precomputed results.
    update:
        Called as ``update(row, value)`` before each run. It may mutate the row
        in place and return ``None``, or return a replacement candidate.
    values:
        Parameter values to test. Unit-bearing arrays are accepted and each row
        receives one scalar quantity from the array.
    observe:
        Called on each ``SimResult`` to produce one per-row observation.
    duration, dt:
        Simulation duration and timestep.
    recording:
        Recording policy used when pool entries must be simulated.
    progress:
        If true, display a Rich live progress table when Rich is available.
    progress_summary:
        Optional formatter for one completed observation row.
    solver_progress:
        Optional progress flag forwarded to ``simulate_pool``.
    """

    base_pool = tuple(pool)
    value_tuple = _normalize_sweep_values(values)
    if len(base_pool) == 0:
        return PoolSweepResult(
            values=value_tuple,
            observations=np.zeros((len(value_tuple), 0), dtype=object),
        )

    progress_display = _SweepProgress(progress)
    observation_rows: list[np.ndarray] = []
    try:
        for index, value in enumerate(value_tuple):
            results = _run_updated_pool(
                base_pool,
                update,
                tuple(value for _ in base_pool),
                duration=duration,
                dt=dt,
                recording=recording,
                progress=solver_progress,
            )
            observations = np.asarray([observe(result) for result in results])
            observation_rows.append(observations)
            progress_display.update(
                label="Pool sweep",
                current_index=index,
                values=value_tuple,
                completed_rows=observation_rows,
                progress_summary=progress_summary,
            )
    finally:
        progress_display.close()

    if not observation_rows:
        return PoolSweepResult(
            values=value_tuple,
            observations=np.zeros((0, len(base_pool)), dtype=object),
        )
    width = observation_rows[0].shape[0]
    if any(row.shape[0] != width for row in observation_rows):
        raise ValueError("pool/update must keep the same number of rows each time.")
    return PoolSweepResult(
        values=value_tuple,
        observations=np.stack(observation_rows, axis=0),
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
    if isinstance(candidate, SimResult):
        result = candidate
    else:
        result = simulate(
            candidate,
            duration=duration,
            dt=dt,
            recording=recording or Recording.voltage(),
        )
    return criterion.evaluate(result)


def _normalize_rows(rows: Sequence[Any]) -> tuple[Any, ...]:
    if units.is_quantity_like(rows):
        magnitudes = np.asarray(rows.magnitude)
        if magnitudes.ndim != 1:
            raise ValueError("rows must be a 1D sequence.")
        row_unit = getattr(rows, "units")
        return tuple(units.Q_(float(value), row_unit) for value in magnitudes)
    normalized = tuple(rows)
    if not normalized:
        return ()
    return normalized


def _normalize_sweep_values(values: Sequence[Any]) -> tuple[Any, ...]:
    if units.is_quantity_like(values):
        magnitudes = np.asarray(values.magnitude)
        if magnitudes.ndim != 1:
            raise ValueError("values must be a 1D sequence.")
        value_unit = getattr(values, "units")
        return tuple(units.Q_(float(value), value_unit) for value in magnitudes)
    normalized = tuple(values)
    if not normalized:
        return ()
    return normalized


def _resolve_threshold_bounds(
    bounds: tuple[Any, Any] | Callable[[Any], tuple[Any, Any]],
    rows: tuple[Any, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if callable(bounds):
        lower: list[float] = []
        upper: list[float] = []
        for row in rows:
            row_low, row_high = bounds(row)
            lower.append(units.require_current_uA(row_low, name="bounds(row)[0]"))
            upper.append(units.require_current_uA(row_high, name="bounds(row)[1]"))
        low_uA = np.asarray(lower, dtype=float)
        high_uA = np.asarray(upper, dtype=float)
    else:
        low_uA = _broadcast_bound(bounds[0], len(rows), name="bounds[0]")
        high_uA = _broadcast_bound(bounds[1], len(rows), name="bounds[1]")
    if np.any(high_uA <= low_uA):
        raise ValueError("bounds must be ordered as (low, high) for every row.")
    return low_uA, high_uA


def _broadcast_bound(value: Any, row_count: int, *, name: str) -> np.ndarray:
    values = _require_current_array_uA(value, name=name)
    if values.ndim == 0:
        return np.full(row_count, float(values.item()), dtype=float)
    if values.shape != (row_count,):
        raise ValueError(f"{name} must be scalar or have shape ({row_count},).")
    return np.asarray(values, dtype=float)


def _require_current_array_uA(value: Any, *, name: str) -> np.ndarray:
    """Normalize a unit-bearing scalar/array/list of current values."""

    if units.is_quantity_like(value):
        return units.require_current_array_uA(value, name=name, dtype=float)
    if isinstance(value, (list, tuple)) and all(units.is_quantity_like(item) for item in value):
        return np.asarray(
            [units.require_current_uA(item, name=name) for item in value],
            dtype=float,
        )
    return units.require_current_array_uA(value, name=name, dtype=float)


def _threshold_converged(
    low_uA: np.ndarray,
    high_uA: np.ndarray,
    *,
    tolerance_uA: float | None,
    relative_tolerance: float | None,
) -> np.ndarray:
    width = np.asarray(high_uA, dtype=float) - np.asarray(low_uA, dtype=float)
    converged = np.zeros(width.shape, dtype=bool)
    if tolerance_uA is not None:
        converged |= width <= tolerance_uA
    if relative_tolerance is not None:
        scale = np.maximum(np.abs(high_uA), np.finfo(float).eps)
        converged |= width <= float(relative_tolerance) * scale
    return converged


def _evaluate_activation_updated_pool(
    pool: tuple[SimulationCandidate, ...],
    update: PoolUpdate,
    tested_current_uA: np.ndarray,
    *,
    duration: Any,
    dt: Any,
    criterion: ActivationCriterion,
    recording: Recording | None,
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
    if all(isinstance(item, SimResult) for item in updated_pool):
        results = tuple(updated_pool)  # type: ignore[assignment]
    elif _can_use_activation_observer(recording, criterion):
        return _evaluate_activation_observer_pool(
            updated_pool,  # type: ignore[arg-type]
            criterion=criterion,
            duration=duration,
            dt=dt,
            progress=progress,
        )
    else:
        pool_result = simulate_pool(
            updated_pool,  # type: ignore[arg-type]
            duration=duration,
            dt=dt,
            recording=recording or Recording.voltage(),
            progress=progress,
        )
        results = tuple(view.to_sim_result() for view in pool_result)
    return np.asarray(
        [criterion.evaluate(result).activated for result in results],
        dtype=bool,
    )


def _run_updated_pool(
    pool: tuple[SimulationCandidate, ...],
    update: PoolUpdate,
    values: tuple[Any, ...],
    *,
    duration: Any,
    dt: Any,
    recording: Recording | None,
    progress: bool | str,
) -> tuple[SimResult, ...]:
    if len(values) != len(pool):
        raise ValueError(
            f"values must contain one value per row; got {len(values)} for {len(pool)} rows."
        )
    updated_pool = tuple(
        _apply_pool_update(row, update, value)
        for row, value in zip(pool, values, strict=True)
    )
    if all(isinstance(item, SimResult) for item in updated_pool):
        return tuple(updated_pool)  # type: ignore[return-value]
    pool_result = simulate_pool(
        updated_pool,  # type: ignore[arg-type]
        duration=duration,
        dt=dt,
        recording=recording or Recording.voltage(),
        progress=progress,
    )
    return tuple(view.to_sim_result() for view in pool_result)


def _evaluate_activation_observer_pool(
    pool: tuple[SimulationCandidate, ...],
    *,
    criterion: ActivationCriterion,
    duration: Any,
    dt: Any,
    progress: bool | str,
) -> np.ndarray:
    """Evaluate activation through compact solver-side observers."""

    activation = _activation_observer_definition(criterion)
    pool_result = simulate_pool(
        pool,  # type: ignore[arg-type]
        duration=duration,
        dt=dt,
        recording=Recording.none(),
        observers=(activation,),
        progress=progress,
    )
    return np.asarray(
        [
            bool(np.asarray(_activation_observation(view, activation.name).values)[0])
            for view in pool_result
        ],
        dtype=bool,
    )


def _activation_observation(result: Any, name: str) -> Any:
    observations = getattr(result, "observations", None)
    if observations is None or name not in observations:
        raise RuntimeError("activation observer result is missing from solver output.")
    return observations[name]


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


def _activation_observer_definition(criterion: ActivationCriterion) -> Activation:
    return Activation(
        threshold=criterion.threshold,
        blanking=criterion.blanking,
        target=criterion.target,
    )


def _apply_threshold_update(
    row: SimulationCandidate,
    update: PoolUpdate,
    current_uA: float,
) -> SimulationCandidate:
    current = units.Q_(current_uA, "microampere")
    updated = update(row, current)
    return row if updated is None else updated


def _apply_pool_update(
    row: SimulationCandidate,
    update: PoolUpdate,
    value: Any,
) -> SimulationCandidate:
    updated = update(row, value)
    return row if updated is None else updated


def _format_row(row: Any) -> str:
    if units.is_quantity_like(row):
        try:
            return f"{float(row.magnitude):.4g} {row.units:~P}"
        except Exception:
            return str(row)
    return str(row)


def _format_sweep_value(value: Any) -> str:
    if units.is_quantity_like(value):
        try:
            return f"{float(value.magnitude):.4g} {value.units:~P}"
        except Exception:
            return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _activation_progress_summary(row: np.ndarray) -> str:
    activated = np.asarray(row, dtype=bool)
    total = int(activated.shape[0])
    count = int(np.sum(activated))
    fraction = 0.0 if total == 0 else count / float(total)
    return f"{count}/{total} ({fraction:.2f})"


class _ThresholdProgress:
    def __init__(self, progress: bool | str) -> None:
        self.progress = progress
        self.mode = "rich" if progress is True else str(progress)
        self._live: Any | None = None
        self._console: Any | None = None

    def update(
        self,
        *,
        iteration: str,
        rows: tuple[Any, ...],
        tested_uA: np.ndarray,
        activated: np.ndarray,
        lower_bound_uA: np.ndarray,
        upper_bound_uA: np.ndarray,
        status: np.ndarray,
    ) -> None:
        if not self.progress:
            return
        if self.mode != "plain":
            try:
                table = self._rich_table(
                    iteration=iteration,
                    rows=rows,
                    tested_uA=tested_uA,
                    activated=activated,
                    lower_bound_uA=lower_bound_uA,
                    upper_bound_uA=upper_bound_uA,
                    status=status,
                )
                if self._live is None:
                    from rich.console import Console
                    from rich.live import Live

                    self._console = Console()
                    self._live = Live(
                        table,
                        console=self._console,
                        refresh_per_second=8,
                        transient=False,
                    )
                    self._live.start()
                else:
                    self._live.update(table)
                return
            except Exception:
                self.mode = "plain"

        self._plain_update(
            iteration=iteration,
            rows=rows,
            tested_uA=tested_uA,
            activated=activated,
            lower_bound_uA=lower_bound_uA,
            upper_bound_uA=upper_bound_uA,
            status=status,
        )

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            if self._console is not None:
                self._console.print()
            self._live = None

    @staticmethod
    def _rich_table(
        *,
        iteration: str,
        rows: tuple[Any, ...],
        tested_uA: np.ndarray,
        activated: np.ndarray,
        lower_bound_uA: np.ndarray,
        upper_bound_uA: np.ndarray,
        status: np.ndarray,
    ) -> Any:
        from rich.table import Table

        table = Table(title=f"Threshold search iteration {iteration}")
        table.add_column("row")
        table.add_column("low (uA)", justify="right")
        table.add_column("high (uA)", justify="right")
        table.add_column("test (uA)", justify="right")
        table.add_column("active", justify="center")
        table.add_column("status")
        for row, low, high, tested, active, state in zip(
            rows,
            lower_bound_uA,
            upper_bound_uA,
            tested_uA,
            activated,
            status,
            strict=True,
        ):
            table.add_row(
                _format_row(row),
                f"{float(low):.3g}",
                f"{float(high):.3g}",
                f"{float(tested):.3g}",
                "yes" if bool(active) else "no",
                str(state),
            )
        return table

    @staticmethod
    def _plain_update(
        *,
        iteration: str,
        rows: tuple[Any, ...],
        tested_uA: np.ndarray,
        activated: np.ndarray,
        lower_bound_uA: np.ndarray,
        upper_bound_uA: np.ndarray,
        status: np.ndarray,
    ) -> None:
        print("\033[2J\033[H", end="")
        print(f"Threshold search iteration {iteration}")
        for row, low, high, tested, active, state in zip(
            rows,
            lower_bound_uA,
            upper_bound_uA,
            tested_uA,
            activated,
            status,
            strict=True,
        ):
            print(
                f"{_format_row(row):>12s}: "
                f"low={float(low):.3g} uA "
                f"high={float(high):.3g} uA "
                f"test={float(tested):.3g} uA "
                f"active={'yes' if bool(active) else 'no'} "
                f"status={state}"
            )


class _SweepProgress:
    def __init__(self, progress: bool | str) -> None:
        self.progress = progress
        self.mode = "rich" if progress is True else str(progress)
        self._live: Any | None = None
        self._console: Any | None = None

    def update(
        self,
        *,
        label: str,
        current_index: int,
        values: tuple[Any, ...],
        completed_rows: list[np.ndarray],
        progress_summary: ProgressSummary | None,
    ) -> None:
        if not self.progress:
            return
        if self.mode != "plain":
            try:
                table = self._rich_table(
                    label=label,
                    current_index=current_index,
                    values=values,
                    completed_rows=completed_rows,
                    progress_summary=progress_summary,
                )
                if self._live is None:
                    from rich.console import Console
                    from rich.live import Live

                    self._console = Console()
                    self._live = Live(
                        table,
                        console=self._console,
                        refresh_per_second=8,
                        transient=False,
                    )
                    self._live.start()
                else:
                    self._live.update(table)
                return
            except Exception:
                self.mode = "plain"

        self._plain_update(
            label=label,
            current_index=current_index,
            values=values,
            completed_rows=completed_rows,
            progress_summary=progress_summary,
        )

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            if self._console is not None:
                self._console.print()
            self._live = None

    @staticmethod
    def _rich_table(
        *,
        label: str,
        current_index: int,
        values: tuple[Any, ...],
        completed_rows: list[np.ndarray],
        progress_summary: ProgressSummary | None,
    ) -> Any:
        from rich.table import Table

        current = _format_sweep_value(values[current_index])
        table = Table(title=f"{label}, current={current}")
        table.add_column("value", justify="right")
        table.add_column("summary", justify="right")
        table.add_column("status", justify="right")
        completed = len(completed_rows)
        for index, value in enumerate(values):
            if index < completed:
                row = completed_rows[index]
                summary = (
                    progress_summary(row)
                    if progress_summary is not None
                    else f"{int(row.shape[0])} rows"
                )
                table.add_row(
                    _format_sweep_value(value),
                    summary,
                    "done",
                )
            else:
                table.add_row(_format_sweep_value(value), "-", "pending")
        return table

    @staticmethod
    def _plain_update(
        *,
        label: str,
        current_index: int,
        values: tuple[Any, ...],
        completed_rows: list[np.ndarray],
        progress_summary: ProgressSummary | None,
    ) -> None:
        print("\033[2J\033[H", end="")
        print(f"{label}, current={_format_sweep_value(values[current_index])}")
        completed = len(completed_rows)
        for index, value in enumerate(values):
            if index < completed:
                row = completed_rows[index]
                summary = (
                    progress_summary(row)
                    if progress_summary is not None
                    else f"{int(row.shape[0])} rows"
                )
                print(f"{_format_sweep_value(value):>12s}: {summary} done")
            else:
                print(f"{_format_sweep_value(value):>12s}: pending")


def _validate_pool_width(values: np.ndarray, expected: int) -> None:
    if values.shape != (expected,):
        raise ValueError(
            "pool/update must return the same number of fibers for every "
            f"threshold evaluation; expected {expected}, got {values.shape[0]}."
        )


__all__ = [
    "PoolSweepResult",
    "RecruitmentCurve",
    "ThresholdCurve",
    "ThresholdHistoryEntry",
    "ThresholdSearchResult",
    "find_activation_threshold",
    "find_activation_threshold_curve",
    "pool_sweep",
    "recruitment_sweep",
]
