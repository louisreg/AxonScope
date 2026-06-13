"""Activation threshold and recruitment protocols.

Protocols choose tested stimulation values and evaluate activation outcomes.
They deliberately do not mutate axons, electrodes, stimuli, or contexts. The
user-provided factory receives each tested value and must return the simulation
or pool that should be run for that value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence, TypeAlias

import numpy as np

from axonscope.axon_simulation import AxonSimulation
from axonscope.axons.axon import Axon
from axonscope.recording import Recording
from axonscope.results import ActivationCriterion, ActivationEvent, SimResult
from axonscope.simulation import simulate, simulate_pool
from axonscope.utils import units


SimulationCandidate: TypeAlias = Axon | AxonSimulation | SimResult
SimulationFactory: TypeAlias = Callable[[Any], SimulationCandidate]
"""Callable that builds one simulation candidate for a tested value."""

PoolFactory: TypeAlias = Callable[
    [Any],
    Sequence[Axon | AxonSimulation] | Sequence[SimResult],
]
"""Callable that builds one simulation pool for a tested value."""

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

    low_uA = units.to_uA(bounds[0])
    high_uA = units.to_uA(bounds[1])
    tolerance_uA = units.to_uA(tolerance)
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


def recruitment_sweep(
    pool_factory: PoolFactory,
    *,
    amplitudes: Any,
    duration: Any,
    dt: Any,
    criterion: ActivationCriterion,
    recording: Recording | None = None,
) -> RecruitmentCurve:
    """Evaluate pool recruitment over sampled amplitudes.

    ``pool_factory`` is called once per sampled amplitude. It owns every
    between-simulation change and should return the pool to evaluate for that
    exact amplitude. The protocol never edits a stimulus or electrode in place.
    """

    amplitudes_uA = units.to_uA_array(amplitudes)
    if amplitudes_uA.ndim != 1:
        raise ValueError("amplitudes must be a 1D array.")
    rows: list[np.ndarray] = []
    for amplitude_uA in amplitudes_uA:
        tested_current = units.Q_(float(amplitude_uA), "microampere")
        pool = tuple(pool_factory(tested_current))
        if all(isinstance(item, SimResult) for item in pool):
            results = tuple(pool)  # type: ignore[assignment]
        else:
            results = tuple(
                simulate_pool(
                    pool,  # type: ignore[arg-type]
                    duration_ms=duration,
                    dt_ms=dt,
                    recording=recording or Recording.voltage(),
                )
            )
        rows.append(
            np.asarray(
                [criterion.evaluate(result).activated for result in results],
                dtype=bool,
            )
        )
    if not rows:
        return RecruitmentCurve(
            amplitudes_uA=np.asarray([], dtype=float),
            activated=np.zeros((0, 0), dtype=bool),
        )
    width = rows[0].shape[0]
    if any(row.shape[0] != width for row in rows):
        raise ValueError("pool_factory must return the same number of fibers each time.")
    return RecruitmentCurve(
        amplitudes_uA=np.asarray(amplitudes_uA, dtype=float),
        activated=np.stack(rows, axis=0),
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
            duration_ms=duration,
            dt_ms=dt,
            recording=recording or Recording.voltage(),
        )
    return criterion.evaluate(result)


__all__ = [
    "RecruitmentCurve",
    "ThresholdHistoryEntry",
    "ThresholdSearchResult",
    "find_activation_threshold",
    "recruitment_sweep",
]
