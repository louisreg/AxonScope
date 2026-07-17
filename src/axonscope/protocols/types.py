"""Shared protocol type aliases."""

from __future__ import annotations

from typing import Any, Callable, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np

from axonscope.analysis import ActivationCriterion
from axonscope.analysis.definitions import Activation
from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.dispatcher.numeric_axis import NumericAxisInput


SimulationCandidate: TypeAlias = Axon | AxonInstance
SimulationFactory: TypeAlias = Callable[[Any], SimulationCandidate]
"""Callable that builds one simulation candidate for a tested value."""

PoolUpdate: TypeAlias = Callable[[SimulationCandidate, Any], SimulationCandidate | None]
"""Callable that updates or replaces one pool row for a tested value."""


@runtime_checkable
class NumericAxisUpdate(Protocol):
    """Typed update that preserves one static execution contract."""

    def __call__(
        self,
        row: SimulationCandidate,
        value: Any,
    ) -> SimulationCandidate | None: ...

    def prepare_numeric_axis(
        self,
        pool: tuple[SimulationCandidate, ...],
    ) -> "NumericAxisInputBuilder": ...


class NumericAxisInputBuilder(Protocol):
    """Prepared immutable source contract for dynamic numeric-axis values."""

    def numeric_axis_input(self, values: tuple[Any, ...]) -> NumericAxisInput: ...

PoolObserver: TypeAlias = Callable[[Any], Any]
"""Callable that extracts one observed value from one per-axon result view."""

ProgressSummary: TypeAlias = Callable[[np.ndarray], str]
"""Callable that formats one completed sweep-observation row for progress."""

ThresholdUpdate: TypeAlias = PoolUpdate
"""Callable that updates or replaces one threshold-search row."""

ThresholdStatus: TypeAlias = Literal["threshold", "below_range", "above_range"]
"""Threshold-search outcome, distinct from per-row ``AnalysisStatus`` values."""

ThresholdCriterion: TypeAlias = ActivationCriterion | Activation
"""Criterion accepted by generic threshold search."""


__all__ = [
    "PoolObserver",
    "PoolUpdate",
    "NumericAxisUpdate",
    "NumericAxisInputBuilder",
    "ProgressSummary",
    "SimulationCandidate",
    "SimulationFactory",
    "ThresholdCriterion",
    "ThresholdStatus",
    "ThresholdUpdate",
]
