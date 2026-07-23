"""Shared protocol type aliases."""

from __future__ import annotations

from typing import Any, Callable, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np

from axonfleet.axon_instance import AxonInstance
from axonfleet.axons.axon import Axon
from axonfleet.dispatcher.numeric_axis import NumericAxisInput


SimulationCandidate: TypeAlias = Axon | AxonInstance
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

ThresholdStatus: TypeAlias = Literal["threshold", "below_range", "above_range"]
"""Threshold-search outcome, distinct from per-row ``AnalysisStatus`` values."""

__all__ = [
    "PoolObserver",
    "PoolUpdate",
    "NumericAxisUpdate",
    "NumericAxisInputBuilder",
    "ProgressSummary",
    "SimulationCandidate",
    "ThresholdStatus",
]
