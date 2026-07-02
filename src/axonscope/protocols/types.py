"""Shared protocol type aliases."""

from __future__ import annotations

from typing import Any, Callable, Literal, TypeAlias

import numpy as np

from axonscope.analysis import ActivationCriterion
from axonscope.analysis.definitions import Activation, ConductionBlock
from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon


SimulationCandidate: TypeAlias = Axon | AxonInstance
SimulationFactory: TypeAlias = Callable[[Any], SimulationCandidate]
"""Callable that builds one simulation candidate for a tested value."""

PoolUpdate: TypeAlias = Callable[[SimulationCandidate, Any], SimulationCandidate | None]
"""Callable that updates or replaces one pool row for a tested value."""

PoolObserver: TypeAlias = Callable[[Any], Any]
"""Callable that extracts one observed value from one per-axon result view."""

ProgressSummary: TypeAlias = Callable[[np.ndarray], str]
"""Callable that formats one completed sweep-observation row for progress."""

ThresholdUpdate: TypeAlias = PoolUpdate
"""Callable that updates or replaces one threshold-search row."""

ThresholdStatus: TypeAlias = Literal["threshold", "below_range", "above_range"]
"""Threshold-search outcome, distinct from per-row ``AnalysisStatus`` values."""

ThresholdCriterion: TypeAlias = ActivationCriterion | Activation | ConductionBlock
"""Criterion accepted by generic threshold search."""


__all__ = [
    "PoolObserver",
    "PoolUpdate",
    "ProgressSummary",
    "SimulationCandidate",
    "SimulationFactory",
    "ThresholdCriterion",
    "ThresholdStatus",
    "ThresholdUpdate",
]
