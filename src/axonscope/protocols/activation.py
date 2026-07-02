"""Activation threshold and recruitment protocol facade."""

from __future__ import annotations

from axonscope.protocols.observer_path import _activation_observations_from_pool_result
from axonscope.protocols.recruitment import recruitment_sweep
from axonscope.protocols.results import (
    PoolSweepResult,
    RecruitmentCurve,
    ThresholdCurve,
    ThresholdHistoryEntry,
    ThresholdSearchResult,
)
from axonscope.protocols.sweep import pool_sweep
from axonscope.protocols.threshold import find_activation_threshold, find_threshold
from axonscope.protocols.types import (
    PoolObserver,
    PoolUpdate,
    ProgressSummary,
    SimulationCandidate,
    SimulationFactory,
    ThresholdCriterion,
    ThresholdStatus,
    ThresholdUpdate,
)


__all__ = [
    "PoolObserver",
    "PoolSweepResult",
    "PoolUpdate",
    "ProgressSummary",
    "RecruitmentCurve",
    "SimulationCandidate",
    "SimulationFactory",
    "ThresholdCriterion",
    "ThresholdCurve",
    "ThresholdHistoryEntry",
    "ThresholdSearchResult",
    "ThresholdStatus",
    "ThresholdUpdate",
    "find_activation_threshold",
    "find_threshold",
    "pool_sweep",
    "recruitment_sweep",
]
