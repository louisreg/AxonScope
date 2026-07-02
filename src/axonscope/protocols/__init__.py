"""High-level simulation protocols."""

from __future__ import annotations

import importlib
from typing import Any

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
from axonscope.protocols.types import ThresholdCriterion, ThresholdStatus


def __getattr__(name: str) -> Any:
    if name == "views":
        return importlib.import_module("axonscope.protocols.views")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PoolSweepResult",
    "RecruitmentCurve",
    "ThresholdCurve",
    "ThresholdCriterion",
    "ThresholdHistoryEntry",
    "ThresholdSearchResult",
    "ThresholdStatus",
    "find_activation_threshold",
    "find_threshold",
    "pool_sweep",
    "recruitment_sweep",
    "views",
]
