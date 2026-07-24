"""High-level simulation protocols."""

from __future__ import annotations

import importlib
from typing import Any

from axonfleet.protocols.recruitment import recruitment_sweep, recruitment_sweep_plan
from axonfleet.protocols.results import (
    PoolSweepResult,
    RecruitmentCurve,
    ThresholdCurve,
)
from axonfleet.protocols.sweep import pool_sweep, pool_sweep_plan
from axonfleet.protocols.threshold import find_threshold, find_threshold_plan
from axonfleet.protocols.types import ThresholdStatus
from axonfleet.protocols.updates import ExtracellularWaveformUpdate


def __getattr__(name: str) -> Any:
    if name == "views":
        return importlib.import_module("axonfleet.protocols.views")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PoolSweepResult",
    "ExtracellularWaveformUpdate",
    "RecruitmentCurve",
    "ThresholdCurve",
    "ThresholdStatus",
    "find_threshold",
    "find_threshold_plan",
    "pool_sweep",
    "pool_sweep_plan",
    "recruitment_sweep",
    "recruitment_sweep_plan",
    "views",
]
