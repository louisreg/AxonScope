"""Public scientific analysis definitions and result containers."""

import importlib
from typing import Any

from axonfleet.analysis.core import (
    AnalysisPopulation,
    AnalysisReport,
    AnalysisRequirements,
    AnalysisResult,
    AnalysisStatus,
    analyze,
)
from axonfleet.analysis.activation import ActivationEvent
from axonfleet.analysis.definitions import (
    Activation,
    ConductionVelocity,
    Latency,
    PeakVoltage,
    SpikeCount,
    SpikeCountEvent,
    VmRaster,
)
from axonfleet.analysis.posthoc import rasterize


def __getattr__(name: str) -> Any:
    if name == "views":
        return importlib.import_module("axonfleet.analysis.views")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Activation",
    "ActivationEvent",
    "AnalysisPopulation",
    "AnalysisReport",
    "AnalysisRequirements",
    "AnalysisResult",
    "AnalysisStatus",
    "ConductionVelocity",
    "Latency",
    "PeakVoltage",
    "SpikeCount",
    "SpikeCountEvent",
    "VmRaster",
    "analyze",
    "rasterize",
    "views",
]
