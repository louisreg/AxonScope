"""Public scientific analysis definitions and result containers."""

import importlib
from typing import Any

from axonscope.analysis.core import (
    AnalysisDefinition,
    AnalysisInputRequirement,
    AnalysisNotApplicableError,
    AnalysisPopulation,
    AnalysisReport,
    AnalysisRequirements,
    AnalysisResult,
    AnalysisStatus,
    MissingAnalysisInputError,
    analyze,
)
from axonscope.analysis.observers import ActivationObserver
from axonscope.analysis.activation import (
    ActivationCriterion,
    ActivationEvent,
    detect_activation,
)
from axonscope.analysis.definitions import (
    Activation,
    ConductionVelocity,
    Latency,
    PeakVoltage,
    SpikeCount,
    SpikeCountEvent,
    VmRaster,
)
from axonscope.analysis.posthoc import (
    average_velocity,
    conduction_velocity,
    peak_voltage,
    rasterize,
    recorded_positions_um,
)


def __getattr__(name: str) -> Any:
    if name == "views":
        return importlib.import_module("axonscope.analysis.views")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Activation",
    "ActivationCriterion",
    "ActivationEvent",
    "AnalysisDefinition",
    "AnalysisInputRequirement",
    "AnalysisNotApplicableError",
    "AnalysisPopulation",
    "AnalysisReport",
    "AnalysisRequirements",
    "AnalysisResult",
    "AnalysisStatus",
    "ConductionVelocity",
    "Latency",
    "MissingAnalysisInputError",
    "PeakVoltage",
    "ActivationObserver",
    "SpikeCount",
    "SpikeCountEvent",
    "VmRaster",
    "average_velocity",
    "analyze",
    "conduction_velocity",
    "detect_activation",
    "peak_voltage",
    "rasterize",
    "recorded_positions_um",
    "views",
]
