"""Public simulation result containers and visualization helpers."""

from axonscope.results.axes import RecordedAxis
from axonscope.results.single import (
    ObservationDict,
    RecordingDict,
    RecordingValue,
    ResultArray,
)
from axonscope.results.pool import (
    CohortResult,
    AxonResultView,
    AxonSimulationResult,
    RecordedSignal,
    RecordingManifest,
)
from axonscope.results.vm_raster import (
    VM_RASTER_OBSERVATION_KEY,
    VmRasterResult,
    unpack_vm_raster_words,
)
import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "visualization":
        return importlib.import_module("axonscope.results.visualization")
    if name in {"plot_raster", "rasterplot"}:
        module = importlib.import_module("axonscope.results.visualization")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "VM_RASTER_OBSERVATION_KEY",
    "ObservationDict",
    "RecordingDict",
    "RecordingValue",
    "RecordedAxis",
    "ResultArray",
    "VmRasterResult",
    "CohortResult",
    "AxonResultView",
    "AxonSimulationResult",
    "RecordedSignal",
    "RecordingManifest",
    "unpack_vm_raster_words",
    "visualization",
    "plot_raster",
    "rasterplot",
]
