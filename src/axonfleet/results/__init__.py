"""Public simulation result containers and view helpers."""

from axonfleet.results.axes import RecordedAxis
from axonfleet.results.pool import (
    AxonResultView,
    AxonSimulationResult,
    RecordingManifest,
)
from axonfleet.results.vm_raster import (
    VM_RASTER_OBSERVATION_KEY,
    VmRasterResult,
)
import importlib
from typing import Any


_VIEW_HELPERS = {
    "plot_recorded_axes",
}


def __getattr__(name: str) -> Any:
    if name == "views":
        return importlib.import_module("axonfleet.results.views")
    if name in _VIEW_HELPERS:
        module = importlib.import_module("axonfleet.results.views")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "VM_RASTER_OBSERVATION_KEY",
    "RecordedAxis",
    "VmRasterResult",
    "AxonResultView",
    "AxonSimulationResult",
    "RecordingManifest",
    "plot_recorded_axes",
    "views",
]
