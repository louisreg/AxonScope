"""Public simulation result containers and view helpers."""

from axonscope.results.axes import RecordedAxis
from axonscope.results.types import (
    ObservationDict,
    RecordingDict,
    RecordingValue,
    ResultArray,
)
from axonscope.results.pool import (
    AxonResultView,
    AxonSimulationResult,
    RecordedSignal,
    RecordingManifest,
)
from axonscope.results.vm_raster import (
    VM_RASTER_OBSERVATION_KEY,
    VmRasterResult,
    activation_values_from_vm_raster,
    unpack_vm_raster_words,
    vm_raster_any_active,
    vm_raster_definition_index,
)
import importlib
from typing import Any


_VIEW_HELPERS = {
    "plot_recorded_axes",
}


def __getattr__(name: str) -> Any:
    if name == "views":
        return importlib.import_module("axonscope.results.views")
    if name in _VIEW_HELPERS:
        module = importlib.import_module("axonscope.results.views")
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
    "AxonResultView",
    "AxonSimulationResult",
    "RecordedSignal",
    "RecordingManifest",
    "activation_values_from_vm_raster",
    "unpack_vm_raster_words",
    "vm_raster_any_active",
    "vm_raster_definition_index",
    "plot_recorded_axes",
    "views",
]
