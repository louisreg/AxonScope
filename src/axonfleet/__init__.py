from .utils import units

from . import (
    analysis,
    analytical,
    axons,
    cache,
    dispatcher,
    identifiers,
    inspection,
    membranes,
    performance,
    positions,
    protocols,
    recording,
    results,
    runtime,
    signals,
    solvers,
    stimulation,
)
from axonfleet.axon_instance import AxonInstance
from axonfleet.benchmarking import (
    benchmark,
    benchmark_report,
    disable_benchmark,
    enable_benchmark,
    reset_benchmark,
)
from axonfleet.identifiers import DriveId
from axonfleet.inspection import SimulationInspection
from axonfleet.population import AxonPopulation
from axonfleet.results import (
    VM_RASTER_OBSERVATION_KEY,
    VmRasterResult,
)
from axonfleet.recording import Recording
from axonfleet.runtime import (
    Device,
    ExecutionPolicy,
    PrecisionPolicy,
    SolverPolicy,
)
from axonfleet.simulation import AxonSimulation
from axonfleet.solvers import (
    BatchOptions,
    DEFAULT_OBSERVER_TIME_CHUNK_STEPS,
)
from axonfleet.stimulation import (
    ExtracellularDrive,
    ExtracellularFootprint,
    ExtracellularStimulation,
    IntracellularCurrentClamp,
)
from axonfleet.stimulation import Stimulus

__version__ = "0.1.0"

_UNIT_ALIASES = {
    "A": "ampere",
    "cm": "centimeter",
    "degC": "degree_Celsius",
    "Hz": "hertz",
    "kHz": "kilohertz",
    "m": "meter",
    "mA": "milliampere",
    "mM": "millimolar",
    "mS": "millisiemens",
    "mA_per_cm2": "milliampere / centimeter ** 2",
    "MOhm_per_cm": "megaohm / centimeter",
    "mV": "millivolt",
    "mm": "millimeter",
    "ms": "millisecond",
    "nA": "nanoampere",
    "ohm": "ohm",
    "ohm_cm": "ohm * centimeter",
    "ohm_cm2": "ohm * centimeter ** 2",
    "ohm_um": "ohm * micrometer",
    "s": "second",
    "S": "siemens",
    "S_per_cm2": "siemens / centimeter ** 2",
    "S_per_m": "siemens / meter",
    "S_per_meter": "siemens / meter",
    "siemens_per_m": "siemens / meter",
    "siemens_per_meter": "siemens / meter",
    "mS_per_cm": "millisiemens / centimeter",
    "mS_per_cm2": "millisiemens / centimeter ** 2",
    "mS_per_centimeter": "millisiemens / centimeter",
    "uA": "microampere",
    "uF": "microfarad",
    "uF_per_cm2": "microfarad / centimeter ** 2",
    "um": "micrometer",
    "us": "microsecond",
    "V": "volt",
}


def __getattr__(name: str):
    if name in _UNIT_ALIASES:
        return getattr(units.ureg, _UNIT_ALIASES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "analysis",
    "analytical",
    "axons",
    "cache",
    "dispatcher",
    "identifiers",
    "inspection",
    "membranes",
    "performance",
    "positions",
    "protocols",
    "recording",
    "results",
    "runtime",
    "signals",
    "solvers",
    "stimulation",
    "units",
    "Stimulus",
    "DriveId",
    "SimulationInspection",
    "Device",
    "ExecutionPolicy",
    "PrecisionPolicy",
    "SolverPolicy",
    "AxonInstance",
    "AxonPopulation",
    "AxonSimulation",
    "benchmark",
    "benchmark_report",
    "disable_benchmark",
    "enable_benchmark",
    "reset_benchmark",
    "BatchOptions",
    "DEFAULT_OBSERVER_TIME_CHUNK_STEPS",
    "Recording",
    "IntracellularCurrentClamp",
    "ExtracellularDrive",
    "ExtracellularFootprint",
    "ExtracellularStimulation",
    "VM_RASTER_OBSERVATION_KEY",
    "VmRasterResult",
    *_UNIT_ALIASES,
]
