import sys as _sys

from .utils import units

from . import (
    axons,
    dispatcher,
    membranes,
    protocols,
    results,
    solvers,
    stimulation,
)
from axonscope.results import analysis, visualization
from axonscope.stimulation import AnalyticalElectrode, Electrode, PointSourceElectrode
from axonscope.axon_simulation import AxonSimulation
from axonscope.results import SimResult
from axonscope.recording import Recording
from axonscope.simulation import simulate, simulate_pool
from axonscope.solvers import SolverOptions
from axonscope.stimulation import (
    AnalyticalExtracellularContext,
    ExtracellularContext,
    IntracellularContext,
    IntracellularCurrentClamp,
    NRVExtracellularContext,
)
from axonscope.stimulation import Stimulus

_sys.modules[__name__ + ".analysis"] = analysis
_sys.modules[__name__ + ".visualization"] = visualization

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
    "axons",
    "analysis",
    "dispatcher",
    "membranes",
    "protocols",
    "results",
    "solvers",
    "stimulation",
    "visualization",
    "units",
    "Stimulus",
    "AxonSimulation",
    "AnalyticalElectrode",
    "AnalyticalExtracellularContext",
    "Electrode",
    "PointSourceElectrode",
    "SolverOptions",
    "simulate",
    "simulate_pool",
    "Recording",
    "IntracellularContext",
    "IntracellularCurrentClamp",
    "ExtracellularContext",
    "NRVExtracellularContext",
    "SimResult",
    *_UNIT_ALIASES,
]
