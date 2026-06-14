from .utils import units

from . import (
    axons,
    dispatcher,
    identifiers,
    membranes,
    positions,
    protocols,
    results,
    signals,
    solvers,
    stimulation,
)
from axonscope.stimulation import AnalyticalElectrode, Electrode, PointSourceElectrode
from axonscope.axon_instance import AxonInstance
from axonscope.identifiers import AxonId, DriveId
from axonscope.population import AxonPopulation
from axonscope.results import SimResult
from axonscope.recording import Recording, RecordingSpatial
from axonscope.signals import Signal
from axonscope.simulation import AxonSimulation, simulate, simulate_pool
from axonscope.solvers import SolverOptions
from axonscope.stimulation import (
    AnalyticalExtracellularContext,
    ExtracellularContext,
    ExtracellularDrive,
    ExtracellularFootprint,
    ExtracellularPotential,
    ExtracellularStimulation,
    IntracellularContext,
    IntracellularCurrentClamp,
    NRVExtracellularContext,
)
from axonscope.stimulation import Stimulus

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
    "dispatcher",
    "identifiers",
    "membranes",
    "positions",
    "protocols",
    "results",
    "signals",
    "solvers",
    "stimulation",
    "units",
    "Stimulus",
    "AxonId",
    "DriveId",
    "AxonInstance",
    "AxonPopulation",
    "AxonSimulation",
    "AnalyticalElectrode",
    "AnalyticalExtracellularContext",
    "Electrode",
    "PointSourceElectrode",
    "SolverOptions",
    "simulate",
    "simulate_pool",
    "Recording",
    "RecordingSpatial",
    "Signal",
    "IntracellularContext",
    "IntracellularCurrentClamp",
    "ExtracellularContext",
    "ExtracellularDrive",
    "ExtracellularFootprint",
    "ExtracellularPotential",
    "ExtracellularStimulation",
    "NRVExtracellularContext",
    "SimResult",
    *_UNIT_ALIASES,
]
