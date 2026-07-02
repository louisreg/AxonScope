"""Semantic quantity names for plain-Python membrane source files.

These names are compiler markers, not runtime numeric classes. They are typed
as ``Any`` so editors do not reject ordinary Python equations such as
``Vm + 40 * mV`` while the source compiler still reads the annotation names
from the AST.
"""

from __future__ import annotations

from typing import Any, TypeAlias


Concentration: TypeAlias = Any
ConcentrationPerCurrentDensityTime: TypeAlias = Any
ConductanceDensity: TypeAlias = Any
CurrentDensity: TypeAlias = Any
Dimensionless: TypeAlias = Any
Gate: TypeAlias = Any
Length: TypeAlias = Any
Rate: TypeAlias = Any
RatePerConcentration: TypeAlias = Any
RatePerVoltage: TypeAlias = Any
ResistanceArea: TypeAlias = Any
Temperature: TypeAlias = Any
Time: TypeAlias = Any
Voltage: TypeAlias = Any


__all__ = [
    "Concentration",
    "ConcentrationPerCurrentDensityTime",
    "ConductanceDensity",
    "CurrentDensity",
    "Dimensionless",
    "Gate",
    "Length",
    "Rate",
    "RatePerConcentration",
    "RatePerVoltage",
    "ResistanceArea",
    "Temperature",
    "Time",
    "Voltage",
]
