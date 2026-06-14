"""Typed public signal selectors."""

from __future__ import annotations

from enum import Enum
from typing import Sequence, TypeAlias


class Signal(Enum):
    """Closed set of signal groups that can be requested from recordings."""

    VM = "Vm"
    GATES = "gates"
    CURRENTS = "currents"
    CONDUCTANCES = "conductances"
    STATES = "states"

    @property
    def result_key(self) -> str:
        """Return the key used in `SimResult.recordings`."""

        return str(self.value)


SignalSelection: TypeAlias = Signal | Sequence[Signal]

VM = Signal.VM
Vm = Signal.VM
VOLTAGE = Signal.VM
GATES = Signal.GATES
CURRENTS = Signal.CURRENTS
CONDUCTANCES = Signal.CONDUCTANCES
STATES = Signal.STATES
STATE_VARIABLES = Signal.STATES


__all__ = [
    "Signal",
    "SignalSelection",
    "VM",
    "Vm",
    "VOLTAGE",
    "GATES",
    "CURRENTS",
    "CONDUCTANCES",
    "STATES",
    "STATE_VARIABLES",
]
