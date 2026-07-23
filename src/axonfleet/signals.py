"""Typed public signal descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeAlias, TypeVar

from axonfleet.identifiers import SignalId


T = TypeVar("T")


@dataclass(frozen=True)
class Signal(Generic[T]):
    """Descriptor for one recorded or observed signal.

    Signals are value objects rather than a closed enum so advanced users can
    introduce domain-specific signals later without changing AxonFleet's public
    type surface.
    """

    id: SignalId
    result_key: str
    unit: Any | None = None
    description: str = ""
    quantity_type: type[T] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, SignalId):
            raise TypeError("Signal.id must be a SignalId.")
        result_key = str(self.result_key).strip()
        if not result_key:
            raise ValueError("Signal.result_key must be a non-empty string.")
        object.__setattr__(self, "result_key", result_key)

    def __str__(self) -> str:
        return str(self.id)


SignalSelection: TypeAlias = Signal[Any] | Sequence[Signal[Any]]

MEMBRANE_VOLTAGE: Signal[float] = Signal(
    id=SignalId("membrane_voltage"),
    result_key="Vm",
    unit="millivolt",
    description="Membrane voltage.",
    quantity_type=float,
)
GATES: Signal[float] = Signal(
    id=SignalId("gates"),
    result_key="gates",
    unit=None,
    description="Membrane gating variables.",
    quantity_type=float,
)
CURRENTS: Signal[float] = Signal(
    id=SignalId("currents"),
    result_key="currents",
    unit="microampere / centimeter ** 2",
    description="Membrane current densities.",
    quantity_type=float,
)
CONDUCTANCES: Signal[float] = Signal(
    id=SignalId("conductances"),
    result_key="conductances",
    unit="millisiemens / centimeter ** 2",
    description="Membrane conductance densities.",
    quantity_type=float,
)
STATE_VARIABLES: Signal[float] = Signal(
    id=SignalId("state_variables"),
    result_key="states",
    unit=None,
    description="Auxiliary membrane state variables.",
    quantity_type=float,
)

Vm = MEMBRANE_VOLTAGE


__all__ = [
    "Signal",
    "SignalSelection",
    "MEMBRANE_VOLTAGE",
    "GATES",
    "CURRENTS",
    "CONDUCTANCES",
    "STATE_VARIABLES",
    "Vm",
]
