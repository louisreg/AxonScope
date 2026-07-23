"""Immutable schema for backend-neutral membrane model IR."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from axonfleet.utils.units import DIMENSIONLESS

from .expressions import Expression, Symbol, symbol


MODEL_IR_SCHEMA_VERSION = "model_ir.v3"


class Variability(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class SymbolKind(Enum):
    INPUT = "input"
    PARAMETER = "parameter"
    STATE = "state"


class SemanticRole(Enum):
    VOLTAGE = "voltage"
    TIME = "time"
    CONDUCTANCE_DENSITY = "conductance_density"
    CURRENT_DENSITY = "current_density"
    RESISTANCE_AREA = "resistance_area"
    TEMPERATURE = "temperature"
    GATE = "gate"
    OCCUPANCY = "occupancy"
    DIMENSIONLESS = "dimensionless"
    UNKNOWN = "unknown"


class GateUpdateKind(Enum):
    RUSH_LARSEN = "rush_larsen"
    CRANK_NICOLSON = "crank_nicolson"


class KineticUpdateKind(Enum):
    BACKWARD_EULER = "backward_euler"


class KineticInitialization(Enum):
    STATIONARY = "stationary"
    DECLARED = "declared"


class LinearizationGateSource(Enum):
    PREVIOUS = "previous"
    PREDICTOR = "predictor"


@dataclass(frozen=True, slots=True)
class QuantitySpec:
    unit: str = DIMENSIONLESS
    dtype: str = "float32"
    role: SemanticRole = SemanticRole.UNKNOWN


@dataclass(frozen=True, slots=True)
class ModelSymbol:
    name: str
    kind: SymbolKind
    quantity: QuantitySpec

    def ref(self) -> Symbol:
        return symbol(self.name)


@dataclass(frozen=True, slots=True)
class Parameter(ModelSymbol):
    variability: Variability = Variability.DYNAMIC
    default: int | float | bool | None = None

    def __init__(
        self,
        name: str,
        quantity: QuantitySpec,
        *,
        variability: Variability = Variability.DYNAMIC,
        default: int | float | bool | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", SymbolKind.PARAMETER)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "variability", variability)
        object.__setattr__(self, "default", default)


@dataclass(frozen=True, slots=True)
class State(ModelSymbol):
    initial: Expression | None = None

    def __init__(
        self,
        name: str,
        quantity: QuantitySpec,
        *,
        initial: Expression | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", SymbolKind.STATE)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "initial", initial)


@dataclass(frozen=True, slots=True)
class Input(ModelSymbol):
    def __init__(self, name: str, quantity: QuantitySpec) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", SymbolKind.INPUT)
        object.__setattr__(self, "quantity", quantity)


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    state: str
    alpha: Expression
    beta: Expression
    update: GateUpdateKind = GateUpdateKind.RUSH_LARSEN
    q10: Expression | None = None


@dataclass(frozen=True, slots=True)
class KineticTransition:
    source: str
    target: str
    rate: Expression


@dataclass(frozen=True, slots=True)
class KineticBlock:
    """One coupled finite-state kinetic mechanism."""

    name: str
    states: tuple[str, ...]
    transitions: tuple[KineticTransition, ...]
    update: KineticUpdateKind = KineticUpdateKind.BACKWARD_EULER
    initialization: KineticInitialization = KineticInitialization.STATIONARY
    conserve_probability: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "transitions", tuple(self.transitions))


@dataclass(frozen=True, slots=True)
class Current:
    """A linearizable outward membrane-current contribution."""

    name: str
    current: Expression
    conductance: Expression
    reversal: Expression
    quantity: QuantitySpec


@dataclass(frozen=True, slots=True)
class Observable:
    name: str
    expression: Expression
    quantity: QuantitySpec


@dataclass(frozen=True, slots=True)
class StateUpdate:
    state: str
    expression: Expression


@dataclass(frozen=True, slots=True)
class Diagnostic:
    name: str
    expression: Expression
    quantity: QuantitySpec


@dataclass(frozen=True, slots=True)
class StepProgram:
    """Explicit membrane-step semantics beyond alpha/beta gate updates.

    Current terms are evaluated against the incoming auxiliary state. Prepare
    and finalize updates are simultaneous within each phase. Prepare updates
    produce the next auxiliary state for finalization, diagnostics, and the
    following solver step.
    """

    prepare_state_updates: tuple[StateUpdate, ...] = ()
    finalize_state_updates: tuple[StateUpdate, ...] = ()
    total_outward_current: Expression | None = None
    explicit_outward_current: Expression | None = None
    correction_current: Expression | None = None
    prepare_gate_source: LinearizationGateSource = LinearizationGateSource.PREDICTOR
    linearization_gate_source: LinearizationGateSource = LinearizationGateSource.PREDICTOR
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prepare_state_updates", tuple(self.prepare_state_updates))
        object.__setattr__(self, "finalize_state_updates", tuple(self.finalize_state_updates))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True, slots=True)
class ModelIR:
    """Backend-neutral membrane model representation."""

    name: str
    inputs: tuple[Input, ...]
    parameters: tuple[Parameter, ...] = ()
    states: tuple[State, ...] = ()
    gates: tuple[Gate, ...] = ()
    kinetics: tuple[KineticBlock, ...] = ()
    currents: tuple[Current, ...] = ()
    observables: tuple[Observable, ...] = ()
    step_program: StepProgram | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = MODEL_IR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "gates", tuple(self.gates))
        object.__setattr__(self, "kinetics", tuple(self.kinetics))
        object.__setattr__(self, "currents", tuple(self.currents))
        object.__setattr__(self, "observables", tuple(self.observables))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
