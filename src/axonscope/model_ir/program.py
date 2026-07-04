"""Backend-neutral executable membrane program metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from axonscope.utils.units import CONDUCTANCE_DENSITY_MS_CM2

from .expressions import BinaryOp, Call, Expression, Literal, Symbol, UnaryOp
from .schema import ModelIR, State
from .serialization import (
    parameterized_hash as _parameterized_hash,
    structural_hash as _structural_hash,
)


@dataclass(frozen=True, slots=True)
class MembraneNameGroups:
    """Unique display names plus the raw column indices behind each name."""

    names: tuple[str, ...]
    indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class MembraneProgram:
    """Backend-neutral execution contract derived from a validated Model IR.

    This is not a public user-facing API. Users write membrane models; AxonScope
    turns them into this internal program before a backend lowers it.
    """

    model: ModelIR
    name: str
    gate_state_names: tuple[str, ...]
    membrane_state_names: tuple[str, ...]
    gate_trace_observable_names: tuple[str, ...]
    gate_names: tuple[str, ...]
    membrane_state_display_names: tuple[str, ...]
    observable_display_names: tuple[str, ...]
    raw_current_names: tuple[str, ...]
    raw_conductance_names: tuple[str, ...]
    current_names: tuple[str, ...]
    current_groups: tuple[tuple[int, ...], ...]
    conductance_names: tuple[str, ...]
    conductance_groups: tuple[tuple[int, ...], ...]
    conductance_parameter_names: tuple[str | None, ...]
    diagnostic_names: tuple[str, ...]
    final_gate_update_mode: str
    source_provenance: Mapping[str, Any]
    codegen_cache: Mapping[str, Any]
    structural_hash: str
    parameterized_hash: str

    @property
    def membrane_states(self) -> tuple[State, ...]:
        names = set(self.membrane_state_names)
        return tuple(state for state in self.model.states if state.name in names)


def membrane_program_from_model_ir(model: ModelIR) -> MembraneProgram:
    """Derive the backend-neutral membrane program contract from Model IR."""

    gate_state_names = tuple(gate.state for gate in model.gates)
    gate_state_set = set(gate_state_names)
    membrane_state_names = tuple(
        state.name for state in model.states if state.name not in gate_state_set
    )
    gate_trace_observable_names = tuple(
        str(name) for name in model.metadata.get("gate_trace_observables", ())
    )
    state_display_names = _component_public_name_map(model, "states")
    observable_display_names = _component_public_name_map(model, "observables")
    default_label = str(model.metadata.get("component_label", model.name))
    public_gate_state_names = tuple(
        state_display_names.get(
            gate.state,
            _qualified_public_name(default_label, gate.state),
        )
        for gate in model.gates
    )
    public_gate_trace_observable_names = tuple(
        observable_display_names.get(
            name,
            _qualified_public_name(default_label, name),
        )
        for name in gate_trace_observable_names
    )
    public_membrane_state_names = tuple(
        state_display_names.get(
            state_name,
            _qualified_public_name(default_label, state_name),
        )
        for state_name in membrane_state_names
    )
    public_observable_names = tuple(
        observable_display_names.get(
            observable.name,
            _qualified_public_name(default_label, observable.name),
        )
        for observable in model.observables
    )
    raw_current_names = tuple(current.name for current in model.currents)
    raw_conductance_names = tuple(
        conductance_name(name) for name in raw_current_names
    )
    current_groups = group_name_indices(raw_current_names)
    conductance_groups = group_name_indices(raw_conductance_names)
    parameter_units = {
        parameter.name: parameter.quantity.unit for parameter in model.parameters
    }
    conductance_parameter_names = tuple(
        conductance_parameter_name(
            current.name,
            current.conductance,
            parameter_units,
        )
        for current in model.currents
    )
    step = model.step_program
    return MembraneProgram(
        model=model,
        name=model.name,
        gate_state_names=gate_state_names,
        membrane_state_names=membrane_state_names,
        gate_trace_observable_names=gate_trace_observable_names,
        gate_names=(*public_gate_state_names, *public_gate_trace_observable_names),
        membrane_state_display_names=public_membrane_state_names,
        observable_display_names=public_observable_names,
        raw_current_names=raw_current_names,
        raw_conductance_names=raw_conductance_names,
        current_names=current_groups.names,
        current_groups=current_groups.indices,
        conductance_names=conductance_groups.names,
        conductance_groups=conductance_groups.indices,
        conductance_parameter_names=conductance_parameter_names,
        diagnostic_names=() if step is None else tuple(d.name for d in step.diagnostics),
        final_gate_update_mode=str(model.metadata.get("final_gate_update", "predictor")),
        source_provenance=MappingProxyType(
            dict(model.metadata.get("source_provenance", {}))
        ),
        codegen_cache=MappingProxyType(dict(model.metadata.get("codegen_cache", {}))),
        structural_hash=_structural_hash(model),
        parameterized_hash=_parameterized_hash(model),
    )


def _component_public_name_map(model: ModelIR, group: str) -> dict[str, str]:
    names = model.metadata.get("component_public_names", {})
    if not isinstance(names, Mapping):
        return {}
    entries = names.get(group, ())
    if isinstance(entries, str):
        return {}
    try:
        pairs = tuple(entries)
    except TypeError:
        return {}
    out: dict[str, str] = {}
    for entry in pairs:
        try:
            internal, public = entry
        except (TypeError, ValueError):
            continue
        out[str(internal)] = str(public)
    return out


def _qualified_public_name(label: str, name: str) -> str:
    return f"{label}.{name}"


def conductance_name(current_name: str) -> str:
    if current_name.startswith("I_"):
        return "g_" + current_name[2:]
    return f"g_{current_name}"


def conductance_parameter_name(
    current_name: str,
    conductance: Expression,
    parameter_units: dict[str, str],
) -> str | None:
    for name in symbol_names(conductance):
        if parameter_units.get(name) == CONDUCTANCE_DENSITY_MS_CM2:
            return name
    suffix = current_name[2:] if current_name.startswith("I_") else current_name
    candidates = (
        f"g{suffix}bar",
        f"g_{suffix}",
        f"g{suffix}",
        suffix,
    )
    for candidate in candidates:
        if parameter_units.get(candidate) == CONDUCTANCE_DENSITY_MS_CM2:
            return candidate
    return None


def group_name_indices(names: tuple[str, ...]) -> MembraneNameGroups:
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        if name not in groups:
            order.append(name)
            groups[name] = []
        groups[name].append(index)
    return MembraneNameGroups(
        names=tuple(order),
        indices=tuple(tuple(groups[name]) for name in order),
    )


def symbol_names(expr: Expression) -> tuple[str, ...]:
    if isinstance(expr, Literal):
        return ()
    if isinstance(expr, Symbol):
        return (expr.name,)
    if isinstance(expr, UnaryOp):
        return symbol_names(expr.operand)
    if isinstance(expr, BinaryOp):
        return (*symbol_names(expr.left), *symbol_names(expr.right))
    if isinstance(expr, Call):
        names: list[str] = []
        for arg in expr.args:
            names.extend(symbol_names(arg))
        return tuple(names)
    return ()


__all__ = [
    "MembraneNameGroups",
    "MembraneProgram",
    "conductance_name",
    "conductance_parameter_name",
    "group_name_indices",
    "membrane_program_from_model_ir",
    "symbol_names",
]
