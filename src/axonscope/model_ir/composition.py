"""Composition utilities for backend-neutral membrane Model IR graphs."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from .expressions import BinaryOp, Call, Expression, Literal, Symbol, UnaryOp
from .schema import Current, Gate, Input, ModelIR, Observable, Parameter, State
from .validation import assert_valid_model_ir


def compose_model_ir(
    components: tuple[ModelIR, ...],
    *,
    name: str = "composite",
    component_labels: tuple[str, ...] | None = None,
) -> ModelIR:
    """Return one Model IR graph for membrane components in parallel."""

    if not components:
        raise ValueError("compose_model_ir requires at least one component.")
    if len(components) == 1:
        return components[0]
    labels = _component_labels(components, component_labels)
    stateful_components = tuple(
        component.name for component in components if component.step_program
    )
    if stateful_components:
        raise ValueError(
            "Composing Model IR step programs is not defined yet; "
            f"stateful components: {stateful_components!r}."
        )

    symbol_counts = Counter(
        symbol.name
        for component in components
        for symbol in (*component.parameters, *component.states)
    )

    inputs = _merge_inputs(components)
    parameters: list[Parameter] = []
    states: list[State] = []
    gates: list[Gate] = []
    currents: list[Current] = []
    observables: list[Observable] = []
    state_public_names: dict[str, str] = {}
    observable_public_names: dict[str, str] = {}
    gate_trace_observables: list[str] = []
    metadata: dict[str, Any] = {
        "source": "axonscope.model_ir.compose_model_ir",
        "components": tuple(component.name for component in components),
        "component_labels": labels,
        "source_provenance": {
            "kind": "composite",
            "components": tuple(
                dict(
                    component.metadata.get(
                        "source_provenance",
                        {
                            "name": component.name,
                            "source": component.metadata.get("source"),
                        },
                    )
                    | {"component_label": labels[index]}
                )
                for index, component in enumerate(components)
            ),
        },
    }
    if any(
        component.metadata.get("final_gate_update") == "post_solve_voltage"
        for component in components
    ):
        metadata["final_gate_update"] = "post_solve_voltage"

    for index, component in enumerate(components):
        label = labels[index]
        renames = _symbol_renames(component, symbol_counts, prefix=f"c{index}__")
        observable_renames = {
            observable.name: f"{label}__{observable.name}"
            for observable in component.observables
        }
        parameters.extend(
            _rename_parameter(parameter, renames)
            for parameter in component.parameters
        )
        for state in component.states:
            renamed_state = _rename_state(state, renames)
            states.append(renamed_state)
            state_public_names[renamed_state.name] = _qualified_public_name(
                label,
                state.name,
            )
        gates.extend(_rename_gate(gate, renames) for gate in component.gates)
        currents.extend(_rename_current(current, renames) for current in component.currents)
        for observable in component.observables:
            renamed_observable = _rename_observable(
                observable,
                renames,
                name=observable_renames[observable.name],
            )
            observables.append(renamed_observable)
            observable_public_names[renamed_observable.name] = _qualified_public_name(
                label,
                observable.name,
            )
        gate_trace_observables.extend(
            observable_renames[name]
            for name in component.metadata.get("gate_trace_observables", ())
            if name in observable_renames
        )
    metadata["component_public_names"] = {
        "states": tuple(state_public_names.items()),
        "observables": tuple(observable_public_names.items()),
    }
    if gate_trace_observables:
        metadata["gate_trace_observables"] = tuple(gate_trace_observables)

    return assert_valid_model_ir(
        ModelIR(
            name=name,
            inputs=inputs,
            parameters=tuple(parameters),
            states=tuple(states),
            gates=tuple(gates),
            currents=tuple(currents),
            observables=tuple(observables),
            metadata=metadata,
        )
    )


def _component_labels(
    components: tuple[ModelIR, ...],
    explicit: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if explicit is not None:
        labels = tuple(str(label) for label in explicit)
        if len(labels) != len(components):
            raise ValueError("component_labels must match Model IR components.")
        if len(set(labels)) != len(labels):
            raise ValueError("component_labels must be unique.")
        return labels
    labels = tuple(component.name for component in components)
    if len(set(labels)) != len(labels):
        raise ValueError(
            "compose_model_ir needs explicit component_labels when component "
            "names are duplicated."
        )
    return labels


def _qualified_public_name(label: str, name: str) -> str:
    return f"{label}.{name}"


def _merge_inputs(components: tuple[ModelIR, ...]) -> tuple[Input, ...]:
    inputs: dict[str, Input] = {}
    for component in components:
        for input_symbol in component.inputs:
            previous = inputs.get(input_symbol.name)
            if previous is None:
                inputs[input_symbol.name] = input_symbol
            elif previous.quantity != input_symbol.quantity:
                raise ValueError(
                    "Cannot compose Model IR inputs with different specs: "
                    f"{input_symbol.name!r}."
                )
    return tuple(inputs.values())


def _symbol_renames(
    component: ModelIR,
    symbol_counts: Counter[str],
    *,
    prefix: str,
) -> dict[str, str]:
    renames: dict[str, str] = {}
    for symbol in (*component.parameters, *component.states):
        if symbol_counts[symbol.name] > 1:
            renames[symbol.name] = f"{prefix}{symbol.name}"
    return renames


def _rename_parameter(parameter: Parameter, renames: dict[str, str]) -> Parameter:
    return Parameter(
        renames.get(parameter.name, parameter.name),
        parameter.quantity,
        variability=parameter.variability,
        default=parameter.default,
    )


def _rename_state(state: State, renames: dict[str, str]) -> State:
    return State(
        renames.get(state.name, state.name),
        state.quantity,
        initial=None if state.initial is None else rewrite_symbols(state.initial, renames),
    )


def _rename_gate(gate: Gate, renames: dict[str, str]) -> Gate:
    return replace(
        gate,
        name=renames.get(gate.name, gate.name),
        state=renames.get(gate.state, gate.state),
        alpha=rewrite_symbols(gate.alpha, renames),
        beta=rewrite_symbols(gate.beta, renames),
        q10=None if gate.q10 is None else rewrite_symbols(gate.q10, renames),
    )


def _rename_current(current: Current, renames: dict[str, str]) -> Current:
    return replace(
        current,
        current=rewrite_symbols(current.current, renames),
        conductance=rewrite_symbols(current.conductance, renames),
        reversal=rewrite_symbols(current.reversal, renames),
    )


def _rename_observable(
    observable: Observable,
    renames: dict[str, str],
    *,
    name: str | None = None,
) -> Observable:
    return replace(
        observable,
        name=observable.name if name is None else name,
        expression=rewrite_symbols(observable.expression, renames),
    )


def rewrite_symbols(expr: Expression, renames: dict[str, str]) -> Expression:
    """Return ``expr`` with symbol names replaced from ``renames``."""

    if isinstance(expr, Literal):
        return expr
    if isinstance(expr, Symbol):
        return Symbol(renames.get(expr.name, expr.name))
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, rewrite_symbols(expr.operand, renames))
    if isinstance(expr, BinaryOp):
        return BinaryOp(
            expr.op,
            rewrite_symbols(expr.left, renames),
            rewrite_symbols(expr.right, renames),
        )
    if isinstance(expr, Call):
        return Call(
            expr.intrinsic,
            tuple(rewrite_symbols(arg, renames) for arg in expr.args),
        )
    raise TypeError(f"Unsupported Model IR expression {type(expr).__name__}.")
