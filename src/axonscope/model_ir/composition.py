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
) -> ModelIR:
    """Return one Model IR graph for membrane components in parallel."""

    if not components:
        raise ValueError("compose_model_ir requires at least one component.")
    if len(components) == 1:
        return components[0]
    stateful_components = tuple(component.name for component in components if component.step_program)
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
    metadata: dict[str, Any] = {
        "source": "axonscope.model_ir.compose_model_ir",
        "components": tuple(component.name for component in components),
        "source_provenance": {
            "kind": "composite",
            "components": tuple(
                dict(
                    component.metadata.get(
                        "source_provenance",
                        {"name": component.name, "source": component.metadata.get("source")},
                    )
                )
                for component in components
            ),
        },
    }
    if any(
        component.metadata.get("final_gate_update") == "post_solve_voltage"
        for component in components
    ):
        metadata["final_gate_update"] = "post_solve_voltage"

    for index, component in enumerate(components):
        renames = _symbol_renames(component, symbol_counts, prefix=f"c{index}__")
        parameters.extend(
            _rename_parameter(parameter, renames)
            for parameter in component.parameters
        )
        states.extend(_rename_state(state, renames) for state in component.states)
        gates.extend(_rename_gate(gate, renames) for gate in component.gates)
        currents.extend(_rename_current(current, renames) for current in component.currents)
        observables.extend(
            _rename_observable(observable, renames)
            for observable in component.observables
        )

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


def _rename_observable(observable: Observable, renames: dict[str, str]) -> Observable:
    return replace(
        observable,
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
