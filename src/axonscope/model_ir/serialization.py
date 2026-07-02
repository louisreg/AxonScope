"""Canonical serialization and hashing for Model IR structures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, cast

from .expressions import (
    BinaryOp,
    BinaryOperator,
    Call,
    Expression,
    Literal,
    Symbol,
    UnaryOp,
    UnaryOperator,
)
from .schema import (
    Current,
    Diagnostic,
    FunctionDef,
    Gate,
    GateUpdateKind,
    Input,
    LinearizationGateSource,
    ModelIR,
    MODEL_IR_SCHEMA_VERSION,
    Observable,
    Parameter,
    QuantitySpec,
    SemanticRole,
    ShapeSpec,
    State,
    StateUpdate,
    StepProgram,
    Variability,
)


def expression_data(expr: Expression) -> dict[str, Any]:
    if isinstance(expr, Literal):
        return {"node": "literal", "value": expr.value, "unit": expr.unit}
    if isinstance(expr, Symbol):
        return {"node": "symbol", "name": expr.name}
    if isinstance(expr, UnaryOp):
        return {
            "node": "unary",
            "op": expr.op,
            "operand": expression_data(expr.operand),
        }
    if isinstance(expr, BinaryOp):
        return {
            "node": "binary",
            "op": expr.op,
            "left": expression_data(expr.left),
            "right": expression_data(expr.right),
        }
    if isinstance(expr, Call):
        return {
            "node": "call",
            "intrinsic": expr.intrinsic,
            "args": [expression_data(arg) for arg in expr.args],
        }
    raise TypeError(f"Unsupported expression node {type(expr).__name__}.")


def canonical_data(value: Any, *, include_dynamic_values: bool = False) -> Any:
    if isinstance(value, Expression):
        return expression_data(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple | list):
        return [
            canonical_data(item, include_dynamic_values=include_dynamic_values)
            for item in value
        ]
    if isinstance(value, Mapping):
        return {
            str(key): canonical_data(val, include_dynamic_values=include_dynamic_values)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if is_dataclass(value) and not isinstance(value, type):
        out: dict[str, Any] = {
            "type": value.__class__.__name__,
        }
        for field in fields(value):
            if isinstance(value, Parameter) and field.name == "default":
                if value.variability is Variability.DYNAMIC and not include_dynamic_values:
                    out["has_default"] = value.default is not None
                    continue
            out[field.name] = canonical_data(
                getattr(value, field.name),
                include_dynamic_values=include_dynamic_values,
            )
        return out
    return value


def canonical_json(value: Any, *, include_dynamic_values: bool = False) -> str:
    return json.dumps(
        canonical_data(value, include_dynamic_values=include_dynamic_values),
        sort_keys=True,
        separators=(",", ":"),
    )


def expression_from_data(data: Any) -> Expression:
    """Restore an expression from canonical serialized data."""

    if not isinstance(data, Mapping):
        raise TypeError("Serialized expression must be a mapping.")
    node = data.get("node")
    if node == "literal":
        return Literal(data["value"], str(data.get("unit", "")))
    if node == "symbol":
        return Symbol(str(data["name"]))
    if node == "unary":
        op = str(data["op"])
        if op != "neg":
            raise TypeError(f"Unsupported unary expression operator {op!r}.")
        return UnaryOp(cast(UnaryOperator, op), expression_from_data(data["operand"]))
    if node == "binary":
        op = str(data["op"])
        if op not in {"add", "sub", "mul", "div", "pow", "lt", "le", "gt", "ge"}:
            raise TypeError(f"Unsupported binary expression operator {op!r}.")
        return BinaryOp(
            cast(BinaryOperator, op),
            expression_from_data(data["left"]),
            expression_from_data(data["right"]),
        )
    if node == "call":
        return Call(
            str(data["intrinsic"]),
            tuple(expression_from_data(arg) for arg in data.get("args", ())),
        )
    raise TypeError(f"Unsupported serialized expression node {node!r}.")


def model_ir_from_json(raw: str) -> ModelIR:
    """Restore a Model IR graph from canonical JSON."""

    return model_ir_from_data(json.loads(raw))


def model_ir_from_data(data: Any) -> ModelIR:
    """Restore a Model IR graph from canonical serialized data."""

    mapping = _typed_mapping(data, "ModelIR")
    schema_version = str(mapping.get("schema_version", MODEL_IR_SCHEMA_VERSION))
    model = ModelIR(
        name=str(mapping["name"]),
        inputs=tuple(_input_from_data(item) for item in mapping.get("inputs", ())),
        parameters=tuple(
            _parameter_from_data(item) for item in mapping.get("parameters", ())
        ),
        states=tuple(_state_from_data(item) for item in mapping.get("states", ())),
        functions=tuple(
            _function_from_data(item) for item in mapping.get("functions", ())
        ),
        gates=tuple(_gate_from_data(item) for item in mapping.get("gates", ())),
        currents=tuple(_current_from_data(item) for item in mapping.get("currents", ())),
        observables=tuple(
            _observable_from_data(item) for item in mapping.get("observables", ())
        ),
        step_program=_optional_step_program_from_data(mapping.get("step_program")),
        metadata=_metadata_from_data(mapping.get("metadata", {})),
        schema_version=schema_version,
    )
    return model


def structural_hash(value: Any) -> str:
    payload = canonical_json(value, include_dynamic_values=False).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=20).hexdigest()


def parameterized_hash(value: Any) -> str:
    payload = canonical_json(value, include_dynamic_values=True).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=20).hexdigest()


def _typed_mapping(data: Any, type_name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"Serialized {type_name} must be a mapping.")
    if data.get("type") != type_name:
        raise TypeError(f"Serialized value must have type {type_name!r}.")
    return data


def _quantity_from_data(data: Any) -> QuantitySpec:
    mapping = _typed_mapping(data, "QuantitySpec")
    return QuantitySpec(
        unit=str(mapping.get("unit", "")),
        shape=_shape_from_data(mapping.get("shape")),
        dtype=str(mapping.get("dtype", "float32")),
        role=SemanticRole(str(mapping.get("role", SemanticRole.UNKNOWN.value))),
    )


def _shape_from_data(data: Any) -> ShapeSpec:
    if data is None:
        return ShapeSpec(())
    mapping = _typed_mapping(data, "ShapeSpec")
    return ShapeSpec(tuple(mapping.get("dims", ())))


def _input_from_data(data: Any) -> Input:
    mapping = _typed_mapping(data, "Input")
    return Input(str(mapping["name"]), _quantity_from_data(mapping["quantity"]))


def _parameter_from_data(data: Any) -> Parameter:
    mapping = _typed_mapping(data, "Parameter")
    default = mapping.get("default")
    return Parameter(
        str(mapping["name"]),
        _quantity_from_data(mapping["quantity"]),
        variability=Variability(str(mapping.get("variability", Variability.DYNAMIC.value))),
        default=None if default is None else float(default),
    )


def _state_from_data(data: Any) -> State:
    mapping = _typed_mapping(data, "State")
    initial = mapping.get("initial")
    return State(
        str(mapping["name"]),
        _quantity_from_data(mapping["quantity"]),
        initial=None if initial is None else expression_from_data(initial),
    )


def _function_from_data(data: Any) -> FunctionDef:
    mapping = _typed_mapping(data, "FunctionDef")
    return FunctionDef(
        str(mapping["name"]),
        tuple(_model_symbol_from_data(item) for item in mapping.get("args", ())),
        expression_from_data(mapping["body"]),
        _quantity_from_data(mapping["quantity"]),
    )


def _model_symbol_from_data(data: Any) -> Input | Parameter | State:
    if not isinstance(data, Mapping):
        raise TypeError("Serialized model symbol must be a mapping.")
    symbol_type = data.get("type")
    if symbol_type == "Input":
        return _input_from_data(data)
    if symbol_type == "Parameter":
        return _parameter_from_data(data)
    if symbol_type == "State":
        return _state_from_data(data)
    raise TypeError(f"Unsupported serialized model symbol {symbol_type!r}.")


def _gate_from_data(data: Any) -> Gate:
    mapping = _typed_mapping(data, "Gate")
    q10 = mapping.get("q10")
    return Gate(
        str(mapping["name"]),
        state=str(mapping["state"]),
        alpha=expression_from_data(mapping["alpha"]),
        beta=expression_from_data(mapping["beta"]),
        update=GateUpdateKind(str(mapping.get("update", GateUpdateKind.RUSH_LARSEN.value))),
        q10=None if q10 is None else expression_from_data(q10),
    )


def _current_from_data(data: Any) -> Current:
    mapping = _typed_mapping(data, "Current")
    return Current(
        str(mapping["name"]),
        current=expression_from_data(mapping["current"]),
        conductance=expression_from_data(mapping["conductance"]),
        reversal=expression_from_data(mapping["reversal"]),
        quantity=_quantity_from_data(mapping["quantity"]),
    )


def _observable_from_data(data: Any) -> Observable:
    mapping = _typed_mapping(data, "Observable")
    return Observable(
        str(mapping["name"]),
        expression=expression_from_data(mapping["expression"]),
        quantity=_quantity_from_data(mapping["quantity"]),
    )


def _state_update_from_data(data: Any) -> StateUpdate:
    mapping = _typed_mapping(data, "StateUpdate")
    return StateUpdate(
        str(mapping["state"]),
        expression_from_data(mapping["expression"]),
    )


def _diagnostic_from_data(data: Any) -> Diagnostic:
    mapping = _typed_mapping(data, "Diagnostic")
    return Diagnostic(
        str(mapping["name"]),
        expression_from_data(mapping["expression"]),
        _quantity_from_data(mapping["quantity"]),
    )


def _optional_step_program_from_data(data: Any) -> StepProgram | None:
    if data is None:
        return None
    mapping = _typed_mapping(data, "StepProgram")
    return StepProgram(
        prepare_state_updates=tuple(
            _state_update_from_data(item)
            for item in mapping.get("prepare_state_updates", ())
        ),
        finalize_state_updates=tuple(
            _state_update_from_data(item)
            for item in mapping.get("finalize_state_updates", ())
        ),
        total_outward_current=_optional_expression_from_data(
            mapping.get("total_outward_current")
        ),
        explicit_outward_current=_optional_expression_from_data(
            mapping.get("explicit_outward_current")
        ),
        correction_current=_optional_expression_from_data(mapping.get("correction_current")),
        prepare_gate_source=LinearizationGateSource(
            str(mapping.get("prepare_gate_source", LinearizationGateSource.PREDICTOR.value))
        ),
        linearization_gate_source=LinearizationGateSource(
            str(
                mapping.get(
                    "linearization_gate_source",
                    LinearizationGateSource.PREDICTOR.value,
                )
            )
        ),
        diagnostics=tuple(
            _diagnostic_from_data(item) for item in mapping.get("diagnostics", ())
        ),
    )


def _optional_expression_from_data(data: Any) -> Expression | None:
    return None if data is None else expression_from_data(data)


def _metadata_from_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        return {}
    metadata = {str(key): _plain_data(value) for key, value in data.items()}
    provenance = metadata.get("source_provenance")
    if isinstance(provenance, dict):
        for key in ("function_names", "intrinsics"):
            if isinstance(provenance.get(key), list):
                provenance[key] = tuple(provenance[key])
    cache = metadata.get("codegen_cache")
    if isinstance(cache, dict):
        for key in ("files", "targets"):
            if isinstance(cache.get(key), list):
                cache[key] = tuple(cache[key])
    source_outputs = metadata.get("source_outputs")
    if isinstance(source_outputs, dict):
        for key in ("all", "currents", "observables"):
            if isinstance(source_outputs.get(key), list):
                source_outputs[key] = tuple(source_outputs[key])
    return metadata


def _plain_data(data: Any) -> Any:
    if isinstance(data, Mapping):
        return {str(key): _plain_data(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_plain_data(item) for item in data]
    return data
