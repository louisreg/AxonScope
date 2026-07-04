"""Compile plain Python membrane source files into the internal model graph."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from axonscope.utils import units

from .expressions import BinaryOp, Call, Expression, Literal, Symbol, UnaryOp, literal, symbol
from .intrinsics import DEFAULT_INTRINSICS, call
from .schema import (
    Current,
    Diagnostic,
    Gate,
    GateUpdateKind,
    Input,
    LinearizationGateSource,
    MODEL_IR_SCHEMA_VERSION,
    ModelIR,
    Observable,
    Parameter,
    QuantitySpec,
    SemanticRole,
    State,
    StateUpdate,
    StepProgram,
    Variability,
)
from .unit_algebra import product_unit, quotient_unit
from .serialization import canonical_json, model_ir_from_json, structural_hash
from .validation import assert_valid_model_ir


SOURCE_CONTRACT_VERSION = "plain_python_membrane.v1"
SOURCE_COMPILER_VERSION = "source_codegen.v11"
SOURCE_CACHE_INDEX_VERSION = "source_cache_index.v1"
STEP_SPECIAL_SYMBOL_UNITS = {
    "Vm_prev": units.VOLTAGE_MV,
    "Vm_new": units.VOLTAGE_MV,
    "I_ion": units.CURRENT_DENSITY_UA_CM2,
    "I_background": units.CURRENT_DENSITY_UA_CM2,
}


class SourceModelCompileError(ValueError):
    """Raised when a plain Python membrane source cannot be compiled."""


def _source_error(node: ast.AST, message: str) -> SourceModelCompileError:
    return SourceModelCompileError(_source_message(node, message))


def _source_message(node: ast.AST, message: str) -> str:
    lineno = getattr(node, "lineno", None)
    if not isinstance(lineno, int):
        return message
    col = getattr(node, "col_offset", None)
    if isinstance(col, int):
        return f"line {lineno}, column {col + 1}: {message}"
    return f"line {lineno}: {message}"


@dataclass(frozen=True, slots=True)
class GeneratedCodeCache:
    """Persistent generated-code cache information for one source model."""

    key: str
    directory: Path
    manifest_path: Path
    cache_hit: bool
    cache_reason: str
    generated_files: tuple[Path, ...]
    loaded_modules: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class SourceModelCompileResult:
    """Compiled source model and generated-code cache metadata."""

    model: ModelIR
    source_hash: str
    source_path: Path
    function_name: str
    cache: GeneratedCodeCache


@dataclass(frozen=True, slots=True)
class _SourceProgram:
    metadata: dict[str, Any]
    functions: tuple[ast.FunctionDef, ...]
    function_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AssignmentRecord:
    name: str
    value: ast.AST
    statement: ast.stmt
    function_name: str


def compile_model_source_file(
    path: str | os.PathLike[str],
    *,
    function_name: str = "equations",
    model_class_name: str | None = None,
    parameter_defaults: dict[str, float] | None = None,
    cache_root: str | os.PathLike[str] | None = None,
    load_generated_modules: tuple[str, ...] = (),
) -> SourceModelCompileResult:
    """Compile one standalone plain-Python membrane source file.

    The source is parsed with ``ast``; equations are not executed through a
    symbolic builder. Dynamic parameter overrides update the returned model
    defaults but do not participate in source hashing or generated-code cache
    identity.
    """

    source_path = Path(path).resolve()
    source_text = source_path.read_text(encoding="utf-8")
    source_text_hash = _source_text_hash(source_text)
    cached = _try_load_compiled_source_cache(
        source_path=source_path,
        source_text_hash=source_text_hash,
        function_name=function_name,
        model_class_name=model_class_name,
        parameter_defaults=parameter_defaults or {},
        cache_root=cache_root,
        load_generated_modules=load_generated_modules,
    )
    if cached is not None:
        return cached

    tree = ast.parse(source_text, filename=str(source_path))
    program = _source_program(
        tree,
        source_path=source_path,
        function_name=function_name,
        model_class_name=model_class_name,
    )
    metadata = program.metadata
    source_hash = _source_hash(tree, metadata=metadata)
    assignments = _compile_assignments(program.functions, metadata)
    model = _build_model_ir(
        metadata,
        assignments=assignments,
        source_path=source_path,
        function_name=",".join(program.function_names),
        source_hash=source_hash,
        parameter_defaults=parameter_defaults or {},
    )
    cache = _ensure_generated_cache(
        model=model,
        source_path=source_path,
        source_text=source_text,
        source_text_hash=source_text_hash,
        functions=program.functions,
        metadata=metadata,
        assignments=assignments,
        source_hash=source_hash,
        cache_root=cache_root,
        load_generated_modules=load_generated_modules,
    )
    _write_source_cache_index(
        source_path=source_path,
        source_text_hash=source_text_hash,
        requested_function_name=function_name,
        compiled_function_name=",".join(program.function_names),
        model_class_name=model_class_name,
        cache=cache,
        source_hash=source_hash,
        cache_root=cache_root,
    )
    model = _with_codegen_cache_metadata(model, cache)
    return SourceModelCompileResult(
        model=model,
        source_hash=source_hash,
        source_path=source_path,
        function_name=",".join(program.function_names),
        cache=cache,
    )


def _source_program(
    tree: ast.Module,
    *,
    source_path: Path,
    function_name: str,
    model_class_name: str | None,
) -> _SourceProgram:
    model_class = _find_model_class(tree, model_class_name=model_class_name)
    scope = ast.Module(body=model_class.body, type_ignores=[]) if model_class else tree
    function_map = {
        node.name: node
        for node in scope.body
        if isinstance(node, ast.FunctionDef)
    }
    declaration = (
        _class_model_declaration(model_class)
        if model_class is not None
        else _find_model_metadata(scope)
    )
    class_parameters = (
        _declared_class_parameters(model_class)
        if model_class is not None
        else {}
    )
    declared_states = _declared_model_states(scope)
    export_groups = _declared_model_export_groups(scope)
    dynamics = _declared_model_dynamics(scope)
    exported_returns = export_groups["currents"] + export_groups["observables"]
    if declaration is None:
        function = _find_function(scope, function_name)
        metadata = _infer_model_metadata(
            source_path,
            functions=(function,),
            returns=exported_returns or None,
            declared_states=declared_states,
            dynamics=dynamics,
        )
        _merge_parameter_metadata(metadata, class_parameters)
        return _SourceProgram(
            metadata=metadata,
            functions=(function,),
            function_names=(function.name,),
        )
    names = tuple(
        declaration.get("functions")
        or _decorated_model_function_names(scope)
        or (function_name,)
    )
    functions = tuple(_lookup_function(function_map, name) for name in names)
    if "inputs" in declaration or "parameters" in declaration or "currents" in declaration:
        metadata = declaration
    else:
        metadata = _infer_model_metadata(
            source_path,
            functions=functions,
            returns=(
                declaration["returns"]
                if "returns" in declaration
                else exported_returns
            ),
            name=declaration.get("name"),
            declared_states=declared_states,
            dynamics=dynamics,
        )
        _merge_parameter_metadata(metadata, class_parameters)
        if isinstance(declaration.get("metadata"), dict):
            metadata["metadata"] = declaration["metadata"]
    _merge_parameter_metadata(metadata, class_parameters)
    if dynamics and "step" not in metadata:
        metadata["step"] = _step_metadata_from_dynamics(
            dynamics,
            local_specs=_local_quantity_specs(functions),
        )
    _merge_source_metadata_descriptions(metadata, declared_states, export_groups)
    return _SourceProgram(metadata=metadata, functions=functions, function_names=names)


def _declared_model_states(tree: ast.Module) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        if node.value is None or not _is_model_state_call(node.value):
            continue
        spec = _annotation_spec(node.annotation, label=node.target.id)
        state_declaration = _model_state_declaration(node.value, label=node.target.id)
        states[node.target.id] = {
            **spec,
            "initial": _metadata_default(
                state_declaration.pop("initial"),
                expected_unit=spec["unit"],
                label=node.target.id,
            ),
            **state_declaration,
        }
    return states


def _is_model_state_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == "state"
    return isinstance(node.func, ast.Attribute) and node.func.attr == "state"


def _model_state_declaration(node: ast.AST, *, label: str) -> dict[str, Any]:
    if not isinstance(node, ast.Call):
        raise SourceModelCompileError(f"State declaration {label!r} must call state(...).")
    values: dict[str, Any] = {}
    if len(node.args) > 1:
        raise SourceModelCompileError(
            f"State declaration {label!r} must use state(initial, ...)."
        )
    if node.args:
        values["initial"] = node.args[0]
    for keyword in node.keywords:
        if keyword.arg is None:
            raise SourceModelCompileError("state(...) does not support **kwargs.")
        if keyword.arg == "initial":
            if "initial" in values:
                raise SourceModelCompileError(
                    f"State declaration {label!r} defines initial twice."
                )
            values["initial"] = keyword.value
        elif keyword.arg in {"description", "source", "bounds"}:
            values[keyword.arg] = _metadata_value(keyword.value)
        else:
            raise SourceModelCompileError(
                f"Unsupported state(...) keyword {keyword.arg!r}."
            )
    if "initial" not in values:
        raise SourceModelCompileError(
            f"State declaration {label!r} must use state(initial)."
        )
    return values


def _declared_model_export_groups(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {
        "currents": [],
        "observables": [],
        "internal": [],
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            decorator = _section_decorator_call(node, "currents")
            if decorator is not None:
                _merge_currents_decorator_exports(groups, decorator)
    _validate_export_groups(groups)
    return {name: tuple(values) for name, values in groups.items()}


def _merge_currents_decorator_exports(
    groups: dict[str, list[str]],
    decorator: ast.Call,
) -> None:
    if decorator.args:
        raise SourceModelCompileError("@currents(...) only supports keyword arguments.")
    for keyword in decorator.keywords:
        if keyword.arg is None:
            raise SourceModelCompileError("@currents(...) does not support **kwargs.")
        if keyword.arg in {"outputs", "currents"}:
            groups["currents"].extend(_name_tuple(keyword.value, label="@currents.outputs"))
        elif keyword.arg == "observables":
            groups["observables"].extend(
                _name_tuple(keyword.value, label="@currents.observables")
            )
        elif keyword.arg == "internal":
            groups["internal"].extend(_name_tuple(keyword.value, label="@currents.internal"))
        else:
            raise SourceModelCompileError(
                f"Unsupported @currents(...) keyword {keyword.arg!r}."
            )


def _validate_export_groups(groups: dict[str, list[str]]) -> None:
    for group_name, values in groups.items():
        duplicate = _first_duplicate(values)
        if duplicate is not None:
            raise SourceModelCompileError(
                f"Duplicate @currents {group_name} name {duplicate!r}."
            )
    current_outputs = set(groups["currents"])
    observable_outputs = set(groups["observables"])
    overlap = sorted(current_outputs & observable_outputs)
    if overlap:
        raise SourceModelCompileError(
            "@currents(...) cannot expose the same expression as both current "
            "and observable: " + ", ".join(overlap)
        )


def _first_duplicate(values: list[str] | tuple[str, ...]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _declared_model_dynamics(tree: ast.Module) -> dict[str, Any] | None:
    declaration: dict[str, Any] | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for section_name in ("initials", "step"):
                decorator = _section_decorator_call(node, section_name)
                if decorator is None or not decorator.keywords:
                    continue
                if declaration is None:
                    declaration = {}
                _merge_dynamics_decorator(
                    declaration,
                    decorator,
                    section_name=section_name,
                )
    return declaration


def _merge_dynamics_decorator(
    declaration: dict[str, Any],
    decorator: ast.Call,
    *,
    section_name: str,
) -> None:
    if decorator.args:
        raise SourceModelCompileError(f"@{section_name}(...) only supports keyword arguments.")
    for keyword in decorator.keywords:
        if keyword.arg is None:
            raise SourceModelCompileError(f"@{section_name}(...) does not support **kwargs.")
        if section_name == "initials" and keyword.arg in {"updates", "initials"}:
            _set_dynamics_key(
                declaration,
                "initials",
                _name_mapping(keyword.value, label=f"@{section_name}.{keyword.arg}"),
                source=f"@{section_name}(...)",
            )
        elif section_name == "step" and keyword.arg in {"prepare", "finalize", "diagnostics"}:
            _set_dynamics_key(
                declaration,
                keyword.arg,
                _name_mapping(keyword.value, label=f"@{section_name}.{keyword.arg}"),
                source=f"@{section_name}(...)",
            )
        elif section_name == "step" and keyword.arg in {
            "total_outward_current",
            "explicit_outward_current",
            "correction_current",
            "prepare_gate_source",
            "linearization_gate_source",
        }:
            _set_dynamics_key(
                declaration,
                keyword.arg,
                _name_reference(keyword.value, label=f"@{section_name}.{keyword.arg}"),
                source=f"@{section_name}(...)",
            )
        else:
            raise SourceModelCompileError(
                f"Unsupported @{section_name}(...) keyword {keyword.arg!r}."
            )


def _set_dynamics_key(
    declaration: dict[str, Any],
    key: str,
    value: Any,
    *,
    source: str,
) -> None:
    if key in declaration:
        raise SourceModelCompileError(
            f"{source} declares dynamics key {key!r} more than once."
        )
    declaration[key] = value


def _merge_source_metadata_descriptions(
    metadata: dict[str, Any],
    declared_states: dict[str, dict[str, Any]],
    export_groups: dict[str, tuple[str, ...]],
) -> None:
    source_metadata = dict(metadata.get("metadata", {}))
    state_docs = {
        name: {
            key: value
            for key, value in spec.items()
            if key in {"description", "source", "bounds"}
        }
        for name, spec in declared_states.items()
        if any(key in spec for key in ("description", "source", "bounds"))
    }
    if state_docs:
        source_metadata.setdefault("states", state_docs)
    if export_groups["internal"]:
        source_metadata.setdefault("internal_outputs", export_groups["internal"])
    if source_metadata:
        metadata["metadata"] = source_metadata


def _is_model_method_call(node: ast.AST, method_name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"model", "self"}
        and node.func.attr == method_name
    )


def _find_model_class(
    tree: ast.Module,
    *,
    model_class_name: str | None,
) -> ast.ClassDef | None:
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and _is_model_class(node)
    ]
    if model_class_name is not None:
        for node in classes:
            if node.name == model_class_name:
                return node
        raise SourceModelCompileError(
            f"Plain Python membrane source does not define Model class {model_class_name!r}."
        )
    if not classes:
        return None
    if len(classes) > 1:
        names = ", ".join(node.name for node in classes)
        raise SourceModelCompileError(
            "Plain Python membrane source defines multiple Model classes; "
            f"select one explicitly: {names}."
        )
    return classes[0]


def _is_model_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "Model":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Model":
            return True
    return False


def _class_model_declaration(node: ast.ClassDef) -> dict[str, Any]:
    for legacy_name, replacement in {
        "exports": "@currents(outputs=..., observables=..., internal=...)",
        "dynamics": "@initials(updates=...) or @step(...)",
    }.items():
        if _class_literal_attr(node, legacy_name) is not None:
            raise SourceModelCompileError(
                f"{node.name}.{legacy_name} is no longer supported; use {replacement}."
            )
    values: dict[str, Any] = {
        "name": _class_model_name(node),
    }
    function_names = _decorated_model_function_names(ast.Module(body=node.body, type_ignores=[]))
    if function_names:
        values["functions"] = function_names
    returns = _class_literal_attr(node, "returns")
    if returns is not None:
        names = _name_tuple(returns, label="returns")
        duplicate = _first_duplicate(names)
        if duplicate is not None:
            raise SourceModelCompileError(
                f"Duplicate Model returns name {duplicate!r}."
            )
        values["returns"] = names
    metadata = _class_metadata_attr(node)
    if metadata:
        values["metadata"] = metadata
    return values


def _class_model_name(node: ast.ClassDef) -> str:
    for attr_name in ("model_kind", "kind", "name"):
        value = _class_literal_attr(node, attr_name)
        if value is None:
            continue
        try:
            metadata_value = _metadata_value(value)
        except SourceModelCompileError:
            continue
        if isinstance(metadata_value, str) and metadata_value:
            return metadata_value
    return _snake_case(node.name)


def _snake_case(name: str) -> str:
    first = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def _class_metadata_attr(node: ast.ClassDef) -> dict[str, Any]:
    value = _class_literal_attr(node, "metadata")
    if value is None:
        return {}
    metadata = _metadata_value(value)
    if not isinstance(metadata, dict):
        raise SourceModelCompileError(f"{node.name}.metadata must be a dictionary.")
    return metadata


def _class_literal_attr(node: ast.ClassDef, name: str) -> ast.AST | None:
    for statement in node.body:
        target_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            target_name = statement.target.id
            value = statement.value
        if target_name == name:
            return value
    return None


def _declared_class_parameters(node: ast.ClassDef) -> dict[str, dict[str, Any]]:
    parameters: dict[str, dict[str, Any]] = {}
    reserved = {
        "dynamics",
        "exports",
        "metadata",
        "model_kind",
        "name",
        "parameter_aliases",
        "parameter_defaults",
        "parameter_units",
        "returns",
    }
    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign):
            continue
        if not isinstance(statement.target, ast.Name):
            continue
        name = statement.target.id
        if name.startswith("_") or name in reserved:
            continue
        if _annotation_is_classvar(statement.annotation):
            continue
        if statement.value is None or _is_model_state_call(statement.value):
            continue
        spec = _annotation_spec(statement.annotation, label=name)
        parameters[name] = {
            **spec,
            "default": _metadata_default(
                statement.value,
                expected_unit=spec["unit"],
                label=name,
            ),
        }
    return parameters


def _annotation_is_classvar(annotation: ast.AST | None) -> bool:
    if isinstance(annotation, ast.Name):
        return annotation.id == "ClassVar"
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "ClassVar"
    if isinstance(annotation, ast.Subscript):
        return _annotation_is_classvar(annotation.value)
    return False


def _merge_parameter_metadata(
    metadata: dict[str, Any],
    parameters: dict[str, dict[str, Any]],
) -> None:
    if not parameters:
        return
    target = metadata.setdefault("parameters", {})
    if not isinstance(target, dict):
        raise SourceModelCompileError("MODEL['parameters'] must be a dictionary.")
    for name, spec in parameters.items():
        _merge_symbol(target, name, spec, section="parameter")


def _find_model_metadata(tree: ast.Module) -> dict[str, Any] | None:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "MODEL":
            value = _metadata_value(node.value)
            if not isinstance(value, dict):
                raise SourceModelCompileError("MODEL must be a dictionary.")
            return value
        if isinstance(target, ast.Name) and target.id == "model":
            return _model_declaration(node.value)
    return None


def _lookup_function(functions: dict[str, ast.FunctionDef], name: str) -> ast.FunctionDef:
    try:
        return functions[name]
    except KeyError as exc:
        raise SourceModelCompileError(
            f"Plain Python membrane source references unknown function {name!r}."
        ) from exc


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SourceModelCompileError(f"Plain Python membrane source must define {name}(...).")


def _decorated_model_function_names(tree: ast.Module) -> tuple[str, ...]:
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if any(_is_model_section_decorator(decorator) for decorator in node.decorator_list):
            names.append(node.name)
    return tuple(names)


def _is_model_section_decorator(node: ast.AST) -> bool:
    if _decorator_section_name(node) is not None:
        return True
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id == "model"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    ):
        return node.func.value.id == "model"
    return False


def _decorator_section_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and node.id in {"currents", "initials", "rates", "step"}:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in {"currents", "initials", "rates", "step"}:
        return node.attr
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"currents", "initials", "rates", "step"}:
            return node.func.id
        if node.func.id == "section" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
        if node.func.id == "mechanism" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return f"mechanism:{arg.value}"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in {"currents", "initials", "rates", "step"}:
            return node.func.attr
        if node.func.attr == "section" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
        if node.func.attr == "mechanism" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return f"mechanism:{arg.value}"
    return None


def _section_decorator_call(function: ast.FunctionDef, section_name: str) -> ast.Call | None:
    for decorator in function.decorator_list:
        if _decorator_section_name(decorator) != section_name:
            continue
        if isinstance(decorator, ast.Call):
            return decorator
    return None


def _model_declaration(node: ast.AST) -> dict[str, Any]:
    if not isinstance(node, ast.Call):
        raise SourceModelCompileError("model must be declared with Model(...).")
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        func_name = node.func.attr
    else:
        func_name = ""
    if func_name != "Model":
        raise SourceModelCompileError("model must be declared with Model(...).")
    values: dict[str, Any] = {}
    function_names: list[str] = []
    if node.args:
        name = _metadata_value(node.args[0])
        if not isinstance(name, str):
            raise SourceModelCompileError("Model(...) first argument must be a name string.")
        values["name"] = name
    for keyword in node.keywords:
        if keyword.arg is None:
            raise SourceModelCompileError("Model(...) does not support **kwargs.")
        if keyword.arg in {"functions", "equations"}:
            function_names.extend(_name_tuple(keyword.value, label=keyword.arg))
        elif keyword.arg == "returns":
            returns = _name_tuple(keyword.value, label=keyword.arg)
            duplicate = _first_duplicate(returns)
            if duplicate is not None:
                raise SourceModelCompileError(
                    f"Duplicate Model(..., returns=...) name {duplicate!r}."
                )
            values[keyword.arg] = returns
        elif keyword.arg == "metadata":
            value = _metadata_value(keyword.value)
            if not isinstance(value, dict):
                raise SourceModelCompileError("Model(..., metadata=...) must be a dictionary.")
            values[keyword.arg] = value
        elif keyword.arg == "name":
            values["name"] = _metadata_value(keyword.value)
        else:
            function_names.extend(_name_tuple(keyword.value, label=keyword.arg))
    if function_names:
        duplicate = _first_duplicate(function_names)
        if duplicate is not None:
            raise SourceModelCompileError(
                f"Duplicate Model(...) source function {duplicate!r}."
            )
        values["functions"] = tuple(function_names)
    return values


def _name_tuple(node: ast.AST, *, label: str) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Tuple | ast.List):
        out: list[str] = []
        for item in node.elts:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                out.append(item.value)
            elif isinstance(item, ast.Name):
                out.append(item.id)
            else:
                raise SourceModelCompileError(
                    f"Model(..., {label}=...) must contain names or strings."
                )
        return tuple(out)
    raise SourceModelCompileError(
        f"Model(..., {label}=...) must be a name, string, tuple, or list."
    )


def _name_reference(node: ast.AST, *, label: str) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    raise SourceModelCompileError(f"model.{label} must be a name or string.")


def _name_mapping(node: ast.AST, *, label: str) -> dict[str, str]:
    if not isinstance(node, ast.Dict):
        raise SourceModelCompileError(f"model.{label} must be a dictionary.")
    mapping: dict[str, str] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None:
            raise SourceModelCompileError(f"model.{label} cannot use ** unpacking.")
        mapping[_name_reference(key, label=label)] = _name_reference(value, label=label)
    return mapping


def _metadata_value(node: ast.AST) -> Any:
    quantity = _quantity_ast(node)
    if quantity is not None:
        magnitude, _unit = quantity
        return magnitude
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        values: dict[Any, Any] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                raise SourceModelCompileError("Metadata dictionaries cannot use ** unpacking.")
            values[_metadata_value(key)] = _metadata_value(value)
        return values
    if isinstance(node, ast.List):
        return [_metadata_value(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_metadata_value(item) for item in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _metadata_value(node.operand)
        if isinstance(value, int | float):
            return -value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        value = _metadata_value(node.operand)
        if isinstance(value, int | float):
            return value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id in {"units", "axs"} and hasattr(units, node.attr):
            return getattr(units, node.attr)
    raise SourceModelCompileError(
        f"Unsupported metadata expression {ast.dump(node, include_attributes=False)}."
    )


def _infer_model_metadata(
    source_path: Path,
    *,
    functions: tuple[ast.FunctionDef, ...],
    returns: tuple[str, ...] | None,
    name: str | None = None,
    declared_states: dict[str, dict[str, Any]] | None = None,
    dynamics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_name = name
    inputs: dict[str, dict[str, Any]] = {}
    states: dict[str, dict[str, Any]] = dict(declared_states or {})
    parameters: dict[str, dict[str, Any]] = {}
    for function in functions:
        args = _function_parameters(function)
        defaults: tuple[ast.AST | None, ...] = (None,) * (
            len(args) - len(function.args.defaults)
        ) + tuple(function.args.defaults)
        for arg, default_node in zip(args, defaults, strict=True):
            spec = _annotation_spec(arg.annotation, label=arg.arg)
            if default_node is None:
                if arg.arg in states:
                    _merge_symbol(states, arg.arg, states[arg.arg], section="state")
                elif spec["role"] == SemanticRole.GATE.value:
                    _merge_symbol(states, arg.arg, spec, section="state")
                else:
                    _merge_symbol(inputs, arg.arg, spec, section="input")
            else:
                _merge_symbol(
                    parameters,
                    arg.arg,
                    {
                        **spec,
                        "default": _metadata_default(
                            default_node,
                            expected_unit=spec["unit"],
                            label=arg.arg,
                        ),
                    },
                    section="parameter",
                    compare_default=False,
                )
    local_specs = _local_quantity_specs(functions)
    if returns is None:
        returns = _return_names(_current_output_function(functions))
    elif not returns:
        returns = _infer_return_names(local_specs)
    currents: dict[str, dict[str, Any]] = {}
    observables: dict[str, dict[str, Any]] = {}
    for name in returns:
        unit = local_specs.get(name, {}).get("unit")
        if unit is None:
            raise SourceModelCompileError(
                f"Returned expression {name!r} must have a unit annotation."
            )
        spec = {"expression": name, "unit": unit, "role": _role_value_for_unit(unit)}
        if name.startswith("I_"):
            currents[name] = {**spec, "name": _public_current_name(name)}
        else:
            observables[name] = spec
    if not currents:
        raise SourceModelCompileError(
            "Plain Python membrane source must return at least one I_* current."
        )
    step = _infer_step_metadata(
        functions,
        states,
        dynamics=dynamics,
        local_specs=local_specs,
    )
    return {
        "name": model_name or source_path.stem,
        "inputs": inputs,
        "states": states,
        "parameters": parameters,
        "currents": currents,
        "observables": observables,
        **({"step": step} if step else {}),
        **({"state_initials": dynamics["initials"]} if dynamics and "initials" in dynamics else {}),
    }


def _function_parameters(function: ast.FunctionDef) -> tuple[ast.arg, ...]:
    args = tuple(function.args.args)
    if args and args[0].arg in {"self", "cls"}:
        return args[1:]
    return args


def _current_output_function(functions: tuple[ast.FunctionDef, ...]) -> ast.FunctionDef:
    for function in functions:
        if function.name == "currents":
            return function
    return functions[-1]


def _public_current_name(expression_name: str) -> str:
    for prefix in ("I_na", "I_k", "I_ca"):
        if expression_name.startswith(prefix + "_"):
            return prefix
    return expression_name


def _infer_step_metadata(
    functions: tuple[ast.FunctionDef, ...],
    states: dict[str, dict[str, Any]],
    *,
    dynamics: dict[str, Any] | None = None,
    local_specs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if dynamics is not None and any(key != "initials" for key in dynamics):
        return _step_metadata_from_dynamics(
            dynamics,
            local_specs=local_specs or _local_quantity_specs(functions),
        )
    if not any(function.name == "step" for function in functions):
        return None
    non_gate_states = tuple(
        name
        for name, spec in states.items()
        if spec.get("role") != SemanticRole.GATE.value
    )
    updates = {
        state_name: f"{state_name}_next"
        for state_name in non_gate_states
    }
    return {
        "prepare_state_updates": updates,
        "total_outward_current": "total_outward_current",
        "explicit_outward_current": "explicit_outward_current",
        "correction_current": "correction_current",
        "linearization_gate_source": LinearizationGateSource.PREDICTOR.value,
    }


def _step_metadata_from_dynamics(
    dynamics: dict[str, Any],
    *,
    local_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    step: dict[str, Any] = {}
    if "prepare" in dynamics:
        step["prepare_state_updates"] = dynamics["prepare"]
    if "finalize" in dynamics:
        step["finalize_state_updates"] = dynamics["finalize"]
    for key in (
        "total_outward_current",
        "explicit_outward_current",
        "correction_current",
        "prepare_gate_source",
        "linearization_gate_source",
    ):
        if key in dynamics:
            step[key] = dynamics[key]
    diagnostics = dynamics.get("diagnostics")
    if diagnostics is not None:
        if not isinstance(diagnostics, dict):
            raise SourceModelCompileError("@step(..., diagnostics=...) must be a dict.")
        step["diagnostics"] = {
            diagnostic_name: _diagnostic_metadata(expression_name, local_specs=local_specs)
            for diagnostic_name, expression_name in diagnostics.items()
        }
    return step


def _diagnostic_metadata(
    expression_name: str,
    *,
    local_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    spec = local_specs.get(expression_name)
    if spec is None:
        raise SourceModelCompileError(
            f"Diagnostic expression {expression_name!r} must have a unit annotation."
        )
    return {"expression": expression_name, **spec}


def _infer_return_names(local_specs: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    currents = [
        name
        for name, spec in local_specs.items()
        if name.startswith("I_") or spec.get("unit") == units.CURRENT_DENSITY_UA_CM2
    ]
    observables = [
        name
        for name, spec in local_specs.items()
        if name.startswith("g_") and spec.get("unit") == units.CONDUCTANCE_DENSITY_MS_CM2
    ]
    if not currents:
        raise SourceModelCompileError(
            "Model(...) without returns must define at least one I_* current."
        )
    return tuple(currents + observables)


def _merge_symbol(
    target: dict[str, dict[str, Any]],
    name: str,
    spec: dict[str, Any],
    *,
    section: str,
    compare_default: bool = True,
) -> None:
    existing = target.get(name)
    if existing is None:
        target[name] = spec
        return
    keys = ("unit", "role", "default") if compare_default else ("unit", "role")
    if any(existing.get(key) != spec.get(key) for key in keys):
        raise SourceModelCompileError(
            f"Conflicting {section} declaration for {name!r}."
        )


def _annotation_spec(annotation: ast.AST | None, *, label: str) -> dict[str, str]:
    unit = _annotation_unit(annotation, label=label)
    role = (
        SemanticRole.GATE.value
        if _annotation_is_gate(annotation)
        else _role_value_for_unit(unit)
    )
    return {"unit": unit, "role": role}


def _annotation_unit(annotation: ast.AST | None, *, label: str) -> str:
    if annotation is None:
        raise SourceModelCompileError(f"{label!r} must have a unit annotation.")
    type_unit = _annotation_type_unit(annotation)
    if type_unit is not None:
        return type_unit
    if isinstance(annotation, ast.Name) and hasattr(units, annotation.id):
        value = getattr(units, annotation.id)
        label_value = _unit_label(value)
        if label_value is not None:
            return label_value
    if isinstance(annotation, ast.Attribute) and isinstance(annotation.value, ast.Name):
        if annotation.value.id in {"units", "axs"} and hasattr(units, annotation.attr):
            value = getattr(units, annotation.attr)
            label_value = _unit_label(value)
            if label_value is not None:
                return label_value
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    raise _source_error(
        annotation,
        f"Unsupported unit annotation for {label!r}: "
        f"{ast.dump(annotation, include_attributes=False)}.",
    )


def _annotation_type_unit(annotation: ast.AST | None) -> str | None:
    name: str | None = None
    if isinstance(annotation, ast.Name):
        name = annotation.id
    elif isinstance(annotation, ast.Attribute):
        name = annotation.attr
    if name is None:
        return None
    return {
        "Concentration": units.CONCENTRATION_MM,
        "ConcentrationPerCurrentDensityTime": units.CONCENTRATION_PER_CURRENT_DENSITY_TIME,
        "ConductanceDensity": units.CONDUCTANCE_DENSITY_MS_CM2,
        "CurrentDensity": units.CURRENT_DENSITY_UA_CM2,
        "Dimensionless": units.DIMENSIONLESS,
        "Gate": units.DIMENSIONLESS,
        "Length": "micrometer",
        "Rate": units.RATE_PER_MS,
        "RatePerConcentration": units.RATE_PER_MS_PER_MM,
        "RatePerVoltage": units.RATE_PER_MS_PER_MV,
        "ResistanceArea": units.RESISTANCE_AREA_OHM_CM2,
        "Temperature": units.TEMPERATURE_DEGC,
        "Time": units.TIME_MS,
        "Voltage": units.VOLTAGE_MV,
    }.get(name)


def _annotation_is_gate(annotation: ast.AST | None) -> bool:
    if isinstance(annotation, ast.Name):
        return annotation.id in {"gate", "Gate"}
    if isinstance(annotation, ast.Attribute) and isinstance(annotation.value, ast.Name):
        return annotation.attr in {"gate", "Gate"}
    return False


def _local_quantity_specs(functions: tuple[ast.FunctionDef, ...]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for function in functions:
        for statement in function.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                specs[statement.target.id] = _annotation_spec(
                    statement.annotation,
                    label=statement.target.id,
                )
    return specs


def _return_names(function: ast.FunctionDef) -> tuple[str, ...]:
    for statement in function.body:
        if not isinstance(statement, ast.Return):
            continue
        value = statement.value
        if isinstance(value, ast.Name):
            return (value.id,)
        if isinstance(value, ast.Tuple | ast.List):
            names = []
            for item in value.elts:
                if not isinstance(item, ast.Name):
                    raise SourceModelCompileError(
                        "Plain Python membrane returns must contain local names."
                    )
                names.append(item.id)
            return tuple(names)
        raise SourceModelCompileError(
            "Plain Python membrane source must return names, e.g. `return I_l, g_l`."
        )
    raise SourceModelCompileError("Plain Python membrane source must return model outputs.")


def _role_value_for_unit(unit: str) -> str:
    if unit == units.VOLTAGE_MV:
        return SemanticRole.VOLTAGE.value
    if unit == units.TIME_MS:
        return SemanticRole.TIME.value
    if unit == units.CONDUCTANCE_DENSITY_MS_CM2:
        return SemanticRole.CONDUCTANCE_DENSITY.value
    if unit == units.CURRENT_DENSITY_UA_CM2:
        return SemanticRole.CURRENT_DENSITY.value
    if unit == units.RESISTANCE_AREA_OHM_CM2:
        return SemanticRole.RESISTANCE_AREA.value
    if unit == units.TEMPERATURE_DEGC:
        return SemanticRole.TEMPERATURE.value
    if unit == units.DIMENSIONLESS:
        return SemanticRole.DIMENSIONLESS.value
    return SemanticRole.UNKNOWN.value


def _compile_assignments(
    functions: tuple[ast.FunctionDef, ...],
    metadata: dict[str, Any],
) -> dict[str, Expression]:
    symbol_units = _metadata_symbol_units(metadata)
    _validate_function_signatures(functions, symbol_units)
    records = _collect_assignment_records(functions)
    ordered_records = _topological_assignment_order(
        records,
        symbol_names=set(symbol_units),
    )
    compiler = _ExpressionCompiler(
        symbols={name: symbol(name) for name in symbol_units},
        symbol_units=symbol_units,
    )
    for record in ordered_records:
        try:
            expression = compiler.expression(record.value)
            compiler.bind(record.name, expression)
        except SourceModelCompileError as exc:
            raise _source_error(record.value, str(exc)) from exc
    assignments = compiler.locals
    exports = _exported_expression_names(metadata)
    missing = sorted(
        name for name in exports if name not in assignments and name not in symbol_units
    )
    if missing:
        raise SourceModelCompileError(
            "MODEL references expression(s) not produced by source functions: "
            + ", ".join(missing)
        )
    return assignments


def _validate_function_signatures(
    functions: tuple[ast.FunctionDef, ...],
    symbol_units: dict[str, str],
) -> None:
    for function in functions:
        if function.args.vararg is not None or function.args.kwarg is not None:
            raise _source_error(
                function,
                "Membrane equation functions cannot use *args/**kwargs.",
            )
        for arg in _function_parameters(function):
            if arg.arg not in symbol_units:
                raise _source_error(
                    arg,
                    f"Function {function.name!r} argument {arg.arg!r} is not declared "
                    "as an input, state, or parameter.",
                )


def _collect_assignment_records(functions: tuple[ast.FunctionDef, ...]) -> tuple[_AssignmentRecord, ...]:
    records: list[_AssignmentRecord] = []
    keep_calls: list[ast.Call] = []
    for function in functions:
        for statement in function.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                continue
            if isinstance(statement, ast.Expr) and _is_model_method_call(
                statement.value,
                "keep",
            ):
                if isinstance(statement.value, ast.Call):
                    _validate_keep_call(statement.value)
                    keep_calls.append(statement.value)
                continue
            target = _assignment_name(statement)
            if target is not None:
                value = _assignment_value(statement)
                if value is None:
                    raise _source_error(
                        statement,
                        "Annotated membrane assignments must define a value.",
                    )
                records.append(
                    _AssignmentRecord(
                        name=target,
                        value=value,
                        statement=statement,
                        function_name=function.name,
                    )
                )
                continue
            if isinstance(statement, ast.Return):
                _ = statement
                break
            raise _source_error(
                statement,
                f"Unsupported statement in membrane equations: {statement.__class__.__name__}.",
            )
    _validate_keep_references(keep_calls, tuple(records))
    return tuple(records)


def _assignment_name(statement: ast.stmt) -> str | None:
    if isinstance(statement, ast.Assign):
        if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            return statement.targets[0].id
        raise _source_error(statement, "Membrane equation assignments must target one local name.")
    if isinstance(statement, ast.AnnAssign):
        if isinstance(statement.target, ast.Name):
            return statement.target.id
        raise _source_error(statement, "Annotated membrane assignments must target one local name.")
    return None


def _assignment_value(statement: ast.stmt) -> ast.AST | None:
    if isinstance(statement, ast.Assign):
        return statement.value
    if isinstance(statement, ast.AnnAssign):
        return statement.value
    return None


def _topological_assignment_order(
    records: tuple[_AssignmentRecord, ...],
    *,
    symbol_names: set[str],
) -> tuple[_AssignmentRecord, ...]:
    by_name: dict[str, _AssignmentRecord] = {}
    for record in records:
        previous = by_name.get(record.name)
        if previous is not None:
            raise _source_error(
                record.statement,
                (
                    f"Duplicate equation assignment {record.name!r}; first defined "
                    f"in function {previous.function_name!r} at line "
                    f"{getattr(previous.statement, 'lineno', '?')}."
                ),
            )
        if record.name in symbol_names:
            raise _source_error(
                record.statement,
                f"Cannot assign over input/state/parameter {record.name!r}.",
            )
        by_name[record.name] = record
    local_names = set(by_name)
    local_dependencies: dict[str, tuple[str, ...]] = {}
    for record in records:
        dependencies = _expression_dependency_names(record.value)
        unknown = sorted(dependencies - local_names - symbol_names)
        if unknown:
            raise _source_error(
                record.value,
                (
                    f"Unknown symbol(s) in equation assignment {record.name!r}: "
                    + ", ".join(unknown)
                ),
            )
        local_dependencies[record.name] = tuple(
            dependency
            for dependency in dependencies
            if dependency in local_names
        )

    ordered: list[_AssignmentRecord] = []
    visiting: list[str] = []
    done: set[str] = set()

    def visit(name: str) -> None:
        if name in done:
            return
        if name in visiting:
            cycle_start = visiting.index(name)
            cycle = visiting[cycle_start:] + [name]
            raise _source_error(
                by_name[name].value,
                "Cycle detected in equation dependencies: " + " -> ".join(cycle),
            )
        visiting.append(name)
        for dependency in local_dependencies[name]:
            visit(dependency)
        visiting.pop()
        done.add(name)
        ordered.append(by_name[name])

    for record in records:
        visit(record.name)
    return tuple(ordered)


def _expression_dependency_names(node: ast.AST) -> set[str]:
    visitor = _ExpressionDependencyVisitor()
    visitor.visit(node)
    return visitor.names


class _ExpressionDependencyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and _unit_name(node) is None:
            self.names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
            self.names.add(node.attr)
            return
        if _unit_name(node) is not None:
            return
        self.generic_visit(node)


def _validate_keep_references(
    keep_calls: list[ast.Call],
    records: tuple[_AssignmentRecord, ...],
) -> None:
    local_names = {record.name for record in records}
    for keep_call in keep_calls:
        for arg in keep_call.args:
            arg_name = _keep_argument_name(arg)
            if arg_name is not None and arg_name not in local_names:
                raise _source_error(
                    arg,
                    f"keep(...) references unknown local equation {arg_name!r}.",
                )


def _validate_keep_call(node: ast.AST) -> None:
    if not isinstance(node, ast.Call):
        return
    if node.keywords:
        raise _source_error(node, "keep(...) does not support keyword arguments.")
    for arg in node.args:
        if _keep_argument_name(arg) is None:
            raise _source_error(arg, "keep(...) arguments must be local names or self.<name>.")


def _keep_argument_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"self", "cls"}
    ):
        return node.attr
    return None


def _metadata_symbol_units(metadata: dict[str, Any]) -> dict[str, str]:
    units_by_name: dict[str, str] = {}
    for section_name in ("inputs", "states", "parameters"):
        for name, spec in metadata.get(section_name, {}).items():
            if not isinstance(spec, dict):
                continue
            units_by_name[name] = _required_str(spec, "unit")
    units_by_name.update(STEP_SPECIAL_SYMBOL_UNITS)
    return units_by_name


def _exported_expression_names(metadata: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for section_name in ("currents", "observables"):
        section = metadata.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for spec in section.values():
            if not isinstance(spec, dict):
                continue
            for key in ("expression", "conductance", "reversal"):
                value = spec.get(key)
                if isinstance(value, str):
                    names.add(value)
    return names


def _build_model_ir(
    metadata: dict[str, Any],
    *,
    assignments: dict[str, Expression],
    source_path: Path,
    function_name: str,
    source_hash: str,
    parameter_defaults: dict[str, float],
) -> ModelIR:
    name = _required_str(metadata, "name")
    inputs = tuple(
        Input(input_name, _quantity(input_spec))
        for input_name, input_spec in _required_dict(metadata, "inputs").items()
    )
    states = tuple(
        _state_from_spec(state_name, state_spec)
        for state_name, state_spec in metadata.get("states", {}).items()
    )
    parameters = tuple(
        _parameter(param_name, param_spec, parameter_defaults)
        for param_name, param_spec in _required_dict(metadata, "parameters").items()
    )
    env: dict[str, Expression] = {
        input_symbol.name: symbol(input_symbol.name)
        for input_symbol in inputs
    }
    env.update({state.name: symbol(state.name) for state in states})
    env.update({parameter.name: symbol(parameter.name) for parameter in parameters})
    env.update(assignments)
    states = _apply_state_initials_from_spec(states, metadata.get("state_initials", {}), env)
    gates = tuple(
        _gate_from_state(state, env)
        for state in states
        if state.quantity.role is SemanticRole.GATE
    )
    currents = tuple(
        _current_from_spec(current_name, current_spec, env)
        for current_name, current_spec in _required_dict(metadata, "currents").items()
    )
    observables = tuple(
        Observable(
            observable_name,
            expression=_lookup_expression(observable_spec["expression"], env),
            quantity=_quantity(observable_spec),
        )
        for observable_name, observable_spec in metadata.get("observables", {}).items()
    )
    step_program = _step_program_from_spec(metadata.get("step"), env)
    model = ModelIR(
        name=name,
        inputs=inputs,
        parameters=parameters,
        states=states,
        gates=gates,
        currents=currents,
        observables=observables,
        step_program=step_program,
        metadata={
            "source": f"{source_path}:{function_name}",
            "source_compiler": SOURCE_COMPILER_VERSION,
            "source_contract": SOURCE_CONTRACT_VERSION,
            "source_hash": source_hash,
            "source_function": function_name,
            "source_outputs": {
                "all": _exported_return_names(metadata),
                "currents": tuple(
                    spec["expression"]
                    for spec in _required_dict(metadata, "currents").values()
                ),
                "observables": tuple(
                    spec["expression"]
                    for spec in metadata.get("observables", {}).values()
                ),
            },
            "source_path": str(source_path),
            "source_provenance": {
                "compiler": SOURCE_COMPILER_VERSION,
                "contract": SOURCE_CONTRACT_VERSION,
                "function_names": tuple(function_name.split(",")),
                "intrinsics": DEFAULT_INTRINSICS.names(),
                "path": str(source_path),
                "schema": MODEL_IR_SCHEMA_VERSION,
                "source_hash": source_hash,
            },
            **dict(metadata.get("metadata", {})),
        },
    )
    return assert_valid_model_ir(model)


def _with_codegen_cache_metadata(model: ModelIR, cache: GeneratedCodeCache) -> ModelIR:
    """Attach stable generated-code cache identity without hit/miss state."""

    metadata = dict(model.metadata)
    metadata["codegen_cache"] = {
        "compiler": SOURCE_COMPILER_VERSION,
        "contract": SOURCE_CONTRACT_VERSION,
        "files": tuple(path.name for path in cache.generated_files),
        "key": cache.key,
        "manifest": cache.manifest_path.name,
        "targets": ("jax", "numpy"),
    }
    return assert_valid_model_ir(replace(model, metadata=metadata))


def _parameter(
    name: str,
    spec: dict[str, Any],
    parameter_defaults: dict[str, float],
) -> Parameter:
    default = parameter_defaults.get(name, spec.get("default"))
    if default is None:
        raise SourceModelCompileError(f"Parameter {name!r} must define a default.")
    return Parameter(
        name,
        _quantity(spec),
        variability=Variability.DYNAMIC,
        default=float(default),
    )


def _gate_from_state(state: State, env: dict[str, Expression]) -> Gate:
    q10 = env.get(f"q10_{state.name}", env.get("q10"))
    return Gate(
        state.name,
        state=state.name,
        alpha=_lookup_expression(f"alpha_{state.name}", env),
        beta=_lookup_expression(f"beta_{state.name}", env),
        update=GateUpdateKind.RUSH_LARSEN,
        q10=q10,
    )


def _current_from_spec(
    current_name: str,
    current_spec: dict[str, Any],
    env: dict[str, Expression],
) -> Current:
    current = _lookup_expression(current_spec["expression"], env)
    if "conductance" in current_spec and "reversal" in current_spec:
        conductance = _lookup_expression(current_spec["conductance"], env)
        reversal = _lookup_expression(current_spec["reversal"], env)
    else:
        conductance, reversal = _infer_linear_current_terms(current)
    return Current(
        str(current_spec.get("name", current_name)),
        current=current,
        conductance=conductance,
        reversal=reversal,
        quantity=_quantity(current_spec),
    )


def _state_from_spec(state_name: str, state_spec: dict[str, Any]) -> State:
    initial = state_spec.get("initial")
    initial_expression = None
    if initial is not None:
        initial_expression = literal(float(initial), unit=_required_str(state_spec, "unit"))
    return State(state_name, _quantity(state_spec), initial=initial_expression)


def _apply_state_initials_from_spec(
    states: tuple[State, ...],
    initial_spec: Any,
    env: dict[str, Expression],
) -> tuple[State, ...]:
    if initial_spec in (None, {}):
        return states
    if not isinstance(initial_spec, dict):
        raise SourceModelCompileError("MODEL['state_initials'] must be a dictionary.")
    known_states = {state.name for state in states}
    unknown = sorted(name for name in initial_spec if name not in known_states)
    if unknown:
        raise SourceModelCompileError(
            "MODEL['state_initials'] references unknown state(s): " + ", ".join(unknown)
        )
    return tuple(
        State(
            state.name,
            state.quantity,
            initial=(
                _lookup_expression(initial_spec[state.name], env)
                if state.name in initial_spec
                else state.initial
            ),
        )
        for state in states
    )


def _step_program_from_spec(spec: Any, env: dict[str, Expression]) -> StepProgram | None:
    if not isinstance(spec, dict):
        return None
    updates = _state_updates_from_spec(spec.get("prepare_state_updates", {}), env, phase="prepare")
    finalize_updates = _state_updates_from_spec(
        spec.get("finalize_state_updates", {}),
        env,
        phase="finalize",
    )
    linearization = _linearization_gate_source(spec.get("linearization_gate_source"))
    prepare_gate_source = _linearization_gate_source(spec.get("prepare_gate_source"))
    diagnostics = _diagnostics_from_spec(spec.get("diagnostics", {}), env)
    return StepProgram(
        prepare_state_updates=updates,
        finalize_state_updates=finalize_updates,
        total_outward_current=_optional_expression(spec.get("total_outward_current"), env),
        explicit_outward_current=_optional_expression(spec.get("explicit_outward_current"), env),
        correction_current=_optional_expression(spec.get("correction_current"), env),
        prepare_gate_source=prepare_gate_source,
        linearization_gate_source=linearization,
        diagnostics=diagnostics,
    )


def _state_updates_from_spec(
    updates_spec: Any,
    env: dict[str, Expression],
    *,
    phase: str,
) -> tuple[StateUpdate, ...]:
    if not isinstance(updates_spec, dict):
        raise SourceModelCompileError(
            f"MODEL['step']['{phase}_state_updates'] must be a dictionary."
        )
    return tuple(
        StateUpdate(state_name, _lookup_expression(expression_name, env))
        for state_name, expression_name in updates_spec.items()
    )


def _diagnostics_from_spec(
    diagnostics_spec: Any,
    env: dict[str, Expression],
) -> tuple[Diagnostic, ...]:
    if not isinstance(diagnostics_spec, dict):
        raise SourceModelCompileError("MODEL['step']['diagnostics'] must be a dictionary.")
    diagnostics: list[Diagnostic] = []
    for name, diagnostic_spec in diagnostics_spec.items():
        if not isinstance(diagnostic_spec, dict):
            raise SourceModelCompileError("Diagnostic specs must be dictionaries.")
        diagnostics.append(
            Diagnostic(
                name,
                _lookup_expression(_required_str(diagnostic_spec, "expression"), env),
                _quantity(diagnostic_spec),
            )
        )
    return tuple(diagnostics)


def _optional_expression(name: Any, env: dict[str, Expression]) -> Expression | None:
    if name is None:
        return None
    if not isinstance(name, str):
        raise SourceModelCompileError("Step expression references must be names.")
    return _lookup_expression(name, env)


def _linearization_gate_source(value: Any) -> LinearizationGateSource:
    if value is None:
        return LinearizationGateSource.PREDICTOR
    if isinstance(value, LinearizationGateSource):
        return value
    try:
        return LinearizationGateSource(str(value))
    except ValueError as exc:
        raise SourceModelCompileError(
            f"Unknown linearization gate source {value!r}."
        ) from exc


def _infer_linear_current_terms(current: Expression) -> tuple[Expression, Expression]:
    if isinstance(current, BinaryOp) and current.op == "mul":
        for conductance, voltage_term in (
            (current.left, current.right),
            (current.right, current.left),
        ):
            if (
                isinstance(voltage_term, BinaryOp)
                and voltage_term.op == "sub"
                and isinstance(voltage_term.left, Symbol)
                and voltage_term.left.name == "Vm"
            ):
                return conductance, voltage_term.right
    raise SourceModelCompileError(
        "Cannot infer conductance/reversal from current expression. "
        "Use the linear form `I_x = g_x * (Vm - E_x)`."
    )


def _quantity(spec: dict[str, Any]) -> QuantitySpec:
    return QuantitySpec(
        unit=_required_str(spec, "unit"),
        role=_role(spec.get("role")),
    )


def _role(value: Any) -> SemanticRole:
    if value is None:
        return SemanticRole.UNKNOWN
    if isinstance(value, SemanticRole):
        return value
    try:
        return SemanticRole(str(value))
    except ValueError as exc:
        raise SourceModelCompileError(f"Unknown semantic role {value!r}.") from exc


def _lookup_expression(name: str, env: dict[str, Expression]) -> Expression:
    try:
        return env[name]
    except KeyError as exc:
        raise SourceModelCompileError(f"Unknown equation expression {name!r}.") from exc


def _required_dict(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise SourceModelCompileError(f"MODEL[{key!r}] must be a dictionary.")
    return value


def _required_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise SourceModelCompileError(f"MODEL field {key!r} must be a non-empty string.")
    return value


class _ExpressionCompiler:
    def __init__(
        self,
        *,
        symbols: dict[str, Expression],
        symbol_units: dict[str, str],
    ) -> None:
        self.symbols = dict(symbols)
        self.symbol_units = dict(symbol_units)
        self.locals: dict[str, Expression] = {}

    def bind(self, name: str, expression: Expression) -> None:
        if name in self.symbols:
            raise SourceModelCompileError(f"Cannot assign over input/parameter {name!r}.")
        self.locals[name] = expression

    def expression(self, node: ast.AST) -> Expression:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool | int | float):
                return literal(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.locals:
                return self.locals[node.id]
            if node.id in self.symbols:
                return self.symbols[node.id]
            unit = _unit_name(node)
            if unit is not None:
                return literal(1.0, unit=unit)
            raise SourceModelCompileError(f"Unknown symbol {node.id!r}.")
        if isinstance(node, ast.Attribute):
            self_symbol = self._self_symbol(node)
            if self_symbol is not None:
                return self_symbol
            unit = _unit_name(node)
            if unit is not None:
                return literal(1.0, unit=unit)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.expression(node.operand)
        if isinstance(node, ast.BinOp):
            return self._binary(node)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.IfExp):
            return call(
                "where",
                self.expression(node.test),
                self.expression(node.body),
                self.expression(node.orelse),
            )
        if isinstance(node, ast.Call):
            return self._call(node)
        raise SourceModelCompileError(
            f"Unsupported expression {ast.dump(node, include_attributes=False)}."
        )

    def _self_symbol(self, node: ast.Attribute) -> Expression | None:
        if not isinstance(node.value, ast.Name) or node.value.id not in {"self", "cls"}:
            return None
        if node.attr in self.locals:
            return self.locals[node.attr]
        if node.attr in self.symbols:
            return self.symbols[node.attr]
        raise SourceModelCompileError(f"Unknown model attribute {node.attr!r}.")

    def _binary(self, node: ast.BinOp) -> Expression:
        unit_literal = self._unit_literal(node)
        if unit_literal is not None:
            return unit_literal
        scaled = self._scaled_symbol_operation(node)
        if scaled is not None:
            return scaled
        left = self.expression(node.left)
        right = self.expression(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise SourceModelCompileError(f"Unsupported binary operator {node.op.__class__.__name__}.")

    def _unit_literal(self, node: ast.BinOp) -> Expression | None:
        quantity = _quantity_ast(node)
        if quantity is not None:
            value, unit = quantity
            return literal(value, unit=unit)
        return None

    def _scaled_symbol_operation(self, node: ast.BinOp) -> Expression | None:
        if isinstance(node.op, ast.Div):
            left_number = _number_literal(node.left)
            right_name = _symbol_reference_name(node.right)
            if (
                left_number is not None
                and right_name is not None
                and self.symbol_units.get(right_name) == units.RESISTANCE_AREA_OHM_CM2
            ):
                return literal(left_number * 1000.0) / self.expression(node.right)
        return None

    def _compare(self, node: ast.Compare) -> Expression:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise SourceModelCompileError("Chained comparisons are not supported in model source.")
        left = self.expression(node.left)
        right = self.expression(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        raise SourceModelCompileError(f"Unsupported comparison operator {op.__class__.__name__}.")

    def _call(self, node: ast.Call) -> Expression:
        if not isinstance(node.func, ast.Name):
            raise SourceModelCompileError("Only direct helper calls are supported in equations.")
        name = node.func.id
        if name not in DEFAULT_INTRINSICS:
            raise SourceModelCompileError(f"Unsupported equation helper {name!r}.")
        if node.keywords:
            raise SourceModelCompileError("Equation helper calls cannot use keyword arguments.")
        return call(name, *(self.expression(arg) for arg in node.args))


def _source_hash(tree: ast.Module, *, metadata: dict[str, Any]) -> str:
    payload = {
        "ast": ast.dump(tree, include_attributes=False),
        "compiler": SOURCE_COMPILER_VERSION,
        "contract": SOURCE_CONTRACT_VERSION,
        "intrinsics": DEFAULT_INTRINSICS.names(),
        "metadata": metadata,
        "schema": MODEL_IR_SCHEMA_VERSION,
        "units": {
            "conductance": units.CONDUCTANCE_DENSITY_MS_CM2,
            "current": units.CURRENT_DENSITY_UA_CM2,
            "resistance_area": units.RESISTANCE_AREA_OHM_CM2,
            "voltage": units.VOLTAGE_MV,
        },
    }
    return _hash_json(payload)


def _try_load_compiled_source_cache(
    *,
    source_path: Path,
    source_text_hash: str,
    function_name: str,
    model_class_name: str | None,
    parameter_defaults: dict[str, float],
    cache_root: str | os.PathLike[str] | None,
    load_generated_modules: tuple[str, ...],
) -> SourceModelCompileResult | None:
    root = _cache_root(cache_root)
    index_path = _source_cache_index_path(
        root,
        source_path=source_path,
        source_text_hash=source_text_hash,
        function_name=function_name,
        model_class_name=model_class_name,
    )
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not _source_cache_index_matches(
        index,
        source_path=source_path,
        source_text_hash=source_text_hash,
        function_name=function_name,
        model_class_name=model_class_name,
    ):
        return None
    cache_key = index.get("cache_key")
    if not isinstance(cache_key, str) or not cache_key:
        return None
    directory = root / cache_key
    generated_files = _generated_cache_files(directory)
    manifest_path = directory / "manifest.json"
    cache_hit, cache_reason = _cache_manifest_status(
        manifest_path,
        key=cache_key,
        generated_files=generated_files,
    )
    if not cache_hit:
        return None
    if not _manifest_source_text_hash_matches(
        manifest_path,
        source_text_hash=source_text_hash,
    ):
        return None
    model = _load_cached_model_ir(directory)
    model = _with_parameter_defaults(model, parameter_defaults)
    cache = GeneratedCodeCache(
        key=cache_key,
        directory=directory,
        manifest_path=manifest_path,
        cache_hit=True,
        cache_reason=cache_reason,
        generated_files=generated_files,
        loaded_modules=_load_generated_modules(
            generated_files,
            key=cache_key,
            targets=load_generated_modules,
        ),
    )
    model = _with_codegen_cache_metadata(model, cache)
    source_hash = str(index.get("source_hash") or model.metadata.get("source_hash") or "")
    compiled_function_name = str(
        index.get("compiled_function_name")
        or model.metadata.get("source_function")
        or function_name
    )
    return SourceModelCompileResult(
        model=model,
        source_hash=source_hash,
        source_path=source_path,
        function_name=compiled_function_name,
        cache=cache,
    )


def _ensure_generated_cache(
    *,
    model: ModelIR,
    source_path: Path,
    source_text: str,
    source_text_hash: str,
    functions: tuple[ast.FunctionDef, ...],
    metadata: dict[str, Any],
    assignments: dict[str, Expression],
    source_hash: str,
    cache_root: str | os.PathLike[str] | None,
    load_generated_modules: tuple[str, ...],
) -> GeneratedCodeCache:
    key_payload = {
        "compiler": SOURCE_COMPILER_VERSION,
        "contract": SOURCE_CONTRACT_VERSION,
        "source_hash": source_hash,
        "structure_hash": structural_hash(
            {
                "functions": [function.name for function in functions],
                "metadata": metadata,
                "source_hash": source_hash,
            }
        ),
        "targets": ("jax", "numpy"),
    }
    key = _hash_json(key_payload)
    root = _cache_root(cache_root)
    directory = root / key
    manifest_path = directory / "manifest.json"
    generated_files = _generated_cache_files(directory)
    cache_hit, cache_reason = _cache_manifest_status(
        manifest_path,
        key=key,
        generated_files=generated_files,
    )
    if cache_hit:
        return GeneratedCodeCache(
            key=key,
            directory=directory,
            manifest_path=manifest_path,
            cache_hit=True,
            cache_reason=cache_reason,
            generated_files=generated_files,
            loaded_modules=_load_generated_modules(
                generated_files,
                key=key,
                targets=load_generated_modules,
            ),
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source_snapshot.py").write_text(source_text, encoding="utf-8")
    graph_json = canonical_json(model, include_dynamic_values=True)
    (directory / "graph.json").write_text(graph_json + "\n", encoding="utf-8")
    (directory / "optimized_graph.json").write_text(
        graph_json + "\n",
        encoding="utf-8",
    )
    (directory / "jax_model.py").write_text(
        _generated_module_source(
            metadata,
            assignments=assignments,
            target="jax",
            key=key,
            source_hash=source_hash,
        ),
        encoding="utf-8",
    )
    (directory / "numpy_model.py").write_text(
        _generated_module_source(
            metadata,
            assignments=assignments,
            target="numpy",
            key=key,
            source_hash=source_hash,
        ),
        encoding="utf-8",
    )
    manifest = {
        "cache_key": key,
        "compiler": SOURCE_COMPILER_VERSION,
        "contract": SOURCE_CONTRACT_VERSION,
        "files": [path.name for path in generated_files],
        "source_hash": source_hash,
        "source_path": str(source_path),
        "source_text_hash": source_text_hash,
        "targets": ["jax", "numpy"],
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return GeneratedCodeCache(
        key=key,
        directory=directory,
        manifest_path=manifest_path,
        cache_hit=False,
        cache_reason=cache_reason,
        generated_files=generated_files,
        loaded_modules=_load_generated_modules(
            generated_files,
            key=key,
            targets=load_generated_modules,
        ),
    )


def _cache_root(cache_root: str | os.PathLike[str] | None) -> Path:
    if cache_root is not None:
        return Path(cache_root).resolve()
    env_root = os.environ.get("AXONSCOPE_MODEL_CODEGEN_CACHE")
    if env_root:
        return Path(env_root).resolve()
    return (Path.cwd() / ".axonscope_cache" / "model_codegen").resolve()


def _generated_cache_files(directory: Path) -> tuple[Path, ...]:
    return (
        directory / "source_snapshot.py",
        directory / "graph.json",
        directory / "optimized_graph.json",
        directory / "jax_model.py",
        directory / "numpy_model.py",
    )


def _source_text_hash(source_text: str) -> str:
    payload = {
        "compiler": SOURCE_COMPILER_VERSION,
        "contract": SOURCE_CONTRACT_VERSION,
        "schema": MODEL_IR_SCHEMA_VERSION,
        "source_text": source_text,
    }
    return _hash_json(payload)


def _source_cache_index_path(
    root: Path,
    *,
    source_path: Path,
    source_text_hash: str,
    function_name: str,
    model_class_name: str | None,
) -> Path:
    key = _hash_json(
        {
            "compiler": SOURCE_COMPILER_VERSION,
            "contract": SOURCE_CONTRACT_VERSION,
            "function_name": function_name,
            "index": SOURCE_CACHE_INDEX_VERSION,
            "intrinsics": DEFAULT_INTRINSICS.names(),
            "model_class_name": model_class_name,
            "path": str(source_path),
            "schema": MODEL_IR_SCHEMA_VERSION,
            "source_text_hash": source_text_hash,
        }
    )
    return root / "_source_index" / f"{key}.json"


def _write_source_cache_index(
    *,
    source_path: Path,
    source_text_hash: str,
    requested_function_name: str,
    compiled_function_name: str,
    model_class_name: str | None,
    cache: GeneratedCodeCache,
    source_hash: str,
    cache_root: str | os.PathLike[str] | None,
) -> None:
    root = _cache_root(cache_root)
    index_path = _source_cache_index_path(
        root,
        source_path=source_path,
        source_text_hash=source_text_hash,
        function_name=requested_function_name,
        model_class_name=model_class_name,
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = {
        "cache_key": cache.key,
        "compiler": SOURCE_COMPILER_VERSION,
        "compiled_function_name": compiled_function_name,
        "contract": SOURCE_CONTRACT_VERSION,
        "index": SOURCE_CACHE_INDEX_VERSION,
        "model_class_name": model_class_name,
        "requested_function_name": requested_function_name,
        "schema": MODEL_IR_SCHEMA_VERSION,
        "source_hash": source_hash,
        "source_path": str(source_path),
        "source_text_hash": source_text_hash,
    }
    index_path.write_text(
        json.dumps(index, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_cache_index_matches(
    index: Any,
    *,
    source_path: Path,
    source_text_hash: str,
    function_name: str,
    model_class_name: str | None,
) -> bool:
    if not isinstance(index, dict):
        return False
    return (
        index.get("compiler") == SOURCE_COMPILER_VERSION
        and index.get("contract") == SOURCE_CONTRACT_VERSION
        and index.get("index") == SOURCE_CACHE_INDEX_VERSION
        and index.get("model_class_name") == model_class_name
        and index.get("requested_function_name") == function_name
        and index.get("schema") == MODEL_IR_SCHEMA_VERSION
        and index.get("source_path") == str(source_path)
        and index.get("source_text_hash") == source_text_hash
    )


def _manifest_source_text_hash_matches(
    manifest_path: Path,
    *,
    source_text_hash: str,
) -> bool:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("source_text_hash") == source_text_hash


def _load_cached_model_ir(directory: Path) -> ModelIR:
    graph_path = directory / "optimized_graph.json"
    if not graph_path.is_file():
        graph_path = directory / "graph.json"
    model = model_ir_from_json(graph_path.read_text(encoding="utf-8"))
    return assert_valid_model_ir(model)


def _with_parameter_defaults(
    model: ModelIR,
    parameter_defaults: dict[str, float],
) -> ModelIR:
    if not parameter_defaults:
        return model
    overrides = {str(name): float(value) for name, value in parameter_defaults.items()}
    parameters = tuple(
        Parameter(
            parameter.name,
            parameter.quantity,
            variability=parameter.variability,
            default=overrides.get(parameter.name, parameter.default),
        )
        for parameter in model.parameters
    )
    return assert_valid_model_ir(replace(model, parameters=parameters))


def _load_generated_modules(
    generated_files: tuple[Path, ...],
    *,
    key: str,
    targets: tuple[str, ...],
) -> dict[str, Any]:
    if not targets:
        return {}
    by_name = {path.name: path for path in generated_files}
    loaded: dict[str, Any] = {}
    for target in dict.fromkeys(targets):
        file_name = _generated_module_file_name(target)
        path = by_name[file_name]
        module_name = f"axonscope_model_codegen_{key}_{target}"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load generated module {path}.")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        if getattr(module, "CACHE_KEY", None) != key:
            raise ImportError(f"Generated module {path} has an unexpected cache key.")
        if getattr(module, "TARGET", None) != target:
            raise ImportError(f"Generated module {path} has an unexpected target.")
        loaded[str(target)] = module
    return loaded


def _generated_module_file_name(target: str) -> str:
    if target == "jax":
        return "jax_model.py"
    if target == "numpy":
        return "numpy_model.py"
    raise ValueError(f"Unknown generated module target {target!r}.")


def _cache_manifest_status(
    manifest_path: Path,
    *,
    key: str,
    generated_files: tuple[Path, ...],
) -> tuple[bool, str]:
    if not manifest_path.is_file():
        return False, "manifest_missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "manifest_invalid_json"
    if manifest.get("cache_key") != key:
        return False, "cache_key_mismatch"
    missing_files = [path.name for path in generated_files if not path.is_file()]
    if missing_files:
        return False, "generated_files_missing:" + ",".join(missing_files)
    return True, "manifest_match"


def _generated_module_source(
    metadata: dict[str, Any],
    *,
    assignments: dict[str, Expression],
    target: str,
    key: str,
    source_hash: str,
) -> str:
    if target == "jax":
        import_line = "import jax.numpy as xp"
    elif target == "numpy":
        import_line = "import numpy as xp"
    else:
        raise ValueError(f"Unknown codegen target {target!r}.")
    output_names = _exported_return_names(metadata)
    selected_assignments = _required_output_assignments(
        assignments,
        output_names=output_names,
    )
    arg_names = _model_step_arg_names(
        metadata,
        assignments=selected_assignments,
        output_names=output_names,
    )
    args = ", ".join(arg_names)
    body_lines = [
        f"{name} = {_expression_source(expression)}"
        for name, expression in selected_assignments.items()
    ]
    body_lines.append("return " + ", ".join(output_names))
    body = "\n".join("    " + line for line in body_lines)
    return (
        "# Generated by AxonScope. Do not edit by hand.\n"
        f"ARG_NAMES = {arg_names!r}\n"
        f"CACHE_KEY = {key!r}\n"
        f"OUTPUT_NAMES = {output_names!r}\n"
        f"SOURCE_HASH = {source_hash!r}\n"
        f"TARGET = {target!r}\n"
        f"{import_line}\n\n"
        "exp = xp.exp\n"
        "expm1 = xp.expm1\n"
        "log = xp.log\n"
        "log1p = xp.log1p\n"
        "sqrt = xp.sqrt\n"
        "abs = xp.abs\n"
        "minimum = xp.minimum\n"
        "maximum = xp.maximum\n"
        "clip = xp.clip\n"
        "where = xp.where\n"
        "tanh = xp.tanh\n\n"
        "dimensionless = 1.0\n"
        "mV = 1.0\n"
        "ms = 1.0\n"
        "mS_per_cm2 = 1.0\n"
        "uA_per_cm2 = 1.0\n"
        "ohm_cm2 = 1.0\n"
        "degC = 1.0\n"
        "per_ms = 1.0\n"
        "per_ms_per_mV = 1.0\n"
        "per_ms_per_mM = 1.0\n"
        "mM = 1.0\n"
        "mM_per_uA_cm2_ms = 1.0\n"
        "gate = 1.0\n\n"
        "def sigmoid(x):\n"
        "    return 1.0 / (1.0 + xp.exp(-x))\n\n"
        "def boltzmann(x, midpoint, slope):\n"
        "    return 1.0 / (1.0 + xp.exp((x - midpoint) / slope))\n\n"
        "def q10(base, celsius, reference):\n"
        "    return xp.power(base, (celsius - reference) / 10.0)\n\n"
        "def alpha_from_inf_tau(x_inf, tau):\n"
        "    return x_inf / tau\n\n"
        "def beta_from_inf_tau(x_inf, tau):\n"
        "    return (1.0 - x_inf) / tau\n\n"
        "def safe_exp(x):\n"
        "    return xp.where(x < -100.0, 0.0, xp.exp(x))\n\n"
        "def vtrap(x, y):\n"
        "    z = x / y\n"
        "    return xp.where(xp.abs(z) < 1e-6, y * (1.0 - z / 2.0), x / (xp.exp(z) - 1.0))\n\n"
        f"def model_step({args}):\n"
        f"{body}\n"
    )


def _model_step_arg_names(
    metadata: dict[str, Any],
    *,
    assignments: dict[str, Expression],
    output_names: tuple[str, ...],
) -> tuple[str, ...]:
    base = tuple(
        name
        for section in ("inputs", "states", "parameters")
        for name in metadata.get(section, {})
    )
    used: set[str] = set()
    for expression in assignments.values():
        used.update(_expression_symbols(expression))
    for output_name in output_names:
        if output_name not in assignments:
            used.add(output_name)
    return (
        tuple(name for name in base if name in used)
        + tuple(name for name in STEP_SPECIAL_SYMBOL_UNITS if name in used)
    )


def _required_output_assignments(
    assignments: dict[str, Expression],
    *,
    output_names: tuple[str, ...],
) -> dict[str, Expression]:
    needed: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in needed or name not in assignments:
            return
        if name in visiting:
            raise SourceModelCompileError(
                f"Cycle detected while generating output {name!r}."
            )
        visiting.add(name)
        for symbol_name in _expression_symbols(assignments[name]):
            visit(symbol_name)
        visiting.remove(name)
        needed.add(name)

    for output_name in output_names:
        visit(output_name)
    return {
        name: expression
        for name, expression in assignments.items()
        if name in needed
    }


def _expression_symbols(expression: Expression) -> set[str]:
    if isinstance(expression, Symbol):
        return {expression.name}
    if isinstance(expression, Literal):
        return set()
    if isinstance(expression, UnaryOp):
        return _expression_symbols(expression.operand)
    if isinstance(expression, BinaryOp):
        return _expression_symbols(expression.left) | _expression_symbols(expression.right)
    if isinstance(expression, Call):
        symbols: set[str] = set()
        for arg in expression.args:
            symbols.update(_expression_symbols(arg))
        return symbols
    return set()


def _exported_return_names(metadata: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        spec["expression"]
        for section in ("currents", "observables")
        for spec in metadata.get(section, {}).values()
    )


def _expression_source(expression: Expression) -> str:
    if isinstance(expression, Symbol):
        return expression.name
    if isinstance(expression, BinaryOp):
        left = _expression_source(expression.left)
        right = _expression_source(expression.right)
        op = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "pow": "**",
            "lt": "<",
            "le": "<=",
            "gt": ">",
            "ge": ">=",
        }[expression.op]
        return f"({left} {op} {right})"
    if isinstance(expression, Literal):
        return repr(expression.value)
    if isinstance(expression, UnaryOp):
        return f"(-{_expression_source(expression.operand)})"
    if isinstance(expression, Call):
        args = ", ".join(_expression_source(arg) for arg in expression.args)
        return f"{expression.intrinsic}({args})"
    raise SourceModelCompileError(
        f"Cannot generate source for expression {expression.__class__.__name__}."
    )


def _hash_json(payload: Any) -> str:
    raw = canonical_json(payload, include_dynamic_values=True).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=20).hexdigest()


def _metadata_default(node: ast.AST, *, expected_unit: str, label: str) -> float:
    quantity = _quantity_ast(node)
    if quantity is not None:
        value, unit = quantity
        if unit == units.DIMENSIONLESS and expected_unit != units.DIMENSIONLESS:
            raise _source_error(
                node,
                f"Default for {label!r} must specify unit {expected_unit!r}.",
            )
        if unit != units.DIMENSIONLESS and unit != expected_unit:
            raise _source_error(
                node,
                f"Default for {label!r} has unit {unit!r}, expected {expected_unit!r}.",
            )
        return float(value)
    value = _metadata_value(node)
    if not isinstance(value, int | float):
        raise _source_error(node, f"Default for {label!r} must be numeric.")
    return float(value)


def _quantity_ast(node: ast.AST) -> tuple[float, str] | None:
    number = _number_literal(node)
    if number is not None:
        return number, units.DIMENSIONLESS
    unit = _unit_name(node)
    if unit is not None:
        return 1.0, unit
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _quantity_ast(node.operand)
        if value is not None:
            magnitude, unit = value
            return -magnitude, unit
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _quantity_ast(node.operand)
    if isinstance(node, ast.BinOp):
        left = _quantity_ast(node.left)
        right = _quantity_ast(node.right)
        if left is None or right is None:
            return None
        left_value, left_unit = left
        right_value, right_unit = right
        if isinstance(node.op, ast.Mult):
            return left_value * right_value, product_unit(left_unit, right_unit)
        if isinstance(node.op, ast.Div):
            return left_value / right_value, quotient_unit(left_unit, right_unit)
    return None


def _number_literal(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _number_literal(node.operand)
        if value is not None:
            return -value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _number_literal(node.operand)
    return None


def _symbol_reference_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id in {"self", "cls"}:
            return node.attr
    return None


def _unit_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and hasattr(units, node.id):
        value = getattr(units, node.id)
        return _unit_label(value)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id in {"units", "axs"} and hasattr(units, node.attr):
            value = getattr(units, node.attr)
            return _unit_label(value)
    return None


def _unit_label(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    label = getattr(value, "label", None)
    if isinstance(label, str):
        return label
    return None


__all__ = [
    "GeneratedCodeCache",
    "SOURCE_COMPILER_VERSION",
    "SOURCE_CONTRACT_VERSION",
    "SourceModelCompileError",
    "SourceModelCompileResult",
    "compile_model_source_file",
]
