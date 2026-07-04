"""Semantic validation for backend-neutral Model IR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from axonscope.utils.units import (
    CONDUCTANCE_DENSITY_MS_CM2,
    CURRENT_DENSITY_UA_CM2,
    DIMENSIONLESS,
    RATE_PER_MS,
    TEMPERATURE_DEGC,
    TIME_MS,
    VOLTAGE_MV,
)

from .expressions import BinaryOp, Call, Expression, Literal, Symbol, UnaryOp
from .intrinsics import DEFAULT_INTRINSICS, IntrinsicRegistry
from .schema import ModelIR, ModelSymbol, QuantitySpec, SemanticRole
from .unit_algebra import (
    is_dimensionless,
    product_unit,
    quotient_unit,
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: str
    message: str


class ModelValidationError(ValueError):
    """Raised when a Model IR definition is semantically invalid."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        joined = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        super().__init__(joined)


def validate_model_ir(
    model: ModelIR,
    *,
    registry: IntrinsicRegistry = DEFAULT_INTRINSICS,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    _validate_source_metadata(model, issues)
    env = _environment(model, issues)

    for state in model.states:
        if state.initial is None:
            continue
        spec = _infer(state.initial, env, registry, issues, f"state.{state.name}.initial")
        _require_same_quantity(
            spec,
            state.quantity,
            issues,
            f"state.{state.name}.initial",
        )

    state_names = {state.name for state in model.states}
    for gate in model.gates:
        if gate.state not in state_names:
            issues.append(
                ValidationIssue(
                    f"gate.{gate.name}",
                    f"references unknown state {gate.state!r}",
                )
            )
        for label, expr in (("alpha", gate.alpha), ("beta", gate.beta)):
            spec = _infer(expr, env, registry, issues, f"gate.{gate.name}.{label}")
            if spec.unit not in {DIMENSIONLESS, RATE_PER_MS} and not is_dimensionless(spec.unit):
                issues.append(
                    ValidationIssue(
                        f"gate.{gate.name}.{label}",
                        f"rate expression must be dimensionless or {RATE_PER_MS!r}, got {spec.unit!r}",
                    )
                )
        if gate.q10 is not None:
            spec = _infer(gate.q10, env, registry, issues, f"gate.{gate.name}.q10")
            if not is_dimensionless(spec.unit):
                issues.append(
                    ValidationIssue(
                        f"gate.{gate.name}.q10",
                        f"q10 expression must be dimensionless, got {spec.unit!r}",
                    )
                )

    for current in model.currents:
        spec = _infer(current.current, env, registry, issues, f"current.{current.name}")
        if spec.unit != current.quantity.unit:
            issues.append(
                ValidationIssue(
                    f"current.{current.name}",
                    f"current expression has unit {spec.unit!r}, expected {current.quantity.unit!r}",
                )
            )
        if current.quantity.unit != CURRENT_DENSITY_UA_CM2:
            issues.append(
                ValidationIssue(
                    f"current.{current.name}",
                    "current quantity must be an outward current density",
                )
            )
        conductance_spec = _infer(
            current.conductance,
            env,
            registry,
            issues,
            f"current.{current.name}.conductance",
        )
        if conductance_spec.unit != CONDUCTANCE_DENSITY_MS_CM2:
            issues.append(
                ValidationIssue(
                    f"current.{current.name}.conductance",
                    f"conductance expression has unit {conductance_spec.unit!r}, "
                    f"expected {CONDUCTANCE_DENSITY_MS_CM2!r}",
                )
            )
        reversal_spec = _infer(
            current.reversal,
            env,
            registry,
            issues,
            f"current.{current.name}.reversal",
        )
        if reversal_spec.unit != VOLTAGE_MV:
            issues.append(
                ValidationIssue(
                    f"current.{current.name}.reversal",
                    f"reversal expression has unit {reversal_spec.unit!r}, "
                    f"expected {VOLTAGE_MV!r}",
                )
            )

    for observable in model.observables:
        spec = _infer(
            observable.expression,
            env,
            registry,
            issues,
            f"observable.{observable.name}",
        )
        _require_same_quantity(spec, observable.quantity, issues, f"observable.{observable.name}")

    if model.step_program is not None:
        _validate_step_program(model, env, registry, issues)

    return tuple(issues)


def _validate_source_metadata(model: ModelIR, issues: list[ValidationIssue]) -> None:
    metadata = model.metadata
    if not isinstance(metadata, Mapping):
        return
    _validate_source_outputs(metadata.get("source_outputs"), issues)
    _validate_source_provenance(metadata, issues)
    source_sections = _validate_source_section_metadata(
        metadata.get("source_sections"),
        path="metadata.source_sections",
        issues=issues,
    )
    source_mechanisms = _validate_source_section_metadata(
        metadata.get("source_mechanisms"),
        path="metadata.source_mechanisms",
        issues=issues,
        require_mechanism_name=False,
    )
    if source_sections is not None and source_mechanisms is not None:
        mechanism_names = {
            str(section.get("mechanism"))
            for section in source_sections
            if isinstance(section.get("mechanism"), str)
        }
        mechanism_names.update(
            str(section["name"]).split(":", 1)[1]
            for section in source_sections
            if isinstance(section.get("name"), str)
            and str(section["name"]).startswith("mechanism:")
        )
        for index, mechanism in enumerate(source_mechanisms):
            name = mechanism.get("name")
            if isinstance(name, str) and name not in mechanism_names:
                issues.append(
                    ValidationIssue(
                        f"metadata.source_mechanisms[{index}]",
                        f"references unknown source mechanism {name!r}",
                    )
                )


def _validate_source_outputs(value: Any, issues: list[ValidationIssue]) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        issues.append(
            ValidationIssue(
                "metadata.source_outputs",
                "source_outputs must be a mapping",
            )
        )
        return
    groups: dict[str, tuple[str, ...]] = {}
    for key in ("currents", "observables", "all"):
        names = _metadata_string_tuple(
            value.get(key),
            path=f"metadata.source_outputs.{key}",
            issues=issues,
        )
        if names is None:
            continue
        groups[key] = names
        duplicate = _first_duplicate_name(names)
        if duplicate is not None:
            issues.append(
                ValidationIssue(
                    f"metadata.source_outputs.{key}",
                    f"duplicate source output name {duplicate!r}",
                )
            )
    if "currents" in groups and "observables" in groups:
        overlap = sorted(set(groups["currents"]) & set(groups["observables"]))
        if overlap:
            issues.append(
                ValidationIssue(
                    "metadata.source_outputs",
                    "current and observable source outputs overlap: "
                    + ", ".join(overlap),
                )
            )
    if all(key in groups for key in ("currents", "observables", "all")):
        expected = (*groups["currents"], *groups["observables"])
        if groups["all"] != expected:
            issues.append(
                ValidationIssue(
                    "metadata.source_outputs.all",
                    "must equal currents followed by observables",
                )
            )


def _validate_source_provenance(
    metadata: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> None:
    provenance = metadata.get("source_provenance")
    source_keys = {
        "source_contract": "contract",
        "source_compiler": "compiler",
        "source_hash": "source_hash",
        "source_path": "path",
    }
    has_source_metadata = (
        any(key in metadata for key in source_keys)
        or "source_function" in metadata
    )
    if not has_source_metadata and provenance is None:
        return
    if not isinstance(provenance, Mapping):
        issues.append(
            ValidationIssue(
                "metadata.source_provenance",
                "source-backed models must carry a source_provenance mapping",
            )
        )
        return
    for metadata_key, provenance_key in source_keys.items():
        metadata_value = metadata.get(metadata_key)
        provenance_value = provenance.get(provenance_key)
        if metadata_value is None or provenance_value is None:
            continue
        if str(metadata_value) != str(provenance_value):
            issues.append(
                ValidationIssue(
                    f"metadata.source_provenance.{provenance_key}",
                    f"does not match metadata.{metadata_key}",
                )
            )
    source_function = metadata.get("source_function")
    if source_function is not None:
        expected = tuple(
            name for name in str(source_function).split(",") if name
        )
        actual = _metadata_string_tuple(
            provenance.get("function_names"),
            path="metadata.source_provenance.function_names",
            issues=issues,
        )
        if actual is not None and actual != expected:
            issues.append(
                ValidationIssue(
                    "metadata.source_provenance.function_names",
                    "does not match metadata.source_function",
                )
            )
    provenance_sections = provenance.get("sections")
    if provenance_sections is not None:
        _validate_source_section_metadata(
            provenance_sections,
            path="metadata.source_provenance.sections",
            issues=issues,
        )


def _validate_source_section_metadata(
    value: Any,
    *,
    path: str,
    issues: list[ValidationIssue],
    require_mechanism_name: bool = True,
) -> tuple[Mapping[str, Any], ...] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        issues.append(ValidationIssue(path, "must be a sequence of mappings"))
        return None
    sections: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, Mapping):
            issues.append(ValidationIssue(item_path, "must be a mapping"))
            continue
        sections.append(item)
        for key in ("name", "function"):
            if not isinstance(item.get(key), str) or not item.get(key):
                issues.append(ValidationIssue(f"{item_path}.{key}", "must be a string"))
        for key in ("assignments", "depends_on"):
            _metadata_string_tuple(
                item.get(key),
                path=f"{item_path}.{key}",
                issues=issues,
            )
        if (
            require_mechanism_name
            and "mechanism" in item
            and not isinstance(item.get("mechanism"), str)
        ):
            issues.append(ValidationIssue(f"{item_path}.mechanism", "must be a string"))
    return tuple(sections)


def _metadata_string_tuple(
    value: Any,
    *,
    path: str,
    issues: list[ValidationIssue],
) -> tuple[str, ...] | None:
    if isinstance(value, str) or not isinstance(value, Sequence):
        issues.append(ValidationIssue(path, "must be a sequence of strings"))
        return None
    names: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            issues.append(ValidationIssue(f"{path}[{index}]", "must be a string"))
            continue
        names.append(item)
    return tuple(names)


def _first_duplicate_name(values: tuple[str, ...]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def assert_valid_model_ir(
    model: ModelIR,
    *,
    registry: IntrinsicRegistry = DEFAULT_INTRINSICS,
) -> ModelIR:
    issues = validate_model_ir(model, registry=registry)
    if issues:
        raise ModelValidationError(issues)
    return model


def _environment(model: ModelIR, issues: list[ValidationIssue]) -> dict[str, QuantitySpec]:
    env: dict[str, QuantitySpec] = {}
    symbols: tuple[ModelSymbol, ...] = (*model.inputs, *model.parameters, *model.states)
    for sym in symbols:
        if sym.name in env:
            issues.append(
                ValidationIssue("symbols", f"duplicate symbol name {sym.name!r}")
            )
        env[sym.name] = sym.quantity
    for current in model.currents:
        if current.name in env:
            issues.append(
                ValidationIssue("currents", f"current name shadows symbol {current.name!r}")
            )
    return env


def _validate_step_program(
    model: ModelIR,
    env: dict[str, QuantitySpec],
    registry: IntrinsicRegistry,
    issues: list[ValidationIssue],
) -> None:
    step = model.step_program
    if step is None:
        return
    step_env = _step_environment(model, env)
    states = {state.name: state for state in model.states}
    gate_state_names = {gate.state for gate in model.gates}

    for phase, updates in (
        ("prepare", step.prepare_state_updates),
        ("finalize", step.finalize_state_updates),
    ):
        for index, update in enumerate(updates):
            state = states.get(update.state)
            path = f"step.{phase}.state_update[{index}]"
            if state is None:
                issues.append(
                    ValidationIssue(path, f"references unknown state {update.state!r}")
                )
                continue
            if update.state in gate_state_names:
                issues.append(
                    ValidationIssue(path, f"cannot update gate state {update.state!r}")
                )
            spec = _infer(update.expression, step_env, registry, issues, path)
            _require_same_quantity(spec, state.quantity, issues, path)

    for name, expression in (
        ("total_outward_current", step.total_outward_current),
        ("explicit_outward_current", step.explicit_outward_current),
        ("correction_current", step.correction_current),
    ):
        if expression is None:
            continue
        spec = _infer(expression, step_env, registry, issues, f"step.{name}")
        if spec.unit != CURRENT_DENSITY_UA_CM2:
            issues.append(
                ValidationIssue(
                    f"step.{name}",
                    f"expression has unit {spec.unit!r}, expected {CURRENT_DENSITY_UA_CM2!r}",
                )
            )

    for diagnostic in step.diagnostics:
        spec = _infer(
            diagnostic.expression,
            step_env,
            registry,
            issues,
            f"step.diagnostic.{diagnostic.name}",
        )
        _require_same_quantity(
            spec,
            diagnostic.quantity,
            issues,
            f"step.diagnostic.{diagnostic.name}",
        )


def _step_environment(
    model: ModelIR,
    env: dict[str, QuantitySpec],
) -> dict[str, QuantitySpec]:
    step_env = dict(env)
    step_env.update(
        {
            "dt": QuantitySpec(unit=TIME_MS, role=SemanticRole.TIME),
            "Vm_prev": QuantitySpec(unit=VOLTAGE_MV, role=SemanticRole.VOLTAGE),
            "Vm_new": QuantitySpec(unit=VOLTAGE_MV, role=SemanticRole.VOLTAGE),
            "I_ion": QuantitySpec(
                unit=CURRENT_DENSITY_UA_CM2,
                role=SemanticRole.CURRENT_DENSITY,
            ),
            "I_background": QuantitySpec(
                unit=CURRENT_DENSITY_UA_CM2,
                role=SemanticRole.CURRENT_DENSITY,
            ),
        }
    )
    for current in model.currents:
        step_env[current.name] = current.quantity
    return step_env


def _infer(
    expr: Expression,
    env: dict[str, QuantitySpec],
    registry: IntrinsicRegistry,
    issues: list[ValidationIssue],
    path: str,
) -> QuantitySpec:
    if isinstance(expr, Literal):
        return QuantitySpec(unit=expr.unit)
    if isinstance(expr, Symbol):
        spec = env.get(expr.name)
        if spec is None:
            issues.append(ValidationIssue(path, f"unknown symbol {expr.name!r}"))
            return QuantitySpec()
        return spec
    if isinstance(expr, UnaryOp):
        return _infer(expr.operand, env, registry, issues, f"{path}.{expr.op}")
    if isinstance(expr, BinaryOp):
        left = _infer(expr.left, env, registry, issues, f"{path}.left")
        right = _infer(expr.right, env, registry, issues, f"{path}.right")
        if expr.op in {"add", "sub"}:
            _require_same_quantity(left, right, issues, path)
            return left
        if expr.op == "mul":
            return QuantitySpec(
                unit=product_unit(left.unit, right.unit),
                shape=_merge_shape(left, right, issues, path),
                dtype=left.dtype,
                role=_role_for_unit(product_unit(left.unit, right.unit)),
            )
        if expr.op == "div":
            unit = quotient_unit(left.unit, right.unit)
            return QuantitySpec(
                unit=unit,
                shape=_merge_shape(left, right, issues, path),
                dtype=left.dtype,
                role=_role_for_unit(unit),
            )
        if expr.op == "pow":
            if not is_dimensionless(right.unit):
                issues.append(ValidationIssue(path, "power exponent must be dimensionless"))
            return left
        if expr.op in {"lt", "le", "gt", "ge"}:
            _require_same_quantity(left, right, issues, path)
            return QuantitySpec(
                unit=DIMENSIONLESS,
                shape=_merge_shape(left, right, issues, path),
                dtype="bool",
                role=SemanticRole.DIMENSIONLESS,
            )
    if isinstance(expr, Call):
        if expr.intrinsic not in registry:
            issues.append(ValidationIssue(path, f"unsupported intrinsic {expr.intrinsic!r}"))
            return QuantitySpec()
        intrinsic = registry.get(expr.intrinsic)
        if not intrinsic.accepts(len(expr.args)):
            issues.append(
                ValidationIssue(
                    path,
                    f"intrinsic {expr.intrinsic!r} got {len(expr.args)} arguments",
                )
            )
        args = [
            _infer(arg, env, registry, issues, f"{path}.{expr.intrinsic}[{i}]")
            for i, arg in enumerate(expr.args)
        ]
        return _infer_intrinsic(expr.intrinsic, args, issues, path)
    issues.append(ValidationIssue(path, f"unsupported expression {type(expr).__name__}"))
    return QuantitySpec()


def _infer_intrinsic(
    name: str,
    args: list[QuantitySpec],
    issues: list[ValidationIssue],
    path: str,
) -> QuantitySpec:
    if not args:
        return QuantitySpec()
    if name in {"exp", "expm1", "log", "log1p", "safe_exp", "sigmoid", "tanh"}:
        if not is_dimensionless(args[0].unit):
            issues.append(
                ValidationIssue(path, f"intrinsic {name!r} requires dimensionless input")
            )
        return QuantitySpec(unit=DIMENSIONLESS, shape=args[0].shape, dtype=args[0].dtype)
    if name == "q10":
        if not is_dimensionless(args[0].unit):
            issues.append(ValidationIssue(path, "q10 base must be dimensionless"))
        if args[1].unit != TEMPERATURE_DEGC:
            issues.append(ValidationIssue(path, "q10 celsius argument must be degC"))
        if args[2].unit != TEMPERATURE_DEGC:
            issues.append(ValidationIssue(path, "q10 reference argument must be degC"))
        return QuantitySpec(unit=DIMENSIONLESS, shape=args[1].shape, dtype=args[1].dtype)
    if name in {"alpha_from_inf_tau", "beta_from_inf_tau"}:
        if not is_dimensionless(args[0].unit):
            issues.append(ValidationIssue(path, f"{name} steady-state value must be dimensionless"))
        if args[1].unit != TIME_MS:
            issues.append(ValidationIssue(path, f"{name} tau argument must be {TIME_MS!r}"))
        return QuantitySpec(unit=RATE_PER_MS, shape=args[0].shape, dtype=args[0].dtype)
    if name in {"abs", "sqrt"}:
        return args[0]
    if name == "clip":
        _require_same_quantity(args[0], args[1], issues, path)
        _require_same_quantity(args[0], args[2], issues, path)
        return args[0]
    if name in {"minimum", "maximum", "vtrap"}:
        _require_same_quantity(args[0], args[1], issues, path)
        return args[0]
    if name == "pow":
        if not is_dimensionless(args[1].unit):
            issues.append(ValidationIssue(path, "power exponent must be dimensionless"))
        return args[0]
    if name == "where":
        if not is_dimensionless(args[0].unit):
            issues.append(ValidationIssue(path, "where condition must be dimensionless"))
        _require_same_quantity(args[1], args[2], issues, path)
        return args[1]
    if name in {"rush_larsen_gate", "cn_gate"}:
        return QuantitySpec(
            unit=DIMENSIONLESS,
            shape=args[0].shape,
            dtype=args[0].dtype,
            role=SemanticRole.GATE,
        )
    return args[0]


def _require_same_quantity(
    left: QuantitySpec,
    right: QuantitySpec,
    issues: list[ValidationIssue],
    path: str,
) -> None:
    if left.unit != right.unit:
        issues.append(
            ValidationIssue(path, f"unit mismatch {left.unit!r} vs {right.unit!r}")
        )
    if left.shape != right.shape:
        issues.append(
            ValidationIssue(path, f"shape mismatch {left.shape.dims!r} vs {right.shape.dims!r}")
        )


def _merge_shape(
    left: QuantitySpec,
    right: QuantitySpec,
    issues: list[ValidationIssue],
    path: str,
):
    if left.shape == right.shape:
        return left.shape
    if left.shape.dims == ():
        return right.shape
    if right.shape.dims == ():
        return left.shape
    issues.append(
        ValidationIssue(path, f"shape mismatch {left.shape.dims!r} vs {right.shape.dims!r}")
    )
    return left.shape


def _role_for_unit(unit: str) -> SemanticRole:
    if unit == CURRENT_DENSITY_UA_CM2:
        return SemanticRole.CURRENT_DENSITY
    if is_dimensionless(unit):
        return SemanticRole.DIMENSIONLESS
    return SemanticRole.UNKNOWN
