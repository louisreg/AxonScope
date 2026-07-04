"""User-facing explanation reports for membrane source models."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TextIO

from axonscope.membranes.generated_code import (
    GeneratedMembraneCodeInspection,
    inspect_generated_code,
)
from axonscope.membranes.model import MembraneModel, ensure_membrane_model
from axonscope.model_ir.expressions import BinaryOp, Call, Expression, Literal, Symbol, UnaryOp
from axonscope.model_ir.schema import ModelIR, ModelSymbol
from axonscope.model_ir.serialization import model_ir_from_json


@dataclass(frozen=True, slots=True)
class MembraneSourceSymbol:
    """One named input, state, or parameter declared by a membrane source model."""

    name: str
    kind: str
    role: str
    unit: str
    default: Any = None


@dataclass(frozen=True, slots=True)
class MembraneStateUpdateExplanation:
    """One source-visible state update used by a generated step program."""

    state: str
    expression: str


@dataclass(frozen=True, slots=True)
class MembraneStepExplanation:
    """Source-facing view of explicit step semantics for a membrane model."""

    state_initials: tuple[MembraneStateUpdateExplanation, ...]
    prepare_state_updates: tuple[MembraneStateUpdateExplanation, ...]
    finalize_state_updates: tuple[MembraneStateUpdateExplanation, ...]
    total_outward_current: str | None
    explicit_outward_current: str | None
    correction_current: str | None
    prepare_gate_source: str
    linearization_gate_source: str
    diagnostics: tuple[MembraneStateUpdateExplanation, ...]


@dataclass(frozen=True, slots=True)
class MembraneEquationDependency:
    """Dependencies for one named intermediate equation."""

    name: str
    depends_on: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MembraneSourceSection:
    """One source function/section in a public membrane model file."""

    name: str
    function_name: str
    docstring: str | None
    arguments: tuple[str, ...]
    assignments: tuple[str, ...]
    dependencies: tuple[MembraneEquationDependency, ...]


@dataclass(frozen=True, slots=True)
class MembraneMechanismExplanation:
    """One named membrane mechanism preserved from source sections."""

    name: str
    function_name: str
    docstring: str | None
    arguments: tuple[str, ...]
    assignments: tuple[str, ...]
    external_dependencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GeneratedTargetExplanation:
    """What one generated backend target keeps for its model_step function."""

    target: str
    path: str
    arg_names: tuple[str, ...]
    output_names: tuple[str, ...]
    retained_assignments: tuple[str, ...]
    pruned_from_model_step: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MembraneComponentExplanation:
    """One public component label in a membrane model explanation."""

    label: str
    model_kind: str


@dataclass(frozen=True, slots=True)
class MembraneRecordingOutputExplanation:
    """Public recording names produced by a membrane model."""

    gates: tuple[str, ...]
    currents: tuple[str, ...]
    conductances: tuple[str, ...]
    states: tuple[str, ...]
    observables: tuple[str, ...]
    current_aggregates: tuple[str, ...]
    conductance_aggregates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MembraneSourceExplanation:
    """Explanation for one standalone membrane source file."""

    model_name: str
    source_path: str
    source_hash: str
    function_names: tuple[str, ...]
    cache_status: str
    cache_reason: str
    cache_key: str
    cache_directory: str
    generated_targets: tuple[str, ...]
    metadata: Mapping[str, Any]
    inputs: tuple[MembraneSourceSymbol, ...]
    parameters: tuple[MembraneSourceSymbol, ...]
    states: tuple[MembraneSourceSymbol, ...]
    gates: tuple[str, ...]
    currents: tuple[str, ...]
    observables: tuple[str, ...]
    diagnostics: tuple[str, ...]
    step: MembraneStepExplanation | None
    source_outputs: Mapping[str, tuple[str, ...]]
    internal_outputs: tuple[str, ...]
    sections: tuple[MembraneSourceSection, ...]
    mechanisms: tuple[MembraneMechanismExplanation, ...]
    targets: tuple[GeneratedTargetExplanation, ...]


@dataclass(frozen=True, slots=True)
class MembraneModelExplanation:
    """Explanation report for a public membrane model descriptor."""

    model_kind: str
    components: tuple[MembraneComponentExplanation, ...]
    recording_outputs: MembraneRecordingOutputExplanation
    sources: tuple[MembraneSourceExplanation, ...]

    def format(self) -> str:
        """Return a compact plain-text explanation."""

        return format_membrane_model_explanation(self)

    def print(self, file: TextIO | None = None) -> None:
        """Print the explanation report."""

        print(self.format(), file=file)


def explain(model: MembraneModel) -> MembraneModelExplanation:
    """Explain how a public membrane model source compiles and generates code.

    The report is intentionally model/source-facing: users see their model
    sections, units, outputs, cache key, and generated backend targets without
    treating AxonScope's internal representation as a public API.
    """

    membrane = ensure_membrane_model(model)
    generated_report = inspect_generated_code(membrane)
    sources = tuple(
        _explain_source(source)
        for source in generated_report.sources
    )
    from axonscope.membranes.compiler import lower_membrane_model_to_ir
    from axonscope.model_ir.program import membrane_program_from_model_ir

    lowered_model = lower_membrane_model_to_ir(membrane)
    program = membrane_program_from_model_ir(lowered_model)
    return MembraneModelExplanation(
        model_kind=membrane.kind,
        components=_component_explanations(membrane),
        recording_outputs=MembraneRecordingOutputExplanation(
            gates=program.gate_names,
            currents=program.current_names,
            conductances=program.conductance_names,
            states=program.membrane_state_display_names,
            observables=program.observable_display_names,
            current_aggregates=_aggregate_names(program.current_names, program.current_groups),
            conductance_aggregates=_aggregate_names(
                program.conductance_names,
                program.conductance_groups,
            ),
        ),
        sources=sources,
    )


def _component_explanations(
    membrane: MembraneModel,
) -> tuple[MembraneComponentExplanation, ...]:
    if membrane.kind != "composite":
        return (MembraneComponentExplanation(label=membrane.kind, model_kind=membrane.kind),)
    labels = membrane.component_labels
    if not labels:
        labels = tuple(component.kind for component in membrane.components)
    return tuple(
        MembraneComponentExplanation(label=label, model_kind=component.kind)
        for label, component in zip(labels, membrane.components, strict=True)
    )


def _aggregate_names(
    names: tuple[str, ...],
    groups: tuple[tuple[int, ...], ...],
) -> tuple[str, ...]:
    return tuple(
        name
        for name, group in zip(names, groups, strict=True)
        if len(group) > 1
    )


def format_membrane_model_explanation(report: MembraneModelExplanation) -> str:
    """Format a membrane-model explanation report as plain text."""

    lines = [
        "AxonScope membrane model explanation",
        f"model={report.model_kind}",
        f"components={_format_components(report.components)}",
        f"recording_outputs={_format_recording_outputs(report.recording_outputs)}",
        "sources:",
    ]
    for source in report.sources:
        lines.extend(
            [
                f"  {source.model_name}:",
                f"    source={source.source_path}",
                f"    source_hash={source.source_hash}",
                f"    functions={source.function_names!r}",
                (
                    f"    cache={source.cache_status}, reason={source.cache_reason}, "
                    f"key={source.cache_key}"
                ),
                f"    cache_directory={source.cache_directory}",
                f"    generated_targets={source.generated_targets!r}",
            ]
        )
        display_name = source.metadata.get("display_name")
        if display_name:
            lines.append(f"    display_name={display_name}")
        lines.extend(
            [
                f"    inputs={_format_symbols(source.inputs)}",
                f"    parameters={_format_symbols(source.parameters)}",
                f"    states={_format_symbols(source.states)}",
                f"    gates={_format_names(source.gates)}",
                f"    currents={_format_names(source.currents)}",
                f"    observables={_format_names(source.observables)}",
                f"    diagnostics={_format_names(source.diagnostics)}",
                f"    step_program={_format_step_program(source.step)}",
                f"    source_outputs={_format_output_groups(source.source_outputs)}",
                f"    internal_outputs={_format_names(source.internal_outputs)}",
                "    sections:",
            ]
        )
        for section in source.sections:
            lines.extend(
                [
                    f"      {section.name} ({section.function_name}):",
                    f"        args={section.arguments!r}",
                    f"        assigns={_format_names(section.assignments)}",
                    f"        dependencies={_format_dependencies(section.dependencies)}",
                ]
            )
            if section.docstring:
                first_line = section.docstring.splitlines()[0].strip()
                lines.append(f"        doc={first_line}")
        if source.mechanisms:
            lines.append("    mechanisms:")
            for mechanism in source.mechanisms:
                lines.extend(
                    [
                        f"      {mechanism.name} ({mechanism.function_name}):",
                        f"        args={mechanism.arguments!r}",
                        f"        assigns={_format_names(mechanism.assignments)}",
                        (
                            "        external_dependencies="
                            f"{_format_names(mechanism.external_dependencies)}"
                        ),
                    ]
                )
                if mechanism.docstring:
                    first_line = mechanism.docstring.splitlines()[0].strip()
                    lines.append(f"        doc={first_line}")
        lines.append("    generated model_step targets:")
        for target in source.targets:
            lines.extend(
                [
                    f"      {target.target}:",
                    f"        path={target.path}",
                    f"        args={_format_names(target.arg_names)}",
                    f"        outputs={_format_names(target.output_names)}",
                    f"        retained_assignments={_format_names(target.retained_assignments)}",
                    (
                        "        pruned_from_model_step="
                        f"{_format_names(target.pruned_from_model_step)}"
                    ),
                ]
            )
    return "\n".join(lines)


def _explain_source(source: GeneratedMembraneCodeInspection) -> MembraneSourceExplanation:
    compiled_model = _compiled_model(source)
    source_tree = _parse_source(source.source_path)
    function_names = tuple(
        name
        for name in source.function_name.split(",")
        if name
    )
    sections = _source_sections(
        source_tree,
        function_names=function_names,
        compiled_model=compiled_model,
    )
    all_assignments = tuple(
        dict.fromkeys(
            assignment
            for section in sections
            for assignment in section.assignments
        )
    )
    targets = _target_explanations(source, all_assignments=all_assignments)
    source_outputs = _source_outputs(compiled_model)
    return MembraneSourceExplanation(
        model_name=source.model_name,
        source_path=source.source_path,
        source_hash=source.source_hash,
        function_names=function_names,
        cache_status=source.cache_status,
        cache_reason=source.cache_reason,
        cache_key=source.cache_key,
        cache_directory=source.cache_directory,
        generated_targets=tuple(str(target) for target in source.manifest.get("targets", ())),
        metadata=dict(compiled_model.metadata),
        inputs=tuple(_symbol(input_symbol) for input_symbol in compiled_model.inputs),
        parameters=tuple(_symbol(parameter) for parameter in compiled_model.parameters),
        states=tuple(_symbol(state) for state in compiled_model.states),
        gates=tuple(gate.name for gate in compiled_model.gates),
        currents=tuple(current.name for current in compiled_model.currents),
        observables=tuple(observable.name for observable in compiled_model.observables),
        diagnostics=_diagnostics(compiled_model),
        step=_step_explanation(compiled_model),
        source_outputs=source_outputs,
        internal_outputs=tuple(
            str(name)
            for name in compiled_model.metadata.get("internal_outputs", ())
        ),
        sections=sections,
        mechanisms=_source_mechanisms(sections),
        targets=targets,
    )


def _compiled_model(source: GeneratedMembraneCodeInspection) -> ModelIR:
    try:
        graph = source.file("optimized_graph.json")
    except KeyError:
        graph = source.file("graph.json")
    return model_ir_from_json(Path(graph.path).read_text(encoding="utf-8"))


def _symbol(symbol: ModelSymbol) -> MembraneSourceSymbol:
    default = getattr(symbol, "default", None)
    return MembraneSourceSymbol(
        name=symbol.name,
        kind=symbol.kind.value,
        role=symbol.quantity.role.value,
        unit=symbol.quantity.unit,
        default=default,
    )


def _diagnostics(model: ModelIR) -> tuple[str, ...]:
    if model.step_program is None:
        return ()
    return tuple(diagnostic.name for diagnostic in model.step_program.diagnostics)


def _step_explanation(model: ModelIR) -> MembraneStepExplanation | None:
    step = model.step_program
    if step is None and not any(state.initial is not None for state in model.states):
        return None
    return MembraneStepExplanation(
        state_initials=tuple(
            MembraneStateUpdateExplanation(state.name, _format_expression(state.initial))
            for state in model.states
            if state.initial is not None
        ),
        prepare_state_updates=tuple(
            MembraneStateUpdateExplanation(update.state, _format_expression(update.expression))
            for update in (() if step is None else step.prepare_state_updates)
        ),
        finalize_state_updates=tuple(
            MembraneStateUpdateExplanation(update.state, _format_expression(update.expression))
            for update in (() if step is None else step.finalize_state_updates)
        ),
        total_outward_current=None if step is None else _optional_expression(step.total_outward_current),
        explicit_outward_current=None if step is None else _optional_expression(step.explicit_outward_current),
        correction_current=None if step is None else _optional_expression(step.correction_current),
        prepare_gate_source=(
            "n/a" if step is None else step.prepare_gate_source.value
        ),
        linearization_gate_source=(
            "n/a" if step is None else step.linearization_gate_source.value
        ),
        diagnostics=tuple(
            MembraneStateUpdateExplanation(
                diagnostic.name,
                _format_expression(diagnostic.expression),
            )
            for diagnostic in (() if step is None else step.diagnostics)
        ),
    )


def _optional_expression(expression: Expression | None) -> str | None:
    return None if expression is None else _format_expression(expression)


def _source_outputs(model: ModelIR) -> Mapping[str, tuple[str, ...]]:
    raw = model.metadata.get("source_outputs", {})
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(group): tuple(str(name) for name in names)
        for group, names in raw.items()
        if isinstance(names, (tuple, list))
    }


def _parse_source(path: str) -> ast.Module:
    source_path = Path(path)
    return ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))


def _source_sections(
    tree: ast.Module,
    *,
    function_names: tuple[str, ...],
    compiled_model: ModelIR,
) -> tuple[MembraneSourceSection, ...]:
    functions = _source_functions(tree, model_name=compiled_model.name)
    selected = tuple(functions[name] for name in function_names if name in functions)
    model_names = _compiled_model_names(compiled_model)
    all_assignment_names = set(model_names)
    all_assignment_names.update(
        assignment
        for function in selected
        for assignment in _assignment_names(function)
    )
    sections: list[MembraneSourceSection] = []
    for function in selected:
        assignments = _assignment_names(function)
        dependencies = tuple(
            MembraneEquationDependency(
                assignment,
                _assignment_dependencies(
                    function,
                    assignment=assignment,
                    known_names=all_assignment_names,
                ),
            )
            for assignment in assignments
        )
        sections.append(
            MembraneSourceSection(
                name=_section_name(function),
                function_name=function.name,
                docstring=ast.get_docstring(function),
                arguments=tuple(argument.arg for argument in function.args.args),
                assignments=assignments,
                dependencies=dependencies,
            )
        )
    return tuple(sections)


def _source_mechanisms(
    sections: tuple[MembraneSourceSection, ...],
) -> tuple[MembraneMechanismExplanation, ...]:
    mechanisms: list[MembraneMechanismExplanation] = []
    for section in sections:
        if not section.name.startswith("mechanism:"):
            continue
        local_names = set(section.assignments)
        dependencies = tuple(
            dict.fromkeys(
                dependency
                for equation in section.dependencies
                for dependency in equation.depends_on
                if dependency not in local_names
            )
        )
        mechanisms.append(
            MembraneMechanismExplanation(
                name=section.name.split(":", 1)[1],
                function_name=section.function_name,
                docstring=section.docstring,
                arguments=section.arguments,
                assignments=section.assignments,
                external_dependencies=dependencies,
            )
        )
    return tuple(mechanisms)


def _source_functions(tree: ast.Module, *, model_name: str) -> dict[str, ast.FunctionDef]:
    functions: dict[str, ast.FunctionDef] = {}
    model_class = _source_model_class(tree, model_name=model_name)
    if model_class is not None:
        return {
            statement.name: statement
            for statement in model_class.body
            if isinstance(statement, ast.FunctionDef)
        }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for statement in node.body:
                if isinstance(statement, ast.FunctionDef):
                    functions[statement.name] = statement
    return functions


def _source_model_class(tree: ast.Module, *, model_name: str) -> ast.ClassDef | None:
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    for node in classes:
        if _class_model_name(node) == model_name:
            return node
    return classes[0] if len(classes) == 1 else None


def _class_model_name(node: ast.ClassDef) -> str:
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
        if target_name in {"model_kind", "kind", "name"} and isinstance(value, ast.Constant):
            if isinstance(value.value, str) and value.value:
                return value.value
    return _snake_case(node.name)


def _snake_case(name: str) -> str:
    first = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def _compiled_model_names(model: ModelIR) -> tuple[str, ...]:
    names: list[str] = []
    for group in (model.inputs, model.states, model.parameters):
        names.extend(symbol.name for symbol in group)
    names.extend(current.name for current in model.currents)
    names.extend(observable.name for observable in model.observables)
    if model.step_program is not None:
        names.extend(diagnostic.name for diagnostic in model.step_program.diagnostics)
    return tuple(dict.fromkeys(names))


def _section_name(function: ast.FunctionDef) -> str:
    for decorator in function.decorator_list:
        name = _decorator_section_name(decorator)
        if name is not None:
            return name
    return function.name


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
    if isinstance(node, ast.Attribute) and _is_model_name(node.value):
        return node.attr
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if not _is_model_name(node.func.value):
            return None
        if node.func.attr == "mechanism" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return f"mechanism:{arg.value}"
        return node.func.attr
    return None


def _is_model_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "model"


def _assignment_names(function: ast.FunctionDef) -> tuple[str, ...]:
    names: list[str] = []
    for statement in function.body:
        target = _assignment_target(statement)
        if target is not None:
            names.append(target)
    return tuple(dict.fromkeys(names))


def _assignment_target(statement: ast.stmt) -> str | None:
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return statement.target.id
    if isinstance(statement, ast.Assign):
        if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            return statement.targets[0].id
    return None


def _assignment_dependencies(
    function: ast.FunctionDef,
    *,
    assignment: str,
    known_names: set[str],
) -> tuple[str, ...]:
    for statement in function.body:
        if _assignment_target(statement) != assignment:
            continue
        value = _assignment_value(statement)
        if value is None:
            return ()
        names = _load_names(value)
        return tuple(name for name in sorted(names & known_names) if name != assignment)
    return ()


def _load_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.add(child.id)
        elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            if child.value.id in {"self", "cls"}:
                names.add(child.attr)
    return names


def _assignment_value(statement: ast.stmt) -> ast.AST | None:
    if isinstance(statement, ast.AnnAssign):
        return statement.value
    if isinstance(statement, ast.Assign):
        return statement.value
    return None


def _target_explanations(
    source: GeneratedMembraneCodeInspection,
    *,
    all_assignments: tuple[str, ...],
) -> tuple[GeneratedTargetExplanation, ...]:
    targets: list[GeneratedTargetExplanation] = []
    for generated in source.files:
        if generated.name not in {"jax_model.py", "numpy_model.py"}:
            continue
        tree = _parse_source(generated.path)
        target = _literal_constant(tree, "TARGET") or generated.name.removesuffix("_model.py")
        arg_names = _literal_tuple_constant(tree, "ARG_NAMES")
        output_names = _literal_tuple_constant(tree, "OUTPUT_NAMES")
        retained = _function_assignment_names(tree, function_name="model_step")
        retained_or_output = set(retained) | set(output_names)
        pruned = tuple(name for name in all_assignments if name not in retained_or_output)
        targets.append(
            GeneratedTargetExplanation(
                target=str(target),
                path=generated.path,
                arg_names=arg_names,
                output_names=output_names,
                retained_assignments=retained,
                pruned_from_model_step=pruned,
            )
        )
    return tuple(targets)


def _literal_tuple_constant(tree: ast.Module, name: str) -> tuple[str, ...]:
    value = _literal_constant(tree, name)
    if not isinstance(value, (tuple, list)):
        return ()
    return tuple(str(item) for item in value)


def _literal_constant(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != name:
            continue
        try:
            return ast.literal_eval(node.value)
        except ValueError:
            return None
    return None


def _function_assignment_names(tree: ast.Module, *, function_name: str) -> tuple[str, ...]:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return _assignment_names(node)
    return ()


def _format_symbols(symbols: tuple[MembraneSourceSymbol, ...]) -> str:
    if not symbols:
        return "()"
    parts = []
    for symbol in symbols:
        default = "" if symbol.default is None else f", default={symbol.default!r}"
        parts.append(f"{symbol.name}({symbol.role}, {symbol.unit}{default})")
    return _format_names(tuple(parts), max_items=8)


def _format_components(components: tuple[MembraneComponentExplanation, ...]) -> str:
    if not components:
        return "()"
    return _format_names(
        tuple(
            f"{component.label}:{component.model_kind}"
            for component in components
        ),
        max_items=8,
    )


def _format_recording_outputs(recording: MembraneRecordingOutputExplanation) -> str:
    groups = (
        ("gates", recording.gates, ()),
        ("currents", recording.currents, recording.current_aggregates),
        ("conductances", recording.conductances, recording.conductance_aggregates),
        ("states", recording.states, ()),
        ("observables", recording.observables, ()),
    )
    parts = []
    for group_name, names, aggregates in groups:
        detail = f"{group_name}: {_format_names(names, max_items=8)}"
        if aggregates:
            detail += f" aggregates={_format_names(aggregates, max_items=8)}"
        parts.append(detail)
    return "{" + "; ".join(parts) + "}"


def _format_output_groups(groups: Mapping[str, tuple[str, ...]]) -> str:
    if not groups:
        return "{}"
    parts = [
        f"{name}: {_format_names(values, max_items=8)}"
        for name, values in groups.items()
    ]
    return "{" + "; ".join(parts) + "}"


def _format_step_program(step: MembraneStepExplanation | None) -> str:
    if step is None:
        return "()"
    parts = []
    if step.state_initials:
        parts.append(f"initials={_format_state_updates(step.state_initials)}")
    if step.prepare_state_updates:
        parts.append(f"prepare={_format_state_updates(step.prepare_state_updates)}")
    if step.finalize_state_updates:
        parts.append(f"finalize={_format_state_updates(step.finalize_state_updates)}")
    solver_terms = {
        "total": step.total_outward_current,
        "explicit": step.explicit_outward_current,
        "correction": step.correction_current,
    }
    visible_terms = tuple(f"{name}={value}" for name, value in solver_terms.items() if value)
    if visible_terms:
        parts.append(
            "solver_terms="
            f"{_format_names(tuple(_short_text(term) for term in visible_terms), max_items=3)}"
        )
    if step.diagnostics:
        parts.append(f"diagnostics={_format_state_updates(step.diagnostics)}")
    if step.prepare_gate_source != "n/a" or step.linearization_gate_source != "n/a":
        parts.append(
            "gate_source="
            f"prepare:{step.prepare_gate_source}, linearization:{step.linearization_gate_source}"
        )
    return "; ".join(parts) if parts else "()"


def _format_state_updates(updates: tuple[MembraneStateUpdateExplanation, ...]) -> str:
    return _format_names(
        tuple(
            f"{update.state}<-{_short_text(update.expression)}"
            for update in updates
        ),
        max_items=8,
    )


def _format_dependencies(dependencies: tuple[MembraneEquationDependency, ...]) -> str:
    if not dependencies:
        return "()"
    parts = []
    for dependency in dependencies[:8]:
        parts.append(f"{dependency.name}<-{_format_names(dependency.depends_on, max_items=6)}")
    if len(dependencies) > 8:
        parts.append(f"... +{len(dependencies) - 8}")
    return "; ".join(parts)


def _format_names(names: tuple[str, ...], *, max_items: int = 12) -> str:
    if not names:
        return "()"
    shown = tuple(names[:max_items])
    suffix = "" if len(names) <= max_items else f", ... +{len(names) - max_items}"
    return "(" + ", ".join(shown) + suffix + ")"


def _short_text(text: str, *, max_chars: int = 96) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 4].rstrip() + " ..."


def _format_expression(expression: Expression) -> str:
    if isinstance(expression, Symbol):
        return expression.name
    if isinstance(expression, Literal):
        if expression.unit in {"", "1", "dimensionless"}:
            return repr(expression.value)
        return f"{expression.value} {expression.unit}"
    if isinstance(expression, UnaryOp):
        return f"(-{_format_expression(expression.operand)})"
    if isinstance(expression, BinaryOp):
        operator = {
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
        return (
            f"({_format_expression(expression.left)} "
            f"{operator} {_format_expression(expression.right)})"
        )
    if isinstance(expression, Call):
        args = ", ".join(_format_expression(arg) for arg in expression.args)
        return f"{expression.intrinsic}({args})"
    return type(expression).__name__


__all__ = [
    "GeneratedTargetExplanation",
    "MembraneComponentExplanation",
    "MembraneEquationDependency",
    "MembraneMechanismExplanation",
    "MembraneModelExplanation",
    "MembraneRecordingOutputExplanation",
    "MembraneSourceExplanation",
    "MembraneSourceSection",
    "MembraneSourceSymbol",
    "MembraneStateUpdateExplanation",
    "MembraneStepExplanation",
    "explain",
    "format_membrane_model_explanation",
]
