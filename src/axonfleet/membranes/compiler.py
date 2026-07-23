"""Internal lowering from public membrane descriptors to the source compiler."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from axonfleet.membranes.model import MembraneModel, Model, ensure_membrane_model
from axonfleet.model_ir.composition import compose_model_ir
from axonfleet.model_ir.schema import ModelIR
from axonfleet.model_ir.source import SourceModelCompileResult, compile_model_source_file
from axonfleet.utils import units


_SOURCE_ROOT = Path(__file__).resolve().parent / "models"


@dataclass(frozen=True, slots=True)
class MembraneLoweringResult:
    """Internal membrane lowering result with source-codegen provenance."""

    model: ModelIR
    source_results: tuple[SourceModelCompileResult, ...]


def parameterized_membrane_model(
    model: MembraneModel,
    params: dict[str, Any] | None = None,
) -> MembraneModel:
    """Return a parameterized copy of a membrane model prototype."""

    if model.kind == "composite":
        if params:
            raise ValueError("Composite membrane models cannot receive parameter overrides.")
        return model
    overrides = dict(params or {})
    base_params = {str(name): float(value) for name, value in dict(model.params).items()}
    if model.source_path is None:
        normalized = normalize_source_parameters(model.kind, overrides)
        source_path = membrane_source_path(model.kind)
    else:
        source_path = Path(model.source_path).resolve()
        normalized = normalize_source_file_parameters(
            source_path,
            overrides,
            model_class_name=model.source_class,
        )
    explicit = {**base_params, **normalized}
    source_defaults = _source_parameter_defaults_for_path(
        source_path,
        model.source_class,
    )
    derived = _derived_parameter_defaults_from_path(
        source_path,
        {**source_defaults, **explicit},
        model_class_name=model.source_class,
    )
    return MembraneModel(
        model.kind,
        {**source_defaults, **derived, **explicit},
        components=model.components,
        source_path=model.source_path,
        source_class=model.source_class,
        dtype=model.dtype,
    )


def source_membrane_model(
    path: str | os.PathLike[str],
    params: dict[str, Any] | None = None,
) -> MembraneModel:
    """Return a public membrane descriptor backed by one standalone source file."""

    source_path = Path(path).resolve()
    compiled = compile_model_source_file(source_path)
    normalized = normalize_source_file_parameters(source_path, params or {})
    derived = _derived_parameter_defaults_from_path(
        source_path,
        normalized,
        model_class_name=None,
    )
    return MembraneModel(
        compiled.model.name,
        {**derived, **normalized},
        source_path=str(source_path),
    )


def lower_membrane_model_to_ir(model: MembraneModel) -> ModelIR:
    """Compile a membrane descriptor through its standalone source file."""

    return lower_membrane_model_with_sources(model).model


def lower_membrane_model_with_sources(
    model: MembraneModel,
    *,
    load_generated_modules: tuple[str, ...] = (),
    generated_targets: tuple[str, ...] = (),
) -> MembraneLoweringResult:
    """Compile a membrane descriptor and retain source/cache compile details."""

    model = ensure_membrane_model(model)
    if model.kind == "composite":
        components: list[ModelIR] = []
        source_results: list[SourceModelCompileResult] = []
        for component in model.components:
            try:
                lowered = lower_membrane_model_with_sources(
                    component,
                    load_generated_modules=load_generated_modules,
                    generated_targets=generated_targets,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Composite membrane components must all lower to Model IR; "
                    f"component kind {component.kind!r} has no source model."
                ) from exc
            components.append(lowered.model)
            source_results.extend(lowered.source_results)
        return MembraneLoweringResult(
            model=compose_model_ir(
                tuple(components),
                component_labels=model.component_labels or None,
            ),
            source_results=tuple(source_results),
        )

    source_path = (
        Path(model.source_path).resolve()
        if model.source_path is not None
        else membrane_source_path(model.kind)
    )
    compiled = compile_model_source_file(
        source_path,
        model_class_name=model.source_class,
        parameter_defaults=_float_parameter_defaults(model.params),
        load_generated_modules=load_generated_modules,
        generated_targets=generated_targets,
    )
    return MembraneLoweringResult(model=compiled.model, source_results=(compiled,))


def membrane_source_path(kind: str) -> Path:
    """Return the source file for a membrane kind without a model-family registry."""

    source_root = _SOURCE_ROOT.resolve()
    source_path = (source_root / f"{kind}.py").resolve()
    if source_path.parent != source_root or not source_path.is_file():
        raise ValueError(f"No membrane source file exists for kind {kind!r}.")
    return source_path


def normalize_source_parameters(kind: str, params: dict[str, Any]) -> dict[str, float]:
    """Convert public overrides to the units declared by the source model."""

    specs = _source_parameter_specs(kind)
    aliases = _source_parameter_aliases(kind)
    return _normalize_parameters(
        params,
        specs=specs,
        aliases=aliases,
        model_label=f"membrane kind {kind!r}",
    )


def normalize_source_file_parameters(
    path: str | os.PathLike[str],
    params: dict[str, Any],
    *,
    model_class_name: str | None = None,
) -> dict[str, float]:
    """Convert public overrides to the units declared by a source model file."""

    source_path = Path(path).resolve()
    specs = _source_parameter_specs_for_path(source_path, model_class_name)
    aliases = _source_parameter_aliases_for_path(source_path, model_class_name)
    return _normalize_parameters(
        params,
        specs=specs,
        aliases=aliases,
        model_label=f"membrane source {source_path}",
    )


def _normalize_parameters(
    params: dict[str, Any],
    *,
    specs: dict[str, str],
    aliases: dict[str, str],
    model_label: str,
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name, value in params.items():
        public_name = str(name)
        source_name = aliases.get(public_name, public_name)
        if source_name not in specs:
            raise ValueError(f"Unknown parameter {public_name!r} for {model_label}.")
        normalized[source_name] = _convert_value_to_unit(
            value,
            unit=specs[source_name],
            name=public_name,
        )
    return normalized


def _float_parameter_defaults(params: Any) -> dict[str, float]:
    return {str(name): float(value) for name, value in dict(params).items()}


@lru_cache(maxsize=None)
def _source_parameter_defaults_for_path(
    source_path: Path,
    model_class_name: str | None,
) -> dict[str, float]:
    model = compile_model_source_file(
        source_path.resolve(),
        model_class_name=model_class_name,
    ).model
    return {
        parameter.name: float(parameter.default)
        for parameter in model.parameters
        if parameter.default is not None
    }


@lru_cache(maxsize=None)
def _source_parameter_specs(kind: str) -> dict[str, str]:
    return _source_parameter_specs_for_path(membrane_source_path(kind), None)


@lru_cache(maxsize=None)
def _source_parameter_specs_for_path(
    source_path: Path,
    model_class_name: str | None,
) -> dict[str, str]:
    source_path = source_path.resolve()
    model = compile_model_source_file(
        source_path,
        model_class_name=model_class_name,
    ).model
    return {parameter.name: parameter.quantity.unit for parameter in model.parameters}


@lru_cache(maxsize=None)
def _source_parameter_aliases(kind: str) -> dict[str, str]:
    return _source_parameter_aliases_for_path(membrane_source_path(kind), None)


@lru_cache(maxsize=None)
def _source_parameter_aliases_for_path(
    source_path: Path,
    model_class_name: str | None,
) -> dict[str, str]:
    source_path = source_path.resolve()
    aliases = _source_parameter_aliases_for_source(source_path, model_class_name)
    if not isinstance(aliases, dict):
        raise TypeError(f"{source_path}.PARAMETER_ALIASES must be a dict when defined.")
    return {str(public): str(source) for public, source in aliases.items()}


def _derived_parameter_defaults_from_path(
    source_path: Path,
    params: dict[str, float],
    *,
    model_class_name: str | None,
) -> dict[str, float]:
    source_class = _source_model_class_from_path(source_path, model_class_name)
    source_module = _source_module_from_path(source_path)
    derived_name, defaults_fn = _source_derived_parameter_function(
        source_class,
        source_module,
    )
    if defaults_fn is None:
        return {}
    signature = inspect.signature(defaults_fn)
    kwargs = {
        name: params[name]
        for name in signature.parameters
        if name in params
    }
    values = defaults_fn(**kwargs)
    if not isinstance(values, dict):
        raise TypeError(f"{source_path}.{derived_name}(...) must return a dict.")
    return {str(name): float(value) for name, value in values.items()}


def _source_derived_parameter_function(
    source_class: type[Model] | None,
    source_module: Any,
) -> tuple[str, Any | None]:
    for source_object in (source_class, source_module):
        if source_object is None:
            continue
        for name in (
            "derive_parameters",
            "derived_parameters",
            "parameter_defaults",
            "_parameter_defaults",
        ):
            defaults_fn = getattr(source_object, name, None)
            if callable(defaults_fn):
                return name, defaults_fn
    return "derive_parameters", None


def _source_parameter_aliases_for_source(
    source_path: Path,
    model_class_name: str | None,
) -> dict[str, Any]:
    source_class = _source_model_class_from_path(source_path, model_class_name)
    if source_class is not None:
        return dict(getattr(source_class, "parameter_aliases", {}))
    return getattr(_source_module_from_path(source_path), "PARAMETER_ALIASES", {})


@lru_cache(maxsize=None)
def _source_model_class_from_path(
    source_path: Path,
    model_class_name: str | None,
) -> type[Model] | None:
    module = _source_module_from_path(source_path)
    if model_class_name is not None:
        value = getattr(module, model_class_name, None)
        if isinstance(value, type) and issubclass(value, Model):
            return value
        return None
    candidates = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, Model)
        and value is not Model
        and value.__module__ == module.__name__
    ]
    return candidates[0] if len(candidates) == 1 else None


@lru_cache(maxsize=None)
def _source_module_from_path(source_path: Path) -> Any:
    source_path = source_path.resolve()
    digest = hashlib.blake2b(str(source_path).encode("utf-8"), digest_size=8).hexdigest()
    module_name = f"axonfleet.membranes.models._source_{digest}"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load membrane source module {source_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _convert_value_to_unit(value: Any, *, unit: str, name: str) -> float:
    same_unit = _same_unit_magnitude(value, unit)
    if same_unit is not None:
        return same_unit
    if unit == units.DIMENSIONLESS:
        return float(value)
    if unit == units.VOLTAGE_MV:
        return _convert_quantity(value, unit, target="millivolt", converter=units.to_mV)
    if unit == units.TEMPERATURE_DEGC:
        return _convert_quantity(value, unit, target="degree_Celsius", converter=units.to_degC)
    if unit == units.CONDUCTANCE_DENSITY_MS_CM2:
        return _convert_quantity(
            value,
            unit,
            target="millisiemens / centimeter ** 2",
            converter=units.to_scalar,
        )
    if unit == units.CURRENT_DENSITY_UA_CM2:
        return _convert_quantity(
            value,
            unit,
            target="microampere / centimeter ** 2",
            converter=units.to_scalar,
        )
    if unit == units.RESISTANCE_AREA_OHM_CM2:
        return _convert_quantity(
            value,
            unit,
            target="ohm * centimeter ** 2",
            converter=units.to_ohm_cm2,
        )
    if unit == units.CONCENTRATION_MM:
        return _convert_quantity(value, unit, target="millimolar", converter=units.to_mM)
    if unit == units.TIME_MS:
        return _convert_quantity(value, unit, target="millisecond", converter=units.to_ms)
    if unit == units.RATE_PER_MS:
        return _convert_quantity(
            value,
            unit,
            target="1 / millisecond",
            converter=units.to_scalar,
        )
    if unit == units.RATE_PER_MS_PER_MV:
        return _convert_quantity(
            value,
            unit,
            target="1 / (millisecond * millivolt)",
            converter=units.to_scalar,
        )
    if unit == units.RATE_PER_MS_PER_MM:
        return _convert_quantity(
            value,
            unit,
            target="1 / (millisecond * millimolar)",
            converter=units.to_scalar,
        )
    if unit == units.CONCENTRATION_PER_CURRENT_DENSITY_TIME:
        return _convert_quantity(
            value,
            unit,
            target="millimolar / (microampere / centimeter ** 2 * millisecond)",
            converter=units.to_scalar,
        )
    if unit == "micrometer":
        return units.require_length_um(value, name=name)
    raise ValueError(f"Parameter {name!r} uses unsupported source unit {unit!r}.")


def _convert_quantity(value: Any, unit: str, *, target: str, converter: Any) -> float:
    same_unit = _same_unit_magnitude(value, unit, target)
    if same_unit is not None:
        return same_unit
    if converter is units.to_scalar:
        return units.to_scalar(value, target)
    return converter(value)


def _same_unit_magnitude(value: Any, *units_: str) -> float | None:
    unit_label = getattr(value, "unit", None)
    if unit_label is None:
        unit_label = getattr(value, "units", None)
    if unit_label is not None and str(unit_label) in set(units_) and hasattr(value, "magnitude"):
        return float(value.magnitude)
    return None


__all__ = [
    "MembraneLoweringResult",
    "lower_membrane_model_to_ir",
    "lower_membrane_model_with_sources",
    "membrane_source_path",
    "normalize_source_file_parameters",
    "normalize_source_parameters",
    "parameterized_membrane_model",
    "source_membrane_model",
]
