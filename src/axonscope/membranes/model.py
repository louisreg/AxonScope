"""Runtime-independent membrane model descriptions."""

from __future__ import annotations

import inspect
import re
import sys
import types
from collections.abc import Callable, Mapping as MappingABC
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Sequence, dataclass_transform

import numpy as np


SectionFunction = Callable[..., Any]


def _freeze_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(params))


def _signature_value(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return ("array", str(arr.dtype), tuple(arr.shape), arr.tobytes())


def _matches_default(value: Any, default: Any) -> bool:
    try:
        return _signature_value(value) == _signature_value(default)
    except Exception:
        try:
            return value == default
        except Exception:
            return False


def _snake_case(name: str) -> str:
    first = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", first).lower()


_COMPONENT_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def section(name: str, **metadata: Any) -> Callable[[SectionFunction], SectionFunction]:
    """Mark a model method as one source-equation section."""

    section_name = str(name)

    def decorate(function: SectionFunction) -> SectionFunction:
        setattr(function, "__axonscope_section__", section_name)
        setattr(function, "__axonscope_section_metadata__", dict(metadata))
        return function

    return decorate


def _section_marker(section_name: str) -> Callable[..., Any]:
    def mark(function: SectionFunction | None = None, **metadata: Any) -> Any:
        decorator = section(section_name, **metadata)
        if function is None:
            return decorator
        return decorator(function)

    return mark


rates = _section_marker("rates")
currents = _section_marker("currents")
initials = _section_marker("initials")
step = _section_marker("step")


def mechanism(name: str, **metadata: Any) -> Callable[[SectionFunction], SectionFunction]:
    """Mark a model method as one named membrane mechanism."""

    return section(f"mechanism:{name}", **metadata)


@dataclass(frozen=True, slots=True)
class StateDeclaration:
    """Runtime placeholder for class-based source-model state declarations."""

    initial: Any
    description: str | None = None
    source: str | None = None
    bounds: tuple[Any, Any] | None = None


def state(
    initial: Any,
    *,
    description: str | None = None,
    source: str | None = None,
    bounds: tuple[Any, Any] | None = None,
) -> Any:
    """Declare a non-gate membrane state in a class-based model."""

    declaration = StateDeclaration(
        initial,
        description=description,
        source=source,
        bounds=bounds,
    )
    return field(default=declaration, init=False, repr=False, compare=False)


@dataclass_transform(kw_only_default=True, frozen_default=True)
class _MembraneModelClass(type):
    """Metaclass marker so editors treat `Model` subclasses like dataclasses."""


@dataclass(frozen=True, kw_only=True)
class Model(metaclass=_MembraneModelClass):
    """Public base class for user-authored membrane models.

    Subclasses are frozen keyword-only dataclasses. Annotated fields with
    defaults are model parameters; decorated methods are equation sections.
    Solver backends never receive a `Model` instance directly: the membrane
    compiler converts it to an internal `MembraneModel` descriptor first.
    """

    dtype: Any = field(default=np.float32, repr=False, compare=False)

    model_kind: ClassVar[str | None] = None
    metadata: ClassVar[Mapping[str, Any]] = MappingProxyType({})
    parameter_aliases: ClassVar[Mapping[str, str]] = MappingProxyType({})

    def __getattr__(self, name: str) -> Any:
        """Type bridge for compiler-provided equation symbols."""

        raise AttributeError(name)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        sys.modules.setdefault(cls.__module__, types.ModuleType(cls.__module__))
        if "__dataclass_fields__" not in cls.__dict__:
            dataclass(frozen=True, kw_only=True)(cls)
        _install_parameter_alias_init(cls)

    @classmethod
    def kind_name(cls) -> str:
        """Return the canonical model name used by generated artifacts."""

        explicit = getattr(cls, "model_kind", None)
        if explicit:
            return str(explicit)
        return _snake_case(cls.__name__)

    @classmethod
    def source_path(cls) -> str:
        """Return the Python source file that defines this model class."""

        source = inspect.getsourcefile(cls)
        if source is None:
            raise TypeError(f"Cannot locate source file for membrane model {cls.__qualname__}.")
        return source

    @classmethod
    def source_class(cls) -> str:
        """Return the class name selected by the source compiler."""

        return cls.__name__

    def _raw_parameter_values(self) -> dict[str, Any]:
        if not is_dataclass(self):
            return {}
        values: dict[str, Any] = {}
        provided_aliases = dict(getattr(self, "_provided_parameter_aliases", {}))
        for field_info in fields(self):
            if (
                not field_info.init
                or field_info.name == "dtype"
                or field_info.name.startswith("_")
            ):
                continue
            value = getattr(self, field_info.name)
            if field_info.default is not MISSING and _matches_default(value, field_info.default):
                continue
            values[provided_aliases.get(field_info.name, field_info.name)] = value
        return values

    def __post_init__(self) -> None:
        values = self._raw_parameter_values()
        if not values:
            return
        from axonscope.membranes.compiler import normalize_source_file_parameters

        normalize_source_file_parameters(
            self.__class__.source_path(),
            values,
            model_class_name=self.__class__.source_class(),
        )

    def to_membrane_model(self) -> "MembraneModel":
        """Return the internal descriptor used by solvers and inspectors."""

        from axonscope.membranes.compiler import parameterized_membrane_model

        descriptor = MembraneModel(
            self.__class__.kind_name(),
            source_path=self.__class__.source_path(),
            source_class=self.__class__.source_class(),
            dtype=self.dtype,
            _implementation=self,
        )
        return parameterized_membrane_model(descriptor, self._raw_parameter_values())

    @property
    def kind(self) -> str:
        """Canonical model kind for this instance."""

        return self.__class__.kind_name()

    @property
    def params(self) -> Mapping[str, Any]:
        """Normalized numeric parameters used by the internal compiler."""

        return self.to_membrane_model().params

    def inspect_generated_code(self, **kwargs: Any) -> Any:
        """Inspect generated compiler artifacts for this membrane model."""

        return self.to_membrane_model().inspect_generated_code(**kwargs)

    def explain(self) -> Any:
        """Explain source sections, units, cache identity, and generated targets."""

        return self.to_membrane_model().explain()

    def keep(self, *values: Any) -> None:
        """Mark intermediates as intentionally retained inside a source section."""

        _ = values


def _install_parameter_alias_init(cls: type[Model]) -> None:
    aliases = {
        str(alias): str(target)
        for alias, target in dict(getattr(cls, "parameter_aliases", {})).items()
    }
    if not aliases:
        return
    original_init = cls.__init__
    public_signature = inspect.signature(cls)

    def __init__(self: Model, *args: Any, **kwargs: Any) -> None:
        provided_aliases: dict[str, str] = {}
        for alias, target in aliases.items():
            if alias not in kwargs:
                continue
            if target in kwargs:
                raise TypeError(
                    f"{cls.__name__} received both {alias!r} and canonical {target!r}."
            )
            provided_aliases[target] = alias
            kwargs[target] = kwargs.pop(alias)
        object.__setattr__(self, "_provided_parameter_aliases", provided_aliases)
        original_init(self, *args, **kwargs)
        object.__setattr__(self, "_provided_parameter_aliases", provided_aliases)

    __init__.__name__ = "__init__"
    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    __init__.__doc__ = original_init.__doc__
    cls.__init__ = __init__  # type: ignore[method-assign]
    cls.__signature__ = public_signature  # type: ignore[attr-defined]


@dataclass(frozen=True)
class MembraneModel:
    """Internal descriptive membrane model specification.

    `MembraneModel` intentionally contains no solver backend, JAX function, or
    compiled compute object. It is the backend-neutral descriptor that axons
    carry internally after public `Model` instances have been normalized.
    """

    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)
    components: tuple["MembraneModel", ...] = ()
    component_labels: tuple[str, ...] = ()
    source_path: str | None = None
    source_class: str | None = None
    dtype: Any = np.float32
    _implementation: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "params", _freeze_params(self.params))
        object.__setattr__(self, "components", tuple(self.components))
        labels = tuple(str(label) for label in self.component_labels)
        if labels and len(labels) != len(self.components):
            raise ValueError(
                "component_labels must match the number of membrane components."
            )
        for label in labels:
            _validate_component_label(label)
        object.__setattr__(self, "component_labels", labels)
        source_path = None if self.source_path is None else str(self.source_path)
        object.__setattr__(self, "source_path", source_path)
        source_class = None if self.source_class is None else str(self.source_class)
        object.__setattr__(self, "source_class", source_class)
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    def _static_signature(self) -> tuple[Any, ...]:
        params = tuple(
            sorted((key, _signature_value(value)) for key, value in self.params.items())
        )
        components = tuple(component._static_signature() for component in self.components)
        return (
            "membrane",
            self.kind,
            self.source_path,
            self.source_class,
            str(self.dtype),
            params,
            self.component_labels,
            components,
        )

    def inspect_generated_code(self, **kwargs: Any) -> Any:
        """Inspect generated compiler artifacts for this membrane model."""

        from axonscope.membranes.generated_code import inspect_generated_code

        return inspect_generated_code(self, **kwargs)

    def explain(self) -> Any:
        """Explain source sections, units, cache identity, and generated targets."""

        from axonscope.membranes.explain import explain

        return explain(self)

    def __hash__(self) -> int:
        return hash(self._static_signature())


def ensure_membrane_model(value: Any) -> MembraneModel:
    """Return `value` as an internal descriptive membrane model."""

    if isinstance(value, MembraneModel):
        return value
    if isinstance(value, type) and issubclass(value, Model):
        return value().to_membrane_model()
    if isinstance(value, Model):
        return value.to_membrane_model()
    raise TypeError(
        "Axon sections require an axonscope.membranes.Model instance or "
        "internal MembraneModel description; "
        f"got {value.__class__.__module__}.{value.__class__.__qualname__}."
    )


def _validate_component_label(label: str) -> None:
    if not _COMPONENT_LABEL_RE.fullmatch(label):
        raise ValueError(
            "Composite component labels must be snake_case identifiers starting "
            f"with a lowercase letter; got {label!r}."
        )


def _component_label_from_kind(component: MembraneModel) -> str:
    label = str(component.kind)
    _validate_component_label(label)
    return label


def _resolve_composite_components(
    components: Mapping[str, Any] | Sequence[Any],
) -> tuple[tuple[MembraneModel, ...], tuple[str, ...]]:
    if isinstance(components, MappingABC):
        labels: list[str] = []
        models: list[MembraneModel] = []
        for label, component in components.items():
            normalized_label = str(label)
            _validate_component_label(normalized_label)
            if normalized_label in labels:
                raise ValueError(
                    f"Composite component label {normalized_label!r} is duplicated."
                )
            labels.append(normalized_label)
            models.append(ensure_membrane_model(component))
        if not models:
            raise ValueError("Composite requires at least one membrane component.")
        return tuple(models), tuple(labels)

    models = tuple(ensure_membrane_model(component) for component in components)
    if not models:
        raise ValueError("Composite requires at least one membrane component.")
    labels = tuple(_component_label_from_kind(component) for component in models)
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        duplicate_list = ", ".join(repr(label) for label in duplicates)
        raise ValueError(
            "Composite received duplicate membrane kinds in a sequence "
            f"({duplicate_list}). Use a mapping with explicit component labels, "
            "for example Composite({'passive_weak': Passive(...), "
            "'passive_strong': Passive(...)})."
        )
    return models, labels


def Composite(components: Mapping[str, Any] | Sequence[Any]) -> MembraneModel:
    """Compose several membrane descriptions on the same section."""

    models, labels = _resolve_composite_components(components)
    return MembraneModel(
        "composite",
        components=models,
        component_labels=labels,
    )


__all__ = [
    "Composite",
    "MembraneModel",
    "Model",
    "StateDeclaration",
    "currents",
    "ensure_membrane_model",
    "initials",
    "mechanism",
    "rates",
    "section",
    "state",
    "step",
]
