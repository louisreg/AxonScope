"""Runtime-independent membrane model descriptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


def _freeze_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(params))


def _signature_value(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return ("array", str(arr.dtype), tuple(arr.shape), arr.tobytes())


@dataclass(frozen=True)
class MembraneModel:
    """Descriptive membrane model specification.

    `MembraneModel` intentionally contains no solver backend, JAX function, or
    compiled compute object. It is the public, DSL-ready description that axons
    carry. Solver runtimes translate it to the current hand-written membrane
    implementations when a simulation starts.
    """

    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)
    components: tuple["MembraneModel", ...] = ()
    dtype: Any = np.float32
    _implementation: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "params", _freeze_params(self.params))
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    def _static_signature(self) -> tuple[Any, ...]:
        params = tuple(
            sorted((key, _signature_value(value)) for key, value in self.params.items())
        )
        components = tuple(component._static_signature() for component in self.components)
        return ("membrane", self.kind, params, components)

    def __hash__(self) -> int:
        return hash(self._static_signature())


def ensure_membrane_model(value: Any) -> MembraneModel:
    """Return `value` as a descriptive membrane model.

    Non-`MembraneModel` values are accepted only as a transitional internal
    escape hatch for existing low-level tests and advanced code paths.
    """

    if isinstance(value, MembraneModel):
        return value
    return MembraneModel(
        "legacy",
        params={"class": value.__class__.__name__},
        dtype=getattr(value, "dtype", np.float32),
        _implementation=value,
    )


def Composite(components: Sequence[Any]) -> MembraneModel:
    """Compose several membrane descriptions on the same section."""

    return MembraneModel(
        "composite",
        components=tuple(ensure_membrane_model(component) for component in components),
    )


__all__ = ["Composite", "MembraneModel", "ensure_membrane_model"]
