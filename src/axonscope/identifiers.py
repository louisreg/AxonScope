"""Opaque public identifiers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Identifier:
    """Base class for typed public identifiers."""

    value: str

    def __post_init__(self) -> None:
        value = str(self.value).strip()
        if not value:
            raise ValueError(f"{type(self).__name__} value must be a non-empty string.")
        object.__setattr__(self, "value", value)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"


@dataclass(frozen=True, repr=False)
class AxonId(_Identifier):
    """Opaque identifier for one axon row or footprint row."""


@dataclass(frozen=True, repr=False)
class DriveId(_Identifier):
    """Opaque identifier for one extracellular drive."""


@dataclass(frozen=True, repr=False)
class SignalId(_Identifier):
    """Opaque identifier for one recorded or observed signal."""


__all__ = ["AxonId", "DriveId", "SignalId"]
