"""Shared result typing aliases."""

from __future__ import annotations

from typing import Any, TypeAlias


ResultArray: TypeAlias = Any
RecordingValue: TypeAlias = ResultArray | dict[str, ResultArray]
RecordingDict: TypeAlias = dict[str, RecordingValue]
ObservationDict: TypeAlias = dict[str, Any]


__all__ = [
    "ObservationDict",
    "RecordingDict",
    "RecordingValue",
    "ResultArray",
]
