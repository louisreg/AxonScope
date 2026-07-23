"""Small validation helpers shared across AxonFleet."""

from __future__ import annotations

from typing import Sequence


def normalize_positive_int(value: int, *, name: str) -> int:
    """Return `value` as a validated positive integer."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer >= 1.")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer >= 1.") from exc
    if count != value:
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if count < 1:
        raise ValueError(f"{name} must be >= 1, got {count}.")
    return count


def require_positive(value: float, *, name: str) -> float:
    """Return `value` after validating it is strictly positive."""

    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def require_non_negative(value: float, *, name: str) -> float:
    """Return `value` after validating it is non-negative."""

    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def normalize_non_empty_string(value: object, *, name: str) -> str:
    """Return `value` as a stripped non-empty string."""

    if value is None:
        raise TypeError(f"{name} must be a non-empty string.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} cannot be empty.")
    return text


def normalize_string_tuple(value: Sequence[str] | str, *, name: str) -> tuple[str, ...]:
    """Return `value` as a tuple of stripped non-empty strings."""

    if isinstance(value, str):
        raw_values = (value,)
    else:
        try:
            raw_values = tuple(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a string or a sequence of strings.") from exc
    return tuple(normalize_non_empty_string(item, name=f"{name} entry") for item in raw_values)
