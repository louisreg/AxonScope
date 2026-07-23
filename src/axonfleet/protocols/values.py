"""Value normalization helpers for protocol parameter sweeps."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from axonfleet.utils import units


def _normalize_rows(rows: Sequence[Any]) -> tuple[Any, ...]:
    if units.is_quantity_like(rows):
        magnitudes = np.asarray(rows.magnitude)
        if magnitudes.ndim != 1:
            raise ValueError("rows must be a 1D sequence.")
        row_unit = getattr(rows, "units")
        return tuple(units.Q_(float(value), row_unit) for value in magnitudes)
    normalized = tuple(rows)
    if not normalized:
        return ()
    return normalized


def _normalize_sweep_values(values: Sequence[Any]) -> tuple[Any, ...]:
    if units.is_quantity_like(values):
        magnitudes = np.asarray(values.magnitude)
        if magnitudes.ndim != 1:
            raise ValueError("values must be a 1D sequence.")
        value_unit = getattr(values, "units")
        return tuple(units.Q_(float(value), value_unit) for value in magnitudes)
    normalized = tuple(values)
    if not normalized:
        return ()
    return normalized


def _require_current_array_uA(value: Any, *, name: str) -> np.ndarray:
    """Normalize a unit-bearing scalar/array/list of current values."""

    if units.is_quantity_like(value):
        return units.require_current_array_uA(value, name=name, dtype=float)
    if isinstance(value, (list, tuple)) and all(units.is_quantity_like(item) for item in value):
        return np.asarray(
            [units.require_current_uA(item, name=name) for item in value],
            dtype=float,
        )
    return units.require_current_array_uA(value, name=name, dtype=float)


def _broadcast_bound(value: Any, row_count: int, *, name: str) -> np.ndarray:
    values = _require_current_array_uA(value, name=name)
    if values.ndim == 0:
        return np.full(row_count, float(values.item()), dtype=float)
    if values.shape != (row_count,):
        raise ValueError(f"{name} must be scalar or have shape ({row_count},).")
    return np.asarray(values, dtype=float)


def _resolve_threshold_bounds(
    bounds: tuple[Any, Any] | Callable[[Any], tuple[Any, Any]],
    rows: tuple[Any, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if callable(bounds):
        lower: list[float] = []
        upper: list[float] = []
        for row in rows:
            row_low, row_high = bounds(row)
            lower.append(units.require_current_uA(row_low, name="bounds(row)[0]"))
            upper.append(units.require_current_uA(row_high, name="bounds(row)[1]"))
        low_uA = np.asarray(lower, dtype=float)
        high_uA = np.asarray(upper, dtype=float)
    else:
        low_uA = _broadcast_bound(bounds[0], len(rows), name="bounds[0]")
        high_uA = _broadcast_bound(bounds[1], len(rows), name="bounds[1]")
    if np.any(high_uA <= low_uA):
        raise ValueError("bounds must be ordered as (low, high) for every row.")
    return low_uA, high_uA


def _threshold_converged(
    low_uA: np.ndarray,
    high_uA: np.ndarray,
    *,
    tolerance_uA: float | None,
    relative_tolerance: float | None,
) -> np.ndarray:
    width = np.asarray(high_uA, dtype=float) - np.asarray(low_uA, dtype=float)
    converged = np.zeros(width.shape, dtype=bool)
    if tolerance_uA is not None:
        converged |= width <= tolerance_uA
    if relative_tolerance is not None:
        scale = np.maximum(np.abs(high_uA), np.finfo(float).eps)
        converged |= width <= float(relative_tolerance) * scale
    return converged


__all__ = [
    "_normalize_rows",
    "_normalize_sweep_values",
    "_require_current_array_uA",
    "_resolve_threshold_bounds",
    "_threshold_converged",
]
