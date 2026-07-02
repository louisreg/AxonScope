"""Shared helpers for AxonScope row/table/text view surfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TextIO

import numpy as np

from axonscope.utils import units


_SHORT_UNIT_ALIASES = {
    "micrometer": "um",
    "millimeter": "mm",
    "millisecond": "ms",
    "millivolt": "mV",
    "microampere": "uA",
    "nanoampere": "nA",
    "second": "s",
    "volt": "V",
}


def display_value(value: Any) -> Any:
    """Return a scalar/list representation suitable for rows and text reports."""

    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        return item
    return array.tolist()


def unit_label(unit: Any | None, *, fallback: str | None = None) -> str:
    """Return a canonical unit label or a caller-provided fallback."""

    if unit is None:
        return "" if fallback is None else fallback
    return units.unit_label(unit) or ("" if fallback is None else fallback)


def unit_text(unit: Any | None, *, fallback: str | None = None) -> str:
    """Return a compact display label for a unit-like value."""

    label = unit_label(unit, fallback=fallback)
    if not label:
        return ""
    return units.short_unit_label(label) or _SHORT_UNIT_ALIASES.get(label, label)


def rows_to_dataframe(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
) -> Any:
    """Return row dictionaries as a pandas DataFrame."""

    import pandas as pd

    return pd.DataFrame(tuple(rows), columns=None if columns is None else list(columns))


def format_summary(
    title: str,
    *,
    summary: Sequence[str] = (),
    rows: Sequence[str] = (),
) -> str:
    """Join title, summary lines, and row lines into one compact report."""

    lines = [title]
    lines.extend(str(line) for line in summary)
    lines.extend(str(line) for line in rows)
    return "\n".join(lines)


def print_summary(text: str, file: TextIO | None = None) -> None:
    """Print a text summary to the provided file object or stdout."""

    print(text, file=file)


__all__ = [
    "display_value",
    "format_summary",
    "print_summary",
    "rows_to_dataframe",
    "unit_label",
    "unit_text",
]
