"""View helpers for high-level protocol result containers."""

from __future__ import annotations

from typing import Any, TextIO

import numpy as np

from axonfleet.views.plotting import axis_label, decorate_axis, ensure_axis
from axonfleet.views.summary import (
    display_value,
    print_summary,
    rows_to_dataframe,
    unit_label,
    unit_text,
)
from axonfleet.utils import units


def recruitment_curve_rows(
    result: Any,
    *,
    unit: Any = "microampere",
) -> tuple[dict[str, Any], ...]:
    """Return one row per sampled recruitment amplitude."""

    unit_name = unit_label(unit, fallback="microampere")
    amplitudes = units.to_array(result.amplitudes, unit_name, dtype=float)
    return tuple(
        {
            "amplitude": float(amplitude),
            "count": int(count),
            "fraction": float(fraction),
        }
        for amplitude, count, fraction in zip(
            amplitudes,
            result.count,
            result.fraction,
            strict=True,
        )
    )


def recruitment_curve_to_dataframe(result: Any, *, unit: Any = "microampere") -> Any:
    """Return a pandas DataFrame for one recruitment curve."""

    return rows_to_dataframe(recruitment_curve_rows(result, unit=unit))


def format_recruitment_curve(result: Any, *, unit: Any = "microampere") -> str:
    """Return a compact recruitment-curve summary."""

    unit_name = unit_text(unit, fallback="microampere")
    lines = [
        "AxonFleet recruitment curve",
        f"values={result.amplitudes_uA.size}, rows={result.activated.shape[1]}",
    ]
    for row in recruitment_curve_rows(result, unit=unit):
        lines.append(
            f"  amplitude={row['amplitude']:g} {unit_name}: "
            f"{row['count']} rows ({row['fraction']:.3f})"
        )
    return "\n".join(lines)


def print_recruitment_curve(
    result: Any,
    file: TextIO | None = None,
    *,
    unit: Any = "microampere",
) -> None:
    """Print a compact recruitment-curve summary."""

    print_summary(format_recruitment_curve(result, unit=unit), file=file)


def plot_recruitment_curve(
    result: Any,
    ax: Any | None = None,
    *,
    unit: Any = "microampere",
    **plot_kwargs: Any,
) -> Any:
    """Plot recruitment fraction versus amplitude."""

    ax = ensure_axis(ax)

    unit_name = unit_label(unit, fallback="microampere")
    unit_display = unit_text(unit, fallback="microampere")
    amplitudes = units.to_array(result.amplitudes, unit_name, dtype=float)
    plot_kwargs.setdefault("marker", "o")
    plot_kwargs.setdefault("linewidth", 2.0)
    ax.plot(amplitudes, result.fraction, **plot_kwargs)
    ax.set_ylim(-0.05, 1.05)
    return decorate_axis(
        ax,
        xlabel=f"Amplitude [{unit_display}]",
        ylabel="Recruitment fraction",
        grid=True,
    )


def plot_recruitment_groups(
    result: Any,
    groups: Any,
    ax: Any | None = None,
    *,
    unit: Any = "microampere",
    include_total: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """Plot recruitment fractions for one or more row groups."""

    ax = ensure_axis(ax)

    group_values = np.asarray(groups, dtype=object)
    if group_values.shape != (result.activated.shape[1],):
        raise ValueError("groups must provide one label per recruited row.")

    unit_name = unit_label(unit, fallback="microampere")
    unit_display = unit_text(unit, fallback="microampere")
    amplitudes = units.to_array(result.amplitudes, unit_name, dtype=float)

    base_kwargs = dict(plot_kwargs)
    base_kwargs.setdefault("marker", "o")
    base_kwargs.setdefault("linewidth", 2.0)
    if include_total:
        kwargs = dict(base_kwargs)
        kwargs.setdefault("label", "all rows")
        ax.plot(amplitudes, result.fraction, **kwargs)

    for group in tuple(dict.fromkeys(group_values.tolist())):
        mask = group_values == group
        if not np.any(mask):
            continue
        kwargs = dict(base_kwargs)
        kwargs.setdefault("label", str(group))
        fraction = np.mean(result.activated[:, mask], axis=1)
        ax.plot(amplitudes, fraction, **kwargs)

    ax.set_ylim(-0.05, 1.05)
    return decorate_axis(
        ax,
        xlabel=f"Amplitude [{unit_display}]",
        ylabel="Recruitment fraction",
        grid=True,
        legend=True,
    )


def pool_sweep_rows(
    result: Any,
    *,
    value_name: str = "value",
    value_unit: Any | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return long-form rows for a generic pool sweep."""

    values = (
        result.value_values(unit=value_unit)
        if value_unit is not None
        else np.asarray(result.values, dtype=object)
    )
    rows = []
    for value_index, value in enumerate(values):
        for row_index, row_label in enumerate(result.row_labels):
            rows.append(
                {
                    "value_index": int(value_index),
                    "row": int(row_index),
                    "row_label": display_value(row_label),
                    value_name: display_value(value),
                    "observation": display_value(result.observations[value_index, row_index]),
                }
            )
    return tuple(rows)


def pool_sweep_to_dataframe(
    result: Any,
    *,
    value_name: str = "value",
    value_unit: Any | None = None,
) -> Any:
    """Return a pandas DataFrame for a generic pool sweep."""

    return rows_to_dataframe(
        pool_sweep_rows(result, value_name=value_name, value_unit=value_unit)
    )


def format_pool_sweep(
    result: Any,
    *,
    value_name: str = "value",
    value_unit: Any | None = None,
) -> str:
    """Return a compact pool-sweep summary."""

    lines = [
        "AxonFleet pool sweep",
        f"values={result.n_values}, rows={result.n_rows}",
    ]
    for row in pool_sweep_rows(result, value_name=value_name, value_unit=value_unit):
        lines.append(
            f"  {value_name}={row[value_name]} row={row['row']}: "
            f"observation={row['observation']}"
        )
    return "\n".join(lines)


def print_pool_sweep(
    result: Any,
    file: TextIO | None = None,
    *,
    value_name: str = "value",
    value_unit: Any | None = None,
) -> None:
    """Print a compact pool-sweep summary."""

    print_summary(
        format_pool_sweep(result, value_name=value_name, value_unit=value_unit),
        file=file,
    )


def plot_pool_sweep(
    result: Any,
    ax: Any | None = None,
    *,
    value_unit: Any | None = None,
    **plot_kwargs: Any,
) -> Any:
    """Plot scalar numeric pool-sweep observations by row."""

    ax = ensure_axis(ax)

    try:
        observations = np.asarray(result.observations, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("pool sweep observations are not scalar numeric values.") from exc
    if observations.ndim != 2:
        raise ValueError("pool sweep plot expects observations shaped (value, row).")

    x = result.value_values(unit=value_unit)
    plot_kwargs.setdefault("marker", "o")
    plot_kwargs.setdefault("linewidth", 1.5)
    for row_index in range(observations.shape[1]):
        row_label = result.row_labels[row_index]
        ax.plot(
            x,
            observations[:, row_index],
            label=f"row {row_label}",
            **plot_kwargs,
        )
    return decorate_axis(
        ax,
        xlabel="value" if value_unit is None else axis_label("value", value_unit),
        ylabel="observation",
        grid=True,
        legend=True,
    )


def threshold_curve_rows(
    result: Any,
    *,
    row_name: str = "row",
    row_unit: Any | None = None,
    threshold_unit: Any = "microampere",
) -> tuple[dict[str, Any], ...]:
    """Return one row per threshold-curve input row."""

    y_unit = unit_label(threshold_unit, fallback="microampere")
    thresholds = units.to_array(result.threshold, y_unit, dtype=float)
    lower_bounds = units.to_array(result.lower_bound, y_unit, dtype=float)
    upper_bounds = units.to_array(result.upper_bound, y_unit, dtype=float)
    row_values = (
        result.row_values(unit=row_unit)
        if row_unit is not None
        else np.asarray(result.row_labels, dtype=object)
    )
    return tuple(
        {
            row_name: display_value(row_value),
            "threshold": float(threshold),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "status": status,
        }
        for row_value, threshold, lower, upper, status in zip(
            row_values,
            thresholds,
            lower_bounds,
            upper_bounds,
            result.status,
            strict=True,
        )
    )


def threshold_curve_to_dataframe(
    result: Any,
    *,
    row_name: str = "row",
    row_unit: Any | None = None,
    threshold_unit: Any = "microampere",
) -> Any:
    """Return a pandas DataFrame summary for one threshold curve."""

    return rows_to_dataframe(
        threshold_curve_rows(
            result,
            row_name=row_name,
            row_unit=row_unit,
            threshold_unit=threshold_unit,
        )
    )


def format_threshold_curve(
    result: Any,
    *,
    row_name: str = "row",
    row_unit: Any | None = None,
    threshold_unit: Any = "microampere",
) -> str:
    """Return a compact threshold-curve summary."""

    threshold_text = unit_text(threshold_unit, fallback="microampere")
    lines = [
        "AxonFleet threshold curve",
        f"rows={len(result.row_labels)}, iterations={result.n_iterations}",
    ]
    for row in threshold_curve_rows(
        result,
        row_name=row_name,
        row_unit=row_unit,
        threshold_unit=threshold_unit,
    ):
        lines.append(
            f"  {row_name}={row[row_name]}: threshold={row['threshold']:g} "
            f"{threshold_text} ({row['status']})"
        )
    return "\n".join(lines)


def print_threshold_curve(
    result: Any,
    file: TextIO | None = None,
    *,
    row_name: str = "row",
    row_unit: Any | None = None,
    threshold_unit: Any = "microampere",
) -> None:
    """Print a compact threshold-curve summary."""

    print_summary(
        format_threshold_curve(
            result,
            row_name=row_name,
            row_unit=row_unit,
            threshold_unit=threshold_unit,
        ),
        file=file,
    )


def plot_threshold_curve(
    result: Any,
    ax: Any | None = None,
    *,
    row_unit: Any | None = None,
    threshold_unit: Any = "microampere",
    **plot_kwargs: Any,
) -> Any:
    """Plot threshold versus row value."""

    ax = ensure_axis(ax)

    x = result.row_values(unit=row_unit)
    y_unit = unit_label(threshold_unit, fallback="microampere")
    y = units.to_array(result.threshold, y_unit, dtype=float)
    row_text = (
        "row"
        if row_unit is None
        else units.short_unit_label(row_unit) or units.unit_label(row_unit) or str(row_unit)
    )
    threshold_text = unit_text(y_unit, fallback="microampere")
    plot_kwargs.setdefault("marker", "o")
    ax.plot(x, y, **plot_kwargs)
    return decorate_axis(
        ax,
        xlabel="row" if row_unit is None else f"row [{row_text}]",
        ylabel=f"threshold [{threshold_text}]",
        grid=True,
    )


__all__ = [
    "format_pool_sweep",
    "format_recruitment_curve",
    "format_threshold_curve",
    "plot_pool_sweep",
    "plot_recruitment_curve",
    "plot_recruitment_groups",
    "plot_threshold_curve",
    "pool_sweep_rows",
    "pool_sweep_to_dataframe",
    "print_pool_sweep",
    "print_recruitment_curve",
    "print_threshold_curve",
    "recruitment_curve_rows",
    "recruitment_curve_to_dataframe",
    "threshold_curve_rows",
    "threshold_curve_to_dataframe",
]
