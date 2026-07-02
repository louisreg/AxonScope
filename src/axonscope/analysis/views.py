"""View helpers for analysis results and reports."""

from __future__ import annotations

from typing import Any, TextIO

import numpy as np

from axonscope.plotting import decorate_axis, ensure_axis
from axonscope.summary_views import (
    display_value,
    format_summary,
    print_summary,
    rows_to_dataframe,
    unit_text,
)
from axonscope.utils import units


def _value_for_display(value: Any) -> Any:
    return display_value(value)


def _unit_text(unit: Any | None) -> str:
    return unit_text(unit)


def analysis_result_rows(result: Any) -> tuple[dict[str, Any], ...]:
    """Return row dictionaries for one ``AnalysisResult``."""

    values = np.asarray(result.values)
    rows = []
    unit_text = _unit_text(result.unit)
    for row_index, (status, row_label) in enumerate(
        zip(result.statuses, result.row_labels, strict=True)
    ):
        rows.append(
            {
                "analysis": result.name,
                "row": int(row_index),
                "row_label": _value_for_display(row_label),
                "value": _value_for_display(values[row_index]),
                "status": getattr(status, "value", str(status)),
                "message": result.messages[row_index],
                "unit": unit_text,
            }
        )
    return tuple(rows)


def analysis_report_rows(report: Any) -> tuple[dict[str, Any], ...]:
    """Return row dictionaries for all analyses in one report."""

    return tuple(row for analysis in report.analyses for row in analysis_result_rows(analysis))


def analysis_result_to_dataframe(result: Any) -> Any:
    """Return one analysis result as a pandas DataFrame."""

    return rows_to_dataframe(analysis_result_rows(result))


def analysis_report_to_dataframe(report: Any) -> Any:
    """Return one analysis report as a pandas DataFrame."""

    return rows_to_dataframe(analysis_report_rows(report))


def format_analysis_result(result: Any) -> str:
    """Return a compact text representation for one analysis result."""

    pop = result.population
    unit_text = _unit_text(result.unit) or "-"
    title = (
        f"{result.name}: valid={pop.n_valid}/{pop.n_total}, "
        f"applicable={pop.n_applicable}, failed={pop.n_failed}, unit={unit_text}"
    )
    rows = []
    for row in analysis_result_rows(result):
        message = "" if not row["message"] else f" message={row['message']}"
        rows.append(
            f"  row {row['row']}: {row['status']} value={row['value']}{message}"
        )
    return format_summary(title, rows=rows)


def format_analysis_report(report: Any) -> str:
    """Return a compact text representation for one analysis report."""

    rows = []
    for result in report.analyses:
        rows.append(format_analysis_result(result))
    return format_summary(
        "AxonScope analysis report",
        summary=(f"analyses={len(report.analyses)}",),
        rows=rows,
    )


def print_analysis_report(
    report: Any,
    file: TextIO | None = None,
) -> None:
    """Print a compact analysis report."""

    print_summary(format_analysis_report(report), file=file)


def print_analysis_result(
    result: Any,
    file: TextIO | None = None,
) -> None:
    """Print a compact analysis result."""

    print_summary(format_analysis_result(result), file=file)


def plot_analysis_result(
    result: Any,
    ax: Any | None = None,
    *,
    x: Any | None = None,
    x_unit: Any | None = None,
    x_label: str = "row",
    y_label: str | None = None,
    title: str | None = None,
    grid: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """Plot one scalar numeric analysis result."""

    ax = ensure_axis(ax)

    y = np.asarray(result.values)
    if y.ndim != 1:
        raise ValueError("analysis result plot expects one scalar value per row.")
    try:
        y = y.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis result values are not numeric.") from exc
    x_values = result.row_values(unit=x_unit) if x is None else np.asarray(x, dtype=float)
    if x_values.shape != y.shape:
        raise ValueError("x values must have the same shape as analysis values.")

    plot_kwargs.setdefault("marker", "o")
    plot_kwargs.setdefault("linewidth", 2.0)
    ax.plot(x_values, y, **plot_kwargs)
    unit_text = _unit_text(result.unit)
    if x_unit is not None and x_label == "row":
        x_label = f"row [{_unit_text(x_unit)}]"
    return decorate_axis(
        ax,
        xlabel=x_label,
        ylabel=y_label or (result.name if not unit_text else f"{result.name} [{unit_text}]"),
        title=title,
        grid=grid,
    )


def plot_analysis_report(report: Any, ax: Any | None = None, **plot_kwargs: Any) -> Any:
    """Plot scalar numeric analysis values by row.

    Non-scalar or non-numeric analyses are skipped. A ``ValueError`` is raised
    only when the report contains no plottable scalar values.
    """

    ax = ensure_axis(ax)

    plotted = False
    for result in report.analyses:
        values = np.asarray(result.values)
        if values.ndim != 1:
            continue
        try:
            y = values.astype(float)
        except (TypeError, ValueError):
            continue
        x = np.arange(y.shape[0], dtype=int)
        kwargs = dict(plot_kwargs)
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("linewidth", 2.0)
        kwargs.setdefault("label", result.name)
        ax.plot(x, y, **kwargs)
        plotted = True

    if not plotted:
        raise ValueError("analysis report has no scalar numeric values to plot.")
    return decorate_axis(
        ax,
        xlabel="row",
        ylabel="analysis value",
        grid=True,
        legend=True,
    )


def plot_spike_raster(
    result: Any,
    ax: Any | None = None,
    *,
    threshold_mV: Any = -10.0,
    min_distance_ms: Any = 1.0,
    peak_height_mV: Any | None = None,
    min_width_ms: Any | None = None,
    spatial_filter: str = "recorded",
    line_half_height: Any = 0.5,
    **line_kwargs: Any,
) -> Any:
    """Plot spikes detected by :func:`axonscope.analysis.rasterize`."""

    ax = ensure_axis(ax)

    from axonscope.analysis.posthoc import rasterize

    line_half_height_um = units.to_um(line_half_height)
    spike_times_ms, spike_positions_um = rasterize(
        result,
        threshold_mV=threshold_mV,
        min_distance_ms=min_distance_ms,
        peak_height_mV=peak_height_mV,
        min_width_ms=min_width_ms,
        spatial_filter=spatial_filter,
    )
    line_kwargs.setdefault("color", "black")
    line_kwargs.setdefault("linewidth", 1.0)
    if spike_times_ms.size:
        ax.vlines(
            spike_times_ms,
            spike_positions_um - line_half_height_um,
            spike_positions_um + line_half_height_um,
            **line_kwargs,
        )
    return decorate_axis(
        ax,
        xlabel="Time [ms]",
        ylabel="Axon position [um]",
        grid=False,
    )


__all__ = [
    "analysis_report_rows",
    "analysis_report_to_dataframe",
    "analysis_result_rows",
    "analysis_result_to_dataframe",
    "format_analysis_report",
    "format_analysis_result",
    "plot_analysis_result",
    "plot_analysis_report",
    "plot_spike_raster",
    "print_analysis_report",
    "print_analysis_result",
]
