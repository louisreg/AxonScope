"""Shared plotting helpers for AxonFleet view modules."""

from __future__ import annotations

from typing import Any

from axonfleet.views.summary import unit_text


def ensure_axis(ax: Any | None = None) -> Any:
    """Return an existing Matplotlib axis or create a new one."""

    if ax is not None:
        return ax

    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    return ax


def axis_label(name: str, unit: Any | None = None) -> str:
    """Return ``name`` with a compact unit suffix when one is provided."""

    text = unit_text(unit)
    return name if not text else f"{name} [{text}]"


def decorate_axis(
    ax: Any,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    grid_axis: str | None = None,
    legend: bool = False,
    legend_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Apply common labels, title, grid, and optional legend to an axis."""

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        kwargs: dict[str, Any] = {"alpha": 0.3}
        if grid_axis is not None:
            kwargs["axis"] = grid_axis
        ax.grid(True, **kwargs)
    if legend:
        ax.legend(**(legend_kwargs or {}))
    return ax


__all__ = [
    "axis_label",
    "decorate_axis",
    "ensure_axis",
]
