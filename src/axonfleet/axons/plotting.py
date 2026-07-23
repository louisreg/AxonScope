"""Plotting helpers for descriptive axon layouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from axonfleet.axons.flattened import flatten_layout
from axonfleet.utils import units

if TYPE_CHECKING:
    from axonfleet.axons.layout import Layout


_UNIT_DISPLAY = {
    "centimeter": "cm",
    "millimeter": "mm",
    "micrometer": "um",
    "meter": "m",
}

_DEFAULT_COLORS = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
    "#59a14f",
)


CompartmentLabels = bool | Literal["auto", "index", "section", "index+section", "none"]


def _unit_display(unit: Any) -> str:
    label = units.unit_label(unit) or "micrometer"
    return _UNIT_DISPLAY.get(label, label)


def _length_values_um_to_unit(values_um: np.ndarray, unit: Any) -> np.ndarray:
    unit_label = units.unit_label(unit) or "micrometer"
    return units.to_array(
        units.Q_(np.asarray(values_um, dtype=float), "micrometer"),
        unit_label,
        dtype=float,
    )


def _color_map(names: tuple[str, ...], section_colors: Mapping[str, str] | None) -> dict[str, str]:
    colors = dict(section_colors or {})
    for name in names:
        colors.setdefault(name, _DEFAULT_COLORS[len(colors) % len(_DEFAULT_COLORS)])
    return colors


def _label_mode(labels: CompartmentLabels, count: int, max_labels: int) -> str | None:
    if labels is False or labels == "none":
        return None
    if labels is True:
        return "index"
    if labels == "auto":
        return "index" if count <= max_labels else None
    return labels


def _compartment_label(index: int, section_name: str, mode: str) -> str:
    if mode == "section":
        return section_name
    if mode == "index+section":
        return f"{index}\n{section_name}"
    return str(index)


def _section_spans(layout: "Layout") -> list[tuple[int, str, float, float, int]]:
    flat = flatten_layout(layout)
    if layout.x_centers_um is not None:
        element = layout.elements[0]
        return [
            (
                0,
                element.section.name,
                float(flat.edges_um[0]),
                float(flat.edges_um[-1]),
                element.compartments,
            )
        ]

    spans: list[tuple[int, str, float, float, int]] = []
    cursor = float(layout.x_shift_um)
    for index, element in enumerate(layout.elements):
        start = cursor
        end = start + element.length_um
        spans.append((index, element.section.name, start, end, element.compartments))
        cursor = end
    return spans


def plot_layout(
    layout: "Layout",
    ax: Any | None = None,
    *,
    position_unit: Any = "micrometer",
    title: str | None = None,
    section_labels: bool = True,
    compartment_labels: CompartmentLabels = "auto",
    max_compartment_labels: int = 80,
    show_compartment_centers: bool = True,
    show_legend: bool = False,
    section_colors: Mapping[str, str] | None = None,
    section_alpha: float = 0.22,
    compartment_alpha: float = 0.72,
    label_fontsize: float = 8.0,
) -> Any:
    """Plot a descriptive layout as sections and numerical compartments.

    Parameters
    ----------
    layout:
        Descriptive `Layout` to plot.
    ax:
        Optional Matplotlib axes. A new axes is created when omitted.
    position_unit:
        Unit used for the x-axis.
    title:
        Optional axes title.
    section_labels:
        Draw one label per placed section.
    compartment_labels:
        Compartment label mode. `True`, `"index"`, and `"auto"` draw indices;
        `"section"` draws section names; `"index+section"` draws both.
    max_compartment_labels:
        Maximum number of compartments labelled when `compartment_labels="auto"`.
    show_compartment_centers:
        Draw a marker at each compartment center.
    show_legend:
        Draw a section-name color legend. Disabled by default because section
        labels are drawn directly on the layout.
    section_colors:
        Optional mapping from section name to Matplotlib color.
    section_alpha, compartment_alpha:
        Fill opacity for section and compartment bands.
    label_fontsize:
        Text size for section and compartment labels.
    """

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(10, 2.6))

    from matplotlib.patches import Patch, Rectangle

    flat = flatten_layout(layout)
    section_names = tuple(dict.fromkeys(flat.section_names))
    colors = _color_map(section_names, section_colors)
    edges = _length_values_um_to_unit(flat.edges_um, position_unit)
    centers = _length_values_um_to_unit(flat.x_um, position_unit)
    spans = []
    for index, name, start_um, end_um, compartments in _section_spans(layout):
        start, end = _length_values_um_to_unit(np.asarray([start_um, end_um]), position_unit)
        spans.append((index, name, float(start), float(end), compartments))

    section_y = 0.64
    section_height = 0.26
    compartment_y = 0.14
    compartment_height = 0.34
    x_span = max(float(edges[-1] - edges[0]), 1e-12)

    for section_index, name, start, end, compartment_count in spans:
        color = colors[name]
        span_fraction = (end - start) / x_span
        ax.add_patch(
            Rectangle(
                (start, section_y),
                end - start,
                section_height,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                alpha=section_alpha,
                zorder=1,
            )
        )
        if section_labels:
            if span_fraction < 0.025:
                label = f"{section_index}: {name}"
                rotation = 90
                fontsize = max(label_fontsize - 1.0, 6.0)
            else:
                label = f"{section_index}: {name}"
                rotation = 0
                fontsize = label_fontsize
            if compartment_count > 1 and span_fraction >= 0.045:
                label = f"{label}\n{compartment_count} comp."
            ax.text(
                0.5 * (start + end),
                section_y + 0.5 * section_height,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                color="0.15",
                rotation=rotation,
                clip_on=True,
                zorder=3,
            )

    for index in range(flat.Nx):
        name = flat.section_names[index]
        start = float(edges[index])
        end = float(edges[index + 1])
        ax.add_patch(
            Rectangle(
                (start, compartment_y),
                end - start,
                compartment_height,
                facecolor=colors[name],
                edgecolor="white",
                linewidth=0.7,
                alpha=compartment_alpha,
                zorder=2,
            )
        )

    label_mode = _label_mode(compartment_labels, flat.Nx, max_compartment_labels)
    if label_mode is not None:
        for index, (center, name) in enumerate(
            zip(centers, flat.section_names, strict=True)
        ):
            ax.text(
                float(center),
                compartment_y + 0.5 * compartment_height,
                _compartment_label(index, name, label_mode),
                ha="center",
                va="center",
                fontsize=label_fontsize,
                color="white",
                rotation=90 if label_mode != "index" else 0,
                clip_on=True,
                zorder=4,
            )

    if show_compartment_centers:
        ax.plot(
            centers,
            np.full_like(centers, compartment_y - 0.04, dtype=float),
            "|",
            color="0.2",
            markersize=7,
            markeredgewidth=1.0,
            zorder=5,
        )

    ax.vlines(
        edges,
        compartment_y,
        compartment_y + compartment_height,
        color="0.35",
        linewidth=0.35,
        alpha=0.55,
    )
    padding = 0.008 * x_span
    ax.set_xlim(float(edges[0]) - padding, float(edges[-1]) + padding)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(
        [
            section_y + 0.5 * section_height,
            compartment_y + 0.5 * compartment_height,
        ]
    )
    ax.set_yticklabels(["sections", "compartments"])
    ax.set_xlabel(f"x [{_unit_display(position_unit)}]")
    ax.set_title("Layout" if title is None else title, loc="left")
    ax.grid(axis="x", color="0.85", linewidth=0.6)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

    if show_legend and len(section_names) > 1:
        handles = [
            Patch(
                facecolor=colors[name],
                edgecolor=colors[name],
                alpha=compartment_alpha,
                label=name,
            )
            for name in section_names
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            ncol=min(len(handles), 4),
            frameon=False,
            fontsize=label_fontsize,
        )

    return ax


__all__: list[str] = []
