"""View helpers for one-axon simulation results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TextIO

import numpy as np

from axonscope.plotting import axis_label, decorate_axis, ensure_axis
from axonscope.summary_views import print_summary, rows_to_dataframe, unit_text
from axonscope.utils import units


def unit_display(unit: Any) -> str:
    """Return a compact display label for a unit-like value."""

    return unit_text(unit)


def time_values(result: Any, *, unit: Any = "millisecond") -> np.ndarray:
    """Return result times as plain values in `unit`."""

    unit_label = units.unit_label(unit) or "millisecond"
    return units.to_array(
        units.Q_(np.asarray(result.t, dtype=float), "millisecond"),
        unit_label,
        dtype=float,
    )


def position_values(result: Any, *, unit: Any = "micrometer") -> np.ndarray:
    """Return recorded axon positions as plain values in `unit`."""

    return result.recorded_axis.position_values(unit=unit)


def voltage_values(result: Any, *, unit: Any = "millivolt") -> np.ndarray:
    """Return membrane voltages as plain values in `unit`."""

    unit_label = units.unit_label(unit) or "millivolt"
    vm = np.asarray(result.Vm, dtype=float)
    return units.to_array(units.Q_(vm, "millivolt"), unit_label, dtype=float)


def peak_voltage_values(result: Any, *, unit: Any = "millivolt") -> np.ndarray:
    """Return the peak membrane voltage for each recorded column."""

    vm = voltage_values(result, unit=unit)
    if vm.ndim != 2:
        raise ValueError(f"result.Vm must be 2D (time, position), got shape {vm.shape}.")
    return np.max(vm, axis=0)


def nearest_position_index(result: Any, position: Any) -> int:
    """Return the recorded column nearest to `position`."""

    positions_um = position_values(result, unit="micrometer")
    target_um = units.to_um(position)
    return int(np.argmin(np.abs(positions_um - target_um)))


def trace_values(
    result: Any,
    *,
    position: Any | None = None,
    index: int | None = None,
    time_unit: Any = "millisecond",
    voltage_unit: Any = "millivolt",
) -> tuple[np.ndarray, np.ndarray]:
    """Return one voltage trace as ``(time, Vm)`` arrays."""

    vm = voltage_values(result, unit=voltage_unit)
    if vm.ndim != 2:
        raise ValueError(f"result.Vm must be 2D (time, position), got shape {vm.shape}.")
    if index is not None and position is not None:
        raise ValueError("Provide either `index` or `position`, not both.")
    if index is None:
        index = (
            nearest_position_index(result, position)
            if position is not None
            else vm.shape[1] // 2
        )
    if index < 0 or index >= vm.shape[1]:
        raise IndexError(f"index {index} is outside Vm columns 0..{vm.shape[1] - 1}.")

    t = time_values(result, unit=time_unit)
    if t.shape[0] != vm.shape[0]:
        raise ValueError("result.t length must match result.Vm time dimension.")
    return t, vm[:, index]


def plot_trace(
    result: Any,
    ax: Any | None = None,
    *,
    position: Any | None = None,
    index: int | None = None,
    time_unit: Any = "millisecond",
    voltage_unit: Any = "millivolt",
    label: str | None = None,
    title: str | None = None,
    grid: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """Plot one membrane-voltage trace."""

    ax = ensure_axis(ax)

    if index is None:
        vm = voltage_values(result, unit=voltage_unit)
        if vm.ndim != 2:
            raise ValueError(
                f"result.Vm must be 2D (time, position), got shape {vm.shape}."
            )
        resolved_index = (
            nearest_position_index(result, position)
            if position is not None
            else vm.shape[1] // 2
        )
    else:
        resolved_index = index
    t, trace = trace_values(
        result,
        position=position,
        index=index,
        time_unit=time_unit,
        voltage_unit=voltage_unit,
    )
    positions = position_values(result, unit="micrometer")

    plot_kwargs.setdefault("linewidth", 2.0)
    if label is not None:
        plot_kwargs.setdefault("label", label)
    ax.plot(t, trace, **plot_kwargs)
    return decorate_axis(
        ax,
        xlabel=axis_label("Time", time_unit),
        ylabel=axis_label("Vm", voltage_unit),
        title=title or f"Vm at x={positions[resolved_index]:g} um",
        grid=grid,
        legend=label is not None,
    )


def plot_traces(
    result: Any,
    ax: Any | None = None,
    *,
    positions: Sequence[Any] | None = None,
    indices: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    time_unit: Any = "millisecond",
    voltage_unit: Any = "millivolt",
    title: str = "Vm traces",
    grid: bool = True,
    legend: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """Plot several membrane-voltage traces from one result."""

    ax = ensure_axis(ax)

    if positions is not None and indices is not None:
        raise ValueError("Provide either `positions` or `indices`, not both.")
    if indices is None:
        if positions is None:
            vm = voltage_values(result, unit=voltage_unit)
            indices = tuple(range(vm.shape[1]))
        else:
            indices = tuple(nearest_position_index(result, position) for position in positions)
    else:
        indices = tuple(int(index) for index in indices)

    if labels is None:
        recorded_positions = position_values(result, unit="micrometer")
        labels = tuple(f"x={recorded_positions[index]:g} um" for index in indices)
    elif len(labels) != len(indices):
        raise ValueError("labels length must match selected trace count.")

    for index, label in zip(indices, labels, strict=True):
        kwargs = dict(plot_kwargs)
        kwargs.setdefault("label", label)
        t, trace = trace_values(
            result,
            index=index,
            time_unit=time_unit,
            voltage_unit=voltage_unit,
        )
        kwargs.setdefault("linewidth", 2.0)
        ax.plot(t, trace, **kwargs)

    return decorate_axis(
        ax,
        xlabel=axis_label("Time", time_unit),
        ylabel=axis_label("Vm", voltage_unit),
        title=title,
        grid=grid,
        legend=legend,
    )


def plot_population_traces(
    results: Any,
    ax: Any | None = None,
    *,
    position: Any | None = None,
    index: int | None = None,
    labels: Sequence[str] | None = None,
    time_unit: Any = "millisecond",
    voltage_unit: Any = "millivolt",
    title: str = "Population Vm traces",
    grid: bool = True,
    legend: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """Plot one membrane-voltage trace per row of a population result."""

    ax = ensure_axis(ax)

    rows = tuple(results)
    if labels is None:
        labels = tuple(f"row {row_index}" for row_index in range(len(rows)))
    elif len(labels) != len(rows):
        raise ValueError("labels length must match population row count.")

    for row, label in zip(rows, labels, strict=True):
        kwargs = dict(plot_kwargs)
        kwargs.setdefault("label", label)
        t, trace = trace_values(
            row,
            position=position,
            index=index,
            time_unit=time_unit,
            voltage_unit=voltage_unit,
        )
        kwargs.setdefault("linewidth", 2.0)
        ax.plot(t, trace, **kwargs)

    return decorate_axis(
        ax,
        xlabel=axis_label("Time", time_unit),
        ylabel=axis_label("Vm", voltage_unit),
        title=title,
        grid=grid,
        legend=legend,
    )


def plot_peak_voltage(
    result: Any,
    ax: Any | None = None,
    *,
    position_unit: Any = "micrometer",
    voltage_unit: Any = "millivolt",
    title: str = "Peak Vm by recorded position",
    grid: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """Plot peak membrane voltage over recorded axon positions."""

    ax = ensure_axis(ax)

    x = position_values(result, unit=position_unit)
    y = peak_voltage_values(result, unit=voltage_unit)
    plot_kwargs.setdefault("linewidth", 2.0)
    ax.plot(x, y, **plot_kwargs)
    return decorate_axis(
        ax,
        xlabel=axis_label("Axon position x", position_unit),
        ylabel=axis_label("Peak Vm", voltage_unit),
        title=title,
        grid=grid,
    )


def plot_recording_group(
    result: Any,
    group: str,
    ax: Any | None = None,
    *,
    position: Any | None = None,
    index: int | None = None,
    labels: Sequence[str] | None = None,
    time_unit: Any = "millisecond",
    title: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    legend: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """Plot all channels from one named recording group at one recorded column."""

    ax = ensure_axis(ax)

    recordings = result.recordings or {}
    values_by_name = recordings.get(group)
    if not isinstance(values_by_name, Mapping):
        raise KeyError(f"recording group {group!r} is not available.")

    vm = voltage_values(result)
    if index is not None and position is not None:
        raise ValueError("Provide either `index` or `position`, not both.")
    if index is None:
        index = (
            nearest_position_index(result, position)
            if position is not None
            else vm.shape[1] // 2
        )
    if index < 0 or index >= vm.shape[1]:
        raise IndexError(f"index {index} is outside recorded columns 0..{vm.shape[1] - 1}.")

    names = tuple(values_by_name)
    if labels is None:
        labels = names
    elif len(labels) != len(names):
        raise ValueError("labels length must match recording group channel count.")

    t = time_values(result, unit=time_unit)
    for name, label in zip(names, labels, strict=True):
        values = np.asarray(values_by_name[name])
        if values.ndim < 2:
            raise ValueError(f"recording group channel {name!r} must include time and position axes.")
        kwargs = dict(plot_kwargs)
        kwargs.setdefault("label", label)
        kwargs.setdefault("linewidth", 2.0)
        ax.plot(t, values[:, index], **kwargs)

    return decorate_axis(
        ax,
        xlabel=axis_label("Time", time_unit),
        ylabel=ylabel or group,
        title=title or group,
        grid=grid,
        legend=legend,
        legend_kwargs={"frameon": False},
    )


def plot_recorded_axis(
    result: Any,
    ax: Any | None = None,
    *,
    selectors: Mapping[str, Any] | None = None,
    markers: Mapping[str, Any] | None = None,
    position_unit: Any = "micrometer",
    title: str = "Recorded Vm positions",
    grid: bool = True,
) -> Any:
    """Plot recorded Vm columns and optional public position selectors."""

    ax = ensure_axis(ax)

    positions = position_values(result, unit=position_unit)
    positions_um = position_values(result, unit="micrometer")
    original_indices = result.recorded_axis.index_values()

    ax.hlines(0.0, positions[0], positions[-1], color="0.75", linewidth=3)
    ax.scatter(positions, np.zeros_like(positions), s=20, color="0.55", label="recorded")

    if markers:
        for label, marker_position in markers.items():
            marker_um = units.to_um(marker_position)
            unit_label = units.unit_label(position_unit) or "micrometer"
            marker_value = units.to_array(units.Q_(marker_um, "micrometer"), unit_label)
            ax.axvline(
                float(marker_value),
                color="crimson",
                linestyle="--",
                linewidth=1.0,
                label=label,
            )

    if selectors:
        for row_index, (label, selector) in enumerate(selectors.items(), start=1):
            columns = selector.columns(
                positions_um=positions_um,
                original_indices=original_indices,
            )
            ax.scatter(
                positions[columns],
                np.full(columns.shape, row_index),
                s=55,
                label=label,
            )
        ax.set_ylim(-0.6, len(selectors) + 0.8)
    else:
        ax.set_ylim(-0.6, 0.8)

    ax.set_yticks([])
    return decorate_axis(
        ax,
        xlabel=axis_label("Axon position", position_unit),
        title=title,
        grid=grid,
        grid_axis="x",
        legend=True,
        legend_kwargs={"ncol": 3, "fontsize": 8, "frameon": False},
    )


def plot_recorded_axes(
    results: Mapping[str, Any],
    ax: Any | None = None,
    *,
    position_unit: Any = "micrometer",
    title: str = "Recorded Vm positions",
    show_indices: bool = True,
    grid: bool = True,
) -> Any:
    """Plot retained Vm positions for several result/recording policies."""

    ax = ensure_axis(ax)

    labels = tuple(results)
    for row_index, (label, result) in enumerate(results.items()):
        row = result if hasattr(result, "recorded_axis") else result[0]
        axis = row.recorded_axis
        positions = axis.position_values(unit=position_unit)
        indices = axis.index_values()
        ax.scatter(
            positions,
            np.full(positions.shape, row_index),
            s=38,
            label=label,
        )
        if show_indices:
            for position, original_index in zip(positions, indices, strict=True):
                ax.text(
                    float(position),
                    row_index + 0.12,
                    str(int(original_index)),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    return decorate_axis(
        ax,
        xlabel=axis_label("Axon position", position_unit),
        title=title,
        grid=grid,
        grid_axis="x",
    )


def plot_map(
    result: Any,
    ax: Any | None = None,
    *,
    time_unit: Any = "millisecond",
    position_unit: Any = "micrometer",
    voltage_unit: Any = "millivolt",
    cmap: str = "viridis",
    colorbar: bool = True,
    title: str = "Vm propagation",
    **imshow_kwargs: Any,
) -> Any:
    """Plot the membrane-voltage space-time map."""

    ax = ensure_axis(ax)

    vm = voltage_values(result, unit=voltage_unit)
    if vm.ndim != 2:
        raise ValueError(f"result.Vm must be 2D (time, position), got shape {vm.shape}.")
    t = time_values(result, unit=time_unit)
    x = position_values(result, unit=position_unit)
    if t.shape[0] != vm.shape[0]:
        raise ValueError("result.t length must match result.Vm time dimension.")
    if x.shape[0] != vm.shape[1]:
        raise ValueError("recorded positions must match result.Vm position dimension.")

    imshow_kwargs.setdefault("aspect", "auto")
    imshow_kwargs.setdefault("origin", "lower")
    imshow_kwargs.setdefault("cmap", cmap)
    image = ax.imshow(
        vm.T,
        extent=[t[0], t[-1], x[0], x[-1]],
        **imshow_kwargs,
    )
    decorate_axis(
        ax,
        xlabel=axis_label("Time", time_unit),
        ylabel=axis_label("Axon position x", position_unit),
        title=f"{title} [{unit_display(voltage_unit)}]",
        grid=False,
    )
    if colorbar:
        ax.figure.colorbar(
            image,
            ax=ax,
            label=f"Vm [{unit_display(voltage_unit)}]",
        )
    return ax


def vm_raster_rows(raster: Any) -> tuple[dict[str, Any], ...]:
    """Return row dictionaries summarizing a ``VmRasterResult``."""

    bits = np.asarray(raster.unpack(), dtype=bool)
    if bits.ndim != 4:
        raise ValueError("VmRaster unpacked values must have shape (batch, raster, probe, time).")

    thresholds = np.asarray(raster.thresholds_mV, dtype=float)
    rows: list[dict[str, Any]] = []
    for batch_index in range(bits.shape[0]):
        probe_indices = _vm_raster_metadata_for_row(raster.probe_indices, raster, batch_index)
        probe_mask = _vm_raster_metadata_for_row(raster.probe_mask, raster, batch_index)
        original_indices = _vm_raster_metadata_for_row(
            raster.original_indices,
            raster,
            batch_index,
        )
        positions_um = _vm_raster_metadata_for_row(raster.positions_um, raster, batch_index)
        for raster_index, name in enumerate(raster.names):
            threshold_mV = float(thresholds[raster_index])
            for probe_index in range(bits.shape[2]):
                if not bool(probe_mask[raster_index, probe_index]):
                    continue
                probe_bits = bits[batch_index, raster_index, probe_index]
                active_indices = np.flatnonzero(probe_bits)
                active_count = int(active_indices.size)
                first_time_ms = (
                    None
                    if active_count == 0
                    else float(active_indices[0]) * float(raster.dt_ms)
                )
                last_time_ms = (
                    None
                    if active_count == 0
                    else float(active_indices[-1]) * float(raster.dt_ms)
                )
                rows.append(
                    {
                        "row": int(batch_index),
                        "raster": int(raster_index),
                        "name": str(name),
                        "probe": int(probe_index),
                        "probe_index": int(probe_indices[raster_index, probe_index]),
                        "original_index": int(original_indices[raster_index, probe_index]),
                        "position_um": float(positions_um[raster_index, probe_index]),
                        "threshold_mV": threshold_mV,
                        "active": bool(active_count > 0),
                        "active_samples": active_count,
                        "first_time_ms": first_time_ms,
                        "last_time_ms": last_time_ms,
                    }
                )
    return tuple(rows)


def vm_raster_to_dataframe(raster: Any) -> Any:
    """Return a VmRaster summary as a pandas DataFrame."""

    return rows_to_dataframe(vm_raster_rows(raster))


def format_vm_raster(raster: Any) -> str:
    """Return a compact text representation of a ``VmRasterResult``."""

    lines = [
        (
            "VmRasterResult "
            f"batch={raster.batch_size}, rasters={raster.raster_count}, "
            f"probes={raster.probe_count}, nt={raster.nt}, dt_ms={raster.dt_ms:g}, "
            f"packed={raster.packed_nbytes} bytes"
        )
    ]
    for row in vm_raster_rows(raster):
        first_time = "-" if row["first_time_ms"] is None else f"{row['first_time_ms']:g} ms"
        lines.append(
            "  "
            f"row {row['row']} {row['name']} probe {row['probe']}: "
            f"original={row['original_index']}, x={row['position_um']:g} um, "
            f"threshold={row['threshold_mV']:g} mV, active={row['active']}, "
            f"samples={row['active_samples']}, first={first_time}"
        )
    return "\n".join(lines)


def print_vm_raster(raster: Any, file: TextIO | None = None) -> None:
    """Print a compact VmRaster summary."""

    print_summary(format_vm_raster(raster), file=file)


def _vm_raster_metadata_for_row(values: Any, raster: Any, row: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3:
        return array[int(row)]
    return array


def plot_vm_raster(
    raster: Any,
    ax: Any | None = None,
    *,
    row: int = 0,
    time_unit: Any = "millisecond",
    title: str = "VmRaster threshold windows",
    grid: bool = True,
) -> Any:
    """Plot packed VmRaster threshold-crossing windows."""

    ax = ensure_axis(ax)

    unpacked = raster.unpack()
    if unpacked.ndim != 4:
        raise ValueError("VmRaster unpacked values must have shape (batch, raster, probe, time).")
    if row < 0 or row >= unpacked.shape[0]:
        raise IndexError(f"row {row} is outside VmRaster batch 0..{unpacked.shape[0] - 1}.")

    time_unit_label = units.unit_label(time_unit) or "millisecond"
    scale = units.to_array(units.Q_(1.0, "millisecond"), time_unit_label, dtype=float)
    dt = float(raster.dt_ms) * float(scale)
    t0 = 0.0
    t1 = float(raster.nt) * dt

    original_indices = _vm_raster_metadata_for_row(raster.original_indices, raster, row)
    probe_mask = _vm_raster_metadata_for_row(raster.probe_mask, raster, row)
    labels = []
    tracks: list[tuple[int, int]] = []
    for definition_index, name in enumerate(raster.names):
        for probe_index in range(raster.probe_count):
            if not bool(probe_mask[definition_index, probe_index]):
                continue
            original_index = int(original_indices[definition_index, probe_index])
            labels.append(f"{name} @ compartment {original_index}")
            tracks.append((definition_index, probe_index))

    for row_index, (definition_index, probe_index) in enumerate(tracks):
        bits = unpacked[row, definition_index, probe_index]
        padded = np.concatenate(([False], np.asarray(bits, dtype=bool), [False]))
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        starts = transitions[0::2]
        stops = transitions[1::2]
        spans = [
            (float(t0 + start * dt), max(float((stop - start) * dt), dt))
            for start, stop in zip(starts, stops, strict=True)
        ]
        if spans:
            ax.broken_barh(
                spans,
                (row_index - 0.35, 0.70),
                facecolors=f"C{row_index}",
                alpha=0.85,
            )

    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(t0, t1)
    return decorate_axis(
        ax,
        xlabel=axis_label("Time", time_unit),
        ylabel="Observer probe",
        title=title,
        grid=grid,
        grid_axis="x",
    )


plot_voltage_trace = plot_trace
plot_voltage_map = plot_map


__all__ = [
    "nearest_position_index",
    "peak_voltage_values",
    "plot_map",
    "plot_peak_voltage",
    "plot_population_traces",
    "plot_recorded_axes",
    "plot_recorded_axis",
    "plot_recording_group",
    "plot_trace",
    "plot_traces",
    "plot_vm_raster",
    "plot_voltage_map",
    "plot_voltage_trace",
    "position_values",
    "time_values",
    "trace_values",
    "unit_display",
    "voltage_values",
    "format_vm_raster",
    "print_vm_raster",
    "vm_raster_rows",
    "vm_raster_to_dataframe",
]
