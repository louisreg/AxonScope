"""User-facing views for AxonScope simulation inspection records."""

from __future__ import annotations

from typing import Any, Sequence, TextIO

import numpy as np

from axonscope.inspection_records import SimulationInspection


def format_simulation_inspection(report: SimulationInspection) -> str:
    """Return a compact multiline text report."""

    policy = report.planning.execution_policy
    if policy is None:
        policy_text = "default"
    else:
        precision = "model"
        if policy.precision is not None:
            precision = policy.precision.solver_dtype
        policy_text = (
            f"runtime={policy.runtime.value}, device={policy.device.kind}, "
            f"precision={precision}"
        )

    lines = [
        "AxonScope solver pipeline inspection",
        "planning:",
        (
            f"  axons={report.planning.axon_count}, Nt={report.planning.step_count}, "
            f"duration={report.planning.duration_ms:g} ms, dt={report.planning.dt_ms:g} ms"
        ),
        f"  execution_policy={policy_text}",
        "dispatch/batch:",
    ]
    for group in report.dispatch_groups:
        route = "batch" if group.will_batch else "scalar"
        padding_text = "yes" if group.has_padding else "no"
        geometry = "shared" if group.geometry_shared else "parameterized"
        method = _dispatch_method_label(group)
        lines.append(
            f"  group {group.group_id}: {route}, method={method}, {group.batch_kind}, "
            f"mode={group.mode}, size={group.size}, nx={group.nx}, "
            f"padding={padding_text}, geometry={geometry}, indices={group.pool_indices}"
        )

    lines.append("padding:")
    for padding_record in report.padding:
        lines.append(
            f"  group {padding_record.group_id}: row_nx={padding_record.row_nx}, "
            f"padded_nx={padding_record.padded_nx}, "
            f"padded_compartments={padding_record.padded_compartments}, "
            f"padded_fraction={padding_record.padded_fraction:.3f}"
        )
    lines.append("prepare:")
    for prepared in report.preparations:
        lines.append(
            f"  group {prepared.group_id}: mode={prepared.mode}, size={prepared.size}, "
            f"nx={prepared.nx}, stimulations={prepared.extracellular_stimulation_count}, "
            f"x_positions={prepared.x_positions_shape}, y={prepared.y_shape}, "
            f"z={prepared.z_shape}, representative={prepared.representative_index}"
        )
    lines.append("membranes:")
    for source in report.membrane_sources:
        lines.append(
            f"  group {source.group_id}: unique={source.unique_membrane_count}, "
            f"kinds={source.kinds}, sources={source.source_count}, "
            f"cache={_cache_status_text(source.cache_statuses)}, "
            f"reasons={_short_tuple(source.cache_reasons, width=24)}, "
            f"keys={_short_tuple(source.cache_keys)}"
        )
    lines.append("lowering:")
    for lowering in report.lowerings:
        iinj_shape = _shape_text(lowering.dense_iinj_shape)
        vstim_shape = _shape_text(lowering.dense_vstim_shape)
        dense_vstim = "yes" if lowering.materializes_dense_vstim else "no"
        lines.append(
            f"  group {lowering.group_id}: route={lowering.route}, "
            f"Iinj={lowering.intracellular_format} shape={iinj_shape}, "
            f"Vext={lowering.extracellular_format} shape={vstim_shape}, "
            f"dense_Vext={dense_vstim}, observer={lowering.observer_format}, "
            f"recording={lowering.recording_mode}->{lowering.kernel_recording_mode}, "
            f"retained_vm_width={lowering.retained_vm_width}"
        )
    lines.append("probes:")
    for probes in report.probes:
        shape = _shape_text(probes.retained_shape)
        lines.append(
            f"  group {probes.group_id}: names={probes.observer_names}, "
            f"row_aware={probes.row_aware}, max_probes={probes.max_probe_count}, "
            f"retained_shape={shape}, retained_bytes={probes.retained_bytes}"
        )
    lines.append("memory:")
    for memory in report.memory:
        lines.append(
            f"  group {memory.group_id}: dtype={memory.dtype}, "
            f"state={_bytes_text(memory.state_bytes)}, "
            f"prepare={_bytes_text(memory.prepared_position_bytes)}, "
            f"Iinj={_bytes_text(memory.dense_iinj_bytes)}, "
            f"Vext={_bytes_text(memory.dense_vstim_bytes)}, "
            f"Vm={_bytes_text(memory.retained_vm_bytes)}, "
            f"observer={_bytes_text(memory.observer_bytes)}, "
            f"total={_bytes_text(memory.total_estimated_bytes)}, "
            f"retained={_bytes_text(memory.retained_public_bytes)}"
        )
    lines.append("kernel:")
    for kernel in report.kernels:
        solver = _solver_route_text(kernel.solver)
        chunking = (
            ""
            if kernel.time_chunk_steps is None
            else f", chunk_steps={kernel.time_chunk_steps}"
        )
        lines.append(
            f"  group {kernel.group_id}: route={kernel.route}, "
            f"kernel={kernel.kernel}, mode={kernel.cable_mode}"
            f", solver={solver}{chunking}"
        )
    lines.append("result assembly:")
    for assembly in report.result_assembly:
        lines.append(
            f"  group {assembly.group_id}: {assembly.record_kind}, "
            f"Vm={assembly.vm_output}, observations={assembly.observation_output}, "
            f"public={assembly.public_result}"
        )
    lines.append("assembly details:")
    for detail in report.assembly_details:
        lines.append(
            f"  group {detail.group_id}: rows={detail.row_count}, "
            f"vm_shape={_shape_text(detail.vm_shape)}, "
            f"observation_shape={_shape_text(detail.observation_shape)}, "
            f"observations_batched={detail.observations_are_batched}, "
            f"public_rows={detail.public_rows}"
        )
    return "\n".join(lines)


def print_simulation_inspection(
    report: SimulationInspection,
    file: TextIO | None = None,
    *,
    rich: bool | None = None,
) -> None:
    """Print the inspection report."""

    if rich is False or file is not None:
        print(format_simulation_inspection(report), file=file)
        return

    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table

    console = Console(width=120)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    policy = report.planning.execution_policy
    if policy is None:
        policy_text = "default"
    else:
        precision = "model"
        if policy.precision is not None:
            precision = policy.precision.solver_dtype
        policy_text = (
            f"runtime={policy.runtime.value}, device={policy.device.kind}, "
            f"precision={precision}"
        )
    summary.add_row("axons", str(report.planning.axon_count))
    summary.add_row("steps", str(report.planning.step_count))
    summary.add_row(
        "time",
        f"{report.planning.duration_ms:g} ms @ {report.planning.dt_ms:g} ms",
    )
    summary.add_row("policy", policy_text)

    dispatch = Table(title="Dispatch And Batch", show_lines=False)
    for column in ("group", "route", "method", "kind", "rows", "Nx", "padding", "indices"):
        dispatch.add_column(column, overflow="fold")
    for group in report.dispatch_groups:
        dispatch.add_row(
            str(group.group_id),
            "batch" if group.will_batch else "scalar",
            _dispatch_method_label(group),
            group.batch_kind,
            str(group.size),
            str(group.nx),
            "yes" if group.has_padding else "no",
            str(group.pool_indices),
        )

    lowering = Table(title="Prepare, Lowering, Kernel", show_lines=False)
    for column in (
        "group",
        "stimulations",
        "Iinj",
        "Vext",
        "observer",
        "recording",
        "kernel",
    ):
        lowering.add_column(column, overflow="fold")
    for prepared, lower, kernel in zip(
        report.preparations,
        report.lowerings,
        report.kernels,
        strict=True,
    ):
        lowering.add_row(
            str(lower.group_id),
            str(prepared.extracellular_stimulation_count),
            _rich_shape_label(
                _display_label(lower.intracellular_format),
                lower.dense_iinj_shape,
            ),
            _rich_shape_label(
                _display_label(lower.extracellular_format),
                lower.dense_vstim_shape,
            ),
            lower.observer_format,
            f"{lower.recording_mode}->{lower.kernel_recording_mode}",
            _display_label(kernel.kernel),
        )

    membranes = Table(title="Membrane Sources", show_lines=False)
    for column in ("group", "unique", "kinds", "sources", "cache", "reason", "keys"):
        membranes.add_column(column, overflow="fold")
    for source in report.membrane_sources:
        membranes.add_row(
            str(source.group_id),
            str(source.unique_membrane_count),
            ", ".join(source.kinds) if source.kinds else "none",
            str(source.source_count),
            _cache_status_text(source.cache_statuses),
            _short_tuple(source.cache_reasons, width=24),
            _short_tuple(source.cache_keys),
        )

    outputs = Table(title="Memory And Result Assembly", show_lines=False)
    for column in (
        "group",
        "public bytes",
        "total bytes",
        "Vm",
        "obs",
        "result",
    ):
        outputs.add_column(column, overflow="fold")
    for memory, detail, assembly in zip(
        report.memory,
        report.assembly_details,
        report.result_assembly,
        strict=True,
    ):
        outputs.add_row(
            str(memory.group_id),
            _bytes_text(memory.retained_public_bytes),
            _bytes_text(memory.total_estimated_bytes),
            _shape_text(detail.vm_shape),
            _shape_text(detail.observation_shape),
            _display_label(assembly.record_kind),
        )

    probes = Table(title="Padding And VmRaster Probes", show_lines=False)
    for column in ("group", "row Nx", "padded", "fraction", "observers", "packed"):
        probes.add_column(column, overflow="fold")
    for padding, probe in zip(report.padding, report.probes, strict=True):
        names = ", ".join(probe.observer_names) if probe.observer_names else "none"
        probes.add_row(
            str(padding.group_id),
            str(padding.row_nx),
            str(padding.padded_compartments),
            f"{padding.padded_fraction:.3f}",
            names,
            _shape_text(probe.retained_shape),
        )

    console.print(
        Panel(
            Group(summary, dispatch, membranes, lowering, outputs, probes),
            title="AxonScope solver pipeline inspection",
            expand=False,
        )
    )


def plot_simulation_inspection(report: SimulationInspection, ax: Any | None = None) -> Any:
    """Plot dispatch groups, spatial widths, retained Vm, and observations."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    labels = [str(group.group_id) for group in report.dispatch_groups]
    sizes = np.asarray([group.size for group in report.dispatch_groups], dtype=float)
    widths = np.asarray([group.nx for group in report.dispatch_groups], dtype=float)
    retained = np.asarray(
        [lowering.retained_vm_width for lowering in report.lowerings],
        dtype=float,
    )
    observation_slots = np.asarray(
        [
            0
            if detail.observation_shape is None
            else int(np.prod(detail.observation_shape[1:]))
            for detail in report.assembly_details
        ],
        dtype=float,
    )
    x = np.arange(len(labels))
    colors = [
        "tab:blue" if group.mode == "single" else "tab:orange"
        for group in report.dispatch_groups
    ]

    ax.bar(x - 0.22, sizes, width=0.22, color=colors, alpha=0.75, label="rows")
    ax.bar(x, retained, width=0.22, color="tab:green", alpha=0.75, label="Vm width")
    ax.bar(
        x + 0.22,
        observation_slots,
        width=0.22,
        color="tab:purple",
        alpha=0.75,
        label="obs slots",
    )
    ax.plot(x, widths, marker="o", color="black", linewidth=1.8, label="Nx")
    for index, group in enumerate(report.dispatch_groups):
        lower = report.lowerings[index]
        ax.text(
            index,
            max(
                sizes[index],
                widths[index],
                retained[index],
                observation_slots[index],
            )
            + 0.5,
            f"{group.batch_kind}\n{_display_label(lower.extracellular_format)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("dispatch groups")
    ax.set_xticks(x, labels)
    ax.set_xlabel("dispatch group")
    ax.set_ylabel("rows / width / Nx")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    return ax


def plot_simulation_inspection_details(
    report: SimulationInspection,
    axes: Sequence[Any] | None = None,
) -> tuple[Any, ...]:
    """Plot padding, memory, VmRaster probes, and result assembly."""

    import matplotlib.pyplot as plt

    if axes is None:
        _, axes_arr = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        axes_tuple = tuple(axes_arr.ravel())
    else:
        axes_tuple = tuple(axes)
        if len(axes_tuple) != 4:
            raise ValueError("plot_details expects four axes.")

    labels = [str(group.group_id) for group in report.dispatch_groups]
    x = np.arange(len(labels))

    padding_ax, memory_ax, probe_ax, assembly_ax = axes_tuple

    padded_values = [item.padded_compartments for item in report.padding]
    padding_colors = ["tab:red" if value else "0.65" for value in padded_values]
    padding_ax.bar(x, padded_values, color=padding_colors, alpha=0.8)
    for index, padding_record in enumerate(report.padding):
        padding_ax.text(
            index,
            padded_values[index] + 0.05,
            f"Nx {padding_record.row_nx}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=20,
        )
    padding_ax.set_title("padding")
    padding_ax.set_xticks(x, labels)
    padding_ax.set_xlabel("dispatch group")
    padding_ax.set_ylabel("compartments")
    padding_ax.grid(True, axis="y", alpha=0.3)

    bottom = np.zeros(len(labels), dtype=float)
    memory_parts = (
        ("state", [item.state_bytes for item in report.memory]),
        ("prepare", [item.prepared_position_bytes for item in report.memory]),
        ("Iinj", [item.dense_iinj_bytes for item in report.memory]),
        ("Vext", [item.dense_vstim_bytes for item in report.memory]),
        ("Vm", [item.retained_vm_bytes for item in report.memory]),
        ("observer", [item.observer_bytes for item in report.memory]),
    )
    memory_scale, memory_unit = _memory_plot_scale(memory_parts)
    for label, values in memory_parts:
        scaled = np.asarray(values, dtype=float) / memory_scale
        memory_ax.bar(x, scaled, bottom=bottom, label=label)
        bottom += scaled
    memory_ax.set_title("memory estimate")
    memory_ax.set_xticks(x, labels)
    memory_ax.set_xlabel("dispatch group")
    memory_ax.set_ylabel(memory_unit)
    memory_ax.grid(True, axis="y", alpha=0.3)
    memory_ax.legend(fontsize=8, frameon=False)

    probe_counts = [item.max_probe_count for item in report.probes]
    probe_ax.bar(x, probe_counts, color="tab:purple", alpha=0.75)
    for index, probe_record in enumerate(report.probes):
        names = (
            ", ".join(probe_record.observer_names)
            if probe_record.observer_names
            else "none"
        )
        probe_ax.text(
            index,
            probe_record.max_probe_count + 0.05,
            names,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=20,
        )
    probe_ax.set_title("VmRaster probes")
    probe_ax.set_xticks(x, labels)
    probe_ax.set_xlabel("dispatch group")
    probe_ax.set_ylabel("max probes")
    probe_ax.set_ylim(0, max(probe_counts + [1]) + 0.5)
    probe_ax.grid(True, axis="y", alpha=0.3)

    vm_widths = [
        0 if detail.vm_shape is None else int(detail.vm_shape[-1])
        for detail in report.assembly_details
    ]
    observation_slots = [
        0 if detail.observation_shape is None else int(np.prod(detail.observation_shape[1:]))
        for detail in report.assembly_details
    ]
    assembly_ax.bar(x - 0.18, vm_widths, width=0.36, label="Vm width")
    assembly_ax.bar(x + 0.18, observation_slots, width=0.36, label="obs slots")
    for index, assembly in enumerate(report.result_assembly):
        assembly_ax.text(
            index,
            max(vm_widths[index], observation_slots[index]) + 0.05,
            assembly.record_kind,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=20,
        )
    assembly_ax.set_title("result assembly")
    assembly_ax.set_xticks(x, labels)
    assembly_ax.set_xlabel("dispatch group")
    assembly_ax.set_ylabel("columns / packed slots")
    assembly_ax.grid(True, axis="y", alpha=0.3)
    assembly_ax.legend(fontsize=8, frameon=False)

    return axes_tuple


def _shape_text(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "-"
    return "x".join(str(dim) for dim in shape)


def _cache_status_text(statuses: tuple[str, ...]) -> str:
    if not statuses:
        return "none"
    counts = {status: statuses.count(status) for status in sorted(set(statuses))}
    return ", ".join(f"{status}={count}" for status, count in counts.items())


def _short_tuple(values: tuple[str, ...], *, width: int = 8, limit: int = 3) -> str:
    if not values:
        return "none"
    shortened = tuple(value[:width] for value in values[:limit])
    suffix = "" if len(values) <= limit else f", +{len(values) - limit}"
    return ", ".join(shortened) + suffix


def _dispatch_method_label(group: Any) -> str:
    prefix = "batch" if group.geometry_shared else "parameter-batch"
    return f"{prefix}-{group.mode}-cable"


_DISPLAY_LABELS = {
    "callable_or_precomputed_per_axon": "callable/precomputed",
    "callable_per_axon": "callable",
    "dense": "dense",
    "compact dispatch cohort": "compact cohort",
    "dispatch row records": "row records",
    "DoubleCableBatchKernel": "double-cable batch",
    "factorized_footprint": "factorized footprint",
    "unsupported row": "unsupported",
    "SingleCableVStimBatchKernel": "single-cable Vstim batch",
    "sparse_current_clamp": "sparse clamp",
    "zero_no_extracellular_stimulation": "zero stimulation",
    "zero_no_intracellular_context": "zero clamp",
}


def _display_label(value: str) -> str:
    return _DISPLAY_LABELS.get(value, value)


def _rich_shape_label(kind: str, shape: tuple[int, ...] | None) -> str:
    shape_text = _shape_text(shape)
    if shape_text == "-":
        return kind
    return f"{kind} {shape_text}"


def _memory_plot_scale(parts: Sequence[tuple[str, Sequence[int]]]) -> tuple[float, str]:
    maximum = max(
        (float(value) for _, values in parts for value in values),
        default=0.0,
    )
    if maximum >= 1024**2:
        return float(1024**2), "MiB"
    if maximum >= 1024:
        return 1024.0, "KiB"
    return 1.0, "bytes"


def _bytes_text(value: int) -> str:
    if value == 0:
        return "0"
    if value < 1024:
        return f"{value} B"
    if value < 1024**2:
        return f"{value / 1024:.2f} KiB"
    return f"{value / (1024**2):.3f} MiB"


def _solver_route_text(solver: Any) -> str:
    route = "auto" if solver.runtime_route is None else str(solver.runtime_route)
    requested = solver.requested
    parts = [route]
    if requested is not None and str(requested) != route:
        parts.append(f"requested={requested}")
    if solver.internal:
        parts.append("internal")
    if solver.options:
        option_text = ",".join(f"{key}={value}" for key, value in solver.options)
        parts.append(option_text)
    return " ".join(parts)


__all__ = [
    "format_simulation_inspection",
    "plot_simulation_inspection",
    "plot_simulation_inspection_details",
    "print_simulation_inspection",
]
