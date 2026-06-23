"""Printable host-side inspection for the solver pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence, TextIO

import numpy as np

from axonscope.analysis.definitions import Activation, ConductionBlock, Latency
from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.backends.jax.recording import batch_options_from_recording
from axonscope.dispatcher.plan import DispatchGroup, build_dispatch_plan
from axonscope.performance import ExecutionPolicy
from axonscope.population import AxonPopulation
from axonscope.preparation.cohort import PreparedCohort
from axonscope.recording import Recording
from axonscope.solvers.options import (
    BatchOptions,
    BatchRecording,
    resolve_double_cable_block_solver,
)
from axonscope.stimulation import (
    IntracellularCurrentClamp,
)
from axonscope.timebase import simulation_step_count
from axonscope.utils import units


@dataclass(frozen=True)
class PlanningInspection:
    """High-level simulation planning summary."""

    axon_count: int
    duration_ms: float
    dt_ms: float
    step_count: int
    execution_policy: ExecutionPolicy | None


@dataclass(frozen=True)
class DispatchGroupInspection:
    """One dispatch/batch group selected by planning."""

    group_id: int
    pool_indices: tuple[int, ...]
    mode: str
    size: int
    nx: int
    batch_kind: str
    geometry_shared: bool
    has_padding: bool
    will_batch: bool


@dataclass(frozen=True)
class PaddingInspection:
    """Spatial padding expected for one dispatch group."""

    group_id: int
    row_nx: tuple[int, ...]
    padded_nx: int
    padded_compartments: int
    padded_fraction: float


@dataclass(frozen=True)
class PreparationInspection:
    """Host-side prepared cohort summary for one dispatch group."""

    group_id: int
    mode: str
    size: int
    nx: int
    context_count: int
    x_positions_shape: tuple[int, ...]
    y_shape: tuple[int, ...]
    z_shape: tuple[int, ...]
    representative_index: int | None


@dataclass(frozen=True)
class LoweringInspection:
    """Predicted host/input lowering for one dispatch group."""

    group_id: int
    route: str
    intracellular_format: str
    extracellular_format: str
    observer_format: str
    recording_mode: str
    kernel_recording_mode: str
    retained_vm_width: int
    dense_iinj_shape: tuple[int, ...] | None
    dense_vstim_shape: tuple[int, ...] | None
    materializes_dense_vstim: bool


@dataclass(frozen=True)
class ProbeInspection:
    """Predicted VmRaster probe lowering for one dispatch group."""

    group_id: int
    observer_names: tuple[str, ...]
    thresholds_mV: tuple[float, ...]
    row_aware: bool
    probe_indices_by_row: tuple[tuple[tuple[int, ...], ...], ...]
    row_probe_counts: tuple[tuple[int, ...], ...]
    max_probe_count: int
    packed_shape: tuple[int, ...] | None
    packed_bytes: int


@dataclass(frozen=True)
class MemoryInspection:
    """Estimated array pressure for one dispatch group."""

    group_id: int
    dtype: str
    state_bytes: int
    prepared_position_bytes: int
    dense_iinj_bytes: int
    dense_vstim_bytes: int
    retained_vm_bytes: int
    vm_raster_bytes: int
    total_estimated_bytes: int
    retained_public_bytes: int

    @property
    def total_estimated_mib(self) -> float:
        return self.total_estimated_bytes / (1024**2)

    @property
    def retained_public_mib(self) -> float:
        return self.retained_public_bytes / (1024**2)


@dataclass(frozen=True)
class KernelInspection:
    """Predicted backend kernel route for one dispatch group."""

    group_id: int
    route: str
    kernel: str
    cable_mode: str
    double_cable_block_solver: str | None
    time_chunk_steps: int | None


@dataclass(frozen=True)
class ResultAssemblyInspection:
    """Predicted solver-output to public-result assembly for one group."""

    group_id: int
    record_kind: str
    vm_output: str
    observation_output: str
    public_result: str


@dataclass(frozen=True)
class AssemblyDetailInspection:
    """Shape-level public result assembly details for one group."""

    group_id: int
    row_count: int
    vm_shape: tuple[int, ...] | None
    observation_shape: tuple[int, ...] | None
    observations_are_batched: bool
    public_rows: int


@dataclass(frozen=True)
class SimulationInspection:
    """Printable inspection report for the solver pipeline."""

    planning: PlanningInspection
    dispatch_groups: tuple[DispatchGroupInspection, ...]
    padding: tuple[PaddingInspection, ...]
    preparations: tuple[PreparationInspection, ...]
    lowerings: tuple[LoweringInspection, ...]
    probes: tuple[ProbeInspection, ...]
    memory: tuple[MemoryInspection, ...]
    kernels: tuple[KernelInspection, ...]
    result_assembly: tuple[ResultAssemblyInspection, ...]
    assembly_details: tuple[AssemblyDetailInspection, ...]

    def format(self) -> str:
        """Return a compact multiline text report."""

        policy = self.planning.execution_policy
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
                f"  axons={self.planning.axon_count}, Nt={self.planning.step_count}, "
                f"duration={self.planning.duration_ms:g} ms, dt={self.planning.dt_ms:g} ms"
            ),
            f"  execution_policy={policy_text}",
            "dispatch/batch:",
        ]
        for group in self.dispatch_groups:
            route = "batch" if group.will_batch else "scalar"
            padding = "yes" if group.has_padding else "no"
            geometry = "shared" if group.geometry_shared else "parameterized"
            lines.append(
                f"  group {group.group_id}: {route}, {group.batch_kind}, "
                f"mode={group.mode}, size={group.size}, nx={group.nx}, "
                f"padding={padding}, geometry={geometry}, indices={group.pool_indices}"
            )

        lines.append("padding:")
        for padding in self.padding:
            lines.append(
                f"  group {padding.group_id}: row_nx={padding.row_nx}, "
                f"padded_nx={padding.padded_nx}, "
                f"padded_compartments={padding.padded_compartments}, "
                f"padded_fraction={padding.padded_fraction:.3f}"
            )
        lines.append("prepare:")
        for prepared in self.preparations:
            lines.append(
                f"  group {prepared.group_id}: mode={prepared.mode}, size={prepared.size}, "
                f"nx={prepared.nx}, contexts={prepared.context_count}, "
                f"x_positions={prepared.x_positions_shape}, y={prepared.y_shape}, "
                f"z={prepared.z_shape}, representative={prepared.representative_index}"
            )
        lines.append("lowering:")
        for lowering in self.lowerings:
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
        for probes in self.probes:
            shape = _shape_text(probes.packed_shape)
            lines.append(
                f"  group {probes.group_id}: names={probes.observer_names}, "
                f"row_aware={probes.row_aware}, max_probes={probes.max_probe_count}, "
                f"packed_shape={shape}, packed_bytes={probes.packed_bytes}"
            )
        lines.append("memory:")
        for memory in self.memory:
            lines.append(
                f"  group {memory.group_id}: dtype={memory.dtype}, "
                f"state={_bytes_text(memory.state_bytes)}, "
                f"prepare={_bytes_text(memory.prepared_position_bytes)}, "
                f"Iinj={_bytes_text(memory.dense_iinj_bytes)}, "
                f"Vext={_bytes_text(memory.dense_vstim_bytes)}, "
                f"Vm={_bytes_text(memory.retained_vm_bytes)}, "
                f"VmRaster={_bytes_text(memory.vm_raster_bytes)}, "
                f"total={_bytes_text(memory.total_estimated_bytes)}, "
                f"retained={_bytes_text(memory.retained_public_bytes)}"
            )
        lines.append("kernel:")
        for kernel in self.kernels:
            block_solver = (
                ""
                if kernel.double_cable_block_solver is None
                else f", block_solver={kernel.double_cable_block_solver}"
            )
            chunking = (
                ""
                if kernel.time_chunk_steps is None
                else f", chunk_steps={kernel.time_chunk_steps}"
            )
            lines.append(
                f"  group {kernel.group_id}: route={kernel.route}, "
                f"kernel={kernel.kernel}, mode={kernel.cable_mode}"
                f"{block_solver}{chunking}"
            )
        lines.append("result assembly:")
        for assembly in self.result_assembly:
            lines.append(
                f"  group {assembly.group_id}: {assembly.record_kind}, "
                f"Vm={assembly.vm_output}, observations={assembly.observation_output}, "
                f"public={assembly.public_result}"
            )
        lines.append("assembly details:")
        for detail in self.assembly_details:
            lines.append(
                f"  group {detail.group_id}: rows={detail.row_count}, "
                f"vm_shape={_shape_text(detail.vm_shape)}, "
                f"observation_shape={_shape_text(detail.observation_shape)}, "
                f"observations_batched={detail.observations_are_batched}, "
                f"public_rows={detail.public_rows}"
            )
        return "\n".join(lines)

    def print(self, file: TextIO | None = None, *, rich: bool | None = None) -> None:
        """Print the inspection report.

        Terminal output uses Rich tables by default. Passing ``file`` keeps the
        stable plain-text representation for logs, tests, and redirected output.
        """

        if rich is False or file is not None:
            print(self.format(), file=file)
            return

        from rich.console import Console, Group
        from rich.panel import Panel
        from rich.table import Table

        console = Console(width=120)

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column()
        policy = self.planning.execution_policy
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
        summary.add_row("axons", str(self.planning.axon_count))
        summary.add_row("steps", str(self.planning.step_count))
        summary.add_row(
            "time",
            f"{self.planning.duration_ms:g} ms @ {self.planning.dt_ms:g} ms",
        )
        summary.add_row("policy", policy_text)

        dispatch = Table(title="Dispatch And Batch", show_lines=False)
        for column in ("group", "route", "kind", "mode", "rows", "Nx", "padding", "indices"):
            dispatch.add_column(column, overflow="fold")
        for group in self.dispatch_groups:
            dispatch.add_row(
                str(group.group_id),
                "batch" if group.will_batch else "scalar",
                group.batch_kind,
                group.mode,
                str(group.size),
                str(group.nx),
                "yes" if group.has_padding else "no",
                str(group.pool_indices),
            )

        lowering = Table(title="Prepare, Lowering, Kernel", show_lines=False)
        for column in (
            "group",
            "contexts",
            "Iinj",
            "Vext",
            "observer",
            "recording",
            "kernel",
        ):
            lowering.add_column(column, overflow="fold")
        for prepared, lower, kernel in zip(
            self.preparations,
            self.lowerings,
            self.kernels,
            strict=True,
        ):
            lowering.add_row(
                str(lower.group_id),
                str(prepared.context_count),
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
            self.memory,
            self.assembly_details,
            self.result_assembly,
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
        for padding, probe in zip(self.padding, self.probes, strict=True):
            names = ", ".join(probe.observer_names) if probe.observer_names else "none"
            probes.add_row(
                str(padding.group_id),
                str(padding.row_nx),
                str(padding.padded_compartments),
                f"{padding.padded_fraction:.3f}",
                names,
                _shape_text(probe.packed_shape),
            )

        console.print(
            Panel(
                Group(summary, dispatch, lowering, outputs, probes),
                title="AxonScope solver pipeline inspection",
                expand=False,
            )
        )

    def plot(self, ax: Any | None = None) -> Any:
        """Plot dispatch groups, spatial widths, retained Vm, and observations."""

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        labels = [str(group.group_id) for group in self.dispatch_groups]
        sizes = np.asarray([group.size for group in self.dispatch_groups], dtype=float)
        widths = np.asarray([group.nx for group in self.dispatch_groups], dtype=float)
        retained = np.asarray(
            [lowering.retained_vm_width for lowering in self.lowerings],
            dtype=float,
        )
        observation_slots = np.asarray(
            [
                0
                if detail.observation_shape is None
                else int(np.prod(detail.observation_shape[1:]))
                for detail in self.assembly_details
            ],
            dtype=float,
        )
        x = np.arange(len(labels))
        colors = [
            "tab:blue" if group.mode == "single" else "tab:orange"
            for group in self.dispatch_groups
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
        for index, group in enumerate(self.dispatch_groups):
            lower = self.lowerings[index]
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

    def plot_details(self, axes: Sequence[Any] | None = None) -> tuple[Any, ...]:
        """Plot padding, memory, VmRaster probes, and result assembly."""

        import matplotlib.pyplot as plt

        if axes is None:
            _, axes_arr = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
            axes_tuple = tuple(axes_arr.ravel())
        else:
            axes_tuple = tuple(axes)
            if len(axes_tuple) != 4:
                raise ValueError("plot_details expects four axes.")

        labels = [str(group.group_id) for group in self.dispatch_groups]
        x = np.arange(len(labels))

        padding_ax, memory_ax, probe_ax, assembly_ax = axes_tuple

        padded_values = [item.padded_compartments for item in self.padding]
        padding_colors = ["tab:red" if value else "0.65" for value in padded_values]
        padding_ax.bar(x, padded_values, color=padding_colors, alpha=0.8)
        for index, item in enumerate(self.padding):
            padding_ax.text(
                index,
                padded_values[index] + 0.05,
                f"Nx {item.row_nx}",
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
            ("state", [item.state_bytes for item in self.memory]),
            ("prepare", [item.prepared_position_bytes for item in self.memory]),
            ("Iinj", [item.dense_iinj_bytes for item in self.memory]),
            ("Vext", [item.dense_vstim_bytes for item in self.memory]),
            ("Vm", [item.retained_vm_bytes for item in self.memory]),
            ("VmRaster", [item.vm_raster_bytes for item in self.memory]),
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

        probe_counts = [item.max_probe_count for item in self.probes]
        probe_ax.bar(x, probe_counts, color="tab:purple", alpha=0.75)
        for index, item in enumerate(self.probes):
            names = ", ".join(item.observer_names) if item.observer_names else "none"
            probe_ax.text(
                index,
                item.max_probe_count + 0.05,
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
            for detail in self.assembly_details
        ]
        observation_slots = [
            0 if detail.observation_shape is None else int(np.prod(detail.observation_shape[1:]))
            for detail in self.assembly_details
        ]
        assembly_ax.bar(x - 0.18, vm_widths, width=0.36, label="Vm width")
        assembly_ax.bar(x + 0.18, observation_slots, width=0.36, label="obs slots")
        for index, assembly in enumerate(self.result_assembly):
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


def inspect_simulation(
    axons: Axon | AxonInstance | AxonPopulation | Iterable[Axon | AxonInstance],
    *,
    duration: Any,
    dt: Any,
    recording: Recording | None = None,
    batch_options: BatchOptions | None = None,
    observers: Sequence[Any] | None = None,
    execution_policy: ExecutionPolicy | None = None,
    print_summary: bool = False,
) -> SimulationInspection:
    """Inspect planning, preparation, lowering, kernels, and result assembly."""

    population = axons if isinstance(axons, AxonPopulation) else AxonPopulation(axons)
    instances = tuple(population.instances)
    duration_ms = units.to_ms(duration)
    dt_ms = units.to_ms(dt)
    step_count = simulation_step_count(duration_ms, dt_ms)
    plan = build_dispatch_plan(instances)
    resolved_batch_options = _inspection_batch_options(
        recording=recording,
        batch_options=batch_options,
    )
    observer_defs = tuple(observers) if observers is not None else None
    groups = tuple(plan.groups)
    dispatch_groups = tuple(_inspect_dispatch_group(group) for group in groups)
    padding = tuple(_inspect_padding(group) for group in groups)
    preparations = tuple(_inspect_prepared_group(group) for group in groups)
    lowerings = tuple(
        _inspect_lowering(
            group,
            step_count=step_count,
            batch_options=resolved_batch_options,
            observers=observer_defs,
        )
        for group in groups
    )
    probes = tuple(
        _inspect_probes(
            group,
            step_count=step_count,
            observers=observer_defs,
        )
        for group in groups
    )
    inspection = SimulationInspection(
        planning=PlanningInspection(
            axon_count=len(instances),
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            step_count=step_count,
            execution_policy=execution_policy,
        ),
        dispatch_groups=dispatch_groups,
        padding=padding,
        preparations=preparations,
        lowerings=lowerings,
        probes=probes,
        memory=tuple(
            _inspect_memory(
                group,
                step_count=step_count,
                lowering=lowering,
                probes=probe,
                execution_policy=execution_policy,
            )
            for group, lowering, probe in zip(groups, lowerings, probes, strict=True)
        ),
        kernels=tuple(
            _inspect_kernel(
                group,
                batch_options=resolved_batch_options,
                execution_policy=execution_policy,
            )
            for group in groups
        ),
        result_assembly=tuple(
            _inspect_result_assembly(
                group,
                step_count=step_count,
                batch_options=resolved_batch_options,
                observers=observer_defs,
            )
            for group in groups
        ),
        assembly_details=tuple(
            _inspect_assembly_details(
                group,
                step_count=step_count,
                batch_options=resolved_batch_options,
                observers=observer_defs,
                probes=probe,
            )
            for group, probe in zip(groups, probes, strict=True)
        ),
    )
    if print_summary:
        inspection.print()
    return inspection


def _inspect_dispatch_group(group: DispatchGroup) -> DispatchGroupInspection:
    return DispatchGroupInspection(
        group_id=int(group.group_id),
        pool_indices=group.pool_indices,
        mode=str(group.mode),
        size=int(group.size),
        nx=int(group.nx),
        batch_kind=group.batch_kind,
        geometry_shared=bool(group.geometry_shared),
        has_padding=bool(group.has_padding),
        will_batch=_can_batch(group),
    )


def _inspect_padding(group: DispatchGroup) -> PaddingInspection:
    row_nx = tuple(int(item.solver_axon.n_compartments) for item in group.items)
    padded_nx = int(group.nx)
    padded_compartments = sum(max(0, padded_nx - nx) for nx in row_nx)
    denominator = max(1, int(group.size) * padded_nx)
    return PaddingInspection(
        group_id=int(group.group_id),
        row_nx=row_nx,
        padded_nx=padded_nx,
        padded_compartments=int(padded_compartments),
        padded_fraction=float(padded_compartments) / float(denominator),
    )


def _inspect_prepared_group(group: DispatchGroup) -> PreparationInspection:
    cohort = PreparedCohort.from_dispatch_group(group)
    representative_index = None
    for item in group.items:
        if item.simulation is cohort.representative:
            representative_index = int(item.index)
            break
    return PreparationInspection(
        group_id=int(cohort.group_id),
        mode=cohort.mode,
        size=int(cohort.size),
        nx=int(cohort.nx),
        context_count=int(cohort.context_count),
        x_positions_shape=tuple(int(value) for value in cohort.x_positions_m.shape),
        y_shape=tuple(int(value) for value in cohort.axon_y_um.shape),
        z_shape=tuple(int(value) for value in cohort.axon_z_um.shape),
        representative_index=representative_index,
    )


def _can_batch(group: DispatchGroup) -> bool:
    return group.size >= 2 and group.mode in {"single", "double"}


def _inspection_batch_options(
    *,
    recording: Recording | None,
    batch_options: BatchOptions | None,
) -> BatchOptions:
    options = BatchOptions.full() if batch_options is None else batch_options
    lowered = batch_options_from_recording(recording, batch_options=options)
    return options if lowered is None else lowered


def _kernel_batch_options(
    group: DispatchGroup,
    options: BatchOptions,
    *,
    observers: tuple[Any, ...] | None,
) -> BatchOptions:
    if not group.has_padding:
        return options
    if options.recording.mode == "none" and observers is not None:
        return options
    return replace(options, recording=BatchRecording.full())


def _inspect_lowering(
    group: DispatchGroup,
    *,
    step_count: int,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None,
) -> LoweringInspection:
    vm_raster_supported = _observers_are_vm_raster_compatible(observers)
    if not _can_batch(group):
        retained_width = group.nx if batch_options.recording.mode != "none" else 0
        if observers is None:
            observer_format = "none"
        elif batch_options.recording.mode == "none" and vm_raster_supported:
            observer_format = "vm_raster"
        elif batch_options.recording.mode == "none":
            observer_format = "unsupported_observer_only"
        else:
            observer_format = "posthoc_from_recorded_vm"
        return LoweringInspection(
            group_id=int(group.group_id),
            route="scalar",
            intracellular_format="callable_per_axon",
            extracellular_format="callable_or_precomputed_per_axon",
            observer_format=observer_format,
            recording_mode=batch_options.recording.mode,
            kernel_recording_mode=batch_options.recording.mode,
            retained_vm_width=int(retained_width),
            dense_iinj_shape=None,
            dense_vstim_shape=None,
            materializes_dense_vstim=False,
        )

    cohort = PreparedCohort.from_dispatch_group(group)
    kernel_options = _kernel_batch_options(group, batch_options, observers=observers)
    observer_plan = (
        observers is not None
        and kernel_options.recording.mode == "none"
        and vm_raster_supported
    )
    sparse_intracellular = (
        group.mode == "single"
        and observer_plan
        and _can_build_sparse_intracellular_from_clamps(cohort.axons)
    )
    has_intracellular = _has_intracellular_contexts(cohort)
    if sparse_intracellular:
        intracellular_format = "sparse_current_clamp"
    elif has_intracellular:
        intracellular_format = "dense"
    else:
        intracellular_format = "zero_no_intracellular_context"

    if group.mode == "single":
        if sparse_intracellular and cohort.context_count == 0:
            extracellular_format = "zero_no_context"
        elif sparse_intracellular and observer_plan and _can_factorize_footprint_rows(cohort):
            extracellular_format = "factorized_footprint"
        else:
            extracellular_format = "dense"
    elif observer_plan and _can_factorize_footprint_rows(cohort):
        extracellular_format = "factorized_footprint"
    else:
        extracellular_format = "dense"

    dense_shape = (int(group.size), int(step_count), int(group.nx))
    dense_iinj_shape = None if intracellular_format != "dense" else dense_shape
    dense_vstim_shape = None if extracellular_format != "dense" else dense_shape
    if observer_plan:
        observer_format = "vm_raster"
    elif observers and kernel_options.recording.mode == "none":
        observer_format = "unsupported_observer_only"
    elif observers:
        observer_format = "posthoc_from_recorded_vm"
    else:
        observer_format = "none"

    return LoweringInspection(
        group_id=int(group.group_id),
        route="batch",
        intracellular_format=intracellular_format,
        extracellular_format=extracellular_format,
        observer_format=observer_format,
        recording_mode=batch_options.recording.mode,
        kernel_recording_mode=kernel_options.recording.mode,
        retained_vm_width=int(kernel_options.recording.width_for(group.nx)),
        dense_iinj_shape=dense_iinj_shape,
        dense_vstim_shape=dense_vstim_shape,
        materializes_dense_vstim=dense_vstim_shape is not None,
    )


def _inspect_probes(
    group: DispatchGroup,
    *,
    step_count: int,
    observers: tuple[Any, ...] | None,
) -> ProbeInspection:
    definitions = _vm_raster_definitions(observers)
    if not definitions:
        return ProbeInspection(
            group_id=int(group.group_id),
            observer_names=(),
            thresholds_mV=(),
            row_aware=_can_batch(group),
            probe_indices_by_row=(),
            row_probe_counts=(),
            max_probe_count=0,
            packed_shape=None,
            packed_bytes=0,
        )

    names = tuple(str(definition.name) for definition in definitions)
    thresholds = tuple(units.to_mV(definition.threshold) for definition in definitions)
    by_row: list[tuple[tuple[int, ...], ...]] = []
    counts: list[tuple[int, ...]] = []
    max_probe_count = 0
    for item in group.items:
        positions_um = np.asarray(item.solver_axon.x_um, dtype=float)
        original_indices = np.arange(positions_um.shape[0], dtype=np.int32)
        row_indices: list[tuple[int, ...]] = []
        row_counts: list[int] = []
        for definition in definitions:
            selected = definition.target.columns(
                positions_um=positions_um,
                original_indices=original_indices,
            )
            original_selected = tuple(int(original_indices[index]) for index in selected)
            row_indices.append(original_selected)
            row_counts.append(len(original_selected))
            max_probe_count = max(max_probe_count, len(original_selected))
        by_row.append(tuple(row_indices))
        counts.append(tuple(row_counts))

    word_count = (int(step_count) + 31) // 32
    packed_shape = (int(group.size), len(definitions), int(max_probe_count), word_count)
    packed_bytes = int(np.prod(packed_shape)) * np.dtype(np.uint32).itemsize
    return ProbeInspection(
        group_id=int(group.group_id),
        observer_names=names,
        thresholds_mV=thresholds,
        row_aware=_can_batch(group),
        probe_indices_by_row=tuple(by_row),
        row_probe_counts=tuple(counts),
        max_probe_count=int(max_probe_count),
        packed_shape=packed_shape,
        packed_bytes=packed_bytes,
    )


def _inspect_memory(
    group: DispatchGroup,
    *,
    step_count: int,
    lowering: LoweringInspection,
    probes: ProbeInspection,
    execution_policy: ExecutionPolicy | None,
) -> MemoryInspection:
    dtype = _inspection_dtype(group, execution_policy)
    itemsize = int(dtype.itemsize)
    state_bytes = int(group.size) * int(group.nx) * itemsize
    prepared_position_bytes = int(group.size) * int(group.nx) * itemsize
    dense_iinj_bytes = _shape_nbytes(lowering.dense_iinj_shape, dtype)
    dense_vstim_bytes = _shape_nbytes(lowering.dense_vstim_shape, dtype)
    retained_vm_bytes = (
        int(group.size) * int(step_count) * int(lowering.retained_vm_width) * itemsize
    )
    vm_raster_bytes = int(probes.packed_bytes) if lowering.observer_format == "vm_raster" else 0
    total_estimated = (
        state_bytes
        + prepared_position_bytes
        + dense_iinj_bytes
        + dense_vstim_bytes
        + retained_vm_bytes
        + vm_raster_bytes
    )
    retained_public = retained_vm_bytes + vm_raster_bytes
    return MemoryInspection(
        group_id=int(group.group_id),
        dtype=str(dtype),
        state_bytes=state_bytes,
        prepared_position_bytes=prepared_position_bytes,
        dense_iinj_bytes=dense_iinj_bytes,
        dense_vstim_bytes=dense_vstim_bytes,
        retained_vm_bytes=retained_vm_bytes,
        vm_raster_bytes=vm_raster_bytes,
        total_estimated_bytes=int(total_estimated),
        retained_public_bytes=int(retained_public),
    )


def _inspect_kernel(
    group: DispatchGroup,
    *,
    batch_options: BatchOptions,
    execution_policy: ExecutionPolicy | None,
) -> KernelInspection:
    if not _can_batch(group):
        kernel = "DoubleCableKernel" if group.mode == "double" else "SingleCableKernel"
        return KernelInspection(
            group_id=int(group.group_id),
            route="scalar",
            kernel=kernel,
            cable_mode=str(group.mode),
            double_cable_block_solver=None,
            time_chunk_steps=None,
        )

    if group.mode == "double":
        kernel = "DoubleCableBatchKernel"
        block_solver = _inspect_double_cable_block_solver(
            batch_options.double_cable_block_solver,
            execution_policy=execution_policy,
        )
    else:
        kernel = "SingleCableVStimBatchKernel"
        block_solver = None
    return KernelInspection(
        group_id=int(group.group_id),
        route="batch",
        kernel=kernel,
        cable_mode=str(group.mode),
        double_cable_block_solver=block_solver,
        time_chunk_steps=batch_options.time_chunk_steps,
    )


def _inspect_result_assembly(
    group: DispatchGroup,
    *,
    step_count: int,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None,
) -> ResultAssemblyInspection:
    if not _can_batch(group):
        vm_output = (
            "none"
            if batch_options.recording.mode == "none"
            else f"Vm[Nt={step_count}, Nx={group.nx}]"
        )
        if observers is None:
            observations = "none"
        elif (
            batch_options.recording.mode == "none"
            and _observers_are_vm_raster_compatible(observers)
        ):
            observations = 'observations["vm_raster"]'
        elif batch_options.recording.mode == "none":
            observations = "unsupported_observer_only"
        else:
            observations = "posthoc_from_recorded_vm"
        return ResultAssemblyInspection(
            group_id=int(group.group_id),
            record_kind="scalar fallback row",
            vm_output=vm_output,
            observation_output=observations,
            public_result="AxonSimulationResult row",
        )

    kernel_options = _kernel_batch_options(group, batch_options, observers=observers)
    width = int(kernel_options.recording.width_for(group.nx))
    observer_only = (
        observers is not None
        and kernel_options.recording.mode == "none"
        and _observers_are_vm_raster_compatible(observers)
    )
    if observer_only:
        return ResultAssemblyInspection(
            group_id=int(group.group_id),
            record_kind="DispatchCohortResult",
            vm_output="none",
            observation_output='observations["vm_raster"]',
            public_result="compact AxonSimulationResult cohort",
        )

    if observers and kernel_options.recording.mode == "none":
        observation_output = "unsupported_observer_only"
    elif observers:
        observation_output = "posthoc_from_recorded_vm"
    else:
        observation_output = "none"
    return ResultAssemblyInspection(
        group_id=int(group.group_id),
        record_kind="DispatchResult rows",
        vm_output=f"Vm[B={group.size}, Nt={step_count}, width={width}]",
        observation_output=observation_output,
        public_result="AxonSimulationResult rows",
    )


def _inspect_assembly_details(
    group: DispatchGroup,
    *,
    step_count: int,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None,
    probes: ProbeInspection | None = None,
) -> AssemblyDetailInspection:
    kernel_options = _kernel_batch_options(group, batch_options, observers=observers)
    width = int(kernel_options.recording.width_for(group.nx))
    vm_shape = None
    if kernel_options.recording.mode != "none":
        if _can_batch(group):
            vm_shape = (int(group.size), int(step_count), width)
        else:
            vm_shape = (int(step_count), int(group.nx))
    probes = (
        _inspect_probes(group, step_count=step_count, observers=observers)
        if probes is None
        else probes
    )
    observation_shape = (
        probes.packed_shape
        if (
            probes.packed_shape is not None
            and kernel_options.recording.mode == "none"
            and _observers_are_vm_raster_compatible(observers)
        )
        else None
    )
    observations_are_batched = (
        observation_shape is not None and kernel_options.recording.mode == "none"
    )
    return AssemblyDetailInspection(
        group_id=int(group.group_id),
        row_count=int(group.size),
        vm_shape=vm_shape,
        observation_shape=observation_shape,
        observations_are_batched=bool(observations_are_batched),
        public_rows=1 if observations_are_batched and _can_batch(group) else int(group.size),
    )


def _inspect_double_cable_block_solver(
    solver: str,
    *,
    execution_policy: ExecutionPolicy | None,
) -> str:
    if solver != "auto":
        return solver
    platform = _execution_policy_platform(execution_policy)
    if platform is None:
        return "auto(default-backend)"
    return resolve_double_cable_block_solver("auto", platform=platform)


def _execution_policy_platform(policy: ExecutionPolicy | None) -> str | None:
    if policy is None:
        return None
    if policy.device.kind in {"cpu", "gpu"}:
        return policy.device.kind
    return None


def _inspection_dtype(
    group: DispatchGroup,
    execution_policy: ExecutionPolicy | None,
) -> np.dtype:
    if execution_policy is not None and execution_policy.precision is not None:
        return np.dtype(execution_policy.precision.solver_dtype)
    return np.dtype(group.items[0].solver_axon.dtype)


def _shape_nbytes(shape: tuple[int, ...] | None, dtype: np.dtype) -> int:
    if shape is None:
        return 0
    return int(np.prod(shape)) * int(dtype.itemsize)


def _vm_raster_definitions(observers: tuple[Any, ...] | None) -> tuple[Any, ...]:
    if observers is None:
        return ()
    return tuple(
        observer
        for observer in observers
        if isinstance(observer, (Activation, Latency, ConductionBlock))
    )


def _observers_are_vm_raster_compatible(observers: tuple[Any, ...] | None) -> bool:
    if observers is None:
        return False
    return bool(observers) and len(_vm_raster_definitions(observers)) == len(observers)


def _has_intracellular_contexts(cohort: PreparedCohort) -> bool:
    return any(getattr(axon, "intracellular_contexts", ()) for axon in cohort.axons)


def _can_build_sparse_intracellular_from_clamps(
    axons: Sequence[AxonInstance],
) -> bool:
    return all(
        isinstance(context, IntracellularCurrentClamp)
        for axon in axons
        for context in getattr(axon, "intracellular_contexts", ())
    )


def _can_factorize_footprint_rows(cohort: PreparedCohort) -> bool:
    rows = cohort.contexts
    if not rows or not any(rows) or any(len(row) != 1 for row in rows):
        return False
    for row in rows:
        context = row[0]
        if len(context.electrodes) != 1:
            return False
        electrode = context.electrodes[0]
        if not hasattr(context, "footprint_for_electrode"):
            return False
        if getattr(electrode, "stimulus", None) is None:
            return False
    return True


def _shape_text(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "-"
    return "x".join(str(dim) for dim in shape)


_DISPLAY_LABELS = {
    "callable_or_precomputed_per_axon": "callable/precomputed",
    "callable_per_axon": "callable",
    "dense": "dense",
    "DispatchCohortResult": "cohort result",
    "DispatchResult rows": "row results",
    "DoubleCableBatchKernel": "double-cable batch",
    "factorized_footprint": "factorized footprint",
    "scalar fallback row": "scalar fallback",
    "SingleCableKernel": "single-cable scalar",
    "SingleCableVStimBatchKernel": "single-cable Vstim batch",
    "sparse_current_clamp": "sparse clamp",
    "zero_no_context": "zero context",
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


__all__ = [
    "AssemblyDetailInspection",
    "DispatchGroupInspection",
    "KernelInspection",
    "LoweringInspection",
    "MemoryInspection",
    "PaddingInspection",
    "PlanningInspection",
    "ProbeInspection",
    "PreparationInspection",
    "ResultAssemblyInspection",
    "SimulationInspection",
    "inspect_simulation",
]
