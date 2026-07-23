"""Performance and memory-estimation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TextIO

import numpy as np

from axonfleet.axon_instance import AxonInstance, as_axon_instance
from axonfleet.axons.axon import Axon
from axonfleet.runtime.execution import (
    batch_options_from_recording,
    benchmark_lower_recording_options,
    benchmark_membrane_output_names,
    benchmark_observer_output_label,
    benchmark_plan_input_lowering,
    benchmark_vm_raster_definitions,
)
from axonfleet.dispatcher.plan import DispatchGroup, build_dispatch_plan
from axonfleet.population import AxonPopulation
from axonfleet.preparation.stimulation_rows import extracellular_stimulation_rows
from axonfleet.recording import Recording
from axonfleet.runtime.policy import (
    Device,
    ExecutionPolicy,
    PrecisionPolicy,
    RuntimeTarget,
    auto as runtime_auto,
    coerce_runtime,
)
from axonfleet.solvers import BatchOptions
from axonfleet.runtime.timebase import simulation_step_count
from axonfleet.utils import units


@dataclass(frozen=True)
class MemoryEstimateItem:
    """One array-like contribution to a simulation memory estimate."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    bytes: int
    role: str
    retained: bool
    note: str = ""

    @property
    def mib(self) -> float:
        """Size in MiB."""

        return self.bytes / (1024**2)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable row."""

        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "bytes": self.bytes,
            "mib": self.mib,
            "role": self.role,
            "retained": self.retained,
            "note": self.note,
        }


@dataclass(frozen=True)
class SimulationEstimateGroup:
    """One dispatch-group contribution to a public simulation estimate."""

    group_id: int
    pool_indices: tuple[int, ...]
    batch_kind: str
    mode: str
    size: int
    nx: int
    row_nx: tuple[int, ...]
    padded_compartments: int
    padded_fraction: float
    dtype: str
    recording_mode: str
    kernel_recording_mode: str
    retained_vm_width: int
    observer_output: str
    intracellular_format: str
    extracellular_format: str
    total_bytes: int
    retained_bytes: int

    @property
    def total_mib(self) -> float:
        """Estimated group memory pressure in MiB."""

        return self.total_bytes / (1024**2)

    @property
    def retained_mib(self) -> float:
        """Estimated retained public output for this group in MiB."""

        return self.retained_bytes / (1024**2)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable group estimate."""

        return {
            "group_id": self.group_id,
            "pool_indices": list(self.pool_indices),
            "batch_kind": self.batch_kind,
            "mode": self.mode,
            "size": self.size,
            "nx": self.nx,
            "row_nx": list(self.row_nx),
            "padded_compartments": self.padded_compartments,
            "padded_fraction": self.padded_fraction,
            "dtype": self.dtype,
            "recording_mode": self.recording_mode,
            "kernel_recording_mode": self.kernel_recording_mode,
            "retained_vm_width": self.retained_vm_width,
            "observer_output": self.observer_output,
            "intracellular_format": self.intracellular_format,
            "extracellular_format": self.extracellular_format,
            "total_bytes": self.total_bytes,
            "total_mib": self.total_mib,
            "retained_bytes": self.retained_bytes,
            "retained_mib": self.retained_mib,
        }


@dataclass(frozen=True)
class SimulationEstimate:
    """Memory and workload estimate for one public simulation definition."""

    axon_count: int
    step_count: int
    max_compartments: int
    duration_ms: float
    dt_ms: float
    runtime: RuntimeTarget
    device: Device
    precision: PrecisionPolicy
    recording_width_max: int
    items: tuple[MemoryEstimateItem, ...]
    groups: tuple[SimulationEstimateGroup, ...]
    warnings: tuple[str, ...]
    recommendations: tuple[str, ...]
    metadata: Mapping[str, Any]

    @property
    def total_bytes(self) -> int:
        """Sum all estimated array contributions."""

        return sum(item.bytes for item in self.items)

    @property
    def total_mib(self) -> float:
        """Total estimated size in MiB."""

        return self.total_bytes / (1024**2)

    @property
    def retained_bytes(self) -> int:
        """Bytes expected to remain visible in public outputs."""

        return sum(item.bytes for item in self.items if item.retained)

    @property
    def retained_mib(self) -> float:
        """Retained public output size in MiB."""

        return self.retained_bytes / (1024**2)

    def item(self, name: str) -> MemoryEstimateItem:
        """Return one estimate row by name."""

        for item in self.items:
            if item.name == name:
                return item
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable estimate."""

        return {
            "axon_count": self.axon_count,
            "step_count": self.step_count,
            "max_compartments": self.max_compartments,
            "duration_ms": self.duration_ms,
            "dt_ms": self.dt_ms,
            "runtime": self.runtime.value,
            "device": {"kind": self.device.kind, "index": self.device.index},
            "precision": {
                "state_dtype": self.precision.state_dtype,
                "solver_dtype": self.precision.solver_dtype,
                "accumulation_dtype": self.precision.accumulation_dtype,
            },
            "recording_width_max": self.recording_width_max,
            "total_bytes": self.total_bytes,
            "total_mib": self.total_mib,
            "retained_bytes": self.retained_bytes,
            "retained_mib": self.retained_mib,
            "items": [item.to_dict() for item in self.items],
            "groups": [group.to_dict() for group in self.groups],
            "warnings": list(self.warnings),
            "recommendations": list(self.recommendations),
            "metadata": dict(self.metadata),
        }

    def rows(self, *, section: str = "items") -> tuple[dict[str, Any], ...]:
        """Return table rows for an estimate section."""

        from axonfleet.performance.views import simulation_estimate_rows

        return simulation_estimate_rows(self, section=section)

    def to_dataframe(self, *, section: str = "items") -> Any:
        """Return one estimate section as a pandas DataFrame."""

        from axonfleet.performance.views import simulation_estimate_to_dataframe

        return simulation_estimate_to_dataframe(self, section=section)

    def format(self) -> str:
        """Format a readable table-like estimate report."""

        from axonfleet.performance.views import format_simulation_estimate

        return format_simulation_estimate(self)

    def print(self, file: TextIO | None = None, *, rich: bool | None = None) -> None:
        """Print the estimate report, using Rich tables for terminals."""

        from axonfleet.performance.views import print_simulation_estimate

        print_simulation_estimate(self, file=file, rich=rich)


def estimate_simulation(
    axons: Axon | AxonInstance | AxonPopulation | Iterable[Axon | AxonInstance],
    *,
    duration: Any,
    dt: Any,
    recording: Recording | None = None,
    batch_options: BatchOptions | None = None,
    observers: Sequence[Any] | None = None,
    population_lifecycle: bool | None = None,
    runtime: RuntimeTarget = runtime_auto,
    device: Device | None = None,
    precision: PrecisionPolicy | None = None,
    memory_budget_bytes: int | None = None,
) -> SimulationEstimate:
    """Estimate memory pressure for a public simulation definition."""

    population = axons if isinstance(axons, AxonPopulation) else AxonPopulation(axons)
    instances = tuple(population.instances)
    observer_defs = tuple(observers) if observers is not None else None
    if (
        recording is not None
        and not recording.voltage
        and not recording.wants_observables
        and not observer_defs
    ):
        raise NotImplementedError("Recording.none() requires solver-side observers.")
    duration_ms = units.to_ms(duration)
    dt_ms = units.to_ms(dt)
    step_count = simulation_step_count(duration_ms, dt_ms)
    resolved_precision = precision or _precision_from_instances(instances)
    resolved_runtime = coerce_runtime(runtime)
    resolved_device = Device.auto() if device is None else device
    max_nx = max(int(instance.axon.n_compartments) for instance in instances)
    axon_count = len(instances)
    is_population_run = (
        not population.is_single
        if population_lifecycle is None
        else bool(population_lifecycle)
    )
    resolved_batch_options = _estimate_batch_options(
        population=population,
        recording=recording,
        batch_options=batch_options,
    )
    plan = build_dispatch_plan(instances)
    estimate_groups = tuple(
        _estimate_dispatch_group(
            group,
            step_count=step_count,
            batch_options=resolved_batch_options,
            recording=recording,
            observers=observer_defs,
            precision=resolved_precision,
        )
        for group in plan.groups
    )
    recording_width_max = max((group.retained_vm_width for group in estimate_groups), default=0)
    intracellular_context_count = _intracellular_context_count(instances)
    max_intracellular_contexts = _max_intracellular_context_count(instances)
    has_intracellular_contexts = intracellular_context_count > 0
    sparse_intracellular_input = any(
        group.intracellular_format == "sparse_current_clamp"
        for group in estimate_groups
    )
    dense_intracellular_input = any(
        group.intracellular_format == "dense"
        for group in estimate_groups
    )
    skipped_dense_iinj_shape = (axon_count, step_count, max_nx)
    skipped_dense_iinj_nbytes = int(np.prod(skipped_dense_iinj_shape)) * int(
        np.dtype(resolved_precision.solver_dtype).itemsize
    )
    if not has_intracellular_contexts:
        intracellular_input_format = "zero_no_intracellular_context"
    elif sparse_intracellular_input:
        intracellular_input_format = "sparse_current_clamp"
    elif dense_intracellular_input:
        intracellular_input_format = "dense"
    else:
        intracellular_input_format = "none"

    extracellular_stimulation_count = _extracellular_stimulation_count(instances)
    extracellular_drive_count = _extracellular_drive_count(instances)
    stimulus_count = _unique_stimulus_count(instances)
    items = _aggregate_estimate_items(
        plan.groups,
        estimate_groups=estimate_groups,
        step_count=step_count,
        max_nx=max_nx,
        axon_count=axon_count,
        recording=recording,
        observers=observer_defs,
        precision=resolved_precision,
        extracellular_drive_count=extracellular_drive_count,
        stimulus_count=stimulus_count,
    )

    warnings, recommendations = _estimate_guidance(
        items,
        groups=estimate_groups,
        extracellular_stimulation_count=extracellular_stimulation_count,
        max_nx=max_nx,
        recording_width_max=recording_width_max,
        memory_budget_bytes=memory_budget_bytes,
        observers=observer_defs,
    )
    if not has_intracellular_contexts:
        recommendations = (
            *recommendations,
            "No intracellular contexts are present; dense zero "
            "Iinj[B,Nt,Nx] is excluded from the estimate "
            f"({skipped_dense_iinj_nbytes / (1024**2):.3f} MiB skipped).",
        )

    metadata: dict[str, Any] = {
        "extracellular_stimulation_count": extracellular_stimulation_count,
        "intracellular_context_count": intracellular_context_count,
        "max_intracellular_contexts": max_intracellular_contexts,
        "intracellular_input_format": intracellular_input_format,
        "extracellular_drive_count": extracellular_drive_count,
        "unique_stimulus_count": stimulus_count,
        "recording_policy": _recording_label(recording, batch_options),
        "population_lifecycle": is_population_run,
    }
    if not has_intracellular_contexts:
        metadata["skipped_dense_iinj_shape"] = list(skipped_dense_iinj_shape)
        metadata["skipped_dense_iinj_nbytes"] = skipped_dense_iinj_nbytes

    return SimulationEstimate(
        axon_count=axon_count,
        step_count=step_count,
        max_compartments=max_nx,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        runtime=resolved_runtime,
        device=resolved_device,
        precision=resolved_precision,
        recording_width_max=recording_width_max,
        items=tuple(items),
        groups=estimate_groups,
        warnings=warnings,
        recommendations=recommendations,
        metadata=metadata,
    )


def _estimate_batch_options(
    *,
    population: AxonPopulation,
    recording: Recording | None,
    batch_options: BatchOptions | None,
) -> BatchOptions:
    options = BatchOptions.full() if batch_options is None else batch_options
    lowered = batch_options_from_recording(recording, batch_options=options)
    return options if lowered is None else lowered


def _estimate_dispatch_group(
    group: DispatchGroup,
    *,
    step_count: int,
    batch_options: BatchOptions,
    recording: Recording | None,
    observers: tuple[Any, ...] | None,
    precision: PrecisionPolicy,
) -> SimulationEstimateGroup:
    dtype = np.dtype(precision.solver_dtype)
    state_dtype = np.dtype(precision.state_dtype)
    kernel_options = benchmark_lower_recording_options(
        group,
        batch_options,
        observers=observers,
    )
    observer_output = benchmark_observer_output_label(
        observers,
        recording_mode=kernel_options.recording.mode,
    )
    observer_plan = observer_output in {
        "activation",
        "first_crossing",
        "spike_summary",
        "spike_events",
        "vm_raster",
    }
    simulations = tuple(item.simulation for item in group.items)
    stimulation_rows = extracellular_stimulation_rows(simulations)
    has_extracellular = any(stimulation_rows)
    planned = benchmark_plan_input_lowering(
        group_mode=group.mode,
        axons=simulations,
        stimulation_rows=stimulation_rows,
        kernel_options=kernel_options,
        observers=observers,
    )
    intracellular_format = planned.intracellular_format
    extracellular_format = planned.extracellular_format

    row_nx = tuple(int(item.solver_axon.n_compartments) for item in group.items)
    padded_compartments = sum(max(0, int(group.nx) - nx) for nx in row_nx)
    padded_fraction = float(padded_compartments) / float(max(1, int(group.size) * int(group.nx)))
    retained_vm_width = int(batch_options.recording.width_for(group.nx))
    if batch_options.recording.mode == "none":
        retained_vm_width = 0
    state_bytes = int(group.size) * int(group.nx) * int(state_dtype.itemsize)
    prepared_position_bytes = int(group.size) * int(group.nx) * int(dtype.itemsize)
    dense_iinj_bytes = (
        int(group.size) * int(step_count) * int(group.nx) * int(dtype.itemsize)
        if intracellular_format == "dense"
        else 0
    )
    sparse_iinj_bytes = 0
    if intracellular_format == "sparse_current_clamp":
        max_slots = _max_intracellular_context_count(simulations)
        sparse_iinj_bytes = (
            int(group.size) * int(step_count) * int(max_slots) * int(dtype.itemsize)
            + int(group.size) * int(max_slots) * np.dtype("int32").itemsize
            + int(group.size) * int(max_slots) * np.dtype("bool").itemsize
        )
    dense_vstim_bytes = (
        int(group.size) * int(step_count) * int(group.nx) * int(dtype.itemsize)
        if extracellular_format == "dense"
        else 0
    )
    footprint_bytes = (
        _extracellular_drive_count(simulations) * int(group.nx) * int(dtype.itemsize)
        if has_extracellular
        else 0
    )
    factorized_current_bytes = (
        _extracellular_drive_count(simulations) * int(step_count) * int(dtype.itemsize)
        if extracellular_format == "factorized_footprint"
        else 0
    )
    retained_vm_bytes = (
        int(group.size) * int(step_count) * int(retained_vm_width) * int(dtype.itemsize)
    )
    observer_bytes = (
        _threshold_observer_nbytes(
            group,
            step_count=step_count,
            observers=observers,
            observer_output=observer_output,
        )
        if observer_plan
        else 0
    )
    observable_bytes = _observable_output_nbytes(
        group,
        recording=recording,
        step_count=step_count,
        dtype=dtype,
    )
    retained_bytes = retained_vm_bytes + observer_bytes + observable_bytes
    total_bytes = (
        state_bytes
        + prepared_position_bytes
        + dense_iinj_bytes
        + sparse_iinj_bytes
        + dense_vstim_bytes
        + footprint_bytes
        + factorized_current_bytes
        + retained_bytes
    )
    return SimulationEstimateGroup(
        group_id=int(group.group_id),
        pool_indices=group.pool_indices,
        batch_kind=group.batch_kind,
        mode=str(group.mode),
        size=int(group.size),
        nx=int(group.nx),
        row_nx=row_nx,
        padded_compartments=int(padded_compartments),
        padded_fraction=padded_fraction,
        dtype=str(dtype),
        recording_mode=batch_options.recording.mode,
        kernel_recording_mode=kernel_options.recording.mode,
        retained_vm_width=retained_vm_width,
        observer_output=observer_output,
        intracellular_format=intracellular_format,
        extracellular_format=extracellular_format,
        total_bytes=int(total_bytes),
        retained_bytes=int(retained_bytes),
    )


def _aggregate_estimate_items(
    groups: Sequence[DispatchGroup],
    *,
    estimate_groups: Sequence[SimulationEstimateGroup],
    step_count: int,
    max_nx: int,
    axon_count: int,
    recording: Recording | None,
    observers: tuple[Any, ...] | None,
    precision: PrecisionPolicy,
    extracellular_drive_count: int,
    stimulus_count: int,
) -> tuple[MemoryEstimateItem, ...]:
    dtype = np.dtype(precision.solver_dtype)
    state_dtype = np.dtype(precision.state_dtype)
    items: list[MemoryEstimateItem] = []
    group_by_id = {group.group_id: estimate for group, estimate in zip(groups, estimate_groups, strict=True)}
    state_bytes = sum(
        int(group.size) * int(group.nx) * int(state_dtype.itemsize)
        for group in groups
    )
    position_bytes = sum(
        int(group.size) * int(group.nx) * int(dtype.itemsize)
        for group in groups
    )
    items.append(
        _item_with_bytes(
            "state.vm",
            (axon_count, max_nx),
            state_dtype,
            bytes=state_bytes,
            role="solver_state",
            retained=False,
            note="current membrane voltage state, summed over dispatch-group padding",
        )
    )
    items.append(
        _item_with_bytes(
            "inputs.positions",
            (axon_count, max_nx),
            dtype,
            bytes=position_bytes,
            role="preparation",
            retained=False,
            note="batched intrinsic positions, summed over dispatch-group padding",
        )
    )

    dense_iinj_groups = [
        group for group in groups if group_by_id[group.group_id].intracellular_format == "dense"
    ]
    if dense_iinj_groups:
        bytes_value = sum(
            int(group.size) * int(step_count) * int(group.nx) * int(dtype.itemsize)
            for group in dense_iinj_groups
        )
        rows = sum(int(group.size) for group in dense_iinj_groups)
        items.append(
            _item_with_bytes(
                "inputs.intracellular_current_density",
                (rows, step_count, max(int(group.nx) for group in dense_iinj_groups)),
                dtype,
                bytes=bytes_value,
                role="kernel_input",
                retained=False,
                note="dense Iinj materialized for groups that need compartment input",
            )
        )

    sparse_groups = [
        group
        for group in groups
        if group_by_id[group.group_id].intracellular_format == "sparse_current_clamp"
    ]
    if sparse_groups:
        sparse_rows = sum(int(group.size) for group in sparse_groups)
        max_slots = max(
            _max_intracellular_context_count(tuple(item.simulation for item in group.items))
            for group in sparse_groups
        )
        items.append(
            _item(
                "inputs.intracellular_current_density_sparse",
                (sparse_rows, step_count, max_slots),
                dtype,
                role="kernel_input",
                retained=False,
                note="observer-only current-clamp path keeps Iinj sparse over compartments",
            )
        )
        items.append(
            _item(
                "inputs.intracellular_current_indices",
                (sparse_rows, max_slots),
                np.dtype("int32"),
                role="kernel_input",
                retained=False,
                note="target compartment per sparse current-clamp slot",
            )
        )
        items.append(
            _item(
                "inputs.intracellular_current_mask",
                (sparse_rows, max_slots),
                np.dtype("bool"),
                role="kernel_input",
                retained=False,
                note="valid sparse current-clamp slots",
            )
        )

    dense_vstim_groups = [
        group for group in groups if group_by_id[group.group_id].extracellular_format == "dense"
    ]
    if dense_vstim_groups:
        bytes_value = sum(
            int(group.size) * int(step_count) * int(group.nx) * int(dtype.itemsize)
            for group in dense_vstim_groups
        )
        rows = sum(int(group.size) for group in dense_vstim_groups)
        items.append(
            _item_with_bytes(
                "inputs.extracellular_potential_mid",
                (rows, step_count, max(int(group.nx) for group in dense_vstim_groups)),
                dtype,
                bytes=bytes_value,
                role="kernel_input",
                retained=False,
                note="dense Vstim materialized for groups that need a time-space field",
            )
        )
    if extracellular_drive_count:
        footprint_bytes = sum(
            _extracellular_drive_count(tuple(item.simulation for item in group.items))
            * int(group.nx)
            * int(dtype.itemsize)
            for group in groups
        )
        items.append(
            _item_with_bytes(
                "footprints.factorized_rows",
                (extracellular_drive_count, max_nx),
                dtype,
                bytes=footprint_bytes,
                role="factorized_reference",
                retained=False,
                note="spatial footprint storage without the time axis",
            )
        )
    if stimulus_count:
        items.append(
            _item(
                "stimuli.sampled_waveforms",
                (stimulus_count, step_count),
                dtype,
                role="factorized_reference",
                retained=False,
                note="temporal stimulus samples without the spatial axis",
            )
        )

    recording_width_max = max((estimate.retained_vm_width for estimate in estimate_groups), default=0)
    vm_bytes = sum(
        int(group.size)
        * int(step_count)
        * int(group_by_id[group.group_id].retained_vm_width)
        * int(dtype.itemsize)
        for group in groups
    )
    items.append(
        _item_with_bytes(
            "outputs.recorded_vm",
            (axon_count, step_count, recording_width_max),
            dtype,
            bytes=vm_bytes,
            role="public_output",
            retained=True,
            note="retained Vm according to the current recording policy",
        )
    )
    items.extend(
        _observable_output_items(
            groups,
            recording=recording,
            step_count=step_count,
            axon_count=axon_count,
            max_nx=max_nx,
            dtype=dtype,
        )
    )
    observer_bytes_by_output = {
        output: sum(
            _threshold_observer_nbytes(
                group,
                step_count=step_count,
                observers=observers,
                observer_output=output,
            )
            for group, estimate in zip(groups, estimate_groups, strict=True)
            if estimate.observer_output == output
        )
        for output in (
            "activation",
            "first_crossing",
            "spike_summary",
            "spike_events",
            "vm_raster",
        )
    }
    for output, output_bytes in observer_bytes_by_output.items():
        if not output_bytes:
            continue
        if output == "activation":
            dtype = np.dtype(bool)
        elif output == "first_crossing":
            dtype = np.dtype("int32")
        elif output in {"spike_summary", "spike_events"}:
            dtype = np.dtype("int32")
        else:
            dtype = np.dtype("uint32")
        items.append(
            _item_with_bytes(
                f"outputs.{output}",
                (axon_count,),
                dtype,
                bytes=output_bytes,
                role="public_output",
                retained=True,
                note=(
                    "solver-side activation flags"
                    if output == "activation"
                    else (
                        "solver-side first-crossing steps"
                        if output == "first_crossing"
                        else (
                            "solver-side per-probe spike summaries"
                            if output in {"spike_summary", "spike_events"}
                            else "packed solver-side VmRaster observations"
                        )
                    )
                ),
            )
        )
    return tuple(item for item in items if item.bytes > 0 or item.retained)


def _item(
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    *,
    role: str,
    retained: bool,
    note: str = "",
) -> MemoryEstimateItem:
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    return _item_with_bytes(
        name,
        shape,
        dtype,
        bytes=size * int(dtype.itemsize),
        role=role,
        retained=retained,
        note=note,
    )


def _item_with_bytes(
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    *,
    bytes: int,
    role: str,
    retained: bool,
    note: str = "",
) -> MemoryEstimateItem:
    return MemoryEstimateItem(
        name=name,
        shape=shape,
        dtype=str(dtype),
        bytes=int(bytes),
        role=role,
        retained=retained,
        note=note,
    )


def _threshold_observer_nbytes(
    group: DispatchGroup,
    *,
    step_count: int,
    observers: tuple[Any, ...] | None,
    observer_output: str,
) -> int:
    definitions = benchmark_vm_raster_definitions(observers)
    if not definitions:
        return 0
    if observer_output == "activation":
        return int(group.size) * len(definitions) * np.dtype(bool).itemsize
    if observer_output == "first_crossing":
        return int(group.size) * len(definitions) * np.dtype(np.int32).itemsize
    max_probe_count = 0
    for item in group.items:
        positions_um = np.asarray(item.solver_axon.x_um, dtype=float)
        original_indices = np.arange(positions_um.shape[0], dtype=np.int32)
        for definition in definitions:
            selected = definition.target.columns(
                positions_um=positions_um,
                original_indices=original_indices,
            )
            max_probe_count = max(max_probe_count, len(selected))
    if observer_output in {"spike_summary", "spike_events"}:
        capacity = getattr(definitions[0], "max_spikes", None)
        state_width = 4 if capacity is None else 5 + int(capacity)
        shape = (
            int(group.size),
            len(definitions),
            int(max_probe_count),
            state_width,
        )
        return int(np.prod(shape)) * np.dtype(np.int32).itemsize
    temporal_stride = int(getattr(definitions[0], "every_n_steps", 1))
    sampled_steps = (int(step_count) + temporal_stride - 1) // temporal_stride
    word_count = (sampled_steps + 31) // 32
    shape = (int(group.size), len(definitions), int(max_probe_count), word_count)
    return int(np.prod(shape)) * np.dtype(np.uint32).itemsize


def _observable_output_nbytes(
    group: DispatchGroup,
    *,
    recording: Recording | None,
    step_count: int,
    dtype: np.dtype[Any],
) -> int:
    if recording is None or not recording.wants_observables:
        return 0
    counts = _observable_group_counts(group, recording=recording)
    width = sum(counts.values())
    return int(group.size) * int(step_count) * int(group.nx) * int(width) * int(dtype.itemsize)


def _observable_output_items(
    groups: Sequence[DispatchGroup],
    *,
    recording: Recording | None,
    step_count: int,
    axon_count: int,
    max_nx: int,
    dtype: np.dtype[Any],
) -> tuple[MemoryEstimateItem, ...]:
    if recording is None or not recording.wants_observables:
        return ()
    result: list[MemoryEstimateItem] = []
    for attr_name, item_name in (
        ("gates", "outputs.gates"),
        ("currents", "outputs.currents"),
        ("conductances", "outputs.conductances"),
        ("state_variables", "outputs.states"),
    ):
        if not getattr(recording, attr_name):
            continue
        group_counts = [
            _observable_group_counts(group, recording=recording)[attr_name]
            for group in groups
        ]
        max_count = max(group_counts, default=0)
        bytes_value = sum(
            int(group.size)
            * int(step_count)
            * int(group.nx)
            * int(count)
            * int(dtype.itemsize)
            for group, count in zip(groups, group_counts, strict=True)
        )
        result.append(
            _item_with_bytes(
                item_name,
                (axon_count, step_count, max_nx, max_count),
                dtype,
                bytes=bytes_value,
                role="public_output",
                retained=True,
                note="retained dense observable group according to the current recording policy",
            )
        )
    return tuple(item for item in result if item.bytes > 0)


def _observable_group_counts(
    group: DispatchGroup,
    *,
    recording: Recording,
) -> dict[str, int]:
    wanted = {
        "gates": bool(recording.gates),
        "currents": bool(recording.currents),
        "conductances": bool(recording.conductances),
        "state_variables": bool(recording.state_variables),
    }
    names: dict[str, set[str]] = {key: set() for key in wanted}
    for item in group.items:
        for model in item.solver_axon.membrane_models:
            if wanted["gates"]:
                names["gates"].update(_call_name_tuple(model, "gate_names"))
            if wanted["currents"]:
                names["currents"].update(_call_name_tuple(model, "current_names"))
            if wanted["conductances"]:
                names["conductances"].update(_call_name_tuple(model, "conductance_names"))
            if wanted["state_variables"]:
                names["state_variables"].update(
                    _call_name_tuple(model, "membrane_state_names")
                )
    return {key: len(value) if wanted[key] else 0 for key, value in names.items()}


def _call_name_tuple(model: Any, method_name: str) -> tuple[str, ...]:
    return benchmark_membrane_output_names(model, method_name)


def _precision_from_instances(instances: Sequence[AxonInstance]) -> PrecisionPolicy:
    itemsize = max(np.dtype(instance.dtype).itemsize for instance in instances)
    dtype = "float64" if itemsize > np.dtype("float32").itemsize else "float32"
    return PrecisionPolicy(dtype, dtype, dtype)


def _extracellular_stimulation_count(instances: Sequence[AxonInstance]) -> int:
    return sum(len(row) for row in extracellular_stimulation_rows(instances))


def _extracellular_drive_count(instances: Sequence[AxonInstance]) -> int:
    total = 0
    for row in extracellular_stimulation_rows(instances):
        for stimulation in row:
            total += len(tuple(getattr(stimulation, "drives", ())))
    return total


def _unique_stimulus_count(instances: Sequence[AxonInstance]) -> int:
    seen: set[int] = set()
    stimulation_rows = extracellular_stimulation_rows(instances)
    for instance, stimulations in zip(instances, stimulation_rows, strict=True):
        for context in getattr(instance, "intracellular_contexts", ()):
            stimulus = getattr(context, "current", None)
            if stimulus is not None:
                seen.add(id(stimulus))
        for stimulation in stimulations:
            for drive in getattr(stimulation, "drives", ()):
                stimulus = getattr(drive, "stimulus", None)
                if stimulus is not None:
                    seen.add(id(stimulus))
    return len(seen)


def _intracellular_context_rows(instances: Sequence[AxonInstance]) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    for instance in instances:
        rows.append(tuple(getattr(instance, "intracellular_contexts", ())))
    return tuple(rows)


def _intracellular_context_count(instances: Sequence[AxonInstance]) -> int:
    return sum(len(row) for row in _intracellular_context_rows(instances))


def _max_intracellular_context_count(instances: Sequence[AxonInstance]) -> int:
    return max((len(row) for row in _intracellular_context_rows(instances)), default=0)


def _estimate_guidance(
    items: Sequence[MemoryEstimateItem],
    *,
    groups: Sequence[SimulationEstimateGroup],
    extracellular_stimulation_count: int,
    max_nx: int,
    recording_width_max: int,
    memory_budget_bytes: int | None,
    observers: Sequence[Any] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    recommendations: list[str] = []
    item_map = {item.name: item for item in items}

    dense_iinj = item_map.get("inputs.intracellular_current_density")
    sparse_iinj = item_map.get("inputs.intracellular_current_density_sparse")
    if dense_iinj is not None and observers and recording_width_max == 0:
        warnings.append(
            "Observer-only run still estimates dense Iinj[B,Nt,Nx] "
            f"({dense_iinj.mib:.3f} MiB); sparse lowering applies only to "
            "batch current-clamp rows."
        )
    if sparse_iinj is not None:
        recommendations.append(
            "Observer-only current-clamp input is estimated with sparse "
            f"compartment slots ({sparse_iinj.mib:.3f} MiB before indices/mask)."
        )
    padded = [group for group in groups if group.padded_compartments]
    if padded:
        warnings.append(
            "Some dispatch groups need spatial padding: "
            + ", ".join(
                f"group {group.group_id} +{group.padded_compartments} compartments"
                for group in padded
            )
            + "."
        )

    dense_vext = item_map.get("inputs.extracellular_potential_mid")
    factorized = item_map.get("footprints.factorized_rows")
    if dense_vext is not None:
        warnings.append(
            "Current batch extracellular lowering may materialize dense "
            f"Vstim[B,Nt,Nx] ({dense_vext.mib:.3f} MiB for this run)."
        )
        if factorized is not None and dense_vext.bytes > max(1, 8 * factorized.bytes):
            recommendations.append(
                "Keep Phase 7.5 focused on in-kernel observer/drive reductions: "
                f"factorized footprints are {factorized.mib:.3f} MiB before the time axis."
            )

    output = item_map.get("outputs.recorded_vm")
    if output is not None and recording_width_max >= max_nx:
        warnings.append(
            "Full Vm recording retains every compartment; large pools should prefer "
            "center/probe recordings or future observer-only runs."
        )
    if observers:
        recommendations.append(
            "Use Recording.none() with solver-side observers to keep packed "
            "VmRaster output instead of retaining Vm[Nt,Nx]."
        )
    if memory_budget_bytes is not None:
        total = sum(item.bytes for item in items)
        if total > int(memory_budget_bytes):
            warnings.append(
                f"Estimated arrays exceed memory_budget_bytes={int(memory_budget_bytes)} "
                f"({total} bytes estimated)."
            )

    if extracellular_stimulation_count == 0:
        recommendations.append(
            "No extracellular stimulations detected; use this as the zero-field baseline "
            "when comparing CPU/GPU hotpath traces."
        )
    return tuple(warnings), tuple(recommendations)


def _recording_label(recording: Recording | None, batch_options: BatchOptions | None) -> str:
    if recording is not None:
        if not recording.voltage:
            return "none"
        return recording.spatial.value
    if batch_options is not None:
        return batch_options.recording.label
    return "full"


__all__ = [
    "MemoryEstimateItem",
    "SimulationEstimate",
    "SimulationEstimateGroup",
    "estimate_simulation",
]
