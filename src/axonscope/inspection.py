"""Host-side inspection builder for the solver pipeline."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.backends.execution import batch_options_from_recording
from axonscope.backends.jax.input_lowering import plan_input_lowering
from axonscope.backends.jax.recording_lowering import (
    lower_batch_recording_options,
    observers_are_vm_raster_compatible,
    vm_raster_definitions,
)
from axonscope.dispatcher.plan import DispatchGroup, build_dispatch_plan
from axonscope.inspection_records import (
    AssemblyDetailInspection,
    DispatchGroupInspection,
    KernelInspection,
    LoweringInspection,
    MemoryInspection,
    MembraneSourceInspection,
    PaddingInspection,
    PlanningInspection,
    PreparationInspection,
    ProbeInspection,
    ResultAssemblyInspection,
    SimulationInspection,
)
from axonscope.membranes.compiler import lower_membrane_model_with_sources
from axonscope.performance import ExecutionPolicy
from axonscope.population import AxonPopulation
from axonscope.preparation.cohort import PreparedCohort
from axonscope.recording import Recording
from axonscope.solvers.options import (
    BatchOptions,
    resolve_double_cable_block_solver,
)
from axonscope.timebase import simulation_step_count
from axonscope.utils import units


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
    dispatch_groups = tuple(
        _inspect_dispatch_group(
            group,
            batch_options=resolved_batch_options,
            observers=observer_defs,
        )
        for group in groups
    )
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
            batch_options=resolved_batch_options,
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
        membrane_sources=tuple(_inspect_membrane_sources(group) for group in groups),
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
                observers=observer_defs,
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


def _inspect_dispatch_group(
    group: DispatchGroup,
    *,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None,
) -> DispatchGroupInspection:
    return DispatchGroupInspection(
        group_id=int(group.group_id),
        pool_indices=group.pool_indices,
        mode=str(group.mode),
        size=int(group.size),
        nx=int(group.nx),
        batch_kind=group.batch_kind,
        geometry_shared=bool(group.geometry_shared),
        has_padding=bool(group.has_padding),
        will_batch=_can_batch(group, batch_options=batch_options, observers=observers),
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
        extracellular_stimulation_count=int(cohort.extracellular_stimulation_count),
        x_positions_shape=tuple(int(value) for value in cohort.x_positions_m.shape),
        y_shape=tuple(int(value) for value in cohort.axon_y_um.shape),
        z_shape=tuple(int(value) for value in cohort.axon_z_um.shape),
        representative_index=representative_index,
    )


def _inspect_membrane_sources(group: DispatchGroup) -> MembraneSourceInspection:
    unique_models: dict[tuple[Any, ...], Any] = {}
    for item in group.items:
        for model in item.solver_axon.membrane_models:
            signature = model._static_signature()
            unique_models.setdefault(signature, model)

    source_results: list[Any] = []
    kinds: list[str] = []
    for model in unique_models.values():
        lowered = lower_membrane_model_with_sources(model)
        kinds.append(str(model.kind))
        source_results.extend(lowered.source_results)

    return MembraneSourceInspection(
        group_id=int(group.group_id),
        unique_membrane_count=len(unique_models),
        kinds=tuple(sorted(set(kinds))),
        source_count=len(source_results),
        cache_statuses=tuple(
            "hit" if result.cache.cache_hit else "miss"
            for result in source_results
        ),
        cache_reasons=tuple(result.cache.cache_reason for result in source_results),
        cache_keys=tuple(result.cache.key for result in source_results),
        source_hashes=tuple(result.source_hash for result in source_results),
        source_paths=tuple(str(result.source_path) for result in source_results),
    )


def _can_batch(
    group: DispatchGroup,
    *,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None,
) -> bool:
    if group.mode not in {"single", "double"}:
        return False
    if group.size >= 2:
        return True
    return observers is not None and batch_options.recording.mode == "none"


def _inspection_batch_options(
    *,
    recording: Recording | None,
    batch_options: BatchOptions | None,
) -> BatchOptions:
    options = BatchOptions.full() if batch_options is None else batch_options
    lowered = batch_options_from_recording(recording, batch_options=options)
    return options if lowered is None else lowered


def _inspect_lowering(
    group: DispatchGroup,
    *,
    step_count: int,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None,
) -> LoweringInspection:
    vm_raster_supported = observers_are_vm_raster_compatible(observers)
    if not _can_batch(group, batch_options=batch_options, observers=observers):
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
    kernel_options = lower_batch_recording_options(
        group,
        batch_options,
        observers=observers,
    )
    observer_plan = (
        observers is not None
        and kernel_options.recording.mode == "none"
        and vm_raster_supported
    )
    planned = plan_input_lowering(
        group_mode=group.mode,
        axons=cohort.axons,
        stimulation_rows=cohort.stimulations,
        kernel_options=kernel_options,
        observers=observers,
        observer_plan=observer_plan,
    )
    intracellular_format = planned.intracellular_format
    extracellular_format = planned.extracellular_format

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
    batch_options: BatchOptions,
) -> ProbeInspection:
    definitions = vm_raster_definitions(observers)
    if not definitions:
        return ProbeInspection(
            group_id=int(group.group_id),
            observer_names=(),
            thresholds_mV=(),
            row_aware=_can_batch(group, batch_options=batch_options, observers=observers),
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
        row_aware=_can_batch(group, batch_options=batch_options, observers=observers),
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
    observers: tuple[Any, ...] | None,
    execution_policy: ExecutionPolicy | None,
) -> KernelInspection:
    if not _can_batch(group, batch_options=batch_options, observers=observers):
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
    if not _can_batch(group, batch_options=batch_options, observers=observers):
        vm_output = (
            "none"
            if batch_options.recording.mode == "none"
            else f"Vm[Nt={step_count}, Nx={group.nx}]"
        )
        if observers is None:
            observations = "none"
        elif (
            batch_options.recording.mode == "none"
            and observers_are_vm_raster_compatible(observers)
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

    kernel_options = lower_batch_recording_options(
        group,
        batch_options,
        observers=observers,
    )
    width = int(kernel_options.recording.width_for(group.nx))
    observer_only = (
        observers is not None
        and kernel_options.recording.mode == "none"
        and observers_are_vm_raster_compatible(observers)
    )
    if observer_only:
        return ResultAssemblyInspection(
            group_id=int(group.group_id),
            record_kind="compact dispatch cohort",
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
        record_kind="dispatch row records",
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
    kernel_options = lower_batch_recording_options(
        group,
        batch_options,
        observers=observers,
    )
    width = int(kernel_options.recording.width_for(group.nx))
    vm_shape: tuple[int, ...] | None = None
    if kernel_options.recording.mode != "none":
        if _can_batch(group, batch_options=batch_options, observers=observers):
            vm_shape = (int(group.size), int(step_count), width)
        else:
            vm_shape = (int(step_count), int(group.nx))
    probes = (
        _inspect_probes(
            group,
            step_count=step_count,
            observers=observers,
            batch_options=batch_options,
        )
        if probes is None
        else probes
    )
    observation_shape = (
        probes.packed_shape
        if (
            probes.packed_shape is not None
            and kernel_options.recording.mode == "none"
            and observers_are_vm_raster_compatible(observers)
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
        public_rows=(
            1
            if observations_are_batched
            and _can_batch(group, batch_options=batch_options, observers=observers)
            else int(group.size)
        ),
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


__all__ = [
    "AssemblyDetailInspection",
    "DispatchGroupInspection",
    "KernelInspection",
    "LoweringInspection",
    "MemoryInspection",
    "MembraneSourceInspection",
    "PaddingInspection",
    "PlanningInspection",
    "ProbeInspection",
    "PreparationInspection",
    "ResultAssemblyInspection",
    "SimulationInspection",
    "inspect_simulation",
]
