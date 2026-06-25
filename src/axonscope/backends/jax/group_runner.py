"""JAX execution for prepared dispatcher groups."""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from axonscope.benchmarking.hotpaths import (
    benchmark_array_metadata,
    benchmark_span,
    benchmark_wait,
    record_benchmark_metadata,
)
from axonscope.dispatcher.plan import DispatchGroup, DispatchItem
from axonscope.dispatcher.progress import ProgressEvent, ProgressStage
from axonscope.dispatcher.results import DispatchCohortResult, DispatchRecord, DispatchResult
from axonscope.backends.jax.input_batches import (
    build_factorized_vstim_midpoint_batch,
    build_intracellular_current_density_batch,
    build_sparse_intracellular_current_density_batch,
    build_vstim_midpoint_and_initial_previous_batch,
    build_vstim_midpoint_batch,
    can_build_sparse_intracellular_current_density_batch,
)
from axonscope.icm.backends import (
    HeterogeneousICMBackend,
    RowIndexedICMBackend,
    _AxNodePassiveFamilyICMBackend,
)
from axonscope.preparation.cohort import PreparedCohort
from axonscope.results.single import SimResult
from axonscope.backends.jax.batch_kernels import (
    BatchKernelResult,
    DoubleCableBatchKernel,
    SingleCableVStimBatchKernel,
)
from axonscope.backends.jax.observer_runtime import (
    build_vm_raster_plan,
)
from axonscope.solvers.options import BatchOptions, BatchRecording, SolverOptions
from axonscope.backends.jax.runtime import (
    CableRuntime,
    ExtracellularRuntime,
    MembraneRuntime,
    SolverRuntime,
    compile_membrane_model,
    prepare_membrane_runtime,
    prepare_simulation_grid,
    prepare_solver_runtime,
    prepare_stimulation_runtime,
)


_BATCH_RUNTIME_CACHE: OrderedDict[tuple[Any, ...], SolverRuntime] = OrderedDict()
_BATCH_STATIC_RUNTIME_CACHE: OrderedDict[tuple[Any, ...], SolverRuntime] = OrderedDict()
_PREPARED_COHORT_CACHE: OrderedDict[tuple[Any, ...], PreparedCohort] = OrderedDict()
_VM_RASTER_PLAN_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_GROUP_RUNNER_CACHE_MAX_SIZE = 64
_DOUBLE_CABLE_SHAPE_BUCKETING_ENV = "AXONSCOPE_EXPERIMENTAL_DOUBLE_CABLE_SHAPE_BUCKETING"
_DOUBLE_CABLE_BATCH_BUCKETS = (16, 32, 64, 128)
_DOUBLE_CABLE_NX_BUCKET_MULTIPLE = 128


@dataclass(frozen=True)
class _FamilyMembraneStack:
    backend: _AxNodePassiveFamilyICMBackend
    gates0_rows: np.ndarray
    background_rows: np.ndarray
    membrane_static: Any
    node_count: int
    passive_count: int
    source: str


def _cache_get(cache: OrderedDict[tuple[Any, ...], Any], key: tuple[Any, ...]) -> Any | None:
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_store(
    cache: OrderedDict[tuple[Any, ...], Any],
    key: tuple[Any, ...],
    value: Any,
) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _GROUP_RUNNER_CACHE_MAX_SIZE:
        cache.popitem(last=False)


def run_jax_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    solver_options: SolverOptions | None,
    observers: tuple[Any, ...] | None = None,
    progress_callback: Any = None,
) -> tuple[DispatchRecord, ...]:
    """Execute one compatible group through the JAX batch backend."""

    if group.mode == "double":
        return _run_double_cable_batch_group(
            group,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            batch_options=batch_options,
            solver_options=solver_options,
            observers=observers,
            progress_callback=progress_callback,
        )
    return _run_single_cable_batch_group(
        group,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        batch_options=batch_options,
        solver_options=solver_options,
        observers=observers,
        progress_callback=progress_callback,
    )


def _dispatch_method(group: DispatchGroup) -> str:
    """Return the public diagnostic label for a dispatch group."""

    prefix = "batch" if group.geometry_shared else "parameter-batch"
    if group.mode == "double":
        return f"{prefix}-double-cable"
    return f"{prefix}-single-cable"


def _double_cable_kernel_group(group: DispatchGroup) -> DispatchGroup:
    """Return a backend-only group padded to stable double-cable JAX shapes."""

    if group.mode != "double" or not _double_cable_shape_bucketing_enabled():
        return group
    target_size = _bucket_batch_size(group.size, buckets=_DOUBLE_CABLE_BATCH_BUCKETS)
    target_nx = _bucket_nx(
        group.nx,
        multiple=_DOUBLE_CABLE_NX_BUCKET_MULTIPLE,
    )
    if target_size == group.size and target_nx == group.nx:
        return group
    if not group.items:
        raise ValueError("cannot bucket an empty dispatch group.")
    padded_items = tuple(group.items) + (group.items[-1],) * (target_size - group.size)
    return DispatchGroup(
        group_id=group.group_id,
        items=padded_items,
        signature=(
            group.signature,
            "double_cable_kernel_bucket_v1",
            int(group.size),
            int(group.nx),
            int(target_size),
            int(target_nx),
        ),
        mode=group.mode,
        nx=int(target_nx),
        geometry_shared=False,
    )


def _double_cable_shape_bucketing_enabled() -> bool:
    """Whether to run the experimental shape-bucketed double-cable kernel path."""

    return os.environ.get(_DOUBLE_CABLE_SHAPE_BUCKETING_ENV, "").strip() in {"1", "true", "yes"}


def _bucket_batch_size(value: int, *, buckets: tuple[int, ...]) -> int:
    requested = int(value)
    if requested <= 0:
        raise ValueError("batch size must be positive.")
    for bucket in buckets:
        if requested <= int(bucket):
            return int(bucket)
    step = int(buckets[-1])
    return ((requested + step - 1) // step) * step


def _bucket_nx(value: int, *, multiple: int) -> int:
    requested = int(value)
    step = int(multiple)
    if requested <= 0 or step <= 0:
        raise ValueError("Nx and bucket multiple must be positive.")
    return ((requested + step - 1) // step) * step


def _record_kernel_bucket_metadata(
    *,
    group: DispatchGroup,
    kernel_group: DispatchGroup,
) -> None:
    record_benchmark_metadata(
        public_group_size=int(group.size),
        public_nx=int(group.nx),
        kernel_group_size=int(kernel_group.size),
        kernel_nx=int(kernel_group.nx),
        kernel_batch_padding_rows=int(kernel_group.size) - int(group.size),
        kernel_spatial_padding=int(kernel_group.nx) - int(group.nx),
        kernel_shape_bucketed=(
            int(kernel_group.size) != int(group.size)
            or int(kernel_group.nx) != int(group.nx)
        ),
    )


def _emit_progress(
    progress_callback: Any,
    group: DispatchGroup,
    stage: str,
    message: str,
    **details: Any,
) -> None:
    """Emit one structured progress event when reporting is enabled."""

    if progress_callback is None:
        return
    event = ProgressEvent(
        stage=cast(ProgressStage, stage),
        group_id=int(group.group_id),
        rows=int(group.size),
        nx=int(group.nx),
        route=_dispatch_method(group),
        message=message,
        details={key: value for key, value in details.items() if value is not None},
    )
    progress_callback(event)


def _batch_wait_target(out: Any) -> Any:
    """Return a JAX/NumPy object that synchronizes a batch kernel result."""

    if out.Vm is not None:
        return out.Vm
    if not out.observations:
        raise RuntimeError("batch kernel produced neither Vm nor observations.")
    first = next(iter(out.observations.values()))
    if hasattr(first, "words"):
        return first.words
    return first.values


def _observer_plan_for_cohort(
    observers: tuple[Any, ...] | None,
    *,
    cohort: PreparedCohort,
    dtype: Any,
    prefer_vm_raster: bool = False,
) -> Any:
    """Lower public observers for one compatible prepared cohort."""

    if observers is None:
        return None
    if not prefer_vm_raster:
        return None
    cache_key = _vm_raster_plan_cache_key(
        observers,
        cohort=cohort,
        dtype=dtype,
    )
    cached = _cache_get(_VM_RASTER_PLAN_CACHE, cache_key)
    if cached is not None:
        record_benchmark_metadata(vm_raster_plan_cache="hit")
        return cached

    row_positions_um = np.asarray(cohort.x_positions_m, dtype=float) * 1e6
    plan = build_vm_raster_plan(
        observers,
        positions_um=row_positions_um,
        original_indices=_cohort_original_indices(cohort),
        dtype=dtype,
    )
    _cache_store(_VM_RASTER_PLAN_CACHE, cache_key, plan)
    record_benchmark_metadata(
        vm_raster_plan_cache="miss",
        vm_raster_count=0 if plan is None else plan.raster_count,
        vm_raster_probe_count=0 if plan is None else plan.probe_count,
    )
    return plan


def _vm_raster_plan_cache_key(
    observers: tuple[Any, ...],
    *,
    cohort: PreparedCohort,
    dtype: Any,
) -> tuple[Any, ...]:
    return (
        "vm_raster_plan_v1",
        str(np.dtype(dtype)),
        _prepared_cohort_signature(cohort),
        tuple(_observer_definition_signature(observer) for observer in observers),
    )


def _observer_definition_signature(observer: Any) -> tuple[Any, ...]:
    signal = getattr(observer, "signal", None)
    signal_id = getattr(signal, "id", repr(signal))
    target = getattr(observer, "target", None)
    return (
        type(observer).__module__,
        type(observer).__qualname__,
        str(getattr(observer, "name", "")),
        str(signal_id),
        repr(target),
        _maybe_millivolt(getattr(observer, "threshold", None)),
        _maybe_millisecond(getattr(observer, "blanking", None)),
    )


def _maybe_millivolt(value: Any) -> float | None:
    if value is None:
        return None
    from axonscope.utils import units

    return float(units.to_mV(value))


def _maybe_millisecond(value: Any) -> float | None:
    if value is None:
        return None
    from axonscope.utils import units

    return float(units.to_ms(value))


def _cohort_original_indices(cohort: PreparedCohort) -> np.ndarray:
    """Return row-aware original compartment indices, with -1 for padding."""

    rows = np.full((cohort.size, cohort.nx), -1, dtype=np.int32)
    for row_index, solver_axon in enumerate(cohort.solver_axons):
        original_nx = int(solver_axon.n_compartments)
        rows[row_index, :original_nx] = np.arange(original_nx, dtype=np.int32)
    return rows


def _representative_item(group: DispatchGroup) -> DispatchItem:
    """Return the row used to compile the shared runtime."""

    for item in group.items:
        if int(item.solver_axon.n_compartments) == int(group.nx):
            return item
    return group.items[0]


def _group_static_signature(group: DispatchGroup) -> tuple[Any, ...]:
    return (
        "dispatch_group_v1",
        group.mode,
        int(group.nx),
        bool(group.geometry_shared),
        bool(group.has_padding),
        tuple(
            (
                int(item.index),
                id(item.simulation),
                id(item.solver_axon),
                item.signature,
                item.membrane_signature,
                item.cable_signature,
            )
            for item in group.items
        ),
    )


def _group_runtime_signature(group: DispatchGroup) -> tuple[Any, ...]:
    """Return a structural key for stimulation-independent solver runtimes."""

    return (
        "dispatch_group_runtime_v1",
        group.mode,
        int(group.nx),
        bool(group.geometry_shared),
        bool(group.has_padding),
        tuple(
            (
                int(item.index),
                item.signature,
                item.membrane_signature,
                item.cable_signature,
            )
            for item in group.items
        ),
    )


def _group_preparation_signature(group: DispatchGroup) -> tuple[Any, ...]:
    return (
        _group_static_signature(group),
        tuple(
            (
                bool(getattr(item.simulation, "use_extracellular", False)),
                tuple(
                    id(context)
                    for context in getattr(item.simulation, "extracellular_contexts", ())
                ),
            )
            for item in group.items
        ),
    )


def _prepared_cohort_signature(cohort: PreparedCohort) -> tuple[Any, ...]:
    return (
        "prepared_cohort_v1",
        int(cohort.group_id),
        str(cohort.mode),
        int(cohort.size),
        int(cohort.nx),
        bool(cohort.geometry_shared),
        bool(cohort.has_padding),
        tuple(id(axon) for axon in cohort.axons),
        tuple(id(solver_axon) for solver_axon in cohort.solver_axons),
        tuple(tuple(id(context) for context in row) for row in cohort.contexts),
        _array_shape_dtype_digest(cohort.x_positions_m),
        _array_shape_dtype_digest(cohort.axon_y_um),
        _array_shape_dtype_digest(cohort.axon_z_um),
    )


def _array_shape_dtype_digest(values: Any) -> tuple[Any, ...]:
    arr = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
    return (
        tuple(int(dim) for dim in arr.shape),
        arr.dtype.str,
        digest,
    )


def _prepare_batch_runtime(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    solver_options: SolverOptions | None,
    mode: str,
    include_extracellular: bool,
    include_area: bool,
) -> SolverRuntime:
    cache_key = (
        "batch_runtime_v1",
        mode,
        _group_runtime_signature(group),
        float(tsim_ms),
        float(dt_ms),
        repr(solver_options),
        bool(include_extracellular),
        bool(include_area),
    )
    cached = _cache_get(_BATCH_RUNTIME_CACHE, cache_key)
    if cached is not None:
        record_benchmark_metadata(batch_runtime_cache="hit")
        return cached

    static_cache_key = (
        "batch_static_runtime_v1",
        mode,
        _group_runtime_signature(group),
        repr(solver_options),
        bool(include_extracellular),
        bool(include_area),
    )
    runtime = _cache_get(_BATCH_STATIC_RUNTIME_CACHE, static_cache_key)
    static_cache_state = "hit"
    if runtime is None:
        representative_item = _representative_item(group)
        with benchmark_span(
            "runtime.prepare.base_runtime",
            group_id=group.group_id,
            group_size=group.size,
            mode=mode,
            nx=group.nx,
        ):
            if group.geometry_shared:
                runtime = prepare_solver_runtime(
                    cast(Any, representative_item.simulation),
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                    solver_axon=representative_item.solver_axon,
                    include_extracellular=include_extracellular,
                    include_area=include_area,
                    precompute_intracellular=False,
                    precompute_extracellular=False,
                    compile_stimulation=False,
                    solver_options=solver_options,
                )
                record_benchmark_metadata(batch_base_runtime_kind="full")
            else:
                runtime = _prepare_parameter_batch_base_runtime(
                    group,
                    representative_item,
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                    mode=mode,
                    include_area=include_area,
                    solver_options=solver_options,
                )
                record_benchmark_metadata(batch_base_runtime_kind="parameter_minimal")
        if not group.geometry_shared:
            if mode == "double":
                runtime = _with_batched_double_cable_runtime(
                    runtime,
                    group,
                    solver_options=solver_options,
                )
            else:
                runtime = _with_batched_single_cable_runtime(runtime, group)
        _cache_store(_BATCH_STATIC_RUNTIME_CACHE, static_cache_key, runtime)
        static_cache_state = "miss"

    runtime = replace(
        runtime,
        grid=prepare_simulation_grid(tsim_ms, dt_ms, runtime.membrane.dtype),
    )

    _cache_store(_BATCH_RUNTIME_CACHE, cache_key, runtime)
    record_benchmark_metadata(
        batch_runtime_cache="miss",
        batch_static_runtime_cache=static_cache_state,
    )
    return runtime


def _prepare_parameter_batch_base_runtime(
    group: DispatchGroup,
    representative_item: DispatchItem,
    *,
    tsim_ms: float,
    dt_ms: float,
    mode: str,
    include_area: bool,
    solver_options: SolverOptions | None,
) -> SolverRuntime:
    """Prepare only representative fields that survive parameter batching."""

    simulation = cast(Any, representative_item.simulation)
    solver_axon = representative_item.solver_axon
    if mode == "double" and _axnode_passive_family_node_description(group) is not None:
        membrane = _minimal_axnode_passive_family_membrane_runtime(
            group,
            representative_item=representative_item,
            solver_options=solver_options,
        )
        record_benchmark_metadata(batch_base_membrane_kind="axnode_passive_placeholder")
    else:
        membrane = prepare_membrane_runtime(
            simulation,
            solver_axon=solver_axon,
            solver_options=solver_options,
        )
        record_benchmark_metadata(batch_base_membrane_kind="full_representative")
    grid = prepare_simulation_grid(tsim_ms, dt_ms, membrane.dtype)
    cable = _cable_runtime_from_numpy_arrays(
        solver_axon,
        dtype_local=membrane.dtype,
        include_area=include_area,
    )
    stimulation = prepare_stimulation_runtime(
        simulation,
        solver_axon,
        membrane.dtype,
        grid=None,
        precompute_intracellular=False,
        precompute_extracellular=False,
        compile_callables=False,
    )
    return SolverRuntime(
        axon=solver_axon,
        grid=grid,
        membrane=membrane,
        cable=cable,
        stimulation=stimulation,
        extracellular=None,
    )


def _minimal_axnode_passive_family_membrane_runtime(
    group: DispatchGroup,
    *,
    representative_item: DispatchItem,
    solver_options: SolverOptions | None,
) -> MembraneRuntime:
    """Build a placeholder membrane runtime replaced by direct family stacking."""

    node_description = _axnode_passive_family_node_description(group)
    if node_description is None:
        raise ValueError("group is not an AxNode/passive family group.")
    node_model = compile_membrane_model(
        node_description,
        solver_options=solver_options,
    )
    dtype_local = node_model.dtype
    nx = int(representative_item.solver_axon.n_compartments)
    return MembraneRuntime(
        backend=None,
        membrane=node_model,
        dtype=dtype_local,
        Nx=nx,
        Vm0_mV=jnp.zeros((nx,), dtype=dtype_local),
        gates0=jnp.zeros((nx, 0), dtype=dtype_local),
        state0=(),
        background_current=jnp.zeros((nx,), dtype=dtype_local),
        observable_names={
            "gates": node_model.gate_names(),
            "currents": node_model.current_names(),
            "conductances": node_model.conductance_names(),
            "states": node_model.membrane_state_names(),
        },
        diagnostic_names=node_model.diagnostic_names(),
    )


def _axnode_passive_family_node_description(group: DispatchGroup) -> Any | None:
    """Return the shared AxNode description when a group is AxNode/passive only."""

    node_description: Any | None = None
    node_signature: tuple[Any, ...] | None = None
    node_count = 0
    passive_count = 0
    for item in group.items:
        solver_axon = item.solver_axon
        membrane_models = tuple(solver_axon.membrane_models)
        if len(membrane_models) != int(solver_axon.n_compartments):
            return None
        for model in membrane_models:
            kind = str(getattr(model, "kind", "")).lower()
            if kind == "axnode":
                signature = model._static_signature()
                if node_signature is None:
                    node_signature = signature
                    node_description = model
                elif signature != node_signature:
                    return None
                node_count += 1
            elif kind == "passive":
                passive_count += 1
            else:
                return None
    if node_description is None or node_count == 0 or passive_count == 0:
        return None
    return node_description


def _prepared_cohort_for_group(group: DispatchGroup) -> PreparedCohort:
    cache_key = ("prepared_cohort_v1", _group_preparation_signature(group))
    cached = _cache_get(_PREPARED_COHORT_CACHE, cache_key)
    if cached is not None:
        record_benchmark_metadata(prepared_cohort_cache="hit")
        return cached

    cohort = PreparedCohort.from_dispatch_group(group)
    _cache_store(_PREPARED_COHORT_CACHE, cache_key, cohort)
    record_benchmark_metadata(prepared_cohort_cache="miss")
    return cohort


def _kernel_batch_options(
    group: DispatchGroup,
    options: BatchOptions,
    *,
    observers: tuple[Any, ...] | None,
) -> BatchOptions:
    """Return solver-kernel options, recording full traces for padded groups."""

    if not group.has_padding:
        return options
    if options.recording.mode == "none" and observers is not None:
        return options
    return replace(
        options,
        recording=BatchRecording.full(),
    )


def _should_use_sparse_intracellular_batch(
    *,
    group: DispatchGroup,
    cohort: PreparedCohort,
    kernel_options: BatchOptions,
    observers: tuple[Any, ...] | None,
) -> bool:
    """Return whether sparse point-clamp lowering can feed this group."""

    return (
        group.mode == "single"
        and observers is not None
        and kernel_options.recording.mode == "none"
        and can_build_sparse_intracellular_current_density_batch(cohort.axons)
    )


def _has_intracellular_contexts(cohort: PreparedCohort) -> bool:
    """Return whether any row has an attached intracellular input."""

    return any(getattr(axon, "intracellular_contexts", ()) for axon in cohort.axons)


def _record_zero_intracellular_metadata(
    *,
    group: DispatchGroup,
    runtime: SolverRuntime,
) -> None:
    """Record skipped dense-Iinj metadata for zero-input cohorts."""

    dtype = np.dtype(runtime.membrane.dtype)
    skipped_shape = (group.size, runtime.grid.Nt, group.nx)
    record_benchmark_metadata(
        input_format="zero_no_intracellular_context",
        skipped_dense_iinj_shape=list(skipped_shape),
        skipped_dense_iinj_nbytes=int(np.prod(skipped_shape)) * int(dtype.itemsize),
    )


def _record_group_memory_estimate(
    *,
    group: DispatchGroup,
    runtime: SolverRuntime,
    cohort: PreparedCohort,
    kernel_options: BatchOptions,
    intracellular_format: str,
    extracellular_format: str,
    include_vstim_previous: bool,
) -> None:
    """Attach a conservative per-group memory estimate to the group span."""

    dtype = np.dtype(runtime.membrane.dtype)
    batch_size = int(group.size)
    nt = int(runtime.grid.Nt)
    nx = int(group.nx)
    itemsize = int(dtype.itemsize)
    positions_nbytes = int(np.asarray(cohort.x_positions_m).nbytes)
    dense_shape = (batch_size, nt, nx)
    dense_nbytes = int(np.prod(dense_shape, dtype=np.int64)) * itemsize
    if extracellular_format == "zero_no_context":
        vstim_mid_nbytes = 0
    elif extracellular_format == "factorized_footprint":
        vstim_mid_nbytes = (nt + batch_size * nx) * itemsize
    else:
        vstim_mid_nbytes = dense_nbytes
    if not include_vstim_previous:
        vstim_previous_nbytes = 0
    elif extracellular_format == "factorized_footprint":
        vstim_previous_nbytes = itemsize
    else:
        vstim_previous_nbytes = batch_size * nx * itemsize
    iinj_dense_nbytes = dense_nbytes if intracellular_format == "dense" else 0
    output_width = int(kernel_options.recording.width_for(nx))
    vm_output_nbytes = batch_size * nt * output_width * itemsize
    components = {
        "positions": positions_nbytes,
        "vstim_mid": vstim_mid_nbytes,
        "vstim_previous": vstim_previous_nbytes,
        "iinj_dense": iinj_dense_nbytes,
        "vm_output": vm_output_nbytes,
    }
    total_nbytes = int(sum(components.values()))
    capacity_bytes = _default_device_memory_capacity_bytes()
    metadata: dict[str, Any] = {
        "memory_estimate_components_nbytes": components,
        "memory_estimate_total_nbytes": total_nbytes,
        "memory_estimate_total_mib": total_nbytes / (1024**2),
        "memory_estimate_dtype": str(dtype),
        "memory_estimate_shape": {
            "batch_size": batch_size,
            "nt": nt,
            "nx": nx,
            "recording_width": output_width,
        },
        "memory_estimate_intracellular_format": intracellular_format,
        "memory_estimate_extracellular_format": extracellular_format,
    }
    if extracellular_format == "factorized_footprint":
        metadata["memory_estimate_vstim_dense_equivalent_nbytes"] = dense_nbytes
    if capacity_bytes is not None and capacity_bytes > 0:
        metadata["device_memory_capacity_bytes"] = int(capacity_bytes)
        metadata["memory_estimate_device_fraction"] = total_nbytes / float(capacity_bytes)
    record_benchmark_metadata(**metadata)


def _default_device_memory_capacity_bytes() -> int | None:
    """Best-effort capacity for the first JAX device, when the backend exposes it."""

    try:
        import jax

        devices = jax.devices()
        if not devices:
            return None
        stats_fn = getattr(devices[0], "memory_stats", None)
        if callable(stats_fn):
            stats = stats_fn() or {}
            for key in (
                "bytes_limit",
                "device_memory_capacity",
                "memory_limit",
                "bytes_reserved",
            ):
                value = stats.get(key)
                if value is not None:
                    return int(value)
    except Exception:
        return None
    return None


def _run_single_cable_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    solver_options: SolverOptions | None,
    observers: tuple[Any, ...] | None,
    progress_callback: Any = None,
) -> tuple[DispatchRecord, ...]:
    """Run a homogeneous single-cable group through imposed-field batching."""

    _emit_progress(progress_callback, group, "prepare", "runtime", mode="single")
    with benchmark_span(
        "runtime.prepare",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        nx=group.nx,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
    ):
        runtime = _prepare_batch_runtime(
            group,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            solver_options=solver_options,
            mode="single",
            include_extracellular=False,
            include_area=False,
        )
        record_benchmark_metadata(
            nt=runtime.grid.Nt,
            nx=runtime.membrane.Nx,
            dtype=str(runtime.membrane.dtype),
        )
    _emit_progress(progress_callback, group, "prepare", "cohort rows")
    with benchmark_span(
        "inputs.positions",
        group_id=group.group_id,
        group_size=group.size,
        nx=group.nx,
    ):
        cohort = _prepared_cohort_for_group(group)
        record_benchmark_metadata(
            **benchmark_array_metadata(
                "x_positions_m",
                cohort.x_positions_m,
                role="positions",
            ),
            context_count=cohort.context_count,
        )
    kernel_options = _kernel_batch_options(group, batch_options, observers=observers)
    _emit_progress(
        progress_callback,
        group,
        "batch",
        "recording plan",
        recording=kernel_options.recording.mode,
        time_chunk_steps=kernel_options.time_chunk_steps,
        observers=0 if observers is None else len(observers),
    )
    with benchmark_span(
        "observer.plan",
        group_id=group.group_id,
        group_size=group.size,
        recording_mode=kernel_options.recording.mode,
    ):
        observer_plan = _observer_plan_for_cohort(
            observers,
            cohort=cohort,
            dtype=runtime.membrane.dtype,
            prefer_vm_raster=kernel_options.recording.mode == "none",
        )
    use_sparse_intracellular = _should_use_sparse_intracellular_batch(
        group=group,
        cohort=cohort,
        kernel_options=kernel_options,
        observers=observers,
    )
    use_zero_intracellular = (
        not use_sparse_intracellular and not _has_intracellular_contexts(cohort)
    )
    use_zero_extracellular = use_sparse_intracellular and cohort.context_count == 0
    intracellular_format = (
        "sparse_current_clamp"
        if use_sparse_intracellular
        else "zero_no_intracellular_context"
        if use_zero_intracellular
        else "dense"
    )
    extracellular_format = "zero_no_context" if use_zero_extracellular else "dense"
    with benchmark_span(
        "inputs.intracellular",
        group_id=group.group_id,
        group_size=group.size,
        nt=runtime.grid.Nt,
        nx=group.nx,
    ):
        if use_sparse_intracellular:
            iinj_mid = build_sparse_intracellular_current_density_batch(
                cohort.axons,
                runtime,
                solver_axons=cohort.solver_axons,
                target_nx=cohort.nx,
            )
            record_benchmark_metadata(
                input_format="sparse_current_clamp",
                target_nx=iinj_mid.target_nx,
                max_sparse_entries=iinj_mid.max_sparse_entries,
                **benchmark_array_metadata(
                    "iinj_density_mid",
                    iinj_mid.density_mid,
                    role="kernel_input",
                ),
                **benchmark_array_metadata(
                    "iinj_indices",
                    iinj_mid.indices,
                    role="kernel_input",
                ),
                **benchmark_array_metadata("iinj_mask", iinj_mid.mask, role="kernel_input"),
            )
        elif use_zero_intracellular:
            iinj_mid = None
            _record_zero_intracellular_metadata(group=group, runtime=runtime)
        else:
            iinj_mid = build_intracellular_current_density_batch(
                cohort.axons,
                runtime,
                solver_axons=cohort.solver_axons,
                target_nx=cohort.nx,
            )
            record_benchmark_metadata(
                input_format="dense",
                **benchmark_array_metadata("iinj_mid", iinj_mid, role="kernel_input"),
            )
    with benchmark_span(
        "inputs.extracellular",
        group_id=group.group_id,
        group_size=group.size,
        nt=runtime.grid.Nt,
        nx=group.nx,
    ):
        if use_zero_extracellular:
            vstim_mid = None
            dtype = np.dtype(runtime.membrane.dtype)
            skipped_shape = (group.size, runtime.grid.Nt, group.nx)
            record_benchmark_metadata(
                input_format="zero_no_context",
                skipped_dense_vstim_shape=list(skipped_shape),
                skipped_dense_vstim_nbytes=int(np.prod(skipped_shape)) * int(dtype.itemsize),
            )
        elif use_sparse_intracellular and observer_plan is not None:
            vstim_mid = build_factorized_vstim_midpoint_batch(
                cohort.representative,
                cohort.contexts,
                tsim_ms=tsim_ms,
                dt_ms=dt_ms,
                x_positions_m=cohort.x_positions_m,
                axon_y_um=cohort.axon_y_um,
                axon_z_um=cohort.axon_z_um,
                dtype_local=runtime.membrane.dtype,
            )
            if vstim_mid is None:
                vstim_mid = build_vstim_midpoint_batch(
                    cohort.representative,
                    cohort.contexts,
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                    x_positions_m=cohort.x_positions_m,
                    axon_y_um=cohort.axon_y_um,
                    axon_z_um=cohort.axon_z_um,
                    dtype_local=runtime.membrane.dtype,
                )
                record_benchmark_metadata(
                    input_format="dense",
                    **benchmark_array_metadata("vstim_mid", vstim_mid, role="kernel_input"),
                )
            else:
                extracellular_format = "factorized_footprint"
                record_benchmark_metadata(
                    input_format="factorized_footprint",
                    target_nx=vstim_mid.target_nx,
                    shared_current=vstim_mid.shared_current,
                    dense_vstim_avoided=True,
                    **benchmark_array_metadata(
                        "vstim_current_mid_A",
                        vstim_mid.current_mid_A,
                        role="kernel_input",
                    ),
                    **benchmark_array_metadata(
                        "vstim_footprint_mV_per_A",
                        vstim_mid.footprint_mV_per_A,
                        role="kernel_input",
                    ),
                )
        else:
            vstim_mid = build_vstim_midpoint_batch(
                cohort.representative,
                cohort.contexts,
                tsim_ms=tsim_ms,
                dt_ms=dt_ms,
                x_positions_m=cohort.x_positions_m,
                axon_y_um=cohort.axon_y_um,
                axon_z_um=cohort.axon_z_um,
                dtype_local=runtime.membrane.dtype,
            )
            record_benchmark_metadata(
                **benchmark_array_metadata("vstim_mid", vstim_mid, role="kernel_input")
            )
    _emit_progress(
        progress_callback,
        group,
        "lowering",
        "inputs",
        intracellular=intracellular_format,
        extracellular=extracellular_format,
        contexts=cohort.context_count,
    )
    _record_group_memory_estimate(
        group=group,
        runtime=runtime,
        cohort=cohort,
        kernel_options=kernel_options,
        intracellular_format=intracellular_format,
        extracellular_format=extracellular_format,
        include_vstim_previous=False,
    )
    _emit_progress(
        progress_callback,
        group,
        "kernel",
        "compiling JAX kernel if needed",
        recording=kernel_options.recording.mode,
        time_chunk_steps=kernel_options.time_chunk_steps,
    )
    with benchmark_span(
        "kernel.enqueue",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        recording_mode=kernel_options.recording.mode,
    ):
        out = SingleCableVStimBatchKernel(
            runtime=runtime,
            Cm_uF_cm2=_group_cm_uF_cm2(group, runtime),
            has_driven_extracellular=cohort.context_count > 0,
        ).run(
            intracellular_current_density_mid=iinj_mid,
            extracellular_potential_mid_mV=vstim_mid,
            options=kernel_options,
            observers=observer_plan,
            progress_callback=progress_callback,
        )
        if out.Vm is not None:
            record_benchmark_metadata(
                **benchmark_array_metadata("Vm", out.Vm, role="kernel_output")
            )
    _emit_progress(
        progress_callback,
        group,
        "kernel",
        "solving JAX kernel",
    )
    with benchmark_span(
        "kernel.wait",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
    ):
        benchmark_wait(_batch_wait_target(out))
    _emit_progress(progress_callback, group, "kernel", "completed JAX kernel")
    _emit_progress(
        progress_callback,
        group,
        "result",
        "assemble batch output",
        output="observations" if out.Vm is None else "Vm",
    )
    with benchmark_span(
        "results.split_batch",
        group_id=group.group_id,
        group_size=group.size,
        recording_mode=kernel_options.recording.mode,
    ):
        return _dispatch_results_from_batch(
            group,
            Vm=out.Vm,
            t=out.t,
            observations=out.observations,
            observer_definitions=observers,
            method=_dispatch_method(group),
            batch_options=batch_options,
            kernel_batch_options=kernel_options,
        )


def _run_double_cable_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    solver_options: SolverOptions | None,
    observers: tuple[Any, ...] | None,
    progress_callback: Any = None,
) -> tuple[DispatchRecord, ...]:
    """Run a homogeneous double-cable group through full double-cable batching."""

    kernel_group = _double_cable_kernel_group(group)
    representative_item = _representative_item(group)
    representative = representative_item.simulation
    _emit_progress(progress_callback, group, "prepare", "runtime", mode="double")
    with benchmark_span(
        "runtime.prepare",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        nx=group.nx,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
    ):
        runtime = _prepare_batch_runtime(
            kernel_group,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            solver_options=solver_options,
            mode="double",
            include_extracellular=True,
            include_area=True,
        )
        _record_kernel_bucket_metadata(group=group, kernel_group=kernel_group)
        record_benchmark_metadata(
            nt=runtime.grid.Nt,
            nx=runtime.membrane.Nx,
            dtype=str(runtime.membrane.dtype),
        )
    _emit_progress(progress_callback, group, "prepare", "cohort rows")
    with benchmark_span(
        "inputs.positions",
        group_id=group.group_id,
        group_size=group.size,
        nx=group.nx,
    ):
        cohort = _prepared_cohort_for_group(kernel_group)
        record_benchmark_metadata(
            **benchmark_array_metadata(
                "x_positions_m",
                cohort.x_positions_m,
                role="positions",
            ),
            context_count=cohort.context_count,
            public_group_size=int(group.size),
            kernel_group_size=int(kernel_group.size),
            public_nx=int(group.nx),
            kernel_nx=int(kernel_group.nx),
        )
    kernel_options = _kernel_batch_options(kernel_group, batch_options, observers=observers)
    _emit_progress(
        progress_callback,
        group,
        "batch",
        "recording plan",
        recording=kernel_options.recording.mode,
        time_chunk_steps=kernel_options.time_chunk_steps,
        observers=0 if observers is None else len(observers),
        block_solver=kernel_options.double_cable_block_solver,
    )
    with benchmark_span(
        "observer.plan",
        group_id=group.group_id,
        group_size=group.size,
        recording_mode=kernel_options.recording.mode,
    ):
        observer_plan = _observer_plan_for_cohort(
            observers,
            cohort=cohort,
            dtype=runtime.membrane.dtype,
            prefer_vm_raster=kernel_options.recording.mode == "none",
        )
    intracellular_format = (
        "dense" if _has_intracellular_contexts(cohort) else "zero_no_intracellular_context"
    )
    extracellular_format = "dense"
    with benchmark_span(
        "inputs.intracellular",
        group_id=group.group_id,
        group_size=group.size,
        nt=runtime.grid.Nt,
        nx=group.nx,
    ):
        if _has_intracellular_contexts(cohort):
            iinj_mid = build_intracellular_current_density_batch(
                cohort.axons,
                runtime,
                solver_axons=cohort.solver_axons,
                target_nx=cohort.nx,
            )
            record_benchmark_metadata(
                input_format="dense",
                **benchmark_array_metadata("iinj_mid", iinj_mid, role="kernel_input"),
            )
        else:
            iinj_mid = None
            _record_zero_intracellular_metadata(group=kernel_group, runtime=runtime)
    with benchmark_span(
        "inputs.extracellular",
        group_id=group.group_id,
        group_size=group.size,
        nt=runtime.grid.Nt,
        nx=group.nx,
    ):
        if observer_plan is not None and kernel_options.recording.mode == "none":
            vstim_mid = build_factorized_vstim_midpoint_batch(
                cohort.representative,
                cohort.contexts,
                tsim_ms=tsim_ms,
                dt_ms=dt_ms,
                x_positions_m=cohort.x_positions_m,
                axon_y_um=cohort.axon_y_um,
                axon_z_um=cohort.axon_z_um,
                dtype_local=runtime.membrane.dtype,
                include_initial_previous=True,
            )
        else:
            vstim_mid = None

        if vstim_mid is None:
            vstim_mid, vstim_previous = build_vstim_midpoint_and_initial_previous_batch(
                cohort.representative,
                cohort.contexts,
                tsim_ms=tsim_ms,
                dt_ms=dt_ms,
                x_positions_m=cohort.x_positions_m,
                axon_y_um=cohort.axon_y_um,
                axon_z_um=cohort.axon_z_um,
                dtype_local=runtime.membrane.dtype,
            )
            record_benchmark_metadata(
                input_format="dense",
                **benchmark_array_metadata("vstim_mid", vstim_mid, role="kernel_input"),
                **benchmark_array_metadata(
                    "vstim_previous",
                    vstim_previous,
                    role="kernel_input",
                ),
            )
        else:
            vstim_previous = None
            extracellular_format = "factorized_footprint"
            record_benchmark_metadata(
                input_format="factorized_footprint",
                target_nx=vstim_mid.target_nx,
                shared_current=vstim_mid.shared_current,
                dense_vstim_avoided=True,
                **benchmark_array_metadata(
                    "vstim_current_mid_A",
                    vstim_mid.current_mid_A,
                    role="kernel_input",
                ),
                **benchmark_array_metadata(
                    "vstim_current_initial_previous_A",
                    vstim_mid.current_initial_previous_A,
                    role="kernel_input",
                ),
                **benchmark_array_metadata(
                    "vstim_footprint_mV_per_A",
                    vstim_mid.footprint_mV_per_A,
                    role="kernel_input",
                ),
            )
    _emit_progress(
        progress_callback,
        group,
        "lowering",
        "inputs",
        intracellular=intracellular_format,
        extracellular=extracellular_format,
        contexts=cohort.context_count,
    )
    _record_group_memory_estimate(
        group=kernel_group,
        runtime=runtime,
        cohort=cohort,
        kernel_options=kernel_options,
        intracellular_format=intracellular_format,
        extracellular_format=extracellular_format,
        include_vstim_previous=True,
    )
    record_benchmark_metadata(
        public_group_size=int(group.size),
        public_nx=int(group.nx),
    )
    _emit_progress(
        progress_callback,
        group,
        "kernel",
        "compiling JAX kernel if needed",
        recording=kernel_options.recording.mode,
        time_chunk_steps=kernel_options.time_chunk_steps,
        block_solver=kernel_options.double_cable_block_solver,
    )
    with benchmark_span(
        "kernel.enqueue",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        recording_mode=kernel_options.recording.mode,
    ):
        out = DoubleCableBatchKernel(
            runtime=runtime,
            Veinit_mV=float(getattr(representative, "Veinit", 0.0)),
            has_driven_extracellular=cohort.context_count > 0,
        ).run(
            intracellular_current_density_mid=iinj_mid,
            extracellular_potential_mid_mV=vstim_mid,
            extracellular_potential_initial_previous_mV=vstim_previous,
            options=kernel_options,
            observers=observer_plan,
            progress_callback=progress_callback,
        )
        if out.Vm is not None:
            record_benchmark_metadata(
                **benchmark_array_metadata("Vm", out.Vm, role="kernel_output")
            )
        out = _trim_batch_kernel_result(out, batch_size=group.size)
    _emit_progress(
        progress_callback,
        group,
        "kernel",
        "solving JAX kernel",
    )
    with benchmark_span(
        "kernel.wait",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
    ):
        benchmark_wait(_batch_wait_target(out))
    _emit_progress(progress_callback, group, "kernel", "completed JAX kernel")
    _emit_progress(
        progress_callback,
        group,
        "result",
        "assemble batch output",
        output="observations" if out.Vm is None else "Vm",
    )
    with benchmark_span(
        "results.split_batch",
        group_id=group.group_id,
        group_size=group.size,
        recording_mode=kernel_options.recording.mode,
    ):
        return _dispatch_results_from_batch(
            group,
            Vm=out.Vm,
            t=out.t,
            observations=out.observations,
            observer_definitions=observers,
            method=_dispatch_method(group),
            batch_options=batch_options,
            kernel_batch_options=kernel_options,
        )


def _with_batched_single_cable_runtime(
    runtime: SolverRuntime,
    group: DispatchGroup,
) -> SolverRuntime:
    """Return `runtime` with cable arrays stacked over the batch axis."""

    with benchmark_span(
        "runtime.prepare.stack_cable",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        nx=group.nx,
    ):
        cable = _stack_cable_runtime(
            group,
            dtype_local=runtime.membrane.dtype,
            include_area=False,
        )
    return replace(
        runtime,
        cable=cable,
    )


def _with_batched_double_cable_runtime(
    runtime: SolverRuntime,
    group: DispatchGroup,
    *,
    solver_options: SolverOptions | None,
) -> SolverRuntime:
    """Return `runtime` with cable and extracellular arrays stacked by row."""

    dtype_local = runtime.membrane.dtype
    with benchmark_span(
        "runtime.prepare.stack_cable",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        nx=group.nx,
    ):
        cable = _stack_cable_runtime(
            group,
            dtype_local=dtype_local,
            include_area=True,
        )
    with benchmark_span(
        "runtime.prepare.stack_extracellular",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        nx=group.nx,
    ):
        extracellular = _stack_extracellular_runtime(group, dtype_local=dtype_local)
    with benchmark_span(
        "runtime.prepare.stack_membrane",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        nx=group.nx,
    ):
        membrane = _stack_membrane_runtime(
            runtime,
            group,
            dtype_local=dtype_local,
            solver_options=solver_options,
        )
    return replace(runtime, membrane=membrane, cable=cable, extracellular=extracellular)


def _stack_extracellular_runtime(
    group: DispatchGroup,
    *,
    dtype_local: jnp.dtype,
) -> ExtracellularRuntime:
    """Stack double-cable extracellular arrays using host-side row preparation."""

    np_dtype = np.dtype(dtype_local)
    rows = tuple(
        _extracellular_runtime_numpy(
            item.solver_axon,
            dtype=np_dtype,
            target_nx=group.nx,
        )
        for item in group.items
    )
    return ExtracellularRuntime(
        Cm_abs=jnp.asarray(np.stack([row.Cm_abs for row in rows], axis=0), dtype=dtype_local),
        Cx_abs=jnp.asarray(np.stack([row.Cx_abs for row in rows], axis=0), dtype=dtype_local),
        Gx_abs=jnp.asarray(np.stack([row.Gx_abs for row in rows], axis=0), dtype=dtype_local),
        Gax_e=jnp.asarray(np.stack([row.Gax_e for row in rows], axis=0), dtype=dtype_local),
        Gax_i=jnp.asarray(np.stack([row.Gax_i for row in rows], axis=0), dtype=dtype_local),
        left_i=jnp.asarray(np.stack([row.left_i for row in rows], axis=0), dtype=dtype_local),
        right_i=jnp.asarray(np.stack([row.right_i for row in rows], axis=0), dtype=dtype_local),
        left_e=jnp.asarray(np.stack([row.left_e for row in rows], axis=0), dtype=dtype_local),
        right_e=jnp.asarray(np.stack([row.right_e for row in rows], axis=0), dtype=dtype_local),
    )


def _stack_membrane_runtime(
    runtime: SolverRuntime,
    group: DispatchGroup,
    *,
    dtype_local: jnp.dtype,
    solver_options: SolverOptions | None,
) -> MembraneRuntime:
    """Stack row-specific membrane initial states and row-selectable backends."""

    np_dtype = np.dtype(dtype_local)
    vm0_rows = np.stack(
        [
            _pad_space_array_numpy(
                np.full(
                    (int(item.solver_axon.n_compartments),),
                    float(getattr(item.simulation, "v_init", 0.0)),
                    dtype=np_dtype,
                ),
                target_nx=group.nx,
                mode="edge",
            )
            for item in group.items
        ],
        axis=0,
    )
    family_stack = _try_stack_axnode_passive_family_membrane_from_group(
        group,
        target_nx=group.nx,
        dtype_local=dtype_local,
        solver_options=solver_options,
    )
    if family_stack is not None:
        record_benchmark_metadata(
            membrane_stack_host_side=True,
            membrane_stack_source=family_stack.source,
            membrane_row_backend="axnode_passive_family",
            membrane_row_backend_branches=1,
            membrane_family_node_compartments=int(family_stack.node_count),
            membrane_family_passive_compartments=int(family_stack.passive_count),
        )
        return replace(
            runtime.membrane,
            backend=family_stack.backend,
            membrane=family_stack.membrane_static,
            Nx=group.nx,
            Vm0_mV=jnp.asarray(vm0_rows, dtype=dtype_local),
            gates0=jnp.asarray(family_stack.gates0_rows, dtype=dtype_local),
            state0=(),
            background_current=jnp.asarray(
                family_stack.background_rows,
                dtype=dtype_local,
            ),
        )
    if runtime.membrane.backend is None:
        raise RuntimeError(
            "internal AxNode/passive placeholder membrane was selected, but direct "
            "family membrane stacking did not match the dispatch group."
        )

    representative_index = next(
        (
            index
            for index, item in enumerate(group.items)
            if int(item.solver_axon.n_compartments) == int(runtime.membrane.Nx)
        ),
        None,
    )
    rows = tuple(
        runtime.membrane
        if representative_index is not None and index == representative_index
        else prepare_membrane_runtime(
            cast(Any, item.simulation),
            solver_axon=item.solver_axon,
            solver_options=solver_options,
        )
        for index, item in enumerate(group.items)
    )
    if any(row.state0 for row in rows):
        raise NotImplementedError(
            "parameter-batched double-cable membranes currently support stateless "
            "membrane components only."
        )
    if any(not row.membrane.supports_stateless_vm_only_fast_path() for row in rows):
        raise NotImplementedError(
            "parameter-batched double-cable membranes currently require membrane "
            "models with the stateless Vm-only fast path."
        )
    family_stack = _try_stack_axnode_passive_family_membrane(
        rows,
        target_nx=group.nx,
        dtype_local=dtype_local,
    )
    if family_stack is None:
        row_backend = RowIndexedICMBackend.from_backends(
            tuple(row.backend for row in rows),
            target_nx=group.nx,
        )
        gates0_rows = np.stack(
            [
                _pad_gate_array_numpy(
                    np.asarray(row.gates0, dtype=np_dtype),
                    target_nx=group.nx,
                    target_gates=row_backend.n_gates_max,
                )
                for row in rows
            ],
            axis=0,
        )
        background_rows = np.stack(
            [
                _pad_space_array_numpy(
                    np.asarray(row.background_current, dtype=np_dtype),
                    target_nx=group.nx,
                    mode="zero",
                )
                for row in rows
            ],
            axis=0,
        )
        membrane_static = runtime.membrane.membrane
        row_backend_kind = "row_indexed"
        row_backend_branches = len(rows)
        family_node_count = 0
        family_passive_count = 0
        family_source = "row_membrane_runtime"
    else:
        row_backend = family_stack.backend
        gates0_rows = family_stack.gates0_rows
        background_rows = family_stack.background_rows
        membrane_static = family_stack.membrane_static
        family_node_count = family_stack.node_count
        family_passive_count = family_stack.passive_count
        row_backend_kind = "axnode_passive_family"
        row_backend_branches = 1
        family_source = family_stack.source
    record_benchmark_metadata(
        membrane_stack_host_side=True,
        membrane_stack_source=family_source,
        membrane_row_backend=row_backend_kind,
        membrane_row_backend_branches=int(row_backend_branches),
        membrane_family_node_compartments=int(family_node_count),
        membrane_family_passive_compartments=int(family_passive_count),
    )
    return replace(
        runtime.membrane,
        backend=row_backend,
        membrane=membrane_static,
        Nx=group.nx,
        Vm0_mV=jnp.asarray(vm0_rows, dtype=dtype_local),
        gates0=jnp.asarray(gates0_rows, dtype=dtype_local),
        state0=(),
        background_current=jnp.asarray(background_rows, dtype=dtype_local),
    )


def _try_stack_axnode_passive_family_membrane_from_group(
    group: DispatchGroup,
    *,
    target_nx: int,
    dtype_local: jnp.dtype,
    solver_options: SolverOptions | None,
) -> _FamilyMembraneStack | None:
    """Fast-path MRG-like rows directly from solver axon membrane descriptions."""

    from axonscope.channel_models.axnode import AxnodeICM

    if not group.items:
        return None
    np_dtype = np.dtype(dtype_local)
    node_model: AxnodeICM | None = None
    node_signature: tuple[Any, ...] | None = None
    gates_rows: list[np.ndarray] = []
    background_rows: list[np.ndarray] = []
    node_gates_by_vm0: dict[float, np.ndarray] = {}
    node_count = 0
    passive_count = 0
    for item in group.items:
        solver_axon = item.solver_axon
        row_nx = int(solver_axon.n_compartments)
        if row_nx > int(target_nx):
            return None
        membrane_models = tuple(solver_axon.membrane_models)
        if len(membrane_models) != row_nx:
            return None
        encoded = np.zeros((int(target_nx), 7), dtype=np_dtype)
        vm0 = float(getattr(item.simulation, "v_init", 0.0))
        for compartment_index, model in enumerate(membrane_models):
            kind = str(getattr(model, "kind", "")).lower()
            if kind == "axnode":
                signature = model._static_signature()
                if node_signature is None:
                    compiled = compile_membrane_model(
                        model,
                        solver_options=solver_options,
                    )
                    if not isinstance(compiled, AxnodeICM):
                        return None
                    if not compiled.supports_stateless_vm_only_fast_path():
                        return None
                    node_signature = signature
                    node_model = compiled
                elif signature != node_signature:
                    return None
                row_node_gates = node_gates_by_vm0.get(vm0)
                if row_node_gates is None:
                    assert node_model is not None
                    row_node_gates = _axnode_initial_gates_numpy(
                        node_model,
                        vm0_mV=vm0,
                        dtype=np_dtype,
                    )
                    node_gates_by_vm0[vm0] = row_node_gates
                encoded[compartment_index, :4] = row_node_gates
                encoded[compartment_index, 6] = np_dtype.type(1.0)
                node_count += 1
            elif kind == "passive":
                params = getattr(model, "params", {})
                try:
                    rm = float(params["Rm"])
                    e_leak = float(params["EL"])
                except (KeyError, TypeError, ValueError):
                    return None
                g_leak = np_dtype.type(1e3) / np.maximum(
                    np_dtype.type(rm),
                    np_dtype.type(1e-18),
                )
                encoded[compartment_index, 4] = g_leak
                encoded[compartment_index, 5] = g_leak * np_dtype.type(e_leak)
                passive_count += 1
            else:
                return None
        gates_rows.append(encoded)
        background_rows.append(np.zeros((int(target_nx),), dtype=np_dtype))
    if node_model is None or passive_count == 0:
        return None
    backend = _AxNodePassiveFamilyICMBackend(
        node_model=node_model,
        target_nx=int(target_nx),
        dtype=dtype_local,
    )
    return _FamilyMembraneStack(
        backend=backend,
        gates0_rows=np.stack(gates_rows, axis=0),
        background_rows=np.stack(background_rows, axis=0),
        membrane_static=node_model,
        node_count=node_count,
        passive_count=passive_count,
        source="solver_axon_membrane_models",
    )


def _try_stack_axnode_passive_family_membrane(
    rows: tuple[MembraneRuntime, ...],
    *,
    target_nx: int,
    dtype_local: jnp.dtype,
) -> _FamilyMembraneStack | None:
    """Encode MRG-like AxNode/passive rows as dynamic row parameters."""

    from axonscope.channel_models.axnode import AxnodeICM
    from axonscope.channel_models.passive import PassiveICM

    if not rows:
        return None
    np_dtype = np.dtype(dtype_local)
    node_model: AxnodeICM | None = None
    node_signature: tuple[Any, ...] | None = None
    gates_rows: list[np.ndarray] = []
    background_rows: list[np.ndarray] = []
    node_count = 0
    passive_count = 0
    for row in rows:
        backend = row.backend
        if not isinstance(backend, HeterogeneousICMBackend):
            return None
        row_nx = int(backend.Nx)
        if row_nx > int(target_nx):
            return None
        row_background = np.asarray(row.background_current, dtype=np_dtype)
        if row_background.shape != (row_nx,) or not np.allclose(row_background, 0.0):
            return None
        row_gates = np.asarray(row.gates0, dtype=np_dtype)
        if row_gates.shape[0] != row_nx:
            return None
        encoded = np.zeros((int(target_nx), 7), dtype=np_dtype)
        gate_cols = min(4, int(row_gates.shape[1]))
        if gate_cols:
            encoded[:row_nx, :gate_cols] = row_gates[:row_nx, :gate_cols]
        for compartment_index, model in enumerate(backend.icm_vec):
            if isinstance(model, AxnodeICM):
                if row_gates.shape[1] < 4:
                    return None
                signature = model._static_signature()
                if node_signature is None:
                    node_signature = signature
                    node_model = model
                elif signature != node_signature:
                    return None
                encoded[compartment_index, 6] = np_dtype.type(1.0)
                node_count += 1
            elif isinstance(model, PassiveICM):
                g_leak = np.asarray(model.g_bar, dtype=np_dtype)[0]
                e_leak = np.asarray(model.E_rev, dtype=np_dtype)[0]
                encoded[compartment_index, 4] = g_leak
                encoded[compartment_index, 5] = g_leak * e_leak
                passive_count += 1
            else:
                return None
        gates_rows.append(encoded)
        background_rows.append(np.zeros((int(target_nx),), dtype=np_dtype))
    if node_model is None or passive_count == 0:
        return None
    backend = _AxNodePassiveFamilyICMBackend(
        node_model=node_model,
        target_nx=int(target_nx),
        dtype=dtype_local,
    )
    return _FamilyMembraneStack(
        backend=backend,
        gates0_rows=np.stack(gates_rows, axis=0),
        background_rows=np.stack(background_rows, axis=0),
        membrane_static=node_model,
        node_count=node_count,
        passive_count=passive_count,
        source="row_membrane_runtime",
    )


def _axnode_initial_gates_numpy(
    node_model: Any,
    *,
    vm0_mV: float,
    dtype: np.dtype,
) -> np.ndarray:
    """NumPy equivalent of AxnodeICM.init_gates for fast host preparation."""

    v = dtype.type(vm0_mV)
    celsius = dtype.type(float(node_model.celsius))

    def safe_exp(x: np.generic) -> np.generic:
        return dtype.type(0.0) if x < dtype.type(-100.0) else dtype.type(np.exp(x))

    def vtrap(
        a_value: float,
        b_value: float,
        c_value: float,
        *,
        flipped: bool = False,
    ) -> np.generic:
        a = dtype.type(a_value)
        b = dtype.type(b_value)
        c = dtype.type(c_value)
        z = (v + b) / c
        num = a * (-(v + b) if flipped else (v + b))
        exp_arg = (v + b) / c if flipped else (-(v + b) / c)
        den = dtype.type(1.0) - safe_exp(exp_arg)
        if abs(float(z)) < 1e-6:
            return a * c
        return num / den

    q10_1 = dtype.type(2.2) ** ((celsius - dtype.type(20.0)) / dtype.type(10.0))
    q10_2 = dtype.type(2.9) ** ((celsius - dtype.type(20.0)) / dtype.type(10.0))
    q10_3 = dtype.type(3.0) ** ((celsius - dtype.type(36.0)) / dtype.type(10.0))

    a_mp = q10_1 * vtrap(0.01, 27.0, 10.2)
    b_mp = q10_1 * vtrap(0.00025, 34.0, 10.0, flipped=True)

    a_m = q10_1 * vtrap(1.86, 21.4, 10.3)
    b_m = q10_1 * vtrap(0.086, 25.7, 9.16, flipped=True)

    a_h = q10_2 * vtrap(0.062, 114.0, 11.0, flipped=True)
    b_h = q10_2 * dtype.type(2.3) / (
        dtype.type(1.0) + safe_exp(-(v + dtype.type(31.8)) / dtype.type(13.4))
    )

    v2 = v + dtype.type(80.0)
    a_s = q10_3 * dtype.type(0.3) / (
        safe_exp((v2 - dtype.type(27.0)) / dtype.type(-5.0)) + dtype.type(1.0)
    )
    b_s = q10_3 * dtype.type(0.03) / (
        safe_exp((v2 + dtype.type(10.0)) / dtype.type(-1.0)) + dtype.type(1.0)
    )

    alpha = np.asarray([a_mp, a_m, a_h, a_s], dtype=dtype)
    beta = np.asarray([b_mp, b_m, b_h, b_s], dtype=dtype)
    return alpha / np.maximum(alpha + beta, dtype.type(1e-12))


def _stack_cable_runtime(
    group: DispatchGroup,
    *,
    dtype_local: jnp.dtype,
    include_area: bool,
) -> CableRuntime:
    """Stack row-specific cable arrays into one batched runtime."""

    np_dtype = np.dtype(dtype_local)
    lower_rows: list[np.ndarray] = []
    diag_rows: list[np.ndarray] = []
    upper_rows: list[np.ndarray] = []
    area_rows: list[np.ndarray] = []
    for item in group.items:
        lower, diag, upper = _diffusion_operator_coeffs_numpy(
            item.solver_axon,
            dtype=np_dtype,
        )
        lower_rows.append(
            _pad_space_array_numpy(lower, target_nx=group.nx, mode="zero")
        )
        diag_rows.append(
            _pad_space_array_numpy(diag, target_nx=group.nx, mode="zero")
        )
        upper_rows.append(
            _pad_space_array_numpy(upper, target_nx=group.nx, mode="zero")
        )
        area_rows.append(
            _pad_space_array_numpy(
                _compartment_area_cm2_numpy(item.solver_axon, dtype=np_dtype)
                if include_area
                else np.zeros((item.solver_axon.n_compartments,), dtype=np_dtype),
                target_nx=group.nx,
                mode="edge",
            )
        )
    return CableRuntime(
        lower=jnp.asarray(np.stack(lower_rows, axis=0), dtype=dtype_local),
        diag=jnp.asarray(np.stack(diag_rows, axis=0), dtype=dtype_local),
        upper=jnp.asarray(np.stack(upper_rows, axis=0), dtype=dtype_local),
        area_cm2=jnp.asarray(np.stack(area_rows, axis=0), dtype=dtype_local),
    )


def _cable_runtime_from_numpy_arrays(
    axon: Any,
    *,
    dtype_local: jnp.dtype,
    include_area: bool,
) -> CableRuntime:
    """Build one cable runtime using host arrays before a single JAX transfer."""

    np_dtype = np.dtype(dtype_local)
    lower, diag, upper = _diffusion_operator_coeffs_numpy(axon, dtype=np_dtype)
    if include_area:
        area = _compartment_area_cm2_numpy(axon, dtype=np_dtype)
    else:
        area = np.zeros((axon.n_compartments,), dtype=np_dtype)
    return CableRuntime(
        lower=jnp.asarray(lower, dtype=dtype_local),
        diag=jnp.asarray(diag, dtype=dtype_local),
        upper=jnp.asarray(upper, dtype=dtype_local),
        area_cm2=jnp.asarray(area, dtype=dtype_local),
    )


def _diffusion_operator_coeffs_numpy(
    axon: Any,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy equivalent of ``diffusion_operator_coeffs`` for batch preparation."""

    nx = int(axon.n_compartments)
    lower = np.zeros((nx,), dtype=dtype)
    diag = np.zeros((nx,), dtype=dtype)
    upper = np.zeros((nx,), dtype=dtype)

    if bool(getattr(axon, "has_heterogeneous_cable_properties", False)):
        lengths_cm = np.asarray(axon.compartment_lengths_um, dtype=dtype) * dtype.type(1e-4)
        diam_um = np.asarray(axon.diam_um, dtype=dtype)
        ra_ohm_cm = np.asarray(axon.Ra_ohm_cm, dtype=dtype)
        cm_uF_cm2 = np.asarray(axon.Cm_uF_cm2, dtype=dtype)

        area_cm2 = np.pi * (diam_um * dtype.type(1e-4)) * lengths_cm
        radius_cm = dtype.type(0.5) * diam_um * dtype.type(1e-4)
        cross_section_cm2 = np.pi * radius_cm**2
        left_half_cm = dtype.type(0.5) * lengths_cm[:-1]
        right_half_cm = dtype.type(0.5) * lengths_cm[1:]
        edge_resistance_ohm = (
            ra_ohm_cm[:-1] * left_half_cm / cross_section_cm2[:-1]
            + ra_ohm_cm[1:] * right_half_cm / cross_section_cm2[1:]
        )
        gax_i_mS = dtype.type(1e3) / np.maximum(edge_resistance_ohm, dtype.type(1e-18))
        cm_abs_uF = cm_uF_cm2 * area_cm2
        lower[1:] = gax_i_mS / cm_abs_uF[1:]
        upper[:-1] = gax_i_mS / cm_abs_uF[:-1]
        diag = -(lower + upper)
        return lower, diag.astype(dtype, copy=False), upper

    h = np.asarray(axon.h_cm, dtype=dtype)
    diffusion = _uniform_diffusion_coefficient_numpy(axon, dtype=dtype)
    if nx >= 2:
        left_coef = dtype.type(2.0) * diffusion / (h[0] ** 2)
        right_coef = dtype.type(2.0) * diffusion / (h[-1] ** 2)
        diag[0] = -left_coef
        upper[0] = left_coef
        lower[-1] = right_coef
        diag[-1] = -right_coef
    if nx > 2:
        h_left = h[:-1]
        h_right = h[1:]
        denom = h_left + h_right
        lower[1:-1] = dtype.type(2.0) * diffusion / (h_left * denom)
        diag[1:-1] = -dtype.type(2.0) * diffusion / (h_left * h_right)
        upper[1:-1] = dtype.type(2.0) * diffusion / (h_right * denom)
    return lower, diag, upper


def _uniform_diffusion_coefficient_numpy(axon: Any, *, dtype: np.dtype) -> np.generic:
    diam_um = np.mean(np.asarray(axon.diam_um, dtype=dtype))
    ra_ohm_cm = np.mean(np.asarray(axon.Ra_ohm_cm, dtype=dtype))
    cm_uF_cm2 = np.mean(np.asarray(axon.Cm_uF_cm2, dtype=dtype))
    radius_cm = dtype.type(0.5) * diam_um * dtype.type(1e-4)
    cm = dtype.type(2.0) * np.pi * radius_cm * cm_uF_cm2 * dtype.type(1e-6)
    ra = ra_ohm_cm / (np.pi * radius_cm**2)
    return dtype.type(1.0) / (ra * cm) / dtype.type(1000.0)


def _compartment_area_cm2_numpy(axon: Any, *, dtype: np.dtype) -> np.ndarray:
    diam = np.asarray(axon.diam_um, dtype=dtype)
    length_cm = np.asarray(axon.compartment_lengths_um, dtype=dtype) * dtype.type(1e-4)
    return np.asarray(np.pi * (diam * dtype.type(1e-4)) * length_cm, dtype=dtype)


def _extracellular_runtime_numpy(
    axon: Any,
    *,
    dtype: np.dtype,
    target_nx: int,
) -> ExtracellularRuntime:
    """Build one padded double-cable extracellular row with NumPy arrays."""

    area = _compartment_area_cm2_numpy(axon, dtype=dtype)
    cm_uF_cm2 = np.asarray(axon.Cm_uF_cm2, dtype=dtype)
    Cm_abs = cm_uF_cm2 * area

    xg = np.asarray(axon.xg_S_cm2, dtype=dtype)
    xc = np.asarray(axon.xc_uF_cm2, dtype=dtype)
    xraxial = np.asarray(axon.xraxial_MOhm_per_cm, dtype=dtype)
    dx_cm = np.asarray(axon.dx_cm, dtype=dtype)

    Cx_abs = xc * area
    Gx_abs = (xg * dtype.type(1e3)) * area

    if int(axon.n_compartments) <= 1:
        Gax_e = np.zeros((0,), dtype=dtype)
    else:
        R_edge_MOhm = (
            xraxial[:-1] * (dtype.type(0.5) * dx_cm[:-1])
            + xraxial[1:] * (dtype.type(0.5) * dx_cm[1:])
        )
        Gax_e = dtype.type(1e-3) / np.maximum(R_edge_MOhm, dtype.type(1e-18))

    lower, _, upper = _diffusion_operator_coeffs_numpy(axon, dtype=dtype)
    Gax_i = dtype.type(0.5) * (upper[:-1] * Cm_abs[:-1] + lower[1:] * Cm_abs[1:])
    left_i = np.concatenate([np.zeros((1,), dtype=dtype), Gax_i])
    right_i = np.concatenate([Gax_i, np.zeros((1,), dtype=dtype)])
    left_e = np.concatenate([np.zeros((1,), dtype=dtype), Gax_e])
    right_e = np.concatenate([Gax_e, np.zeros((1,), dtype=dtype)])

    return ExtracellularRuntime(
        Cm_abs=_pad_space_array_numpy(Cm_abs, target_nx=target_nx, mode="edge"),
        Cx_abs=_pad_space_array_numpy(Cx_abs, target_nx=target_nx, mode="edge"),
        Gx_abs=_pad_space_array_numpy(Gx_abs, target_nx=target_nx, mode="edge"),
        Gax_e=_pad_edge_array_numpy(Gax_e, target_nx=target_nx),
        Gax_i=_pad_edge_array_numpy(Gax_i, target_nx=target_nx),
        left_i=_pad_space_array_numpy(left_i, target_nx=target_nx, mode="zero"),
        right_i=_pad_space_array_numpy(right_i, target_nx=target_nx, mode="zero"),
        left_e=_pad_space_array_numpy(left_e, target_nx=target_nx, mode="zero"),
        right_e=_pad_space_array_numpy(right_e, target_nx=target_nx, mode="zero"),
    )


def _pad_space_array_numpy(
    values: np.ndarray,
    *,
    target_nx: int,
    mode: str,
) -> np.ndarray:
    """Pad one host compartment-space array to ``target_nx``."""

    arr = np.asarray(values)
    pad_count = int(target_nx) - int(arr.shape[0])
    if pad_count < 0:
        raise ValueError(
            f"target_nx must be >= array width, got target_nx={target_nx}, "
            f"width={arr.shape[0]}."
        )
    if pad_count == 0:
        return arr
    if mode == "zero":
        pad_values = np.zeros((pad_count,), dtype=arr.dtype)
    elif mode == "edge":
        pad_values = np.broadcast_to(arr[-1], (pad_count,)).astype(arr.dtype, copy=False)
    else:
        raise ValueError(f"unknown padding mode: {mode!r}.")
    return np.concatenate([arr, pad_values], axis=0)


def _pad_edge_array_numpy(values: np.ndarray, *, target_nx: int) -> np.ndarray:
    """Pad one host edge-space array with zero coupling into padded compartments."""

    arr = np.asarray(values)
    target_edges = max(int(target_nx) - 1, 0)
    pad_count = target_edges - int(arr.shape[0])
    if pad_count < 0:
        raise ValueError(
            f"target_nx={target_nx} is too small for edge width={arr.shape[0]}."
        )
    if pad_count == 0:
        return arr
    return np.concatenate([arr, np.zeros((pad_count,), dtype=arr.dtype)], axis=0)


def _pad_gate_array_numpy(
    values: np.ndarray,
    *,
    target_nx: int,
    target_gates: int,
) -> np.ndarray:
    """Pad one host gate matrix to shared spatial and gate widths."""

    arr = np.asarray(values)
    pad_nx = int(target_nx) - int(arr.shape[0])
    pad_gates = int(target_gates) - int(arr.shape[1])
    if pad_nx < 0 or pad_gates < 0:
        raise ValueError(
            "target_nx/target_gates must be >= gate array shape, got "
            f"targets=({target_nx}, {target_gates}) and shape={arr.shape}."
        )
    if pad_gates:
        arr = np.concatenate(
            [arr, np.zeros((arr.shape[0], pad_gates), dtype=arr.dtype)],
            axis=1,
        )
    if pad_nx:
        arr = np.concatenate(
            [arr, np.zeros((pad_nx, arr.shape[1]), dtype=arr.dtype)],
            axis=0,
        )
    return arr


def _group_cm_uF_cm2(group: DispatchGroup, runtime: SolverRuntime) -> jnp.ndarray:
    """Return shared or row-specific membrane capacitance density arrays."""

    dtype_local = runtime.membrane.dtype
    if group.geometry_shared:
        return jnp.asarray(runtime.axon.Cm_uF_cm2, dtype=dtype_local)
    return jnp.stack(
        [
            jnp.asarray(item.solver_axon.Cm_uF_cm2, dtype=dtype_local)
            for item in group.items
        ],
        axis=0,
    )


def _posthoc_observations_for_row(
    item: DispatchItem,
    *,
    row_vm: np.ndarray,
    t: Any,
    record_indices: tuple[int, ...] | None,
    observer_definitions: tuple[Any, ...],
) -> dict[str, Any]:
    """Evaluate observers post-hoc when Vm was intentionally recorded."""

    row_result = SimResult(
        item.simulation.axon,
        row_vm,
        np.asarray(t),
        record_indices=record_indices,
        simulation=item.simulation,
    )
    observations = {}
    for definition in observer_definitions:
        analysis = row_result.analyze(definition)
        observations[analysis.name] = analysis
    return observations


def _trim_batch_kernel_result(
    out: BatchKernelResult,
    *,
    batch_size: int,
) -> BatchKernelResult:
    """Drop backend-only padded batch rows before public result assembly."""

    size = int(batch_size)
    Vm = None if out.Vm is None else out.Vm[:size]
    observations = _trim_observations_batch(out.observations, batch_size=size)
    return BatchKernelResult(Vm=Vm, t=out.t, observations=observations)


def _trim_observations_batch(
    observations: dict[str, Any] | None,
    *,
    batch_size: int,
) -> dict[str, Any] | None:
    if observations is None:
        return None
    return {
        name: _trim_observation_batch(value, batch_size=batch_size)
        for name, value in observations.items()
    }


def _trim_observation_batch(value: Any, *, batch_size: int) -> Any:
    current_size = getattr(value, "batch_size", None)
    if current_size is None or int(current_size) <= int(batch_size):
        return value
    slice_batch = getattr(value, "slice_batch", None)
    concat_batch = getattr(type(value), "concat_batch", None)
    if callable(slice_batch) and callable(concat_batch):
        return concat_batch([slice_batch(index) for index in range(int(batch_size))])
    raise TypeError(
        f"cannot trim padded observation result type {type(value).__name__}."
    )


def _dispatch_results_from_batch(
    group: DispatchGroup,
    *,
    Vm: jnp.ndarray | None,
    t: jnp.ndarray,
    observations: dict[str, Any] | None,
    observer_definitions: tuple[Any, ...] | None,
    method: str,
    batch_options: BatchOptions,
    kernel_batch_options: BatchOptions,
) -> tuple[DispatchRecord, ...]:
    """Convert a batched solver output to compact dispatch records."""

    vm_values = None if Vm is None else np.asarray(Vm)
    kernel_indices = kernel_batch_options.recording.indices_for(group.nx)
    kernel_record_indices = (
        None if kernel_indices is None else tuple(int(value) for value in kernel_indices)
    )

    if vm_values is None and observations is not None:
        return (
            DispatchCohortResult(
                indices=tuple(item.index for item in group.items),
                axons=tuple(item.simulation.axon for item in group.items),
                simulations=tuple(item.simulation for item in group.items),
                Vm=None,
                t=t,
                group_id=group.group_id,
                method=method,
                record_indices=tuple(None for _ in group.items),
                observations=observations,
                group_size=group.size,
                batch_kind=group.batch_kind,
                geometry_shared=group.geometry_shared,
                has_padding=group.has_padding,
            ),
        )

    results = []
    for row_index, item in enumerate(group.items):
        original_nx = int(item.solver_axon.n_compartments)
        row_vm = None if vm_values is None else vm_values[row_index]
        record_indices = kernel_record_indices

        if row_vm is not None and kernel_indices is None:
            row_vm = row_vm[:, :original_nx]
            requested_indices = batch_options.recording.indices_for(original_nx)
            if requested_indices is not None:
                row_vm = np.take(row_vm, np.asarray(requested_indices), axis=1)
                record_indices = tuple(int(value) for value in requested_indices)
            else:
                record_indices = None
        if row_vm is None:
            record_indices = None

        row_observations = observations
        observations_are_batched = row_observations is not None
        if row_observations is None and observer_definitions and row_vm is not None:
            row_observations = _posthoc_observations_for_row(
                item,
                row_vm=row_vm,
                t=t,
                record_indices=record_indices,
                observer_definitions=observer_definitions,
            )
            observations_are_batched = False

        results.append(
            DispatchResult(
                index=item.index,
                axon=item.simulation.axon,
                simulation=item.simulation,
                Vm=row_vm,
                t=t,
                group_id=group.group_id,
                method=method,
                record_indices=record_indices,
                observations=row_observations,
                observations_are_batched=observations_are_batched,
                group_size=group.size,
                batch_kind=group.batch_kind,
                geometry_shared=group.geometry_shared,
                has_padding=group.has_padding,
            )
        )
    return tuple(results)


__all__ = ["run_jax_batch_group"]
