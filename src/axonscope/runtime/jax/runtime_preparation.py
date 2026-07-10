"""Runtime preparation and host-side stacking for JAX batch execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from axonscope.benchmarking import benchmark_span, record_benchmark_metadata
from axonscope.runtime.jax.runtime import (
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
from axonscope.runtime.jax.runtime_caches import (
    get_batch_runtime,
    get_batch_static_runtime,
    get_prepared_cohort,
    store_batch_runtime,
    store_batch_static_runtime,
    store_prepared_cohort,
)
from axonscope.runtime.jax.membrane_backend import (
    GatedLeakStackMembraneBackend,
    HeterogeneousMembraneBackend,
    RowIndexedMembraneBackend,
    membrane_backend_model,
    membrane_static_signature,
)
from axonscope.dispatcher.plan import DispatchGroup, DispatchItem
from axonscope.preparation.cohort import PreparedCohort
from axonscope.preparation.runtime_batches import extracellular_stimulation_rows
from axonscope.solvers.options import SolverOptions


@dataclass(frozen=True)
class GatedLeakMembraneStack:
    backend: GatedLeakStackMembraneBackend
    gates0_rows: np.ndarray
    background_rows: np.ndarray
    membrane_static: Any
    gated_count: int
    leak_count: int
    source: str


@dataclass(frozen=True)
class _GatedLeakMember:
    role: str
    model: Any | None = None
    leak_g: float = 0.0
    leak_ge: float = 0.0


@dataclass(frozen=True)
class _ExtracellularRuntimeNumpy:
    Cm_abs: np.ndarray
    Cx_abs: np.ndarray
    Gx_abs: np.ndarray
    Gax_e: np.ndarray
    Gax_i: np.ndarray
    left_i: np.ndarray
    right_i: np.ndarray
    left_e: np.ndarray
    right_e: np.ndarray


_EXTRACELLULAR_SPACE_FIELDS = (
    "Cm_abs",
    "Cx_abs",
    "Gx_abs",
    "left_i",
    "right_i",
    "left_e",
    "right_e",
)
_EXTRACELLULAR_EDGE_FIELDS = ("Gax_e", "Gax_i")


def representative_item(group: DispatchGroup) -> DispatchItem:
    """Return the row used to compile the shared runtime."""

    for item in group.items:
        if int(item.solver_axon.n_compartments) == int(group.nx):
            return item
    return group.items[0]


def prepare_batch_runtime(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    solver_options: SolverOptions | None,
    mode: str,
    include_extracellular: bool,
    include_area: bool,
    backend_context: Any | None = None,
) -> SolverRuntime:
    backend_scope = _backend_context_cache_key(backend_context)
    cache_key = (
        "batch_runtime_v1",
        mode,
        _group_runtime_signature(group),
        backend_scope,
        float(tsim_ms),
        float(dt_ms),
        repr(solver_options),
        bool(include_extracellular),
        bool(include_area),
    )
    cached = get_batch_runtime(cache_key)
    if cached is not None:
        record_benchmark_metadata(batch_runtime_cache="hit")
        return cached

    static_cache_key = (
        "batch_static_runtime_v1",
        mode,
        _group_runtime_signature(group),
        backend_scope,
        repr(solver_options),
        bool(include_extracellular),
        bool(include_area),
    )
    runtime = get_batch_static_runtime(static_cache_key)
    static_cache_state = "hit"
    if runtime is None:
        item = representative_item(group)
        with benchmark_span(
            "runtime.prepare.base_runtime",
            group_id=group.group_id,
            group_size=group.size,
            mode=mode,
            nx=group.nx,
        ):
            if group.geometry_shared:
                runtime = prepare_solver_runtime(
                    cast(Any, item.simulation),
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                    solver_axon=item.solver_axon,
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
                    item,
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
        store_batch_static_runtime(static_cache_key, runtime)
        static_cache_state = "miss"

    runtime = replace(
        runtime,
        grid=prepare_simulation_grid(tsim_ms, dt_ms, runtime.membrane.dtype),
    )

    store_batch_runtime(cache_key, runtime)
    record_benchmark_metadata(
        batch_runtime_cache="miss",
        batch_static_runtime_cache=static_cache_state,
    )
    return runtime


def prepared_cohort_for_group(group: DispatchGroup) -> PreparedCohort:
    cache_key = ("prepared_cohort_v1", _group_preparation_signature(group))
    cached = get_prepared_cohort(cache_key)
    if cached is not None:
        record_benchmark_metadata(prepared_cohort_cache="hit")
        return _with_current_stimulation_rows(cached, group)

    cohort = PreparedCohort.from_dispatch_group(group)
    store_prepared_cohort(cache_key, cohort)
    record_benchmark_metadata(prepared_cohort_cache="miss")
    return cohort


def _with_current_stimulation_rows(
    cohort: PreparedCohort,
    group: DispatchGroup,
) -> PreparedCohort:
    axons = tuple(item.simulation for item in group.items)
    stimulations = extracellular_stimulation_rows(axons)
    representative = representative_item(group).simulation
    if (
        _same_objects(cohort.axons, axons)
        and _same_stimulation_rows(cohort.stimulations, stimulations)
        and cohort.representative is representative
    ):
        return cohort
    return replace(
        cohort,
        representative=representative,
        axons=axons,
        stimulations=stimulations,
    )


def _same_objects(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return len(left) == len(right) and all(
        a is b for a, b in zip(left, right, strict=True)
    )


def _same_stimulation_rows(
    left: tuple[tuple[Any, ...], ...],
    right: tuple[tuple[Any, ...], ...],
) -> bool:
    if len(left) != len(right):
        return False
    return all(_same_objects(a, b) for a, b in zip(left, right, strict=True))


def _backend_context_cache_key(context: Any | None) -> tuple[Any, ...] | None:
    """Return the backend/runtime policy part of runtime cache identity."""

    if context is None:
        return None
    policy = getattr(context, "policy", None)
    runtime = getattr(policy, "runtime", None)
    device_request = getattr(policy, "device", None)
    precision = getattr(policy, "precision", None)
    solver_engine = getattr(context, "solver_engine", None)
    resolved_device = getattr(context, "device", None)
    resolved_device_key = None
    if resolved_device is not None:
        resolved_device_key = (
            getattr(resolved_device, "platform", None),
            getattr(resolved_device, "id", None),
            str(resolved_device),
        )
    return (
        "backend_context_v1",
        getattr(runtime, "value", runtime),
        None
        if device_request is None
        else (
            getattr(device_request, "kind", None),
            getattr(device_request, "index", None),
        ),
        getattr(context, "platform", None),
        resolved_device_key,
        None
        if precision is None
        else (
            precision.state_dtype,
            precision.solver_dtype,
            precision.accumulation_dtype,
        ),
        None
        if solver_engine is None
        else (
            getattr(solver_engine, "name", None),
            getattr(solver_engine, "platform", None),
            getattr(solver_engine, "double_cable_block_solver", None),
            getattr(solver_engine, "allow_internal_double_cable_block_solver", None),
            getattr(solver_engine, "tiled_thomas_block_b", None),
        ),
    )


def stack_extracellular_runtime(
    group: DispatchGroup,
    *,
    dtype_local: jnp.dtype,
) -> ExtracellularRuntime:
    """Stack double-cable extracellular arrays using host-side row preparation."""

    np_dtype = np.dtype(dtype_local)
    batch_size = len(group.items)
    target_nx = int(group.nx)
    target_edges = max(target_nx - 1, 0)
    space_rows = np.empty(
        (len(_EXTRACELLULAR_SPACE_FIELDS), batch_size, target_nx),
        dtype=np_dtype,
    )
    edge_rows = np.empty(
        (len(_EXTRACELLULAR_EDGE_FIELDS), batch_size, target_edges),
        dtype=np_dtype,
    )
    row_cache: dict[tuple[Any, ...], _ExtracellularRuntimeNumpy] = {}

    for row_index, item in enumerate(group.items):
        cache_key = (
            item.cable_signature,
            int(item.solver_axon.n_compartments),
            target_nx,
            np_dtype.str,
        )
        row = row_cache.get(cache_key)
        if row is None:
            row = _extracellular_runtime_numpy(
                item.solver_axon,
                dtype=np_dtype,
                target_nx=target_nx,
            )
            row_cache[cache_key] = row
        for field_index, field_name in enumerate(_EXTRACELLULAR_SPACE_FIELDS):
            space_rows[field_index, row_index] = getattr(row, field_name)
        for field_index, field_name in enumerate(_EXTRACELLULAR_EDGE_FIELDS):
            edge_rows[field_index, row_index] = getattr(row, field_name)

    space = jnp.asarray(space_rows, dtype=dtype_local)
    edge = jnp.asarray(edge_rows, dtype=dtype_local)
    record_benchmark_metadata(
        extracellular_stack_rows=batch_size,
        extracellular_stack_unique_rows=len(row_cache),
        extracellular_stack_cache_hits=batch_size - len(row_cache),
    )
    return ExtracellularRuntime(
        Cm_abs=space[0],
        Cx_abs=space[1],
        Gx_abs=space[2],
        Gax_e=edge[0],
        Gax_i=edge[1],
        left_i=space[3],
        right_i=space[4],
        left_e=space[5],
        right_e=space[6],
    )


def try_stack_gated_leak_membrane_from_group(
    group: DispatchGroup,
    *,
    target_nx: int,
    dtype_local: jnp.dtype,
    solver_options: SolverOptions | None,
) -> GatedLeakMembraneStack | None:
    """Fast-path gated/leak rows from structural membrane capabilities."""

    if not group.items:
        return None
    np_dtype = np.dtype(dtype_local)
    gated_model: Any | None = None
    gated_signature: tuple[Any, ...] | None = None
    gated_count = 0
    leak_count = 0
    members_by_row: list[list[_GatedLeakMember]] = []
    compiled_by_signature: dict[tuple[Any, ...], Any] = {}

    for item in group.items:
        solver_axon = item.solver_axon
        row_nx = int(solver_axon.n_compartments)
        if row_nx > int(target_nx):
            return None
        membrane_models = tuple(solver_axon.membrane_models)
        if len(membrane_models) != row_nx:
            return None
        row_members: list[_GatedLeakMember] = []
        for compartment_index, model in enumerate(membrane_models):
            _ = compartment_index
            signature = membrane_static_signature(model)
            compiled = compiled_by_signature.get(signature)
            if compiled is None:
                compiled = compile_membrane_model(model, solver_options=solver_options)
                compiled_by_signature[signature] = compiled
            executable = membrane_backend_model(compiled)
            member = _gated_leak_member(executable, dtype=np_dtype)
            if member is None:
                return None
            if member.role == "gated":
                executable_signature = membrane_static_signature(executable)
                if gated_signature is None:
                    gated_signature = executable_signature
                    gated_model = executable
                elif executable_signature != gated_signature:
                    return None
                gated_count += 1
            else:
                leak_count += 1
            row_members.append(member)
        members_by_row.append(row_members)

    if gated_model is None or gated_count == 0 or leak_count == 0:
        return None
    gated_gate_count = len(gated_model.gate_names())
    gated_channel_count = int(gated_model.g_bar.shape[0])
    gates_rows, background_rows = _encode_gated_leak_members(
        members_by_row,
        group=group,
        gated_model=gated_model,
        gated_gate_count=gated_gate_count,
        target_nx=target_nx,
        dtype_local=dtype_local,
        np_dtype=np_dtype,
    )
    backend = GatedLeakStackMembraneBackend(
        gated_model=gated_model,
        target_nx=int(target_nx),
        dtype=dtype_local,
        gated_gate_count=gated_gate_count,
        gated_channel_count=gated_channel_count,
    )
    return GatedLeakMembraneStack(
        backend=backend,
        gates0_rows=gates_rows,
        background_rows=background_rows,
        membrane_static=gated_model,
        gated_count=gated_count,
        leak_count=leak_count,
        source="solver_axon_membrane_models",
    )


def _gated_leak_member(model: Any, *, dtype: np.dtype) -> _GatedLeakMember | None:
    """Classify a compiled membrane by structural capabilities, not family name."""

    if model.membrane_state_specs():
        return None
    gate_count = len(model.gate_names())
    if gate_count > 0:
        if not model.supports_stateless_vm_only_fast_path():
            return None
        return _GatedLeakMember(role="gated", model=model)
    if int(model.g_bar.shape[0]) != 1:
        return None
    g = dtype.type(np.asarray(model.g_bar, dtype=dtype)[0])
    e_rev = dtype.type(np.asarray(model.E_rev, dtype=dtype)[0])
    return _GatedLeakMember(
        role="leak",
        leak_g=float(g),
        leak_ge=float(g * e_rev),
    )


def _encode_gated_leak_members(
    members_by_row: list[list[_GatedLeakMember]],
    *,
    group: DispatchGroup,
    gated_model: Any,
    gated_gate_count: int,
    target_nx: int,
    dtype_local: jnp.dtype,
    np_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    leak_g_col = int(gated_gate_count)
    leak_ge_col = int(gated_gate_count) + 1
    gated_mask_col = int(gated_gate_count) + 2
    encoded_width = int(gated_gate_count) + 3
    gated_gates_by_vm0: dict[float, np.ndarray] = {}
    gates_rows: list[np.ndarray] = []
    background_rows: list[np.ndarray] = []
    for item, row_members in zip(group.items, members_by_row, strict=True):
        encoded = np.zeros((int(target_nx), encoded_width), dtype=np_dtype)
        vm0 = float(getattr(item.simulation, "v_init", 0.0))
        row_gated_gates = gated_gates_by_vm0.get(vm0)
        if row_gated_gates is None:
            row_gated_gates = np.asarray(
                gated_model.init_gates(
                    jnp.asarray([vm0], dtype=dtype_local),
                )[0],
                dtype=np_dtype,
            )
            gated_gates_by_vm0[vm0] = row_gated_gates
        for compartment_index, member in enumerate(row_members):
            if member.role == "gated":
                encoded[compartment_index, :gated_gate_count] = row_gated_gates
                encoded[compartment_index, gated_mask_col] = np_dtype.type(1.0)
            else:
                encoded[compartment_index, leak_g_col] = np_dtype.type(member.leak_g)
                encoded[compartment_index, leak_ge_col] = np_dtype.type(member.leak_ge)
        gates_rows.append(encoded)
        background_rows.append(np.zeros((int(target_nx),), dtype=np_dtype))
    return np.stack(gates_rows, axis=0), np.stack(background_rows, axis=0)


def group_cm_uF_cm2(group: DispatchGroup, runtime: SolverRuntime) -> jnp.ndarray:
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
    return _group_static_signature(group)


def _prepare_parameter_batch_base_runtime(
    group: DispatchGroup,
    item: DispatchItem,
    *,
    tsim_ms: float,
    dt_ms: float,
    mode: str,
    include_area: bool,
    solver_options: SolverOptions | None,
) -> SolverRuntime:
    """Prepare only representative fields that survive parameter batching."""

    simulation = cast(Any, item.simulation)
    solver_axon = item.solver_axon
    _ = mode
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
        extracellular = stack_extracellular_runtime(group, dtype_local=dtype_local)
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
    gated_leak_stack = try_stack_gated_leak_membrane_from_group(
        group,
        target_nx=group.nx,
        dtype_local=dtype_local,
        solver_options=solver_options,
    )
    if gated_leak_stack is not None:
        record_benchmark_metadata(
            membrane_stack_host_side=True,
            membrane_stack_source=gated_leak_stack.source,
            membrane_row_backend="gated_leak_stack",
            membrane_row_backend_branches=1,
            membrane_stack_gated_compartments=int(gated_leak_stack.gated_count),
            membrane_stack_leak_compartments=int(gated_leak_stack.leak_count),
        )
        return replace(
            runtime.membrane,
            backend=gated_leak_stack.backend,
            membrane=gated_leak_stack.membrane_static,
            Nx=group.nx,
            Vm0_mV=jnp.asarray(vm0_rows, dtype=dtype_local),
            gates0=jnp.asarray(gated_leak_stack.gates0_rows, dtype=dtype_local),
            state0=(),
            background_current=jnp.asarray(
                gated_leak_stack.background_rows,
                dtype=dtype_local,
            ),
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
    gated_leak_stack = _try_stack_gated_leak_membrane(
        rows,
        target_nx=group.nx,
        dtype_local=dtype_local,
    )
    row_backend: Any
    if gated_leak_stack is None:
        row_backend = RowIndexedMembraneBackend.from_backends(
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
        stack_gated_count = 0
        stack_leak_count = 0
        stack_source = "row_membrane_runtime"
    else:
        row_backend = gated_leak_stack.backend
        gates0_rows = gated_leak_stack.gates0_rows
        background_rows = gated_leak_stack.background_rows
        membrane_static = gated_leak_stack.membrane_static
        stack_gated_count = gated_leak_stack.gated_count
        stack_leak_count = gated_leak_stack.leak_count
        row_backend_kind = "gated_leak_stack"
        row_backend_branches = 1
        stack_source = gated_leak_stack.source
    record_benchmark_metadata(
        membrane_stack_host_side=True,
        membrane_stack_source=stack_source,
        membrane_row_backend=row_backend_kind,
        membrane_row_backend_branches=int(row_backend_branches),
        membrane_stack_gated_compartments=int(stack_gated_count),
        membrane_stack_leak_compartments=int(stack_leak_count),
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


def _try_stack_gated_leak_membrane(
    rows: tuple[MembraneRuntime, ...],
    *,
    target_nx: int,
    dtype_local: jnp.dtype,
) -> GatedLeakMembraneStack | None:
    """Encode gated/leak row layouts as dynamic row parameters."""

    if not rows:
        return None
    np_dtype = np.dtype(dtype_local)
    gated_model: Any | None = None
    gated_signature: tuple[Any, ...] | None = None
    gated_count = 0
    leak_count = 0
    members_by_row: list[list[_GatedLeakMember]] = []
    gates_by_row: list[np.ndarray] = []
    for row in rows:
        backend = row.backend
        if not isinstance(backend, HeterogeneousMembraneBackend):
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
        row_members: list[_GatedLeakMember] = []
        for compartment_index, model in enumerate(backend.membrane_models):
            _ = compartment_index
            member = _gated_leak_member(model, dtype=np_dtype)
            if member is None:
                return None
            if member.role == "gated":
                signature = membrane_static_signature(model)
                if gated_signature is None:
                    gated_signature = signature
                    gated_model = model
                elif signature != gated_signature:
                    return None
                gated_count += 1
            else:
                leak_count += 1
            row_members.append(member)
        members_by_row.append(row_members)
        gates_by_row.append(row_gates)
    if gated_model is None or gated_count == 0 or leak_count == 0:
        return None
    gated_gate_count = len(gated_model.gate_names())
    gated_channel_count = int(gated_model.g_bar.shape[0])
    encoded = _encode_gated_leak_runtime_members(
        members_by_row,
        gates_by_row,
        gated_gate_count=gated_gate_count,
        target_nx=target_nx,
        np_dtype=np_dtype,
    )
    if encoded is None:
        return None
    gates_rows, background_rows = encoded
    backend = GatedLeakStackMembraneBackend(
        gated_model=gated_model,
        target_nx=int(target_nx),
        dtype=dtype_local,
        gated_gate_count=gated_gate_count,
        gated_channel_count=gated_channel_count,
    )
    return GatedLeakMembraneStack(
        backend=backend,
        gates0_rows=gates_rows,
        background_rows=background_rows,
        membrane_static=gated_model,
        gated_count=gated_count,
        leak_count=leak_count,
        source="row_membrane_runtime",
    )


def _encode_gated_leak_runtime_members(
    members_by_row: list[list[_GatedLeakMember]],
    gates_by_row: list[np.ndarray],
    *,
    gated_gate_count: int,
    target_nx: int,
    np_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray] | None:
    leak_g_col = int(gated_gate_count)
    leak_ge_col = int(gated_gate_count) + 1
    gated_mask_col = int(gated_gate_count) + 2
    encoded_width = int(gated_gate_count) + 3
    gates_rows: list[np.ndarray] = []
    background_rows: list[np.ndarray] = []
    for row_members, row_gates in zip(members_by_row, gates_by_row, strict=True):
        if int(row_gates.shape[1]) < int(gated_gate_count):
            return None
        encoded = np.zeros((int(target_nx), encoded_width), dtype=np_dtype)
        for compartment_index, member in enumerate(row_members):
            if member.role == "gated":
                encoded[compartment_index, :gated_gate_count] = row_gates[
                    compartment_index,
                    :gated_gate_count,
                ]
                encoded[compartment_index, gated_mask_col] = np_dtype.type(1.0)
            else:
                encoded[compartment_index, leak_g_col] = np_dtype.type(member.leak_g)
                encoded[compartment_index, leak_ge_col] = np_dtype.type(
                    member.leak_ge
                )
        gates_rows.append(encoded)
        background_rows.append(np.zeros((int(target_nx),), dtype=np_dtype))
    return np.stack(gates_rows, axis=0), np.stack(background_rows, axis=0)


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
) -> _ExtracellularRuntimeNumpy:
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

    return _ExtracellularRuntimeNumpy(
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


__all__ = [
    "GatedLeakMembraneStack",
    "group_cm_uF_cm2",
    "prepare_batch_runtime",
    "prepared_cohort_for_group",
    "representative_item",
    "stack_extracellular_runtime",
    "try_stack_gated_leak_membrane_from_group",
]
