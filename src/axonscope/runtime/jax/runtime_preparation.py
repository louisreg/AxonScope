"""Runtime preparation and host-side stacking for JAX batch execution."""

from __future__ import annotations

import hashlib
import weakref
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from axonscope.benchmarking import benchmark_span, record_benchmark_metadata
from axonscope.runtime.host_preparation import (
    EXTRACELLULAR_EDGE_FIELDS,
    EXTRACELLULAR_SPACE_FIELDS,
    ExtracellularRuntimeArrays,
    compartment_area_cm2_numpy,
    diffusion_operator_coeffs_numpy,
    extracellular_runtime_numpy,
    pad_gate_array_numpy,
    pad_space_array_numpy,
)
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
    get_prepared_cohort_identity,
    store_batch_runtime,
    store_batch_static_runtime,
    store_prepared_cohort,
    store_prepared_cohort_identity,
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
    unique_rows: int = 0
    cache_hits: int = 0


@dataclass(frozen=True)
class _GatedLeakMember:
    role: str
    model: Any | None = None
    leak_g: float = 0.0
    leak_ge: float = 0.0


@dataclass(frozen=True)
class _EncodedGatedLeakRow:
    gates: np.ndarray
    background: np.ndarray
    gated_model: Any
    gated_signature: tuple[Any, ...]
    gated_count: int
    leak_count: int


_GROUP_SIGNATURE_CACHE_MAX_SIZE = 128
_GROUP_STATIC_SIGNATURE_CACHE: OrderedDict[
    int,
    tuple[weakref.ReferenceType[DispatchGroup], tuple[Any, ...]],
] = OrderedDict()
_GROUP_RUNTIME_SIGNATURE_CACHE: OrderedDict[
    int,
    tuple[weakref.ReferenceType[DispatchGroup], tuple[Any, ...]],
] = OrderedDict()
_GROUP_CM_CACHE: OrderedDict[
    tuple[int, int, str],
    tuple[weakref.ReferenceType[DispatchGroup], jnp.ndarray],
] = OrderedDict()


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
    runtime_context: Any | None = None,
) -> SolverRuntime:
    runtime_scope = _runtime_context_cache_key(runtime_context)
    group_signature = _group_runtime_signature(group)
    cache_key = (
        "batch_runtime_v1",
        mode,
        group_signature,
        runtime_scope,
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
        group_signature,
        runtime_scope,
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


def prepared_cohort_for_current_group(group: DispatchGroup) -> PreparedCohort:
    """Return a cohort for an unchanged dispatch group object.

    The structural cache remains the conservative path and refreshes current
    stimulation rows. This exact-group cache is for hot execution of a dispatch
    plan that has already been validated as current by the caller.
    """

    cached = get_prepared_cohort_identity(group)
    if cached is not None:
        record_benchmark_metadata(prepared_cohort_identity_cache="hit")
        return cached

    cohort = prepared_cohort_for_group(group)
    store_prepared_cohort_identity(group, cohort)
    record_benchmark_metadata(prepared_cohort_identity_cache="miss")
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


def _runtime_context_cache_key(context: Any | None) -> tuple[Any, ...] | None:
    """Return the runtime policy part of runtime cache identity."""

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
        "runtime_context_v1",
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
            getattr(solver_engine, "single_cable_solver", None),
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
        (len(EXTRACELLULAR_SPACE_FIELDS), batch_size, target_nx),
        dtype=np_dtype,
    )
    edge_rows = np.empty(
        (len(EXTRACELLULAR_EDGE_FIELDS), batch_size, target_edges),
        dtype=np_dtype,
    )
    row_cache: dict[tuple[Any, ...], ExtracellularRuntimeArrays] = {}

    for row_index, item in enumerate(group.items):
        cache_key = (
            item.cable_signature,
            int(item.solver_axon.n_compartments),
            target_nx,
            np_dtype.str,
        )
        row = row_cache.get(cache_key)
        if row is None:
            row = extracellular_runtime_numpy(
                item.solver_axon,
                dtype=np_dtype,
                target_nx=target_nx,
            )
            row_cache[cache_key] = row
        for field_index, field_name in enumerate(EXTRACELLULAR_SPACE_FIELDS):
            space_rows[field_index, row_index] = getattr(row, field_name)
        for field_index, field_name in enumerate(EXTRACELLULAR_EDGE_FIELDS):
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
    gates_rows: list[np.ndarray] = []
    background_rows: list[np.ndarray] = []
    compiled_by_signature: dict[tuple[Any, ...], Any] = {}
    encoded_row_cache: dict[tuple[Any, ...], _EncodedGatedLeakRow] = {}
    row_cache_hits = 0

    for item in group.items:
        solver_axon = item.solver_axon
        row_nx = int(solver_axon.n_compartments)
        if row_nx > int(target_nx):
            return None
        row_key = _gated_leak_group_row_cache_key(
            item,
            target_nx=target_nx,
            np_dtype=np_dtype,
            solver_options=solver_options,
        )
        encoded = encoded_row_cache.get(row_key)
        if encoded is None:
            encoded = _encode_gated_leak_group_row(
                item,
                target_nx=target_nx,
                dtype_local=dtype_local,
                np_dtype=np_dtype,
                solver_options=solver_options,
                compiled_by_signature=compiled_by_signature,
            )
            if encoded is None:
                return None
            encoded_row_cache[row_key] = encoded
        else:
            row_cache_hits += 1
        if gated_signature is None:
            gated_signature = encoded.gated_signature
            gated_model = encoded.gated_model
        elif encoded.gated_signature != gated_signature:
            return None
        gated_count += int(encoded.gated_count)
        leak_count += int(encoded.leak_count)
        gates_rows.append(encoded.gates)
        background_rows.append(encoded.background)

    if gated_model is None or gated_count == 0 or leak_count == 0:
        return None
    gated_gate_count = len(gated_model.gate_names())
    gated_channel_count = int(gated_model.g_bar.shape[0])
    backend = GatedLeakStackMembraneBackend(
        gated_model=gated_model,
        target_nx=int(target_nx),
        dtype=dtype_local,
        gated_gate_count=gated_gate_count,
        gated_channel_count=gated_channel_count,
    )
    return GatedLeakMembraneStack(
        backend=backend,
        gates0_rows=np.stack(gates_rows, axis=0),
        background_rows=np.stack(background_rows, axis=0),
        membrane_static=gated_model,
        gated_count=gated_count,
        leak_count=leak_count,
        source="solver_axon_membrane_models",
        unique_rows=len(encoded_row_cache),
        cache_hits=row_cache_hits,
    )


def _gated_leak_group_row_cache_key(
    item: DispatchItem,
    *,
    target_nx: int,
    np_dtype: np.dtype,
    solver_options: SolverOptions | None,
) -> tuple[Any, ...]:
    return (
        "gated_leak_group_row_v1",
        item.membrane_signature,
        int(item.solver_axon.n_compartments),
        int(target_nx),
        float(getattr(item.simulation, "v_init", 0.0)),
        np_dtype.str,
        repr(solver_options),
    )


def _encode_gated_leak_group_row(
    item: DispatchItem,
    *,
    target_nx: int,
    dtype_local: jnp.dtype,
    np_dtype: np.dtype,
    solver_options: SolverOptions | None,
    compiled_by_signature: dict[tuple[Any, ...], Any],
) -> _EncodedGatedLeakRow | None:
    solver_axon = item.solver_axon
    row_nx = int(solver_axon.n_compartments)
    membrane_models = tuple(solver_axon.membrane_models)
    if len(membrane_models) != row_nx:
        return None

    gated_model: Any | None = None
    gated_signature: tuple[Any, ...] | None = None
    gated_count = 0
    leak_count = 0
    members: list[_GatedLeakMember] = []
    for model in membrane_models:
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
        members.append(member)

    if gated_model is None or gated_signature is None:
        return None
    gated_gate_count = len(gated_model.gate_names())
    leak_g_col = int(gated_gate_count)
    leak_ge_col = int(gated_gate_count) + 1
    gated_mask_col = int(gated_gate_count) + 2
    encoded_width = int(gated_gate_count) + 3
    gates = np.zeros((int(target_nx), encoded_width), dtype=np_dtype)
    vm0 = float(getattr(item.simulation, "v_init", 0.0))
    row_gated_gates = np.asarray(
        gated_model.init_gates(
            jnp.asarray([vm0], dtype=dtype_local),
        )[0],
        dtype=np_dtype,
    )
    for compartment_index, member in enumerate(members):
        if member.role == "gated":
            gates[compartment_index, :gated_gate_count] = row_gated_gates
            gates[compartment_index, gated_mask_col] = np_dtype.type(1.0)
        else:
            gates[compartment_index, leak_g_col] = np_dtype.type(member.leak_g)
            gates[compartment_index, leak_ge_col] = np_dtype.type(member.leak_ge)

    return _EncodedGatedLeakRow(
        gates=gates,
        background=np.zeros((int(target_nx),), dtype=np_dtype),
        gated_model=gated_model,
        gated_signature=gated_signature,
        gated_count=gated_count,
        leak_count=leak_count,
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
    with benchmark_span(
        "kernel.prepare_cm",
        mode=group.mode,
        group_size=group.size,
        geometry_shared=group.geometry_shared,
        dtype=str(dtype_local),
    ):
        if group.geometry_shared:
            record_benchmark_metadata(group_cm_cache="shared")
            return jnp.asarray(runtime.axon.Cm_uF_cm2, dtype=dtype_local)
        cache_key = (id(group), id(runtime), str(dtype_local))
        cached = _GROUP_CM_CACHE.get(cache_key)
        if cached is not None:
            ref, values = cached
            if ref() is group:
                _GROUP_CM_CACHE.move_to_end(cache_key)
                record_benchmark_metadata(group_cm_cache="hit")
                return values
            _GROUP_CM_CACHE.pop(cache_key, None)

        np_dtype = np.dtype(dtype_local)
        host_values = np.stack(
            [
                np.asarray(item.solver_axon.Cm_uF_cm2, dtype=np_dtype)
                for item in group.items
            ],
            axis=0,
        )
        host_values = np.ascontiguousarray(host_values)
        host_values.setflags(write=False)
        values = jnp.asarray(host_values, dtype=dtype_local)
        _GROUP_CM_CACHE[cache_key] = (weakref.ref(group), values)
        _GROUP_CM_CACHE.move_to_end(cache_key)
        while len(_GROUP_CM_CACHE) > _GROUP_SIGNATURE_CACHE_MAX_SIZE:
            _GROUP_CM_CACHE.popitem(last=False)
        record_benchmark_metadata(
            group_cm_cache="miss",
            group_cm_lowering="host_stack",
            group_cm_nbytes=int(host_values.nbytes),
        )
        return values


def _group_static_signature(group: DispatchGroup) -> tuple[Any, ...]:
    return _cached_group_signature(
        group,
        cache=_GROUP_STATIC_SIGNATURE_CACHE,
        metadata_key="group_static_signature_cache",
        builder=_build_group_static_signature,
    )


def _build_group_static_signature(group: DispatchGroup) -> tuple[Any, ...]:
    rows_digest = _digest_group_items(
        group.items,
        include_identity=True,
    )
    return (
        "dispatch_group_v3",
        group.mode,
        int(group.nx),
        bool(group.geometry_shared),
        bool(group.has_padding),
        int(group.size),
        _digest_signature_value(group.signature, cache={}),
        rows_digest,
    )


def _group_runtime_signature(group: DispatchGroup) -> tuple[Any, ...]:
    """Return a structural key for stimulation-independent solver runtimes."""

    return _cached_group_signature(
        group,
        cache=_GROUP_RUNTIME_SIGNATURE_CACHE,
        metadata_key="group_runtime_signature_cache",
        builder=_build_group_runtime_signature,
    )


def _build_group_runtime_signature(group: DispatchGroup) -> tuple[Any, ...]:
    """Build the uncached structural key for stimulation-independent runtimes."""

    rows_digest = _digest_group_items(
        group.items,
        include_identity=False,
    )
    return (
        "dispatch_group_runtime_v3",
        group.mode,
        int(group.nx),
        bool(group.geometry_shared),
        bool(group.has_padding),
        int(group.size),
        _digest_signature_value(group.signature, cache={}),
        rows_digest,
    )


def _group_preparation_signature(group: DispatchGroup) -> tuple[Any, ...]:
    return _group_static_signature(group)


def _digest_group_items(
    items: tuple[DispatchItem, ...],
    *,
    include_identity: bool,
) -> str:
    token_cache: dict[int, str] = {}
    hasher = hashlib.blake2b(digest_size=16)
    for item in items:
        _update_digest_int(hasher, int(item.index))
        if include_identity:
            _update_digest_int(hasher, id(item.simulation))
            _update_digest_int(hasher, id(item.solver_axon))
        hasher.update(_digest_signature_value(item.membrane_signature, token_cache).encode())
        hasher.update(b"\0")
        hasher.update(_digest_signature_value(item.cable_signature, token_cache).encode())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _digest_signature_value(value: Any, cache: dict[int, str]) -> str:
    cache_key = id(value)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.blake2b(repr(value).encode("utf-8"), digest_size=16).hexdigest()
    cache[cache_key] = digest
    return digest


def _update_digest_int(hasher: Any, value: int) -> None:
    hasher.update(int(value).to_bytes(8, byteorder="little", signed=False))


def _cached_group_signature(
    group: DispatchGroup,
    *,
    cache: OrderedDict[int, tuple[weakref.ReferenceType[DispatchGroup], tuple[Any, ...]]],
    metadata_key: str,
    builder: Callable[[DispatchGroup], tuple[Any, ...]],
) -> tuple[Any, ...]:
    cache_key = id(group)
    cached = cache.get(cache_key)
    if cached is not None:
        ref, signature = cached
        if ref() is group:
            cache.move_to_end(cache_key)
            record_benchmark_metadata(**{metadata_key: "hit"})
            return signature
        cache.pop(cache_key, None)

    signature = builder(group)
    cache[cache_key] = (weakref.ref(group), signature)
    cache.move_to_end(cache_key)
    while len(cache) > _GROUP_SIGNATURE_CACHE_MAX_SIZE:
        cache.popitem(last=False)
    record_benchmark_metadata(**{metadata_key: "miss"})
    return signature


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
            pad_space_array_numpy(
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
            membrane_stack_unique_rows=int(gated_leak_stack.unique_rows),
            membrane_stack_cache_hits=int(gated_leak_stack.cache_hits),
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
                pad_gate_array_numpy(
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
                pad_space_array_numpy(
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
        membrane_stack_unique_rows=int(gated_leak_stack.unique_rows)
        if gated_leak_stack is not None
        else 0,
        membrane_stack_cache_hits=int(gated_leak_stack.cache_hits)
        if gated_leak_stack is not None
        else 0,
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
        lower, diag, upper = diffusion_operator_coeffs_numpy(
            item.solver_axon,
            dtype=np_dtype,
        )
        lower_rows.append(
            pad_space_array_numpy(lower, target_nx=group.nx, mode="zero")
        )
        diag_rows.append(
            pad_space_array_numpy(diag, target_nx=group.nx, mode="zero")
        )
        upper_rows.append(
            pad_space_array_numpy(upper, target_nx=group.nx, mode="zero")
        )
        area_rows.append(
            pad_space_array_numpy(
                compartment_area_cm2_numpy(item.solver_axon, dtype=np_dtype)
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
    lower, diag, upper = diffusion_operator_coeffs_numpy(axon, dtype=np_dtype)
    if include_area:
        area = compartment_area_cm2_numpy(axon, dtype=np_dtype)
    else:
        area = np.zeros((axon.n_compartments,), dtype=np_dtype)
    return CableRuntime(
        lower=jnp.asarray(lower, dtype=dtype_local),
        diag=jnp.asarray(diag, dtype=dtype_local),
        upper=jnp.asarray(upper, dtype=dtype_local),
        area_cm2=jnp.asarray(area, dtype=dtype_local),
    )


__all__ = [
    "GatedLeakMembraneStack",
    "group_cm_uF_cm2",
    "prepare_batch_runtime",
    "prepared_cohort_for_current_group",
    "prepared_cohort_for_group",
    "representative_item",
    "stack_extracellular_runtime",
    "try_stack_gated_leak_membrane_from_group",
]
