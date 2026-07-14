"""JAX row stacking for batch runtimes."""

from __future__ import annotations

import weakref
from collections import OrderedDict
from dataclasses import replace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from axonscope.benchmarking import benchmark_span, record_benchmark_metadata
from axonscope.dispatcher.plan import DispatchGroup
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
from axonscope.runtime.jax.membranes.backend import RowIndexedMembraneBackend
from axonscope.runtime.jax.membranes.stacking import (
    try_stack_gated_leak_membrane_from_group,
    try_stack_gated_leak_membrane_from_runtime_rows,
)
from axonscope.runtime.jax.preparation.base import prepare_membrane_runtime
from axonscope.runtime.jax.types import (
    CableRuntime,
    ExtracellularRuntime,
    MembraneRuntime,
    SolverRuntime,
)
from axonscope.solvers.options import SolverOptions


_GROUP_CM_CACHE_MAX_SIZE = 128
_GROUP_CM_CACHE: OrderedDict[
    tuple[int, int, str],
    tuple[weakref.ReferenceType[DispatchGroup], jnp.ndarray],
] = OrderedDict()


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
        while len(_GROUP_CM_CACHE) > _GROUP_CM_CACHE_MAX_SIZE:
            _GROUP_CM_CACHE.popitem(last=False)
        record_benchmark_metadata(
            group_cm_cache="miss",
            group_cm_lowering="host_stack",
            group_cm_nbytes=int(host_values.nbytes),
        )
        return values


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
            membrane_stack_host_leak_models=int(
                gated_leak_stack.host_leak_model_count
            ),
            membrane_stack_jax_compiled_models=int(
                gated_leak_stack.jax_compiled_model_count
            ),
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
    gated_leak_stack = try_stack_gated_leak_membrane_from_runtime_rows(
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
    "group_cm_uF_cm2",
    "stack_extracellular_runtime",
]
