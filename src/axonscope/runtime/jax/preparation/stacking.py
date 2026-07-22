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
from axonscope.preparation.axon_rows import MaterializedAxonRows
from axonscope.preparation.membrane_rows import MembraneRowPlan
from axonscope.runtime.host_preparation import (
    cable_runtime_rows_numpy,
    compartment_area_cm2_numpy,
    diffusion_operator_coeffs_numpy,
    extracellular_runtime_rows_numpy,
    pad_gate_array_numpy,
    pad_space_array_numpy,
)
from axonscope.runtime.jax.membranes.backend import RowIndexedMembraneBackend
from axonscope.runtime.jax.membranes.stacking import (
    try_stack_gated_leak_membrane_from_group,
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
    tuple[weakref.ReferenceType[MaterializedAxonRows], jnp.ndarray],
] = OrderedDict()


def stack_extracellular_runtime(
    materialized_axons: MaterializedAxonRows,
    *,
    dtype_local: jnp.dtype,
) -> ExtracellularRuntime:
    """Stack double-cable extracellular arrays using host-side row preparation."""

    np_dtype = np.dtype(dtype_local)
    template_rows = extracellular_runtime_rows_numpy(
        materialized_axons,
        dtype=np_dtype,
    )
    population_rows = {
        field: jnp.asarray(
            materialized_axons.gather_space(getattr(template_rows, field)),
            dtype=dtype_local,
        )
        for field in (
            "Cm_abs",
            "Cx_abs",
            "Gx_abs",
            "Gax_e",
            "Gax_i",
            "left_i",
            "right_i",
            "left_e",
            "right_e",
        )
    }
    record_benchmark_metadata(
        extracellular_stack_rows=materialized_axons.size,
        extracellular_stack_unique_rows=materialized_axons.template_count,
        extracellular_stack_cache_hits=(
            materialized_axons.size - materialized_axons.template_count
        ),
        extracellular_stack_lowering="vectorized_template_rows",
    )
    return ExtracellularRuntime(
        **population_rows,
    )


def group_cm_uF_cm2(
    group: DispatchGroup,
    runtime: SolverRuntime,
    materialized_axons: MaterializedAxonRows,
) -> jnp.ndarray:
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
        cache_key = (id(materialized_axons), id(runtime), str(dtype_local))
        cached = _GROUP_CM_CACHE.get(cache_key)
        if cached is not None:
            ref, values = cached
            if ref() is materialized_axons:
                _GROUP_CM_CACHE.move_to_end(cache_key)
                record_benchmark_metadata(group_cm_cache="hit")
                return values
            _GROUP_CM_CACHE.pop(cache_key, None)

        np_dtype = np.dtype(dtype_local)
        host_values = materialized_axons.gather_space(
            np.asarray(materialized_axons.Cm_uF_cm2, dtype=np_dtype)
        )
        host_values = np.ascontiguousarray(host_values)
        host_values.setflags(write=False)
        values = jnp.asarray(host_values, dtype=dtype_local)
        _GROUP_CM_CACHE[cache_key] = (weakref.ref(materialized_axons), values)
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
    materialized_axons: MaterializedAxonRows,
    membrane_rows: MembraneRowPlan,
    *,
    solver_options: SolverOptions | None,
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
            materialized_axons,
            dtype_local=runtime.membrane.dtype,
            include_area=False,
        )
    membrane_signatures = {item.membrane_signature for item in group.items}
    if len(membrane_signatures) == 1:
        return replace(runtime, cable=cable)
    return replace(
        _with_batched_membrane_runtime(
            runtime,
            group,
            membrane_rows,
            solver_options=solver_options,
        ),
        cable=cable,
    )


def _with_batched_double_cable_runtime(
    runtime: SolverRuntime,
    group: DispatchGroup,
    materialized_axons: MaterializedAxonRows,
    membrane_rows: MembraneRowPlan,
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
            materialized_axons,
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
        extracellular = stack_extracellular_runtime(
            materialized_axons,
            dtype_local=dtype_local,
        )
    return replace(
        _with_batched_membrane_runtime(
            runtime,
            group,
            membrane_rows,
            solver_options=solver_options,
        ),
        cable=cable,
        extracellular=extracellular,
    )


def _with_batched_membrane_runtime(
    runtime: SolverRuntime,
    group: DispatchGroup,
    membrane_rows: MembraneRowPlan,
    *,
    solver_options: SolverOptions | None,
) -> SolverRuntime:
    """Stack row-specific membranes while preserving shared cable arrays."""

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
            membrane_rows,
            dtype_local=runtime.membrane.dtype,
            solver_options=solver_options,
        )
    return replace(runtime, membrane=membrane)


def _stack_membrane_runtime(
    runtime: SolverRuntime,
    group: DispatchGroup,
    membrane_rows: MembraneRowPlan,
    *,
    dtype_local: jnp.dtype,
    solver_options: SolverOptions | None,
) -> MembraneRuntime:
    """Stack row-specific membrane initial states and row-selectable backends."""

    np_dtype = np.dtype(dtype_local)
    with benchmark_span("runtime.prepare.membrane_vm0_rows"):
        unique_vm0 = np.asarray(
            [signature[2] for signature in membrane_rows.signatures],
            dtype=np_dtype,
        )
        row_vm0 = unique_vm0[membrane_rows.row_parameter_indices]
        vm0_rows = np.ascontiguousarray(
            np.broadcast_to(row_vm0[:, None], (membrane_rows.size, group.nx))
        )
    with benchmark_span("runtime.prepare.membrane_encode_rows"):
        gated_leak_stack = try_stack_gated_leak_membrane_from_group(
            group,
            membrane_rows,
            target_nx=group.nx,
            dtype_local=dtype_local,
            solver_options=solver_options,
            compiled_models_by_signature=_compiled_membrane_models_by_signature(
                runtime,
                group,
            ),
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
        with benchmark_span("runtime.prepare.membrane_device_arrays"):
            vm0_device = jnp.asarray(vm0_rows, dtype=dtype_local)
            gates0_device = jnp.asarray(
                gated_leak_stack.gates0_rows,
                dtype=dtype_local,
            )
            background_device = jnp.asarray(
                gated_leak_stack.background_rows,
                dtype=dtype_local,
            )
            parameter_rows_device = {
                name: jnp.asarray(values, dtype=dtype_local)
                for name, values in gated_leak_stack.parameter_rows.items()
            }
        return replace(
            runtime.membrane,
            backend=gated_leak_stack.backend,
            membrane=gated_leak_stack.membrane_static,
            Nx=group.nx,
            Vm0_mV=vm0_device,
            gates0=gates0_device,
            state0=(),
            background_current=background_device,
            parameter_rows=parameter_rows_device or None,
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
    record_benchmark_metadata(
        membrane_stack_host_side=True,
        membrane_stack_source="row_membrane_runtime",
        membrane_row_backend="row_indexed",
        membrane_row_backend_branches=len(rows),
        membrane_stack_gated_compartments=0,
        membrane_stack_leak_compartments=0,
        membrane_stack_unique_rows=0,
        membrane_stack_cache_hits=0,
    )
    return replace(
        runtime.membrane,
        backend=row_backend,
        membrane=runtime.membrane.membrane,
        Nx=group.nx,
        Vm0_mV=jnp.asarray(vm0_rows, dtype=dtype_local),
        gates0=jnp.asarray(gates0_rows, dtype=dtype_local),
        state0=(),
        background_current=jnp.asarray(background_rows, dtype=dtype_local),
    )


def _compiled_membrane_models_by_signature(
    runtime: SolverRuntime,
    group: DispatchGroup,
) -> dict[tuple[Any, ...], Any]:
    """Index representative compiled models by descriptive signature."""

    representative = next(
        (
            item
            for item in group.items
            if item.solver_axon is runtime.axon
        ),
        None,
    )
    if representative is None:
        return {}
    backend = runtime.membrane.backend
    compiled_models = getattr(backend, "membrane_models", None)
    if compiled_models is None:
        uniform_model = getattr(backend, "ion_channel", None)
        if uniform_model is None:
            return {}
        compiled_models = (uniform_model,) * len(representative.membrane_signature)
    if len(compiled_models) != len(representative.membrane_signature):
        return {}
    return {
        signature: compiled
        for signature, compiled in zip(
            representative.membrane_signature,
            compiled_models,
            strict=True,
        )
    }


def _stack_cable_runtime(
    materialized_axons: MaterializedAxonRows,
    *,
    dtype_local: jnp.dtype,
    include_area: bool,
) -> CableRuntime:
    """Stack row-specific cable arrays into one batched runtime."""

    template_rows = cable_runtime_rows_numpy(
        materialized_axons,
        dtype=np.dtype(dtype_local),
        include_area=include_area,
    )
    record_benchmark_metadata(
        cable_stack_rows=materialized_axons.size,
        cable_stack_unique_rows=materialized_axons.template_count,
        cable_stack_cache_hits=(
            materialized_axons.size - materialized_axons.template_count
        ),
        cable_stack_lowering="vectorized_template_rows",
    )
    return CableRuntime(
        lower=jnp.asarray(
            materialized_axons.gather_space(template_rows.lower),
            dtype=dtype_local,
        ),
        diag=jnp.asarray(
            materialized_axons.gather_space(template_rows.diag),
            dtype=dtype_local,
        ),
        upper=jnp.asarray(
            materialized_axons.gather_space(template_rows.upper),
            dtype=dtype_local,
        ),
        area_cm2=jnp.asarray(
            materialized_axons.gather_space(template_rows.area_cm2),
            dtype=dtype_local,
        ),
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
