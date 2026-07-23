"""JAX batch runtime construction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from axonfleet.benchmarking import benchmark_span, record_benchmark_metadata
from axonfleet.dispatcher.plan import DispatchGroup, DispatchItem
from axonfleet.preparation.axon_rows import MaterializedAxonRows
from axonfleet.preparation.membrane_rows import MembraneRowPlan
from axonfleet.runtime.group_preparation import (
    group_runtime_signature,
    representative_item,
    runtime_context_cache_key,
)
from axonfleet.runtime.jax.preparation.caches import (
    get_batch_runtime,
    get_batch_static_runtime,
    store_batch_runtime,
    store_batch_static_runtime,
)
from axonfleet.runtime.jax.preparation.stacking import (
    _cable_runtime_from_numpy_arrays,
    _with_batched_double_cable_runtime,
    _with_batched_membrane_runtime,
    _with_batched_single_cable_runtime,
)
from axonfleet.runtime.jax.preparation.base import (
    prepare_membrane_runtime,
    prepare_simulation_grid,
    prepare_solver_runtime,
)
from axonfleet.runtime.jax.types import SolverRuntime


def prepare_batch_runtime(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    mode: str,
    include_extracellular: bool,
    include_area: bool,
    materialized_axons: MaterializedAxonRows,
    membrane_rows: MembraneRowPlan,
    runtime_context: Any | None = None,
) -> SolverRuntime:
    runtime_scope = runtime_context_cache_key(runtime_context)
    group_signature = group_runtime_signature(group)
    cache_key = (
        "batch_runtime_v1",
        mode,
        group_signature,
        runtime_scope,
        float(tsim_ms),
        float(dt_ms),
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
                )
                record_benchmark_metadata(batch_base_runtime_kind="full")
            else:
                runtime = _prepare_parameter_batch_base_runtime(
                    group,
                    item,
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                    include_area=include_area,
                )
                record_benchmark_metadata(batch_base_runtime_kind="parameter_minimal")
        if not group.geometry_shared:
            if mode == "double":
                runtime = _with_batched_double_cable_runtime(
                    runtime,
                    group,
                    materialized_axons,
                    membrane_rows,
                )
            else:
                runtime = _with_batched_single_cable_runtime(
                    runtime,
                    group,
                    materialized_axons,
                    membrane_rows,
                )
        elif len({item.membrane_signature for item in group.items}) > 1:
            runtime = _with_batched_membrane_runtime(
                runtime,
                group,
                membrane_rows,
            )
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


def _prepare_parameter_batch_base_runtime(
    group: DispatchGroup,
    item: DispatchItem,
    *,
    tsim_ms: float,
    dt_ms: float,
    include_area: bool,
) -> SolverRuntime:
    """Prepare only representative fields that survive parameter batching."""

    simulation = cast(Any, item.simulation)
    solver_axon = item.solver_axon
    membrane = prepare_membrane_runtime(
        simulation,
        solver_axon=solver_axon,
    )
    record_benchmark_metadata(batch_base_membrane_kind="full_representative")
    grid = prepare_simulation_grid(tsim_ms, dt_ms, membrane.dtype)
    cable = _cable_runtime_from_numpy_arrays(
        solver_axon,
        dtype_local=membrane.dtype,
        include_area=include_area,
    )
    return SolverRuntime(
        axon=solver_axon,
        grid=grid,
        membrane=membrane,
        cable=cable,
        extracellular=None,
    )


__all__ = [
    "prepare_batch_runtime",
]
