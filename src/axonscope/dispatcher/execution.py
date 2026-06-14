from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import jax.numpy as jnp

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.dispatcher.plan import DispatchGroup, DispatchItem, build_dispatch_plan
from axonscope.dispatcher.progress import DispatchProgress, ProgressOption
from axonscope.dispatcher.runtime_batches import (
    axon_transverse_positions_um,
    build_intracellular_current_density_batch,
    build_vstim_initial_previous_batch,
    build_vstim_midpoint_batch,
    extracellular_context_rows,
    x_positions_batch_m,
)
from axonscope.results import SimResult
from axonscope.solvers import (
    BatchOptions,
    BatchRecording,
    CrankNicholson,
    DoubleCableBatchKernel,
    SolverOptions,
    SingleCableVStimBatchKernel,
)
from axonscope.solvers.runtime import (
    CableRuntime,
    ExtracellularRuntime,
    MembraneRuntime,
    SolverRuntime,
    prepare_cable_runtime,
    prepare_extracellular_runtime,
    prepare_membrane_runtime,
    prepare_solver_runtime,
)
from axonscope.icm.backends import RowIndexedICMBackend
from axonscope.utils import units


@dataclass(frozen=True)
class DispatchResult:
    """Raw execution result for one axon before conversion to ``SimResult``."""

    index: int
    axon: Axon
    simulation: AxonInstance
    Vm: jnp.ndarray
    t: jnp.ndarray
    group_id: int
    method: str
    record_indices: tuple[int, ...] | None = None
    group_size: int = 1
    batch_kind: str = "scalar"
    geometry_shared: bool = True
    has_padding: bool = False


def run_pool(
    axons: Sequence[Axon | AxonInstance],
    *,
    tsim_ms: Any,
    dt_ms: Any,
    solver_options: SolverOptions | None = None,
    batch_options: BatchOptions | None = None,
    progress: ProgressOption = False,
) -> tuple[DispatchResult, ...]:
    """Run an axon pool and return one raw dispatch result per input simulation.

    Public code should generally call ``axonscope.simulate_pool`` so these raw
    dispatch results are converted to ``SimResult`` objects. Plain numeric times
    are interpreted as milliseconds; Pint-like quantities are converted at this
    boundary. ``progress`` enables optional Rich/plain progress reporting at the
    dispatch-group level and, for chunked batch runs, at the kernel-chunk level.
    """

    if not axons:
        raise ValueError("axons must contain at least one Axon.")
    tsim_ms = units.to_ms(tsim_ms)
    dt_ms = units.to_ms(dt_ms)
    if tsim_ms <= 0.0:
        raise ValueError("tsim_ms must be > 0.")
    if dt_ms <= 0.0:
        raise ValueError("dt_ms must be > 0.")

    resolved_batch_options = (
        BatchOptions.full() if batch_options is None else batch_options
    )
    plan = build_dispatch_plan(axons)

    results: list[DispatchResult | None] = [None] * len(plan.items)
    with DispatchProgress(progress, plan) as progress_reporter:
        for group in plan.groups:
            progress_reporter.start_group(group)
            if _can_run_batch_group(group):
                group_results = _run_batch_group(
                    group,
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                    batch_options=resolved_batch_options,
                    solver_options=solver_options,
                    progress_callback=progress_reporter.kernel_callback(group),
                )
            else:
                group_results = _run_scalar_group(
                    group,
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                    solver_options=solver_options,
                )
                callback = progress_reporter.kernel_callback(group)
                if callback is not None:
                    callback(1, 1)
            progress_reporter.finish_group(group)
            for result in group_results:
                results[result.index] = result

    if any(result is None for result in results):
        raise RuntimeError("pool dispatch did not produce all axon results.")
    return tuple(result for result in results if result is not None)


def _run_scalar_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    solver_options: SolverOptions | None,
) -> tuple[DispatchResult, ...]:
    """Execute a dispatch group through scalar solves."""

    solver = CrankNicholson(solver_options=solver_options)
    return tuple(
        _dispatch_result_from_sim(
            item,
            solver.solve(item.simulation, tsim=tsim_ms, dt=dt_ms),
            group_id=group.group_id,
        )
        for item in group.items
    )


def _can_run_batch_group(group: DispatchGroup) -> bool:
    """Return whether a dispatch group can use the current batch kernels."""

    if group.size < 2:
        return False
    return group.mode in {"single", "double"}


def _run_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    solver_options: SolverOptions | None,
    progress_callback: Any = None,
) -> tuple[DispatchResult, ...]:
    """Execute one compatible group through a batch solver kernel."""

    if group.mode == "double":
        return _run_double_cable_batch_group(
            group,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            batch_options=batch_options,
            solver_options=solver_options,
            progress_callback=progress_callback,
        )
    return _run_single_cable_batch_group(
        group,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        batch_options=batch_options,
        solver_options=solver_options,
        progress_callback=progress_callback,
    )


def _dispatch_method(group: DispatchGroup) -> str:
    """Return the public diagnostic label for a dispatch group."""

    if group.size < 2:
        return "scalar"
    prefix = "batch" if group.geometry_shared else "parameter-batch"
    if group.mode == "double":
        return f"{prefix}-double-cable"
    return f"{prefix}-single-cable"


def _representative_item(group: DispatchGroup) -> DispatchItem:
    """Return the row used to compile the shared runtime."""

    for item in group.items:
        if int(item.solver_axon.n_compartments) == int(group.nx):
            return item
    return group.items[0]


def _kernel_batch_options(group: DispatchGroup, options: BatchOptions) -> BatchOptions:
    """Return solver-kernel options, recording full traces for padded groups."""

    if not group.has_padding:
        return options
    return replace(
        options,
        recording=BatchRecording.full(),
    )


def _run_single_cable_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    solver_options: SolverOptions | None,
    progress_callback: Any = None,
) -> tuple[DispatchResult, ...]:
    """Run a homogeneous single-cable group through imposed-field batching."""

    representative = _representative_item(group).simulation
    runtime = prepare_solver_runtime(
        representative,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=False,
        precompute_extracellular=False,
        solver_options=solver_options,
    )
    if not group.geometry_shared:
        runtime = _with_batched_single_cable_runtime(runtime, group)
    axons = tuple(item.simulation for item in group.items)
    iinj_mid = build_intracellular_current_density_batch(
        axons,
        runtime,
        solver_axons=tuple(item.solver_axon for item in group.items),
        target_nx=group.nx,
    )
    vstim_mid = build_vstim_midpoint_batch(
        representative,
        extracellular_context_rows(axons),
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        x_positions_m=x_positions_batch_m(axons, target_nx=group.nx),
        axon_y_um=axon_transverse_positions_um(axons)[0],
        axon_z_um=axon_transverse_positions_um(axons)[1],
        dtype_local=runtime.membrane.dtype,
    )
    kernel_options = _kernel_batch_options(group, batch_options)
    out = SingleCableVStimBatchKernel(
        runtime=runtime,
        Cm_uF_cm2=_group_cm_uF_cm2(group, runtime),
    ).run(
        intracellular_current_density_mid=iinj_mid,
        extracellular_potential_mid_mV=vstim_mid,
        options=kernel_options,
        progress_callback=progress_callback,
    )
    return _dispatch_results_from_batch(
        group,
        Vm=out.Vm,
        t=out.t,
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
    progress_callback: Any = None,
) -> tuple[DispatchResult, ...]:
    """Run a homogeneous double-cable group through full double-cable batching."""

    representative = _representative_item(group).simulation
    runtime = prepare_solver_runtime(
        representative,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        include_extracellular=True,
        include_area=True,
        precompute_intracellular=False,
        precompute_extracellular=False,
        solver_options=solver_options,
    )
    if not group.geometry_shared:
        runtime = _with_batched_double_cable_runtime(
            runtime,
            group,
            solver_options=solver_options,
        )
    axons = tuple(item.simulation for item in group.items)
    iinj_mid = build_intracellular_current_density_batch(
        axons,
        runtime,
        solver_axons=tuple(item.solver_axon for item in group.items),
        target_nx=group.nx,
    )
    contexts = extracellular_context_rows(axons)
    x_positions = x_positions_batch_m(axons, target_nx=group.nx)
    axon_y_um, axon_z_um = axon_transverse_positions_um(axons)
    vstim_mid = build_vstim_midpoint_batch(
        representative,
        contexts,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        x_positions_m=x_positions,
        axon_y_um=axon_y_um,
        axon_z_um=axon_z_um,
        dtype_local=runtime.membrane.dtype,
    )
    vstim_previous = build_vstim_initial_previous_batch(
        representative,
        contexts,
        dt_ms=dt_ms,
        x_positions_m=x_positions,
        axon_y_um=axon_y_um,
        axon_z_um=axon_z_um,
        dtype_local=runtime.membrane.dtype,
    )
    kernel_options = _kernel_batch_options(group, batch_options)
    out = DoubleCableBatchKernel(
        runtime=runtime,
        Veinit_mV=float(getattr(representative, "Veinit", 0.0)),
    ).run(
        intracellular_current_density_mid=iinj_mid,
        extracellular_potential_mid_mV=vstim_mid,
        extracellular_potential_initial_previous_mV=vstim_previous,
        options=kernel_options,
        progress_callback=progress_callback,
    )
    return _dispatch_results_from_batch(
        group,
        Vm=out.Vm,
        t=out.t,
        method=_dispatch_method(group),
        batch_options=batch_options,
        kernel_batch_options=kernel_options,
    )


def _with_batched_single_cable_runtime(
    runtime: SolverRuntime,
    group: DispatchGroup,
) -> SolverRuntime:
    """Return `runtime` with cable arrays stacked over the batch axis."""

    return replace(
        runtime,
        cable=_stack_cable_runtime(
            group,
            dtype_local=runtime.membrane.dtype,
            include_area=False,
        ),
    )


def _with_batched_double_cable_runtime(
    runtime: SolverRuntime,
    group: DispatchGroup,
    *,
    solver_options: SolverOptions | None,
) -> SolverRuntime:
    """Return `runtime` with cable and extracellular arrays stacked by row."""

    dtype_local = runtime.membrane.dtype
    cable = _stack_cable_runtime(
        group,
        dtype_local=dtype_local,
        include_area=True,
    )
    extracellular_rows = [
        _pad_extracellular_runtime(
            prepare_extracellular_runtime(item.solver_axon, dtype_local, cable_row),
            target_nx=group.nx,
        )
        for item, cable_row in zip(
            group.items,
            _row_cable_runtimes(group, dtype_local=dtype_local, include_area=True),
            strict=True,
        )
    ]
    extracellular = ExtracellularRuntime(
        Cm_abs=jnp.stack([row.Cm_abs for row in extracellular_rows], axis=0),
        Cx_abs=jnp.stack([row.Cx_abs for row in extracellular_rows], axis=0),
        Gx_abs=jnp.stack([row.Gx_abs for row in extracellular_rows], axis=0),
        Gax_e=jnp.stack([row.Gax_e for row in extracellular_rows], axis=0),
        Gax_i=jnp.stack([row.Gax_i for row in extracellular_rows], axis=0),
        left_i=jnp.stack([row.left_i for row in extracellular_rows], axis=0),
        right_i=jnp.stack([row.right_i for row in extracellular_rows], axis=0),
        left_e=jnp.stack([row.left_e for row in extracellular_rows], axis=0),
        right_e=jnp.stack([row.right_e for row in extracellular_rows], axis=0),
    )
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

    rows = tuple(
        prepare_membrane_runtime(
            item.simulation,
            solver_axon=item.solver_axon,
            solver_options=solver_options,
        )
        for item in group.items
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
    row_backend = RowIndexedICMBackend.from_backends(
        tuple(row.backend for row in rows),
        target_nx=group.nx,
    )
    return replace(
        runtime.membrane,
        backend=row_backend,
        Nx=group.nx,
        Vm0_mV=jnp.stack(
            [
                _pad_space_array(row.Vm0_mV, target_nx=group.nx, mode="edge")
                for row in rows
            ],
            axis=0,
        ),
        gates0=jnp.stack(
            [
                _pad_gate_array(
                    row.gates0,
                    target_nx=group.nx,
                    target_gates=row_backend.n_gates_max,
                )
                for row in rows
            ],
            axis=0,
        ),
        state0=(),
        background_current=jnp.stack(
            [
                _pad_space_array(row.background_current, target_nx=group.nx, mode="zero")
                for row in rows
            ],
            axis=0,
        ),
    )


def _row_cable_runtimes(
    group: DispatchGroup,
    *,
    dtype_local: jnp.dtype,
    include_area: bool,
) -> tuple[CableRuntime, ...]:
    """Return one cable runtime per row in a dispatch group."""

    return tuple(
        prepare_cable_runtime(
            item.solver_axon,
            dtype_local,
            include_area=include_area,
        )
        for item in group.items
    )


def _stack_cable_runtime(
    group: DispatchGroup,
    *,
    dtype_local: jnp.dtype,
    include_area: bool,
) -> CableRuntime:
    """Stack row-specific cable arrays into one batched runtime."""

    rows = _row_cable_runtimes(
        group,
        dtype_local=dtype_local,
        include_area=include_area,
    )
    return CableRuntime(
        lower=jnp.stack(
            [_pad_space_array(row.lower, target_nx=group.nx, mode="zero") for row in rows],
            axis=0,
        ),
        diag=jnp.stack(
            [_pad_space_array(row.diag, target_nx=group.nx, mode="zero") for row in rows],
            axis=0,
        ),
        upper=jnp.stack(
            [_pad_space_array(row.upper, target_nx=group.nx, mode="zero") for row in rows],
            axis=0,
        ),
        area_cm2=jnp.stack(
            [_pad_space_array(row.area_cm2, target_nx=group.nx, mode="edge") for row in rows],
            axis=0,
        ),
    )


def _pad_space_array(
    values: jnp.ndarray,
    *,
    target_nx: int,
    mode: str,
) -> jnp.ndarray:
    """Pad one compartment-space array to ``target_nx``."""

    arr = jnp.asarray(values)
    pad_count = int(target_nx) - int(arr.shape[0])
    if pad_count < 0:
        raise ValueError(
            f"target_nx must be >= array width, got target_nx={target_nx}, "
            f"width={arr.shape[0]}."
        )
    if pad_count == 0:
        return arr
    if mode == "zero":
        pad_values = jnp.zeros((pad_count,), dtype=arr.dtype)
    elif mode == "edge":
        pad_values = jnp.broadcast_to(arr[-1], (pad_count,))
    else:
        raise ValueError(f"unknown padding mode: {mode!r}.")
    return jnp.concatenate([arr, pad_values], axis=0)


def _pad_edge_array(values: jnp.ndarray, *, target_nx: int) -> jnp.ndarray:
    """Pad one edge-space array with zero coupling into padded compartments."""

    arr = jnp.asarray(values)
    target_edges = max(int(target_nx) - 1, 0)
    pad_count = target_edges - int(arr.shape[0])
    if pad_count < 0:
        raise ValueError(
            f"target_nx={target_nx} is too small for edge width={arr.shape[0]}."
        )
    if pad_count == 0:
        return arr
    return jnp.concatenate([arr, jnp.zeros((pad_count,), dtype=arr.dtype)], axis=0)


def _pad_gate_array(
    values: jnp.ndarray,
    *,
    target_nx: int,
    target_gates: int,
) -> jnp.ndarray:
    """Pad one gate matrix to shared spatial and gate widths."""

    arr = jnp.asarray(values)
    pad_nx = int(target_nx) - int(arr.shape[0])
    pad_gates = int(target_gates) - int(arr.shape[1])
    if pad_nx < 0 or pad_gates < 0:
        raise ValueError(
            "target_nx/target_gates must be >= gate array shape, got "
            f"targets=({target_nx}, {target_gates}) and shape={arr.shape}."
        )
    if pad_gates:
        arr = jnp.concatenate(
            [arr, jnp.zeros((arr.shape[0], pad_gates), dtype=arr.dtype)],
            axis=1,
        )
    if pad_nx:
        arr = jnp.concatenate(
            [arr, jnp.zeros((pad_nx, arr.shape[1]), dtype=arr.dtype)],
            axis=0,
        )
    return arr


def _pad_extracellular_runtime(
    runtime: ExtracellularRuntime,
    *,
    target_nx: int,
) -> ExtracellularRuntime:
    """Pad double-cable extracellular arrays to a shared batch width."""

    return ExtracellularRuntime(
        Cm_abs=_pad_space_array(runtime.Cm_abs, target_nx=target_nx, mode="edge"),
        Cx_abs=_pad_space_array(runtime.Cx_abs, target_nx=target_nx, mode="edge"),
        Gx_abs=_pad_space_array(runtime.Gx_abs, target_nx=target_nx, mode="edge"),
        Gax_e=_pad_edge_array(runtime.Gax_e, target_nx=target_nx),
        Gax_i=_pad_edge_array(runtime.Gax_i, target_nx=target_nx),
        left_i=_pad_space_array(runtime.left_i, target_nx=target_nx, mode="zero"),
        right_i=_pad_space_array(runtime.right_i, target_nx=target_nx, mode="zero"),
        left_e=_pad_space_array(runtime.left_e, target_nx=target_nx, mode="zero"),
        right_e=_pad_space_array(runtime.right_e, target_nx=target_nx, mode="zero"),
    )


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


def _dispatch_results_from_batch(
    group: DispatchGroup,
    *,
    Vm: jnp.ndarray,
    t: jnp.ndarray,
    method: str,
    batch_options: BatchOptions,
    kernel_batch_options: BatchOptions,
) -> tuple[DispatchResult, ...]:
    """Split a batched solver output into per-axon dispatch results."""

    kernel_indices = kernel_batch_options.recording.indices_for(group.nx)
    kernel_record_indices = (
        None if kernel_indices is None else tuple(int(value) for value in kernel_indices)
    )
    results = []
    for row_index, item in enumerate(group.items):
        original_nx = int(item.solver_axon.n_compartments)
        row_vm = Vm[row_index]
        record_indices = kernel_record_indices

        if kernel_indices is None:
            row_vm = row_vm[:, :original_nx]
            requested_indices = batch_options.recording.indices_for(original_nx)
            if requested_indices is not None:
                row_vm = jnp.take(row_vm, jnp.asarray(requested_indices), axis=1)
                record_indices = tuple(int(value) for value in requested_indices)
            else:
                record_indices = None

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
                group_size=group.size,
                batch_kind=group.batch_kind,
                geometry_shared=group.geometry_shared,
                has_padding=group.has_padding,
            )
        )
    return tuple(results)


def _dispatch_result_from_sim(
    item: DispatchItem,
    sim: SimResult,
    *,
    group_id: int,
) -> DispatchResult:
    """Convert a scalar ``SimResult`` to a raw dispatch result."""

    return DispatchResult(
        index=item.index,
        axon=item.simulation.axon,
        simulation=item.simulation,
        Vm=sim.Vm,
        t=sim.t,
        group_id=group_id,
        method="scalar",
        record_indices=None,
        group_size=1,
        batch_kind="scalar",
        geometry_shared=True,
        has_padding=False,
    )


__all__ = [
    "DispatchResult",
    "run_pool",
]
