"""JAX execution for prepared dispatcher groups."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from axonscope.benchmarking.hotpaths import (
    benchmark_array_metadata,
    benchmark_span,
    benchmark_wait,
    record_benchmark_metadata,
)
from axonscope.dispatcher.plan import DispatchGroup, DispatchItem
from axonscope.dispatcher.results import DispatchResult
from axonscope.backends.jax.input_batches import (
    build_intracellular_current_density_batch,
    build_sparse_intracellular_current_density_batch,
    build_vstim_initial_previous_batch,
    build_vstim_midpoint_batch,
    can_build_sparse_intracellular_current_density_batch,
)
from axonscope.icm.backends import RowIndexedICMBackend
from axonscope.preparation.cohort import PreparedCohort
from axonscope.results import SimResult
from axonscope.solvers.batch_kernels import (
    DoubleCableBatchKernel,
    SingleCableVStimBatchKernel,
)
from axonscope.solvers.observer_runtime import build_solver_observer_plan
from axonscope.solvers.options import BatchOptions, BatchRecording, SolverOptions
from axonscope.solvers.runtime import (
    CableRuntime,
    ExtracellularRuntime,
    MembraneRuntime,
    SolverRuntime,
    prepare_extracellular_runtime,
    prepare_membrane_runtime,
    prepare_solver_runtime,
)


def run_jax_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    solver_options: SolverOptions | None,
    observers: tuple[Any, ...] | None = None,
    progress_callback: Any = None,
) -> tuple[DispatchResult, ...]:
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

    if group.size < 2:
        return "scalar"
    prefix = "batch" if group.geometry_shared else "parameter-batch"
    if group.mode == "double":
        return f"{prefix}-double-cable"
    return f"{prefix}-single-cable"


def _batch_wait_target(out: Any) -> Any:
    """Return a JAX/NumPy object that synchronizes a batch kernel result."""

    if out.Vm is not None:
        return out.Vm
    if not out.observations:
        raise RuntimeError("batch kernel produced neither Vm nor observations.")
    first = next(iter(out.observations.values()))
    return first.values


def _observer_plan_for_cohort(
    observers: tuple[Any, ...] | None,
    *,
    cohort: PreparedCohort,
    dtype: Any,
) -> Any:
    """Lower public observers for one compatible prepared cohort."""

    if observers is None:
        return None
    positions_um = np.asarray(cohort.x_positions_m[0], dtype=float) * 1e6
    return build_solver_observer_plan(
        observers,
        positions_um=positions_um,
        dtype=dtype,
    )


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


def _run_single_cable_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    solver_options: SolverOptions | None,
    observers: tuple[Any, ...] | None,
    progress_callback: Any = None,
) -> tuple[DispatchResult, ...]:
    """Run a homogeneous single-cable group through imposed-field batching."""

    representative = _representative_item(group).simulation
    with benchmark_span(
        "runtime.prepare",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        nx=group.nx,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
    ):
        runtime = prepare_solver_runtime(
            representative,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            include_extracellular=False,
            include_area=False,
            precompute_intracellular=False,
            precompute_extracellular=False,
            compile_stimulation=False,
            solver_options=solver_options,
        )
        if not group.geometry_shared:
            runtime = _with_batched_single_cable_runtime(runtime, group)
        record_benchmark_metadata(
            nt=runtime.grid.Nt,
            nx=runtime.membrane.Nx,
            dtype=str(runtime.membrane.dtype),
        )
    with benchmark_span(
        "inputs.positions",
        group_id=group.group_id,
        group_size=group.size,
        nx=group.nx,
    ):
        cohort = PreparedCohort.from_dispatch_group(group)
        record_benchmark_metadata(
            **benchmark_array_metadata(
                "x_positions_m",
                cohort.x_positions_m,
                role="positions",
            ),
            context_count=cohort.context_count,
        )
    kernel_options = _kernel_batch_options(group, batch_options)
    observer_plan = _observer_plan_for_cohort(
        observers,
        cohort=cohort,
        dtype=runtime.membrane.dtype,
    )
    use_sparse_intracellular = _should_use_sparse_intracellular_batch(
        group=group,
        cohort=cohort,
        kernel_options=kernel_options,
        observers=observers,
    )
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
    with benchmark_span(
        "kernel.wait",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
    ):
        benchmark_wait(_batch_wait_target(out))
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
) -> tuple[DispatchResult, ...]:
    """Run a homogeneous double-cable group through full double-cable batching."""

    representative = _representative_item(group).simulation
    with benchmark_span(
        "runtime.prepare",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        nx=group.nx,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
    ):
        runtime = prepare_solver_runtime(
            representative,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            include_extracellular=True,
            include_area=True,
            precompute_intracellular=False,
            precompute_extracellular=False,
            compile_stimulation=False,
            solver_options=solver_options,
        )
        if not group.geometry_shared:
            runtime = _with_batched_double_cable_runtime(
                runtime,
                group,
                solver_options=solver_options,
            )
        record_benchmark_metadata(
            nt=runtime.grid.Nt,
            nx=runtime.membrane.Nx,
            dtype=str(runtime.membrane.dtype),
        )
    with benchmark_span(
        "inputs.positions",
        group_id=group.group_id,
        group_size=group.size,
        nx=group.nx,
    ):
        cohort = PreparedCohort.from_dispatch_group(group)
        record_benchmark_metadata(
            **benchmark_array_metadata(
                "x_positions_m",
                cohort.x_positions_m,
                role="positions",
            ),
            context_count=cohort.context_count,
        )
    with benchmark_span(
        "inputs.intracellular",
        group_id=group.group_id,
        group_size=group.size,
        nt=runtime.grid.Nt,
        nx=group.nx,
    ):
        iinj_mid = build_intracellular_current_density_batch(
            cohort.axons,
            runtime,
            solver_axons=cohort.solver_axons,
            target_nx=cohort.nx,
        )
        record_benchmark_metadata(
            **benchmark_array_metadata("iinj_mid", iinj_mid, role="kernel_input")
        )
    with benchmark_span(
        "inputs.extracellular",
        group_id=group.group_id,
        group_size=group.size,
        nt=runtime.grid.Nt,
        nx=group.nx,
    ):
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
        vstim_previous = build_vstim_initial_previous_batch(
            cohort.representative,
            cohort.contexts,
            dt_ms=dt_ms,
            x_positions_m=cohort.x_positions_m,
            axon_y_um=cohort.axon_y_um,
            axon_z_um=cohort.axon_z_um,
            dtype_local=runtime.membrane.dtype,
        )
        record_benchmark_metadata(
            **benchmark_array_metadata("vstim_mid", vstim_mid, role="kernel_input"),
            **benchmark_array_metadata(
                "vstim_previous",
                vstim_previous,
                role="kernel_input",
            ),
        )
    kernel_options = _kernel_batch_options(group, batch_options)
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
            progress_callback=progress_callback,
        )
        if out.Vm is not None:
            record_benchmark_metadata(
                **benchmark_array_metadata("Vm", out.Vm, role="kernel_output")
            )
    with benchmark_span(
        "kernel.wait",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
    ):
        benchmark_wait(_batch_wait_target(out))
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
        _cable_runtime_from_numpy_arrays(
            item.solver_axon,
            dtype_local=dtype_local,
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
) -> tuple[DispatchResult, ...]:
    """Split a batched solver output into per-axon dispatch results."""

    vm_values = None if Vm is None else np.asarray(Vm)
    kernel_indices = kernel_batch_options.recording.indices_for(group.nx)
    kernel_record_indices = (
        None if kernel_indices is None else tuple(int(value) for value in kernel_indices)
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
