"""JAX execution for prepared dispatcher groups."""

from __future__ import annotations

from typing import Any, cast

from axonscope.benchmarking import (
    active_benchmark_session,
    benchmark_array_metadata,
    benchmark_span,
    benchmark_wait,
    record_benchmark_metadata,
)
from axonscope.dispatcher.plan import DispatchGroup
from axonscope.dispatcher.progress import ProgressEvent, ProgressStage
from axonscope.dispatcher._records import DispatchRecord
from axonscope.runtime.jax.benchmark_metadata import (
    record_extracellular_lowering_metadata,
    record_group_memory_estimate,
    record_intracellular_lowering_metadata,
)
from axonscope.runtime.jax.batch_results import (
    dispatch_results_from_batch,
    trim_batch_kernel_result,
)
from axonscope.runtime.jax.input_lowering import (
    lower_double_cable_extracellular_input,
    lower_double_cable_intracellular_input,
    lower_single_cable_extracellular_input,
    lower_single_cable_intracellular_input,
)
from axonscope.runtime.jax.output_plan import OutputPlan
from axonscope.runtime.jax.recording_lowering import (
    lower_batch_recording_options,
    lower_observers_for_cohort,
    row_recording_indices_for_group,
)
from axonscope.runtime.jax.runtime_preparation import (
    group_cm_uF_cm2,
    prepare_batch_runtime,
    prepared_cohort_for_current_group,
    representative_item,
)
from axonscope.runtime.jax.shape_bucketing import (
    double_cable_kernel_group,
    record_kernel_bucket_metadata,
)
from axonscope.runtime.jax.batch_kernels import (
    BatchKernelResult,
    DoubleCableBatchKernel,
    SingleCableVStimBatchKernel,
)
from axonscope.runtime.jax.observer_runtime import finalize_vm_raster_state
from axonscope.solvers.options import BatchOptions, SolverOptions


def run_jax_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    solver_options: SolverOptions | None,
    observers: tuple[Any, ...] | None = None,
    progress_callback: Any = None,
    backend_context: Any | None = None,
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
            backend_context=backend_context,
        )
    return _run_single_cable_batch_group(
        group,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        batch_options=batch_options,
        solver_options=solver_options,
        observers=observers,
        progress_callback=progress_callback,
        backend_context=backend_context,
    )


def _dispatch_method(group: DispatchGroup) -> str:
    """Return the public diagnostic label for a dispatch group."""

    prefix = "batch" if group.geometry_shared else "parameter-batch"
    if group.mode == "double":
        return f"{prefix}-double-cable"
    return f"{prefix}-single-cable"


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
    if out.pending_observation is not None:
        return out.pending_observation.state
    if not out.observations:
        raise RuntimeError("batch kernel produced neither Vm nor observations.")
    first = next(iter(out.observations.values()))
    if hasattr(first, "words"):
        return first.words
    return first.values


def _finalize_pending_batch_observation(
    out: BatchKernelResult,
    *,
    group: DispatchGroup,
    mode: str,
) -> BatchKernelResult:
    """Finalize observer output after the explicit group-level device wait."""

    pending = out.pending_observation
    if pending is None:
        return out
    with benchmark_span(
        "kernel.finalize_observer",
        mode=mode,
        observer="vm_raster",
        group_id=group.group_id,
        group_size=group.size,
        synchronized_before_finalize=True,
        wait_span="kernel.wait",
    ):
        observations = cast(
            dict[str, object],
            finalize_vm_raster_state(
                pending.plan,
                pending.state,
                nt=pending.nt,
                dt_ms=pending.dt_ms,
                synchronize=False,
            ),
        )
    return BatchKernelResult(
        Vm=out.Vm,
        t=out.t,
        observations=observations,
        pending_observation=None,
    )


def _benchmark_double_cable_block_solver_override() -> str | None:
    session = active_benchmark_session()
    if session is None:
        return None
    options = session.metadata.get("benchmark_options")
    if not isinstance(options, dict):
        return None
    solver = options.get("benchmark_double_cable_block_solver")
    if solver in (None, ""):
        return None
    return str(solver)


def _benchmark_observer_state_scope_override() -> str | None:
    session = active_benchmark_session()
    if session is None:
        return None
    options = session.metadata.get("benchmark_options")
    if not isinstance(options, dict):
        return None
    scope = options.get("benchmark_observer_state_scope")
    if scope in (None, "", "default"):
        return None
    return str(scope)


def _context_double_cable_block_solver_override(
    backend_context: Any | None,
) -> tuple[str | None, bool, int | None]:
    if backend_context is None:
        return None, False, None
    solver = getattr(backend_context, "double_cable_block_solver", None)
    if solver in (None, ""):
        return None, False, None
    return (
        str(solver),
        bool(getattr(backend_context, "double_cable_block_solver_allow_internal", False)),
        getattr(backend_context, "double_cable_tiled_thomas_block_b", None),
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
    backend_context: Any | None = None,
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
        runtime = prepare_batch_runtime(
            group,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            solver_options=solver_options,
            mode="single",
            include_extracellular=False,
            include_area=False,
            backend_context=backend_context,
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
        cohort = prepared_cohort_for_current_group(group)
        record_benchmark_metadata(
            **benchmark_array_metadata(
                "x_positions_m",
                cohort.x_positions_m,
                role="positions",
            ),
            extracellular_stimulation_count=cohort.extracellular_stimulation_count,
        )
    lowered_options = lower_batch_recording_options(
        group,
        batch_options,
        observers=observers,
    )
    kernel_options = OutputPlan.from_batch_options(
        lowered_options,
        observers=observers,
        row_record_indices=row_recording_indices_for_group(
            group,
            lowered_options.recording,
        ),
    )
    _emit_progress(
        progress_callback,
        group,
        "batch",
        "recording plan",
        recording=kernel_options.recording.mode,
        time_chunk_steps=kernel_options.time_chunk_steps,
        output_sink=kernel_options.sink,
        observers=0 if observers is None else len(observers),
    )
    with benchmark_span(
        "observer.plan",
        group_id=group.group_id,
        group_size=group.size,
        recording_mode=kernel_options.recording.mode,
    ):
        observer_plan = lower_observers_for_cohort(
            observers,
            cohort=cohort,
            dtype=runtime.membrane.dtype,
            prefer_vm_raster=kernel_options.recording.mode == "none",
        )
    with benchmark_span(
        "inputs.intracellular",
        group_id=group.group_id,
        group_size=group.size,
        nt=runtime.grid.Nt,
        nx=group.nx,
    ):
        intracellular = lower_single_cable_intracellular_input(
            group=group,
            cohort=cohort,
            runtime=runtime,
            kernel_options=kernel_options,
            observers=observers,
        )
        record_intracellular_lowering_metadata(
            intracellular,
            group=group,
            runtime=runtime,
        )
    with benchmark_span(
        "inputs.extracellular",
        group_id=group.group_id,
        group_size=group.size,
        nt=runtime.grid.Nt,
        nx=group.nx,
    ):
        extracellular = lower_single_cable_extracellular_input(
            group=group,
            cohort=cohort,
            runtime=runtime,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            intracellular=intracellular,
            observer_plan=observer_plan,
        )
        record_extracellular_lowering_metadata(
            extracellular,
            group=group,
            runtime=runtime,
        )
    _emit_progress(
        progress_callback,
        group,
        "lowering",
        "inputs",
        intracellular=intracellular.format,
        extracellular=extracellular.format,
        stimulations=cohort.extracellular_stimulation_count,
    )
    record_group_memory_estimate(
        group=group,
        runtime=runtime,
        cohort=cohort,
        kernel_options=kernel_options,
        intracellular_format=intracellular.format,
        extracellular_format=extracellular.format,
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
        timing_role="host_enqueue",
        device_synchronization=False,
        explicit_wait_span="kernel.wait",
    ):
        out = SingleCableVStimBatchKernel(
            runtime=runtime,
            Cm_uF_cm2=group_cm_uF_cm2(group, runtime),
            has_driven_extracellular=cohort.extracellular_stimulation_count > 0,
        ).run(
            intracellular_current_density_mid=intracellular.midpoint,
            extracellular_potential_mid_mV=extracellular.midpoint,
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
        timing_role="device_synchronization",
        device_synchronization=True,
        includes_device_solver_work=True,
    ):
        benchmark_wait(_batch_wait_target(out))
    out = _finalize_pending_batch_observation(
        out,
        group=group,
        mode=group.mode,
    )
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
        return dispatch_results_from_batch(
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
    backend_context: Any | None = None,
) -> tuple[DispatchRecord, ...]:
    """Run a homogeneous double-cable group through full double-cable batching."""

    kernel_group = double_cable_kernel_group(group)
    representative = representative_item(group).simulation
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
        runtime = prepare_batch_runtime(
            kernel_group,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            solver_options=solver_options,
            mode="double",
            include_extracellular=True,
            include_area=True,
            backend_context=backend_context,
        )
        record_kernel_bucket_metadata(group=group, kernel_group=kernel_group)
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
        cohort = prepared_cohort_for_current_group(kernel_group)
        record_benchmark_metadata(
            **benchmark_array_metadata(
                "x_positions_m",
                cohort.x_positions_m,
                role="positions",
            ),
            extracellular_stimulation_count=cohort.extracellular_stimulation_count,
            public_group_size=int(group.size),
            kernel_group_size=int(kernel_group.size),
            public_nx=int(group.nx),
            kernel_nx=int(kernel_group.nx),
        )
    lowered_options = lower_batch_recording_options(
        kernel_group,
        batch_options,
        observers=observers,
    )
    kernel_options = OutputPlan.from_batch_options(
        lowered_options,
        observers=observers,
        row_record_indices=row_recording_indices_for_group(
            kernel_group,
            lowered_options.recording,
        ),
    )
    policy_block_solver, policy_allow_internal, policy_block_b = (
        _context_double_cable_block_solver_override(backend_context)
    )
    benchmark_block_solver = _benchmark_double_cable_block_solver_override()
    benchmark_observer_state_scope = _benchmark_observer_state_scope_override()
    _emit_progress(
        progress_callback,
        group,
        "batch",
        "recording plan",
        recording=kernel_options.recording.mode,
        time_chunk_steps=kernel_options.time_chunk_steps,
        output_sink=kernel_options.sink,
        observers=0 if observers is None else len(observers),
        policy_block_solver=policy_block_solver,
        benchmark_block_solver=benchmark_block_solver,
        benchmark_observer_state_scope=benchmark_observer_state_scope,
    )
    with benchmark_span(
        "observer.plan",
        group_id=group.group_id,
        group_size=group.size,
        recording_mode=kernel_options.recording.mode,
    ):
        observer_plan = lower_observers_for_cohort(
            observers,
            cohort=cohort,
            dtype=runtime.membrane.dtype,
            prefer_vm_raster=kernel_options.recording.mode == "none",
        )
    with benchmark_span(
        "inputs.intracellular",
        group_id=group.group_id,
        group_size=group.size,
        nt=runtime.grid.Nt,
        nx=group.nx,
    ):
        intracellular = lower_double_cable_intracellular_input(
            cohort=cohort,
            runtime=runtime,
        )
        record_intracellular_lowering_metadata(
            intracellular,
            group=kernel_group,
            runtime=runtime,
        )
    with benchmark_span(
        "inputs.extracellular",
        group_id=group.group_id,
        group_size=group.size,
        nt=runtime.grid.Nt,
        nx=group.nx,
    ):
        extracellular = lower_double_cable_extracellular_input(
            cohort=cohort,
            runtime=runtime,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            observer_plan=observer_plan,
            kernel_options=kernel_options,
        )
        record_extracellular_lowering_metadata(
            extracellular,
            group=kernel_group,
            runtime=runtime,
        )
    _emit_progress(
        progress_callback,
        group,
        "lowering",
        "inputs",
        intracellular=intracellular.format,
        extracellular=extracellular.format,
        stimulations=cohort.extracellular_stimulation_count,
    )
    record_group_memory_estimate(
        group=kernel_group,
        runtime=runtime,
        cohort=cohort,
        kernel_options=kernel_options,
        intracellular_format=intracellular.format,
        extracellular_format=extracellular.format,
        include_vstim_previous=True,
    )
    record_benchmark_metadata(
        public_group_size=int(group.size),
        public_nx=int(group.nx),
    )
    if benchmark_block_solver is not None:
        record_benchmark_metadata(
            benchmark_double_cable_block_solver=benchmark_block_solver
        )
    if benchmark_observer_state_scope is not None:
        record_benchmark_metadata(
            benchmark_observer_state_scope=benchmark_observer_state_scope
        )
    if policy_block_solver is not None:
        record_benchmark_metadata(
            execution_policy_double_cable_block_solver=policy_block_solver,
            execution_policy_double_cable_block_solver_internal=policy_allow_internal,
            execution_policy_tiled_thomas_block_b=policy_block_b,
        )
    _emit_progress(
        progress_callback,
        group,
        "kernel",
        "compiling JAX kernel if needed",
        recording=kernel_options.recording.mode,
        time_chunk_steps=kernel_options.time_chunk_steps,
        policy_block_solver=policy_block_solver,
        benchmark_block_solver=benchmark_block_solver,
        benchmark_observer_state_scope=benchmark_observer_state_scope,
    )
    with benchmark_span(
        "kernel.enqueue",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        recording_mode=kernel_options.recording.mode,
        benchmark_double_cable_block_solver=benchmark_block_solver,
        timing_role="host_enqueue",
        device_synchronization=False,
        explicit_wait_span="kernel.wait",
    ):
        out = DoubleCableBatchKernel(
            runtime=runtime,
            Veinit_mV=float(getattr(representative, "Veinit", 0.0)),
            has_driven_extracellular=cohort.extracellular_stimulation_count > 0,
        ).run(
            intracellular_current_density_mid=intracellular.midpoint,
            extracellular_potential_mid_mV=extracellular.midpoint,
            extracellular_potential_initial_previous_mV=extracellular.initial_previous,
            options=kernel_options,
            observers=observer_plan,
            progress_callback=progress_callback,
            double_cable_block_solver=policy_block_solver,
            allow_internal_double_cable_block_solver=policy_allow_internal,
            double_cable_tiled_thomas_block_b=policy_block_b,
            benchmark_double_cable_block_solver=benchmark_block_solver,
            benchmark_observer_state_scope=benchmark_observer_state_scope,
        )
        if out.Vm is not None:
            record_benchmark_metadata(
                **benchmark_array_metadata("Vm", out.Vm, role="kernel_output")
            )
    with benchmark_span(
        "kernel.trim_batch_output",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        recording_mode=kernel_options.recording.mode,
    ):
        out = trim_batch_kernel_result(out, batch_size=group.size)
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
        timing_role="device_synchronization",
        device_synchronization=True,
        includes_device_solver_work=True,
    ):
        benchmark_wait(_batch_wait_target(out))
    out = _finalize_pending_batch_observation(
        out,
        group=group,
        mode=group.mode,
    )
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
        return dispatch_results_from_batch(
            group,
            Vm=out.Vm,
            t=out.t,
            observations=out.observations,
            observer_definitions=observers,
            method=_dispatch_method(group),
            batch_options=batch_options,
            kernel_batch_options=kernel_options,
        )


__all__ = ["run_jax_batch_group"]
