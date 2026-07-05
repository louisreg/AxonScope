"""JAX execution for prepared dispatcher groups."""

from __future__ import annotations

from typing import Any, cast

from axonscope.benchmarking import (
    benchmark_array_metadata,
    benchmark_span,
    benchmark_wait,
    record_benchmark_metadata,
)
from axonscope.dispatcher.plan import DispatchGroup
from axonscope.dispatcher.progress import ProgressEvent, ProgressStage
from axonscope.dispatcher._records import DispatchRecord
from axonscope.backends.jax.benchmark_metadata import (
    record_extracellular_lowering_metadata,
    record_group_memory_estimate,
    record_intracellular_lowering_metadata,
)
from axonscope.backends.jax.batch_results import (
    dispatch_results_from_batch,
    trim_batch_kernel_result,
)
from axonscope.backends.jax.input_lowering import (
    lower_double_cable_extracellular_input,
    lower_double_cable_intracellular_input,
    lower_single_cable_extracellular_input,
    lower_single_cable_intracellular_input,
)
from axonscope.backends.jax.recording_lowering import (
    lower_batch_recording_options,
    lower_observers_for_cohort,
)
from axonscope.backends.jax.runtime_preparation import (
    group_cm_uF_cm2,
    prepare_batch_runtime,
    prepared_cohort_for_group,
    representative_item,
)
from axonscope.backends.jax.shape_bucketing import (
    double_cable_kernel_group,
    record_kernel_bucket_metadata,
)
from axonscope.backends.jax.batch_kernels import (
    DoubleCableBatchKernel,
    SingleCableVStimBatchKernel,
)
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
    if not out.observations:
        raise RuntimeError("batch kernel produced neither Vm nor observations.")
    first = next(iter(out.observations.values()))
    if hasattr(first, "words"):
        return first.words
    return first.values


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
        cohort = prepared_cohort_for_group(group)
        record_benchmark_metadata(
            **benchmark_array_metadata(
                "x_positions_m",
                cohort.x_positions_m,
                role="positions",
            ),
            extracellular_stimulation_count=cohort.extracellular_stimulation_count,
        )
    kernel_options = lower_batch_recording_options(
        group,
        batch_options,
        observers=observers,
    )
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
        cohort = prepared_cohort_for_group(kernel_group)
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
    kernel_options = lower_batch_recording_options(
        kernel_group,
        batch_options,
        observers=observers,
    )
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
            has_driven_extracellular=cohort.extracellular_stimulation_count > 0,
        ).run(
            intracellular_current_density_mid=intracellular.midpoint,
            extracellular_potential_mid_mV=extracellular.midpoint,
            extracellular_potential_initial_previous_mV=extracellular.initial_previous,
            options=kernel_options,
            observers=observer_plan,
            progress_callback=progress_callback,
        )
        if out.Vm is not None:
            record_benchmark_metadata(
                **benchmark_array_metadata("Vm", out.Vm, role="kernel_output")
            )
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
