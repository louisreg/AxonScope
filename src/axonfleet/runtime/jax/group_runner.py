"""JAX execution for prepared dispatcher groups."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

from axonfleet.benchmarking import benchmark_span, record_benchmark_metadata
from axonfleet.runtime.benchmarking import (
    active_benchmark_session,
    benchmark_array_metadata,
    benchmark_wait,
)
from axonfleet.dispatcher.plan import DispatchGroup
from axonfleet.dispatcher.progress import ProgressEvent, ProgressStage
from axonfleet.dispatcher._records import DispatchRecord
from axonfleet.runtime.jax.benchmarking.metadata import (
    record_extracellular_lowering_metadata,
    record_group_memory_estimate,
    record_intracellular_lowering_metadata,
)
from axonfleet.runtime.jax.recording.results import (
    batch_wait_target,
    finalize_pending_batch_observation,
    trim_batch_kernel_result,
)
from axonfleet.runtime.inputs.contracts import (
    PreparedRuntimeInputSummary,
    extracellular_mode_from_format,
)
from axonfleet.runtime.outputs.assembly import dispatch_results_from_batch
from axonfleet.runtime.jax.inputs.lowering import (
    JAX_DOUBLE_CABLE_INPUT_CONTRACT,
    JAX_SINGLE_CABLE_INPUT_CONTRACT,
    lower_double_cable_extracellular_input,
    lower_double_cable_intracellular_input,
    lower_numeric_axis_input,
    lower_single_cable_extracellular_input,
    lower_single_cable_intracellular_input,
)
from axonfleet.runtime.outputs.contracts import OutputPlan
from axonfleet.runtime.recording import (
    lower_batch_recording_options,
    row_recording_indices_for_group,
)
from axonfleet.runtime.jax.recording.lowering import (
    lower_observers_for_cohort,
)
from axonfleet.runtime.jax.preparation.runtime import prepare_batch_runtime
from axonfleet.runtime.jax.preparation.stacking import group_cm_uF_cm2
from axonfleet.runtime.jax.kernels.double_cable import DoubleCableBatchKernel
from axonfleet.runtime.jax.kernels.single_cable import SingleCableVStimBatchKernel
from axonfleet.runtime.jax.policy.engine_types import (
    CPU_SINGLE_CABLE_SOLVER,
    GPU_SINGLE_CABLE_SOLVER,
)
from axonfleet.solvers.options import BatchOptions
from axonfleet.recording import RecordingPlan


@dataclass(frozen=True)
class _PreparedJaxBatchGroup:
    """Shared host-side state before cable-specific input lowering."""

    runtime: Any
    cohort: Any


@dataclass(frozen=True)
class _LoweredJaxBatchInputs:
    """Concrete input payloads lowered for one JAX batch kernel."""

    intracellular: Any
    extracellular: Any


@dataclass(frozen=True)
class PendingJaxBatchGroup:
    """Device work enqueued for one dispatcher group, awaiting synchronization."""

    group: DispatchGroup
    output: Any
    batch_options: BatchOptions
    kernel_options: OutputPlan
    observers: tuple[Any, ...] | None
    progress_callback: Any


def enqueue_jax_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None = None,
    recording_plan: RecordingPlan | None = None,
    progress_callback: Any = None,
    runtime_context: Any | None = None,
    preparation_cache: Any | None = None,
) -> PendingJaxBatchGroup:
    """Prepare and enqueue one JAX batch group without waiting for device work."""

    if group.mode == "double":
        _require_preparation_cache(preparation_cache)
        return _enqueue_double_cable_batch_group(
            group,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            batch_options=batch_options,
            observers=observers,
            recording_plan=recording_plan,
            progress_callback=progress_callback,
            runtime_context=runtime_context,
            preparation_cache=preparation_cache,
        )
    if group.mode != "single":
        raise ValueError(
            f"Unsupported JAX dispatch group mode {group.mode!r}; "
            "expected 'single' or 'double'."
        )
    _require_preparation_cache(preparation_cache)
    return _enqueue_single_cable_batch_group(
        group,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        batch_options=batch_options,
        observers=observers,
        recording_plan=recording_plan,
        progress_callback=progress_callback,
        runtime_context=runtime_context,
        preparation_cache=preparation_cache,
    )


def _require_preparation_cache(preparation_cache: Any | None) -> None:
    if preparation_cache is None or not callable(
        getattr(preparation_cache, "for_current_group", None)
    ):
        raise RuntimeError(
            "JAX group execution requires the Runner-owned preparation cache."
        )


def finalize_jax_batch_group(pending: PendingJaxBatchGroup) -> tuple[DispatchRecord, ...]:
    """Synchronize and assemble one previously enqueued JAX batch group."""

    group = pending.group
    out = _wait_for_batch_kernel_output(
        pending.output,
        group=group,
        progress_callback=pending.progress_callback,
    )
    return _dispatch_batch_kernel_output(
        out,
        group=group,
        batch_options=pending.batch_options,
        kernel_options=pending.kernel_options,
        observers=pending.observers,
        progress_callback=pending.progress_callback,
    )


def _prepare_jax_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    progress_callback: Any,
    runtime_context: Any | None,
    preparation_cache: Any,
) -> _PreparedJaxBatchGroup:
    """Prepare runtime arrays and cohort rows through the shared host path."""

    _emit_progress(progress_callback, group, "prepare", "runtime", mode=group.mode)
    with benchmark_span(
        "runtime.prepare",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        nx=group.nx,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
    ):
        with benchmark_span(
            "runtime.prepare.materialize_axons",
            group_id=group.group_id,
            group_size=group.size,
            mode=group.mode,
            nx=group.nx,
        ):
            cohort = preparation_cache.for_current_group(group)
            record_benchmark_metadata(
                materialized_axon_rows=cohort.materialized_axons.size,
                materialized_axon_templates=cohort.materialized_axons.template_count,
                materialized_axon_translated_rows=(
                    cohort.materialized_axons.translated_row_count
                ),
                materialized_axon_nbytes=cohort.materialized_axons.nbytes,
                membrane_parameter_rows=cohort.membrane_rows.unique_count,
                membrane_parameter_cache_hits=cohort.membrane_rows.cache_hits,
                membrane_unique_models=cohort.membrane_rows.unique_model_count,
            )
        runtime = prepare_batch_runtime(
            group,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            mode=group.mode,
            include_extracellular=group.mode == "double",
            include_area=group.mode == "double",
            materialized_axons=cohort.materialized_axons,
            membrane_rows=cohort.membrane_rows,
            runtime_context=runtime_context,
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
        record_benchmark_metadata(
            **benchmark_array_metadata(
                "x_positions_m",
                cohort.x_positions_m,
                role="positions",
            ),
            extracellular_stimulation_count=cohort.extracellular_stimulation_count,
        )

    return _PreparedJaxBatchGroup(
        runtime=runtime,
        cohort=cohort,
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
        route=group.dispatch_method,
        message=message,
        details={key: value for key, value in details.items() if value is not None},
    )
    progress_callback(event)


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


def _runtime_context_solver_engine(runtime_context: Any | None) -> Any | None:
    if runtime_context is None:
        return None
    return getattr(runtime_context, "solver_engine", None)


def _runtime_context_platform(runtime_context: Any | None) -> str:
    platform = getattr(runtime_context, "platform", None)
    if platform is None:
        solver_engine = _runtime_context_solver_engine(runtime_context)
        platform = getattr(solver_engine, "platform", None)
    return str(platform).lower()


def _guard_gpu_observer_extracellular_route(
    *,
    runtime_context: Any | None,
    observer_plan: Any,
    kernel_options: OutputPlan,
    extracellular_stimulation_count: int,
    lowered_inputs: _LoweredJaxBatchInputs,
) -> None:
    """Reject dense Vext fallback on the compact GPU observer route."""

    if not _requires_factorized_gpu_observer_route(
        runtime_context=runtime_context,
        observer_plan=observer_plan,
        kernel_options=kernel_options,
        extracellular_stimulation_count=extracellular_stimulation_count,
    ):
        return
    if lowered_inputs.extracellular.format != "factorized_footprint":
        reason = lowered_inputs.extracellular.dense_fallback_reason or "unknown"
        raise RuntimeError(
            "JAX GPU observer-only execution resolved to dense extracellular "
            f"input ({reason}); expected the factorized footprint route. "
            "Split unsupported stimulation rows or use an explicitly recorded "
            "Vm workflow."
        )


def _guard_single_cable_gpu_solver_route(*, runtime_context: Any | None) -> str:
    """Require the retained Triton solve for JAX CUDA single-cable execution."""

    solver_engine = _runtime_context_solver_engine(runtime_context)
    if _runtime_context_platform(runtime_context) not in {"cuda", "gpu"}:
        return CPU_SINGLE_CABLE_SOLVER

    route = None if solver_engine is None else solver_engine.single_cable_solver
    if route != GPU_SINGLE_CABLE_SOLVER:
        raise RuntimeError(
            "JAX GPU single-cable execution resolved to the wrong solver route "
            f"{route!r}; expected {GPU_SINGLE_CABLE_SOLVER!r}."
        )

    from axonfleet.runtime.jax.kernels.triton_single_cable import (
        single_cable_triton_dependency_skip_reason,
    )

    skip_reason = single_cable_triton_dependency_skip_reason()
    if skip_reason is not None:
        raise RuntimeError(skip_reason)
    return GPU_SINGLE_CABLE_SOLVER


def _requires_factorized_gpu_observer_route(
    *,
    runtime_context: Any | None,
    observer_plan: Any,
    kernel_options: OutputPlan,
    extracellular_stimulation_count: int,
) -> bool:
    """Return whether dense Vext is forbidden before input materialization."""

    platform = getattr(runtime_context, "platform", None)
    if platform is None:
        solver_engine = _runtime_context_solver_engine(runtime_context)
        platform = getattr(solver_engine, "platform", None)
    return bool(
        _runtime_context_platform(runtime_context) in {"cuda", "gpu", "metal", "rocm"}
        and observer_plan is not None
        and kernel_options.recording.mode == "none"
        and extracellular_stimulation_count > 0
    )


def _lower_output_plan_for_group(
    *,
    group: DispatchGroup,
    runtime: Any,
    cohort: Any,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None,
    progress_callback: Any,
    progress_details: dict[str, Any] | None = None,
) -> tuple[OutputPlan, Any]:
    """Lower recording and observers through the shared non-solver path."""

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
    details = {
        "recording": kernel_options.recording.mode,
        "time_chunk_steps": kernel_options.time_chunk_steps,
        "output_sink": kernel_options.sink,
        "observers": 0 if observers is None else len(observers),
    }
    if progress_details is not None:
        details.update(progress_details)
    _emit_progress(
        progress_callback,
        group,
        "batch",
        "recording plan",
        **details,
    )
    with benchmark_span(
        "observer.plan",
        group_id=group.group_id,
        group_size=group.size,
        recording_mode=kernel_options.recording.mode,
    ):
        observer_plan = (
            lower_observers_for_cohort(
                observers,
                cohort=cohort,
                dtype=runtime.membrane.dtype,
            )
            if kernel_options.recording.mode == "none"
            else None
        )
    return kernel_options, observer_plan


def _wait_for_batch_kernel_output(
    out: Any,
    *,
    group: DispatchGroup,
    progress_callback: Any,
) -> Any:
    """Synchronize one batch kernel result and finalize pending observations."""

    _emit_progress(
        progress_callback,
        group,
        "kernel",
        "waiting for JAX work",
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
        benchmark_wait(batch_wait_target(out))
    out = finalize_pending_batch_observation(
        out,
        group=group,
        mode=group.mode,
    )
    _emit_progress(progress_callback, group, "kernel", "completed JAX work")
    return out


def _dispatch_batch_kernel_output(
    out: Any,
    *,
    group: DispatchGroup,
    batch_options: BatchOptions,
    kernel_options: OutputPlan,
    observers: tuple[Any, ...] | None,
    progress_callback: Any,
) -> tuple[DispatchRecord, ...]:
    """Assemble one batch kernel output into dispatcher records."""

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
            recordings=out.recordings,
            observations=out.observations,
            observer_definitions=observers,
            method=group.dispatch_method,
            batch_options=batch_options,
            kernel_batch_options=kernel_options,
        )


def _record_lowered_input_progress_and_memory(
    *,
    group: DispatchGroup,
    runtime: Any,
    cohort: Any,
    kernel_options: OutputPlan,
    lowered_inputs: _LoweredJaxBatchInputs,
    progress_callback: Any,
) -> None:
    """Record shared input-lowering progress and memory metadata."""

    _emit_progress(
        progress_callback,
        group,
        "lowering",
        "inputs",
        intracellular=lowered_inputs.intracellular.format,
        extracellular=lowered_inputs.extracellular.format,
        stimulations=cohort.extracellular_stimulation_count,
    )
    record_group_memory_estimate(
        group=group,
        runtime=runtime,
        cohort=cohort,
        kernel_options=kernel_options,
        intracellular_format=lowered_inputs.intracellular.format,
        extracellular_format=lowered_inputs.extracellular.format,
        include_vstim_previous=group.mode == "double",
    )


def _record_prepared_runtime_input_contract(
    *,
    group: DispatchGroup,
    runtime: Any,
    kernel_options: OutputPlan,
    lowered_inputs: _LoweredJaxBatchInputs,
    observers: tuple[Any, ...] | None,
    solver_policy: str,
) -> None:
    """Validate and record the runtime-neutral contract for one prepared batch."""

    contract = (
        JAX_DOUBLE_CABLE_INPUT_CONTRACT
        if group.mode == "double"
        else JAX_SINGLE_CABLE_INPUT_CONTRACT
    )
    extracellular_mode = extracellular_mode_from_format(
        lowered_inputs.extracellular.format,
        explicit_mode=lowered_inputs.extracellular.mode,
    )
    summary = PreparedRuntimeInputSummary(
        cable="double-cable" if group.mode == "double" else "single-cable",
        batch_size=int(group.size),
        nx=int(runtime.membrane.Nx),
        nt=int(runtime.grid.Nt),
        dtype=str(runtime.membrane.dtype),
        has_padding=bool(group.has_padding),
        row_specific_parameters=not bool(group.geometry_shared),
        recording_mode=kernel_options.recording.mode,
        output_sink=kernel_options.sink,
        observer_count=0 if observers is None else len(observers),
        time_chunk_steps=kernel_options.time_chunk_steps,
        solver_policy=solver_policy,
        intracellular_format=lowered_inputs.intracellular.format,
        extracellular_format=lowered_inputs.extracellular.format,
        extracellular_mode=extracellular_mode,
        extracellular_requires_initial_previous=(
            contract.extracellular.requires_initial_previous
        ),
        extracellular_has_initial_previous=_extracellular_has_initial_previous(
            lowered_inputs.extracellular
        ),
    )
    summary.validate_against(contract)
    record_benchmark_metadata(
        **contract.as_metadata(prefix="runtime_input_contract_"),
        **summary.as_metadata(prefix="prepared_input_contract_"),
    )


def _extracellular_has_initial_previous(extracellular: Any) -> bool:
    if extracellular.initial_previous is not None:
        return True
    factorized = extracellular.factorized
    return bool(
        factorized is not None
        and getattr(factorized, "current_initial_previous_A", None) is not None
    )


def _group_numeric_axis_shape(group: DispatchGroup) -> tuple[int, int] | None:
    axis_input = group.numeric_axis
    if axis_input is None:
        return None
    source_size = group.numeric_axis_source_size
    if source_size is None:
        raise RuntimeError("numeric-axis dispatch group is missing its source size.")
    return int(source_size), int(axis_input.size)


def _lower_group_numeric_axis(
    lowered_inputs: _LoweredJaxBatchInputs,
    *,
    group: DispatchGroup,
    runtime: Any,
    tsim_ms: float,
    dt_ms: float,
) -> _LoweredJaxBatchInputs:
    axis_input = group.numeric_axis
    if axis_input is None:
        return lowered_inputs
    source_size, _axis_size = _group_numeric_axis_shape(group)
    with benchmark_span(
        "inputs.numeric_axis",
        source_size=source_size,
        axis_size=axis_input.size,
        logical_batch_size=axis_input.size * source_size,
        kernel_batch_size=group.size,
        mode=group.mode,
    ):
        extracellular = lower_numeric_axis_input(
            lowered_inputs.extracellular,
            axis_input,
            source_size=source_size,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            dtype_local=runtime.membrane.dtype,
            include_initial_previous=group.mode == "double",
        )
    return replace(
        lowered_inputs,
        extracellular=extracellular,
    )


def _emit_kernel_compile_progress(
    *,
    group: DispatchGroup,
    kernel_options: OutputPlan,
    progress_callback: Any,
    **details: Any,
) -> None:
    """Emit the common pre-enqueue kernel progress event."""

    _emit_progress(
        progress_callback,
        group,
        "kernel",
        "compiling JAX kernel if needed",
        recording=kernel_options.recording.mode,
        time_chunk_steps=kernel_options.time_chunk_steps,
        **details,
    )


def _record_kernel_output_metadata(out: Any) -> None:
    """Record benchmark metadata for retained Vm kernel outputs."""

    if out.Vm is not None:
        record_benchmark_metadata(
            **benchmark_array_metadata("Vm", out.Vm, role="kernel_output")
        )


def _lower_single_cable_inputs(
    *,
    group: DispatchGroup,
    runtime: Any,
    cohort: Any,
    kernel_options: OutputPlan,
    observer_plan: Any,
    observers: tuple[Any, ...] | None,
    tsim_ms: float,
    dt_ms: float,
    require_factorized_extracellular: bool,
) -> _LoweredJaxBatchInputs:
    """Lower and record single-cable kernel inputs."""

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
            cohort=cohort,
            runtime=runtime,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            intracellular=intracellular,
            observer_plan=observer_plan,
            require_factorized=require_factorized_extracellular,
            numeric_axis_shape=_group_numeric_axis_shape(group),
        )
        record_extracellular_lowering_metadata(
            extracellular,
            group=group,
            runtime=runtime,
        )
    return _LoweredJaxBatchInputs(
        intracellular=intracellular,
        extracellular=extracellular,
    )


def _lower_double_cable_inputs(
    *,
    group: DispatchGroup,
    runtime: Any,
    cohort: Any,
    tsim_ms: float,
    dt_ms: float,
    require_factorized_extracellular: bool,
) -> _LoweredJaxBatchInputs:
    """Lower and record double-cable kernel inputs."""

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
        extracellular = lower_double_cable_extracellular_input(
            cohort=cohort,
            runtime=runtime,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            require_factorized=require_factorized_extracellular,
            numeric_axis_shape=_group_numeric_axis_shape(group),
        )
        record_extracellular_lowering_metadata(
            extracellular,
            group=group,
            runtime=runtime,
        )
    return _LoweredJaxBatchInputs(
        intracellular=intracellular,
        extracellular=extracellular,
    )


def _enqueue_single_cable_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None,
    recording_plan: RecordingPlan | None,
    progress_callback: Any = None,
    runtime_context: Any | None = None,
    preparation_cache: Any,
) -> PendingJaxBatchGroup:
    """Enqueue a homogeneous single-cable group through imposed-field batching."""

    policy_single_cable_solver = _guard_single_cable_gpu_solver_route(
        runtime_context=runtime_context
    )

    prepared = _prepare_jax_batch_group(
        group,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        progress_callback=progress_callback,
        runtime_context=runtime_context,
        preparation_cache=preparation_cache,
    )
    runtime = prepared.runtime
    cohort = prepared.cohort
    kernel_options, observer_plan = _lower_output_plan_for_group(
        group=group,
        runtime=runtime,
        cohort=cohort,
        batch_options=batch_options,
        observers=observers,
        progress_callback=progress_callback,
    )
    lowered_inputs = _lower_single_cable_inputs(
        group=group,
        runtime=runtime,
        cohort=cohort,
        kernel_options=kernel_options,
        observer_plan=observer_plan,
        observers=observers,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        require_factorized_extracellular=_requires_factorized_gpu_observer_route(
            runtime_context=runtime_context,
            observer_plan=observer_plan,
            kernel_options=kernel_options,
            extracellular_stimulation_count=cohort.extracellular_stimulation_count,
        ),
    )
    lowered_inputs = _lower_group_numeric_axis(
        lowered_inputs,
        group=group,
        runtime=runtime,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
    )
    _guard_gpu_observer_extracellular_route(
        runtime_context=runtime_context,
        observer_plan=observer_plan,
        kernel_options=kernel_options,
        extracellular_stimulation_count=cohort.extracellular_stimulation_count,
        lowered_inputs=lowered_inputs,
    )
    _record_lowered_input_progress_and_memory(
        group=group,
        runtime=runtime,
        cohort=cohort,
        kernel_options=kernel_options,
        lowered_inputs=lowered_inputs,
        progress_callback=progress_callback,
    )
    _record_prepared_runtime_input_contract(
        group=group,
        runtime=runtime,
        kernel_options=kernel_options,
        lowered_inputs=lowered_inputs,
        observers=observers,
        solver_policy=policy_single_cable_solver,
    )
    record_benchmark_metadata(
        execution_policy_single_cable_solver=policy_single_cable_solver
    )
    _emit_kernel_compile_progress(
        group=group,
        kernel_options=kernel_options,
        progress_callback=progress_callback,
    )
    with benchmark_span(
        "kernel.enqueue",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        recording_mode=kernel_options.recording.mode,
        timing_role="enqueue_may_execute_deferred_work",
        device_synchronization=False,
        explicit_wait_span="kernel.wait",
    ):
        out = SingleCableVStimBatchKernel(
            runtime=runtime,
            Cm_uF_cm2=group_cm_uF_cm2(
                group,
                runtime,
                cohort.materialized_axons,
            ),
            has_driven_extracellular=cohort.extracellular_stimulation_count > 0,
        ).run(
            intracellular_current_density_mid=lowered_inputs.intracellular.midpoint,
            extracellular_potential_mid_mV=lowered_inputs.extracellular.midpoint,
            options=kernel_options,
            observers=observer_plan,
            recording_plan=recording_plan,
            progress_callback=progress_callback,
        )
        _record_kernel_output_metadata(out)
    return PendingJaxBatchGroup(
        group=group,
        output=out,
        batch_options=batch_options,
        kernel_options=kernel_options,
        observers=observers,
        progress_callback=progress_callback,
    )


def _enqueue_double_cable_batch_group(
    group: DispatchGroup,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None,
    recording_plan: RecordingPlan | None,
    progress_callback: Any = None,
    runtime_context: Any | None = None,
    preparation_cache: Any,
) -> PendingJaxBatchGroup:
    """Enqueue a homogeneous double-cable group through full double-cable batching."""
    prepared = _prepare_jax_batch_group(
        group,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        progress_callback=progress_callback,
        runtime_context=runtime_context,
        preparation_cache=preparation_cache,
    )
    runtime = prepared.runtime
    cohort = prepared.cohort
    solver_engine = _runtime_context_solver_engine(runtime_context)
    policy_block_solver = (
        None if solver_engine is None else solver_engine.double_cable_block_solver
    )
    policy_block_b = None if solver_engine is None else solver_engine.tiled_thomas_block_b
    benchmark_observer_state_scope = _benchmark_observer_state_scope_override()
    kernel_options, observer_plan = _lower_output_plan_for_group(
        group=group,
        runtime=runtime,
        cohort=cohort,
        batch_options=batch_options,
        observers=observers,
        progress_callback=progress_callback,
        progress_details={
            "policy_block_solver": policy_block_solver,
            "benchmark_observer_state_scope": benchmark_observer_state_scope,
        },
    )
    require_factorized_extracellular = _requires_factorized_gpu_observer_route(
        runtime_context=runtime_context,
        observer_plan=observer_plan,
        kernel_options=kernel_options,
        extracellular_stimulation_count=cohort.extracellular_stimulation_count,
    )
    lowered_inputs = _lower_double_cable_inputs(
        group=group,
        runtime=runtime,
        cohort=cohort,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        require_factorized_extracellular=require_factorized_extracellular,
    )
    lowered_inputs = _lower_group_numeric_axis(
        lowered_inputs,
        group=group,
        runtime=runtime,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
    )
    _guard_gpu_observer_extracellular_route(
        runtime_context=runtime_context,
        observer_plan=observer_plan,
        kernel_options=kernel_options,
        extracellular_stimulation_count=cohort.extracellular_stimulation_count,
        lowered_inputs=lowered_inputs,
    )
    _record_lowered_input_progress_and_memory(
        group=group,
        runtime=runtime,
        cohort=cohort,
        kernel_options=kernel_options,
        lowered_inputs=lowered_inputs,
        progress_callback=progress_callback,
    )
    _record_prepared_runtime_input_contract(
        group=group,
        runtime=runtime,
        kernel_options=kernel_options,
        lowered_inputs=lowered_inputs,
        observers=observers,
        solver_policy=str(policy_block_solver or "default"),
    )
    if benchmark_observer_state_scope is not None:
        record_benchmark_metadata(
            benchmark_observer_state_scope=benchmark_observer_state_scope
        )
    if policy_block_solver is not None:
        record_benchmark_metadata(
            execution_policy_double_cable_block_solver=policy_block_solver,
            execution_policy_double_cable_block_solver_internal=(
                solver_engine is not None and solver_engine.platform == "gpu"
            ),
            execution_policy_tiled_thomas_block_b=policy_block_b,
        )
    _emit_kernel_compile_progress(
        group=group,
        kernel_options=kernel_options,
        progress_callback=progress_callback,
        policy_block_solver=policy_block_solver,
        benchmark_observer_state_scope=benchmark_observer_state_scope,
    )
    with benchmark_span(
        "kernel.enqueue",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        recording_mode=kernel_options.recording.mode,
        timing_role="enqueue_may_execute_deferred_work",
        device_synchronization=False,
        explicit_wait_span="kernel.wait",
    ):
        out = DoubleCableBatchKernel(
            runtime=runtime,
            has_driven_extracellular=cohort.extracellular_stimulation_count > 0,
            solver_engine=solver_engine,
        ).run(
            intracellular_current_density_mid=lowered_inputs.intracellular.midpoint,
            extracellular_potential_mid_mV=lowered_inputs.extracellular.midpoint,
            extracellular_potential_initial_previous_mV=(
                lowered_inputs.extracellular.initial_previous
            ),
            options=kernel_options,
            observers=observer_plan,
            recording_plan=recording_plan,
            progress_callback=progress_callback,
            benchmark_observer_state_scope=benchmark_observer_state_scope,
            require_compact_factorized_extracellular=require_factorized_extracellular,
        )
        _record_kernel_output_metadata(out)
    with benchmark_span(
        "kernel.trim_batch_output",
        group_id=group.group_id,
        group_size=group.size,
        mode=group.mode,
        recording_mode=kernel_options.recording.mode,
    ):
        out = trim_batch_kernel_result(out, batch_size=group.size)
    return PendingJaxBatchGroup(
        group=group,
        output=out,
        batch_options=batch_options,
        kernel_options=kernel_options,
        observers=observers,
        progress_callback=progress_callback,
    )
