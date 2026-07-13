"""Single-cable JAX batch kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, cast

import jax.numpy as jnp

from axonscope.benchmarking import benchmark_span
from axonscope.runtime.input_payloads import (
    FactorizedExtracellularPotentialBatch,
    SparseIntracellularCurrentDensityBatch,
)
from axonscope.runtime.jax.inputs.payloads import (
    materialize_factorized_extracellular_potential_batch,
    materialize_sparse_intracellular_current_density_batch,
)
from axonscope.runtime.jax.cable_geometry import Array
from axonscope.runtime.jax.recording.observer import (
    PendingVmRasterObservation,
    VmRasterPlan,
    VmRasterState,
    init_vm_raster_state,
)
from axonscope.runtime.jax.types import SolverRuntime
from axonscope.solvers.options import BatchOptions

from .chunking import (
    _combine_vm_raster_chunk_states,
    _concat_trace_chunks,
    _init_local_vm_raster_chunk_template,
    _normalize_time_chunk_steps,
    _resolve_vm_raster_observer_state_scope,
    _time_chunks,
    _vm_raster_probe_tables_for_kernel,
)
from .factorized import (
    _factorized_current_mid_rows,
    _single_cable_factorized_forcing_footprint_for_batch,
)
from .inputs import (
    _as_cached_batched_scalar_or_space_array,
    _as_cached_batched_space_array,
    _as_batched_scalar_or_space_array,
    _as_batched_space_array,
    _as_batched_time_space_array,
    _as_factorized_extracellular_potential_batch,
    _as_sparse_intracellular_current_density_batch,
    _cached_broadcast_batch_leading,
    _cached_single_cable_tridiagonal_coefficients,
    _normalize_batch_options,
    _resolve_output_recording,
)
from axonscope.runtime.jax.recording.results import BatchKernelResult
from axonscope.recording import RecordingPlan

from .single_cable_scans import (
    _run_single_cable_factorized_vstim_batch_observer_scan,
    _run_single_cable_factorized_vstim_batch_sparse_observer_scan,
    _run_single_cable_factorized_vstim_batch_stateful_scan,
    _run_single_cable_shared_rank1_vstim_batch_sparse_observer_scan,
    _run_single_cable_vstim_batch_observer_scan,
    _run_single_cable_vstim_batch_stateful_scan,
    _run_single_cable_zero_vstim_batch_sparse_observer_scan,
)


class _RecordedTrace(NamedTuple):
    Vm: Array
    recordings: dict[str, Any] | None


@dataclass(frozen=True)
class SingleCableVStimBatchKernel:
    """Batch-oriented imposed-field kernel for homogeneous single-cable axons.

    The batch axis represents independent extracellular fields sharing the same
    axon geometry, membrane model, initial state, and time grid. This is the
    first GPU-friendly shape: ``Vstim[B, Nt, Nx] -> Vm[B, Nt, Nx]``.
    """

    runtime: SolverRuntime
    Cm_uF_cm2: Array
    has_driven_extracellular: bool | None = None

    def run(
        self,
        *,
        extracellular_potential_mid_mV: (
            Array | FactorizedExtracellularPotentialBatch | None
        ) = None,
        intracellular_current_density_mid: (
            Array | SparseIntracellularCurrentDensityBatch | None
        ) = None,
        options: BatchOptions | None = None,
        observers: VmRasterPlan | None = None,
        recording_plan: RecordingPlan | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BatchKernelResult:
        runtime = self.runtime
        if runtime.extracellular is not None:
            raise ValueError(
                "SingleCableVStimBatchKernel expects a scalar single-cable runtime; "
                "prepare it with include_extracellular=False."
            )

        membrane_runtime = runtime.membrane
        grid = runtime.grid
        cable = runtime.cable
        dtype_local = membrane_runtime.dtype
        with benchmark_span(
            "kernel.prepare_inputs",
            mode="single",
            nx=membrane_runtime.Nx,
            nt=grid.Nt,
        ):
            has_driven_extracellular = (
                runtime.stimulation.has_driven_extracellular
                if self.has_driven_extracellular is None
                else bool(self.has_driven_extracellular)
            )

            vext_mid = (
                runtime.stimulation.extracellular_potential_mid_mV
                if extracellular_potential_mid_mV is None
                else extracellular_potential_mid_mV
            )
            iinj_mid = (
                runtime.stimulation.intracellular_current_density_mid
                if intracellular_current_density_mid is None
                else intracellular_current_density_mid
            )

            factorized_vext = None
            if isinstance(vext_mid, FactorizedExtracellularPotentialBatch):
                factorized_vext = _as_factorized_extracellular_potential_batch(
                    "extracellular_potential_mid_mV",
                    vext_mid,
                    nt=grid.Nt,
                    nx=membrane_runtime.Nx,
                    dtype_local=dtype_local,
                )
                vext_batch = None
                batch_size = factorized_vext.batch_size
            elif vext_mid is None:
                if has_driven_extracellular:
                    raise ValueError("extracellular_potential_mid_mV is required for Vstim batching.")
                if not isinstance(iinj_mid, SparseIntracellularCurrentDensityBatch):
                    raise ValueError(
                        "extracellular_potential_mid_mV is required unless sparse "
                        "observer-only current input defines the batch size."
                    )
                batch_size = iinj_mid.batch_size
                vext_batch = None
            else:
                vext_batch = _as_batched_time_space_array(
                    "extracellular_potential_mid_mV",
                    vext_mid,
                    nt=grid.Nt,
                    nx=membrane_runtime.Nx,
                    dtype_local=dtype_local,
                )
                batch_size = int(vext_batch.shape[0])

            sparse_iinj = None
            if isinstance(iinj_mid, SparseIntracellularCurrentDensityBatch):
                sparse_iinj = _as_sparse_intracellular_current_density_batch(
                    "intracellular_current_density_mid",
                    iinj_mid,
                    nt=grid.Nt,
                    nx=membrane_runtime.Nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )
            iinj_batch = None
            if sparse_iinj is None and iinj_mid is None:
                iinj_batch = jnp.zeros(
                    (batch_size, grid.Nt, membrane_runtime.Nx),
                    dtype=dtype_local,
                )
            elif sparse_iinj is None:
                iinj_batch = _as_batched_time_space_array(
                    "intracellular_current_density_mid",
                    cast(Any, iinj_mid),
                    nt=grid.Nt,
                    nx=membrane_runtime.Nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )

            options = _normalize_batch_options(options)
            record_idx, record_full = _resolve_output_recording(
                options,
                nx=membrane_runtime.Nx,
            )
            record_voltage = (
                options.recording.mode != "none"
                and (recording_plan is None or recording_plan.voltage)
            )
            record_outputs = _recording_output_flags(recording_plan)
            chunk_steps = _normalize_time_chunk_steps(options.time_chunk_steps, nt=grid.Nt)
            stateless_vm_only = bool(
                membrane_runtime.membrane.supports_stateless_vm_only_fast_path()
            )
        if observers is not None and not record_voltage:
            if sparse_iinj is not None:
                if vext_batch is None:
                    if factorized_vext is not None:
                        observer_state = _run_single_cable_factorized_vstim_batch_sparse_observer_chunks(
                            runtime=runtime,
                            Cm_uF_cm2=jnp.asarray(self.Cm_uF_cm2, dtype=dtype_local),
                            observers=observers,
                            has_driven_extracellular=has_driven_extracellular,
                            stateless_vm_only=stateless_vm_only,
                            intracellular_current_density_mid=sparse_iinj,
                            extracellular_potential_mid_mV=factorized_vext,
                            time_chunk_steps=chunk_steps,
                            progress_callback=progress_callback,
                        )
                    else:
                        observer_state = _run_single_cable_zero_vstim_batch_sparse_observer_chunks(
                            runtime=runtime,
                            Cm_uF_cm2=jnp.asarray(self.Cm_uF_cm2, dtype=dtype_local),
                            observers=observers,
                            stateless_vm_only=stateless_vm_only,
                            intracellular_current_density_mid=sparse_iinj,
                            batch_size=batch_size,
                            time_chunk_steps=chunk_steps,
                            progress_callback=progress_callback,
                        )
                else:
                    raise ValueError(
                        "single-cable observer-only sparse current input requires "
                        "factorized or zero extracellular input; dense Vstim with "
                        "sparse observer input is not a supported kernel route."
                    )
            else:
                assert iinj_batch is not None
                if vext_batch is None and factorized_vext is not None:
                    observer_state = _run_single_cable_factorized_vstim_batch_observer_chunks(
                        runtime=runtime,
                        Cm_uF_cm2=jnp.asarray(self.Cm_uF_cm2, dtype=dtype_local),
                        observers=observers,
                        has_driven_extracellular=has_driven_extracellular,
                        stateless_vm_only=stateless_vm_only,
                        intracellular_current_density_mid=iinj_batch,
                        extracellular_potential_mid_mV=factorized_vext,
                        time_chunk_steps=chunk_steps,
                        progress_callback=progress_callback,
                    )
                else:
                    assert vext_batch is not None
                    observer_state = _run_single_cable_vstim_batch_observer_chunks(
                        runtime=runtime,
                        Cm_uF_cm2=jnp.asarray(self.Cm_uF_cm2, dtype=dtype_local),
                        observers=observers,
                        has_driven_extracellular=has_driven_extracellular,
                        stateless_vm_only=stateless_vm_only,
                        intracellular_current_density_mid=iinj_batch,
                        extracellular_potential_mid_mV=vext_batch,
                        time_chunk_steps=chunk_steps,
                        progress_callback=progress_callback,
                    )
            return BatchKernelResult(
                Vm=None,
                t=grid.t_vec_ms,
                pending_observation=PendingVmRasterObservation(
                    plan=observers,
                    state=observer_state,
                    nt=grid.Nt,
                    dt_ms=grid.dt_ms,
                ),
            )
        if factorized_vext is not None and record_voltage:
            if iinj_batch is None:
                assert sparse_iinj is not None
                with benchmark_span(
                    "kernel.materialize_inputs",
                    mode="single",
                    input="sparse_iinj",
                    group_size=batch_size,
                ):
                    iinj_batch = materialize_sparse_intracellular_current_density_batch(sparse_iinj)
            out = _run_single_cable_factorized_vstim_batch_array_chunks(
                runtime=runtime,
                Cm_uF_cm2=jnp.asarray(self.Cm_uF_cm2, dtype=dtype_local),
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
                intracellular_current_density_mid=iinj_batch,
                extracellular_potential_mid_mV=factorized_vext,
                record_indices=record_idx,
                record_full=record_full,
                time_chunk_steps=chunk_steps,
                record_outputs=record_outputs,
                progress_callback=progress_callback,
            )
            return BatchKernelResult(
                Vm=out.Vm if record_voltage else None,
                t=grid.t_vec_ms,
                recordings=_recordings_for_plan(
                    recording_plan,
                    out,
                    observable_names=membrane_runtime.observable_names,
                ),
            )
        if factorized_vext is not None and vext_batch is None:
            with benchmark_span(
                "kernel.materialize_inputs",
                mode="single",
                input="factorized_vext",
                group_size=batch_size,
            ):
                vext_batch = materialize_factorized_extracellular_potential_batch(
                    factorized_vext
                )
        if iinj_batch is None:
            assert sparse_iinj is not None
            with benchmark_span(
                "kernel.materialize_inputs",
                mode="single",
                input="sparse_iinj",
                group_size=batch_size,
            ):
                iinj_batch = materialize_sparse_intracellular_current_density_batch(sparse_iinj)
        if vext_batch is None:
            raise ValueError("extracellular_potential_mid_mV is required when recording Vm.")
        out = _run_single_cable_vstim_batch_array_chunks(
            runtime=runtime,
            Cm_uF_cm2=jnp.asarray(self.Cm_uF_cm2, dtype=dtype_local),
            has_driven_extracellular=has_driven_extracellular,
            stateless_vm_only=stateless_vm_only,
            intracellular_current_density_mid=iinj_batch,
            extracellular_potential_mid_mV=vext_batch,
            record_indices=record_idx,
            record_full=record_full,
            time_chunk_steps=chunk_steps,
            record_outputs=record_outputs,
            progress_callback=progress_callback,
        )
        return BatchKernelResult(
            Vm=out.Vm if record_voltage else None,
            t=grid.t_vec_ms,
            recordings=_recordings_for_plan(
                recording_plan,
                out,
                observable_names=membrane_runtime.observable_names,
            ),
        )

def _run_single_cable_vstim_batch_array_chunks(
    *,
    runtime: SolverRuntime,
    Cm_uF_cm2: Array,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    record_indices: Array,
    record_full: bool,
    time_chunk_steps: int | None,
    record_outputs: dict[str, bool],
    progress_callback: Callable[[int, int], None] | None,
) -> _RecordedTrace:
    membrane_runtime = runtime.membrane
    grid = runtime.grid
    cable = runtime.cable
    dtype_local = membrane_runtime.dtype
    dt = jnp.asarray(grid.dt_ms, dtype=dtype_local)
    batch_size = int(extracellular_potential_mid_mV.shape[0])
    with benchmark_span(
        "kernel.prepare_arrays",
        mode="single",
        variant="dense_vstim",
        output="full_vm" if record_full else "probe_vm",
        group_size=batch_size,
        nx=membrane_runtime.Nx,
        nt=grid.Nt,
        time_chunk_steps=time_chunk_steps,
    ):
        lower = _as_cached_batched_space_array(
            "lower", cable.lower, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        diag = _as_cached_batched_space_array(
            "diag", cable.diag, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        upper = _as_cached_batched_space_array(
            "upper", cable.upper, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        cm = _as_cached_batched_scalar_or_space_array(
            "Cm_uF_cm2",
            Cm_uF_cm2,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        background = _as_cached_batched_space_array(
            "I_background",
            membrane_runtime.background_current,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
    Vm, gates, state = _initial_single_cable_batch_state(runtime, batch_size)
    chunks = []
    recording_chunks: list[dict[str, Any]] = []

    chunk_ranges = tuple(_time_chunks(grid.Nt, time_chunk_steps))
    for chunk_index, (start, stop) in enumerate(chunk_ranges, start=1):
        with benchmark_span(
            "kernel.chunk_setup",
            mode="single",
            variant="dense_vstim",
            output="full_vm" if record_full else "probe_vm",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
        ):
            iinj_chunk = intracellular_current_density_mid[:, start:stop]
            vext_chunk = extracellular_potential_mid_mV[:, start:stop]
        with benchmark_span(
            "kernel.dispatch_jax",
            mode="single",
            variant="dense_vstim",
            output="full_vm" if record_full else "probe_vm",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
        ):
            Vm, gates, state, trace, recording_trace = _run_single_cable_vstim_batch_stateful_scan(
                backend=membrane_runtime.backend,
                membrane=membrane_runtime.membrane,
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
                record_full=record_full,
                record_gates=record_outputs["gates"],
                record_currents=record_outputs["currents"],
                record_conductances=record_outputs["conductances"],
                record_states=record_outputs["states"],
                lower=lower,
                diag=diag,
                upper=upper,
                dl=-dt * lower,
                d_static=jnp.ones_like(diag) - dt * diag,
                du=-dt * upper,
                Cm_uF_cm2=cm,
                I_background=background,
                Vm0_mV=Vm,
                gates0=gates,
                state0=state,
                intracellular_current_density_mid=iinj_chunk,
                extracellular_potential_mid_mV=vext_chunk,
                record_indices=record_indices,
                dt_ms=dt,
            )
        with benchmark_span(
            "kernel.chunk_bookkeeping",
            mode="single",
            variant="dense_vstim",
            output="full_vm" if record_full else "probe_vm",
            group_size=batch_size,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
        ):
            chunks.append(trace)
            recording_chunks.append(recording_trace)
            if progress_callback is not None:
                progress_callback(chunk_index, len(chunk_ranges))

    return _RecordedTrace(
        Vm=_concat_trace_chunks(chunks),
        recordings=_concat_recording_chunks(recording_chunks),
    )

def _run_single_cable_factorized_vstim_batch_array_chunks(
    *,
    runtime: SolverRuntime,
    Cm_uF_cm2: Array,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: FactorizedExtracellularPotentialBatch,
    record_indices: Array,
    record_full: bool,
    time_chunk_steps: int | None,
    record_outputs: dict[str, bool],
    progress_callback: Callable[[int, int], None] | None,
) -> _RecordedTrace:
    membrane_runtime = runtime.membrane
    grid = runtime.grid
    cable = runtime.cable
    dtype_local = membrane_runtime.dtype
    dt = jnp.asarray(grid.dt_ms, dtype=dtype_local)
    batch_size = extracellular_potential_mid_mV.batch_size
    with benchmark_span(
        "kernel.prepare_arrays",
        mode="single",
        variant="factorized_vstim",
        output="full_vm" if record_full else "probe_vm",
        group_size=batch_size,
        nx=membrane_runtime.Nx,
        nt=grid.Nt,
        time_chunk_steps=time_chunk_steps,
    ):
        current_rows_mid_A = _factorized_current_mid_rows(
            extracellular_potential_mid_mV,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        lower = _as_cached_batched_space_array(
            "lower", cable.lower, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        diag = _as_cached_batched_space_array(
            "diag", cable.diag, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        upper = _as_cached_batched_space_array(
            "upper", cable.upper, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        cm = _as_cached_batched_scalar_or_space_array(
            "Cm_uF_cm2",
            Cm_uF_cm2,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        background = _as_cached_batched_space_array(
            "I_background",
            membrane_runtime.background_current,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        forcing_footprint_mV_per_A = _single_cable_factorized_forcing_footprint_for_batch(
            extracellular_potential_mid_mV,
            lower=lower,
            upper=upper,
            lower_cache_source=cable.lower,
            upper_cache_source=cable.upper,
            dtype_local=dtype_local,
        )
    Vm, gates, state = _initial_single_cable_batch_state(runtime, batch_size)
    chunks = []
    recording_chunks: list[dict[str, Any]] = []

    chunk_ranges = tuple(_time_chunks(grid.Nt, time_chunk_steps))
    for chunk_index, (start, stop) in enumerate(chunk_ranges, start=1):
        with benchmark_span(
            "kernel.chunk_setup",
            mode="single",
            variant="factorized_vstim",
            output="full_vm" if record_full else "probe_vm",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
        ):
            iinj_chunk = intracellular_current_density_mid[:, start:stop]
            current_chunk = current_rows_mid_A[:, :, start:stop]
        with benchmark_span(
            "kernel.dispatch_jax",
            mode="single",
            variant="factorized_vstim",
            output="full_vm" if record_full else "probe_vm",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
        ):
            Vm, gates, state, trace, recording_trace = _run_single_cable_factorized_vstim_batch_stateful_scan(
                backend=membrane_runtime.backend,
                membrane=membrane_runtime.membrane,
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
                record_full=record_full,
                record_gates=record_outputs["gates"],
                record_currents=record_outputs["currents"],
                record_conductances=record_outputs["conductances"],
                record_states=record_outputs["states"],
                lower=lower,
                diag=diag,
                upper=upper,
                dl=-dt * lower,
                d_static=jnp.ones_like(diag) - dt * diag,
                du=-dt * upper,
                Cm_uF_cm2=cm,
                I_background=background,
                Vm0_mV=Vm,
                gates0=gates,
                state0=state,
                intracellular_current_density_mid=iinj_chunk,
                extracellular_current_mid_A=current_chunk,
                extracellular_forcing_footprint_mV_per_A=forcing_footprint_mV_per_A,
                record_indices=record_indices,
                dt_ms=dt,
            )
        with benchmark_span(
            "kernel.chunk_bookkeeping",
            mode="single",
            variant="factorized_vstim",
            output="full_vm" if record_full else "probe_vm",
            group_size=batch_size,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
        ):
            chunks.append(trace)
            recording_chunks.append(recording_trace)
            if progress_callback is not None:
                progress_callback(chunk_index, len(chunk_ranges))

    return _RecordedTrace(
        Vm=_concat_trace_chunks(chunks),
        recordings=_concat_recording_chunks(recording_chunks),
    )

def _run_single_cable_factorized_vstim_batch_observer_chunks(
    *,
    runtime: SolverRuntime,
    Cm_uF_cm2: Array,
    observers: VmRasterPlan,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: FactorizedExtracellularPotentialBatch,
    time_chunk_steps: int | None,
    progress_callback: Callable[[int, int], None] | None,
) -> VmRasterState:
    membrane_runtime = runtime.membrane
    grid = runtime.grid
    cable = runtime.cable
    dtype_local = membrane_runtime.dtype
    dt = jnp.asarray(grid.dt_ms, dtype=dtype_local)
    batch_size = extracellular_potential_mid_mV.batch_size
    with benchmark_span(
        "kernel.prepare_arrays",
        mode="single",
        variant="factorized_vstim",
        output="observer_only",
        observer="vm_raster",
        group_size=batch_size,
        nx=membrane_runtime.Nx,
        nt=grid.Nt,
        time_chunk_steps=time_chunk_steps,
    ):
        current_rows_mid_A = _factorized_current_mid_rows(
            extracellular_potential_mid_mV,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        lower = _as_cached_batched_space_array(
            "lower", cable.lower, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        diag = _as_cached_batched_space_array(
            "diag", cable.diag, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        upper = _as_cached_batched_space_array(
            "upper", cable.upper, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        cm = _as_cached_batched_scalar_or_space_array(
            "Cm_uF_cm2",
            Cm_uF_cm2,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        background = _as_cached_batched_space_array(
            "I_background",
            membrane_runtime.background_current,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        forcing_footprint_mV_per_A = _single_cable_factorized_forcing_footprint_for_batch(
            extracellular_potential_mid_mV,
            lower=lower,
            upper=upper,
            lower_cache_source=cable.lower,
            upper_cache_source=cable.upper,
            dtype_local=dtype_local,
        )
    Vm, gates, state = _initial_single_cable_batch_state(runtime, batch_size)
    raster_probe_indices, raster_probe_mask = _vm_raster_probe_tables_for_kernel(
        observers,
        batch_size=batch_size,
    )

    chunk_ranges = tuple(_time_chunks(grid.Nt, time_chunk_steps))
    resolved_observer_state_scope = _resolve_vm_raster_observer_state_scope(
        None,
        time_chunk_steps=time_chunk_steps,
    )
    local_observer_chunks = resolved_observer_state_scope == "chunk"
    observer_chunk_state_template = _init_local_vm_raster_chunk_template(
        observers,
        batch_size=batch_size,
        chunk_ranges=chunk_ranges,
        mode="single",
        variant="factorized_vstim",
        time_chunk_steps=time_chunk_steps,
        enabled=local_observer_chunks,
    )
    observer_state = (
        None
        if local_observer_chunks
        else init_vm_raster_state(observers, batch_size=batch_size, nt=grid.Nt)
    )
    observer_chunk_states: list[VmRasterState] = []
    observer_chunk_starts: list[int] = []
    observer_chunk_lengths: list[int] = []
    for chunk_index, (start, stop) in enumerate(chunk_ranges, start=1):
        with benchmark_span(
            "kernel.chunk_setup",
            mode="single",
            observer="vm_raster",
            variant="factorized_vstim",
            output="observer_only",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
            observer_state_scope="chunk" if local_observer_chunks else "full",
        ):
            observer_state0 = (
                observer_chunk_state_template
                if local_observer_chunks
                else observer_state
            )
            assert observer_state0 is not None
            iinj_chunk = intracellular_current_density_mid[:, start:stop]
            current_chunk = current_rows_mid_A[:, :, start:stop]
        with benchmark_span(
            "kernel.dispatch_jax",
            mode="single",
            observer="vm_raster",
            variant="factorized_vstim",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
            observer_state_scope="chunk" if local_observer_chunks else "full",
        ):
            Vm, gates, state, observer_state = _run_single_cable_factorized_vstim_batch_observer_scan(
                backend=membrane_runtime.backend,
                membrane=membrane_runtime.membrane,
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
                dl=-dt * lower,
                d_static=jnp.ones_like(diag) - dt * diag,
                du=-dt * upper,
                Cm_uF_cm2=cm,
                I_background=background,
                Vm0_mV=Vm,
                gates0=gates,
                state0=state,
                observer_state0=observer_state0,
                raster_probe_indices=raster_probe_indices,
                raster_probe_mask=raster_probe_mask,
                raster_thresholds_mV=observers.thresholds_mV,
                intracellular_current_density_mid=iinj_chunk,
                extracellular_current_mid_A=current_chunk,
                extracellular_forcing_footprint_mV_per_A=forcing_footprint_mV_per_A,
                time_start_index=jnp.asarray(
                    0 if local_observer_chunks else start,
                    dtype=jnp.int32,
                ),
                dt_ms=dt,
            )
        with benchmark_span(
            "kernel.chunk_bookkeeping",
            mode="single",
            observer="vm_raster",
            variant="factorized_vstim",
            output="observer_only",
            group_size=batch_size,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
        ):
            if local_observer_chunks:
                observer_chunk_states.append(observer_state)
                observer_chunk_starts.append(start)
                observer_chunk_lengths.append(stop - start)
            if progress_callback is not None:
                progress_callback(chunk_index, len(chunk_ranges))

    if local_observer_chunks:
        return _combine_vm_raster_chunk_states(
            observer_chunk_states,
            starts=observer_chunk_starts,
            lengths=observer_chunk_lengths,
            nt=grid.Nt,
            mode="single",
            variant="factorized_vstim",
            time_chunk_steps=time_chunk_steps,
        )
    assert observer_state is not None
    return observer_state

def _run_single_cable_vstim_batch_observer_chunks(
    *,
    runtime: SolverRuntime,
    Cm_uF_cm2: Array,
    observers: VmRasterPlan,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    time_chunk_steps: int | None,
    progress_callback: Callable[[int, int], None] | None,
) -> VmRasterState:
    membrane_runtime = runtime.membrane
    grid = runtime.grid
    cable = runtime.cable
    dtype_local = membrane_runtime.dtype
    dt = jnp.asarray(grid.dt_ms, dtype=dtype_local)
    batch_size = int(extracellular_potential_mid_mV.shape[0])
    with benchmark_span(
        "kernel.prepare_arrays",
        mode="single",
        variant="dense_vstim",
        output="observer_only",
        observer="vm_raster",
        group_size=batch_size,
        nx=membrane_runtime.Nx,
        nt=grid.Nt,
        time_chunk_steps=time_chunk_steps,
    ):
        lower = _as_batched_space_array(
            "lower", cable.lower, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        diag = _as_batched_space_array(
            "diag", cable.diag, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        upper = _as_batched_space_array(
            "upper", cable.upper, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        cm = _as_batched_scalar_or_space_array(
            "Cm_uF_cm2",
            Cm_uF_cm2,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        background = _as_batched_space_array(
            "I_background",
            membrane_runtime.background_current,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
    Vm, gates, state = _initial_single_cable_batch_state(runtime, batch_size)
    raster_probe_indices, raster_probe_mask = _vm_raster_probe_tables_for_kernel(
        observers,
        batch_size=batch_size,
    )

    chunk_ranges = tuple(_time_chunks(grid.Nt, time_chunk_steps))
    resolved_observer_state_scope = _resolve_vm_raster_observer_state_scope(
        None,
        time_chunk_steps=time_chunk_steps,
    )
    local_observer_chunks = resolved_observer_state_scope == "chunk"
    observer_chunk_state_template = _init_local_vm_raster_chunk_template(
        observers,
        batch_size=batch_size,
        chunk_ranges=chunk_ranges,
        mode="single",
        variant="dense_vstim",
        time_chunk_steps=time_chunk_steps,
        enabled=local_observer_chunks,
    )
    observer_state = (
        None
        if local_observer_chunks
        else init_vm_raster_state(observers, batch_size=batch_size, nt=grid.Nt)
    )
    observer_chunk_states: list[VmRasterState] = []
    observer_chunk_starts: list[int] = []
    observer_chunk_lengths: list[int] = []
    for chunk_index, (start, stop) in enumerate(chunk_ranges, start=1):
        with benchmark_span(
            "kernel.chunk_setup",
            mode="single",
            observer="vm_raster",
            variant="dense_vstim",
            output="observer_only",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
            observer_state_scope="chunk" if local_observer_chunks else "full",
        ):
            observer_state0 = (
                observer_chunk_state_template
                if local_observer_chunks
                else observer_state
            )
            assert observer_state0 is not None
            iinj_chunk = intracellular_current_density_mid[:, start:stop]
            vext_chunk = extracellular_potential_mid_mV[:, start:stop]
        with benchmark_span(
            "kernel.dispatch_jax",
            mode="single",
            observer="vm_raster",
            variant="dense_vstim",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
            observer_state_scope="chunk" if local_observer_chunks else "full",
        ):
            Vm, gates, state, observer_state = _run_single_cable_vstim_batch_observer_scan(
                backend=membrane_runtime.backend,
                membrane=membrane_runtime.membrane,
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
                lower=lower,
                diag=diag,
                upper=upper,
                dl=-dt * lower,
                d_static=jnp.ones_like(diag) - dt * diag,
                du=-dt * upper,
                Cm_uF_cm2=cm,
                I_background=background,
                Vm0_mV=Vm,
                gates0=gates,
                state0=state,
                observer_state0=observer_state0,
                raster_probe_indices=raster_probe_indices,
                raster_probe_mask=raster_probe_mask,
                raster_thresholds_mV=observers.thresholds_mV,
                intracellular_current_density_mid=iinj_chunk,
                extracellular_potential_mid_mV=vext_chunk,
                time_start_index=jnp.asarray(
                    0 if local_observer_chunks else start,
                    dtype=jnp.int32,
                ),
                dt_ms=dt,
            )
        with benchmark_span(
            "kernel.chunk_bookkeeping",
            mode="single",
            observer="vm_raster",
            variant="dense_vstim",
            output="observer_only",
            group_size=batch_size,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
        ):
            if local_observer_chunks:
                observer_chunk_states.append(observer_state)
                observer_chunk_starts.append(start)
                observer_chunk_lengths.append(stop - start)
            if progress_callback is not None:
                progress_callback(chunk_index, len(chunk_ranges))

    if local_observer_chunks:
        return _combine_vm_raster_chunk_states(
            observer_chunk_states,
            starts=observer_chunk_starts,
            lengths=observer_chunk_lengths,
            nt=grid.Nt,
            mode="single",
            variant="dense_vstim",
            time_chunk_steps=time_chunk_steps,
        )
    assert observer_state is not None
    return observer_state

def _run_single_cable_factorized_vstim_batch_sparse_observer_chunks(
    *,
    runtime: SolverRuntime,
    Cm_uF_cm2: Array,
    observers: VmRasterPlan,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    intracellular_current_density_mid: SparseIntracellularCurrentDensityBatch,
    extracellular_potential_mid_mV: FactorizedExtracellularPotentialBatch,
    time_chunk_steps: int | None,
    progress_callback: Callable[[int, int], None] | None,
) -> VmRasterState:
    membrane_runtime = runtime.membrane
    grid = runtime.grid
    cable = runtime.cable
    dtype_local = membrane_runtime.dtype
    dt = jnp.asarray(grid.dt_ms, dtype=dtype_local)
    batch_size = extracellular_potential_mid_mV.batch_size
    shared_rank1_current = (
        extracellular_potential_mid_mV.shared_current
        and extracellular_potential_mid_mV.drive_count == 1
    )
    has_sparse_iinj = intracellular_current_density_mid.max_sparse_entries > 0
    current_layout = "shared_rank1" if shared_rank1_current else "batched"
    with benchmark_span(
        "kernel.prepare_arrays",
        mode="single",
        variant="factorized_sparse_vstim",
        output="observer_only",
        observer="vm_raster",
        group_size=batch_size,
        nx=membrane_runtime.Nx,
        nt=grid.Nt,
        time_chunk_steps=time_chunk_steps,
        current_layout=current_layout,
    ):
        current_rows_mid_A = (
            None
            if shared_rank1_current
            else _factorized_current_mid_rows(
                extracellular_potential_mid_mV,
                dtype_local=dtype_local,
                batch_size=batch_size,
            )
        )
        lower = _as_cached_batched_space_array(
            "lower", cable.lower, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        diag = _as_cached_batched_space_array(
            "diag", cable.diag, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        upper = _as_cached_batched_space_array(
            "upper", cable.upper, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        cm = _as_cached_batched_scalar_or_space_array(
            "Cm_uF_cm2",
            Cm_uF_cm2,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        background = _as_cached_batched_space_array(
            "I_background",
            membrane_runtime.background_current,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        dl, d_static, du = _cached_single_cable_tridiagonal_coefficients(
            lower=lower,
            diag=diag,
            upper=upper,
            dt=dt,
            dt_ms=grid.dt_ms,
        )
    Vm, gates, state = _initial_single_cable_batch_state(runtime, batch_size)
    with benchmark_span(
        "kernel.prepare_observer_tables",
        mode="single",
        variant="factorized_sparse_vstim",
        output="observer_only",
        observer="vm_raster",
        group_size=batch_size,
    ):
        raster_probe_indices, raster_probe_mask = _vm_raster_probe_tables_for_kernel(
            observers,
            batch_size=batch_size,
        )
    with benchmark_span(
        "kernel.prepare_chunk_ranges",
        mode="single",
        variant="factorized_sparse_vstim",
        output="observer_only",
        observer="vm_raster",
        nt=grid.Nt,
        time_chunk_steps=time_chunk_steps,
    ):
        chunk_ranges = tuple(_time_chunks(grid.Nt, time_chunk_steps))
        resolved_observer_state_scope = _resolve_vm_raster_observer_state_scope(
            None,
            time_chunk_steps=time_chunk_steps,
        )
        local_observer_chunks = resolved_observer_state_scope == "chunk"
    observer_chunk_state_template = _init_local_vm_raster_chunk_template(
        observers,
        batch_size=batch_size,
        chunk_ranges=chunk_ranges,
        mode="single",
        variant="factorized_sparse_vstim",
        time_chunk_steps=time_chunk_steps,
        enabled=local_observer_chunks,
    )
    if local_observer_chunks:
        observer_state = None
    else:
        with benchmark_span(
            "kernel.prepare_observer_state",
            mode="single",
            variant="factorized_sparse_vstim",
            output="observer_only",
            observer="vm_raster",
            group_size=batch_size,
            nt=grid.Nt,
            time_chunk_steps=time_chunk_steps,
        ):
            observer_state = init_vm_raster_state(
                observers,
                batch_size=batch_size,
                nt=grid.Nt,
            )
    observer_chunk_states: list[VmRasterState] = []
    observer_chunk_starts: list[int] = []
    observer_chunk_lengths: list[int] = []
    with benchmark_span(
        "kernel.prepare_factorized_current",
        mode="single",
        variant="factorized_sparse_vstim",
        output="observer_only",
        group_size=batch_size,
        current_layout=current_layout,
        current_rank=1 if shared_rank1_current else getattr(current_rows_mid_A, "ndim", None),
    ):
        current_mid_A = jnp.asarray(
            (
                extracellular_potential_mid_mV.current_mid_A
                if shared_rank1_current
                else current_rows_mid_A
            ),
            dtype=dtype_local,
        )
    forcing_footprint_mV_per_A = _single_cable_factorized_forcing_footprint_for_batch(
        extracellular_potential_mid_mV,
        lower=lower,
        upper=upper,
        lower_cache_source=cable.lower,
        upper_cache_source=cable.upper,
        dtype_local=dtype_local,
    )
    for chunk_index, (start, stop) in enumerate(chunk_ranges, start=1):
        with benchmark_span(
            "kernel.chunk_setup",
            mode="single",
            observer="vm_raster",
            variant="factorized_sparse_vstim",
            output="observer_only",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
            observer_state_scope="chunk" if local_observer_chunks else "full",
            current_layout=current_layout,
        ):
            observer_state0 = (
                observer_chunk_state_template
                if local_observer_chunks
                else observer_state
            )
            assert observer_state0 is not None
            iinj_values_chunk = intracellular_current_density_mid.density_mid[:, start:stop]
            current_chunk = (
                current_mid_A[start:stop]
                if shared_rank1_current
                else current_mid_A[:, :, start:stop]
            )
        with benchmark_span(
            "kernel.dispatch_jax",
            mode="single",
            observer="vm_raster",
            variant="factorized_sparse_vstim",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
            observer_state_scope="chunk" if local_observer_chunks else "full",
            current_layout=current_layout,
        ):
            common_kwargs = dict(
                backend=membrane_runtime.backend,
                membrane=membrane_runtime.membrane,
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
                dl=dl,
                d_static=d_static,
                du=du,
                Cm_uF_cm2=cm,
                I_background=background,
                Vm0_mV=Vm,
                gates0=gates,
                state0=state,
                observer_state0=observer_state0,
                raster_probe_indices=raster_probe_indices,
                raster_probe_mask=raster_probe_mask,
                raster_thresholds_mV=observers.thresholds_mV,
                intracellular_current_density_values_mid=iinj_values_chunk,
                intracellular_current_density_indices=intracellular_current_density_mid.indices,
                intracellular_current_density_mask=intracellular_current_density_mid.mask,
                extracellular_current_mid_A=current_chunk,
                extracellular_forcing_footprint_mV_per_A=forcing_footprint_mV_per_A,
                time_start_index=jnp.asarray(
                    0 if local_observer_chunks else start,
                    dtype=jnp.int32,
                ),
                dt_ms=dt,
            )
            if shared_rank1_current:
                Vm, gates, state, observer_state = (
                    _run_single_cable_shared_rank1_vstim_batch_sparse_observer_scan(
                        has_sparse_iinj=has_sparse_iinj,
                        **common_kwargs,
                    )
                )
            else:
                Vm, gates, state, observer_state = (
                    _run_single_cable_factorized_vstim_batch_sparse_observer_scan(
                        **common_kwargs,
                    )
                )
        with benchmark_span(
            "kernel.chunk_bookkeeping",
            mode="single",
            observer="vm_raster",
            variant="factorized_sparse_vstim",
            output="observer_only",
            group_size=batch_size,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
        ):
            if local_observer_chunks:
                observer_chunk_states.append(observer_state)
                observer_chunk_starts.append(start)
                observer_chunk_lengths.append(stop - start)
            if progress_callback is not None:
                progress_callback(chunk_index, len(chunk_ranges))

    if local_observer_chunks:
        return _combine_vm_raster_chunk_states(
            observer_chunk_states,
            starts=observer_chunk_starts,
            lengths=observer_chunk_lengths,
            nt=grid.Nt,
            mode="single",
            variant="factorized_sparse_vstim",
            time_chunk_steps=time_chunk_steps,
        )
    assert observer_state is not None
    return observer_state


def _run_single_cable_zero_vstim_batch_sparse_observer_chunks(
    *,
    runtime: SolverRuntime,
    Cm_uF_cm2: Array,
    observers: VmRasterPlan,
    stateless_vm_only: bool,
    intracellular_current_density_mid: SparseIntracellularCurrentDensityBatch,
    batch_size: int,
    time_chunk_steps: int | None,
    progress_callback: Callable[[int, int], None] | None,
) -> VmRasterState:
    membrane_runtime = runtime.membrane
    grid = runtime.grid
    cable = runtime.cable
    dtype_local = membrane_runtime.dtype
    dt = jnp.asarray(grid.dt_ms, dtype=dtype_local)
    with benchmark_span(
        "kernel.prepare_arrays",
        mode="single",
        variant="zero_sparse_vstim",
        output="observer_only",
        observer="vm_raster",
        group_size=batch_size,
        nx=membrane_runtime.Nx,
        nt=grid.Nt,
        time_chunk_steps=time_chunk_steps,
    ):
        lower = _as_batched_space_array(
            "lower", cable.lower, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        diag = _as_batched_space_array(
            "diag", cable.diag, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        upper = _as_batched_space_array(
            "upper", cable.upper, nx=membrane_runtime.Nx, dtype_local=dtype_local, batch_size=batch_size
        )
        cm = _as_batched_scalar_or_space_array(
            "Cm_uF_cm2",
            Cm_uF_cm2,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        background = _as_batched_space_array(
            "I_background",
            membrane_runtime.background_current,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
    Vm, gates, state = _initial_single_cable_batch_state(runtime, batch_size)
    raster_probe_indices, raster_probe_mask = _vm_raster_probe_tables_for_kernel(
        observers,
        batch_size=batch_size,
    )

    chunk_ranges = tuple(_time_chunks(grid.Nt, time_chunk_steps))
    resolved_observer_state_scope = _resolve_vm_raster_observer_state_scope(
        None,
        time_chunk_steps=time_chunk_steps,
    )
    local_observer_chunks = resolved_observer_state_scope == "chunk"
    observer_chunk_state_template = _init_local_vm_raster_chunk_template(
        observers,
        batch_size=batch_size,
        chunk_ranges=chunk_ranges,
        mode="single",
        variant="zero_sparse_vstim",
        time_chunk_steps=time_chunk_steps,
        enabled=local_observer_chunks,
    )
    observer_state = (
        None
        if local_observer_chunks
        else init_vm_raster_state(observers, batch_size=batch_size, nt=grid.Nt)
    )
    observer_chunk_states: list[VmRasterState] = []
    observer_chunk_starts: list[int] = []
    observer_chunk_lengths: list[int] = []
    for chunk_index, (start, stop) in enumerate(chunk_ranges, start=1):
        with benchmark_span(
            "kernel.chunk_setup",
            mode="single",
            observer="vm_raster",
            variant="zero_sparse_vstim",
            output="observer_only",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
            observer_state_scope="chunk" if local_observer_chunks else "full",
        ):
            observer_state0 = (
                observer_chunk_state_template
                if local_observer_chunks
                else observer_state
            )
            assert observer_state0 is not None
            iinj_values_chunk = intracellular_current_density_mid.density_mid[:, start:stop]
        with benchmark_span(
            "kernel.dispatch_jax",
            mode="single",
            observer="vm_raster",
            variant="zero_sparse_vstim",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
            observer_state_scope="chunk" if local_observer_chunks else "full",
        ):
            Vm, gates, state, observer_state = _run_single_cable_zero_vstim_batch_sparse_observer_scan(
                backend=membrane_runtime.backend,
                membrane=membrane_runtime.membrane,
                stateless_vm_only=stateless_vm_only,
                lower=lower,
                diag=diag,
                upper=upper,
                dl=-dt * lower,
                d_static=jnp.ones_like(diag) - dt * diag,
                du=-dt * upper,
                Cm_uF_cm2=cm,
                I_background=background,
                Vm0_mV=Vm,
                gates0=gates,
                state0=state,
                observer_state0=observer_state0,
                raster_probe_indices=raster_probe_indices,
                raster_probe_mask=raster_probe_mask,
                raster_thresholds_mV=observers.thresholds_mV,
                intracellular_current_density_values_mid=iinj_values_chunk,
                intracellular_current_density_indices=intracellular_current_density_mid.indices,
                intracellular_current_density_mask=intracellular_current_density_mid.mask,
                time_start_index=jnp.asarray(
                    0 if local_observer_chunks else start,
                    dtype=jnp.int32,
                ),
                dt_ms=dt,
            )
        with benchmark_span(
            "kernel.chunk_bookkeeping",
            mode="single",
            observer="vm_raster",
            variant="zero_sparse_vstim",
            output="observer_only",
            group_size=batch_size,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
        ):
            if local_observer_chunks:
                observer_chunk_states.append(observer_state)
                observer_chunk_starts.append(start)
                observer_chunk_lengths.append(stop - start)
            if progress_callback is not None:
                progress_callback(chunk_index, len(chunk_ranges))

    if local_observer_chunks:
        return _combine_vm_raster_chunk_states(
            observer_chunk_states,
            starts=observer_chunk_starts,
            lengths=observer_chunk_lengths,
            nt=grid.Nt,
            mode="single",
            variant="zero_sparse_vstim",
            time_chunk_steps=time_chunk_steps,
        )
    assert observer_state is not None
    return observer_state


def _initial_single_cable_batch_state(
    runtime: SolverRuntime,
    batch_size: int,
) -> tuple[Array, Array, tuple[Array, ...]]:
    with benchmark_span(
        "kernel.prepare_state",
        mode="single",
        group_size=batch_size,
        nx=runtime.membrane.Nx,
    ):
        membrane_runtime = runtime.membrane
        Vm = _cached_broadcast_batch_leading(membrane_runtime.Vm0_mV, batch_size)
        gates = _cached_broadcast_batch_leading(membrane_runtime.gates0, batch_size)
        state = tuple(
            _cached_broadcast_batch_leading(values, batch_size)
            for values in membrane_runtime.state0
        )
        return Vm, gates, state


def _recording_output_flags(plan: RecordingPlan | None) -> dict[str, bool]:
    return {
        "gates": bool(plan is not None and plan.gates),
        "currents": bool(plan is not None and plan.currents),
        "conductances": bool(plan is not None and plan.conductances),
        "states": bool(plan is not None and plan.state_variables),
    }


def _concat_recording_chunks(
    chunks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not chunks:
        return None
    names = tuple(chunks[0])
    out: dict[str, Any] = {}
    for name in names:
        out[name] = _concat_trace_chunks([chunk[name] for chunk in chunks])
    return out


def _recording_group(
    values: Any,
    names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        name: values[..., index]
        for index, name in enumerate(names)
        if index < int(values.shape[-1])
    }


def _recordings_for_plan(
    plan: RecordingPlan | None,
    trace: _RecordedTrace,
    *,
    observable_names: dict[str, tuple[str, ...]],
) -> dict[str, Any] | None:
    if plan is None:
        return {"Vm": trace.Vm}
    recordings: dict[str, Any] = {}
    if plan.voltage:
        recordings["Vm"] = trace.Vm
    if trace.recordings is not None:
        if plan.gates:
            group = _recording_group(
                trace.recordings["gates"],
                observable_names.get("gates", ()),
            )
            if group:
                recordings["gates"] = group
        if plan.currents:
            group = _recording_group(
                trace.recordings["currents"],
                observable_names.get("currents", ()),
            )
            if group:
                recordings["currents"] = group
        if plan.conductances:
            group = _recording_group(
                trace.recordings["conductances"],
                observable_names.get("conductances", ()),
            )
            if group:
                recordings["conductances"] = group
        if plan.state_variables:
            group = _recording_group(
                trace.recordings["states"],
                observable_names.get("states", ()),
            )
            if group:
                recordings["states"] = group
    return recordings or None

__all__ = [
    "SingleCableVStimBatchKernel",
]
