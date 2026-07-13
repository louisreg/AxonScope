"""Double-cable JAX batch kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp

from axonscope.benchmarking import benchmark_span
from axonscope.runtime.input_payloads import FactorizedExtracellularPotentialBatch
from axonscope.runtime.jax.inputs.payloads import (
    materialize_factorized_extracellular_potential_initial_previous,
    materialize_factorized_extracellular_potential_batch,
)
from axonscope.runtime.jax.cable_geometry import Array
from axonscope.runtime.jax.recording.observer import (
    PendingVmRasterObservation,
    VmRasterPlan,
    VmRasterState,
    init_vm_raster_state,
)
from axonscope.runtime.jax.policy.engine_types import JaxSolverEngine
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
    _double_cable_factorized_vext_can_stay_compact,
    _factorized_current_initial_previous_rows,
    _factorized_current_mid_rows,
)
from .inputs import (
    _as_batched_edge_array,
    _as_batched_space_array,
    _as_batched_time_space_array,
    _as_cached_batched_space_array,
    _as_edge_array,
    _as_factorized_extracellular_potential_batch,
    _as_scalar_or_space_array,
    _as_space_array,
    _cached_broadcast_batch_leading,
    _cached_constant_batched_space_array,
    _normalize_batch_options,
    _resolve_output_recording,
)
from .double_cable_cpu import (
    _run_double_cable_batch_observer_scan,
    _run_double_cable_batch_stateful_scan,
)
from .double_cable_gpu import (
    _run_double_cable_batch_observer_integrated_scan,
    _run_double_cable_batch_stateful_integrated_scan,
    _use_batch_native_double_cable_integrated_solver,
)
from axonscope.runtime.jax.recording.results import BatchKernelResult


_GPU_PLATFORMS = frozenset({"cuda", "gpu", "metal", "rocm"})
_CPU_DOUBLE_CABLE_BLOCK_SOLVER = "thomas"
_GPU_DOUBLE_CABLE_BLOCK_SOLVER = "jax_triton_loop_xb"
_SUPPORTED_DOUBLE_CABLE_BLOCK_SOLVERS = frozenset({_CPU_DOUBLE_CABLE_BLOCK_SOLVER})
_INTERNAL_DOUBLE_CABLE_BLOCK_SOLVERS = frozenset({_GPU_DOUBLE_CABLE_BLOCK_SOLVER})
_DEFAULT_TRITON_TILED_THOMAS_BLOCK_B = 32

def _resolve_double_cable_kernel_block_solver(
    solver: str,
    *,
    batch_size: int,
) -> str:
    if solver in _SUPPORTED_DOUBLE_CABLE_BLOCK_SOLVERS:
        return solver
    if solver in _INTERNAL_DOUBLE_CABLE_BLOCK_SOLVERS:
        return solver
    raise ValueError(
        "double_cable_block_solver must be 'thomas' on CPU or the resolved "
        "JAX GPU tiled-Thomas route."
    )

def _resolve_double_cable_run_solver_settings(
    solver_engine: JaxSolverEngine | None,
    *,
    platform: str,
) -> tuple[str, int]:
    normalized_platform = platform.lower()
    solver = (
        None if solver_engine is None else solver_engine.double_cable_block_solver
    )
    tiled_thomas_block_b = (
        None if solver_engine is None else solver_engine.tiled_thomas_block_b
    )
    if solver in (None, ""):
        resolved = (
            _GPU_DOUBLE_CABLE_BLOCK_SOLVER
            if normalized_platform in _GPU_PLATFORMS
            else _CPU_DOUBLE_CABLE_BLOCK_SOLVER
        )
        _guard_double_cable_gpu_route(
            platform=normalized_platform,
            block_solver=resolved,
        )
        return resolved, _normalize_tiled_thomas_block_b(tiled_thomas_block_b)
    if solver == "auto":
        raise ValueError(
            "double_cable_block_solver must be resolved before kernel dispatch; "
            "use ExecutionPolicy solvers for public auto selection."
        )
    if solver in _INTERNAL_DOUBLE_CABLE_BLOCK_SOLVERS:
        if normalized_platform not in _GPU_PLATFORMS:
            raise RuntimeError(
                f"Double-cable GPU solver {solver!r} requires a JAX GPU backend."
            )
        _guard_double_cable_gpu_route(
            platform=normalized_platform,
            block_solver=solver,
        )
        return solver, _normalize_tiled_thomas_block_b(tiled_thomas_block_b)
    if solver in _SUPPORTED_DOUBLE_CABLE_BLOCK_SOLVERS:
        if normalized_platform in _GPU_PLATFORMS:
            _guard_double_cable_gpu_route(
                platform=normalized_platform,
                block_solver=solver,
            )
        return solver, _normalize_tiled_thomas_block_b(tiled_thomas_block_b)
    raise ValueError(
        "double_cable_block_solver must be 'thomas' on CPU or the resolved "
        "JAX GPU tiled-Thomas route."
    )


def _guard_double_cable_gpu_route(*, platform: str, block_solver: str) -> None:
    if platform in _GPU_PLATFORMS and block_solver != _GPU_DOUBLE_CABLE_BLOCK_SOLVER:
        raise RuntimeError(
            "JAX GPU double-cable execution resolved to the non-GPU route "
            f"{block_solver!r}; expected {_GPU_DOUBLE_CABLE_BLOCK_SOLVER!r}. "
            "Pass a GPU ExecutionPolicy and keep public GPU solver policy on "
            "auto/tiled_thomas."
        )

def _resolve_double_cable_run_block_solver(
    solver_engine: JaxSolverEngine | None,
    *,
    platform: str,
) -> str:
    block_solver, _ = _resolve_double_cable_run_solver_settings(
        solver_engine,
        platform=platform,
    )
    return block_solver

def _normalize_tiled_thomas_block_b(block_b: int | None) -> int:
    if block_b is None:
        return _DEFAULT_TRITON_TILED_THOMAS_BLOCK_B
    value = int(block_b)
    if value < 1:
        raise ValueError("tiled Thomas block_b must be >= 1.")
    return value

@dataclass(frozen=True)
class DoubleCableBatchKernel:
    """Batch-oriented full double-cable kernel with shared axon structure.

    This intentionally keeps the first pool constraint simple: all batch
    rows share geometry, membrane model, extracellular parameters, initial
    state, and time grid. Only imposed ``Vstim`` and optional ``Iinj`` vary.
    """

    runtime: SolverRuntime
    Veinit_mV: float = 0.0
    has_driven_extracellular: bool | None = None

    def run(
        self,
        *,
        extracellular_potential_mid_mV: Array | FactorizedExtracellularPotentialBatch | None = None,
        extracellular_potential_initial_previous_mV: Array | None = None,
        intracellular_current_density_mid: Array | None = None,
        options: BatchOptions | None = None,
        observers: VmRasterPlan | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        solver_engine: JaxSolverEngine | None = None,
        benchmark_observer_state_scope: str | None = None,
    ) -> BatchKernelResult:
        runtime = self.runtime
        extracellular = runtime.extracellular
        if extracellular is None:
            raise ValueError(
                "DoubleCableBatchKernel requires extracellular runtime arrays; "
                "prepare it with include_extracellular=True."
            )

        membrane_runtime = runtime.membrane
        grid = runtime.grid
        cable = runtime.cable
        dtype_local = membrane_runtime.dtype
        nx = membrane_runtime.Nx

        with benchmark_span("kernel.prepare_inputs", mode="double", nx=nx, nt=grid.Nt):
            vext_mid = (
                runtime.stimulation.extracellular_potential_mid_mV
                if extracellular_potential_mid_mV is None
                else extracellular_potential_mid_mV
            )
            if vext_mid is None:
                raise ValueError(
                    "extracellular_potential_mid_mV is required for double-cable batching."
                )
            factorized_vext = None
            factorized_source = None
            if isinstance(vext_mid, FactorizedExtracellularPotentialBatch):
                factorized_source = _as_factorized_extracellular_potential_batch(
                    "extracellular_potential_mid_mV",
                    vext_mid,
                    nt=grid.Nt,
                    nx=nx,
                    dtype_local=dtype_local,
                )
                if _double_cable_factorized_vext_can_stay_compact(factorized_source):
                    factorized_vext = factorized_source
                    vext_batch = None
                    batch_size = factorized_vext.batch_size
                else:
                    with benchmark_span(
                        "kernel.materialize_inputs",
                        mode="double",
                        input="factorized_vext",
                    ):
                        vext_batch = materialize_factorized_extracellular_potential_batch(
                            factorized_source
                        )
                    batch_size = factorized_source.batch_size
            else:
                vext_batch = _as_batched_time_space_array(
                    "extracellular_potential_mid_mV",
                    vext_mid,
                    nt=grid.Nt,
                    nx=nx,
                    dtype_local=dtype_local,
                )
                batch_size = int(vext_batch.shape[0])

            if factorized_vext is None:
                vext_previous = (
                    runtime.stimulation.extracellular_potential_initial_previous_mV
                    if extracellular_potential_initial_previous_mV is None
                    else extracellular_potential_initial_previous_mV
                )
                if vext_previous is None and factorized_source is not None:
                    with benchmark_span(
                        "kernel.materialize_inputs",
                        mode="double",
                        input="factorized_vext_previous",
                        group_size=batch_size,
                    ):
                        vext_previous = (
                            materialize_factorized_extracellular_potential_initial_previous(
                                factorized_source
                            )
                        )
                if vext_previous is None:
                    raise ValueError(
                        "extracellular_potential_initial_previous_mV is required "
                        "for double-cable batching."
                    )
                vext_previous_batch = _as_batched_space_array(
                    "extracellular_potential_initial_previous_mV",
                    vext_previous,
                    nx=nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )
            else:
                vext_previous_batch = None

            iinj_mid = (
                runtime.stimulation.intracellular_current_density_mid
                if intracellular_current_density_mid is None
                else intracellular_current_density_mid
            )
            if iinj_mid is None:
                iinj_batch = None
            else:
                iinj_batch = _as_batched_time_space_array(
                    "intracellular_current_density_mid",
                    iinj_mid,
                    nt=grid.Nt,
                    nx=nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )

            options = _normalize_batch_options(options)
            record_idx, record_full = _resolve_output_recording(options, nx=nx)
            chunk_steps = _normalize_time_chunk_steps(options.time_chunk_steps, nt=grid.Nt)
            has_driven_extracellular = (
                runtime.stimulation.has_driven_extracellular
                if self.has_driven_extracellular is None
                else bool(self.has_driven_extracellular)
            )
            stateless_vm_only = bool(
                membrane_runtime.membrane.supports_stateless_vm_only_fast_path()
            )
            block_solver, tiled_thomas_block_b = (
                _resolve_double_cable_run_solver_settings(
                    solver_engine,
                    platform=_effective_double_cable_platform(solver_engine),
                )
            )
        if observers is not None and options.recording.mode == "none":
            if factorized_vext is not None:
                observer_state = _run_double_cable_batch_observer_chunks(
                    runtime=runtime,
                    Veinit_mV=float(self.Veinit_mV),
                    observers=observers,
                    has_driven_extracellular=has_driven_extracellular,
                    stateless_vm_only=stateless_vm_only,
                    double_cable_block_solver=block_solver,
                    tiled_thomas_block_b=tiled_thomas_block_b,
                    intracellular_current_density_mid=iinj_batch,
                    extracellular_potential_mid_mV=factorized_vext,
                    extracellular_potential_initial_previous_mV=None,
                    time_chunk_steps=chunk_steps,
                    observer_state_scope=benchmark_observer_state_scope,
                    progress_callback=progress_callback,
                )
            else:
                assert vext_batch is not None
                assert vext_previous_batch is not None
                observer_state = _run_double_cable_batch_observer_chunks(
                    runtime=runtime,
                    Veinit_mV=float(self.Veinit_mV),
                    observers=observers,
                    has_driven_extracellular=has_driven_extracellular,
                    stateless_vm_only=stateless_vm_only,
                    double_cable_block_solver=block_solver,
                    tiled_thomas_block_b=tiled_thomas_block_b,
                    intracellular_current_density_mid=iinj_batch,
                    extracellular_potential_mid_mV=vext_batch,
                    extracellular_potential_initial_previous_mV=vext_previous_batch,
                    time_chunk_steps=chunk_steps,
                    observer_state_scope=benchmark_observer_state_scope,
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
        out = _run_double_cable_batch_array_chunks(
            runtime=runtime,
            Veinit_mV=float(self.Veinit_mV),
            has_driven_extracellular=has_driven_extracellular,
            stateless_vm_only=stateless_vm_only,
            double_cable_block_solver=block_solver,
            tiled_thomas_block_b=tiled_thomas_block_b,
            intracellular_current_density_mid=iinj_batch,
            extracellular_potential_mid_mV=(
                cast(Any, factorized_vext)
                if factorized_vext is not None
                else cast(Any, vext_batch)
            ),
            extracellular_potential_initial_previous_mV=cast(Any, vext_previous_batch),
            record_indices=record_idx,
            record_full=record_full,
            time_chunk_steps=chunk_steps,
            progress_callback=progress_callback,
        )
        return BatchKernelResult(Vm=out, t=grid.t_vec_ms)


def _effective_double_cable_platform(solver_engine: JaxSolverEngine | None) -> str:
    if solver_engine is not None and solver_engine.platform:
        return str(solver_engine.platform)
    return str(jax.default_backend())

def _run_double_cable_batch_array_chunks(
    *,
    runtime: SolverRuntime,
    Veinit_mV: float,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    double_cable_block_solver: str,
    tiled_thomas_block_b: int | None,
    intracellular_current_density_mid: Array | None,
    extracellular_potential_mid_mV: Array | FactorizedExtracellularPotentialBatch,
    extracellular_potential_initial_previous_mV: Array | None,
    record_indices: Array,
    record_full: bool,
    time_chunk_steps: int | None,
    progress_callback: Callable[[int, int], None] | None,
) -> Array:
    membrane_runtime = runtime.membrane
    extracellular = runtime.extracellular
    if extracellular is None:
        raise ValueError("double-cable batch chunks require extracellular runtime arrays.")
    grid = runtime.grid
    dtype_local = membrane_runtime.dtype
    nx = membrane_runtime.Nx
    factorized_vext = isinstance(
        extracellular_potential_mid_mV,
        FactorizedExtracellularPotentialBatch,
    )
    if factorized_vext:
        factorized_batch = _as_factorized_extracellular_potential_batch(
            "extracellular_potential_mid_mV",
            extracellular_potential_mid_mV,
            nt=grid.Nt,
            nx=nx,
            dtype_local=dtype_local,
        )
        if factorized_batch.drive_count != 1:
            raise ValueError("double-cable compact factorized Vext requires one drive.")
        batch_size = factorized_batch.batch_size
        current_rows_mid_A = _factorized_current_mid_rows(
            factorized_batch,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        if int(current_rows_mid_A.shape[1]) != 1:
            raise ValueError("double-cable compact factorized Vext requires one current row.")
        if (
            factorized_batch.shared_current
            and factorized_batch.current_row_scales is None
        ):
            factorized_current_mid_A = jnp.asarray(
                factorized_batch.current_mid_A,
                dtype=dtype_local,
            )
        else:
            factorized_current_mid_A = current_rows_mid_A[:, 0, :]
        factorized_previous_current_A = _factorized_current_initial_previous_rows(
            factorized_batch,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        factorized_footprint_mV_per_A = jnp.asarray(
            factorized_batch.footprint_mV_per_A,
            dtype=dtype_local,
        )
    else:
        batch_size = int(cast(Any, extracellular_potential_mid_mV).shape[0])
        factorized_current_mid_A = None
        factorized_previous_current_A = None
        factorized_footprint_mV_per_A = None
    kernel_block_solver = _resolve_double_cable_kernel_block_solver(
        double_cable_block_solver,
        batch_size=batch_size,
    )
    kernel_tiled_thomas_block_b = _normalize_tiled_thomas_block_b(tiled_thomas_block_b)
    (
        area_cm2,
        Cm_abs,
        Cx_abs,
        Gx_abs,
        Gax_e,
        Gax_i,
        left_i,
        right_i,
        left_e,
        right_e,
        background,
        shared_coefficients,
    ) = _prepare_double_cable_batch_arrays(
        runtime=runtime,
        batch_size=batch_size,
        output="full_vm" if record_full else "probe_vm",
        variant=kernel_block_solver,
        time_chunk_steps=time_chunk_steps,
        factorized_vext=factorized_vext,
    )
    Vi, Ve, gates, state = _initial_double_cable_batch_state(runtime, batch_size, Veinit_mV)
    previous = extracellular_potential_initial_previous_mV
    chunks = []

    chunk_ranges = tuple(_time_chunks(grid.Nt, time_chunk_steps))
    for chunk_index, (start, stop) in enumerate(chunk_ranges, start=1):
        with benchmark_span(
            "kernel.chunk_setup",
            mode="double",
            variant=kernel_block_solver,
            output="full_vm" if record_full else "probe_vm",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
            shared_coefficients=shared_coefficients,
            tiled_thomas_block_b=(
                kernel_tiled_thomas_block_b
                if kernel_block_solver == "jax_triton_loop_xb"
                else None
            ),
        ):
            if factorized_vext:
                vext_chunk = None
                assert factorized_current_mid_A is not None
                current_chunk = (
                    factorized_current_mid_A[start:stop]
                    if jnp.asarray(factorized_current_mid_A).ndim == 1
                    else factorized_current_mid_A[:, start:stop]
                )
            else:
                vext_chunk = cast(Any, extracellular_potential_mid_mV)[:, start:stop]
                current_chunk = None
            iinj_chunk = (
                None
                if intracellular_current_density_mid is None
                else intracellular_current_density_mid[:, start:stop]
            )
        with benchmark_span(
            "kernel.dispatch_jax",
            mode="double",
            variant=kernel_block_solver,
            output="full_vm" if record_full else "probe_vm",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
            block_solver=kernel_block_solver,
            shared_coefficients=shared_coefficients,
            tiled_thomas_block_b=(
                kernel_tiled_thomas_block_b
                if kernel_block_solver == "jax_triton_loop_xb"
                else None
            ),
        ):
            if _use_batch_native_double_cable_integrated_solver(
                kernel_block_solver,
                batch_size=batch_size,
            ):
                Vi, Ve, gates, state, trace = _run_double_cable_batch_stateful_integrated_scan(
                    backend=membrane_runtime.backend,
                    membrane=membrane_runtime.membrane,
                    has_driven_extracellular=has_driven_extracellular,
                    stateless_vm_only=stateless_vm_only,
                    record_full=record_full,
                    double_cable_block_solver=kernel_block_solver,
                    tiled_thomas_block_b=kernel_tiled_thomas_block_b,
                    Vi0_mV=Vi,
                    Ve0_mV=Ve,
                    gates0=gates,
                    state0=state,
                    area_cm2=area_cm2,
                    Cm_abs=Cm_abs,
                    Cx_abs=Cx_abs,
                    Gx_abs=Gx_abs,
                    Gax_e=Gax_e,
                    Gax_i=Gax_i,
                    left_i=left_i,
                    right_i=right_i,
                    left_e=left_e,
                    right_e=right_e,
                    I_background=background,
                    intracellular_current_density_mid=iinj_chunk,
                    extracellular_potential_mid_mV=vext_chunk,
                    extracellular_potential_initial_previous_mV=previous,
                    extracellular_current_mid_A=current_chunk,
                    extracellular_current_initial_previous_A=factorized_previous_current_A,
                    extracellular_footprint_mV_per_A=factorized_footprint_mV_per_A,
                    row_indices=jnp.arange(batch_size, dtype=jnp.int32),
                    record_indices=record_indices,
                    dt_ms=jnp.asarray(grid.dt_ms, dtype=dtype_local),
                )
            else:
                Vi, Ve, gates, state, trace = _run_double_cable_batch_stateful_scan(
                    backend=membrane_runtime.backend,
                    membrane=membrane_runtime.membrane,
                    has_driven_extracellular=has_driven_extracellular,
                    stateless_vm_only=stateless_vm_only,
                    record_full=record_full,
                    double_cable_block_solver=kernel_block_solver,
                    Vi0_mV=Vi,
                    Ve0_mV=Ve,
                    gates0=gates,
                    state0=state,
                    area_cm2=area_cm2,
                    Cm_abs=Cm_abs,
                    Cx_abs=Cx_abs,
                    Gx_abs=Gx_abs,
                    Gax_e=Gax_e,
                    Gax_i=Gax_i,
                    left_i=left_i,
                    right_i=right_i,
                    left_e=left_e,
                    right_e=right_e,
                    I_background=background,
                    intracellular_current_density_mid=iinj_chunk,
                    extracellular_potential_mid_mV=vext_chunk,
                    extracellular_potential_initial_previous_mV=previous,
                    extracellular_current_mid_A=current_chunk,
                    extracellular_current_initial_previous_A=factorized_previous_current_A,
                    extracellular_footprint_mV_per_A=factorized_footprint_mV_per_A,
                    row_indices=jnp.arange(batch_size, dtype=jnp.int32),
                    record_indices=record_indices,
                    dt_ms=jnp.asarray(grid.dt_ms, dtype=dtype_local),
                )
        with benchmark_span(
            "kernel.chunk_bookkeeping",
            mode="double",
            variant=kernel_block_solver,
            output="full_vm" if record_full else "probe_vm",
            group_size=batch_size,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
        ):
            if factorized_vext:
                assert current_chunk is not None
                factorized_previous_current_A = (
                    current_chunk[-1]
                    if jnp.asarray(current_chunk).ndim == 1
                    else current_chunk[:, -1]
                )
            else:
                assert vext_chunk is not None
                previous = vext_chunk[:, -1]
            chunks.append(trace)
            if progress_callback is not None:
                progress_callback(chunk_index, len(chunk_ranges))

    return _concat_trace_chunks(chunks)

def _prepare_double_cable_batch_arrays(
    *,
    runtime: SolverRuntime,
    batch_size: int,
    output: str,
    variant: str,
    time_chunk_steps: int | None,
    factorized_vext: bool,
    observer: str | None = None,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    bool,
]:
    membrane_runtime = runtime.membrane
    extracellular = runtime.extracellular
    if extracellular is None:
        raise ValueError("double-cable batch chunks require extracellular runtime arrays.")
    dtype_local = membrane_runtime.dtype
    nx = membrane_runtime.Nx
    shared_coefficients = (
        jnp.asarray(runtime.cable.area_cm2).ndim == 1
        and jnp.asarray(extracellular.Cm_abs).ndim == 1
        and jnp.asarray(extracellular.Cx_abs).ndim == 1
        and jnp.asarray(extracellular.Gx_abs).ndim == 1
        and jnp.asarray(extracellular.Gax_e).ndim == 1
        and jnp.asarray(extracellular.Gax_i).ndim == 1
        and jnp.asarray(extracellular.left_i).ndim == 1
        and jnp.asarray(extracellular.right_i).ndim == 1
        and jnp.asarray(extracellular.left_e).ndim == 1
        and jnp.asarray(extracellular.right_e).ndim == 1
        and jnp.asarray(membrane_runtime.background_current).ndim <= 1
    )
    metadata: dict[str, Any] = {
        "mode": "double",
        "variant": variant,
        "output": output,
        "group_size": batch_size,
        "nx": nx,
        "time_chunk_steps": time_chunk_steps,
        "shared_coefficients": shared_coefficients,
        "factorized_vext": factorized_vext,
    }
    if observer is not None:
        metadata["observer"] = observer
    with benchmark_span("kernel.prepare_arrays", **metadata):
        with benchmark_span("kernel.prepare_double_coefficients", **metadata):
            if shared_coefficients:
                area_cm2 = _as_space_array(
                    "area_cm2",
                    runtime.cable.area_cm2,
                    nx=nx,
                    dtype_local=dtype_local,
                )
                Cm_abs = _as_space_array(
                    "Cm_abs", extracellular.Cm_abs, nx=nx, dtype_local=dtype_local
                )
                Cx_abs = _as_space_array(
                    "Cx_abs", extracellular.Cx_abs, nx=nx, dtype_local=dtype_local
                )
                Gx_abs = _as_space_array(
                    "Gx_abs", extracellular.Gx_abs, nx=nx, dtype_local=dtype_local
                )
                Gax_e = _as_edge_array(
                    "Gax_e", extracellular.Gax_e, nx=nx, dtype_local=dtype_local
                )
                Gax_i = _as_edge_array(
                    "Gax_i", extracellular.Gax_i, nx=nx, dtype_local=dtype_local
                )
                left_i = _as_space_array(
                    "left_i", extracellular.left_i, nx=nx, dtype_local=dtype_local
                )
                right_i = _as_space_array(
                    "right_i",
                    extracellular.right_i,
                    nx=nx,
                    dtype_local=dtype_local,
                )
                left_e = _as_space_array(
                    "left_e", extracellular.left_e, nx=nx, dtype_local=dtype_local
                )
                right_e = _as_space_array(
                    "right_e",
                    extracellular.right_e,
                    nx=nx,
                    dtype_local=dtype_local,
                )
                background = _as_scalar_or_space_array(
                    "I_background",
                    membrane_runtime.background_current,
                    nx=nx,
                    dtype_local=dtype_local,
                )
            else:
                area_cm2 = _as_batched_space_array(
                    "area_cm2",
                    runtime.cable.area_cm2,
                    nx=nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )
                Cm_abs = _as_batched_space_array(
                    "Cm_abs",
                    extracellular.Cm_abs,
                    nx=nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )
                Cx_abs = _as_batched_space_array(
                    "Cx_abs",
                    extracellular.Cx_abs,
                    nx=nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )
                Gx_abs = _as_batched_space_array(
                    "Gx_abs",
                    extracellular.Gx_abs,
                    nx=nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )
                Gax_e = _as_batched_edge_array(
                    "Gax_e",
                    extracellular.Gax_e,
                    nx=nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )
                Gax_i = _as_batched_edge_array(
                    "Gax_i",
                    extracellular.Gax_i,
                    nx=nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )
                left_i = _as_batched_space_array(
                    "left_i",
                    extracellular.left_i,
                    nx=nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )
                right_i = _as_batched_space_array(
                    "right_i",
                    extracellular.right_i,
                    nx=nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )
                left_e = _as_batched_space_array(
                    "left_e",
                    extracellular.left_e,
                    nx=nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )
                right_e = _as_batched_space_array(
                    "right_e",
                    extracellular.right_e,
                    nx=nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )
                background = _as_batched_space_array(
                    "I_background",
                    membrane_runtime.background_current,
                    nx=nx,
                    dtype_local=dtype_local,
                    batch_size=batch_size,
                )
    return (
        area_cm2,
        Cm_abs,
        Cx_abs,
        Gx_abs,
        Gax_e,
        Gax_i,
        left_i,
        right_i,
        left_e,
        right_e,
        background,
        shared_coefficients,
    )

def _run_double_cable_batch_observer_chunks(
    *,
    runtime: SolverRuntime,
    Veinit_mV: float,
    observers: VmRasterPlan,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    double_cable_block_solver: str,
    tiled_thomas_block_b: int | None,
    intracellular_current_density_mid: Array | None,
    extracellular_potential_mid_mV: Array | FactorizedExtracellularPotentialBatch,
    extracellular_potential_initial_previous_mV: Array | None,
    time_chunk_steps: int | None,
    progress_callback: Callable[[int, int], None] | None,
    observer_state_scope: str | None = None,
) -> VmRasterState:
    membrane_runtime = runtime.membrane
    extracellular = runtime.extracellular
    if extracellular is None:
        raise ValueError("double-cable observer chunks require extracellular runtime arrays.")
    grid = runtime.grid
    dtype_local = membrane_runtime.dtype
    nx = membrane_runtime.Nx
    factorized_vext = (
        extracellular_potential_mid_mV
        if isinstance(extracellular_potential_mid_mV, FactorizedExtracellularPotentialBatch)
        else None
    )
    if factorized_vext is not None:
        if factorized_vext.current_initial_previous_A is None:
            raise ValueError(
                "factorized double-cable observer batches require "
                "current_initial_previous_A."
            )
        previous_current = jnp.asarray(factorized_vext.current_initial_previous_A)
        previous_shape_ok = previous_current.ndim == 0 or previous_current.shape == (
            factorized_vext.batch_size,
        )
        if not previous_shape_ok:
            with benchmark_span(
                "kernel.materialize_inputs",
                mode="double",
                input="factorized_vext",
                group_size=factorized_vext.batch_size,
            ):
                dense_vext = materialize_factorized_extracellular_potential_batch(
                    factorized_vext
                )
            with benchmark_span(
                "kernel.materialize_inputs",
                mode="double",
                input="factorized_vext_previous",
                group_size=factorized_vext.batch_size,
            ):
                dense_previous = (
                    materialize_factorized_extracellular_potential_initial_previous(
                        factorized_vext
                    )
                )
            return _run_double_cable_batch_observer_chunks(
                runtime=runtime,
                Veinit_mV=Veinit_mV,
                observers=observers,
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
                double_cable_block_solver=double_cable_block_solver,
                tiled_thomas_block_b=tiled_thomas_block_b,
                intracellular_current_density_mid=intracellular_current_density_mid,
                extracellular_potential_mid_mV=dense_vext,
                extracellular_potential_initial_previous_mV=dense_previous,
                time_chunk_steps=time_chunk_steps,
                observer_state_scope=observer_state_scope,
                progress_callback=progress_callback,
            )
        batch_size = factorized_vext.batch_size
    else:
        batch_size = int(cast(Any, extracellular_potential_mid_mV).shape[0])
        if extracellular_potential_initial_previous_mV is None:
            raise ValueError(
                "extracellular_potential_initial_previous_mV is required."
            )
    kernel_block_solver = _resolve_double_cable_kernel_block_solver(
        double_cable_block_solver,
        batch_size=batch_size,
    )
    kernel_tiled_thomas_block_b = _normalize_tiled_thomas_block_b(tiled_thomas_block_b)
    (
        area_cm2,
        Cm_abs,
        Cx_abs,
        Gx_abs,
        Gax_e,
        Gax_i,
        left_i,
        right_i,
        left_e,
        right_e,
        background,
        shared_coefficients,
    ) = _prepare_double_cable_batch_arrays(
        runtime=runtime,
        batch_size=batch_size,
        output="observer_only",
        variant=kernel_block_solver,
        time_chunk_steps=time_chunk_steps,
        factorized_vext=factorized_vext is not None,
        observer="vm_raster",
    )
    Vi, Ve, gates, state = _initial_double_cable_batch_state(runtime, batch_size, Veinit_mV)
    raster_probe_indices, raster_probe_mask = _vm_raster_probe_tables_for_kernel(
        observers,
        batch_size=batch_size,
    )
    previous = extracellular_potential_initial_previous_mV
    previous_current_A = None
    factorized_current_mid_A = None
    factorized_footprint_mV_per_A = None
    if factorized_vext is not None:
        with benchmark_span(
            "kernel.prepare_factorized_vext",
            mode="double",
            output="observer_only",
            observer="vm_raster",
            variant=kernel_block_solver,
            group_size=batch_size,
            nx=nx,
            nt=grid.Nt,
            time_chunk_steps=time_chunk_steps,
            drive_count=factorized_vext.drive_count,
            shared_current=factorized_vext.shared_current,
            footprint_rank=jnp.asarray(factorized_vext.footprint_mV_per_A).ndim,
            tiled_thomas_block_b=(
                kernel_tiled_thomas_block_b
                if kernel_block_solver == "jax_triton_loop_xb"
                else None
            ),
        ):
            previous_current_A = jnp.asarray(
                factorized_vext.current_initial_previous_A,
                dtype=dtype_local,
            )
            factorized_current_mid_A = jnp.asarray(
                factorized_vext.current_mid_A,
                dtype=dtype_local,
            )
            factorized_footprint_mV_per_A = jnp.asarray(
                factorized_vext.footprint_mV_per_A,
                dtype=dtype_local,
            )

    chunk_ranges = tuple(_time_chunks(grid.Nt, time_chunk_steps))
    resolved_observer_state_scope = _resolve_vm_raster_observer_state_scope(
        observer_state_scope,
        time_chunk_steps=time_chunk_steps,
    )
    local_observer_chunks = resolved_observer_state_scope == "chunk"
    observer_chunk_state_template = _init_local_vm_raster_chunk_template(
        observers,
        batch_size=batch_size,
        chunk_ranges=chunk_ranges,
        mode="double",
        variant=kernel_block_solver,
        time_chunk_steps=time_chunk_steps,
        enabled=local_observer_chunks,
    )
    observer_state = None
    if not local_observer_chunks:
        with benchmark_span(
            "kernel.prepare_observer_state",
            mode="double",
            output="observer_only",
            observer="vm_raster",
            variant=kernel_block_solver,
            state_scope="full",
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
    for chunk_index, (start, stop) in enumerate(chunk_ranges, start=1):
        with benchmark_span(
            "kernel.chunk_setup",
            mode="double",
            observer="vm_raster",
            variant=kernel_block_solver,
            output="observer_only",
            factorized_vext=factorized_vext is not None,
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
            observer_state_scope="chunk" if local_observer_chunks else "full",
            benchmark_observer_state_scope=observer_state_scope,
            resolved_observer_state_scope=resolved_observer_state_scope,
            tiled_thomas_block_b=(
                kernel_tiled_thomas_block_b
                if kernel_block_solver == "jax_triton_loop_xb"
                else None
            ),
        ):
            observer_state0 = (
                observer_chunk_state_template
                if local_observer_chunks
                else observer_state
            )
            if factorized_vext is None:
                vext_chunk = cast(Any, extracellular_potential_mid_mV)[:, start:stop]
                current_chunk = None
            else:
                assert factorized_current_mid_A is not None
                vext_chunk = None
                current_chunk = (
                    factorized_current_mid_A[start:stop]
                    if factorized_current_mid_A.ndim == 1
                    else factorized_current_mid_A[:, start:stop]
                )
            iinj_chunk = (
                None
                if intracellular_current_density_mid is None
                else intracellular_current_density_mid[:, start:stop]
            )
            assert observer_state0 is not None
            time_start_index = jnp.asarray(
                0 if local_observer_chunks else start,
                dtype=jnp.int32,
            )
        if _use_batch_native_double_cable_integrated_solver(
            kernel_block_solver,
            batch_size=batch_size,
        ):
            with benchmark_span(
                "kernel.dispatch_jax",
                mode="double",
                observer="vm_raster",
                variant=kernel_block_solver,
                factorized_vext=factorized_vext is not None,
                group_size=batch_size,
                time_chunk_steps=time_chunk_steps,
                chunk_steps=stop - start,
                chunk_index=chunk_index,
                chunk_count=len(chunk_ranges),
                observer_state_scope="chunk" if local_observer_chunks else "full",
                benchmark_observer_state_scope=observer_state_scope,
                resolved_observer_state_scope=resolved_observer_state_scope,
            ):
                Vi, Ve, gates, state, observer_state = _run_double_cable_batch_observer_integrated_scan(
                    backend=membrane_runtime.backend,
                    membrane=membrane_runtime.membrane,
                    has_driven_extracellular=has_driven_extracellular,
                    stateless_vm_only=stateless_vm_only,
                    double_cable_block_solver=kernel_block_solver,
                    tiled_thomas_block_b=kernel_tiled_thomas_block_b,
                    Vi0_mV=Vi,
                    Ve0_mV=Ve,
                    gates0=gates,
                    state0=state,
                    observer_state0=observer_state0,
                    raster_probe_indices=raster_probe_indices,
                    raster_probe_mask=raster_probe_mask,
                    raster_thresholds_mV=observers.thresholds_mV,
                    area_cm2=area_cm2,
                    Cm_abs=Cm_abs,
                    Cx_abs=Cx_abs,
                    Gx_abs=Gx_abs,
                    Gax_e=Gax_e,
                    Gax_i=Gax_i,
                    left_i=left_i,
                    right_i=right_i,
                    left_e=left_e,
                    right_e=right_e,
                    I_background=background,
                    intracellular_current_density_mid=iinj_chunk,
                    extracellular_potential_mid_mV=vext_chunk,
                    extracellular_potential_initial_previous_mV=previous,
                    row_indices=jnp.arange(batch_size, dtype=jnp.int32),
                    time_start_index=time_start_index,
                    dt_ms=jnp.asarray(grid.dt_ms, dtype=dtype_local),
                    extracellular_current_mid_A=current_chunk,
                    extracellular_current_initial_previous_A=previous_current_A,
                    extracellular_footprint_mV_per_A=factorized_footprint_mV_per_A,
                )
        else:
            with benchmark_span(
                "kernel.dispatch_jax",
                mode="double",
                observer="vm_raster",
                variant=kernel_block_solver,
                factorized_vext=factorized_vext is not None,
                group_size=batch_size,
                time_chunk_steps=time_chunk_steps,
                chunk_steps=stop - start,
                chunk_index=chunk_index,
                chunk_count=len(chunk_ranges),
                observer_state_scope="chunk" if local_observer_chunks else "full",
                benchmark_observer_state_scope=observer_state_scope,
                resolved_observer_state_scope=resolved_observer_state_scope,
            ):
                Vi, Ve, gates, state, observer_state = _run_double_cable_batch_observer_scan(
                    backend=membrane_runtime.backend,
                    membrane=membrane_runtime.membrane,
                    has_driven_extracellular=has_driven_extracellular,
                    stateless_vm_only=stateless_vm_only,
                    double_cable_block_solver=kernel_block_solver,
                    Vi0_mV=Vi,
                    Ve0_mV=Ve,
                    gates0=gates,
                    state0=state,
                    observer_state0=observer_state0,
                    raster_probe_indices=raster_probe_indices,
                    raster_probe_mask=raster_probe_mask,
                    raster_thresholds_mV=observers.thresholds_mV,
                    area_cm2=area_cm2,
                    Cm_abs=Cm_abs,
                    Cx_abs=Cx_abs,
                    Gx_abs=Gx_abs,
                    Gax_e=Gax_e,
                    Gax_i=Gax_i,
                    left_i=left_i,
                    right_i=right_i,
                    left_e=left_e,
                    right_e=right_e,
                    I_background=background,
                    intracellular_current_density_mid=iinj_chunk,
                    extracellular_potential_mid_mV=vext_chunk,
                    extracellular_potential_initial_previous_mV=previous,
                    row_indices=jnp.arange(batch_size, dtype=jnp.int32),
                    time_start_index=time_start_index,
                    dt_ms=jnp.asarray(grid.dt_ms, dtype=dtype_local),
                    extracellular_current_mid_A=current_chunk,
                    extracellular_current_initial_previous_A=previous_current_A,
                    extracellular_footprint_mV_per_A=factorized_footprint_mV_per_A,
                )
        with benchmark_span(
            "kernel.chunk_bookkeeping",
            mode="double",
            observer="vm_raster",
            variant=kernel_block_solver,
            output="observer_only",
            group_size=batch_size,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
        ):
            if factorized_vext is None:
                previous = cast(Any, vext_chunk)[:, -1]
            else:
                assert current_chunk is not None
                previous_current_A = (
                    current_chunk[-1]
                    if current_chunk.ndim == 1
                    else current_chunk[:, -1]
                )
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
            mode="double",
            variant=kernel_block_solver,
            time_chunk_steps=time_chunk_steps,
        )
    assert observer_state is not None
    return observer_state

def _initial_double_cable_batch_state(
    runtime: SolverRuntime,
    batch_size: int,
    Veinit_mV: float,
) -> tuple[Array, Array, Array, tuple[Array, ...]]:
    with benchmark_span(
        "kernel.prepare_state",
        mode="double",
        group_size=batch_size,
        nx=runtime.membrane.Nx,
    ):
        membrane_runtime = runtime.membrane
        dtype_local = membrane_runtime.dtype
        nx = membrane_runtime.Nx
        Ve = _cached_constant_batched_space_array(
            "double_cable_Veinit_mV",
            Veinit_mV,
            nx=nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        Vm = _as_cached_batched_space_array(
            "Vm0_mV",
            membrane_runtime.Vm0_mV,
            nx=nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        Vi = Vm if float(Veinit_mV) == 0.0 else Vm + Ve
        gates = _as_initial_double_cable_batch_state_array(
            "gates0",
            membrane_runtime.gates0,
            batch_size=batch_size,
            dtype_local=dtype_local,
        )
        state = tuple(
            _as_initial_double_cable_batch_state_array(
                f"state0[{index}]",
                values,
                batch_size=batch_size,
                dtype_local=dtype_local,
            )
            for index, values in enumerate(membrane_runtime.state0)
        )
        return Vi, Ve, gates, state

def _as_initial_double_cable_batch_state_array(
    name: str,
    values: Array,
    *,
    batch_size: int,
    dtype_local: jnp.dtype,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim >= 3 and int(arr.shape[0]) == int(batch_size):
        return arr
    if arr.ndim >= 3 and int(arr.shape[0]) == 1:
        return jnp.broadcast_to(arr, (int(batch_size), *arr.shape[1:]))
    if arr.ndim >= 3:
        raise ValueError(
            f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}."
        )
    return _cached_broadcast_batch_leading(
        arr,
        batch_size,
    )

__all__ = [
    "DoubleCableBatchKernel",
    "_prepare_double_cable_batch_arrays",
    "_resolve_double_cable_kernel_block_solver",
    "_resolve_double_cable_run_block_solver",
    "_use_batch_native_double_cable_integrated_solver",
]
