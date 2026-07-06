from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp

from axonscope.benchmarking import benchmark_span

from .batch_inputs import (
    FactorizedExtracellularPotentialBatch,
    SparseIntracellularCurrentDensityBatch,
    materialize_factorized_extracellular_potential_initial_previous,
    materialize_factorized_extracellular_potential_batch,
    materialize_sparse_intracellular_current_density_batch,
)
from .common import (
    Array,
    apply_diffusion_operator,
    solve_block_tridiagonal_2x2_pcr,
    solve_block_tridiagonal_2x2_pcr_soa,
    solve_block_tridiagonal_2x2_pcr_soa_batched,
    solve_block_tridiagonal_2x2_scalar,
)
from .kernels import _run_double_cable_vm_scan, _run_single_cable_vstim_vm_scan
from .observer_runtime import (
    VmRasterPlan,
    VmRasterState,
    combine_vm_raster_chunk_states,
    finalize_vm_raster_state,
    init_vm_raster_state,
    update_vm_raster_state_batch_from_tables,
    update_vm_raster_state_scalar_from_tables,
)
from axonscope.solvers.options import (
    BatchOptions,
    BatchRecording,
    resolve_double_cable_block_solver,
)
from .runtime import SolverRuntime


@dataclass(frozen=True)
class BatchKernelResult:
    """Raw batched solver-kernel output before packaging public simulations."""

    Vm: Array | None
    t: Array
    observations: dict[str, object] | None = None


_DOUBLE_CABLE_PCR_SOA_MAX_BATCH = 4096
_DOUBLE_CABLE_BATCH_NATIVE_PCR_SOA_MIN_BATCH = 16


def _resolve_double_cable_kernel_block_solver(
    solver: str,
    *,
    batch_size: int,
) -> str:
    if solver == "pcr_adaptive":
        return "pcr_soa" if batch_size <= _DOUBLE_CABLE_PCR_SOA_MAX_BATCH else "pcr"
    return solver


def _use_batch_native_double_cable_pcr_soa_solver(
    solver: str,
    *,
    batch_size: int,
) -> bool:
    return solver == "pcr_soa" and int(batch_size) >= _DOUBLE_CABLE_BATCH_NATIVE_PCR_SOA_MIN_BATCH


def _vm_raster_probe_tables_for_kernel(
    plan: VmRasterPlan,
    *,
    batch_size: int,
) -> tuple[Array, Array]:
    with benchmark_span(
        "kernel.prepare_observer_tables",
        observer="vm_raster",
        group_size=batch_size,
        raster_count=plan.raster_count,
        probe_count=plan.probe_count,
        row_aware=plan.row_aware,
    ):
        indices = jnp.asarray(plan.probe_indices)
        mask = jnp.asarray(plan.probe_mask)
        if indices.ndim == 2:
            shape = (int(batch_size),) + tuple(indices.shape)
            return (
                jnp.broadcast_to(indices[None, :, :], shape),
                jnp.broadcast_to(mask[None, :, :], shape),
            )
        return indices, mask


def _resolve_double_cable_run_block_solver(
    solver: str,
    *,
    platform: str,
) -> str:
    return resolve_double_cable_block_solver(solver, platform=platform)


def _double_cable_block_solve_fn(solver: str):
    if solver == "pcr_soa":
        return solve_block_tridiagonal_2x2_pcr_soa
    if solver == "pcr":
        return solve_block_tridiagonal_2x2_pcr
    if solver == "thomas":
        return solve_block_tridiagonal_2x2_scalar
    raise ValueError(f"Unsupported double-cable block solver: {solver!r}")


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
    ),
)
def _run_single_cable_vstim_batch_vm_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    lower: Array,
    diag: Array,
    upper: Array,
    dl: Array,
    d_static: Array,
    du: Array,
    Cm_uF_cm2: Array,
    I_background: Array,
    Vm0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    dt_ms: Array,
) -> Array:
    """Run the imposed-Vstim single-cable scan over a leading batch axis."""

    def one_batch(Iinj_mid: Array, vext_mid: Array) -> Array:
        vstim_forcing_mid = jax.vmap(
            lambda values: apply_diffusion_operator(values, lower, diag, upper)
        )(vext_mid)
        return _run_single_cable_vstim_vm_scan(
            backend=backend,
            membrane=membrane,
            has_driven_extracellular=has_driven_extracellular,
            stateless_vm_only=stateless_vm_only,
            lower=lower,
            diag=diag,
            upper=upper,
            dl=dl,
            d_static=d_static,
            du=du,
            Cm_uF_cm2=Cm_uF_cm2,
            I_background=I_background,
            Vm0_mV=Vm0_mV,
            gates0=gates0,
            state0=state0,
            intracellular_current_density_mid=Iinj_mid,
            extracellular_diffusion_forcing_mid=vstim_forcing_mid,
            dt_ms=dt_ms,
        )

    return jax.vmap(one_batch)(
        intracellular_current_density_mid,
        extracellular_potential_mid_mV,
    )


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
    ),
)
def _run_double_cable_batch_vm_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    Vi0_mV: Array,
    Ve0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    area_cm2: Array,
    Cm_abs: Array,
    Cx_abs: Array,
    Gx_abs: Array,
    Gax_e: Array,
    Gax_i: Array,
    left_i: Array,
    right_i: Array,
    left_e: Array,
    right_e: Array,
    I_background: Array,
    intracellular_current_density_mid: Array | None,
    extracellular_potential_mid_mV: Array,
    extracellular_potential_initial_previous_mV: Array,
    dt_ms: Array,
) -> Array:
    """Run the full double-cable scan over a leading batch axis."""

    def one_batch(
        Iinj_mid: Array | None,
        vext_mid: Array,
        vext_previous: Array,
    ) -> Array:
        if Iinj_mid is None:
            Iinj_mid = jnp.zeros_like(vext_mid)
        return _run_double_cable_vm_scan(
            backend=backend,
            membrane=membrane,
            has_driven_extracellular=has_driven_extracellular,
            stateless_vm_only=stateless_vm_only,
            Vi0_mV=Vi0_mV,
            Ve0_mV=Ve0_mV,
            gates0=gates0,
            state0=state0,
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
            I_background=I_background,
            intracellular_current_density_mid=Iinj_mid,
            extracellular_potential_mid_mV=vext_mid,
            extracellular_potential_initial_previous_mV=vext_previous,
            dt_ms=dt_ms,
        )

    iinj_in_axes = None if intracellular_current_density_mid is None else 0
    return jax.vmap(one_batch, in_axes=(iinj_in_axes, 0, 0))(
        intracellular_current_density_mid,
        extracellular_potential_mid_mV,
        extracellular_potential_initial_previous_mV,
    )


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "record_full",
    ),
)
def _run_single_cable_vstim_batch_stateful_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    record_full: bool,
    lower: Array,
    diag: Array,
    upper: Array,
    dl: Array,
    d_static: Array,
    du: Array,
    Cm_uF_cm2: Array,
    I_background: Array,
    Vm0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    record_indices: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], Array]:
    """Run one time chunk and return final batch state plus recorded Vm."""

    def one_batch(
        Vm0_row,
        gates0_row,
        state0_row,
        lower_row,
        diag_row,
        upper_row,
        dl_row,
        d_static_row,
        du_row,
        Cm_row,
        I_background_row,
        Iinj_mid,
        vext_mid,
    ):
        vstim_forcing_mid = jax.vmap(
            lambda values: apply_diffusion_operator(values, lower_row, diag_row, upper_row)
        )(vext_mid)

        def step(carry, step_inputs):
            Iinj, vstim_force = step_inputs
            Vm, gates, *extra = carry
            extra = tuple(extra)

            gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt_ms)
            if stateless_vm_only:
                linearization_gates = gates if has_driven_extracellular else gates_pred
                explicit_outward_current = I_background_row
                correction_current = jnp.zeros_like(Vm)
            else:
                Iion_pred = backend.currents(V_mV=Vm, gates=gates_pred)
                step_plan_pred = membrane.prepare_membrane_step(
                    V_mV=Vm,
                    gates_prev=gates,
                    gates_new=gates_pred,
                    state=extra,
                    dt=dt_ms,
                    I_ion=Iion_pred,
                    I_background=I_background_row,
                )
                linearization_gates = step_plan_pred.linearization_gates
                if has_driven_extracellular:
                    linearization_gates = gates
                explicit_outward_current = step_plan_pred.explicit_outward_current
                correction_current = step_plan_pred.correction_current

            Gm, GE = backend.membrane_conductance_terms(linearization_gates)
            d = d_static_row + (dt_ms / Cm_row) * Gm
            rhs = (
                Vm
                + dt_ms * vstim_force
                + (dt_ms / Cm_row)
                * (
                    GE
                    + Iinj
                    - explicit_outward_current
                    - correction_current
                )
            )
            Vm_new = jax.lax.linalg.tridiagonal_solve(dl_row, d, du_row, rhs[:, None])[:, 0]

            if stateless_vm_only:
                output = Vm_new if record_full else jnp.take(Vm_new, record_indices, axis=0)
                return (Vm_new, gates_pred, *extra), output

            gates_new = membrane.final_gate_update(
                gates_prev=gates,
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                dt=dt_ms,
                gates_predictor=gates_pred,
            )
            Iion_new = backend.currents(V_mV=Vm_new, gates=gates_new)
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state=extra,
                dt=dt_ms,
                I_ion=Iion_new,
                I_background=I_background_row,
            )
            state_new = membrane.finalize_membrane_step(
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=extra,
                step_plan=step_plan,
                dt=dt_ms,
            )
            output = Vm_new if record_full else jnp.take(Vm_new, record_indices, axis=0)
            return (Vm_new, gates_new, *state_new), output

        final_carry, trace = jax.lax.scan(
            step,
            (Vm0_row, gates0_row, *state0_row),
            (Iinj_mid, vstim_forcing_mid),
        )
        return final_carry[0], final_carry[1], tuple(final_carry[2:]), trace

    state_axes = tuple(0 for _ in state0)
    return jax.vmap(one_batch, in_axes=(0, 0, state_axes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
        Vm0_mV,
        gates0,
        state0,
        lower,
        diag,
        upper,
        dl,
        d_static,
        du,
        Cm_uF_cm2,
        I_background,
        intracellular_current_density_mid,
        extracellular_potential_mid_mV,
    )


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "record_full",
    ),
)
def _run_single_cable_factorized_vstim_batch_stateful_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    record_full: bool,
    lower: Array,
    diag: Array,
    upper: Array,
    dl: Array,
    d_static: Array,
    du: Array,
    Cm_uF_cm2: Array,
    I_background: Array,
    Vm0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    intracellular_current_density_mid: Array,
    extracellular_current_mid_A: Array,
    extracellular_forcing_footprint_mV_per_A: Array,
    record_indices: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], Array]:
    """Run one recorded-Vm chunk with factorized extracellular forcing."""

    def one_batch(
        Vm0_row,
        gates0_row,
        state0_row,
        lower_row,
        diag_row,
        upper_row,
        dl_row,
        d_static_row,
        du_row,
        Cm_row,
        I_background_row,
        Iinj_mid,
        current_mid_row_A,
        forcing_footprint_mV_per_A,
    ):
        def step(carry, step_inputs):
            Iinj, current_A = step_inputs
            Vm, gates, *extra = carry
            extra = tuple(extra)
            vstim_force = jnp.sum(
                current_A[:, None] * forcing_footprint_mV_per_A,
                axis=0,
            )

            gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt_ms)
            if stateless_vm_only:
                linearization_gates = gates if has_driven_extracellular else gates_pred
                explicit_outward_current = I_background_row
                correction_current = jnp.zeros_like(Vm)
            else:
                Iion_pred = backend.currents(V_mV=Vm, gates=gates_pred)
                step_plan_pred = membrane.prepare_membrane_step(
                    V_mV=Vm,
                    gates_prev=gates,
                    gates_new=gates_pred,
                    state=extra,
                    dt=dt_ms,
                    I_ion=Iion_pred,
                    I_background=I_background_row,
                )
                linearization_gates = step_plan_pred.linearization_gates
                if has_driven_extracellular:
                    linearization_gates = gates
                explicit_outward_current = step_plan_pred.explicit_outward_current
                correction_current = step_plan_pred.correction_current

            Gm, GE = backend.membrane_conductance_terms(linearization_gates)
            d = d_static_row + (dt_ms / Cm_row) * Gm
            rhs = (
                Vm
                + dt_ms * vstim_force
                + (dt_ms / Cm_row)
                * (
                    GE
                    + Iinj
                    - explicit_outward_current
                    - correction_current
                )
            )
            Vm_new = jax.lax.linalg.tridiagonal_solve(dl_row, d, du_row, rhs[:, None])[:, 0]

            if stateless_vm_only:
                output = Vm_new if record_full else jnp.take(Vm_new, record_indices, axis=0)
                return (Vm_new, gates_pred, *extra), output

            gates_new = membrane.final_gate_update(
                gates_prev=gates,
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                dt=dt_ms,
                gates_predictor=gates_pred,
            )
            Iion_new = backend.currents(V_mV=Vm_new, gates=gates_new)
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state=extra,
                dt=dt_ms,
                I_ion=Iion_new,
                I_background=I_background_row,
            )
            state_new = membrane.finalize_membrane_step(
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=extra,
                step_plan=step_plan,
                dt=dt_ms,
            )
            output = Vm_new if record_full else jnp.take(Vm_new, record_indices, axis=0)
            return (Vm_new, gates_new, *state_new), output

        final_carry, trace = jax.lax.scan(
            step,
            (Vm0_row, gates0_row, *state0_row),
            (Iinj_mid, jnp.swapaxes(current_mid_row_A, 0, 1)),
        )
        return final_carry[0], final_carry[1], tuple(final_carry[2:]), trace

    state_axes = tuple(0 for _ in state0)
    return jax.vmap(
        one_batch,
        in_axes=(0, 0, state_axes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )(
        Vm0_mV,
        gates0,
        state0,
        lower,
        diag,
        upper,
        dl,
        d_static,
        du,
        Cm_uF_cm2,
        I_background,
        intracellular_current_density_mid,
        extracellular_current_mid_A,
        extracellular_forcing_footprint_mV_per_A,
    )


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
    ),
)
def _run_single_cable_vstim_batch_observer_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    lower: Array,
    diag: Array,
    upper: Array,
    dl: Array,
    d_static: Array,
    du: Array,
    Cm_uF_cm2: Array,
    I_background: Array,
    Vm0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    observer_state0: Array,
    raster_probe_indices: Array,
    raster_probe_mask: Array,
    raster_thresholds_mV: Array,
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    time_start_index: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], VmRasterState]:
    """Run one time chunk while packing VmRaster words in the scan."""

    def one_batch(
        Vm0_row,
        gates0_row,
        state0_row,
        observer_state_row,
        raster_probe_indices_row,
        raster_probe_mask_row,
        lower_row,
        diag_row,
        upper_row,
        dl_row,
        d_static_row,
        du_row,
        Cm_row,
        I_background_row,
        Iinj_mid,
        vext_mid,
    ):
        vstim_forcing_mid = jax.vmap(
            lambda values: apply_diffusion_operator(values, lower_row, diag_row, upper_row)
        )(vext_mid)

        def step(carry, step_inputs):
            Iinj, vstim_force, local_step = step_inputs
            Vm, gates, observer_state, *extra = carry
            extra = tuple(extra)

            gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt_ms)
            if stateless_vm_only:
                linearization_gates = gates if has_driven_extracellular else gates_pred
                explicit_outward_current = I_background_row
                correction_current = jnp.zeros_like(Vm)
            else:
                Iion_pred = backend.currents(V_mV=Vm, gates=gates_pred)
                step_plan_pred = membrane.prepare_membrane_step(
                    V_mV=Vm,
                    gates_prev=gates,
                    gates_new=gates_pred,
                    state=extra,
                    dt=dt_ms,
                    I_ion=Iion_pred,
                    I_background=I_background_row,
                )
                linearization_gates = step_plan_pred.linearization_gates
                if has_driven_extracellular:
                    linearization_gates = gates
                explicit_outward_current = step_plan_pred.explicit_outward_current
                correction_current = step_plan_pred.correction_current

            Gm, GE = backend.membrane_conductance_terms(linearization_gates)
            d = d_static_row + (dt_ms / Cm_row) * Gm
            rhs = (
                Vm
                + dt_ms * vstim_force
                + (dt_ms / Cm_row)
                * (
                    GE
                    + Iinj
                    - explicit_outward_current
                    - correction_current
                )
            )
            Vm_new = jax.lax.linalg.tridiagonal_solve(dl_row, d, du_row, rhs[:, None])[:, 0]

            if stateless_vm_only:
                observer_state = update_vm_raster_state_scalar_from_tables(
                    observer_state,
                    vm_mV=Vm_new,
                    step_index=time_start_index + local_step,
                    probe_indices=raster_probe_indices_row,
                    probe_mask=raster_probe_mask_row,
                    thresholds_mV=raster_thresholds_mV,
                )
                return (Vm_new, gates_pred, observer_state, *extra), None

            gates_new = membrane.final_gate_update(
                gates_prev=gates,
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                dt=dt_ms,
                gates_predictor=gates_pred,
            )
            Iion_new = backend.currents(V_mV=Vm_new, gates=gates_new)
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state=extra,
                dt=dt_ms,
                I_ion=Iion_new,
                I_background=I_background_row,
            )
            state_new = membrane.finalize_membrane_step(
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=extra,
                step_plan=step_plan,
                dt=dt_ms,
            )
            observer_state = update_vm_raster_state_scalar_from_tables(
                observer_state,
                vm_mV=Vm_new,
                step_index=time_start_index + local_step,
                probe_indices=raster_probe_indices_row,
                probe_mask=raster_probe_mask_row,
                thresholds_mV=raster_thresholds_mV,
            )
            return (Vm_new, gates_new, observer_state, *state_new), None

        final_carry, _ = jax.lax.scan(
            step,
            (Vm0_row, gates0_row, observer_state_row, *state0_row),
            (
                Iinj_mid,
                vstim_forcing_mid,
                jnp.arange(Iinj_mid.shape[0], dtype=jnp.asarray(time_start_index).dtype),
            ),
        )
        return final_carry[0], final_carry[1], tuple(final_carry[3:]), final_carry[2]

    state_axes = tuple(0 for _ in state0)
    observer_axes = 0
    return jax.vmap(
        one_batch,
        in_axes=(0, 0, state_axes, observer_axes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )(
        Vm0_mV,
        gates0,
        state0,
        observer_state0,
        raster_probe_indices,
        raster_probe_mask,
        lower,
        diag,
        upper,
        dl,
        d_static,
        du,
        Cm_uF_cm2,
        I_background,
        intracellular_current_density_mid,
        extracellular_potential_mid_mV,
    )


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
    ),
)
def _run_single_cable_vstim_batch_sparse_observer_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    lower: Array,
    diag: Array,
    upper: Array,
    dl: Array,
    d_static: Array,
    du: Array,
    Cm_uF_cm2: Array,
    I_background: Array,
    Vm0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    observer_state0: Array,
    raster_probe_indices: Array,
    raster_probe_mask: Array,
    raster_thresholds_mV: Array,
    intracellular_current_density_values_mid: Array,
    intracellular_current_density_indices: Array,
    intracellular_current_density_mask: Array,
    extracellular_potential_mid_mV: Array,
    time_start_index: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], VmRasterState]:
    """Run one observer chunk with sparse point-clamp intracellular input."""

    def one_batch(
        Vm0_row,
        gates0_row,
        state0_row,
        observer_state_row,
        raster_probe_indices_row,
        raster_probe_mask_row,
        lower_row,
        diag_row,
        upper_row,
        dl_row,
        d_static_row,
        du_row,
        Cm_row,
        I_background_row,
        Iinj_values_mid,
        Iinj_indices,
        Iinj_mask,
        vext_mid,
    ):
        vstim_forcing_mid = jax.vmap(
            lambda values: apply_diffusion_operator(values, lower_row, diag_row, upper_row)
        )(vext_mid)
        safe_iinj_indices = jnp.where(Iinj_mask, Iinj_indices, 0)

        def step(carry, step_inputs):
            Iinj_values, vstim_force, local_step = step_inputs
            Vm, gates, observer_state, *extra = carry
            extra = tuple(extra)
            Iinj = jnp.zeros_like(Vm).at[safe_iinj_indices].add(
                jnp.where(Iinj_mask, Iinj_values, 0.0)
            )

            gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt_ms)
            if stateless_vm_only:
                linearization_gates = gates if has_driven_extracellular else gates_pred
                explicit_outward_current = I_background_row
                correction_current = jnp.zeros_like(Vm)
            else:
                Iion_pred = backend.currents(V_mV=Vm, gates=gates_pred)
                step_plan_pred = membrane.prepare_membrane_step(
                    V_mV=Vm,
                    gates_prev=gates,
                    gates_new=gates_pred,
                    state=extra,
                    dt=dt_ms,
                    I_ion=Iion_pred,
                    I_background=I_background_row,
                )
                linearization_gates = step_plan_pred.linearization_gates
                if has_driven_extracellular:
                    linearization_gates = gates
                explicit_outward_current = step_plan_pred.explicit_outward_current
                correction_current = step_plan_pred.correction_current

            Gm, GE = backend.membrane_conductance_terms(linearization_gates)
            d = d_static_row + (dt_ms / Cm_row) * Gm
            rhs = (
                Vm
                + dt_ms * vstim_force
                + (dt_ms / Cm_row)
                * (
                    GE
                    + Iinj
                    - explicit_outward_current
                    - correction_current
                )
            )
            Vm_new = jax.lax.linalg.tridiagonal_solve(dl_row, d, du_row, rhs[:, None])[:, 0]

            if stateless_vm_only:
                observer_state = update_vm_raster_state_scalar_from_tables(
                    observer_state,
                    vm_mV=Vm_new,
                    step_index=time_start_index + local_step,
                    probe_indices=raster_probe_indices_row,
                    probe_mask=raster_probe_mask_row,
                    thresholds_mV=raster_thresholds_mV,
                )
                return (Vm_new, gates_pred, observer_state, *extra), None

            gates_new = membrane.final_gate_update(
                gates_prev=gates,
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                dt=dt_ms,
                gates_predictor=gates_pred,
            )
            Iion_new = backend.currents(V_mV=Vm_new, gates=gates_new)
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state=extra,
                dt=dt_ms,
                I_ion=Iion_new,
                I_background=I_background_row,
            )
            state_new = membrane.finalize_membrane_step(
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=extra,
                step_plan=step_plan,
                dt=dt_ms,
            )
            observer_state = update_vm_raster_state_scalar_from_tables(
                observer_state,
                vm_mV=Vm_new,
                step_index=time_start_index + local_step,
                probe_indices=raster_probe_indices_row,
                probe_mask=raster_probe_mask_row,
                thresholds_mV=raster_thresholds_mV,
            )
            return (Vm_new, gates_new, observer_state, *state_new), None

        final_carry, _ = jax.lax.scan(
            step,
            (Vm0_row, gates0_row, observer_state_row, *state0_row),
            (
                Iinj_values_mid,
                vstim_forcing_mid,
                jnp.arange(
                    Iinj_values_mid.shape[0],
                    dtype=jnp.asarray(time_start_index).dtype,
                ),
            ),
        )
        return final_carry[0], final_carry[1], tuple(final_carry[3:]), final_carry[2]

    state_axes = tuple(0 for _ in state0)
    observer_axes = 0
    return jax.vmap(
        one_batch,
        in_axes=(
            0,
            0,
            state_axes,
            observer_axes,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
    )(
        Vm0_mV,
        gates0,
        state0,
        observer_state0,
        raster_probe_indices,
        raster_probe_mask,
        lower,
        diag,
        upper,
        dl,
        d_static,
        du,
        Cm_uF_cm2,
        I_background,
        intracellular_current_density_values_mid,
        intracellular_current_density_indices,
        intracellular_current_density_mask,
        extracellular_potential_mid_mV,
    )


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
    ),
)
def _run_single_cable_factorized_vstim_batch_sparse_observer_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    dl: Array,
    d_static: Array,
    du: Array,
    Cm_uF_cm2: Array,
    I_background: Array,
    Vm0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    observer_state0: Array,
    raster_probe_indices: Array,
    raster_probe_mask: Array,
    raster_thresholds_mV: Array,
    intracellular_current_density_values_mid: Array,
    intracellular_current_density_indices: Array,
    intracellular_current_density_mask: Array,
    extracellular_current_mid_A: Array,
    extracellular_forcing_footprint_mV_per_A: Array,
    time_start_index: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], VmRasterState]:
    """Run one observer chunk with sparse Iinj and pre-lowered factorized Vstim."""

    def one_batch(
        Vm0_row,
        gates0_row,
        state0_row,
        observer_state_row,
        raster_probe_indices_row,
        raster_probe_mask_row,
        dl_row,
        d_static_row,
        du_row,
        Cm_row,
        I_background_row,
        Iinj_values_mid,
        Iinj_indices,
        Iinj_mask,
        current_mid_row_A,
        forcing_footprint_mV_per_A,
    ):
        safe_iinj_indices = jnp.where(Iinj_mask, Iinj_indices, 0)

        def step(carry, step_inputs):
            Iinj_values, current_A, local_step = step_inputs
            Vm, gates, observer_state, *extra = carry
            extra = tuple(extra)
            Iinj = jnp.zeros_like(Vm).at[safe_iinj_indices].add(
                jnp.where(Iinj_mask, Iinj_values, 0.0)
            )
            vstim_force = jnp.sum(
                current_A[:, None] * forcing_footprint_mV_per_A,
                axis=0,
            )

            gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt_ms)
            if stateless_vm_only:
                linearization_gates = gates if has_driven_extracellular else gates_pred
                explicit_outward_current = I_background_row
                correction_current = jnp.zeros_like(Vm)
            else:
                Iion_pred = backend.currents(V_mV=Vm, gates=gates_pred)
                step_plan_pred = membrane.prepare_membrane_step(
                    V_mV=Vm,
                    gates_prev=gates,
                    gates_new=gates_pred,
                    state=extra,
                    dt=dt_ms,
                    I_ion=Iion_pred,
                    I_background=I_background_row,
                )
                linearization_gates = step_plan_pred.linearization_gates
                if has_driven_extracellular:
                    linearization_gates = gates
                explicit_outward_current = step_plan_pred.explicit_outward_current
                correction_current = step_plan_pred.correction_current

            Gm, GE = backend.membrane_conductance_terms(linearization_gates)
            d = d_static_row + (dt_ms / Cm_row) * Gm
            rhs = (
                Vm
                + dt_ms * vstim_force
                + (dt_ms / Cm_row)
                * (
                    GE
                    + Iinj
                    - explicit_outward_current
                    - correction_current
                )
            )
            Vm_new = jax.lax.linalg.tridiagonal_solve(dl_row, d, du_row, rhs[:, None])[:, 0]

            if stateless_vm_only:
                observer_state = update_vm_raster_state_scalar_from_tables(
                    observer_state,
                    vm_mV=Vm_new,
                    step_index=time_start_index + local_step,
                    probe_indices=raster_probe_indices_row,
                    probe_mask=raster_probe_mask_row,
                    thresholds_mV=raster_thresholds_mV,
                )
                return (Vm_new, gates_pred, observer_state, *extra), None

            gates_new = membrane.final_gate_update(
                gates_prev=gates,
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                dt=dt_ms,
                gates_predictor=gates_pred,
            )
            Iion_new = backend.currents(V_mV=Vm_new, gates=gates_new)
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state=extra,
                dt=dt_ms,
                I_ion=Iion_new,
                I_background=I_background_row,
            )
            state_new = membrane.finalize_membrane_step(
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=extra,
                step_plan=step_plan,
                dt=dt_ms,
            )
            observer_state = update_vm_raster_state_scalar_from_tables(
                observer_state,
                vm_mV=Vm_new,
                step_index=time_start_index + local_step,
                probe_indices=raster_probe_indices_row,
                probe_mask=raster_probe_mask_row,
                thresholds_mV=raster_thresholds_mV,
            )
            return (Vm_new, gates_new, observer_state, *state_new), None

        final_carry, _ = jax.lax.scan(
            step,
            (Vm0_row, gates0_row, observer_state_row, *state0_row),
            (
                Iinj_values_mid,
                jnp.swapaxes(current_mid_row_A, 0, 1),
                jnp.arange(
                    Iinj_values_mid.shape[0],
                    dtype=jnp.asarray(time_start_index).dtype,
                ),
            ),
        )
        return final_carry[0], final_carry[1], tuple(final_carry[3:]), final_carry[2]

    state_axes = tuple(0 for _ in state0)
    observer_axes = 0
    return jax.vmap(
        one_batch,
        in_axes=(
            0,
            0,
            state_axes,
            observer_axes,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
    )(
        Vm0_mV,
        gates0,
        state0,
        observer_state0,
        raster_probe_indices,
        raster_probe_mask,
        dl,
        d_static,
        du,
        Cm_uF_cm2,
        I_background,
        intracellular_current_density_values_mid,
        intracellular_current_density_indices,
        intracellular_current_density_mask,
        extracellular_current_mid_A,
        extracellular_forcing_footprint_mV_per_A,
    )


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "stateless_vm_only",
    ),
)
def _run_single_cable_zero_vstim_batch_sparse_observer_scan(
    *,
    backend,
    membrane,
    stateless_vm_only: bool,
    lower: Array,
    diag: Array,
    upper: Array,
    dl: Array,
    d_static: Array,
    du: Array,
    Cm_uF_cm2: Array,
    I_background: Array,
    Vm0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    observer_state0: Array,
    raster_probe_indices: Array,
    raster_probe_mask: Array,
    raster_thresholds_mV: Array,
    intracellular_current_density_values_mid: Array,
    intracellular_current_density_indices: Array,
    intracellular_current_density_mask: Array,
    time_start_index: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], VmRasterState]:
    """Run one observer chunk with sparse point clamps and zero extracellular drive."""

    def one_batch(
        Vm0_row,
        gates0_row,
        state0_row,
        observer_state_row,
        raster_probe_indices_row,
        raster_probe_mask_row,
        lower_row,
        diag_row,
        upper_row,
        dl_row,
        d_static_row,
        du_row,
        Cm_row,
        I_background_row,
        Iinj_values_mid,
        Iinj_indices,
        Iinj_mask,
    ):
        safe_iinj_indices = jnp.where(Iinj_mask, Iinj_indices, 0)

        def step(carry, step_inputs):
            Iinj_values, local_step = step_inputs
            Vm, gates, observer_state, *extra = carry
            extra = tuple(extra)
            Iinj = jnp.zeros_like(Vm).at[safe_iinj_indices].add(
                jnp.where(Iinj_mask, Iinj_values, 0.0)
            )

            gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt_ms)
            if stateless_vm_only:
                linearization_gates = gates_pred
                explicit_outward_current = I_background_row
                correction_current = jnp.zeros_like(Vm)
            else:
                Iion_pred = backend.currents(V_mV=Vm, gates=gates_pred)
                step_plan_pred = membrane.prepare_membrane_step(
                    V_mV=Vm,
                    gates_prev=gates,
                    gates_new=gates_pred,
                    state=extra,
                    dt=dt_ms,
                    I_ion=Iion_pred,
                    I_background=I_background_row,
                )
                linearization_gates = step_plan_pred.linearization_gates
                explicit_outward_current = step_plan_pred.explicit_outward_current
                correction_current = step_plan_pred.correction_current

            Gm, GE = backend.membrane_conductance_terms(linearization_gates)
            d = d_static_row + (dt_ms / Cm_row) * Gm
            rhs = (
                Vm
                + (dt_ms / Cm_row)
                * (
                    GE
                    + Iinj
                    - explicit_outward_current
                    - correction_current
                )
            )
            Vm_new = jax.lax.linalg.tridiagonal_solve(dl_row, d, du_row, rhs[:, None])[:, 0]

            if stateless_vm_only:
                observer_state = update_vm_raster_state_scalar_from_tables(
                    observer_state,
                    vm_mV=Vm_new,
                    step_index=time_start_index + local_step,
                    probe_indices=raster_probe_indices_row,
                    probe_mask=raster_probe_mask_row,
                    thresholds_mV=raster_thresholds_mV,
                )
                return (Vm_new, gates_pred, observer_state, *extra), None

            gates_new = membrane.final_gate_update(
                gates_prev=gates,
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                dt=dt_ms,
                gates_predictor=gates_pred,
            )
            Iion_new = backend.currents(V_mV=Vm_new, gates=gates_new)
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state=extra,
                dt=dt_ms,
                I_ion=Iion_new,
                I_background=I_background_row,
            )
            state_new = membrane.finalize_membrane_step(
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=extra,
                step_plan=step_plan,
                dt=dt_ms,
            )
            observer_state = update_vm_raster_state_scalar_from_tables(
                observer_state,
                vm_mV=Vm_new,
                step_index=time_start_index + local_step,
                probe_indices=raster_probe_indices_row,
                probe_mask=raster_probe_mask_row,
                thresholds_mV=raster_thresholds_mV,
            )
            return (Vm_new, gates_new, observer_state, *state_new), None

        final_carry, _ = jax.lax.scan(
            step,
            (Vm0_row, gates0_row, observer_state_row, *state0_row),
            (
                Iinj_values_mid,
                jnp.arange(
                    Iinj_values_mid.shape[0],
                    dtype=jnp.asarray(time_start_index).dtype,
                ),
            ),
        )
        return final_carry[0], final_carry[1], tuple(final_carry[3:]), final_carry[2]

    state_axes = tuple(0 for _ in state0)
    observer_axes = 0
    return jax.vmap(
        one_batch,
        in_axes=(
            0,
            0,
            state_axes,
            observer_axes,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
    )(
        Vm0_mV,
        gates0,
        state0,
        observer_state0,
        raster_probe_indices,
        raster_probe_mask,
        lower,
        diag,
        upper,
        dl,
        d_static,
        du,
        Cm_uF_cm2,
        I_background,
        intracellular_current_density_values_mid,
        intracellular_current_density_indices,
        intracellular_current_density_mask,
    )


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "record_full",
        "double_cable_block_solver",
    ),
)
def _run_double_cable_batch_stateful_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    record_full: bool,
    double_cable_block_solver: str,
    Vi0_mV: Array,
    Ve0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    area_cm2: Array,
    Cm_abs: Array,
    Cx_abs: Array,
    Gx_abs: Array,
    Gax_e: Array,
    Gax_i: Array,
    left_i: Array,
    right_i: Array,
    left_e: Array,
    right_e: Array,
    I_background: Array,
    intracellular_current_density_mid: Array | None,
    extracellular_potential_mid_mV: Array,
    extracellular_potential_initial_previous_mV: Array,
    row_indices: Array,
    record_indices: Array,
    dt_ms: Array,
) -> tuple[Array, Array, Array, tuple[Array, ...], Array]:
    """Run one double-cable time chunk and return final batch state."""

    if intracellular_current_density_mid is None:
        intracellular_current_abs_mid = None
    else:
        area_for_iinj = (
            area_cm2[None, None, :]
            if jnp.asarray(area_cm2).ndim == 1
            else area_cm2[:, None, :]
        )
        intracellular_current_abs_mid = intracellular_current_density_mid * area_for_iinj

    def one_batch(
        Vi0_row,
        Ve0_row,
        gates0_row,
        state0_row,
        area_row,
        Cm_abs_row,
        Cx_abs_row,
        Gx_abs_row,
        Gax_e_row,
        Gax_i_row,
        left_i_row,
        right_i_row,
        left_e_row,
        right_e_row,
        I_background_row,
        Iinj_abs_mid,
        vext_mid,
        vext_prev0,
        row_index,
    ):
        cm_over_dt = Cm_abs_row / dt_ms
        cx_over_dt = Cx_abs_row / dt_ms
        off_i = -Gax_i_row
        off_e = -Gax_e_row
        vext_previous_mV = jnp.concatenate([vext_prev0[None, :], vext_mid[:-1]], axis=0)
        extracellular_rhs_drive = (
            (cx_over_dt + Gx_abs_row)[None, :] * vext_mid
            - cx_over_dt[None, :] * vext_previous_mV
        )

        def solve_vi_vperi(
            Vi: Array,
            Ve: Array,
            gates_new: Array,
            Iinj_abs: Array,
            I_outward_den: Array,
            I_corr_den: Array,
            extracellular_drive_abs: Array,
        ) -> tuple[Array, Array]:
            row_terms = getattr(backend, "membrane_conductance_terms_for_row", None)
            if callable(row_terms):
                Gm_den, GE_den = row_terms(row_index, gates_new)
            else:
                Gm_den, GE_den = backend.membrane_conductance_terms(gates_new)
            Gm_abs = Gm_den * area_row
            GE_abs = GE_den * area_row

            I_outward_abs = I_outward_den * area_row
            I_corr_abs = I_corr_den * area_row
            Vm = Vi - Ve

            a00 = cm_over_dt + Gm_abs + left_i_row + right_i_row
            a01 = -(cm_over_dt + Gm_abs)
            rhs0 = (
                cm_over_dt * Vm
                + GE_abs
                + Iinj_abs
                - I_outward_abs
                - I_corr_abs
            )

            a10 = a01
            a11 = (
                cm_over_dt
                + Gm_abs
                + cx_over_dt
                + Gx_abs_row
                + left_e_row
                + right_e_row
            )
            rhs1 = (
                -cm_over_dt * Vm
                - GE_abs
                + cx_over_dt * Ve
                + extracellular_drive_abs
                + I_outward_abs
                + I_corr_abs
            )

            solve_block = _double_cable_block_solve_fn(double_cable_block_solver)
            return solve_block(
                a00,
                a01,
                a10,
                a11,
                off_i,
                off_e,
                rhs0,
                rhs1,
            )

        def step(
            carry,
            step_inputs,
        ):
            if Iinj_abs_mid is None:
                Iinj_abs = jnp.zeros_like(area_row)
                extracellular_drive_abs = step_inputs
            else:
                Iinj_abs, extracellular_drive_abs = step_inputs
            Vi, Ve, gates, *extra = carry
            extra = tuple(extra)
            Vm = Vi - Ve

            row_gate_update = getattr(backend, "cn_gate_update_for_row", None)
            if callable(row_gate_update):
                gates_pred = row_gate_update(
                    row_index,
                    g_prev=gates,
                    V_mV=Vm,
                    dt=dt_ms,
                )
            else:
                gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt_ms)

            if stateless_vm_only:
                linearization_gates = gates if has_driven_extracellular else gates_pred
                explicit_outward_current = I_background_row
                correction_current = jnp.zeros_like(Vm)
            else:
                row_currents = getattr(backend, "currents_for_row", None)
                Iion_pred = (
                    row_currents(row_index, V_mV=Vm, gates=gates_pred)
                    if callable(row_currents)
                    else backend.currents(V_mV=Vm, gates=gates_pred)
                )
                step_plan_pred = membrane.prepare_membrane_step(
                    V_mV=Vm,
                    gates_prev=gates,
                    gates_new=gates_pred,
                    state=extra,
                    dt=dt_ms,
                    I_ion=Iion_pred,
                    I_background=I_background_row,
                )
                linearization_gates = step_plan_pred.linearization_gates
                if has_driven_extracellular:
                    linearization_gates = gates
                explicit_outward_current = step_plan_pred.explicit_outward_current
                correction_current = step_plan_pred.correction_current

            Vi_new, Ve_new = solve_vi_vperi(
                Vi=Vi,
                Ve=Ve,
                gates_new=linearization_gates,
                Iinj_abs=Iinj_abs,
                I_outward_den=explicit_outward_current,
                I_corr_den=correction_current,
                extracellular_drive_abs=extracellular_drive_abs,
            )
            Vm_new = Vi_new - Ve_new

            if stateless_vm_only:
                output = Vm_new if record_full else jnp.take(Vm_new, record_indices, axis=0)
                return (Vi_new, Ve_new, gates_pred, *extra), output

            gates_new = membrane.final_gate_update(
                gates_prev=gates,
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                dt=dt_ms,
                gates_predictor=gates_pred,
            )
            row_currents = getattr(backend, "currents_for_row", None)
            Iion_new = (
                row_currents(row_index, V_mV=Vm_new, gates=gates_new)
                if callable(row_currents)
                else backend.currents(V_mV=Vm_new, gates=gates_new)
            )
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state=extra,
                dt=dt_ms,
                I_ion=Iion_new,
                I_background=I_background_row,
            )
            state_new = membrane.finalize_membrane_step(
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=extra,
                step_plan=step_plan,
                dt=dt_ms,
            )
            output = Vm_new if record_full else jnp.take(Vm_new, record_indices, axis=0)
            return (Vi_new, Ve_new, gates_new, *state_new), output

        scan_inputs = (
            extracellular_rhs_drive
            if Iinj_abs_mid is None
            else (Iinj_abs_mid, extracellular_rhs_drive)
        )
        final_carry, trace = jax.lax.scan(
            step,
            (Vi0_row, Ve0_row, gates0_row, *state0_row),
            scan_inputs,
        )
        return final_carry[0], final_carry[1], final_carry[2], tuple(final_carry[3:]), trace

    state_axes = tuple(0 for _ in state0)
    space_in_axes = None if jnp.asarray(area_cm2).ndim == 1 else 0
    edge_in_axes = None if jnp.asarray(Gax_i).ndim == 1 else 0
    background_in_axes = None if jnp.asarray(I_background).ndim <= 1 else 0
    iinj_in_axes = None if intracellular_current_abs_mid is None else 0
    return jax.vmap(
        one_batch,
        in_axes=(
            0,
            0,
            0,
            state_axes,
            space_in_axes,
            space_in_axes,
            space_in_axes,
            space_in_axes,
            edge_in_axes,
            edge_in_axes,
            space_in_axes,
            space_in_axes,
            space_in_axes,
            space_in_axes,
            background_in_axes,
            iinj_in_axes,
            0,
            0,
            0,
        ),
    )(
        Vi0_mV,
        Ve0_mV,
        gates0,
        state0,
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
        I_background,
        intracellular_current_abs_mid,
        extracellular_potential_mid_mV,
        extracellular_potential_initial_previous_mV,
        row_indices,
    )


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "record_full",
        "double_cable_block_solver",
    ),
)
def _run_double_cable_batch_stateful_pcr_soa_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    record_full: bool,
    double_cable_block_solver: str,
    Vi0_mV: Array,
    Ve0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    area_cm2: Array,
    Cm_abs: Array,
    Cx_abs: Array,
    Gx_abs: Array,
    Gax_e: Array,
    Gax_i: Array,
    left_i: Array,
    right_i: Array,
    left_e: Array,
    right_e: Array,
    I_background: Array,
    intracellular_current_density_mid: Array | None,
    extracellular_potential_mid_mV: Array,
    extracellular_potential_initial_previous_mV: Array,
    row_indices: Array,
    record_indices: Array,
    dt_ms: Array,
) -> tuple[Array, Array, Array, tuple[Array, ...], Array]:
    """Run one time chunk using a batch-native double-cable block solver."""

    batch_size = int(Vi0_mV.shape[0])
    nx = int(Vi0_mV.shape[1])

    def batch_space(values: Array) -> Array:
        arr = jnp.asarray(values)
        if arr.ndim == 0:
            return jnp.broadcast_to(arr, (batch_size, nx))
        if arr.ndim == 1:
            return jnp.broadcast_to(arr[None, :], (batch_size, nx))
        return arr

    area_batch = batch_space(area_cm2)
    cm_over_dt = batch_space(Cm_abs) / dt_ms
    cx_over_dt = batch_space(Cx_abs) / dt_ms
    Gx_abs_batch = batch_space(Gx_abs)
    left_i_batch = batch_space(left_i)
    right_i_batch = batch_space(right_i)
    left_e_batch = batch_space(left_e)
    right_e_batch = batch_space(right_e)
    background_batch = batch_space(I_background)

    off_i = -jnp.asarray(Gax_i)
    off_e = -jnp.asarray(Gax_e)
    vext_previous_mV = jnp.concatenate(
        [
            extracellular_potential_initial_previous_mV[:, None, :],
            extracellular_potential_mid_mV[:, :-1, :],
        ],
        axis=1,
    )
    extracellular_rhs_drive = (
        (cx_over_dt + Gx_abs_batch)[:, None, :] * extracellular_potential_mid_mV
        - cx_over_dt[:, None, :] * vext_previous_mV
    )

    if intracellular_current_density_mid is None:
        intracellular_current_abs_mid = None
    else:
        intracellular_current_abs_mid = intracellular_current_density_mid * area_batch[:, None, :]

    state_axes = tuple(0 for _ in state0)

    def batch_gate_update(gates: Array, Vm: Array) -> Array:
        row_gate_update = getattr(backend, "cn_gate_update_for_row", None)
        if callable(row_gate_update):
            return jax.vmap(
                lambda row_index, gates_row, vm_row: row_gate_update(
                    row_index,
                    g_prev=gates_row,
                    V_mV=vm_row,
                    dt=dt_ms,
                )
            )(row_indices, gates, Vm)
        return jax.vmap(
            lambda gates_row, vm_row: backend.cn_gate_update(
                g_prev=gates_row,
                V_mV=vm_row,
                dt=dt_ms,
            )
        )(gates, Vm)

    def batch_currents(Vm: Array, gates: Array) -> Array:
        row_currents = getattr(backend, "currents_for_row", None)
        if callable(row_currents):
            return jax.vmap(
                lambda row_index, vm_row, gates_row: row_currents(
                    row_index,
                    V_mV=vm_row,
                    gates=gates_row,
                )
            )(row_indices, Vm, gates)
        return jax.vmap(
            lambda vm_row, gates_row: backend.currents(V_mV=vm_row, gates=gates_row)
        )(Vm, gates)

    def batch_membrane_conductance_terms(gates: Array) -> tuple[Array, Array]:
        row_terms = getattr(backend, "membrane_conductance_terms_for_row", None)
        if callable(row_terms):
            return jax.vmap(
                lambda row_index, gates_row: row_terms(row_index, gates_row)
            )(row_indices, gates)
        return jax.vmap(backend.membrane_conductance_terms)(gates)

    def batch_prepare_membrane_step(
        Vm: Array,
        gates_prev: Array,
        gates_new: Array,
        state: tuple[Array, ...],
        I_ion: Array,
        I_background_row: Array,
    ):
        return jax.vmap(
            lambda vm_row, gates_prev_row, gates_new_row, state_row, iion_row, ibg_row: (
                membrane.prepare_membrane_step(
                    V_mV=vm_row,
                    gates_prev=gates_prev_row,
                    gates_new=gates_new_row,
                    state=state_row,
                    dt=dt_ms,
                    I_ion=iion_row,
                    I_background=ibg_row,
                )
            ),
            in_axes=(0, 0, 0, state_axes, 0, 0),
        )(Vm, gates_prev, gates_new, state, I_ion, I_background_row)

    def batch_final_gate_update(
        gates_prev: Array,
        Vm_prev: Array,
        Vm_new: Array,
        gates_predictor: Array,
    ) -> Array:
        return jax.vmap(
            lambda gates_prev_row, vm_prev_row, vm_new_row, gates_predictor_row: (
                membrane.final_gate_update(
                    gates_prev=gates_prev_row,
                    V_mV_prev=vm_prev_row,
                    V_mV_new=vm_new_row,
                    dt=dt_ms,
                    gates_predictor=gates_predictor_row,
                )
            )
        )(gates_prev, Vm_prev, Vm_new, gates_predictor)

    def batch_finalize_membrane_step(
        Vm_prev: Array,
        Vm_new: Array,
        gates_prev: Array,
        gates_new: Array,
        state_prev: tuple[Array, ...],
        step_plan,
    ) -> tuple[Array, ...]:
        return jax.vmap(
            lambda vm_prev_row, vm_new_row, gates_prev_row, gates_new_row, state_row, step_plan_row: (
                membrane.finalize_membrane_step(
                    V_mV_prev=vm_prev_row,
                    V_mV_new=vm_new_row,
                    gates_prev=gates_prev_row,
                    gates_new=gates_new_row,
                    state_prev=state_row,
                    step_plan=step_plan_row,
                    dt=dt_ms,
                )
            ),
            in_axes=(0, 0, 0, 0, state_axes, 0),
        )(Vm_prev, Vm_new, gates_prev, gates_new, state_prev, step_plan)

    def solve_vi_vperi(
        Vi: Array,
        Ve: Array,
        gates_new: Array,
        Iinj_abs: Array,
        I_outward_den: Array,
        I_corr_den: Array,
        extracellular_drive_abs: Array,
    ) -> tuple[Array, Array]:
        Gm_den, GE_den = batch_membrane_conductance_terms(gates_new)
        Gm_abs = Gm_den * area_batch
        GE_abs = GE_den * area_batch

        I_outward_abs = I_outward_den * area_batch
        I_corr_abs = I_corr_den * area_batch
        Vm = Vi - Ve

        a00 = cm_over_dt + Gm_abs + left_i_batch + right_i_batch
        a01 = -(cm_over_dt + Gm_abs)
        rhs0 = cm_over_dt * Vm + GE_abs + Iinj_abs - I_outward_abs - I_corr_abs

        a10 = a01
        a11 = (
            cm_over_dt
            + Gm_abs
            + cx_over_dt
            + Gx_abs_batch
            + left_e_batch
            + right_e_batch
        )
        rhs1 = (
            -cm_over_dt * Vm
            - GE_abs
            + cx_over_dt * Ve
            + extracellular_drive_abs
            + I_outward_abs
            + I_corr_abs
        )

        if double_cable_block_solver == "pcr_soa":
            return solve_block_tridiagonal_2x2_pcr_soa_batched(
                a00,
                a01,
                a10,
                a11,
                off_i,
                off_e,
                rhs0,
                rhs1,
            )
        raise ValueError(
            f"Unsupported batch-native double-cable block solver: {double_cable_block_solver!r}"
        )

    def step(carry, step_inputs):
        if intracellular_current_abs_mid is None:
            Iinj_abs = jnp.zeros_like(area_batch)
            extracellular_drive_abs = step_inputs
        else:
            Iinj_abs, extracellular_drive_abs = step_inputs
        Vi, Ve, gates, *extra = carry
        extra = tuple(extra)
        Vm = Vi - Ve

        gates_pred = batch_gate_update(gates, Vm)
        if stateless_vm_only:
            linearization_gates = gates if has_driven_extracellular else gates_pred
            explicit_outward_current = background_batch
            correction_current = jnp.zeros_like(Vm)
        else:
            Iion_pred = batch_currents(Vm, gates_pred)
            step_plan_pred = batch_prepare_membrane_step(
                Vm,
                gates,
                gates_pred,
                extra,
                Iion_pred,
                background_batch,
            )
            linearization_gates = step_plan_pred.linearization_gates
            if has_driven_extracellular:
                linearization_gates = gates
            explicit_outward_current = step_plan_pred.explicit_outward_current
            correction_current = step_plan_pred.correction_current

        Vi_new, Ve_new = solve_vi_vperi(
            Vi=Vi,
            Ve=Ve,
            gates_new=linearization_gates,
            Iinj_abs=Iinj_abs,
            I_outward_den=explicit_outward_current,
            I_corr_den=correction_current,
            extracellular_drive_abs=extracellular_drive_abs,
        )
        Vm_new = Vi_new - Ve_new

        output = Vm_new if record_full else jnp.take(Vm_new, record_indices, axis=1)
        if stateless_vm_only:
            return (Vi_new, Ve_new, gates_pred, *extra), output

        gates_new = batch_final_gate_update(gates, Vm, Vm_new, gates_pred)
        Iion_new = batch_currents(Vm_new, gates_new)
        step_plan = batch_prepare_membrane_step(
            Vm_new,
            gates,
            gates_new,
            extra,
            Iion_new,
            background_batch,
        )
        state_new = batch_finalize_membrane_step(
            Vm,
            Vm_new,
            gates,
            gates_new,
            extra,
            step_plan,
        )
        return (Vi_new, Ve_new, gates_new, *state_new), output

    scan_inputs = (
        jnp.swapaxes(extracellular_rhs_drive, 0, 1)
        if intracellular_current_abs_mid is None
        else (
            jnp.swapaxes(intracellular_current_abs_mid, 0, 1),
            jnp.swapaxes(extracellular_rhs_drive, 0, 1),
        )
    )
    final_carry, trace = jax.lax.scan(
        step,
        (Vi0_mV, Ve0_mV, gates0, *state0),
        scan_inputs,
    )
    return (
        final_carry[0],
        final_carry[1],
        final_carry[2],
        tuple(final_carry[3:]),
        jnp.swapaxes(trace, 0, 1),
    )


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "double_cable_block_solver",
    ),
)
def _run_double_cable_batch_observer_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    double_cable_block_solver: str,
    Vi0_mV: Array,
    Ve0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    observer_state0: VmRasterState,
    raster_probe_indices: Array,
    raster_probe_mask: Array,
    raster_thresholds_mV: Array,
    area_cm2: Array,
    Cm_abs: Array,
    Cx_abs: Array,
    Gx_abs: Array,
    Gax_e: Array,
    Gax_i: Array,
    left_i: Array,
    right_i: Array,
    left_e: Array,
    right_e: Array,
    I_background: Array,
    intracellular_current_density_mid: Array | None,
    extracellular_potential_mid_mV: Array | None,
    extracellular_potential_initial_previous_mV: Array | None,
    row_indices: Array,
    time_start_index: Array,
    dt_ms: Array,
    extracellular_current_mid_A: Array | None = None,
    extracellular_current_initial_previous_A: Array | None = None,
    extracellular_footprint_mV_per_A: Array | None = None,
) -> tuple[Array, Array, Array, tuple[Array, ...], VmRasterState]:
    """Run one observer-only chunk with the row-wise double-cable solver."""

    use_factorized_vext = extracellular_footprint_mV_per_A is not None
    if intracellular_current_density_mid is None:
        intracellular_current_abs_mid = None
    else:
        area_for_iinj = (
            area_cm2[None, None, :]
            if jnp.asarray(area_cm2).ndim == 1
            else area_cm2[:, None, :]
        )
        intracellular_current_abs_mid = intracellular_current_density_mid * area_for_iinj

    if use_factorized_vext:
        if extracellular_current_mid_A is None:
            raise ValueError("extracellular_current_mid_A is required.")
        if extracellular_current_initial_previous_A is None:
            raise ValueError("extracellular_current_initial_previous_A is required.")
        current_mid_A = jnp.asarray(extracellular_current_mid_A)
        current_initial_previous_A = jnp.asarray(extracellular_current_initial_previous_A)
        if current_mid_A.ndim == 1:
            current_previous_A = jnp.concatenate(
                [
                    current_initial_previous_A.reshape((1,)),
                    current_mid_A[:-1],
                ],
                axis=0,
            )
        elif current_mid_A.ndim == 2:
            batch_size = int(current_mid_A.shape[0])
            initial_previous = (
                jnp.broadcast_to(current_initial_previous_A, (batch_size,))
                if current_initial_previous_A.ndim == 0
                else current_initial_previous_A
            )
            current_previous_A = jnp.concatenate(
                [
                    initial_previous.reshape((batch_size, 1)),
                    current_mid_A[:, :-1],
                ],
                axis=1,
            )
        else:
            raise ValueError("extracellular_current_mid_A must have shape (Nt,) or (B, Nt).")
        footprint_mV_per_A = jnp.asarray(extracellular_footprint_mV_per_A)
        vext_mid_for_vmap = None
        vext_prev_for_vmap = None
    else:
        if extracellular_potential_mid_mV is None:
            raise ValueError("extracellular_potential_mid_mV is required.")
        if extracellular_potential_initial_previous_mV is None:
            raise ValueError("extracellular_potential_initial_previous_mV is required.")
        current_mid_A = None
        current_previous_A = None
        footprint_mV_per_A = None
        vext_mid_for_vmap = extracellular_potential_mid_mV
        vext_prev_for_vmap = extracellular_potential_initial_previous_mV

    def one_batch(
        Vi0_row,
        Ve0_row,
        gates0_row,
        state0_row,
        area_row,
        Cm_abs_row,
        Cx_abs_row,
        Gx_abs_row,
        Gax_e_row,
        Gax_i_row,
        left_i_row,
        right_i_row,
        left_e_row,
        right_e_row,
        I_background_row,
        Iinj_abs_mid,
        vext_mid,
        vext_prev0,
        current_mid,
        current_previous,
        footprint,
        observer_state_row,
        raster_probe_indices_row,
        raster_probe_mask_row,
        row_index,
    ):
        cm_over_dt = Cm_abs_row / dt_ms
        cx_over_dt = Cx_abs_row / dt_ms
        off_i = -Gax_i_row
        off_e = -Gax_e_row
        if use_factorized_vext:
            step_count = int(current_mid.shape[0])
        else:
            vext_previous_mV = jnp.concatenate([vext_prev0[None, :], vext_mid[:-1]], axis=0)
            extracellular_rhs_drive = (
                (cx_over_dt + Gx_abs_row)[None, :] * vext_mid
                - cx_over_dt[None, :] * vext_previous_mV
            )
            step_count = int(extracellular_rhs_drive.shape[0])

        def solve_vi_vperi(
            Vi: Array,
            Ve: Array,
            gates_new: Array,
            Iinj_abs: Array,
            I_outward_den: Array,
            I_corr_den: Array,
            extracellular_drive_abs: Array,
        ) -> tuple[Array, Array]:
            row_terms = getattr(backend, "membrane_conductance_terms_for_row", None)
            if callable(row_terms):
                Gm_den, GE_den = row_terms(row_index, gates_new)
            else:
                Gm_den, GE_den = backend.membrane_conductance_terms(gates_new)
            Gm_abs = Gm_den * area_row
            GE_abs = GE_den * area_row

            I_outward_abs = I_outward_den * area_row
            I_corr_abs = I_corr_den * area_row
            Vm = Vi - Ve

            a00 = cm_over_dt + Gm_abs + left_i_row + right_i_row
            a01 = -(cm_over_dt + Gm_abs)
            rhs0 = (
                cm_over_dt * Vm
                + GE_abs
                + Iinj_abs
                - I_outward_abs
                - I_corr_abs
            )

            a10 = a01
            a11 = (
                cm_over_dt
                + Gm_abs
                + cx_over_dt
                + Gx_abs_row
                + left_e_row
                + right_e_row
            )
            rhs1 = (
                -cm_over_dt * Vm
                - GE_abs
                + cx_over_dt * Ve
                + extracellular_drive_abs
                + I_outward_abs
                + I_corr_abs
            )

            solve_block = _double_cable_block_solve_fn(double_cable_block_solver)
            return solve_block(
                a00,
                a01,
                a10,
                a11,
                off_i,
                off_e,
                rhs0,
                rhs1,
            )

        def step(carry, step_inputs):
            if use_factorized_vext:
                if Iinj_abs_mid is None:
                    current_A, previous_current_A, local_step = step_inputs
                    Iinj_abs = jnp.zeros_like(area_row)
                else:
                    Iinj_abs, current_A, previous_current_A, local_step = step_inputs
                extracellular_drive_abs = (
                    (
                        (cx_over_dt + Gx_abs_row) * current_A
                        - cx_over_dt * previous_current_A
                    )
                    * footprint
                )
            else:
                if Iinj_abs_mid is None:
                    extracellular_drive_abs, local_step = step_inputs
                    Iinj_abs = jnp.zeros_like(area_row)
                else:
                    Iinj_abs, extracellular_drive_abs, local_step = step_inputs

            Vi, Ve, gates, observer_state, *extra = carry
            extra = tuple(extra)
            Vm = Vi - Ve

            row_gate_update = getattr(backend, "cn_gate_update_for_row", None)
            if callable(row_gate_update):
                gates_pred = row_gate_update(
                    row_index,
                    g_prev=gates,
                    V_mV=Vm,
                    dt=dt_ms,
                )
            else:
                gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt_ms)

            if stateless_vm_only:
                linearization_gates = gates if has_driven_extracellular else gates_pred
                explicit_outward_current = I_background_row
                correction_current = jnp.zeros_like(Vm)
            else:
                row_currents = getattr(backend, "currents_for_row", None)
                Iion_pred = (
                    row_currents(row_index, V_mV=Vm, gates=gates_pred)
                    if callable(row_currents)
                    else backend.currents(V_mV=Vm, gates=gates_pred)
                )
                step_plan_pred = membrane.prepare_membrane_step(
                    V_mV=Vm,
                    gates_prev=gates,
                    gates_new=gates_pred,
                    state=extra,
                    dt=dt_ms,
                    I_ion=Iion_pred,
                    I_background=I_background_row,
                )
                linearization_gates = step_plan_pred.linearization_gates
                if has_driven_extracellular:
                    linearization_gates = gates
                explicit_outward_current = step_plan_pred.explicit_outward_current
                correction_current = step_plan_pred.correction_current

            Vi_new, Ve_new = solve_vi_vperi(
                Vi=Vi,
                Ve=Ve,
                gates_new=linearization_gates,
                Iinj_abs=Iinj_abs,
                I_outward_den=explicit_outward_current,
                I_corr_den=correction_current,
                extracellular_drive_abs=extracellular_drive_abs,
            )
            Vm_new = Vi_new - Ve_new

            observer_state = update_vm_raster_state_scalar_from_tables(
                observer_state,
                vm_mV=Vm_new,
                step_index=time_start_index + local_step,
                probe_indices=raster_probe_indices_row,
                probe_mask=raster_probe_mask_row,
                thresholds_mV=raster_thresholds_mV,
            )
            if stateless_vm_only:
                return (Vi_new, Ve_new, gates_pred, observer_state, *extra), None

            gates_new = membrane.final_gate_update(
                gates_prev=gates,
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                dt=dt_ms,
                gates_predictor=gates_pred,
            )
            row_currents = getattr(backend, "currents_for_row", None)
            Iion_new = (
                row_currents(row_index, V_mV=Vm_new, gates=gates_new)
                if callable(row_currents)
                else backend.currents(V_mV=Vm_new, gates=gates_new)
            )
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state=extra,
                dt=dt_ms,
                I_ion=Iion_new,
                I_background=I_background_row,
            )
            state_new = membrane.finalize_membrane_step(
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=extra,
                step_plan=step_plan,
                dt=dt_ms,
            )
            return (Vi_new, Ve_new, gates_new, observer_state, *state_new), None

        local_steps = jnp.arange(step_count, dtype=jnp.asarray(time_start_index).dtype)
        if use_factorized_vext:
            scan_inputs = (
                (current_mid, current_previous, local_steps)
                if Iinj_abs_mid is None
                else (Iinj_abs_mid, current_mid, current_previous, local_steps)
            )
        else:
            scan_inputs = (
                (extracellular_rhs_drive, local_steps)
                if Iinj_abs_mid is None
                else (Iinj_abs_mid, extracellular_rhs_drive, local_steps)
            )
        final_carry, _ = jax.lax.scan(
            step,
            (Vi0_row, Ve0_row, gates0_row, observer_state_row, *state0_row),
            scan_inputs,
        )
        return (
            final_carry[0],
            final_carry[1],
            final_carry[2],
            tuple(final_carry[4:]),
            final_carry[3],
        )

    state_axes = tuple(0 for _ in state0)
    space_in_axes = None if jnp.asarray(area_cm2).ndim == 1 else 0
    edge_in_axes = None if jnp.asarray(Gax_i).ndim == 1 else 0
    background_in_axes = None if jnp.asarray(I_background).ndim <= 1 else 0
    iinj_in_axes = None if intracellular_current_abs_mid is None else 0
    vext_in_axes = None if use_factorized_vext else 0
    vext_prev_in_axes = None if use_factorized_vext else 0
    current_in_axes = (
        None
        if current_mid_A is None or jnp.asarray(current_mid_A).ndim == 1
        else 0
    )
    current_previous_in_axes = (
        None
        if current_previous_A is None or jnp.asarray(current_previous_A).ndim == 1
        else 0
    )
    footprint_in_axes = None if footprint_mV_per_A is None else 0
    return jax.vmap(
        one_batch,
        in_axes=(
            0,
            0,
            0,
            state_axes,
            space_in_axes,
            space_in_axes,
            space_in_axes,
            space_in_axes,
            edge_in_axes,
            edge_in_axes,
            space_in_axes,
            space_in_axes,
            space_in_axes,
            space_in_axes,
            background_in_axes,
            iinj_in_axes,
            vext_in_axes,
            vext_prev_in_axes,
            current_in_axes,
            current_previous_in_axes,
            footprint_in_axes,
            0,
            0,
            0,
            0,
        ),
    )(
        Vi0_mV,
        Ve0_mV,
        gates0,
        state0,
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
        I_background,
        intracellular_current_abs_mid,
        vext_mid_for_vmap,
        vext_prev_for_vmap,
        current_mid_A,
        current_previous_A,
        footprint_mV_per_A,
        observer_state0,
        raster_probe_indices,
        raster_probe_mask,
        row_indices,
    )


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "double_cable_block_solver",
    ),
)
def _run_double_cable_batch_observer_pcr_soa_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    double_cable_block_solver: str,
    Vi0_mV: Array,
    Ve0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    observer_state0: VmRasterState,
    raster_probe_indices: Array,
    raster_probe_mask: Array,
    raster_thresholds_mV: Array,
    area_cm2: Array,
    Cm_abs: Array,
    Cx_abs: Array,
    Gx_abs: Array,
    Gax_e: Array,
    Gax_i: Array,
    left_i: Array,
    right_i: Array,
    left_e: Array,
    right_e: Array,
    I_background: Array,
    intracellular_current_density_mid: Array | None,
    extracellular_potential_mid_mV: Array | None,
    extracellular_potential_initial_previous_mV: Array | None,
    row_indices: Array,
    time_start_index: Array,
    dt_ms: Array,
    extracellular_current_mid_A: Array | None = None,
    extracellular_current_initial_previous_A: Array | None = None,
    extracellular_footprint_mV_per_A: Array | None = None,
) -> tuple[Array, Array, Array, tuple[Array, ...], VmRasterState]:
    """Run one observer-only chunk using the batch-native PCR/SoA solver."""

    batch_size = int(Vi0_mV.shape[0])
    nx = int(Vi0_mV.shape[1])

    def batch_space(values: Array) -> Array:
        arr = jnp.asarray(values)
        if arr.ndim == 0:
            return jnp.broadcast_to(arr, (batch_size, nx))
        if arr.ndim == 1:
            return jnp.broadcast_to(arr[None, :], (batch_size, nx))
        return arr

    area_batch = batch_space(area_cm2)
    cm_over_dt = batch_space(Cm_abs) / dt_ms
    cx_over_dt = batch_space(Cx_abs) / dt_ms
    Gx_abs_batch = batch_space(Gx_abs)
    left_i_batch = batch_space(left_i)
    right_i_batch = batch_space(right_i)
    left_e_batch = batch_space(left_e)
    right_e_batch = batch_space(right_e)
    background_batch = batch_space(I_background)

    off_i = -jnp.asarray(Gax_i)
    off_e = -jnp.asarray(Gax_e)
    use_factorized_vext = extracellular_footprint_mV_per_A is not None
    if use_factorized_vext:
        if extracellular_current_mid_A is None:
            raise ValueError("extracellular_current_mid_A is required.")
        if extracellular_current_initial_previous_A is None:
            raise ValueError("extracellular_current_initial_previous_A is required.")
        footprint_batch = batch_space(extracellular_footprint_mV_per_A)
        current_mid_A = jnp.asarray(extracellular_current_mid_A)
        current_initial_previous_A = jnp.asarray(extracellular_current_initial_previous_A)
        if current_mid_A.ndim == 1:
            current_previous_A = jnp.concatenate(
                [
                    current_initial_previous_A.reshape((1,)),
                    current_mid_A[:-1],
                ],
                axis=0,
            )
            step_count = int(current_mid_A.shape[0])
        elif current_mid_A.ndim == 2:
            current_previous_A = jnp.concatenate(
                [
                    current_initial_previous_A.reshape((batch_size, 1)),
                    current_mid_A[:, :-1],
                ],
                axis=1,
            )
            step_count = int(current_mid_A.shape[1])
        else:
            raise ValueError("extracellular_current_mid_A must have shape (Nt,) or (B, Nt).")
    else:
        if extracellular_potential_mid_mV is None:
            raise ValueError("extracellular_potential_mid_mV is required.")
        if extracellular_potential_initial_previous_mV is None:
            raise ValueError("extracellular_potential_initial_previous_mV is required.")
        vext_previous_mV = jnp.concatenate(
            [
                extracellular_potential_initial_previous_mV[:, None, :],
                extracellular_potential_mid_mV[:, :-1, :],
            ],
            axis=1,
        )
        extracellular_rhs_drive = (
            (cx_over_dt + Gx_abs_batch)[:, None, :] * extracellular_potential_mid_mV
            - cx_over_dt[:, None, :] * vext_previous_mV
        )
        step_count = int(extracellular_rhs_drive.shape[1])

    if intracellular_current_density_mid is None:
        intracellular_current_abs_mid = None
    else:
        intracellular_current_abs_mid = intracellular_current_density_mid * area_batch[:, None, :]

    state_axes = tuple(0 for _ in state0)

    def batch_gate_update(gates: Array, Vm: Array) -> Array:
        row_gate_update = getattr(backend, "cn_gate_update_for_row", None)
        if callable(row_gate_update):
            return jax.vmap(
                lambda row_index, gates_row, vm_row: row_gate_update(
                    row_index,
                    g_prev=gates_row,
                    V_mV=vm_row,
                    dt=dt_ms,
                )
            )(row_indices, gates, Vm)
        return jax.vmap(
            lambda gates_row, vm_row: backend.cn_gate_update(
                g_prev=gates_row,
                V_mV=vm_row,
                dt=dt_ms,
            )
        )(gates, Vm)

    def batch_currents(Vm: Array, gates: Array) -> Array:
        row_currents = getattr(backend, "currents_for_row", None)
        if callable(row_currents):
            return jax.vmap(
                lambda row_index, vm_row, gates_row: row_currents(
                    row_index,
                    V_mV=vm_row,
                    gates=gates_row,
                )
            )(row_indices, Vm, gates)
        return jax.vmap(
            lambda vm_row, gates_row: backend.currents(V_mV=vm_row, gates=gates_row)
        )(Vm, gates)

    def batch_membrane_conductance_terms(gates: Array) -> tuple[Array, Array]:
        row_terms = getattr(backend, "membrane_conductance_terms_for_row", None)
        if callable(row_terms):
            return jax.vmap(
                lambda row_index, gates_row: row_terms(row_index, gates_row)
            )(row_indices, gates)
        return jax.vmap(backend.membrane_conductance_terms)(gates)

    def batch_prepare_membrane_step(
        Vm: Array,
        gates_prev: Array,
        gates_new: Array,
        state: tuple[Array, ...],
        I_ion: Array,
        I_background_row: Array,
    ):
        return jax.vmap(
            lambda vm_row, gates_prev_row, gates_new_row, state_row, iion_row, ibg_row: (
                membrane.prepare_membrane_step(
                    V_mV=vm_row,
                    gates_prev=gates_prev_row,
                    gates_new=gates_new_row,
                    state=state_row,
                    dt=dt_ms,
                    I_ion=iion_row,
                    I_background=ibg_row,
                )
            ),
            in_axes=(0, 0, 0, state_axes, 0, 0),
        )(Vm, gates_prev, gates_new, state, I_ion, I_background_row)

    def batch_final_gate_update(
        gates_prev: Array,
        Vm_prev: Array,
        Vm_new: Array,
        gates_predictor: Array,
    ) -> Array:
        return jax.vmap(
            lambda gates_prev_row, vm_prev_row, vm_new_row, gates_predictor_row: (
                membrane.final_gate_update(
                    gates_prev=gates_prev_row,
                    V_mV_prev=vm_prev_row,
                    V_mV_new=vm_new_row,
                    dt=dt_ms,
                    gates_predictor=gates_predictor_row,
                )
            )
        )(gates_prev, Vm_prev, Vm_new, gates_predictor)

    def batch_finalize_membrane_step(
        Vm_prev: Array,
        Vm_new: Array,
        gates_prev: Array,
        gates_new: Array,
        state_prev: tuple[Array, ...],
        step_plan,
    ) -> tuple[Array, ...]:
        return jax.vmap(
            lambda vm_prev_row, vm_new_row, gates_prev_row, gates_new_row, state_row, step_plan_row: (
                membrane.finalize_membrane_step(
                    V_mV_prev=vm_prev_row,
                    V_mV_new=vm_new_row,
                    gates_prev=gates_prev_row,
                    gates_new=gates_new_row,
                    state_prev=state_row,
                    step_plan=step_plan_row,
                    dt=dt_ms,
                )
            ),
            in_axes=(0, 0, 0, 0, state_axes, 0),
        )(Vm_prev, Vm_new, gates_prev, gates_new, state_prev, step_plan)

    def solve_vi_vperi(
        Vi: Array,
        Ve: Array,
        gates_new: Array,
        Iinj_abs: Array,
        I_outward_den: Array,
        I_corr_den: Array,
        extracellular_drive_abs: Array,
    ) -> tuple[Array, Array]:
        Gm_den, GE_den = batch_membrane_conductance_terms(gates_new)
        Gm_abs = Gm_den * area_batch
        GE_abs = GE_den * area_batch

        I_outward_abs = I_outward_den * area_batch
        I_corr_abs = I_corr_den * area_batch
        Vm = Vi - Ve

        a00 = cm_over_dt + Gm_abs + left_i_batch + right_i_batch
        a01 = -(cm_over_dt + Gm_abs)
        rhs0 = cm_over_dt * Vm + GE_abs + Iinj_abs - I_outward_abs - I_corr_abs

        a10 = a01
        a11 = (
            cm_over_dt
            + Gm_abs
            + cx_over_dt
            + Gx_abs_batch
            + left_e_batch
            + right_e_batch
        )
        rhs1 = (
            -cm_over_dt * Vm
            - GE_abs
            + cx_over_dt * Ve
            + extracellular_drive_abs
            + I_outward_abs
            + I_corr_abs
        )

        if double_cable_block_solver == "pcr_soa":
            return solve_block_tridiagonal_2x2_pcr_soa_batched(
                a00,
                a01,
                a10,
                a11,
                off_i,
                off_e,
                rhs0,
                rhs1,
            )
        raise ValueError(
            f"Unsupported batch-native double-cable block solver: {double_cable_block_solver!r}"
        )

    def current_to_space(value: Array) -> Array:
        value = jnp.asarray(value)
        return value if value.ndim == 0 else value[:, None]

    def step(carry, step_inputs):
        if use_factorized_vext:
            if intracellular_current_abs_mid is None:
                current_A, previous_current_A, local_step = step_inputs
                Iinj_abs = jnp.zeros_like(area_batch)
            else:
                Iinj_abs, current_A, previous_current_A, local_step = step_inputs
            extracellular_drive_abs = (
                (
                    (cx_over_dt + Gx_abs_batch) * current_to_space(current_A)
                    - cx_over_dt * current_to_space(previous_current_A)
                )
                * footprint_batch
            )
        else:
            if intracellular_current_abs_mid is None:
                extracellular_drive_abs, local_step = step_inputs
                Iinj_abs = jnp.zeros_like(area_batch)
            else:
                Iinj_abs, extracellular_drive_abs, local_step = step_inputs
        Vi, Ve, gates, observer_state, *extra = carry
        extra = tuple(extra)
        Vm = Vi - Ve

        gates_pred = batch_gate_update(gates, Vm)
        if stateless_vm_only:
            linearization_gates = gates if has_driven_extracellular else gates_pred
            explicit_outward_current = background_batch
            correction_current = jnp.zeros_like(Vm)
        else:
            Iion_pred = batch_currents(Vm, gates_pred)
            step_plan_pred = batch_prepare_membrane_step(
                Vm,
                gates,
                gates_pred,
                extra,
                Iion_pred,
                background_batch,
            )
            linearization_gates = step_plan_pred.linearization_gates
            if has_driven_extracellular:
                linearization_gates = gates
            explicit_outward_current = step_plan_pred.explicit_outward_current
            correction_current = step_plan_pred.correction_current

        Vi_new, Ve_new = solve_vi_vperi(
            Vi=Vi,
            Ve=Ve,
            gates_new=linearization_gates,
            Iinj_abs=Iinj_abs,
            I_outward_den=explicit_outward_current,
            I_corr_den=correction_current,
            extracellular_drive_abs=extracellular_drive_abs,
        )
        Vm_new = Vi_new - Ve_new

        observer_state = update_vm_raster_state_batch_from_tables(
            observer_state,
            vm_mV=Vm_new,
            step_index=time_start_index + local_step,
            probe_indices=raster_probe_indices,
            probe_mask=raster_probe_mask,
            thresholds_mV=raster_thresholds_mV,
        )
        if stateless_vm_only:
            return (Vi_new, Ve_new, gates_pred, observer_state, *extra), None

        gates_new = batch_final_gate_update(gates, Vm, Vm_new, gates_pred)
        Iion_new = batch_currents(Vm_new, gates_new)
        step_plan = batch_prepare_membrane_step(
            Vm_new,
            gates,
            gates_new,
            extra,
            Iion_new,
            background_batch,
        )
        state_new = batch_finalize_membrane_step(
            Vm,
            Vm_new,
            gates,
            gates_new,
            extra,
            step_plan,
        )
        return (Vi_new, Ve_new, gates_new, observer_state, *state_new), None

    local_steps = jnp.arange(
        step_count,
        dtype=jnp.asarray(time_start_index).dtype,
    )
    if use_factorized_vext:
        current_scan_A = (
            current_mid_A
            if current_mid_A.ndim == 1
            else jnp.swapaxes(current_mid_A, 0, 1)
        )
        previous_scan_A = (
            current_previous_A
            if current_previous_A.ndim == 1
            else jnp.swapaxes(current_previous_A, 0, 1)
        )
        scan_inputs = (
            (current_scan_A, previous_scan_A, local_steps)
            if intracellular_current_abs_mid is None
            else (
                jnp.swapaxes(intracellular_current_abs_mid, 0, 1),
                current_scan_A,
                previous_scan_A,
                local_steps,
            )
        )
    else:
        scan_inputs = (
            (jnp.swapaxes(extracellular_rhs_drive, 0, 1), local_steps)
            if intracellular_current_abs_mid is None
            else (
                jnp.swapaxes(intracellular_current_abs_mid, 0, 1),
                jnp.swapaxes(extracellular_rhs_drive, 0, 1),
                local_steps,
            )
        )
    final_carry, _ = jax.lax.scan(
        step,
        (Vi0_mV, Ve0_mV, gates0, observer_state0, *state0),
        scan_inputs,
    )
    return (
        final_carry[0],
        final_carry[1],
        final_carry[2],
        tuple(final_carry[4:]),
        final_carry[3],
    )


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

            dt = jnp.asarray(grid.dt_ms, dtype=dtype_local)
            lower, diag, upper = cable.lower, cable.diag, cable.upper
            options = _normalize_batch_options(options)
            record_idx, record_full = _resolve_recording(options.recording, nx=membrane_runtime.Nx)
            record_voltage = options.recording.mode != "none"
            chunk_steps = _normalize_time_chunk_steps(options.time_chunk_steps, nt=grid.Nt)
            stateless_vm_only = bool(
                membrane_runtime.membrane.supports_stateless_vm_only_fast_path()
            )
            shared_cable = (
                jnp.asarray(cable.lower).ndim == 1
                and jnp.asarray(cable.diag).ndim == 1
                and jnp.asarray(cable.upper).ndim == 1
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
                    observer_state = _run_single_cable_vstim_batch_sparse_observer_chunks(
                        runtime=runtime,
                        Cm_uF_cm2=jnp.asarray(self.Cm_uF_cm2, dtype=dtype_local),
                        observers=observers,
                        has_driven_extracellular=has_driven_extracellular,
                        stateless_vm_only=stateless_vm_only,
                        intracellular_current_density_mid=sparse_iinj,
                        extracellular_potential_mid_mV=vext_batch,
                        time_chunk_steps=chunk_steps,
                        progress_callback=progress_callback,
                    )
            else:
                assert iinj_batch is not None
                if vext_batch is None and factorized_vext is not None:
                    with benchmark_span(
                        "kernel.materialize_inputs",
                        mode="single",
                        input="factorized_vext",
                        group_size=batch_size,
                    ):
                        vext_batch = materialize_factorized_extracellular_potential_batch(
                            factorized_vext
                        )
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
            with benchmark_span(
                "kernel.finalize_observer",
                mode="single",
                observer="vm_raster",
                group_size=batch_size,
            ):
                observations = cast(
                    dict[str, object],
                    finalize_vm_raster_state(
                        observers,
                        observer_state,
                        nt=grid.Nt,
                        dt_ms=grid.dt_ms,
                    ),
                )
            return BatchKernelResult(
                Vm=None,
                t=grid.t_vec_ms,
                observations=observations,
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
                progress_callback=progress_callback,
            )
            return BatchKernelResult(Vm=out, t=grid.t_vec_ms)
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
        if (
            record_full
            and chunk_steps is None
            and shared_cable
        ):
            with benchmark_span(
                "kernel.dispatch_jax",
                mode="single",
                variant="dense_vstim_full_scan",
                output="full_vm",
                group_size=batch_size,
                time_chunk_steps=chunk_steps,
                chunk_steps=grid.Nt,
                chunk_index=1,
                chunk_count=1,
            ):
                out = _run_single_cable_vstim_batch_vm_scan(
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
                    Cm_uF_cm2=jnp.asarray(self.Cm_uF_cm2, dtype=dtype_local),
                    I_background=membrane_runtime.background_current,
                    Vm0_mV=membrane_runtime.Vm0_mV,
                    gates0=membrane_runtime.gates0,
                    state0=membrane_runtime.state0,
                    intracellular_current_density_mid=iinj_batch,
                    extracellular_potential_mid_mV=vext_batch,
                    dt_ms=dt,
                )
            with benchmark_span(
                "kernel.chunk_bookkeeping",
                mode="single",
                variant="dense_vstim_full_scan",
                output="full_vm",
                group_size=batch_size,
                chunk_index=1,
                chunk_count=1,
            ):
                if progress_callback is not None:
                    progress_callback(1, 1)
        else:
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
                progress_callback=progress_callback,
            )
        return BatchKernelResult(Vm=out, t=grid.t_vec_ms)

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
        benchmark_double_cable_block_solver: str | None = None,
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
                factorized_previous_is_scalar = (
                    factorized_source.current_initial_previous_A is not None
                    and jnp.asarray(factorized_source.current_initial_previous_A).ndim == 0
                )
                if factorized_source.shared_current and factorized_previous_is_scalar:
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
            record_idx, record_full = _resolve_recording(options.recording, nx=nx)
            chunk_steps = _normalize_time_chunk_steps(options.time_chunk_steps, nt=grid.Nt)
            has_driven_extracellular = (
                runtime.stimulation.has_driven_extracellular
                if self.has_driven_extracellular is None
                else bool(self.has_driven_extracellular)
            )
            stateless_vm_only = bool(
                membrane_runtime.membrane.supports_stateless_vm_only_fast_path()
            )
            shared_cable = (
                jnp.asarray(cable.area_cm2).ndim == 1
                and jnp.asarray(extracellular.Cm_abs).ndim == 1
                and jnp.asarray(extracellular.Gax_i).ndim == 1
                and jnp.asarray(extracellular.Gax_e).ndim == 1
            )
            block_solver = _resolve_double_cable_run_block_solver(
                options.double_cable_block_solver
                if benchmark_double_cable_block_solver is None
                else benchmark_double_cable_block_solver,
                platform=jax.default_backend(),
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
                    intracellular_current_density_mid=iinj_batch,
                    extracellular_potential_mid_mV=factorized_vext,
                    extracellular_potential_initial_previous_mV=None,
                    time_chunk_steps=chunk_steps,
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
                    intracellular_current_density_mid=iinj_batch,
                    extracellular_potential_mid_mV=vext_batch,
                    extracellular_potential_initial_previous_mV=vext_previous_batch,
                    time_chunk_steps=chunk_steps,
                    progress_callback=progress_callback,
                )
            with benchmark_span(
                "kernel.finalize_observer",
                mode="double",
                observer="vm_raster",
                group_size=batch_size,
            ):
                observations = cast(
                    dict[str, object],
                    finalize_vm_raster_state(
                        observers,
                        observer_state,
                        nt=grid.Nt,
                        dt_ms=grid.dt_ms,
                    ),
                )
            return BatchKernelResult(
                Vm=None,
                t=grid.t_vec_ms,
                observations=observations,
            )
        if factorized_vext is not None:
            with benchmark_span(
                "kernel.materialize_inputs",
                mode="double",
                input="factorized_vext",
                group_size=batch_size,
            ):
                vext_batch = materialize_factorized_extracellular_potential_batch(
                    factorized_vext
                )
            with benchmark_span(
                "kernel.materialize_inputs",
                mode="double",
                input="factorized_vext_previous",
                group_size=batch_size,
            ):
                vext_previous_batch = materialize_factorized_extracellular_potential_initial_previous(
                    factorized_vext
                )
        if (
            record_full
            and chunk_steps is None
            and shared_cable
            and block_solver == "thomas"
        ):
            with benchmark_span(
                "kernel.prepare_state",
                mode="double",
                variant="dense_vstim_full_scan",
                group_size=batch_size,
                nx=nx,
            ):
                Ve0 = jnp.full(
                    (nx,),
                    jnp.asarray(self.Veinit_mV, dtype=dtype_local),
                    dtype=dtype_local,
                )
                Vm0 = membrane_runtime.Vm0_mV
            with benchmark_span(
                "kernel.dispatch_jax",
                mode="double",
                variant="dense_vstim_full_scan",
                output="full_vm",
                group_size=batch_size,
                time_chunk_steps=chunk_steps,
                chunk_steps=grid.Nt,
                chunk_index=1,
                chunk_count=1,
                block_solver=block_solver,
            ):
                out = _run_double_cable_batch_vm_scan(
                    backend=membrane_runtime.backend,
                    membrane=membrane_runtime.membrane,
                    has_driven_extracellular=has_driven_extracellular,
                    stateless_vm_only=stateless_vm_only,
                    Vi0_mV=Vm0 + Ve0,
                    Ve0_mV=Ve0,
                    gates0=membrane_runtime.gates0,
                    state0=membrane_runtime.state0,
                    area_cm2=cable.area_cm2,
                    Cm_abs=extracellular.Cm_abs,
                    Cx_abs=extracellular.Cx_abs,
                    Gx_abs=extracellular.Gx_abs,
                    Gax_e=extracellular.Gax_e,
                    Gax_i=extracellular.Gax_i,
                    left_i=extracellular.left_i,
                    right_i=extracellular.right_i,
                    left_e=extracellular.left_e,
                    right_e=extracellular.right_e,
                    I_background=membrane_runtime.background_current,
                    intracellular_current_density_mid=iinj_batch,
                    extracellular_potential_mid_mV=cast(Any, vext_batch),
                    extracellular_potential_initial_previous_mV=cast(Any, vext_previous_batch),
                    dt_ms=jnp.asarray(grid.dt_ms, dtype=dtype_local),
                )
            with benchmark_span(
                "kernel.chunk_bookkeeping",
                mode="double",
                variant="dense_vstim_full_scan",
                output="full_vm",
                group_size=batch_size,
                chunk_index=1,
                chunk_count=1,
            ):
                if progress_callback is not None:
                    progress_callback(1, 1)
        else:
            out = _run_double_cable_batch_array_chunks(
                runtime=runtime,
                Veinit_mV=float(self.Veinit_mV),
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
                double_cable_block_solver=block_solver,
                intracellular_current_density_mid=iinj_batch,
                extracellular_potential_mid_mV=cast(Any, vext_batch),
                extracellular_potential_initial_previous_mV=cast(Any, vext_previous_batch),
                record_indices=record_idx,
                record_full=record_full,
                time_chunk_steps=chunk_steps,
                progress_callback=progress_callback,
            )
        return BatchKernelResult(Vm=out, t=grid.t_vec_ms)

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
    progress_callback: Callable[[int, int], None] | None,
) -> Array:
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
    chunks = []

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
            Vm, gates, state, trace = _run_single_cable_vstim_batch_stateful_scan(
                backend=membrane_runtime.backend,
                membrane=membrane_runtime.membrane,
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
                record_full=record_full,
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
            if progress_callback is not None:
                progress_callback(chunk_index, len(chunk_ranges))

    return _concat_trace_chunks(chunks)


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
    progress_callback: Callable[[int, int], None] | None,
) -> Array:
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
        forcing_footprint_mV_per_A = _single_cable_factorized_forcing_footprint(
            extracellular_potential_mid_mV.footprint_mV_per_A,
            lower=lower,
            upper=upper,
            dtype_local=dtype_local,
        )
    Vm, gates, state = _initial_single_cable_batch_state(runtime, batch_size)
    chunks = []

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
            Vm, gates, state, trace = _run_single_cable_factorized_vstim_batch_stateful_scan(
                backend=membrane_runtime.backend,
                membrane=membrane_runtime.membrane,
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
                record_full=record_full,
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
            if progress_callback is not None:
                progress_callback(chunk_index, len(chunk_ranges))

    return _concat_trace_chunks(chunks)


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
    local_observer_chunks = time_chunk_steps is not None
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
                init_vm_raster_state(observers, batch_size=batch_size, nt=stop - start)
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


def _run_single_cable_vstim_batch_sparse_observer_chunks(
    *,
    runtime: SolverRuntime,
    Cm_uF_cm2: Array,
    observers: VmRasterPlan,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    intracellular_current_density_mid: SparseIntracellularCurrentDensityBatch,
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
        variant="sparse_vstim",
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
    local_observer_chunks = time_chunk_steps is not None
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
            variant="sparse_vstim",
            output="observer_only",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
            observer_state_scope="chunk" if local_observer_chunks else "full",
        ):
            observer_state0 = (
                init_vm_raster_state(observers, batch_size=batch_size, nt=stop - start)
                if local_observer_chunks
                else observer_state
            )
            assert observer_state0 is not None
            iinj_values_chunk = intracellular_current_density_mid.density_mid[:, start:stop]
            vext_chunk = extracellular_potential_mid_mV[:, start:stop]
        with benchmark_span(
            "kernel.dispatch_jax",
            mode="single",
            observer="vm_raster",
            variant="sparse_vstim",
            group_size=batch_size,
            time_chunk_steps=time_chunk_steps,
            chunk_steps=stop - start,
            chunk_index=chunk_index,
            chunk_count=len(chunk_ranges),
            observer_state_scope="chunk" if local_observer_chunks else "full",
        ):
            Vm, gates, state, observer_state = _run_single_cable_vstim_batch_sparse_observer_scan(
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
                intracellular_current_density_values_mid=iinj_values_chunk,
                intracellular_current_density_indices=intracellular_current_density_mid.indices,
                intracellular_current_density_mask=intracellular_current_density_mid.mask,
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
            variant="sparse_vstim",
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
            variant="sparse_vstim",
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
    ):
        current_rows_mid_A = _factorized_current_mid_rows(
            extracellular_potential_mid_mV,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
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
    local_observer_chunks = time_chunk_steps is not None
    observer_state = (
        None
        if local_observer_chunks
        else init_vm_raster_state(observers, batch_size=batch_size, nt=grid.Nt)
    )
    observer_chunk_states: list[VmRasterState] = []
    observer_chunk_starts: list[int] = []
    observer_chunk_lengths: list[int] = []
    current_mid_A = jnp.asarray(
        current_rows_mid_A,
        dtype=dtype_local,
    )
    forcing_footprint_mV_per_A = _single_cable_factorized_forcing_footprint(
        extracellular_potential_mid_mV.footprint_mV_per_A,
        lower=lower,
        upper=upper,
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
        ):
            observer_state0 = (
                init_vm_raster_state(observers, batch_size=batch_size, nt=stop - start)
                if local_observer_chunks
                else observer_state
            )
            assert observer_state0 is not None
            iinj_values_chunk = intracellular_current_density_mid.density_mid[:, start:stop]
            current_chunk = current_mid_A[:, :, start:stop]
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
        ):
            Vm, gates, state, observer_state = _run_single_cable_factorized_vstim_batch_sparse_observer_scan(
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


def _factorized_current_mid_rows(
    batch: FactorizedExtracellularPotentialBatch,
    *,
    dtype_local: Any,
    batch_size: int,
) -> Array:
    current = jnp.asarray(batch.current_mid_A, dtype=dtype_local)
    if current.ndim == 1:
        return jnp.broadcast_to(current[None, None, :], (batch_size, 1, current.shape[0]))
    if current.ndim == 2:
        return current[:, None, :]
    if current.ndim == 3:
        return current
    raise ValueError(
        "factorized current_mid_A must have shape (Nt,), (B, Nt), or (B, K, Nt), "
        f"got {current.shape}."
    )


def _single_cable_factorized_forcing_footprint(
    footprint_mV_per_A: Array,
    *,
    lower: Array,
    upper: Array,
    dtype_local: Any,
) -> Array:
    """Lower factorized Vstim footprints to diffusion forcing footprints once."""

    with benchmark_span("kernel.prepare_factorized_forcing", mode="single"):
        footprint = jnp.asarray(footprint_mV_per_A, dtype=dtype_local)
        lower_rows = jnp.asarray(lower, dtype=dtype_local)
        upper_rows = jnp.asarray(upper, dtype=dtype_local)
        if footprint.ndim == 3:
            batch_size, drive_count, nx = footprint.shape
            flattened = footprint.reshape((batch_size * drive_count, nx))
            lower_rows = jnp.broadcast_to(
                lower_rows[:, None, :],
                (batch_size, drive_count, nx),
            ).reshape((batch_size * drive_count, nx))
            upper_rows = jnp.broadcast_to(
                upper_rows[:, None, :],
                (batch_size, drive_count, nx),
            ).reshape((batch_size * drive_count, nx))
            forcing = _single_cable_factorized_forcing_footprint(
                flattened,
                lower=lower_rows,
                upper=upper_rows,
                dtype_local=dtype_local,
            )
            return forcing.reshape((batch_size, drive_count, nx))
        if footprint.ndim != 2:
            raise ValueError(
                "factorized single-cable footprints must have shape (B, Nx) or (B, K, Nx), "
                f"got {footprint.shape}."
            )
        out = jnp.zeros_like(footprint)
        nx = int(footprint.shape[1])
        if nx >= 2:
            out = out.at[:, 0].set(upper_rows[:, 0] * (footprint[:, 1] - footprint[:, 0]))
            out = out.at[:, -1].set(lower_rows[:, -1] * (footprint[:, -2] - footprint[:, -1]))
        if nx > 2:
            out = out.at[:, 1:-1].set(
                lower_rows[:, 1:-1] * (footprint[:, :-2] - footprint[:, 1:-1])
                + upper_rows[:, 1:-1] * (footprint[:, 2:] - footprint[:, 1:-1])
            )
        return out


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
    local_observer_chunks = time_chunk_steps is not None
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
                init_vm_raster_state(observers, batch_size=batch_size, nt=stop - start)
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


def _run_double_cable_batch_array_chunks(
    *,
    runtime: SolverRuntime,
    Veinit_mV: float,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    double_cable_block_solver: str,
    intracellular_current_density_mid: Array | None,
    extracellular_potential_mid_mV: Array,
    extracellular_potential_initial_previous_mV: Array,
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
    batch_size = int(extracellular_potential_mid_mV.shape[0])
    kernel_block_solver = _resolve_double_cable_kernel_block_solver(
        double_cable_block_solver,
        batch_size=batch_size,
    )
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
    if shared_coefficients:
        area_cm2 = _as_space_array(
            "area_cm2", runtime.cable.area_cm2, nx=nx, dtype_local=dtype_local
        )
        Cm_abs = _as_space_array("Cm_abs", extracellular.Cm_abs, nx=nx, dtype_local=dtype_local)
        Cx_abs = _as_space_array("Cx_abs", extracellular.Cx_abs, nx=nx, dtype_local=dtype_local)
        Gx_abs = _as_space_array("Gx_abs", extracellular.Gx_abs, nx=nx, dtype_local=dtype_local)
        Gax_e = _as_edge_array("Gax_e", extracellular.Gax_e, nx=nx, dtype_local=dtype_local)
        Gax_i = _as_edge_array("Gax_i", extracellular.Gax_i, nx=nx, dtype_local=dtype_local)
        left_i = _as_space_array("left_i", extracellular.left_i, nx=nx, dtype_local=dtype_local)
        right_i = _as_space_array(
            "right_i", extracellular.right_i, nx=nx, dtype_local=dtype_local
        )
        left_e = _as_space_array("left_e", extracellular.left_e, nx=nx, dtype_local=dtype_local)
        right_e = _as_space_array(
            "right_e", extracellular.right_e, nx=nx, dtype_local=dtype_local
        )
        background = _as_scalar_or_space_array(
            "I_background",
            membrane_runtime.background_current,
            nx=nx,
            dtype_local=dtype_local,
        )
    else:
        area_cm2 = _as_batched_space_array(
            "area_cm2", runtime.cable.area_cm2, nx=nx, dtype_local=dtype_local, batch_size=batch_size
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
            "Gax_e", extracellular.Gax_e, nx=nx, dtype_local=dtype_local, batch_size=batch_size
        )
        Gax_i = _as_batched_edge_array(
            "Gax_i", extracellular.Gax_i, nx=nx, dtype_local=dtype_local, batch_size=batch_size
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
        ):
            vext_chunk = extracellular_potential_mid_mV[:, start:stop]
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
        ):
            if _use_batch_native_double_cable_pcr_soa_solver(
                kernel_block_solver,
                batch_size=batch_size,
            ):
                Vi, Ve, gates, state, trace = _run_double_cable_batch_stateful_pcr_soa_scan(
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
            previous = vext_chunk[:, -1]
            chunks.append(trace)
            if progress_callback is not None:
                progress_callback(chunk_index, len(chunk_ranges))

    return _concat_trace_chunks(chunks)


def _run_double_cable_batch_observer_chunks(
    *,
    runtime: SolverRuntime,
    Veinit_mV: float,
    observers: VmRasterPlan,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    double_cable_block_solver: str,
    intracellular_current_density_mid: Array | None,
    extracellular_potential_mid_mV: Array | FactorizedExtracellularPotentialBatch,
    extracellular_potential_initial_previous_mV: Array | None,
    time_chunk_steps: int | None,
    progress_callback: Callable[[int, int], None] | None,
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
                intracellular_current_density_mid=intracellular_current_density_mid,
                extracellular_potential_mid_mV=dense_vext,
                extracellular_potential_initial_previous_mV=dense_previous,
                time_chunk_steps=time_chunk_steps,
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
    if shared_coefficients:
        area_cm2 = _as_space_array(
            "area_cm2", runtime.cable.area_cm2, nx=nx, dtype_local=dtype_local
        )
        Cm_abs = _as_space_array("Cm_abs", extracellular.Cm_abs, nx=nx, dtype_local=dtype_local)
        Cx_abs = _as_space_array("Cx_abs", extracellular.Cx_abs, nx=nx, dtype_local=dtype_local)
        Gx_abs = _as_space_array("Gx_abs", extracellular.Gx_abs, nx=nx, dtype_local=dtype_local)
        Gax_e = _as_edge_array("Gax_e", extracellular.Gax_e, nx=nx, dtype_local=dtype_local)
        Gax_i = _as_edge_array("Gax_i", extracellular.Gax_i, nx=nx, dtype_local=dtype_local)
        left_i = _as_space_array("left_i", extracellular.left_i, nx=nx, dtype_local=dtype_local)
        right_i = _as_space_array(
            "right_i", extracellular.right_i, nx=nx, dtype_local=dtype_local
        )
        left_e = _as_space_array("left_e", extracellular.left_e, nx=nx, dtype_local=dtype_local)
        right_e = _as_space_array(
            "right_e", extracellular.right_e, nx=nx, dtype_local=dtype_local
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
    Vi, Ve, gates, state = _initial_double_cable_batch_state(runtime, batch_size, Veinit_mV)
    raster_probe_indices, raster_probe_mask = _vm_raster_probe_tables_for_kernel(
        observers,
        batch_size=batch_size,
    )
    previous = extracellular_potential_initial_previous_mV
    previous_current_A = (
        None
        if factorized_vext is None
        else jnp.asarray(factorized_vext.current_initial_previous_A, dtype=dtype_local)
    )
    factorized_current_mid_A = (
        None
        if factorized_vext is None
        else jnp.asarray(factorized_vext.current_mid_A, dtype=dtype_local)
    )
    factorized_footprint_mV_per_A = (
        None
        if factorized_vext is None
        else jnp.asarray(factorized_vext.footprint_mV_per_A, dtype=dtype_local)
    )

    chunk_ranges = tuple(_time_chunks(grid.Nt, time_chunk_steps))
    local_observer_chunks = time_chunk_steps is not None
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
        ):
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
            observer_state0 = (
                init_vm_raster_state(observers, batch_size=batch_size, nt=stop - start)
                if local_observer_chunks
                else observer_state
            )
            assert observer_state0 is not None
            time_start_index = jnp.asarray(
                0 if local_observer_chunks else start,
                dtype=jnp.int32,
            )
        if kernel_block_solver == "pcr_soa":
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
            ):
                Vi, Ve, gates, state, observer_state = _run_double_cable_batch_observer_pcr_soa_scan(
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
        Vm = _broadcast_batch_leading(membrane_runtime.Vm0_mV, batch_size)
        gates = _broadcast_batch_leading(membrane_runtime.gates0, batch_size)
        state = tuple(
            _broadcast_batch_leading(values, batch_size)
            for values in membrane_runtime.state0
        )
        return Vm, gates, state


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
        Ve = jnp.full(
            (batch_size, nx),
            jnp.asarray(Veinit_mV, dtype=dtype_local),
            dtype=dtype_local,
        )
        Vm = _as_batched_space_array(
            "Vm0_mV",
            membrane_runtime.Vm0_mV,
            nx=nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        Vi = Vm + Ve
        gates = _as_batched_row_array(
            "gates0",
            membrane_runtime.gates0,
            row_shape=(nx, membrane_runtime.backend.n_gates_max),
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        state = tuple(
            _broadcast_batch_leading(values, batch_size)
            for values in membrane_runtime.state0
        )
        return Vi, Ve, gates, state


def _broadcast_batch_leading(values: Array, batch_size: int) -> Array:
    arr = jnp.asarray(values)
    return jnp.broadcast_to(arr, (batch_size, *arr.shape))


def _normalize_batch_options(options: BatchOptions | None) -> BatchOptions:
    return BatchOptions.full() if options is None else options


def _resolve_recording(recording: BatchRecording, *, nx: int) -> tuple[Array, bool]:
    indices = recording.indices_for(nx)
    if indices is None:
        return jnp.arange(nx, dtype=jnp.int32), True
    return jnp.asarray(indices, dtype=jnp.int32), False


def _combine_vm_raster_chunk_states(
    states: list[VmRasterState],
    *,
    starts: list[int],
    lengths: list[int],
    nt: int,
    mode: str,
    variant: str,
    time_chunk_steps: int | None,
) -> VmRasterState:
    with benchmark_span(
        "kernel.combine_observer_chunks",
        mode=mode,
        variant=variant,
        observer="vm_raster",
        observer_state_scope="chunk",
        chunk_count=len(states),
        chunk_steps_min=min(lengths) if lengths else None,
        chunk_steps_max=max(lengths) if lengths else None,
        time_chunk_steps=time_chunk_steps,
        nt=nt,
    ):
        return combine_vm_raster_chunk_states(
            states,
            starts=starts,
            lengths=lengths,
            nt=nt,
        )


def _normalize_time_chunk_steps(time_chunk_steps: int | None, *, nt: int) -> int | None:
    if time_chunk_steps is None:
        return None
    steps = int(time_chunk_steps)
    if steps < 1:
        raise ValueError("time_chunk_steps must be >= 1.")
    return min(steps, int(nt))


def _time_chunks(nt: int, time_chunk_steps: int | None):
    chunk_steps = nt if time_chunk_steps is None else time_chunk_steps
    for start in range(0, nt, chunk_steps):
        yield start, min(start + chunk_steps, nt)


def _concat_trace_chunks(chunks: list[Array]) -> Array:
    with benchmark_span(
        "kernel.concat_trace_chunks",
        chunk_count=len(chunks),
        single_chunk=len(chunks) == 1,
    ):
        if not chunks:
            raise ValueError("at least one time chunk is required.")
        if len(chunks) == 1:
            return chunks[0]
        return jnp.concatenate(chunks, axis=1)


def _as_sparse_intracellular_current_density_batch(
    name: str,
    values: SparseIntracellularCurrentDensityBatch,
    *,
    nt: int,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int,
) -> SparseIntracellularCurrentDensityBatch:
    density_mid = jnp.asarray(values.density_mid, dtype=dtype_local)
    indices = jnp.asarray(values.indices, dtype=jnp.int32)
    mask = jnp.asarray(values.mask, dtype=bool)
    if int(values.target_nx) != int(nx):
        raise ValueError(f"{name}.target_nx must be {nx}, got {values.target_nx}.")
    if density_mid.ndim != 3:
        raise ValueError(f"{name}.density_mid must have shape (B, Nt, K).")
    if density_mid.shape[:2] != (batch_size, nt):
        raise ValueError(
            f"{name}.density_mid must have leading shape (B, Nt)="
            f"({batch_size}, {nt}), got {density_mid.shape}."
        )
    sparse_shape = (batch_size, int(density_mid.shape[2]))
    if indices.shape != sparse_shape:
        raise ValueError(f"{name}.indices must have shape {sparse_shape}, got {indices.shape}.")
    if mask.shape != sparse_shape:
        raise ValueError(f"{name}.mask must have shape {sparse_shape}, got {mask.shape}.")
    if bool(jnp.any(jnp.where(mask, (indices < 0) | (indices >= nx), False))):
        raise ValueError(f"{name}.indices contains an out-of-range compartment index.")
    return SparseIntracellularCurrentDensityBatch(
        density_mid=density_mid,
        indices=indices,
        mask=mask,
        target_nx=nx,
    )


def _as_factorized_extracellular_potential_batch(
    name: str,
    values: FactorizedExtracellularPotentialBatch,
    *,
    nt: int,
    nx: int,
    dtype_local: jnp.dtype,
) -> FactorizedExtracellularPotentialBatch:
    current_mid_A = jnp.asarray(values.current_mid_A, dtype=dtype_local)
    current_initial_previous_A = (
        None
        if values.current_initial_previous_A is None
        else jnp.asarray(values.current_initial_previous_A, dtype=dtype_local)
    )
    footprint_mV_per_A = jnp.asarray(values.footprint_mV_per_A, dtype=dtype_local)
    if int(values.target_nx) != int(nx):
        raise ValueError(f"{name}.target_nx must be {nx}, got {values.target_nx}.")
    if footprint_mV_per_A.ndim not in {2, 3} or footprint_mV_per_A.shape[-1] != nx:
        raise ValueError(
            f"{name}.footprint_mV_per_A must have shape (B, Nx) or "
            f"(B, K, Nx) with Nx={nx}, "
            f"got {footprint_mV_per_A.shape}."
        )
    batch_size = int(footprint_mV_per_A.shape[0])
    drive_count = 1 if footprint_mV_per_A.ndim == 2 else int(footprint_mV_per_A.shape[1])
    if footprint_mV_per_A.ndim == 2 and current_mid_A.ndim == 1:
        if current_mid_A.shape != (nt,):
            raise ValueError(
                f"{name}.current_mid_A must have shape (Nt,)=({nt},), "
                f"got {current_mid_A.shape}."
            )
    elif footprint_mV_per_A.ndim == 2 and current_mid_A.ndim == 2:
        if current_mid_A.shape != (batch_size, nt):
            raise ValueError(
                f"{name}.current_mid_A must have shape (B, Nt)="
                f"({batch_size}, {nt}), got {current_mid_A.shape}."
            )
    elif footprint_mV_per_A.ndim == 3 and current_mid_A.ndim == 3:
        if current_mid_A.shape != (batch_size, drive_count, nt):
            raise ValueError(
                f"{name}.current_mid_A must have shape (B, K, Nt)="
                f"({batch_size}, {drive_count}, {nt}), got {current_mid_A.shape}."
            )
    else:
        raise ValueError(
            f"{name}.current_mid_A shape {current_mid_A.shape} is incompatible "
            f"with footprint shape {footprint_mV_per_A.shape}."
        )
    if current_initial_previous_A is not None:
        if footprint_mV_per_A.ndim == 2:
            valid_previous = (
                current_initial_previous_A.ndim == 0
                or current_initial_previous_A.shape == (batch_size,)
            )
            expected = f"scalar or (B,)=({batch_size},)"
        else:
            valid_previous = current_initial_previous_A.shape == (batch_size, drive_count)
            expected = f"(B, K)=({batch_size}, {drive_count})"
        if not valid_previous:
            raise ValueError(
                f"{name}.current_initial_previous_A must have shape {expected}, "
                f"got {current_initial_previous_A.shape}."
            )
    return FactorizedExtracellularPotentialBatch(
        current_mid_A=current_mid_A,
        footprint_mV_per_A=footprint_mV_per_A,
        target_nx=nx,
        current_initial_previous_A=current_initial_previous_A,
    )


def _as_batched_time_space_array(
    name: str,
    values: Array,
    *,
    nt: int,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int | None = None,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 2:
        if arr.shape != (nt, nx):
            raise ValueError(
                f"{name} must have shape (Nt, Nx)=({nt}, {nx}) "
                f"or (B, Nt, Nx), got {arr.shape}."
            )
        arr = arr[jnp.newaxis, :, :]
    elif arr.ndim == 3:
        if arr.shape[1:] != (nt, nx):
            raise ValueError(
                f"{name} must have trailing shape (Nt, Nx)=({nt}, {nx}), "
                f"got {arr.shape}."
            )
    else:
        raise ValueError(
            f"{name} must have shape (Nt, Nx) or (B, Nt, Nx), got {arr.shape}."
        )

    if batch_size is None:
        return arr
    if arr.shape[0] == batch_size:
        return arr
    if arr.shape[0] == 1:
        return jnp.broadcast_to(arr, (batch_size, nt, nx))
    raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")


def _as_space_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim != 1 or arr.shape != (nx,):
        raise ValueError(f"{name} must have shape (Nx,)=({nx},), got {arr.shape}.")
    return arr


def _as_edge_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
) -> Array:
    edge_count = max(int(nx) - 1, 0)
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim != 1 or arr.shape != (edge_count,):
        raise ValueError(
            f"{name} must have shape (Nx-1,)=({edge_count},), got {arr.shape}."
        )
    return arr


def _as_scalar_or_space_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 0:
        return arr
    return _as_space_array(name, arr, nx=nx, dtype_local=dtype_local)


def _as_batched_space_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int | None = None,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 1:
        if arr.shape != (nx,):
            raise ValueError(
                f"{name} must have shape (Nx,)=({nx},) or (B, Nx), got {arr.shape}."
            )
        arr = arr[jnp.newaxis, :]
    elif arr.ndim == 2:
        if arr.shape[1:] != (nx,):
            raise ValueError(
                f"{name} must have trailing shape (Nx,)=({nx},), got {arr.shape}."
            )
    else:
        raise ValueError(f"{name} must have shape (Nx,) or (B, Nx), got {arr.shape}.")

    if batch_size is None:
        return arr
    if arr.shape[0] == batch_size:
        return arr
    if arr.shape[0] == 1:
        return jnp.broadcast_to(arr, (batch_size, nx))
    raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")


def _as_batched_edge_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int,
) -> Array:
    edge_count = max(int(nx) - 1, 0)
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 1:
        if arr.shape != (edge_count,):
            raise ValueError(
                f"{name} must have shape (Nx-1,)=({edge_count},) or "
                f"(B, Nx-1), got {arr.shape}."
            )
        arr = arr[jnp.newaxis, :]
    elif arr.ndim == 2:
        if arr.shape[1:] != (edge_count,):
            raise ValueError(
                f"{name} must have trailing shape (Nx-1,)=({edge_count},), got {arr.shape}."
            )
    else:
        raise ValueError(f"{name} must have shape (Nx-1,) or (B, Nx-1), got {arr.shape}.")

    if arr.shape[0] == batch_size:
        return arr
    if arr.shape[0] == 1:
        return jnp.broadcast_to(arr, (batch_size, edge_count))
    raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")


def _as_batched_scalar_or_space_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr[jnp.newaxis], (batch_size,))
    if arr.ndim == 1 and arr.shape == (batch_size,):
        return arr
    return _as_batched_space_array(
        name,
        arr,
        nx=nx,
        dtype_local=dtype_local,
        batch_size=batch_size,
    )


def _as_batched_row_array(
    name: str,
    values: Array,
    *,
    row_shape: tuple[int, ...],
    dtype_local: jnp.dtype,
    batch_size: int,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.shape == row_shape:
        return jnp.broadcast_to(arr[jnp.newaxis, ...], (batch_size, *row_shape))
    if arr.ndim == len(row_shape) + 1 and arr.shape[1:] == row_shape:
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            return jnp.broadcast_to(arr, (batch_size, *row_shape))
        raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")
    raise ValueError(f"{name} must have shape {row_shape} or (B, *{row_shape}), got {arr.shape}.")


__all__ = [
    "BatchKernelResult",
    "BatchOptions",
    "BatchRecording",
    "SingleCableVStimBatchKernel",
    "DoubleCableBatchKernel",
]
