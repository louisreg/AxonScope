from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, cast

import jax
import jax.numpy as jnp

from .batch_inputs import (
    SparseIntracellularCurrentDensityBatch,
    materialize_sparse_intracellular_current_density_batch,
)
from .common import Array, apply_diffusion_operator, solve_block_tridiagonal_2x2_scalar
from .kernels import _run_double_cable_vm_scan, _run_single_cable_vstim_vm_scan
from .observer_runtime import (
    ObserverState,
    SolverObserverPlan,
    finalize_observer_state,
    init_observer_state,
    update_observer_state_scalar,
)
from .options import BatchOptions, BatchRecording
from .runtime import SolverRuntime


@dataclass(frozen=True)
class BatchKernelResult:
    """Raw batched solver-kernel output before packaging public simulations."""

    Vm: Array | None
    t: Array
    observations: dict[str, object] | None = None


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
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    extracellular_potential_initial_previous_mV: Array,
    dt_ms: Array,
) -> Array:
    """Run the full double-cable scan over a leading batch axis."""

    def one_batch(Iinj_mid: Array, vext_mid: Array, vext_previous: Array) -> Array:
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

    return jax.vmap(one_batch)(
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
    observer_state0: tuple[Array, ...],
    observer_kind_codes: Array,
    observer_indices: Array,
    observer_mask: Array,
    observer_original_indices: Array,
    observer_positions_um: Array,
    observer_thresholds_mV: Array,
    observer_blanking_ms: Array,
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    time_start_index: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], tuple[Array, ...]]:
    """Run one time chunk while updating compact observers in the scan."""

    def one_batch(
        Vm0_row,
        gates0_row,
        state0_row,
        observer_state_row,
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
                observer_state = update_observer_state_scalar(
                    observer_state,
                    vm_mV=Vm_new,
                    time_ms=(time_start_index + local_step + 1) * dt_ms,
                    kind_codes=observer_kind_codes,
                    indices=observer_indices,
                    mask=observer_mask,
                    original_indices=observer_original_indices,
                    positions_um=observer_positions_um,
                    thresholds_mV=observer_thresholds_mV,
                    blanking_ms=observer_blanking_ms,
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
            observer_state = update_observer_state_scalar(
                observer_state,
                vm_mV=Vm_new,
                time_ms=(time_start_index + local_step + 1) * dt_ms,
                kind_codes=observer_kind_codes,
                indices=observer_indices,
                mask=observer_mask,
                original_indices=observer_original_indices,
                positions_um=observer_positions_um,
                thresholds_mV=observer_thresholds_mV,
                blanking_ms=observer_blanking_ms,
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
    observer_axes = tuple(0 for _ in observer_state0)
    return jax.vmap(
        one_batch,
        in_axes=(0, 0, state_axes, observer_axes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )(
        Vm0_mV,
        gates0,
        state0,
        observer_state0,
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
    observer_state0: tuple[Array, ...],
    observer_kind_codes: Array,
    observer_indices: Array,
    observer_mask: Array,
    observer_original_indices: Array,
    observer_positions_um: Array,
    observer_thresholds_mV: Array,
    observer_blanking_ms: Array,
    intracellular_current_density_values_mid: Array,
    intracellular_current_density_indices: Array,
    intracellular_current_density_mask: Array,
    extracellular_potential_mid_mV: Array,
    time_start_index: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], tuple[Array, ...]]:
    """Run one observer chunk with sparse point-clamp intracellular input."""

    def one_batch(
        Vm0_row,
        gates0_row,
        state0_row,
        observer_state_row,
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
                observer_state = update_observer_state_scalar(
                    observer_state,
                    vm_mV=Vm_new,
                    time_ms=(time_start_index + local_step + 1) * dt_ms,
                    kind_codes=observer_kind_codes,
                    indices=observer_indices,
                    mask=observer_mask,
                    original_indices=observer_original_indices,
                    positions_um=observer_positions_um,
                    thresholds_mV=observer_thresholds_mV,
                    blanking_ms=observer_blanking_ms,
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
            observer_state = update_observer_state_scalar(
                observer_state,
                vm_mV=Vm_new,
                time_ms=(time_start_index + local_step + 1) * dt_ms,
                kind_codes=observer_kind_codes,
                indices=observer_indices,
                mask=observer_mask,
                original_indices=observer_original_indices,
                positions_um=observer_positions_um,
                thresholds_mV=observer_thresholds_mV,
                blanking_ms=observer_blanking_ms,
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
    observer_axes = tuple(0 for _ in observer_state0)
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
        "record_full",
    ),
)
def _run_double_cable_batch_stateful_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    record_full: bool,
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
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    extracellular_potential_initial_previous_mV: Array,
    row_indices: Array,
    record_indices: Array,
    dt_ms: Array,
) -> tuple[Array, Array, Array, tuple[Array, ...], Array]:
    """Run one double-cable time chunk and return final batch state."""

    intracellular_current_abs_mid = intracellular_current_density_mid * area_cm2[:, None, :]

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

            return solve_block_tridiagonal_2x2_scalar(
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

        final_carry, trace = jax.lax.scan(
            step,
            (Vi0_row, Ve0_row, gates0_row, *state0_row),
            (Iinj_abs_mid, extracellular_rhs_drive),
        )
        return final_carry[0], final_carry[1], final_carry[2], tuple(final_carry[3:]), trace

    state_axes = tuple(0 for _ in state0)
    return jax.vmap(
        one_batch,
        in_axes=(0, 0, 0, state_axes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
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
        extracellular_potential_mid_mV: Array | None = None,
        intracellular_current_density_mid: Array | None = None,
        options: BatchOptions | None = None,
        observers: SolverObserverPlan | None = None,
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

        vext_mid = (
            runtime.stimulation.extracellular_potential_mid_mV
            if extracellular_potential_mid_mV is None
            else extracellular_potential_mid_mV
        )
        if vext_mid is None:
            raise ValueError("extracellular_potential_mid_mV is required for Vstim batching.")

        vext_batch = _as_batched_time_space_array(
            "extracellular_potential_mid_mV",
            vext_mid,
            nt=grid.Nt,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
        )
        batch_size = int(vext_batch.shape[0])

        iinj_mid = (
            runtime.stimulation.intracellular_current_density_mid
            if intracellular_current_density_mid is None
            else intracellular_current_density_mid
        )
        if iinj_mid is None:
            raise ValueError("intracellular_current_density_mid is required for Vstim batching.")
        sparse_iinj = (
            _as_sparse_intracellular_current_density_batch(
                "intracellular_current_density_mid",
                iinj_mid,
                nt=grid.Nt,
                nx=membrane_runtime.Nx,
                dtype_local=dtype_local,
                batch_size=batch_size,
            )
            if isinstance(iinj_mid, SparseIntracellularCurrentDensityBatch)
            else None
        )
        iinj_batch = None
        if sparse_iinj is None:
            iinj_batch = _as_batched_time_space_array(
                "intracellular_current_density_mid",
                iinj_mid,
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
        has_driven_extracellular = (
            runtime.stimulation.has_driven_extracellular
            if self.has_driven_extracellular is None
            else bool(self.has_driven_extracellular)
        )
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
                observations=cast(
                    dict[str, object],
                    finalize_observer_state(observers, observer_state),
                ),
            )
        if iinj_batch is None:
            assert sparse_iinj is not None
            iinj_batch = materialize_sparse_intracellular_current_density_batch(sparse_iinj)
        if record_full and chunk_steps is None and shared_cable:
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
        extracellular_potential_mid_mV: Array | None = None,
        extracellular_potential_initial_previous_mV: Array | None = None,
        intracellular_current_density_mid: Array | None = None,
        options: BatchOptions | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
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

        vext_mid = (
            runtime.stimulation.extracellular_potential_mid_mV
            if extracellular_potential_mid_mV is None
            else extracellular_potential_mid_mV
        )
        if vext_mid is None:
            raise ValueError(
                "extracellular_potential_mid_mV is required for double-cable batching."
            )
        vext_batch = _as_batched_time_space_array(
            "extracellular_potential_mid_mV",
            vext_mid,
            nt=grid.Nt,
            nx=nx,
            dtype_local=dtype_local,
        )
        batch_size = int(vext_batch.shape[0])

        vext_previous = (
            runtime.stimulation.extracellular_potential_initial_previous_mV
            if extracellular_potential_initial_previous_mV is None
            else extracellular_potential_initial_previous_mV
        )
        if vext_previous is None:
            raise ValueError(
                "extracellular_potential_initial_previous_mV is required for double-cable batching."
            )
        vext_previous_batch = _as_batched_space_array(
            "extracellular_potential_initial_previous_mV",
            vext_previous,
            nx=nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )

        iinj_mid = (
            runtime.stimulation.intracellular_current_density_mid
            if intracellular_current_density_mid is None
            else intracellular_current_density_mid
        )
        if iinj_mid is None:
            raise ValueError(
                "intracellular_current_density_mid is required for double-cable batching."
            )
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
        if record_full and chunk_steps is None and shared_cable:
            Ve0 = jnp.full(
                (nx,),
                jnp.asarray(self.Veinit_mV, dtype=dtype_local),
                dtype=dtype_local,
            )
            Vm0 = membrane_runtime.Vm0_mV
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
                extracellular_potential_mid_mV=vext_batch,
                extracellular_potential_initial_previous_mV=vext_previous_batch,
                dt_ms=jnp.asarray(grid.dt_ms, dtype=dtype_local),
            )
            if progress_callback is not None:
                progress_callback(1, 1)
        else:
            out = _run_double_cable_batch_array_chunks(
                runtime=runtime,
                Veinit_mV=float(self.Veinit_mV),
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
                intracellular_current_density_mid=iinj_batch,
                extracellular_potential_mid_mV=vext_batch,
                extracellular_potential_initial_previous_mV=vext_previous_batch,
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
            intracellular_current_density_mid=intracellular_current_density_mid[:, start:stop],
            extracellular_potential_mid_mV=extracellular_potential_mid_mV[:, start:stop],
            record_indices=record_indices,
            dt_ms=dt,
        )
        chunks.append(trace)
        if progress_callback is not None:
            progress_callback(chunk_index, len(chunk_ranges))

    return _concat_trace_chunks(chunks)


def _run_single_cable_vstim_batch_observer_chunks(
    *,
    runtime: SolverRuntime,
    Cm_uF_cm2: Array,
    observers: SolverObserverPlan,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    time_chunk_steps: int | None,
    progress_callback: Callable[[int, int], None] | None,
) -> ObserverState:
    membrane_runtime = runtime.membrane
    grid = runtime.grid
    cable = runtime.cable
    dtype_local = membrane_runtime.dtype
    dt = jnp.asarray(grid.dt_ms, dtype=dtype_local)
    batch_size = int(extracellular_potential_mid_mV.shape[0])
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
    observer_state = init_observer_state(observers, batch_size=batch_size)

    chunk_ranges = tuple(_time_chunks(grid.Nt, time_chunk_steps))
    for chunk_index, (start, stop) in enumerate(chunk_ranges, start=1):
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
            observer_state0=observer_state,
            observer_kind_codes=observers.kind_codes,
            observer_indices=observers.indices,
            observer_mask=observers.mask,
            observer_original_indices=observers.original_indices,
            observer_positions_um=observers.positions_um,
            observer_thresholds_mV=observers.thresholds_mV,
            observer_blanking_ms=observers.blanking_ms,
            intracellular_current_density_mid=intracellular_current_density_mid[:, start:stop],
            extracellular_potential_mid_mV=extracellular_potential_mid_mV[:, start:stop],
            time_start_index=jnp.asarray(start, dtype=jnp.int32),
            dt_ms=dt,
        )
        if progress_callback is not None:
            progress_callback(chunk_index, len(chunk_ranges))

    return observer_state


def _run_single_cable_vstim_batch_sparse_observer_chunks(
    *,
    runtime: SolverRuntime,
    Cm_uF_cm2: Array,
    observers: SolverObserverPlan,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    intracellular_current_density_mid: SparseIntracellularCurrentDensityBatch,
    extracellular_potential_mid_mV: Array,
    time_chunk_steps: int | None,
    progress_callback: Callable[[int, int], None] | None,
) -> ObserverState:
    membrane_runtime = runtime.membrane
    grid = runtime.grid
    cable = runtime.cable
    dtype_local = membrane_runtime.dtype
    dt = jnp.asarray(grid.dt_ms, dtype=dtype_local)
    batch_size = int(extracellular_potential_mid_mV.shape[0])
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
    observer_state = init_observer_state(observers, batch_size=batch_size)

    chunk_ranges = tuple(_time_chunks(grid.Nt, time_chunk_steps))
    for chunk_index, (start, stop) in enumerate(chunk_ranges, start=1):
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
            observer_state0=observer_state,
            observer_kind_codes=observers.kind_codes,
            observer_indices=observers.indices,
            observer_mask=observers.mask,
            observer_original_indices=observers.original_indices,
            observer_positions_um=observers.positions_um,
            observer_thresholds_mV=observers.thresholds_mV,
            observer_blanking_ms=observers.blanking_ms,
            intracellular_current_density_values_mid=(
                intracellular_current_density_mid.density_mid[:, start:stop]
            ),
            intracellular_current_density_indices=intracellular_current_density_mid.indices,
            intracellular_current_density_mask=intracellular_current_density_mid.mask,
            extracellular_potential_mid_mV=extracellular_potential_mid_mV[:, start:stop],
            time_start_index=jnp.asarray(start, dtype=jnp.int32),
            dt_ms=dt,
        )
        if progress_callback is not None:
            progress_callback(chunk_index, len(chunk_ranges))

    return observer_state


def _run_double_cable_batch_array_chunks(
    *,
    runtime: SolverRuntime,
    Veinit_mV: float,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    intracellular_current_density_mid: Array,
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
        vext_chunk = extracellular_potential_mid_mV[:, start:stop]
        Vi, Ve, gates, state, trace = _run_double_cable_batch_stateful_scan(
            backend=membrane_runtime.backend,
            membrane=membrane_runtime.membrane,
            has_driven_extracellular=has_driven_extracellular,
            stateless_vm_only=stateless_vm_only,
            record_full=record_full,
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
            intracellular_current_density_mid=intracellular_current_density_mid[:, start:stop],
            extracellular_potential_mid_mV=vext_chunk,
            extracellular_potential_initial_previous_mV=previous,
            row_indices=jnp.arange(batch_size, dtype=jnp.int32),
            record_indices=record_indices,
            dt_ms=jnp.asarray(grid.dt_ms, dtype=dtype_local),
        )
        previous = vext_chunk[:, -1]
        chunks.append(trace)
        if progress_callback is not None:
            progress_callback(chunk_index, len(chunk_ranges))

    return _concat_trace_chunks(chunks)


def _initial_single_cable_batch_state(
    runtime: SolverRuntime,
    batch_size: int,
) -> tuple[Array, Array, tuple[Array, ...]]:
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


def _normalize_time_chunk_steps(time_chunk_steps: int | None, *, nt: int) -> int | None:
    if time_chunk_steps is None:
        return None
    steps = int(time_chunk_steps)
    if steps < 1:
        raise ValueError("time_chunk_steps must be >= 1.")
    if steps >= nt:
        return None
    return steps


def _time_chunks(nt: int, time_chunk_steps: int | None):
    chunk_steps = nt if time_chunk_steps is None else time_chunk_steps
    for start in range(0, nt, chunk_steps):
        yield start, min(start + chunk_steps, nt)


def _concat_trace_chunks(chunks: list[Array]) -> Array:
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
