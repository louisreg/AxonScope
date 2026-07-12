"""CPU Thomas double-cable JAX batch scans."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from axonscope.runtime.jax.recording.observer import (
    VmRasterState,
    update_vm_raster_state_scalar_from_tables,
)

from .block_tridiagonal import solve_block_tridiagonal_2x2_scalar
from ..cable_geometry import Array
from .inputs import _record_vm_row


_CPU_DOUBLE_CABLE_BLOCK_SOLVER = "thomas"


def _double_cable_block_solve_fn(solver: str):
    if solver == _CPU_DOUBLE_CABLE_BLOCK_SOLVER:
        return solve_block_tridiagonal_2x2_scalar
    raise ValueError(f"Unsupported double-cable block solver: {solver!r}")

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
    extracellular_potential_mid_mV: Array | None,
    extracellular_potential_initial_previous_mV: Array | None,
    row_indices: Array,
    record_indices: Array,
    dt_ms: Array,
    extracellular_current_mid_A: Array | None = None,
    extracellular_current_initial_previous_A: Array | None = None,
    extracellular_footprint_mV_per_A: Array | None = None,
) -> tuple[Array, Array, Array, tuple[Array, ...], Array]:
    """Run one double-cable time chunk and return final batch state."""

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
        row_index,
        record_indices_row,
    ):
        cm_over_dt = Cm_abs_row / dt_ms
        cx_over_dt = Cx_abs_row / dt_ms
        off_i = -Gax_i_row
        off_e = -Gax_e_row
        if not use_factorized_vext:
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
            if use_factorized_vext:
                if Iinj_abs_mid is None:
                    current_A, previous_current_A = step_inputs
                    Iinj_abs = jnp.zeros_like(area_row)
                else:
                    Iinj_abs, current_A, previous_current_A = step_inputs
                extracellular_drive_abs = (
                    (
                        (cx_over_dt + Gx_abs_row) * current_A
                        - cx_over_dt * previous_current_A
                    )
                    * footprint
                )
            else:
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
                output = _record_vm_row(
                    Vm_new,
                    record_indices_row,
                    record_full=record_full,
                )
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
            output = _record_vm_row(
                Vm_new,
                record_indices_row,
                record_full=record_full,
            )
            return (Vi_new, Ve_new, gates_new, *state_new), output

        if use_factorized_vext:
            scan_inputs = (
                (current_mid, current_previous)
                if Iinj_abs_mid is None
                else (Iinj_abs_mid, current_mid, current_previous)
            )
        else:
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
    record_indices_axes = 0 if jnp.asarray(record_indices).ndim == 2 else None
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
            record_indices_axes,
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
        row_indices,
        record_indices,
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

__all__ = [
    "_run_double_cable_batch_observer_scan",
    "_run_double_cable_batch_stateful_scan",
]
