"""GPU tiled-Thomas double-cable JAX batch scans."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from axonscope.runtime.jax.recording.observer import (
    VmRasterState,
    update_vm_raster_state_batch_from_tables,
)

from . import double_cable_step as solver_core
from ..cable_geometry import Array
from .double_cable_linear import (
    prepare_double_cable_linear_system_static_terms,
    prepare_double_cable_linear_system_static_terms_xb,
)
from .inputs import _record_vm_batch


_GPU_DOUBLE_CABLE_BLOCK_SOLVER = "jax_triton_loop_xb"
_INTERNAL_DOUBLE_CABLE_BLOCK_SOLVERS = frozenset({_GPU_DOUBLE_CABLE_BLOCK_SOLVER})


def _use_batch_native_double_cable_integrated_solver(
    solver: str,
    *,
    batch_size: int,
) -> bool:
    return solver in _INTERNAL_DOUBLE_CABLE_BLOCK_SOLVERS

@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "record_full",
        "double_cable_block_solver",
        "tiled_thomas_block_b",
    ),
)
def _run_double_cable_batch_stateful_integrated_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    record_full: bool,
    double_cable_block_solver: str,
    tiled_thomas_block_b: int,
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
    """Run one time chunk using a batch-native double-cable block solver."""

    batch_size = int(Vi0_mV.shape[0])
    nx = int(Vi0_mV.shape[1])

    linear_static = prepare_double_cable_linear_system_static_terms(
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
        dt_ms=dt_ms,
        batch_size=batch_size,
        nx=nx,
    )
    linear_static_xb = (
        prepare_double_cable_linear_system_static_terms_xb(
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
            dt_ms=dt_ms,
            batch_size=batch_size,
            nx=nx,
        )
        if double_cable_block_solver == "jax_triton_loop_xb"
        else None
    )
    area_batch = linear_static.area
    background_abs = linear_static.background_abs
    zero_abs = linear_static.zero_abs

    def batch_space(values: Array) -> Array:
        arr = jnp.asarray(values)
        if arr.ndim == 0:
            return jnp.broadcast_to(arr, (batch_size, nx))
        if arr.ndim == 1:
            return jnp.broadcast_to(arr[None, :], (batch_size, nx))
        return arr

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
        elif current_mid_A.ndim == 2:
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
            linear_static.cx_plus_gx[:, None, :] * extracellular_potential_mid_mV
            - linear_static.cx_over_dt[:, None, :] * vext_previous_mV
        )

    if intracellular_current_density_mid is None:
        intracellular_current_abs_mid = None
    else:
        intracellular_current_abs_mid = intracellular_current_density_mid * area_batch[:, None, :]

    batch_gate_update = partial(
        solver_core.batch_gate_update,
        backend=backend,
        row_indices=row_indices,
        dt_ms=dt_ms,
    )
    batch_currents = partial(
        solver_core.batch_currents,
        backend=backend,
        row_indices=row_indices,
    )
    batch_prepare_membrane_step = partial(
        solver_core.batch_prepare_membrane_step,
        membrane=membrane,
        dt_ms=dt_ms,
    )
    batch_final_gate_update = partial(
        solver_core.batch_final_gate_update,
        membrane=membrane,
        dt_ms=dt_ms,
    )
    batch_finalize_membrane_step = partial(
        solver_core.batch_finalize_membrane_step,
        membrane=membrane,
        dt_ms=dt_ms,
    )

    solve_vi_vperi = partial(
        solver_core.solve_double_cable_batch_step,
        backend=backend,
        row_indices=row_indices,
        area_batch=area_batch,
        linear_static=linear_static,
        linear_static_xb=linear_static_xb,
        batch_size=batch_size,
        nx=nx,
        double_cable_block_solver=double_cable_block_solver,
        tiled_thomas_block_b=tiled_thomas_block_b,
    )

    def current_to_space(value: Array) -> Array:
        value = jnp.asarray(value)
        return value if value.ndim == 0 else value[:, None]

    def step(carry, step_inputs):
        if use_factorized_vext:
            if intracellular_current_abs_mid is None:
                current_A, previous_current_A = step_inputs
                Iinj_abs = jnp.zeros_like(area_batch)
            else:
                Iinj_abs, current_A, previous_current_A = step_inputs
            extracellular_drive_abs = (
                (
                    linear_static.cx_plus_gx * current_to_space(current_A)
                    - linear_static.cx_over_dt * current_to_space(previous_current_A)
                )
                * footprint_batch
            )
        else:
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
            explicit_outward_current_abs = background_abs
            correction_current_abs = zero_abs
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
            explicit_outward_current_abs = step_plan_pred.explicit_outward_current * area_batch
            correction_current_abs = step_plan_pred.correction_current * area_batch

        Vi_new, Ve_new = solve_vi_vperi(
            Vi=Vi,
            Ve=Ve,
            gates_new=linearization_gates,
            Iinj_abs=Iinj_abs,
            I_outward_abs=explicit_outward_current_abs,
            I_corr_abs=correction_current_abs,
            extracellular_drive_abs=extracellular_drive_abs,
        )
        Vm_new = Vi_new - Ve_new

        output = _record_vm_batch(
            Vm_new,
            record_indices,
            record_full=record_full,
        )
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
            (current_scan_A, previous_scan_A)
            if intracellular_current_abs_mid is None
            else (
                jnp.swapaxes(intracellular_current_abs_mid, 0, 1),
                current_scan_A,
                previous_scan_A,
            )
        )
    else:
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
        "tiled_thomas_block_b",
    ),
)
def _run_double_cable_batch_observer_integrated_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    double_cable_block_solver: str,
    tiled_thomas_block_b: int,
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
    """Run one observer-only chunk using the batch-native tiled-Thomas solver."""

    batch_size = int(Vi0_mV.shape[0])
    nx = int(Vi0_mV.shape[1])

    linear_static = prepare_double_cable_linear_system_static_terms(
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
        dt_ms=dt_ms,
        batch_size=batch_size,
        nx=nx,
    )
    linear_static_xb = (
        prepare_double_cable_linear_system_static_terms_xb(
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
            dt_ms=dt_ms,
            batch_size=batch_size,
            nx=nx,
        )
        if double_cable_block_solver == "jax_triton_loop_xb"
        else None
    )
    area_batch = linear_static.area
    background_abs = linear_static.background_abs
    zero_abs = linear_static.zero_abs

    def batch_space(values: Array) -> Array:
        arr = jnp.asarray(values)
        if arr.ndim == 0:
            return jnp.broadcast_to(arr, (batch_size, nx))
        if arr.ndim == 1:
            return jnp.broadcast_to(arr[None, :], (batch_size, nx))
        return arr
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
            linear_static.cx_plus_gx[:, None, :] * extracellular_potential_mid_mV
            - linear_static.cx_over_dt[:, None, :] * vext_previous_mV
        )
        step_count = int(extracellular_rhs_drive.shape[1])

    if intracellular_current_density_mid is None:
        intracellular_current_abs_mid = None
    else:
        intracellular_current_abs_mid = intracellular_current_density_mid * area_batch[:, None, :]

    batch_gate_update = partial(
        solver_core.batch_gate_update,
        backend=backend,
        row_indices=row_indices,
        dt_ms=dt_ms,
    )
    batch_currents = partial(
        solver_core.batch_currents,
        backend=backend,
        row_indices=row_indices,
    )
    batch_prepare_membrane_step = partial(
        solver_core.batch_prepare_membrane_step,
        membrane=membrane,
        dt_ms=dt_ms,
    )
    batch_final_gate_update = partial(
        solver_core.batch_final_gate_update,
        membrane=membrane,
        dt_ms=dt_ms,
    )
    batch_finalize_membrane_step = partial(
        solver_core.batch_finalize_membrane_step,
        membrane=membrane,
        dt_ms=dt_ms,
    )

    solve_vi_vperi = partial(
        solver_core.solve_double_cable_batch_step,
        backend=backend,
        row_indices=row_indices,
        area_batch=area_batch,
        linear_static=linear_static,
        linear_static_xb=linear_static_xb,
        batch_size=batch_size,
        nx=nx,
        double_cable_block_solver=double_cable_block_solver,
        tiled_thomas_block_b=tiled_thomas_block_b,
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
                    linear_static.cx_plus_gx * current_to_space(current_A)
                    - linear_static.cx_over_dt * current_to_space(previous_current_A)
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
            explicit_outward_current_abs = background_abs
            correction_current_abs = zero_abs
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
            explicit_outward_current_abs = step_plan_pred.explicit_outward_current * area_batch
            correction_current_abs = step_plan_pred.correction_current * area_batch

        Vi_new, Ve_new = solve_vi_vperi(
            Vi=Vi,
            Ve=Ve,
            gates_new=linearization_gates,
            Iinj_abs=Iinj_abs,
            I_outward_abs=explicit_outward_current_abs,
            I_corr_abs=correction_current_abs,
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

__all__ = [
    "_run_double_cable_batch_observer_integrated_scan",
    "_run_double_cable_batch_stateful_integrated_scan",
    "_use_batch_native_double_cable_integrated_solver",
]
