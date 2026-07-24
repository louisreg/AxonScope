"""GPU tiled-Thomas double-cable JAX batch scans."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from axonfleet.runtime.jax.membranes.backend import advance_stateless_membrane_terms
from axonfleet.runtime.jax.recording.observer import (
    ObserverRetention,
    ThresholdObserverState,
    update_threshold_observer_state_batch_from_tables,
)

from . import double_cable_step as solver_core
from ..cable_geometry import Array
from .double_cable_linear import (
    double_cable_space_from_xb,
    prepare_double_cable_linear_system_static_terms_xb,
)
from .inputs import _record_vm_batch
from .dense_recording import (
    empty_recording_matrix_batch,
    record_matrix_batch,
)


_GPU_DOUBLE_CABLE_BLOCK_SOLVER = "jax_triton_loop_xb"
_INTERNAL_DOUBLE_CABLE_BLOCK_SOLVERS = frozenset({_GPU_DOUBLE_CABLE_BLOCK_SOLVER})


def _factorized_current_previous_batch(
    current_mid_A: Array,
    current_initial_previous_A: Array,
    *,
    batch_size: int,
) -> Array:
    current = jnp.asarray(current_mid_A)
    initial = jnp.asarray(current_initial_previous_A)
    if current.ndim == 1:
        return jnp.concatenate([initial.reshape((1,)), current[:-1]], axis=0)
    if current.ndim == 2:
        initial_rows = (
            jnp.broadcast_to(initial, (batch_size,)) if initial.ndim == 0 else initial
        )
        return jnp.concatenate(
            [initial_rows.reshape((batch_size, 1)), current[:, :-1]],
            axis=1,
        )
    if current.ndim == 3:
        drive_count = int(current.shape[1])
        initial_rows = (
            jnp.broadcast_to(initial[None, :], (batch_size, drive_count))
            if initial.ndim == 1
            else initial
        )
        return jnp.concatenate(
            [initial_rows.reshape((batch_size, drive_count, 1)), current[:, :, :-1]],
            axis=2,
        )
    raise ValueError(
        "extracellular_current_mid_A must have shape (Nt,), (B, Nt), or (B, S, Nt)."
    )


def _factorized_extracellular_drive_batch(
    *,
    cx_plus_gx_batch: Array,
    cx_over_dt_batch: Array,
    current_A: Array,
    previous_current_A: Array,
    footprint_batch: Array,
) -> Array:
    footprint = jnp.asarray(footprint_batch)
    current = jnp.asarray(current_A)
    previous = jnp.asarray(previous_current_A)
    if footprint.ndim == 2:
        current_space = current if current.ndim == 0 else current[:, None]
        previous_space = previous if previous.ndim == 0 else previous[:, None]
        return (
            cx_plus_gx_batch * current_space
            - cx_over_dt_batch * previous_space
        ) * footprint
    per_drive = (
        cx_plus_gx_batch[:, None, :] * current[:, :, None]
        - cx_over_dt_batch[:, None, :] * previous[:, :, None]
    ) * footprint
    return jnp.sum(per_drive, axis=1)


def _use_batch_native_double_cable_integrated_solver(
    solver: str,
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
        "record_gates",
        "record_occupancies",
        "record_currents",
        "record_conductances",
        "record_states",
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
    record_gates: bool,
    record_occupancies: bool,
    record_currents: bool,
    record_conductances: bool,
    record_states: bool,
    tiled_thomas_block_b: int,
    Vi0_mV: Array,
    Ve0_mV: Array,
    gates0: Array,
    membrane_parameters: dict[str, Array] | None,
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

    linear_static_xb = prepare_double_cable_linear_system_static_terms_xb(
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
    area_batch = double_cable_space_from_xb(linear_static_xb.area)
    background_abs = double_cable_space_from_xb(linear_static_xb.background_abs)
    zero_abs = double_cable_space_from_xb(linear_static_xb.zero_abs)
    cx_plus_gx_batch = double_cable_space_from_xb(linear_static_xb.cx_plus_gx)
    cx_over_dt_batch = double_cable_space_from_xb(linear_static_xb.cx_over_dt)

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
        current_previous_A = _factorized_current_previous_batch(
            current_mid_A,
            current_initial_previous_A,
            batch_size=batch_size,
        )
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
            cx_plus_gx_batch[:, None, :] * extracellular_potential_mid_mV
            - cx_over_dt_batch[:, None, :] * vext_previous_mV
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
        linear_static_xb=linear_static_xb,
        batch_size=batch_size,
        nx=nx,
        tiled_thomas_block_b=tiled_thomas_block_b,
    )

    def step(carry, step_inputs):
        if use_factorized_vext:
            if intracellular_current_abs_mid is None:
                current_A, previous_current_A = step_inputs
                Iinj_abs = jnp.zeros_like(area_batch)
            else:
                Iinj_abs, current_A, previous_current_A = step_inputs
            extracellular_drive_abs = _factorized_extracellular_drive_batch(
                cx_plus_gx_batch=cx_plus_gx_batch,
                cx_over_dt_batch=cx_over_dt_batch,
                current_A=current_A,
                previous_current_A=previous_current_A,
                footprint_batch=footprint_batch,
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

        membrane_terms = None
        if stateless_vm_only:
            gates_pred, Gm_den, GE_den = advance_stateless_membrane_terms(
                backend,
                gates=gates,
                static_gates=None,
                V_mV=Vm,
                dt_ms=dt_ms,
                linearize_previous=False,
                parameters=membrane_parameters,
            )
            linearization_gates = gates_pred
            membrane_terms = (Gm_den, GE_den)
            explicit_outward_current_abs = background_abs
            correction_current_abs = zero_abs
        else:
            gates_pred = batch_gate_update(gates, Vm)
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
            membrane_terms=membrane_terms,
        )
        Vm_new = Vi_new - Ve_new

        vm_output = _record_vm_batch(
            Vm_new,
            record_indices,
            record_full=record_full,
        )
        if stateless_vm_only:
            output = (
                vm_output,
                {
                    "gates": (
                        record_matrix_batch(
                            membrane.gate_trace_matrix(gates_pred, extra, Vm_new),
                            record_indices,
                            record_full=record_full,
                        )
                        if record_gates
                        else empty_recording_matrix_batch(Vm_new)
                    ),
                    "occupancies": (
                        record_matrix_batch(
                            membrane.occupancy_trace_matrix(gates_pred),
                            record_indices,
                            record_full=record_full,
                        )
                        if record_occupancies
                        else empty_recording_matrix_batch(Vm_new)
                    ),
                    "currents": (
                        record_matrix_batch(
                            membrane.ionic_current_trace_matrix(
                                Vm_new, gates_pred, extra
                            ),
                            record_indices,
                            record_full=record_full,
                        )
                        if record_currents
                        else empty_recording_matrix_batch(Vm_new)
                    ),
                    "conductances": (
                        record_matrix_batch(
                            membrane.conductance_trace_matrix(
                                gates_pred, extra, Vm_new
                            ),
                            record_indices,
                            record_full=record_full,
                        )
                        if record_conductances
                        else empty_recording_matrix_batch(Vm_new)
                    ),
                    "states": (
                        record_matrix_batch(
                            membrane.membrane_state_trace_matrix(extra),
                            record_indices,
                            record_full=record_full,
                        )
                        if record_states
                        else empty_recording_matrix_batch(Vm_new)
                    ),
                },
            )
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
        output = (
            vm_output,
            {
                "gates": (
                    record_matrix_batch(
                        membrane.gate_trace_matrix(gates_new, state_new, Vm_new),
                        record_indices,
                        record_full=record_full,
                    )
                    if record_gates
                    else empty_recording_matrix_batch(Vm_new)
                ),
                "occupancies": (
                    record_matrix_batch(
                        membrane.occupancy_trace_matrix(gates_new),
                        record_indices,
                        record_full=record_full,
                    )
                    if record_occupancies
                    else empty_recording_matrix_batch(Vm_new)
                ),
                "currents": (
                    record_matrix_batch(
                            membrane.recorded_ionic_current_trace_matrix(
                                Vm,
                                Vm_new,
                                gates,
                                gates_new,
                                extra,
                                state_new,
                                step_plan,
                                Iion_new,
                            ),
                        record_indices,
                        record_full=record_full,
                    )
                    if record_currents
                    else empty_recording_matrix_batch(Vm_new)
                ),
                "conductances": (
                    record_matrix_batch(
                        membrane.conductance_trace_matrix(
                            gates_new, state_new, Vm_new
                        ),
                        record_indices,
                        record_full=record_full,
                    )
                    if record_conductances
                    else empty_recording_matrix_batch(Vm_new)
                ),
                "states": (
                    record_matrix_batch(
                        membrane.membrane_state_trace_matrix(state_new),
                        record_indices,
                        record_full=record_full,
                    )
                    if record_states
                    else empty_recording_matrix_batch(Vm_new)
                ),
            },
        )
        return (Vi_new, Ve_new, gates_new, *state_new), output

    if use_factorized_vext:
        current_scan_A = (
            current_mid_A
            if current_mid_A.ndim == 1
            else jnp.moveaxis(current_mid_A, -1, 0)
        )
        previous_scan_A = (
            current_previous_A
            if current_previous_A.ndim == 1
            else jnp.moveaxis(current_previous_A, -1, 0)
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
        jax.tree.map(lambda values: jnp.swapaxes(values, 0, 1), trace),
    )

@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "tiled_thomas_block_b",
        "observer_retention",
        "raster_temporal_stride",
    ),
)
def _run_double_cable_batch_observer_integrated_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    tiled_thomas_block_b: int,
    observer_retention: ObserverRetention,
    Vi0_mV: Array,
    Ve0_mV: Array,
    gates0: Array,
    membrane_parameters: dict[str, Array] | None,
    state0: tuple[Array, ...],
    observer_state0: ThresholdObserverState,
    raster_probe_indices: Array,
    raster_probe_mask: Array,
    raster_thresholds_mV: Array,
    raster_blanking_ms: Array,
    raster_reset_thresholds_mV: Array,
    raster_refractory_ms: Array,
    raster_temporal_stride: int,
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
) -> tuple[Array, Array, Array, tuple[Array, ...], ThresholdObserverState]:
    """Run one observer-only chunk using the batch-native tiled-Thomas solver."""

    batch_size = int(Vi0_mV.shape[0])
    nx = int(Vi0_mV.shape[1])

    linear_static_xb = prepare_double_cable_linear_system_static_terms_xb(
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
    use_node_first_state = bool(
        stateless_vm_only
        and getattr(backend, "supports_node_first_batch", False)
    )
    if use_node_first_state:
        area_batch = linear_static_xb.area
        background_abs = linear_static_xb.background_abs
        zero_abs = linear_static_xb.zero_abs
    else:
        area_batch = double_cable_space_from_xb(linear_static_xb.area)
        background_abs = double_cable_space_from_xb(linear_static_xb.background_abs)
        zero_abs = double_cable_space_from_xb(linear_static_xb.zero_abs)
    cx_plus_gx_batch = double_cable_space_from_xb(linear_static_xb.cx_plus_gx)
    cx_over_dt_batch = double_cable_space_from_xb(linear_static_xb.cx_over_dt)

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
        current_previous_A = _factorized_current_previous_batch(
            current_mid_A,
            current_initial_previous_A,
            batch_size=batch_size,
        )
        step_count = int(current_mid_A.shape[-1])
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
            cx_plus_gx_batch[:, None, :] * extracellular_potential_mid_mV
            - cx_over_dt_batch[:, None, :] * vext_previous_mV
        )
        step_count = int(extracellular_rhs_drive.shape[1])

    if intracellular_current_density_mid is None:
        intracellular_current_abs_mid = None
    else:
        intracellular_area = (
            jnp.swapaxes(area_batch, 0, 1)
            if use_node_first_state
            else area_batch
        )
        intracellular_current_abs_mid = (
            intracellular_current_density_mid * intracellular_area[:, None, :]
        )

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
        linear_static_xb=linear_static_xb,
        batch_size=batch_size,
        nx=nx,
        tiled_thomas_block_b=tiled_thomas_block_b,
        return_node_first=use_node_first_state,
    )

    gates0_scan = jnp.swapaxes(gates0, 0, 1) if use_node_first_state else gates0
    split_scan_gates = getattr(backend, "split_scan_gates", None)
    merge_scan_gates = getattr(backend, "merge_scan_gates", None)
    if (
        use_node_first_state
        and callable(split_scan_gates)
        and callable(merge_scan_gates)
    ):
        gates0_scan, static_scan_gates = split_scan_gates(gates0_scan)
    else:
        static_scan_gates = None

    def step(carry, step_inputs):
        if use_factorized_vext:
            if intracellular_current_abs_mid is None:
                current_A, previous_current_A, local_step = step_inputs
                Iinj_abs = jnp.zeros_like(area_batch)
            else:
                Iinj_abs, current_A, previous_current_A, local_step = step_inputs
            extracellular_drive_abs = _factorized_extracellular_drive_batch(
                cx_plus_gx_batch=cx_plus_gx_batch,
                cx_over_dt_batch=cx_over_dt_batch,
                current_A=current_A,
                previous_current_A=previous_current_A,
                footprint_batch=footprint_batch,
            )
            if use_node_first_state:
                extracellular_drive_abs = jnp.swapaxes(
                    extracellular_drive_abs,
                    0,
                    1,
                )
        else:
            if intracellular_current_abs_mid is None:
                extracellular_drive_abs, local_step = step_inputs
                Iinj_abs = jnp.zeros_like(area_batch)
            else:
                Iinj_abs, extracellular_drive_abs, local_step = step_inputs
        if use_node_first_state:
            if intracellular_current_abs_mid is not None:
                Iinj_abs = jnp.swapaxes(Iinj_abs, 0, 1)
            if not use_factorized_vext:
                extracellular_drive_abs = jnp.swapaxes(
                    extracellular_drive_abs,
                    0,
                    1,
                )
        Vi, Ve, gates, observer_state, *extra = carry
        extra = tuple(extra)
        Vm = Vi - Ve

        membrane_terms = None
        if stateless_vm_only:
            gates_pred, Gm_den, GE_den = advance_stateless_membrane_terms(
                backend,
                gates=gates,
                static_gates=static_scan_gates,
                V_mV=Vm,
                dt_ms=dt_ms,
                linearize_previous=False,
                parameters=membrane_parameters,
            )
            linearization_gates = gates_pred
            membrane_terms = (Gm_den, GE_den)
            explicit_outward_current_abs = background_abs
            correction_current_abs = zero_abs
        else:
            gates_pred = batch_gate_update(gates, Vm)
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
            static_gates=static_scan_gates,
            membrane_terms=membrane_terms,
        )
        Vm_new = Vi_new - Ve_new

        observer_state = update_threshold_observer_state_batch_from_tables(
            observer_state,
            vm_mV=(
                jnp.swapaxes(Vm_new, 0, 1)
                if use_node_first_state
                else Vm_new
            ),
            step_index=time_start_index + local_step,
            probe_indices=raster_probe_indices,
            probe_mask=raster_probe_mask,
            thresholds_mV=raster_thresholds_mV,
            blanking_ms=raster_blanking_ms,
            reset_thresholds_mV=raster_reset_thresholds_mV,
            refractory_ms=raster_refractory_ms,
            temporal_stride=raster_temporal_stride,
            dt_ms=dt_ms,
            retention=observer_retention,
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
            else jnp.moveaxis(current_mid_A, -1, 0)
        )
        previous_scan_A = (
            current_previous_A
            if current_previous_A.ndim == 1
            else jnp.moveaxis(current_previous_A, -1, 0)
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
        (
            jnp.swapaxes(Vi0_mV, 0, 1) if use_node_first_state else Vi0_mV,
            jnp.swapaxes(Ve0_mV, 0, 1) if use_node_first_state else Ve0_mV,
            gates0_scan,
            observer_state0,
            *state0,
        ),
        scan_inputs,
    )
    final_gates = final_carry[2]
    if static_scan_gates is not None:
        final_gates = merge_scan_gates(final_gates, static_scan_gates)
    return (
        (
            jnp.swapaxes(final_carry[0], 0, 1)
            if use_node_first_state
            else final_carry[0]
        ),
        (
            jnp.swapaxes(final_carry[1], 0, 1)
            if use_node_first_state
            else final_carry[1]
        ),
        (
            jnp.swapaxes(final_gates, 0, 1)
            if use_node_first_state
            else final_gates
        ),
        tuple(final_carry[4:]),
        final_carry[3],
    )
