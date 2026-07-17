"""Jitted scan bodies for the single-cable batch kernel."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from axonscope.runtime.jax.cable_geometry import Array, apply_diffusion_operator
from axonscope.runtime.jax.recording.observer import (
    ObserverRetention,
    ThresholdObserverState,
    update_threshold_observer_state_scalar_from_tables,
)

from .inputs import _record_vm_row


def _record_matrix_row(values: Array, record_indices: Array, *, record_full: bool) -> Array:
    if record_full:
        return values
    return jnp.take(values, record_indices, axis=0)


def _empty_recording_matrix(vm: Array) -> Array:
    return jnp.zeros((vm.shape[0], 0), dtype=vm.dtype)


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "record_full",
        "record_gates",
        "record_currents",
        "record_conductances",
        "record_states",
    ),
)
def _run_single_cable_vstim_batch_stateful_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    record_full: bool,
    record_gates: bool,
    record_currents: bool,
    record_conductances: bool,
    record_states: bool,
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
) -> tuple[Array, Array, tuple[Array, ...], Array, dict[str, Array]]:
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
        record_indices_row,
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
                output = _record_vm_row(
                    Vm_new,
                    record_indices_row,
                    record_full=record_full,
                )
                gates_out = (
                    _record_matrix_row(
                        membrane.gate_trace_matrix(gates_pred, extra),
                        record_indices_row,
                        record_full=record_full,
                    )
                    if record_gates
                    else _empty_recording_matrix(Vm_new)
                )
                currents_out = (
                    _record_matrix_row(
                        membrane.ionic_current_trace_matrix(Vm_new, gates_pred, extra),
                        record_indices_row,
                        record_full=record_full,
                    )
                    if record_currents
                    else _empty_recording_matrix(Vm_new)
                )
                conductances_out = (
                    _record_matrix_row(
                        membrane.conductance_trace_matrix(gates_pred, extra),
                        record_indices_row,
                        record_full=record_full,
                    )
                    if record_conductances
                    else _empty_recording_matrix(Vm_new)
                )
                states_out = (
                    _record_matrix_row(
                        membrane.membrane_state_trace_matrix(extra),
                        record_indices_row,
                        record_full=record_full,
                    )
                    if record_states
                    else _empty_recording_matrix(Vm_new)
                )
                return (Vm_new, gates_pred, *extra), (
                    output,
                    {
                        "gates": gates_out,
                        "currents": currents_out,
                        "conductances": conductances_out,
                        "states": states_out,
                    },
                )

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
            output = _record_vm_row(
                Vm_new,
                record_indices_row,
                record_full=record_full,
            )
            gates_out = (
                _record_matrix_row(
                    membrane.gate_trace_matrix(gates_new, state_new),
                    record_indices_row,
                    record_full=record_full,
                )
                if record_gates
                else _empty_recording_matrix(Vm_new)
            )
            currents_out = (
                _record_matrix_row(
                    membrane.ionic_current_trace_matrix(Vm_new, gates_new, state_new),
                    record_indices_row,
                    record_full=record_full,
                )
                if record_currents
                else _empty_recording_matrix(Vm_new)
            )
            conductances_out = (
                _record_matrix_row(
                    membrane.conductance_trace_matrix(gates_new, state_new),
                    record_indices_row,
                    record_full=record_full,
                )
                if record_conductances
                else _empty_recording_matrix(Vm_new)
            )
            states_out = (
                _record_matrix_row(
                    membrane.membrane_state_trace_matrix(state_new),
                    record_indices_row,
                    record_full=record_full,
                )
                if record_states
                else _empty_recording_matrix(Vm_new)
            )
            return (Vm_new, gates_new, *state_new), (
                output,
                {
                    "gates": gates_out,
                    "currents": currents_out,
                    "conductances": conductances_out,
                    "states": states_out,
                },
            )

        final_carry, trace = jax.lax.scan(
            step,
            (Vm0_row, gates0_row, *state0_row),
            (Iinj_mid, vstim_forcing_mid),
        )
        vm_trace, recording_trace = trace
        return final_carry[0], final_carry[1], tuple(final_carry[2:]), vm_trace, recording_trace

    state_axes = tuple(0 for _ in state0)
    record_indices_axes = 0 if jnp.asarray(record_indices).ndim == 2 else None
    return jax.vmap(
        one_batch,
        in_axes=(
            0,
            0,
            state_axes,
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
            record_indices_axes,
        ),
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
        extracellular_potential_mid_mV,
        record_indices,
    )

@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "record_full",
        "record_gates",
        "record_currents",
        "record_conductances",
        "record_states",
    ),
)
def _run_single_cable_factorized_vstim_batch_stateful_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    record_full: bool,
    record_gates: bool,
    record_currents: bool,
    record_conductances: bool,
    record_states: bool,
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
) -> tuple[Array, Array, tuple[Array, ...], Array, dict[str, Array]]:
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
        record_indices_row,
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
                output = _record_vm_row(
                    Vm_new,
                    record_indices_row,
                    record_full=record_full,
                )
                gates_out = (
                    _record_matrix_row(
                        membrane.gate_trace_matrix(gates_pred, extra),
                        record_indices_row,
                        record_full=record_full,
                    )
                    if record_gates
                    else _empty_recording_matrix(Vm_new)
                )
                currents_out = (
                    _record_matrix_row(
                        membrane.ionic_current_trace_matrix(Vm_new, gates_pred, extra),
                        record_indices_row,
                        record_full=record_full,
                    )
                    if record_currents
                    else _empty_recording_matrix(Vm_new)
                )
                conductances_out = (
                    _record_matrix_row(
                        membrane.conductance_trace_matrix(gates_pred, extra),
                        record_indices_row,
                        record_full=record_full,
                    )
                    if record_conductances
                    else _empty_recording_matrix(Vm_new)
                )
                states_out = (
                    _record_matrix_row(
                        membrane.membrane_state_trace_matrix(extra),
                        record_indices_row,
                        record_full=record_full,
                    )
                    if record_states
                    else _empty_recording_matrix(Vm_new)
                )
                return (Vm_new, gates_pred, *extra), (
                    output,
                    {
                        "gates": gates_out,
                        "currents": currents_out,
                        "conductances": conductances_out,
                        "states": states_out,
                    },
                )

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
            output = _record_vm_row(
                Vm_new,
                record_indices_row,
                record_full=record_full,
            )
            gates_out = (
                _record_matrix_row(
                    membrane.gate_trace_matrix(gates_new, state_new),
                    record_indices_row,
                    record_full=record_full,
                )
                if record_gates
                else _empty_recording_matrix(Vm_new)
            )
            currents_out = (
                _record_matrix_row(
                    membrane.ionic_current_trace_matrix(Vm_new, gates_new, state_new),
                    record_indices_row,
                    record_full=record_full,
                )
                if record_currents
                else _empty_recording_matrix(Vm_new)
            )
            conductances_out = (
                _record_matrix_row(
                    membrane.conductance_trace_matrix(gates_new, state_new),
                    record_indices_row,
                    record_full=record_full,
                )
                if record_conductances
                else _empty_recording_matrix(Vm_new)
            )
            states_out = (
                _record_matrix_row(
                    membrane.membrane_state_trace_matrix(state_new),
                    record_indices_row,
                    record_full=record_full,
                )
                if record_states
                else _empty_recording_matrix(Vm_new)
            )
            return (Vm_new, gates_new, *state_new), (
                output,
                {
                    "gates": gates_out,
                    "currents": currents_out,
                    "conductances": conductances_out,
                    "states": states_out,
                },
            )

        final_carry, trace = jax.lax.scan(
            step,
            (Vm0_row, gates0_row, *state0_row),
            (Iinj_mid, jnp.swapaxes(current_mid_row_A, 0, 1)),
        )
        vm_trace, recording_trace = trace
        return final_carry[0], final_carry[1], tuple(final_carry[2:]), vm_trace, recording_trace

    state_axes = tuple(0 for _ in state0)
    record_indices_axes = 0 if jnp.asarray(record_indices).ndim == 2 else None
    return jax.vmap(
        one_batch,
        in_axes=(
            0,
            0,
            state_axes,
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
            record_indices_axes,
        ),
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
        record_indices,
    )

@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "observer_retention",
    ),
)
def _run_single_cable_vstim_batch_observer_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    observer_retention: ObserverRetention,
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
    raster_blanking_ms: Array,
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    time_start_index: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], ThresholdObserverState]:
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
                observer_state = update_threshold_observer_state_scalar_from_tables(
                    observer_state,
                    vm_mV=Vm_new,
                    step_index=time_start_index + local_step,
                    probe_indices=raster_probe_indices_row,
                    probe_mask=raster_probe_mask_row,
                    thresholds_mV=raster_thresholds_mV,
                    blanking_ms=raster_blanking_ms,
                    dt_ms=dt_ms,
                    retention=observer_retention,
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
            observer_state = update_threshold_observer_state_scalar_from_tables(
                observer_state,
                vm_mV=Vm_new,
                step_index=time_start_index + local_step,
                probe_indices=raster_probe_indices_row,
                probe_mask=raster_probe_mask_row,
                thresholds_mV=raster_thresholds_mV,
                blanking_ms=raster_blanking_ms,
                dt_ms=dt_ms,
                retention=observer_retention,
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
        "observer_retention",
    ),
)
def _run_single_cable_factorized_vstim_batch_observer_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    observer_retention: ObserverRetention,
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
    raster_blanking_ms: Array,
    intracellular_current_density_mid: Array,
    extracellular_current_mid_A: Array,
    extracellular_forcing_footprint_mV_per_A: Array,
    time_start_index: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], ThresholdObserverState]:
    """Run one observer chunk with dense Iinj and pre-lowered factorized Vstim."""

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
        Iinj_mid,
        current_mid_row_A,
        forcing_footprint_mV_per_A,
    ):
        def step(carry, step_inputs):
            Iinj, current_A, local_step = step_inputs
            Vm, gates, observer_state, *extra = carry
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
                observer_state = update_threshold_observer_state_scalar_from_tables(
                    observer_state,
                    vm_mV=Vm_new,
                    step_index=time_start_index + local_step,
                    probe_indices=raster_probe_indices_row,
                    probe_mask=raster_probe_mask_row,
                    thresholds_mV=raster_thresholds_mV,
                blanking_ms=raster_blanking_ms,
                dt_ms=dt_ms,
                retention=observer_retention,
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
            observer_state = update_threshold_observer_state_scalar_from_tables(
                observer_state,
                vm_mV=Vm_new,
                step_index=time_start_index + local_step,
                probe_indices=raster_probe_indices_row,
                probe_mask=raster_probe_mask_row,
                thresholds_mV=raster_thresholds_mV,
            blanking_ms=raster_blanking_ms,
            dt_ms=dt_ms,
            retention=observer_retention,
            )
            return (Vm_new, gates_new, observer_state, *state_new), None

        final_carry, _ = jax.lax.scan(
            step,
            (Vm0_row, gates0_row, observer_state_row, *state0_row),
            (
                Iinj_mid,
                jnp.swapaxes(current_mid_row_A, 0, 1),
                jnp.arange(
                    Iinj_mid.shape[0],
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
        "observer_retention",
    ),
)
def _run_single_cable_factorized_vstim_batch_sparse_observer_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    observer_retention: ObserverRetention,
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
    raster_blanking_ms: Array,
    intracellular_current_density_values_mid: Array,
    intracellular_current_density_indices: Array,
    intracellular_current_density_mask: Array,
    extracellular_current_mid_A: Array,
    extracellular_forcing_footprint_mV_per_A: Array,
    time_start_index: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], ThresholdObserverState]:
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
                observer_state = update_threshold_observer_state_scalar_from_tables(
                    observer_state,
                    vm_mV=Vm_new,
                    step_index=time_start_index + local_step,
                    probe_indices=raster_probe_indices_row,
                    probe_mask=raster_probe_mask_row,
                    thresholds_mV=raster_thresholds_mV,
                blanking_ms=raster_blanking_ms,
                dt_ms=dt_ms,
                retention=observer_retention,
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
            observer_state = update_threshold_observer_state_scalar_from_tables(
                observer_state,
                vm_mV=Vm_new,
                step_index=time_start_index + local_step,
                probe_indices=raster_probe_indices_row,
                probe_mask=raster_probe_mask_row,
                thresholds_mV=raster_thresholds_mV,
            blanking_ms=raster_blanking_ms,
            dt_ms=dt_ms,
            retention=observer_retention,
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
        "has_driven_extracellular",
        "stateless_vm_only",
        "has_sparse_iinj",
        "observer_retention",
    ),
)
def _run_single_cable_shared_rank1_vstim_batch_sparse_observer_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    has_sparse_iinj: bool,
    observer_retention: ObserverRetention,
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
    raster_blanking_ms: Array,
    intracellular_current_density_values_mid: Array,
    intracellular_current_density_indices: Array,
    intracellular_current_density_mask: Array,
    extracellular_current_mid_A: Array,
    extracellular_forcing_footprint_mV_per_A: Array,
    time_start_index: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], ThresholdObserverState]:
    """Run sparse observer chunks when all rows share one extracellular current."""

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
        forcing_footprint_mV_per_A,
    ):
        safe_iinj_indices = (
            jnp.where(Iinj_mask, Iinj_indices, 0)
            if has_sparse_iinj
            else None
        )

        def step(carry, step_inputs):
            Iinj_values, current_A, local_step = step_inputs
            Vm, gates, observer_state, *extra = carry
            extra = tuple(extra)
            Iinj = (
                jnp.zeros_like(Vm).at[safe_iinj_indices].add(
                    jnp.where(Iinj_mask, Iinj_values, 0.0)
                )
                if has_sparse_iinj
                else jnp.zeros_like(Vm)
            )
            vstim_force = current_A * forcing_footprint_mV_per_A

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
                observer_state = update_threshold_observer_state_scalar_from_tables(
                    observer_state,
                    vm_mV=Vm_new,
                    step_index=time_start_index + local_step,
                    probe_indices=raster_probe_indices_row,
                    probe_mask=raster_probe_mask_row,
                    thresholds_mV=raster_thresholds_mV,
                blanking_ms=raster_blanking_ms,
                dt_ms=dt_ms,
                retention=observer_retention,
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
            observer_state = update_threshold_observer_state_scalar_from_tables(
                observer_state,
                vm_mV=Vm_new,
                step_index=time_start_index + local_step,
                probe_indices=raster_probe_indices_row,
                probe_mask=raster_probe_mask_row,
                thresholds_mV=raster_thresholds_mV,
            blanking_ms=raster_blanking_ms,
            dt_ms=dt_ms,
            retention=observer_retention,
            )
            return (Vm_new, gates_new, observer_state, *state_new), None

        final_carry, _ = jax.lax.scan(
            step,
            (Vm0_row, gates0_row, observer_state_row, *state0_row),
            (
                Iinj_values_mid,
                extracellular_current_mid_A,
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
        extracellular_forcing_footprint_mV_per_A,
    )

@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "stateless_vm_only",
        "observer_retention",
    ),
)
def _run_single_cable_zero_vstim_batch_sparse_observer_scan(
    *,
    backend,
    membrane,
    stateless_vm_only: bool,
    observer_retention: ObserverRetention,
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
    raster_blanking_ms: Array,
    intracellular_current_density_values_mid: Array,
    intracellular_current_density_indices: Array,
    intracellular_current_density_mask: Array,
    time_start_index: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], ThresholdObserverState]:
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
                observer_state = update_threshold_observer_state_scalar_from_tables(
                    observer_state,
                    vm_mV=Vm_new,
                    step_index=time_start_index + local_step,
                    probe_indices=raster_probe_indices_row,
                    probe_mask=raster_probe_mask_row,
                    thresholds_mV=raster_thresholds_mV,
                blanking_ms=raster_blanking_ms,
                dt_ms=dt_ms,
                retention=observer_retention,
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
            observer_state = update_threshold_observer_state_scalar_from_tables(
                observer_state,
                vm_mV=Vm_new,
                step_index=time_start_index + local_step,
                probe_indices=raster_probe_indices_row,
                probe_mask=raster_probe_mask_row,
                thresholds_mV=raster_thresholds_mV,
            blanking_ms=raster_blanking_ms,
            dt_ms=dt_ms,
            retention=observer_retention,
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
