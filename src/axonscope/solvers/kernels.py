from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from .common import (
    Array,
    apply_diffusion_operator,
    build_cn_tridiagonal,
    solve_block_tridiagonal_2x2_scalar,
)
from .recording import observable_matrices, package_recordings
from .runtime import SolverRuntime


@dataclass(frozen=True)
class KernelResult:
    """Raw solver-kernel output before packaging into a public SimResult."""

    Vm: Array
    t: Array
    diagnostics: dict[str, Array] | None = None
    recordings: dict[str, dict[str, Array]] | None = None


def _precomputed_step_pair(
    values_mid: Array,
    initial_previous: Array,
    n: int,
) -> tuple[Array, Array]:
    current = values_mid[n]
    previous_idx = jnp.maximum(n - 1, 0)
    previous = jnp.where(n == 0, initial_previous, values_mid[previous_idx])
    return current, previous


def _intracellular_current_at(runtime: SolverRuntime, n: int, t_mid: Array) -> Array:
    samples = runtime.stimulation.intracellular_current_density_mid
    if samples is not None:
        return samples[n]
    return runtime.stimulation.intracellular_current_density(t_mid)


def _scalar_like(value: Array, scalar: float) -> Array:
    return jnp.asarray(scalar, dtype=jnp.asarray(value).dtype)


@partial(jax.jit, static_argnames=("backend", "membrane", "stateless_vm_only"))
def _run_single_cable_vm_scan(
    *,
    backend,
    membrane,
    stateless_vm_only: bool,
    lower: Array,
    diag: Array,
    upper: Array,
    dl: Array,
    d: Array,
    du: Array,
    Cm_uF_cm2: Array,
    I_background: Array,
    Vm0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    intracellular_current_density_mid: Array,
    step_indices: Array,
    dt_ms: Array,
) -> Array:
    """Jitted Vm-only single-cable scan.

    This is the performance path used by the public solver when no optional
    diagnostics or observables are requested. It consumes only prepared solver
    arrays plus static membrane/backend methods, which keeps the critical time
    loop close to the future batch-kernel shape.
    """

    def step(carry, n: int):
        Vm, gates, *extra = carry
        extra = tuple(extra)

        gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt_ms)
        Iion_pred = backend.currents(V_mV=Vm, gates=gates_pred)
        step_plan_pred = membrane.prepare_membrane_step(
            V_mV=Vm,
            gates_prev=gates,
            gates_new=gates_pred,
            state=extra,
            dt=dt_ms,
            I_ion=Iion_pred,
            I_background=I_background,
        )

        diffusion = apply_diffusion_operator(Vm, lower, diag, upper)
        rhs = Vm + _scalar_like(dt_ms, 0.5) * dt_ms * diffusion + (dt_ms / Cm_uF_cm2) * (
            intracellular_current_density_mid[n]
            - step_plan_pred.total_outward_current
            - step_plan_pred.correction_current
        )
        Vm_new = jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:, None])[:, 0]

        if stateless_vm_only:
            return (Vm_new, gates_pred, *extra), Vm_new

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
            I_background=I_background,
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
        return (Vm_new, gates_new, *state_new), Vm_new

    init_carry = (Vm0_mV, gates0, *state0)
    _, Vm_trace = jax.lax.scan(step, init_carry, step_indices)
    return Vm_trace


@partial(
    jax.jit,
    static_argnames=("backend", "membrane", "has_driven_extracellular", "stateless_vm_only"),
)
def _run_single_cable_vstim_vm_scan(
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
    extracellular_diffusion_forcing_mid: Array,
    dt_ms: Array,
) -> Array:
    """Jitted Vm-only single-cable scan with imposed extracellular forcing.

    The imposed extracellular field is treated as a known mid-step potential:
    Vi = Vm + Vstim. This is the scalar limit of the current double-cable
    extracellular solve, with implicit axial diffusion and the known axial
    drive ``dt * L(Vstim_mid)`` on the RHS.
    """

    def step(carry, step_inputs):
        Iinj, vstim_force = step_inputs
        Vm, gates, *extra = carry
        extra = tuple(extra)

        gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt_ms)
        if stateless_vm_only:
            linearization_gates = gates if has_driven_extracellular else gates_pred
            explicit_outward_current = I_background
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
                I_background=I_background,
            )
            linearization_gates = step_plan_pred.linearization_gates
            if has_driven_extracellular:
                linearization_gates = gates
            explicit_outward_current = step_plan_pred.explicit_outward_current
            correction_current = step_plan_pred.correction_current

        Gm, GE = backend.membrane_conductance_terms(linearization_gates)
        d = d_static + (dt_ms / Cm_uF_cm2) * Gm
        rhs = (
            Vm
            + dt_ms * vstim_force
            + (dt_ms / Cm_uF_cm2)
            * (
                GE
                + Iinj
                - explicit_outward_current
                - correction_current
            )
        )
        Vm_new = jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:, None])[:, 0]

        if stateless_vm_only:
            return (Vm_new, gates_pred, *extra), Vm_new

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
            I_background=I_background,
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
        return (Vm_new, gates_new, *state_new), Vm_new

    init_carry = (Vm0_mV, gates0, *state0)
    _, Vm_trace = jax.lax.scan(
        step,
        init_carry,
        (intracellular_current_density_mid, extracellular_diffusion_forcing_mid),
    )
    return Vm_trace


@partial(
    jax.jit,
    static_argnames=("backend", "membrane", "has_driven_extracellular", "stateless_vm_only"),
)
def _run_double_cable_vm_scan(
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
    """Jitted Vm-only double-cable scan for the optimized extracellular path."""
    cm_over_dt = Cm_abs / dt_ms
    cx_over_dt = Cx_abs / dt_ms
    vext_previous_mV = jnp.concatenate(
        [
            extracellular_potential_initial_previous_mV[None, :],
            extracellular_potential_mid_mV[:-1],
        ],
        axis=0,
    )
    extracellular_rhs_drive = (
        (cx_over_dt + Gx_abs)[None, :] * extracellular_potential_mid_mV
        - cx_over_dt[None, :] * vext_previous_mV
    )
    intracellular_current_abs_mid = intracellular_current_density_mid * area_cm2[None, :]
    off_i = -Gax_i
    off_e = -Gax_e

    def solve_vi_vperi(
        Vi: Array,
        Ve: Array,
        gates_new: Array,
        Iinj_abs: Array,
        I_outward_den: Array,
        I_corr_den: Array,
        extracellular_drive_abs: Array,
    ) -> tuple[Array, Array]:
        Gm_den, GE_den = backend.membrane_conductance_terms(gates_new)
        Gm_abs = Gm_den * area_cm2
        GE_abs = GE_den * area_cm2

        I_outward_abs = I_outward_den * area_cm2
        I_corr_abs = I_corr_den * area_cm2
        Vm = Vi - Ve

        a00 = cm_over_dt + Gm_abs + left_i + right_i
        a01 = -(cm_over_dt + Gm_abs)
        rhs0 = (
            cm_over_dt * Vm
            + GE_abs
            + Iinj_abs
            - I_outward_abs
            - I_corr_abs
        )

        a10 = a01
        a11 = cm_over_dt + Gm_abs + cx_over_dt + Gx_abs + left_e + right_e
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

        gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt_ms)

        if stateless_vm_only:
            linearization_gates = gates if has_driven_extracellular else gates_pred
            explicit_outward_current = I_background
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
                I_background=I_background,
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
            return (Vi_new, Ve_new, gates_pred, *extra), Vm_new

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
            I_background=I_background,
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
        return (Vi_new, Ve_new, gates_new, *state_new), Vm_new

    init_carry = (Vi0_mV, Ve0_mV, gates0, *state0)
    _, Vm_trace = jax.lax.scan(
        step,
        init_carry,
        (intracellular_current_abs_mid, extracellular_rhs_drive),
    )
    return Vm_trace


def _pack_scan_result(
    runtime: SolverRuntime,
    out,
    *,
    record_diagnostics: bool,
    record_observables: bool,
) -> KernelResult:
    membrane_runtime = runtime.membrane
    diagnostic_names = membrane_runtime.diagnostic_names
    observable_names = membrane_runtime.observable_names

    if record_observables and record_diagnostics and diagnostic_names:
        recordings = package_recordings(
            observable_names,
            out[1],
            out[2],
            out[3],
            out[4],
        )
        diagnostics = {
            name: values
            for name, values in zip(diagnostic_names, out[5:], strict=False)
        }
        return KernelResult(
            Vm=out[0],
            t=runtime.grid.t_vec_ms,
            diagnostics=diagnostics,
            recordings=recordings,
        )

    if record_observables:
        recordings = package_recordings(
            observable_names,
            out[1],
            out[2],
            out[3],
            out[4],
        )
        return KernelResult(Vm=out[0], t=runtime.grid.t_vec_ms, recordings=recordings)

    if record_diagnostics and diagnostic_names:
        diagnostics = {
            name: values
            for name, values in zip(diagnostic_names, out[1:], strict=False)
        }
        return KernelResult(Vm=out[0], t=runtime.grid.t_vec_ms, diagnostics=diagnostics)

    return KernelResult(Vm=out, t=runtime.grid.t_vec_ms)


@dataclass(frozen=True)
class SingleCableKernel:
    """Optimized scalar Crank-Nicolson kernel for one cable state Vm."""

    runtime: SolverRuntime
    Cm_uF_cm2: Array

    def run(
        self,
        *,
        record_diagnostics: bool = False,
        record_observables: bool = False,
    ) -> KernelResult:
        runtime = self.runtime
        membrane_runtime = runtime.membrane
        grid = runtime.grid
        cable = runtime.cable

        backend = membrane_runtime.backend
        membrane = membrane_runtime.membrane
        diagnostic_names = membrane_runtime.diagnostic_names
        dtype_local = membrane_runtime.dtype
        dt = dtype_local(grid.dt_ms)

        lower, diag, upper = cable.lower, cable.diag, cable.upper
        dl, d, du = build_cn_tridiagonal(lower, diag, upper, grid.dt_ms, dtype_local)
        dt_arr = jnp.asarray(grid.dt_ms, dtype=dtype_local)
        dl_vstim = -dt_arr * lower
        d_vstim_static = jnp.ones_like(diag) - dt_arr * diag
        du_vstim = -dt_arr * upper
        Cm = jnp.asarray(self.Cm_uF_cm2, dtype=dtype_local)
        I_bg = membrane_runtime.background_current
        state0 = membrane_runtime.state0
        Iinj_mid = runtime.stimulation.intracellular_current_density_mid
        vext_mid = runtime.stimulation.extracellular_potential_mid_mV
        has_imposed_vstim = vext_mid is not None

        if (
            not record_observables
            and not record_diagnostics
            and Iinj_mid is not None
            and has_imposed_vstim
        ):
            vstim_forcing_mid = jax.vmap(
                lambda values: apply_diffusion_operator(values, lower, diag, upper)
            )(vext_mid)
            out = _run_single_cable_vstim_vm_scan(
                backend=backend,
                membrane=membrane,
                has_driven_extracellular=runtime.stimulation.has_driven_extracellular,
                stateless_vm_only=bool(membrane.supports_stateless_vm_only_fast_path()),
                lower=lower,
                diag=diag,
                upper=upper,
                dl=dl_vstim,
                d_static=d_vstim_static,
                du=du_vstim,
                Cm_uF_cm2=Cm,
                I_background=I_bg,
                Vm0_mV=membrane_runtime.Vm0_mV,
                gates0=membrane_runtime.gates0,
                state0=state0,
                intracellular_current_density_mid=Iinj_mid,
                extracellular_diffusion_forcing_mid=vstim_forcing_mid,
                dt_ms=dt_arr,
            )
            return KernelResult(Vm=out, t=runtime.grid.t_vec_ms)

        if (
            not record_observables
            and not record_diagnostics
            and Iinj_mid is not None
        ):
            out = _run_single_cable_vm_scan(
                backend=backend,
                membrane=membrane,
                stateless_vm_only=bool(membrane.supports_stateless_vm_only_fast_path()),
                lower=lower,
                diag=diag,
                upper=upper,
                dl=dl,
                d=d,
                du=du,
                Cm_uF_cm2=Cm,
                I_background=I_bg,
                Vm0_mV=membrane_runtime.Vm0_mV,
                gates0=membrane_runtime.gates0,
                state0=state0,
                intracellular_current_density_mid=Iinj_mid,
                step_indices=jnp.arange(grid.Nt, dtype=jnp.int32),
                dt_ms=jnp.asarray(grid.dt_ms, dtype=dtype_local),
            )
            return KernelResult(Vm=out, t=runtime.grid.t_vec_ms)

        def step(carry, n: int):
            Vm, gates, *extra = carry
            extra = tuple(extra)
            t_mid = dtype_local(n) * dt + dtype_local(0.5) * dt

            gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=grid.dt_ms)
            Iion_pred = backend.currents(V_mV=Vm, gates=gates_pred)
            Iinj = _intracellular_current_at(runtime, n, t_mid)
            vstim_forcing = (
                apply_diffusion_operator(vext_mid[n], lower, diag, upper)
                if has_imposed_vstim
                else jnp.zeros_like(Vm)
            )
            step_plan_pred = membrane.prepare_membrane_step(
                V_mV=Vm,
                gates_prev=gates,
                gates_new=gates_pred,
                state=extra,
                dt=grid.dt_ms,
                I_ion=Iion_pred,
                I_background=I_bg,
            )

            if has_imposed_vstim:
                linearization_gates = step_plan_pred.linearization_gates
                if runtime.stimulation.has_driven_extracellular:
                    linearization_gates = gates
                Gm, GE = backend.membrane_conductance_terms(linearization_gates)
                d_step = d_vstim_static + (dt / Cm) * Gm
                rhs = (
                    Vm
                    + dt * vstim_forcing
                    + (dt / Cm)
                    * (
                        GE
                        + Iinj
                        - step_plan_pred.explicit_outward_current
                        - step_plan_pred.correction_current
                    )
                )
                Vm_new = jax.lax.linalg.tridiagonal_solve(
                    dl_vstim,
                    d_step,
                    du_vstim,
                    rhs[:, None],
                )[:, 0]
            else:
                diffusion = apply_diffusion_operator(Vm, lower, diag, upper)
                rhs = (
                    Vm
                    + dtype_local(0.5) * dt * diffusion
                    + (dt / Cm)
                    * (
                        Iinj
                        - step_plan_pred.total_outward_current
                        - step_plan_pred.correction_current
                    )
                )
                Vm_new = jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:, None])[:, 0]

            gates_new = membrane.final_gate_update(
                gates_prev=gates,
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                dt=grid.dt_ms,
                gates_predictor=gates_pred,
            )
            Iion_new = backend.currents(V_mV=Vm_new, gates=gates_new)
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state=extra,
                dt=grid.dt_ms,
                I_ion=Iion_new,
                I_background=I_bg,
            )
            state_new = membrane.finalize_membrane_step(
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=extra,
                step_plan=step_plan,
                dt=grid.dt_ms,
            )
            carry_out = (Vm_new, gates_new, *state_new)

            if record_observables:
                gate_obs, current_obs, conductance_obs, state_obs = observable_matrices(
                    membrane, Vm_new, gates_new, state_new
                )
                if record_diagnostics and diagnostic_names:
                    diag_vals = membrane.compute_step_diagnostics(
                        V_mV_prev=Vm,
                        V_mV_new=Vm_new,
                        gates_prev=gates,
                        gates_new=gates_new,
                        state_prev=extra,
                        state_new=state_new,
                        step_plan=step_plan,
                        I_ion=Iion_new,
                    )
                    return carry_out, (
                        Vm_new,
                        gate_obs,
                        current_obs,
                        conductance_obs,
                        state_obs,
                        *diag_vals,
                    )
                return carry_out, (Vm_new, gate_obs, current_obs, conductance_obs, state_obs)

            if record_diagnostics and diagnostic_names:
                diag_vals = membrane.compute_step_diagnostics(
                    V_mV_prev=Vm,
                    V_mV_new=Vm_new,
                    gates_prev=gates,
                    gates_new=gates_new,
                    state_prev=extra,
                    state_new=state_new,
                    step_plan=step_plan,
                    I_ion=Iion_new,
                )
                return carry_out, (Vm_new, *diag_vals)

            return carry_out, Vm_new

        init_carry = (
            membrane_runtime.Vm0_mV,
            membrane_runtime.gates0,
            *state0,
        )
        _, out = jax.lax.scan(step, init_carry, jnp.arange(grid.Nt))
        return _pack_scan_result(
            runtime,
            out,
            record_diagnostics=record_diagnostics,
            record_observables=record_observables,
        )


@dataclass(frozen=True)
class DoubleCableKernel:
    """Optimized full double-cable Crank-Nicolson kernel for Vi/Vperi."""

    runtime: SolverRuntime
    Veinit_mV: float = 0.0

    def run(
        self,
        *,
        record_diagnostics: bool = False,
        record_observables: bool = False,
    ) -> KernelResult:
        runtime = self.runtime
        membrane_runtime = runtime.membrane
        grid = runtime.grid
        cable = runtime.cable
        stimulation = runtime.stimulation
        extracellular = runtime.extracellular
        if extracellular is None:
            raise ValueError("DoubleCableKernel requires extracellular runtime arrays.")

        backend = membrane_runtime.backend
        membrane = membrane_runtime.membrane
        diagnostic_names = membrane_runtime.diagnostic_names
        dtype_local = membrane_runtime.dtype
        dt = dtype_local(grid.dt_ms)
        Nx = membrane_runtime.Nx

        Vm0 = membrane_runtime.Vm0_mV
        Ve0 = jnp.full((Nx,), dtype_local(self.Veinit_mV), dtype=dtype_local)
        Vi0 = Vm0 + Ve0
        gates0 = membrane_runtime.gates0
        state0 = membrane_runtime.state0
        I_bg = membrane_runtime.background_current

        area = cable.area_cm2
        Cm_abs = extracellular.Cm_abs
        Cx_abs = extracellular.Cx_abs
        Gx_abs = extracellular.Gx_abs
        Gax_e = extracellular.Gax_e
        Gax_i = extracellular.Gax_i
        left_i = extracellular.left_i
        right_i = extracellular.right_i
        left_e = extracellular.left_e
        right_e = extracellular.right_e

        vext_fun = stimulation.extracellular_potential_mV
        vext_mid_all = stimulation.extracellular_potential_mid_mV
        vext_initial_previous = stimulation.extracellular_potential_initial_previous_mV
        has_driven_extracellular = stimulation.has_driven_extracellular
        Iinj_mid = stimulation.intracellular_current_density_mid

        if (
            not record_observables
            and not record_diagnostics
            and Iinj_mid is not None
            and vext_mid_all is not None
            and vext_initial_previous is not None
        ):
            out = _run_double_cable_vm_scan(
                backend=backend,
                membrane=membrane,
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=bool(membrane.supports_stateless_vm_only_fast_path()),
                Vi0_mV=Vi0,
                Ve0_mV=Ve0,
                gates0=gates0,
                state0=state0,
                area_cm2=area,
                Cm_abs=Cm_abs,
                Cx_abs=Cx_abs,
                Gx_abs=Gx_abs,
                Gax_e=Gax_e,
                Gax_i=Gax_i,
                left_i=left_i,
                right_i=right_i,
                left_e=left_e,
                right_e=right_e,
                I_background=I_bg,
                intracellular_current_density_mid=Iinj_mid,
                extracellular_potential_mid_mV=vext_mid_all,
                extracellular_potential_initial_previous_mV=vext_initial_previous,
                dt_ms=jnp.asarray(grid.dt_ms, dtype=dtype_local),
            )
            return KernelResult(Vm=out, t=runtime.grid.t_vec_ms)

        def solve_vi_vperi(
            Vi: Array,
            Ve: Array,
            gates_new: Array,
            Iinj_den: Array,
            I_outward_den: Array,
            I_corr_den: Array,
            extracellular_potential_mV: Array,
            Vext_old_mV: Array,
        ) -> tuple[Array, Array]:
            Gm_den, GE_den = backend.membrane_conductance_terms(gates_new)
            Gm_abs = Gm_den * area
            GE_abs = GE_den * area

            Iinj_abs = Iinj_den * area
            I_outward_abs = I_outward_den * area
            I_corr_abs = I_corr_den * area
            Vm = Vi - Ve

            a00 = Cm_abs / dt + Gm_abs + left_i + right_i
            a01 = -(Cm_abs / dt + Gm_abs)
            rhs0 = (
                (Cm_abs / dt) * Vm
                + GE_abs
                + Iinj_abs
                - I_outward_abs
                - I_corr_abs
            )

            a10 = a01
            a11 = Cm_abs / dt + Gm_abs + Cx_abs / dt + Gx_abs + left_e + right_e
            rhs1 = (
                -(Cm_abs / dt) * Vm
                - GE_abs
                + (Cx_abs / dt) * Ve
                - (Cx_abs / dt) * Vext_old_mV
                + (Cx_abs / dt + Gx_abs) * extracellular_potential_mV
                + I_outward_abs
                + I_corr_abs
            )

            return solve_block_tridiagonal_2x2_scalar(
                a00,
                a01,
                a10,
                a11,
                -Gax_i,
                -Gax_e,
                rhs0,
                rhs1,
            )

        def extracellular_drive_at(n: int, t_mid: Array) -> tuple[Array, Array]:
            if vext_mid_all is not None and vext_initial_previous is not None:
                return _precomputed_step_pair(vext_mid_all, vext_initial_previous, n)
            return vext_fun(t_mid), vext_fun(t_mid - dt)

        def step(carry, n: int):
            Vi, Ve, gates, *extra = carry
            extra = tuple(extra)
            Vm = Vi - Ve
            t_mid = dtype_local(n) * dt + dtype_local(0.5) * dt

            gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=grid.dt_ms)
            Iion_pred = backend.currents(V_mV=Vm, gates=gates_pred)
            Iinj = _intracellular_current_at(runtime, n, t_mid)
            step_plan_pred = membrane.prepare_membrane_step(
                V_mV=Vm,
                gates_prev=gates,
                gates_new=gates_pred,
                state=extra,
                dt=grid.dt_ms,
                I_ion=Iion_pred,
                I_background=I_bg,
            )
            linearization_gates = step_plan_pred.linearization_gates
            if has_driven_extracellular:
                linearization_gates = gates

            Vext, Vext_old = extracellular_drive_at(n, t_mid)
            Vi_new, Ve_new = solve_vi_vperi(
                Vi=Vi,
                Ve=Ve,
                gates_new=linearization_gates,
                Iinj_den=Iinj,
                I_outward_den=step_plan_pred.explicit_outward_current,
                I_corr_den=step_plan_pred.correction_current,
                extracellular_potential_mV=Vext,
                Vext_old_mV=Vext_old,
            )
            Vm_new = Vi_new - Ve_new

            gates_new = membrane.final_gate_update(
                gates_prev=gates,
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                dt=grid.dt_ms,
                gates_predictor=gates_pred,
            )
            Iion_new = backend.currents(V_mV=Vm_new, gates=gates_new)
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state=extra,
                dt=grid.dt_ms,
                I_ion=Iion_new,
                I_background=I_bg,
            )
            state_new = membrane.finalize_membrane_step(
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=extra,
                step_plan=step_plan,
                dt=grid.dt_ms,
            )
            carry_out = (Vi_new, Ve_new, gates_new, *state_new)

            if record_observables:
                gate_obs, current_obs, conductance_obs, state_obs = observable_matrices(
                    membrane, Vm_new, gates_new, state_new
                )
                if record_diagnostics and diagnostic_names:
                    diag_vals = membrane.compute_step_diagnostics(
                        V_mV_prev=Vm,
                        V_mV_new=Vm_new,
                        gates_prev=gates,
                        gates_new=gates_new,
                        state_prev=extra,
                        state_new=state_new,
                        step_plan=step_plan,
                        I_ion=Iion_new,
                    )
                    return carry_out, (
                        Vm_new,
                        gate_obs,
                        current_obs,
                        conductance_obs,
                        state_obs,
                        *diag_vals,
                    )
                return carry_out, (Vm_new, gate_obs, current_obs, conductance_obs, state_obs)

            if record_diagnostics and diagnostic_names:
                diag_vals = membrane.compute_step_diagnostics(
                    V_mV_prev=Vm,
                    V_mV_new=Vm_new,
                    gates_prev=gates,
                    gates_new=gates_new,
                    state_prev=extra,
                    state_new=state_new,
                    step_plan=step_plan,
                    I_ion=Iion_new,
                )
                return carry_out, (Vm_new, *diag_vals)

            return carry_out, Vm_new

        init_carry = (Vi0, Ve0, gates0, *state0)
        _, out = jax.lax.scan(step, init_carry, jnp.arange(grid.Nt))
        return _pack_scan_result(
            runtime,
            out,
            record_diagnostics=record_diagnostics,
            record_observables=record_observables,
        )
