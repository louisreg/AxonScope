from __future__ import annotations

from dataclasses import dataclass

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
        Cm = jnp.asarray(self.Cm_uF_cm2, dtype=dtype_local)
        I_bg = membrane_runtime.background_current
        state0 = membrane_runtime.state0

        def step(carry, n: int):
            Vm, gates, *extra = carry
            extra = tuple(extra)
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

            diffusion = apply_diffusion_operator(Vm, lower, diag, upper)
            rhs = Vm + dtype_local(0.5) * dt * diffusion + (dt / Cm) * (
                Iinj
                - step_plan_pred.total_outward_current
                - step_plan_pred.correction_current
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

        def solve_vi_vperi(
            Vi: Array,
            Ve: Array,
            gates_new: Array,
            Iinj_den: Array,
            I_outward_den: Array,
            I_corr_den: Array,
            Vext_mV: Array,
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
                + (Cx_abs / dt + Gx_abs) * Vext_mV
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
                Vext_mV=Vext,
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
