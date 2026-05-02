"""
crank_nicholson.py
==================

Numerical solvers for the cable equation used by AxonScope.

This module contains the official optimized Crank-Nicholson solver.
Reference and prototype variants live in `axonscope.solvers.experimental`.

All solvers are written to be JAX-friendly:
- they operate on JAX arrays (`jnp.ndarray`),
- loop over time using `jax.lax.scan` so the entire time loop can be jitted,
- avoid Python-side mutation of arrays,
- pre-extract functional objects from the axon/ion channel model when appropriate,
- GPU-ready

Development notes:
    - Optimize heterogeneous multi-compartment membrane backends
    - Provide saving methods with filtering (eg. Iion, gating variables...)
    - Provide chunk simulation
    - Provide batching capabilities
    - Add "on-the-fly" processing capabilities (eg rasterizing, filtering,...)
    - and other things

Author: l.regnacq
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from axonscope.axons.base import AxonBase
from axonscope.simresult import SimResult

from .base import Solver
from .common import solve_block_tridiagonal_2x2_scalar
from .kernels import DoubleCableKernel, SingleCableKernel
from .recording import observable_matrices, package_recordings
from .runtime import prepare_solver_runtime


def _precomputed_vext_step(
    vext_mid_all: jnp.ndarray,
    vext_initial_previous: jnp.ndarray,
    n: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    Vext = vext_mid_all[n]
    previous_idx = jnp.maximum(n - 1, 0)
    Vext_old = jnp.where(n == 0, vext_initial_previous, vext_mid_all[previous_idx])
    return Vext, Vext_old


def _solve_with_extracellular_generic(
    axon: AxonBase,
    tsim: float,
    dt: float,
    record_diagnostics: bool = False,
    record_observables: bool = False,
) -> SimResult:
    """Generic extracellular (Vi/Ve) integration shared by the solver family."""
    runtime = prepare_solver_runtime(
        axon,
        tsim,
        dt,
        include_extracellular=True,
        include_area=True,
    )
    membrane_runtime = runtime.membrane
    grid = runtime.grid
    cable = runtime.cable
    stimulation = runtime.stimulation
    extracellular = runtime.extracellular
    assert extracellular is not None

    Nx: int = membrane_runtime.Nx
    Nt: int = grid.Nt
    backend = membrane_runtime.backend
    membrane = membrane_runtime.membrane
    diagnostic_names = membrane_runtime.diagnostic_names
    observable_names = membrane_runtime.observable_names
    dtype_local = membrane_runtime.dtype

    Vm0: jnp.ndarray = membrane_runtime.Vm0_mV
    Ve0: jnp.ndarray = jnp.full((Nx,), dtype_local(getattr(axon, "Veinit", 0.0)), dtype=dtype_local)
    Vi0: jnp.ndarray = Vm0 + Ve0
    gates0: jnp.ndarray = membrane_runtime.gates0
    state0 = membrane_runtime.state0

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

    inj_fun = stimulation.intracellular_current_density
    vext_fun = stimulation.extracellular_potential_mV
    vext_mid_all = stimulation.extracellular_potential_mid_mV
    vext_initial_previous = stimulation.extracellular_potential_initial_previous_mV
    I_bg = membrane_runtime.background_current

    def step(carry, n):
        Vi_old, Ve_old, gates, *state_prev = carry
        state_prev = tuple(state_prev)
        Vm_old = Vi_old - Ve_old
        t_mid: float = dtype_local(n) * dt + dt / 2.0

        Iinj_abs = inj_fun(t_mid) * area
        if vext_mid_all is not None and vext_initial_previous is not None:
            Vext, Vext_old = _precomputed_vext_step(
                vext_mid_all,
                vext_initial_previous,
                n,
            )
        else:
            Vext = vext_fun(t_mid)
            Vext_old = vext_fun(t_mid - dt)

        def one_linear_solve(
            linearization_gates: jnp.ndarray,
            outward_current: jnp.ndarray,
            correction_current: jnp.ndarray,
        ):
            Gm_den, GE_den = backend.membrane_conductance_terms(linearization_gates)
            Gm_abs = Gm_den * area
            GE_abs = GE_den * area
            I_outward_abs = outward_current * area
            I_corr_abs = correction_current * area

            a00 = Cm_abs / dt + Gm_abs + left_i + right_i
            a01 = -(Cm_abs / dt + Gm_abs)
            rhs0 = (Cm_abs / dt) * Vm_old + GE_abs + Iinj_abs - I_outward_abs - I_corr_abs

            a10 = a01
            a11 = Cm_abs / dt + Gm_abs + Cx_abs / dt + Gx_abs + left_e + right_e
            rhs1 = (
                -(Cm_abs / dt) * Vm_old
                - GE_abs
                + (Cx_abs / dt) * Ve_old
                - (Cx_abs / dt) * Vext_old
                + (Cx_abs / dt + Gx_abs) * Vext
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

        def iter_body(_, state):
            Vm_guess, gates_last, step_plan_last, Vi_last, Ve_last = state
            gates_new = backend.cn_gate_update(g_prev=gates, V_mV=Vm_guess, dt=dt)
            Iion = backend.currents(V_mV=Vm_guess, gates=gates_new)
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_guess,
                gates_prev=gates,
                gates_new=gates_new,
                state=state_prev,
                dt=dt,
                I_ion=Iion,
                I_background=I_bg,
            )
            Vi_new, Ve_new = one_linear_solve(
                step_plan.linearization_gates,
                step_plan.explicit_outward_current,
                step_plan.correction_current,
            )
            Vm_new = Vi_new - Ve_new
            return Vm_new, gates_new, step_plan, Vi_new, Ve_new

        Vm_init = Vm_old
        Vi_init = Vi_old
        Ve_init = Ve_old
        gates_init = backend.cn_gate_update(g_prev=gates, V_mV=Vm_old, dt=dt)
        Iion_init = backend.currents(V_mV=Vm_old, gates=gates_init)
        step_plan_init = membrane.prepare_membrane_step(
            V_mV=Vm_old,
            gates_prev=gates,
            gates_new=gates_init,
            state=state_prev,
            dt=dt,
            I_ion=Iion_init,
            I_background=I_bg,
        )
        Vm_new, gates_new, step_plan_new, Vi_new, Ve_new = jax.lax.fori_loop(
            0, 3, iter_body, (Vm_init, gates_init, step_plan_init, Vi_init, Ve_init)
        )
        state_new = membrane.finalize_membrane_step(
            V_mV_prev=Vm_old,
            V_mV_new=Vm_new,
            gates_prev=gates,
            gates_new=gates_new,
            state_prev=state_prev,
            step_plan=step_plan_new,
            dt=dt,
        )
        carry_out = (Vi_new, Ve_new, gates_new, *state_new)
        if record_observables:
            gate_obs, current_obs, conductance_obs, state_obs = observable_matrices(
                membrane, Vm_new, gates_new, state_new
            )
            if record_diagnostics and diagnostic_names:
                diag_vals = membrane.compute_step_diagnostics(
                    V_mV_prev=Vm_old,
                    V_mV_new=Vm_new,
                    gates_prev=gates,
                    gates_new=gates_new,
                    state_prev=state_prev,
                    state_new=state_new,
                    step_plan=step_plan_new,
                    I_ion=backend.currents(V_mV=Vm_old, gates=gates_new),
                )
                return carry_out, (Vm_new, gate_obs, current_obs, conductance_obs, state_obs, *diag_vals)
            return carry_out, (Vm_new, gate_obs, current_obs, conductance_obs, state_obs)

        if record_diagnostics and diagnostic_names:
            diag_vals = membrane.compute_step_diagnostics(
                V_mV_prev=Vm_old,
                V_mV_new=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=state_prev,
                state_new=state_new,
                step_plan=step_plan_new,
                I_ion=backend.currents(V_mV=Vm_old, gates=gates_new),
            )
            return carry_out, (Vm_new, *diag_vals)
        return carry_out, Vm_new

    init_carry = (Vi0, Ve0, gates0, *state0)
    t_vec: jnp.ndarray = grid.t_vec_ms
    if record_observables and record_diagnostics and diagnostic_names:
        _, out = jax.lax.scan(step, init_carry, jnp.arange(Nt))
        V_all = out[0]
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
        return SimResult(axon, V_all, t_vec, diagnostics=diagnostics, recordings=recordings)

    if record_observables:
        _, out = jax.lax.scan(step, init_carry, jnp.arange(Nt))
        V_all = out[0]
        recordings = package_recordings(
            observable_names,
            out[1],
            out[2],
            out[3],
            out[4],
        )
        return SimResult(axon, V_all, t_vec, recordings=recordings)

    if record_diagnostics and diagnostic_names:
        _, diag_out = jax.lax.scan(step, init_carry, jnp.arange(Nt))
        V_all = diag_out[0]
        diagnostics = {
            name: values
            for name, values in zip(diagnostic_names, diag_out[1:], strict=False)
        }
        return SimResult(axon, V_all, t_vec, diagnostics=diagnostics)

    _, V_all = jax.lax.scan(step, init_carry, jnp.arange(Nt))
    return SimResult(axon, V_all, t_vec)


def _maybe_solve_with_extracellular_generic(
    axon: AxonBase,
    tsim: float,
    dt: float,
    record_diagnostics: bool = False,
    record_observables: bool = False,
) -> Optional[SimResult]:
    if bool(getattr(axon, "use_extracellular", False)):
        return _solve_with_extracellular_generic(
            axon,
            tsim,
            dt,
            record_diagnostics=record_diagnostics,
            record_observables=record_observables,
        )
    return None


# -----------------------------------------------------------------------------
# Optimized Crank–Nicolson using tridiagonal solver
# -----------------------------------------------------------------------------
class CrankNicholson(Solver):
    """
    Optimized Crank–Nicolson (Hines) solver for the 1D cable equation.

    Description
    -----------
    This solver implements the **Hines tridiagonal method**, the algorithm used
    internally by the NEURON simulator for integrating multi-compartment cable
    models. It is the optimized form of the Crank–Nicolson (CN) scheme, where
    the axial diffusion operator produces a strictly tridiagonal matrix.  
    Instead of forming the dense matrix `(I - (dt/2) L)`, we directly use its
    three diagonals and solve the system using `jax.lax.linalg.tridiagonal_solve`,
    which is memory-efficient, compute-efficient, highly JIT-friendly and GPU-ready. 

    The CN discretization of the cable equation reads:

        (I - (dt/2) L) · V_{n+1}
            = (I + (dt/2) L) · V_n  +  dt * (I_inj - I_ion)

    where L is the discrete Laplacian.  
    Because L is tridiagonal for a 1D cable, the matrix system is trivially
    solved with the **Thomas algorithm** (tridiagonal LU), exactly as in the
    classical Hines method.

    This implementation matches the same Crank-Nicolson update as the dense
    solver above:
    - diffusion term treated with the Hines matrix,
    - membrane currents added explicitly on the right-hand side,
    - gating variables updated over one full step with the exact `cnexp`
      helper `update_gates`.

    Implementation notes
    --------------------
    - The three coefficient vectors `dl, d, du` correspond to the lower,
      diagonal, and upper entries of the Hines matrix `(I - (dt/2) L)` and are
      constant in time for uniform compartments.
    - `update_gates` advances gating variables using the same frozen-voltage
      exponential update as NEURON `cnexp`.
    - The tridiagonal solve returns `V_{n+1}` directly; there is no separate
      half-step extrapolation.
    - Boundary conditions are sealed-end (zero-flux / Neumann), matching the
      diffusion operator rows at the two cable ends.
    - This method is O(N) in memory and time, compared to O(N²) for the dense
      solver, and is identical in structure to NEURON’s internal cable solver.

    Reference
    ---------
    Hines, M. (1984).
        *Efficient computation of branched nerve equations.*
        International Journal of Bio-Medical Computing, 15(1), 69–76.

    """


    def __init__(self) -> None:
        pass

    def solve(
        self,
        axon: AxonBase,
        tsim: float,
        dt: float,
        record_diagnostics: bool = False,
        record_observables: bool = False,
    ) -> SimResult:
        """
        Execute an optimized Crank–Nicolson simulation.

        Parameters
        ----------
        axon : AxonBase
            Axon model with ion channel model, geometry and stimulus.
        tsim : float
            Total simulation time in ms.
        dt : float
            Time step in ms.

        Returns
        -------
        SimResult
            Contains V_all (Nt × Nx) and t_vec (Nt).
        """
        if not bool(getattr(axon, "prefer_inline_extracellular_solver", False)):
            extracellular_res = _maybe_solve_with_extracellular_generic(
                axon,
                tsim,
                dt,
                record_diagnostics=record_diagnostics,
                record_observables=record_observables,
            )
            if extracellular_res is not None:
                return extracellular_res

        use_extracellular = bool(getattr(axon, "use_extracellular", False))
        runtime = prepare_solver_runtime(
            axon,
            tsim,
            dt,
            include_extracellular=use_extracellular,
            include_area=use_extracellular,
            precompute_intracellular=True,
        )
        if use_extracellular:
            kernel = DoubleCableKernel(
                runtime=runtime,
                Veinit_mV=float(getattr(axon, "Veinit", 0.0)),
            )
        else:
            kernel = SingleCableKernel(
                runtime=runtime,
                Cm_uF_cm2=jnp.asarray(axon.Cm, dtype=runtime.membrane.dtype),
            )

        out = kernel.run(
            record_diagnostics=record_diagnostics,
            record_observables=record_observables,
        )
        return SimResult(
            axon,
            out.Vm,
            out.t,
            diagnostics=out.diagnostics,
            recordings=out.recordings,
        )
