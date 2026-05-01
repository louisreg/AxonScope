"""
CrankNicholson.py
==========

Numerical solvers for the cable equation used by AxonScope.

This module implements three time-stepping schemes for one-dimensional cable
equations with distributed ion-channel currents:

* CrankNicholson_unoptimized : Crank–Nicolson using a dense linear solve.
* CrankNicholson : optimized Crank–Nicolson using tridiagonal solver.

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
import jax
import jax.numpy as jnp
from typing import Optional, Tuple

from axonscope.axons.base import AxonBase
from axonscope.simresult import SimResult

from .base import Solver
from .common import (
    initial_voltage,
    diffusion_operator_coeffs,
    apply_diffusion_operator,
    Carry,
    Array,
    build_cn_tridiagonal,
    build_dense_from_tridiagonal,
    compartment_area_cm2,
    extracellular_absolute_arrays,
    solve_block_tridiagonal_2x2,
)
from .stimulus_runtime import (
    build_extracellular_potential_fn,
    build_intracellular_current_density_fn,
)


def _observable_names(membrane) -> dict[str, tuple[str, ...]]:
    return {
        "gates": membrane.gate_names(),
        "currents": membrane.current_names(),
        "conductances": membrane.conductance_names(),
        "states": membrane.membrane_state_names(),
    }


def _observable_matrices(
    membrane,
    V_mV: jnp.ndarray,
    gates: jnp.ndarray,
    state: tuple[jnp.ndarray, ...],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    Nx = V_mV.shape[0]
    gate_obs = membrane.gate_trace_matrix(gates, state)
    current_obs = membrane.ionic_current_trace_matrix(V_mV, gates, state)
    conductance_obs = membrane.conductance_trace_matrix(gates, state)
    if membrane.membrane_state_names():
        state_obs = membrane.membrane_state_trace_matrix(state)
    else:
        state_obs = jnp.zeros((Nx, 0), dtype=V_mV.dtype)
    return gate_obs, current_obs, conductance_obs, state_obs


def _package_recordings(
    names: dict[str, tuple[str, ...]],
    gate_obs: jnp.ndarray,
    current_obs: jnp.ndarray,
    conductance_obs: jnp.ndarray,
    state_obs: jnp.ndarray,
) -> dict[str, dict[str, jnp.ndarray]]:
    recordings: dict[str, dict[str, jnp.ndarray]] = {}
    packed = {
        "gates": gate_obs,
        "currents": current_obs,
        "conductances": conductance_obs,
        "states": state_obs,
    }
    for group_name, group_names in names.items():
        if not group_names:
            continue
        values = packed[group_name]
        group_recordings: dict[str, jnp.ndarray] = {}
        sum_duplicates = group_name in {"currents", "conductances"}
        for i, name in enumerate(group_names):
            column = values[:, :, i]
            if sum_duplicates and name in group_recordings:
                group_recordings[name] = group_recordings[name] + column
            else:
                group_recordings[name] = column
        recordings[group_name] = group_recordings
    return recordings

def _solve_with_extracellular_generic(
    axon: AxonBase,
    tsim: float,
    dt: float,
    record_diagnostics: bool = False,
    record_observables: bool = False,
) -> SimResult:
    """Generic extracellular (Vi/Ve) integration shared by the solver family."""
    Nx: int = axon.Nx
    Nt: int = int(jnp.ceil(tsim / dt))
    backend = axon.build_icm_backend()
    membrane = axon.ion_channel
    diagnostic_names = membrane.diagnostic_names()
    observable_names = _observable_names(membrane)
    dtype_local = backend.dtype

    Vm0: jnp.ndarray = initial_voltage(axon, Nx, dtype_local)
    Ve0: jnp.ndarray = jnp.full((Nx,), dtype_local(getattr(axon, "Veinit", 0.0)), dtype=dtype_local)
    Vi0: jnp.ndarray = Vm0 + Ve0
    gates0: jnp.ndarray = backend.init_gates(V0_mV=Vm0)
    state0 = membrane.init_membrane_state(Nx=Nx, dtype_local=dtype_local, V0_mV=Vm0)

    lower, diag, upper = diffusion_operator_coeffs(axon, dtype_local)
    area = compartment_area_cm2(axon, dtype_local)
    Cm_abs, Cx_abs, Gx_abs, Gax_e = extracellular_absolute_arrays(axon, dtype_local)
    Gax_i = 0.5 * (upper[:-1] * Cm_abs[:-1] + lower[1:] * Cm_abs[1:])
    left_i = jnp.concatenate([jnp.zeros((1,), dtype=dtype_local), Gax_i])
    right_i = jnp.concatenate([Gax_i, jnp.zeros((1,), dtype=dtype_local)])
    left_e = jnp.concatenate([jnp.zeros((1,), dtype=dtype_local), Gax_e])
    right_e = jnp.concatenate([Gax_e, jnp.zeros((1,), dtype=dtype_local)])

    inj_fun = build_intracellular_current_density_fn(axon)
    vext_fun = build_extracellular_potential_fn(axon)
    I_bg = backend.background_current()

    def step(carry, n):
        Vi_old, Ve_old, gates, *state_prev = carry
        state_prev = tuple(state_prev)
        Vm_old = Vi_old - Ve_old
        t_mid: float = dtype_local(n) * dt + dt / 2.0

        Iinj_abs = inj_fun(t_mid) * area
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

            A_diag = jnp.zeros((Nx, 2, 2), dtype=dtype_local)
            A_lower = jnp.zeros((Nx, 2, 2), dtype=dtype_local)
            A_upper = jnp.zeros((Nx, 2, 2), dtype=dtype_local)
            rhs = jnp.zeros((Nx, 2), dtype=dtype_local)

            A_diag = A_diag.at[:, 0, 0].set(Cm_abs / dt + Gm_abs + left_i + right_i)
            A_diag = A_diag.at[:, 0, 1].set(-(Cm_abs / dt + Gm_abs))
            rhs = rhs.at[:, 0].set((Cm_abs / dt) * Vm_old + GE_abs + Iinj_abs - I_outward_abs - I_corr_abs)

            A_diag = A_diag.at[:, 1, 0].set(-(Cm_abs / dt + Gm_abs))
            A_diag = A_diag.at[:, 1, 1].set(Cm_abs / dt + Gm_abs + Cx_abs / dt + Gx_abs + left_e + right_e)
            rhs = rhs.at[:, 1].set(
                -(Cm_abs / dt) * Vm_old
                - GE_abs
                + (Cx_abs / dt) * Ve_old
                - (Cx_abs / dt) * Vext_old
                + (Cx_abs / dt + Gx_abs) * Vext
                + I_outward_abs
                + I_corr_abs
            )

            A_lower = A_lower.at[1:, 0, 0].set(-Gax_i)
            A_upper = A_upper.at[:-1, 0, 0].set(-Gax_i)
            A_lower = A_lower.at[1:, 1, 1].set(-Gax_e)
            A_upper = A_upper.at[:-1, 1, 1].set(-Gax_e)

            sol = solve_block_tridiagonal_2x2(A_lower, A_diag, A_upper, rhs)
            return sol[:, 0], sol[:, 1]

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
            gate_obs, current_obs, conductance_obs, state_obs = _observable_matrices(
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
    t_vec: jnp.ndarray = (jnp.arange(Nt, dtype=dtype_local) + 1.0) * dt
    if record_observables and record_diagnostics and diagnostic_names:
        _, out = jax.lax.scan(step, init_carry, jnp.arange(Nt))
        V_all = out[0]
        recordings = _package_recordings(
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
        recordings = _package_recordings(
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


def _single_cable_backend_setup(
    axon: AxonBase,
) -> tuple[object, jnp.dtype, int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build the generic membrane backend used by single-cable solvers."""
    backend = axon.build_icm_backend()
    dtype_local = backend.dtype
    Nx = axon.Nx
    V0 = initial_voltage(axon, Nx, dtype_local)
    gates0 = backend.init_gates(V0_mV=V0)
    I_bg = backend.background_current()
    return backend, dtype_local, Nx, V0, gates0, I_bg


def _cn_channel_step(
    backend,
    V_mV: jnp.ndarray,
    gates: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generic channel update shared by Crank-Nicholson solver variants."""
    gates_new = backend.cn_gate_update(g_prev=gates, V_mV=V_mV, dt=dt)
    Iion = backend.currents(V_mV=V_mV, gates=gates_new)
    Gtot = backend.total_conductance(gates_new)
    return gates_new, Iion, Gtot


def _single_cable_diffusion_setup(
    axon: AxonBase,
    dtype_local: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return diffusion_operator_coeffs(axon, dtype_local)


def _implicit_fast_linear_terms(
    Cm: float,
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dl = -0.5 * Cm * lower
    du = -0.5 * Cm * upper
    d_static = (Cm / dt) - 0.5 * Cm * diag
    return dl, du, d_static


def _linearized_cn_rhs(
    V: jnp.ndarray,
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    Cm: float,
    dt: float,
    Iinj: jnp.ndarray,
    Iion: jnp.ndarray,
    I_bg: jnp.ndarray,
    Gtot: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    LV = apply_diffusion_operator(V, lower, diag, upper)
    rhs = (
        Cm / dt * V
        + 0.5 * Cm * LV
        + (Iinj - Iion - I_bg)
        + 0.5 * Gtot * V
    )
    return LV, rhs




# -----------------------------------------------------------------------------
# Crank–Nicolson (unoptimized, dense solve)
# -----------------------------------------------------------------------------
class CrankNicholson_unoptimized(Solver):
    """
    Crank–Nicolson scheme using a dense linear solver.

    Description
    -----------
    This solver implements the classical **Hines method** used in the NEURON
    simulator for cable equations, but expressed here in its dense-matrix form
    for clarity (the optimized solver below uses the tridiagonal Hines layout).

    The Hines approach is a semi-implicit **Crank–Nicolson** (CN) discretization
    of the cable equation, where the diffusion (axial) term is treated
    implicitly and the membrane ionic currents are treated explicitly or with a
    CN-exp update of gating variables.

    For a linear cable segment, the CN discretization writes:

        (I - (dt/2) * L) · V_{n+1}
            = (I + (dt/2) * L) · V_n  +  dt * (I_inj - I_ion)

    where L is the discrete Laplacian operator. This method is unconditionally
    stable for the passive cable and corresponds exactly to the numerical scheme
    introduced by **Hines (1984)** for multi-compartment neuronal models.

    In this “unoptimized” implementation we assemble the full dense matrix
    `(I - (dt/2) L)` and solve the linear system using `jnp.linalg.solve`.  
    This is correct but not memory- or compute-efficient for large Nx.  
    The optimized solver below replaces this with the classical **tridiagonal
    Hines matrix solver**, identical in structure to NEURON’s internal method.

    Notes on implementation
    -----------------------
    - Boundary conditions are sealed-end (zero-flux / Neumann).
    - Gating variables are advanced with the exact `update_gates` CNEXP helper
      over one full step before evaluating ionic currents.
    - Equivalent to the Hines method but implemented using dense matrices for
      pedagogical clarity.

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
        Run CN using dense linear algebra.

        Parameters
        ----------
        axon : AxonBase
            Axon model with ion channel dynamics and geometry.
        tsim : float
            Simulation duration (ms).
        dt : float
            Time step (ms).

        Returns
        -------
        SimResult
            Voltage traces V_all and time vector t_vec.
        """
        extracellular_res = _maybe_solve_with_extracellular_generic(
            axon, tsim, dt, record_diagnostics=record_diagnostics
        )
        if extracellular_res is not None:
            return extracellular_res

        Nx: int = axon.Nx
        Nt: int = int(jnp.ceil(tsim / dt))

        backend = axon.build_icm_backend()
        dtype_local = backend.dtype
        V0: jnp.ndarray = initial_voltage(axon, Nx, dtype_local)
        gates0: jnp.ndarray = backend.init_gates(V0_mV=V0)

        t_vec: jnp.ndarray = (jnp.arange(Nt, dtype=dtype_local) + 1.0) * dt

        lower, diag, upper = _single_cable_diffusion_setup(axon, dtype_local)
        dl, d, du = build_cn_tridiagonal(lower, diag, upper, dt, dtype_local)
        A: jnp.ndarray = build_dense_from_tridiagonal(dl, d, du, dtype_local)
        I_bg = backend.background_current()
        inj_fun = build_intracellular_current_density_fn(axon)

        def step(carry: Carry, n: int) -> Tuple[Carry, jnp.ndarray]:
            """
            One Crank–Nicolson step using dense solve.

            Parameters
            ----------
            carry : tuple (V, gates)
            n : int
                Time index.

            Returns
            -------
            carry_out : tuple (V_new, gates_new)
            V_new : ndarray
                Voltage after the direct CN solve.
            """
            V, gates = carry
            t_mid: float = dtype_local(n) * dt + dt / 2.0

            # NEURON channel mechanisms use cnexp; with V frozen over the step
            # this exponential update is exact for the linear gate ODE.
            gates_new: jnp.ndarray = backend.cn_gate_update(g_prev=gates, V_mV=V, dt=dt)

            # Ionic currents are evaluated using the updated gates and current V.
            Iion: jnp.ndarray = backend.currents(V_mV=V, gates=gates_new)

            Iinj: jnp.ndarray = inj_fun(t_mid)

            diffusion = apply_diffusion_operator(V, lower, diag, upper)
            rhs: jnp.ndarray = V + 0.5 * dt * diffusion + (dt / axon.Cm) * (Iinj - Iion - I_bg)

            # Solve the Crank-Nicolson system directly for V_{n+1}.
            V_new: jnp.ndarray = jnp.linalg.solve(A, rhs)

            return (V_new, gates_new), V_new

        (_, _), V_all = jax.lax.scan(step, (V0, gates0), jnp.arange(Nt))
        return SimResult(axon, V_all, t_vec)


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

        Nx: int = axon.Nx
        Nt: int = int(jnp.ceil(tsim / dt))
        backend = axon.build_icm_backend()
        membrane = axon.ion_channel
        diagnostic_names = membrane.diagnostic_names()
        observable_names = _observable_names(membrane)
        dtype_local = backend.dtype
        use_extracellular = bool(getattr(axon, "use_extracellular", False))

        Vm0: jnp.ndarray = initial_voltage(axon, Nx, dtype_local)
        Ve0: jnp.ndarray = jnp.full((Nx,), dtype_local(getattr(axon, "Veinit", 0.0)), dtype=dtype_local)
        Vi0: jnp.ndarray = Vm0 + Ve0
        gates0: jnp.ndarray = backend.init_gates(V0_mV=Vm0)

        lower, diag, upper = _single_cable_diffusion_setup(axon, dtype_local)
        dl, d, du = build_cn_tridiagonal(lower, diag, upper, dt, dtype_local)
        Cm = jnp.asarray(axon.Cm, dtype=dtype_local)

        inj_fun = build_intracellular_current_density_fn(axon)
        vext_fun = build_extracellular_potential_fn(axon)
        I_bg = backend.background_current()
        state0 = membrane.init_membrane_state(Nx=Nx, dtype_local=dtype_local, V0_mV=Vm0)
        n_extra = len(state0)

        area = compartment_area_cm2(axon, dtype_local)
        has_driven_extracellular = bool(use_extracellular and getattr(axon, "extracellular_contexts", ()))

        if use_extracellular:
            Cm_abs, Cx_abs, Gx_abs, Gax_e = extracellular_absolute_arrays(axon, dtype_local)
            Gax_i = 0.5 * (upper[:-1] * Cm_abs[:-1] + lower[1:] * Cm_abs[1:])
            left_i = jnp.concatenate([jnp.zeros((1,), dtype=dtype_local), Gax_i])
            right_i = jnp.concatenate([Gax_i, jnp.zeros((1,), dtype=dtype_local)])
            left_e = jnp.concatenate([jnp.zeros((1,), dtype=dtype_local), Gax_e])
            right_e = jnp.concatenate([Gax_e, jnp.zeros((1,), dtype=dtype_local)])

        def _unpack_voltage_state(carry, n_extra: int):
            if use_extracellular:
                Vi, Ve, gates, *extra = carry
                if len(extra) != n_extra:
                    raise ValueError(f"Expected {n_extra} extra state arrays, got {len(extra)}.")
                Vm = Vi - Ve
                return Vi, Ve, Vm, gates, tuple(extra)

            Vm, gates, *extra = carry
            if len(extra) != n_extra:
                raise ValueError(f"Expected {n_extra} extra state arrays, got {len(extra)}.")
            Vi = Vm
            Ve = jnp.zeros_like(Vm)
            return Vi, Ve, Vm, gates, tuple(extra)

        def _pack_voltage_state(
            Vi_new: jnp.ndarray,
            Ve_new: jnp.ndarray,
            Vm_new: jnp.ndarray,
            gates_new: jnp.ndarray,
            extra_state: tuple[jnp.ndarray, ...],
        ):
            if use_extracellular:
                return (Vi_new, Ve_new, gates_new, *extra_state)
            return (Vm_new, gates_new, *extra_state)

        def _channel_step(Vm: jnp.ndarray, gates: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            gates_new = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt)
            Iion = backend.currents(V_mV=Vm, gates=gates_new)
            return gates_new, Iion

        def _solve_intracellular_extracellular(
            Vi: jnp.ndarray,
            Ve: jnp.ndarray,
            gates_new: jnp.ndarray,
            Iinj_den: jnp.ndarray,
            I_bg_den: jnp.ndarray,
            I_corr_den: jnp.ndarray,
            Vext_mV: jnp.ndarray,
            Vext_old_mV: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            Gm_den, GE_den = backend.membrane_conductance_terms(gates_new)
            Gm_abs = Gm_den * area
            GE_abs = GE_den * area

            Iinj_abs = Iinj_den * area
            I_bg_abs = I_bg_den * area
            I_corr_abs = I_corr_den * area
            Vm = Vi - Ve

            A_diag = jnp.zeros((Nx, 2, 2), dtype=dtype_local)
            A_lower = jnp.zeros((Nx, 2, 2), dtype=dtype_local)
            A_upper = jnp.zeros((Nx, 2, 2), dtype=dtype_local)
            rhs = jnp.zeros((Nx, 2), dtype=dtype_local)

            A_diag = A_diag.at[:, 0, 0].set(Cm_abs / dt + Gm_abs + left_i + right_i)
            A_diag = A_diag.at[:, 0, 1].set(-(Cm_abs / dt + Gm_abs))
            rhs = rhs.at[:, 0].set((Cm_abs / dt) * Vm + GE_abs + Iinj_abs - I_bg_abs - I_corr_abs)

            A_diag = A_diag.at[:, 1, 0].set(-(Cm_abs / dt + Gm_abs))
            A_diag = A_diag.at[:, 1, 1].set(Cm_abs / dt + Gm_abs + Cx_abs / dt + Gx_abs + left_e + right_e)
            rhs = rhs.at[:, 1].set(
                -(Cm_abs / dt) * Vm
                - GE_abs
                + (Cx_abs / dt) * Ve
                - (Cx_abs / dt) * Vext_old_mV
                + (Cx_abs / dt + Gx_abs) * Vext_mV
                + I_bg_abs
                + I_corr_abs
            )

            A_lower = A_lower.at[1:, 0, 0].set(-Gax_i)
            A_upper = A_upper.at[:-1, 0, 0].set(-Gax_i)
            A_lower = A_lower.at[1:, 1, 1].set(-Gax_e)
            A_upper = A_upper.at[:-1, 1, 1].set(-Gax_e)

            sol = solve_block_tridiagonal_2x2(A_lower, A_diag, A_upper, rhs)
            return sol[:, 0], sol[:, 1]

        def _solve_voltage_step(
            Vi: jnp.ndarray,
            Ve: jnp.ndarray,
            Vm: jnp.ndarray,
            gates_new: jnp.ndarray,
            t_mid: float,
            Iinj_den: jnp.ndarray,
            I_outward_den: jnp.ndarray,
            I_corr_den: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            if use_extracellular:
                Vext = vext_fun(t_mid)
                Vext_old = vext_fun(t_mid - dt)
                Vi_new, Ve_new = _solve_intracellular_extracellular(
                    Vi=Vi,
                    Ve=Ve,
                    gates_new=gates_new,
                    Iinj_den=Iinj_den,
                    I_bg_den=I_outward_den,
                    I_corr_den=I_corr_den,
                    Vext_mV=Vext,
                    Vext_old_mV=Vext_old,
                )
                return Vi_new, Ve_new, Vi_new - Ve_new

            diffusion = apply_diffusion_operator(Vm, lower, diag, upper)
            rhs = Vm + dtype_local(0.5) * dt * diffusion + (dtype_local(dt) / Cm) * (
                Iinj_den - I_outward_den - I_corr_den
            )
            Vm_new = jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:, None])[:, 0]
            return Vm_new, jnp.zeros_like(Vm_new), Vm_new

        def step(carry: Carry, n: int):
            Vi, Ve, Vm, gates, extra = _unpack_voltage_state(carry, n_extra=n_extra)
            t_mid: float = dtype_local(n) * dt + dt / 2.0

            gates_pred, Iion_pred = _channel_step(Vm, gates)
            Iinj: jnp.ndarray = inj_fun(t_mid)
            step_plan_pred = membrane.prepare_membrane_step(
                V_mV=Vm,
                gates_prev=gates,
                gates_new=gates_pred,
                state=extra,
                dt=dt,
                I_ion=Iion_pred,
                I_background=I_bg,
            )
            linearization_gates = step_plan_pred.linearization_gates
            if has_driven_extracellular:
                linearization_gates = gates
            Vi_new, Ve_new, Vm_new = _solve_voltage_step(
                Vi=Vi,
                Ve=Ve,
                Vm=Vm,
                gates_new=linearization_gates,
                t_mid=t_mid,
                Iinj_den=Iinj,
                I_outward_den=(
                    step_plan_pred.explicit_outward_current
                    if use_extracellular
                    else step_plan_pred.total_outward_current
                ),
                I_corr_den=step_plan_pred.correction_current,
            )
            gates_new = membrane.final_gate_update(
                gates_prev=gates,
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                dt=dt,
                gates_predictor=gates_pred,
            )
            Iion_new = backend.currents(V_mV=Vm_new, gates=gates_new)
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state=extra,
                dt=dt,
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
                dt=dt,
            )
            carry_out = _pack_voltage_state(Vi_new, Ve_new, Vm_new, gates_new, state_new)

            if record_observables:
                gate_obs, current_obs, conductance_obs, state_obs = _observable_matrices(
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
            (Vi0, Ve0, gates0, *state0)
            if use_extracellular
            else (Vm0, gates0, *state0)
        )
        t_vec = (jnp.arange(Nt, dtype=dtype_local) + dtype_local(1.0)) * dt
        if record_observables and record_diagnostics and diagnostic_names:
            _, out = jax.lax.scan(step, init_carry, jnp.arange(Nt))
            V_all = out[0]
            recordings = _package_recordings(
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
            recordings = _package_recordings(
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

# -----------------------------------------------------------------------------
# Semi-implicit Crank–Nicolson (linearized ionic currents)
# -----------------------------------------------------------------------------
class CrankNicholsonSemiImplicit(Solver):

    def solve(
        self,
        axon: AxonBase,
        tsim: float,
        dt: float,
        record_diagnostics: bool = False,
    ) -> SimResult:
        extracellular_res = _maybe_solve_with_extracellular_generic(
            axon, tsim, dt, record_diagnostics=record_diagnostics
        )
        if extracellular_res is not None:
            return extracellular_res

        backend, dtype, Nx, V, gates, I_bg = _single_cable_backend_setup(axon)
        Nt = int(jnp.ceil(tsim / dt))

        Cm = jnp.asarray(axon.Cm, dtype=dtype)
        dt_local = jnp.asarray(dt, dtype=dtype)

        # -------------------------
        # Diffusion operator (matrix form ONLY here)
        # -------------------------
        lower, diag, upper = _single_cable_diffusion_setup(axon, dtype)

        # CN matrix: (I - dt/2 L)
        dl = -0.5 * dt_local * lower
        du = -0.5 * dt_local * upper
        d_static = 1.0 - 0.5 * dt_local * diag

        # -------------------------
        # Channels
        # -------------------------
        inj_fun = build_intracellular_current_density_fn(axon)

        # -------------------------
        # Step
        # -------------------------
        def step(carry, n):
            V, gates = carry

            t_mid = n * dt + 0.5 * dt

            gates_new, Iion, G_tot = _cn_channel_step(backend, V, gates, dt)

            Iinj = inj_fun(t_mid)

            # -------------------------------------------------------
            # CN diffusion (FULL implicit, tridiagonal)
            # -------------------------------------------------------
            LV = apply_diffusion_operator(V, lower, diag, upper)

            rhs = (
                V
                + 0.5 * dt_local * LV
                + (dt_local / Cm) * (Iinj - Iion - I_bg)
                + (dt_local / (2.0 * Cm)) * G_tot * V
            )

            # effective diagonal includes ionic linearization
            d = d_static + (dt_local / (2.0 * Cm)) * G_tot

            # solve tridiagonal system (THIS is the key fix)
            V_new = jax.lax.linalg.tridiagonal_solve(
                dl, d, du, rhs[:, None]
            )[:, 0]

            return (V_new, gates_new), V_new

        (_, _), V_all = jax.lax.scan(
            step, (V, gates), jnp.arange(Nt)
        )

        t_vec = (jnp.arange(Nt, dtype=dtype) + 1.0) * dt

        return SimResult(axon, V_all, t_vec)

# -----------------------------------------------------------------------------
# Fully implicit Crank–Nicolson (Newton iteration per step)
# -----------------------------------------------------------------------------
class CrankNicholsonImplicit(Solver):
    """
    Fully implicit Crank–Nicolson solver using a Newton iteration per step.

    Description
    -----------
    Unlike the semi-implicit variant that linearizes I_ion only once around
    V_n, this solver solves the full nonlinear CN system at each step:

        F(V_{n+1}) = Cm (V_{n+1} - V_n) / dt
                   - (1/2) ( L V_{n+1} + L V_n )
                   + (1/2) ( I_ion(V_{n+1}, g_{n+1/2}) + I_ion(V_n, g_{n+1/2}) )
                   - I_inj^{n+1/2}
                   = 0

    The nonlinear system is solved by Newton iteration:

        V^{(k+1)} = V^{(k)} - J^{-1} · F(V^{(k)})

    where the Jacobian J = ∂F/∂V is tridiagonal:
        J = (Cm/dt) I - (1/2) L + (1/2) diag(G_tot(V^{(k)}))

    Since J is tridiagonal, each Newton step has the same cost as one
    tridiagonal solve. In practice 2–3 Newton iterations per step are
    enough to converge to machine precision for typical AP dynamics.

    When to use
    -----------
    - Very large dt on stiff systems (dt ≥ 0.5 ms on HH).
    - Strongly nonlinear channels (e.g. calcium dynamics, persistent Na+).
    - When accuracy at large dt matters more than per-step speed.

    For moderate dt, the semi-implicit solver is usually faster overall
    because it avoids the Newton loop; this solver becomes competitive
    when the semi-implicit scheme would need to reduce dt for accuracy.

    Implementation notes
    --------------------
    - Gating variables are advanced once per step (CN-exp), not inside the
      Newton loop, to keep the Jacobian tridiagonal. This is the standard
      operator-splitting convention used by NEURON for implicit modes.
    - The Newton loop uses a fixed number of iterations (`n_newton`) so the
      whole step is JIT-friendly via `jax.lax.fori_loop`. Convergence is
      robust for typical AP simulations; tune `n_newton` if needed.
    - The diffusion Laplacian `LV = D · d²V/dx²` is built with interior
      centered differences and zeroed at the endpoints.

    Reference
    ---------
    Hines, M. (1984). *Efficient computation of branched nerve equations.*
        Int. J. Bio-Med. Comput., 15(1), 69–76.
    """

    def __init__(self, n_newton: int = 3) -> None:
        """
        Parameters
        ----------
        n_newton : int
            Number of Newton iterations per time step. Default is 3, which
            is typically enough to reach machine precision on HH-like models.
        """
        self.n_newton = int(n_newton)

    def solve(self, axon: AxonBase, tsim: float, dt: float) -> SimResult:
        """
        Execute a fully implicit Crank–Nicolson simulation.

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
        extracellular_res = _maybe_solve_with_extracellular_generic(axon, tsim, dt)
        if extracellular_res is not None:
            return extracellular_res

        backend, dtype_local, Nx, V0, gates0, I_bg = _single_cable_backend_setup(axon)
        Nt: int = int(jnp.ceil(tsim / dt))
        n_newton: int = self.n_newton

        # Geometry / constants
        Cm: float = axon.Cm
        Cm_over_dt: float = Cm / dt

        # Sealed-end Neumann diffusion operator (shared helpers)
        lower, diag, upper = _single_cable_diffusion_setup(axon, dtype_local)

        # Newton Jacobian off-diagonals: J = (Cm/dt) I - (1/2) L + diag(G_tot/2)
        dl_base: jnp.ndarray = -0.5 * lower
        du_base: jnp.ndarray = -0.5 * upper
        # Static diagonal: Cm/dt - (1/2) * L_diag  (L_diag is negative, so this adds)
        d_static: jnp.ndarray = dtype_local(Cm_over_dt) * jnp.ones((Nx,), dtype=dtype_local) - 0.5 * diag

        inj_fun = build_intracellular_current_density_fn(axon)

        def laplacian(V: jnp.ndarray) -> jnp.ndarray:
            """Compute L @ V = D * d²V/dx² with sealed-end Neumann BCs."""
            return apply_diffusion_operator(V, lower, diag, upper)

        def residual(V_k: jnp.ndarray, V_n: jnp.ndarray, gates_new: jnp.ndarray,
                     Iinj: jnp.ndarray, LV_n: jnp.ndarray,
                     Iion_n: jnp.ndarray) -> jnp.ndarray:
            """
            Nonlinear residual F(V_k) for the implicit CN step.

            F(V_k) = Cm/dt (V_k - V_n)
                   - (1/2)(L V_k + L V_n)
                   + (1/2)(I_ion(V_k, g) + I_ion(V_n, g))
                   - I_inj
            """
            LV_k = laplacian(V_k)
            Iion_k = backend.currents(V_mV=V_k, gates=gates_new)
            return (
                Cm_over_dt * (V_k - V_n)
                - 0.5 * (LV_k + LV_n)
                + 0.5 * (Iion_k + Iion_n)
                - Iinj
                + I_bg
            )

        def step(carry: Carry, n: int) -> Tuple[Carry, jnp.ndarray]:
            """
            Single fully-implicit CN step: gating update + Newton iteration.

            Parameters
            ----------
            carry : tuple (V, gates)
            n : int
                Current time-step index.

            Returns
            -------
            carry_out : tuple (V_new, gates_new)
            V_new : ndarray
            """
            V, gates = carry
            t_mid: float = dtype_local(n) * dt + dt / 2.0

            # Advance gates once (operator splitting, keeps Jacobian tridiagonal)
            gates_new = backend.cn_gate_update(g_prev=gates, V_mV=V, dt=dt)

            # Quantities that do not change during the Newton loop
            Iinj: jnp.ndarray = inj_fun(t_mid)
            LV_n: jnp.ndarray = laplacian(V)
            Iion_n = backend.currents(V_mV=V, gates=gates_new)

            def newton_body(_: int, V_k: jnp.ndarray) -> jnp.ndarray:
                """One Newton iteration: V_{k+1} = V_k - J^{-1} F(V_k)."""
                # Residual at current iterate
                F_k = residual(V_k, V, gates_new, Iinj, LV_n, Iion_n)

                # Local conductance slope at V_k, used for the Jacobian diagonal
                G_tot_k = backend.total_conductance(gates=gates_new)

                # Jacobian diagonal: Cm/dt - (1/2)*L_diag + (1/2)*G_tot
                d_k = d_static + 0.5 * G_tot_k

                dV = jax.lax.linalg.tridiagonal_solve(
                    dl_base, d_k, du_base, F_k[:, None]
                )[:, 0]

                return V_k - dV

            # Initial guess: V_n (could also use semi-implicit predictor)
            V_new: jnp.ndarray = jax.lax.fori_loop(0, n_newton, newton_body, V)

            return (V_new, gates_new), V_new

        (_, _), V_all = jax.lax.scan(step, (V0, gates0), jnp.arange(Nt))
        t_vec: jnp.ndarray = (jnp.arange(Nt, dtype=dtype_local) + 1.0) * dt

        return SimResult(axon, V_all, t_vec)


class CrankNicholsonImplicitFast(Solver):

    def solve(self, axon, tsim, dt):
        extracellular_res = _maybe_solve_with_extracellular_generic(axon, tsim, dt)
        if extracellular_res is not None:
            return extracellular_res

        backend, dtype, Nx, V, gates, I_bg = _single_cable_backend_setup(axon)
        Nt = int(jnp.ceil(tsim / dt))

        Cm = axon.Cm
        lower, diag, upper = _single_cable_diffusion_setup(axon, dtype)
        dl, du, d_static = _implicit_fast_linear_terms(Cm, lower, diag, upper, dt)

        inj_fun = build_intracellular_current_density_fn(axon)

        def step(carry, n):
            V, gates = carry
            t_mid = (n + 0.5) * dt

            gates_new, Iion, Gtot = _cn_channel_step(backend, V, gates, dt)

            Iinj = inj_fun(t_mid)

            _, rhs = _linearized_cn_rhs(V, lower, diag, upper, Cm, dt, Iinj, Iion, I_bg, Gtot)

            d = d_static + 0.5 * Gtot

            V_new = jax.lax.linalg.tridiagonal_solve(
                dl, d, du, rhs[:, None]
            )[:, 0]

            return (V_new, gates_new), V_new

        (_, _), V_all = jax.lax.scan(step, (V, gates), jnp.arange(Nt))

        t_vec = (jnp.arange(Nt, dtype=dtype) + 1.0) * dt

        return SimResult(axon, V_all, t_vec)
    
class CrankNicholsonImplicitFastMultiStep(Solver):

    def __init__(self, K: int = 5):
        self.K = int(K)

    def solve(self, axon, tsim, dt):
        extracellular_res = _maybe_solve_with_extracellular_generic(axon, tsim, dt)
        if extracellular_res is not None:
            return extracellular_res

        backend, dtype, Nx, V, gates, I_bg = _single_cable_backend_setup(axon)
        K = self.K

        Nt = int(jnp.ceil(tsim / dt))

        Cm = axon.Cm
        lower, diag, upper = _single_cable_diffusion_setup(axon, dtype)
        dl, du, d_static = _implicit_fast_linear_terms(Cm, lower, diag, upper, dt)

        inj_fun = build_intracellular_current_density_fn(axon)

        def step(V, gates, t):

            gates_new, Iion, Gtot = _cn_channel_step(backend, V, gates, dt)

            Iinj = inj_fun(t + 0.5 * dt)

            _, rhs = _linearized_cn_rhs(V, lower, diag, upper, Cm, dt, Iinj, Iion, I_bg, Gtot)

            d = d_static + 0.5 * Gtot

            V_new = jax.lax.linalg.tridiagonal_solve(
                dl, d, du, rhs[:, None]
            )[:, 0]

            return V_new, gates_new

        def block_step(carry, b):
            V, gates, t0 = carry

            def body(carry, k):
                V, gates, t = carry
                V_new, gates_new = step(V, gates, t)
                return (V_new, gates_new, t + dt), V_new

            (V, gates, t_end), Vs = jax.lax.scan(
                body,
                (V, gates, t0),
                jnp.arange(K)
            )

            return (V, gates, t_end), Vs

        n_full_blocks = Nt // K
        n_tail = Nt % K

        (V, gates, t_end), V_blocks = jax.lax.scan(
            block_step,
            (V, gates, dtype(0.0)),
            jnp.arange(n_full_blocks)
        )

        V_all = V_blocks.reshape((-1, Nx)) if n_full_blocks > 0 else jnp.zeros((0, Nx), dtype=dtype)

        if n_tail > 0:
            def tail_body(carry, k):
                V, gates, t = carry
                V_new, gates_new = step(V, gates, t)
                return (V_new, gates_new, t + dt), V_new

            (V, gates, t_end), V_tail = jax.lax.scan(
                tail_body,
                (V, gates, t_end),
                jnp.arange(n_tail)
            )
            V_all = jnp.concatenate([V_all, V_tail], axis=0)

        t_vec = (jnp.arange(Nt, dtype=dtype) + 1.0) * dt

        return SimResult(axon, V_all, t_vec)
    


class CrankNicholsonQuasiNewtonFast(Solver):

    def solve(self, axon, tsim, dt):
        extracellular_res = _maybe_solve_with_extracellular_generic(axon, tsim, dt)
        if extracellular_res is not None:
            return extracellular_res

        backend, dtype, Nx, V, gates, I_bg = _single_cable_backend_setup(axon)
        Nt = int(jnp.ceil(tsim / dt))

        Cm = axon.Cm
        lower, diag, upper = _single_cable_diffusion_setup(axon, dtype)

        inj_fun = build_intracellular_current_density_fn(axon)

        # -------------------------
        # CN diffusion operators
        # -------------------------
        dl, du, d_static = _implicit_fast_linear_terms(Cm, lower, diag, upper, dt)

        # relaxation (important for stability)
        omega = 0.6

        def step(carry, n):
            V, gates = carry
            t_mid = (n + 0.5) * dt

            # -------------------------
            # gating (unchanged stable CNEXP)
            # -------------------------
            gates_new, Iion, Gtot = _cn_channel_step(backend, V, gates, dt)

            Iinj = inj_fun(t_mid)

            # -------------------------
            # freeze Jacobian at V_n
            # -------------------------
            d = d_static + 0.5 * Gtot

            # LU factorization ONLY ONCE
            def linear_solve(rhs):
                return jax.lax.linalg.tridiagonal_solve(
                    dl, d, du, rhs[:, None]
                )[:, 0]

            # -------------------------
            # RHS CN
            # -------------------------
            _, rhs = _linearized_cn_rhs(V, lower, diag, upper, Cm, dt, Iinj, Iion, I_bg, Gtot)

            # -------------------------
            # STEP 1: quasi-Newton solve
            # -------------------------
            V_new = linear_solve(rhs)

            # -------------------------
            # STEP 2: 1 correction only (cheap)
            # -------------------------
            for _ in range(1):
                Iion_new = backend.currents(V_mV=V_new, gates=gates_new)
                LV_new = apply_diffusion_operator(V_new, lower, diag, upper)
                rhs_corr = (
                    Cm / dt * V
                    + 0.5 * Cm * LV_new
                    + (Iinj - Iion_new - I_bg)
                    + 0.5 * Gtot * V
                )

                V_corr = linear_solve(rhs_corr)

                # relaxation
                V_new = (1 - omega) * V_new + omega * V_corr

            return (V_new, gates_new), V_new

        (_, _), V_all = jax.lax.scan(
            step,
            (V, gates),
            jnp.arange(Nt)
        )

        t_vec = (jnp.arange(Nt, dtype=dtype) + 1.0) * dt

        return SimResult(axon, V_all, t_vec)
