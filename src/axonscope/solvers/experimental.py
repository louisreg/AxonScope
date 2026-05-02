"""Experimental and reference Crank-Nicholson solver variants.

Kept variants:
- ``CrankNicholsonVStimForcing``: imposed-field single-cable extracellular path.
- ``CrankNicholson_unoptimized``: dense reference implementation.
- ``CrankNicholsonSemiImplicit``: one-shot ionic-current linearization.
- ``CrankNicholsonImplicit``: Newton-style nonlinear solve prototype.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from axonscope.axons.base import AxonBase
from axonscope.simresult import SimResult

from .base import Solver
from .common import (
    Carry,
    apply_diffusion_operator,
    build_cn_tridiagonal,
    build_dense_from_tridiagonal,
    diffusion_operator_coeffs,
    initial_voltage,
)
from .crank_nicholson import _maybe_solve_with_extracellular_generic
from .kernels import SingleCableKernel
from .runtime import prepare_membrane_runtime, prepare_solver_runtime
from .stimulus_runtime import build_intracellular_current_density_fn


def _single_cable_backend_setup(
    axon: AxonBase,
) -> tuple[object, jnp.dtype, int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build the generic membrane backend used by single-cable solvers."""
    runtime = prepare_membrane_runtime(axon)
    return (
        runtime.backend,
        runtime.dtype,
        runtime.Nx,
        runtime.Vm0_mV,
        runtime.gates0,
        runtime.background_current,
    )


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


class CrankNicholsonVStimForcing(Solver):
    """Single-cable Crank-Nicolson with an imposed extracellular potential.

    This prototype treats extracellular stimulation as a prescribed field
    ``Vstim(t, x)`` rather than as a dynamic periaxonal state. The cable solve
    remains scalar on Vm and adds the known forcing term ``L(Vstim)``.
    """

    def solve(
        self,
        axon: AxonBase,
        tsim: float,
        dt: float,
        record_diagnostics: bool = False,
        record_observables: bool = False,
    ) -> SimResult:
        if bool(getattr(axon, "has_heterogeneous_cable_properties", False)):
            raise ValueError(
                "CrankNicholsonVStimForcing is a single-cable solver; "
                "use CrankNicholson for heterogeneous/double-cable axons."
            )

        runtime = prepare_solver_runtime(
            axon,
            tsim,
            dt,
            include_extracellular=False,
            include_area=False,
            precompute_intracellular=True,
            precompute_extracellular=True,
        )
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

        lower, diag, upper = diffusion_operator_coeffs(axon, dtype_local)
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
        lower, diag, upper = diffusion_operator_coeffs(axon, dtype)

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
                   - (Cm/2) ( L V_{n+1} + L V_n )
                   + (1/2) ( I_ion(V_{n+1}, g_{n+1/2}) + I_ion(V_n, g_{n+1/2}) )
                   - I_inj^{n+1/2}
                   = 0

    The nonlinear system is solved by Newton iteration:

        V^{(k+1)} = V^{(k)} - J^{-1} · F(V^{(k)})

    where the Jacobian J = ∂F/∂V is tridiagonal:
        J = (Cm/dt) I - (Cm/2) L + (1/2) diag(G_tot(V^{(k)}))

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
        lower, diag, upper = diffusion_operator_coeffs(axon, dtype_local)

        # Newton Jacobian: Cm/dt * I - 0.5 * Cm * L + diag(Gtot/2).
        dl_base: jnp.ndarray = -0.5 * dtype_local(Cm) * lower
        du_base: jnp.ndarray = -0.5 * dtype_local(Cm) * upper
        d_static: jnp.ndarray = (
            dtype_local(Cm_over_dt) * jnp.ones((Nx,), dtype=dtype_local)
            - 0.5 * dtype_local(Cm) * diag
        )

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
                   - (Cm/2)(L V_k + L V_n)
                   + (1/2)(I_ion(V_k, g) + I_ion(V_n, g))
                   - I_inj
            """
            LV_k = laplacian(V_k)
            Iion_k = backend.currents(V_mV=V_k, gates=gates_new)
            return (
                Cm_over_dt * (V_k - V_n)
                - 0.5 * Cm * (LV_k + LV_n)
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

                # Jacobian diagonal: Cm/dt - (Cm/2)*L_diag + (1/2)*G_tot
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


__all__ = [
    "CrankNicholsonVStimForcing",
    "CrankNicholson_unoptimized",
    "CrankNicholsonSemiImplicit",
    "CrankNicholsonImplicit",
]
