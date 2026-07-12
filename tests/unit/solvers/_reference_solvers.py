"""Reference one-row solver variants used by tests.

Kept variants:
- ``DenseSingleCableReference``: dense reference implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from axonscope.axon_instance import AxonInstance, as_axon_instance
from axonscope.axons.axon import Axon
from axonscope.runtime.jax.cable_geometry import (
    apply_diffusion_operator,
    diffusion_operator_coeffs,
    initial_voltage,
)
from axonscope.runtime.jax.preparation.base import prepare_membrane_runtime
from axonscope.runtime.row_output import RowRecordingOutput
from axonscope.runtime.solver_axon import build_solver_axon
from axonscope.solvers.options import SolverOptions
from axonscope.timebase import resolve_time_args, simulation_step_count

from axonscope.runtime.jax.inputs.intracellular import (
    build_intracellular_current_density_fn,
)

Carry = tuple[jnp.ndarray, jnp.ndarray]


def _build_cn_tridiagonal(
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    dt: float,
    dtype_local: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dt_local = dtype_local(dt)
    dl = -0.5 * dt_local * lower
    d = jnp.ones_like(diag, dtype=dtype_local) - 0.5 * dt_local * diag
    du = -0.5 * dt_local * upper
    return dl, d, du


def _build_dense_from_tridiagonal(
    dl: jnp.ndarray,
    d: jnp.ndarray,
    du: jnp.ndarray,
    dtype_local: jnp.dtype,
) -> jnp.ndarray:
    nx = d.shape[0]
    matrix = jnp.zeros((nx, nx), dtype=dtype_local)
    idx = jnp.arange(nx)
    matrix = matrix.at[idx, idx].set(d)
    matrix = matrix.at[idx[1:], idx[:-1]].set(dl[1:])
    matrix = matrix.at[idx[:-1], idx[1:]].set(du[:-1])
    return matrix


class DenseSingleCableReference:
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
    - Gating variables are advanced through the membrane program's gate update
      over one full step before evaluating ionic currents.
    - Equivalent to the Hines method but implemented using dense matrices for
      pedagogical clarity.

    Reference
    ---------
    Hines, M. (1984).
        *Efficient computation of branched nerve equations.*
        International Journal of Bio-Medical Computing, 15(1), 69–76.
    """

    def __init__(self, *, solver_options: SolverOptions | None = None) -> None:
        self.solver_options = (
            SolverOptions() if solver_options is None else solver_options
        )

    def solve(
        self,
        axon: Axon | AxonInstance,
        tsim: float | None = None,
        dt: float | None = None,
    ) -> RowRecordingOutput:
        """
        Run CN using dense linear algebra.

        Parameters
        ----------
        axon : Axon
            Axon model with ion channel dynamics and geometry.
        tsim : float
            Simulation duration (ms).
        dt : float
            Time step (ms).

        Returns
        -------
            RowRecordingOutput
            Voltage traces V_all and time vector t_vec.
        """
        simulation = as_axon_instance(axon)
        duration, step = resolve_time_args(tsim=tsim, dt=dt)
        if bool(getattr(simulation, "use_extracellular", False)):
            raise ValueError(
                "DenseSingleCableReference is an intracellular single-cable "
                "dense reference; use the batch route for extracellular cases."
            )

        solver_axon = build_solver_axon(simulation)
        Nx: int = solver_axon.n_compartments
        Nt: int = simulation_step_count(duration, step)

        membrane_runtime = prepare_membrane_runtime(
            simulation,
            solver_axon=solver_axon,
            solver_options=self.solver_options,
        )
        if membrane_runtime.state0:
            raise NotImplementedError(
                "DenseSingleCableReference only supports stateless membrane models."
            )
        backend = membrane_runtime.backend
        dtype_local = membrane_runtime.dtype
        V0: jnp.ndarray = initial_voltage(simulation, Nx, dtype_local)
        gates0: jnp.ndarray = membrane_runtime.gates0

        t_vec: jnp.ndarray = (jnp.arange(Nt, dtype=dtype_local) + 1.0) * step

        lower, diag, upper = diffusion_operator_coeffs(solver_axon, dtype_local)
        dl, d, du = _build_cn_tridiagonal(lower, diag, upper, step, dtype_local)
        A: jnp.ndarray = _build_dense_from_tridiagonal(dl, d, du, dtype_local)
        I_bg = membrane_runtime.background_current
        inj_fun = build_intracellular_current_density_fn(
            simulation,
            solver_axon=solver_axon,
        )

        def scan_step(carry: Carry, n: int) -> tuple[Carry, jnp.ndarray]:
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
            t_mid: float = dtype_local(n) * step + step / 2.0

            # NEURON channel mechanisms use cnexp; with V frozen over the step
            # this exponential update is exact for the linear gate ODE.
            gates_new: jnp.ndarray = backend.cn_gate_update(
                g_prev=gates,
                V_mV=V,
                dt=step,
            )

            # Ionic currents are evaluated using the updated gates and current V.
            Iion: jnp.ndarray = backend.currents(V_mV=V, gates=gates_new)

            Iinj: jnp.ndarray = inj_fun(t_mid)

            diffusion = apply_diffusion_operator(V, lower, diag, upper)
            rhs: jnp.ndarray = (
                V
                + 0.5 * step * diffusion
                + (step / jnp.asarray(solver_axon.Cm_uF_cm2, dtype=dtype_local))
                * (Iinj - Iion - I_bg)
            )

            # Solve the Crank-Nicolson system directly for V_{n+1}.
            V_new: jnp.ndarray = jnp.linalg.solve(A, rhs)

            return (V_new, gates_new), V_new

        (_, _), V_all = jax.lax.scan(scan_step, (V0, gates0), jnp.arange(Nt))
        return RowRecordingOutput(simulation.axon, V_all, t_vec, simulation=simulation)


__all__ = [
    "DenseSingleCableReference",
]
