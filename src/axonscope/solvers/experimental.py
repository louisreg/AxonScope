"""Reference and prototype Crank-Nicholson solver variants.

Kept variants:
- ``CrankNicholsonVStimForcing``: imposed-field single-cable extracellular path.
- ``CrankNicholson_unoptimized``: dense reference implementation.
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
from .runtime import prepare_solver_runtime
from .stimulus_runtime import build_intracellular_current_density_fn


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


__all__ = [
    "CrankNicholsonVStimForcing",
    "CrankNicholson_unoptimized",
]
