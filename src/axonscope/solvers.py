"""
solvers.py
==========

Numerical solvers for the cable equation used by AxonScope.

This module implements three time-stepping schemes for one-dimensional cable
equations with distributed ion-channel currents:

* Euler        : forward Euler, explicit, simple and pedagogical.
* CrankNicholson_unoptimized : Crank–Nicolson using a dense linear solve.
* CrankNicholson : optimized Crank–Nicolson using tridiagonal solver.

All solvers are written to be JAX-friendly:
- they operate on JAX arrays (`jnp.ndarray`),
- loop over time using `jax.lax.scan` so the entire time loop can be jitted,
- avoid Python-side mutation of arrays,
- pre-extract functional objects from the axon/ion channel model when appropriate,
- GPU-ready

TODO: 
    - Implement multi-compartment version
    - Provide saving methods with filtering (eg. Iion, Gating Variables...)
    - Provide chunk simulation
    - Provide batching capabilities
    - Add "on-the-fly" processing capabilities (eg rasterizing, filtering,...)
    - and other things
    

Author: l.regnacq
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Callable, Tuple

from axonscope.axons import AxonBase
from axonscope.simresult import SimResult
from axonscope.benchmark import Benchmark
from axonscope.icm_compute import Gating

bench = Benchmark()


# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
Array = jnp.ndarray
Carry = Tuple[Array, Array]  # generic (V, gates) carry used by scan


# -----------------------------------------------------------------------------
# Abstract solver base class
# -----------------------------------------------------------------------------
class Solver(ABC):
    """
    Abstract base class for temporal solvers of the cable equation.

    Concrete solver classes must implement `solve(axon, tsim, dt)` and return a
    :class:`SimResult` object containing the voltage traces and time vector.

    The cable equation solved is (per compartment):
        C_m * dV/dt = D * C_m * d^2V/dx^2 - I_ion(V, gates) + I_inj(t)

    where:
    - V is membrane voltage in mV
    - C_m is membrane capacitance (µF/cm²)
    - D is axial diffusion coefficient (cm²/ms)
    - I_ion is ionic current density (µA/cm²)
    - I_inj is external injected current density (µA/cm²)
    """

    @abstractmethod
    def solve(self, axon: AxonBase, tsim: float, dt: float) -> SimResult:
        """
        Run simulation for a given axon.

        Parameters
        ----------
        axon : AxonBase
            Axon object providing geometry, ion channel model and stimulus.
        tsim : float
            Total simulation time (ms).
        dt : float
            Time step (ms).

        Returns
        -------
        SimResult
            Simulation result containing V_all (Nt × Nx) and t_vec (Nt).
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Euler explicit solver
# -----------------------------------------------------------------------------
class Euler(Solver):
    """
    Explicit forward Euler solver for the cable equation.

    Description
    -----------
    The solver advances voltage using a forward Euler discretization in time
    and a centered second-order finite difference for the spatial second
    derivative (Laplacian). Gating variables are updated using the
    `update_gates` CNEXP-like helper.

    Discretization (for interior spatial index i):
        d2Vdx2[i] ≈ (V[i+1] - 2 V[i] + V[i-1]) / dx^2
        dV/dt = (D * d2Vdx2 * C_m - I_ion + I_inj) / C_m

        V_new = V + dt * dV/dt

    Stability
    ---------
    The explicit Euler scheme is conditionally stable. For diffusion term
    roughly dt <= dx^2 / (2*D) is required. 

    Implementation notes
    --------------------
    - Uses `jax.lax.scan` over time so the kernel can be jitted and it GPU-ready.
    - Carry is the tuple (V, gates); `scan` collects V at each step.
    """

    def __init__(self) -> None:
        """Create an Euler solver instance."""
        pass

    @bench.benchmark(level=1)
    def solve(self, axon: AxonBase, tsim: float, dt: float) -> SimResult:
        """
        Execute an Euler simulation.

        Parameters
        ----------
        axon : AxonBase
            Model containing geometry, ion channel model and stimulation function.
        tsim : float
            Total simulation time in ms.
        dt : float
            Time step in ms.

        Returns
        -------
        SimResult
            object containing (V_all, t_vec) where V_all shape is (Nt, Nx).
        """
        Nt: int = int(jnp.ceil(tsim / dt))
        Nx: int = axon.Nx

        dtype_local = axon.ion_channel.dtype
        # Initial conditions
        V0: jnp.ndarray = jnp.full((Nx,), axon.Vinit, dtype=dtype_local)
        gates0: jnp.ndarray = axon.ion_channel.init_gates(V0_mV=V0)

        # Time vector
        t_vec: jnp.ndarray = jnp.arange(Nt, dtype=dtype_local) * dt

        # Extract ion channel helpers (function objects) once for efficiency
        g_funcs: Callable[..., jnp.ndarray] = axon.ion_channel.g_funcs
        E_rev: jnp.ndarray = axon.ion_channel.E_rev
        g_bar: jnp.ndarray = axon.ion_channel.g_bar
        alpha_fun: Callable[[jnp.ndarray], jnp.ndarray] = axon.ion_channel.alpha_funcs
        beta_fun: Callable[[jnp.ndarray], jnp.ndarray] = axon.ion_channel.beta_funcs
        q10: float = axon.ion_channel.q10
        Cm: float = axon.Cm

        dx2: float = axon.dx_cm ** 2

        def step(carry: Carry, n: int) -> Tuple[Carry, jnp.ndarray]:
            """
            Single Euler step executed inside `lax.scan`.

            Parameters
            ----------
            carry : tuple (V, gates)
                V : ndarray (Nx,)
                gates : ndarray (Nx, n_gates)
            n : int
                Current time-step index (0..Nt-1)

            Returns
            -------
            carry_out : tuple (V_new, gates_new)
            V_new : ndarray (Nx,)
                Voltage after this time step (also collected by scan).
            """
            V, gates = carry
            t: float = dtype_local(n) * dt

            # Gating variables update
            gates_new: jnp.ndarray = Gating.update_gates(
                gates=gates,
                alpha_fun=alpha_fun,
                beta_fun=beta_fun,
                V=V,
                dt=dt,
                q10=q10
            )

            # Spatial second derivative with Dirichlet BCs enforced by
            # leaving the endpoints unchanged (we set them explicitly below).
            d2Vdx2 = jnp.zeros_like(V).at[1:-1].set(
                (V[2:] - 2.0 * V[1:-1] + V[:-2]) / dx2
            )

            # Diffusive current 
            Idiff = axon.D * d2Vdx2 * Cm

            # Ionic currents from ion channel model (vectorized)
            Iion = Gating.compute_currents(
                V=V,
                gates=gates_new,
                g_bar=g_bar,
                g_func=g_funcs,
                E_rev=E_rev
            )

            # External injection current (vectorized over compartments)
            Iinj = axon.Iinj_uAcm2(t)

            # Time derivative and Euler update
            dVdt = (Idiff - Iion + Iinj) / Cm
            V_new = V + dt * dVdt

            # Enforce Dirichlet boundary conditions (fixed end potentials)
            V_new = V_new.at[0].set(axon.Vinit).at[-1].set(axon.Vinit)

            return (V_new, gates_new), V_new

        # Run scan: it returns (final_carry, all_outputs)
        (_, _), V_all = jax.lax.scan(step, (V0, gates0), jnp.arange(Nt))
        return SimResult(axon, V_all, t_vec)


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
    - Boundary conditions are enforced by modifying RHS rows and clamping the
      endpoints of the solution.
    - Gating variables are advanced with `half_step_gates` (CN-exp style)
      before evaluating ionic currents at the half-step.
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

    @bench.benchmark(level=1)
    def solve(self, axon: AxonBase, tsim: float, dt: float) -> SimResult:
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
        Nx: int = axon.Nx
        Nt: int = int(jnp.ceil(tsim / dt))

        dtype_local = axon.ion_channel.dtype

        V0: jnp.ndarray = jnp.full((Nx,), axon.Vinit, dtype=dtype_local)
        gates0: jnp.ndarray = axon.ion_channel.init_gates(V0_mV=V0)

        V_all: jnp.ndarray = jnp.zeros((Nt, Nx), dtype=dtype_local)
        t_vec: jnp.ndarray = jnp.arange(Nt, dtype=dtype_local) * dt

        dx2: float = axon.dx_cm ** 2
        alpha = axon.D * (dt / 2.0) / dx2

        # Build dense tridiagonal matrix A = I + 2*alpha on diag, -alpha off-diag
        A: jnp.ndarray = jnp.eye(Nx) * (1.0 + 2.0 * alpha)
        A = A.at[jnp.arange(1, Nx - 1), jnp.arange(0, Nx - 2)].set(-alpha)
        A = A.at[jnp.arange(1, Nx - 1), jnp.arange(2, Nx)].set(-alpha)
        # enforce Dirichlet rows
        A = A.at[0, :].set(0.0).at[0, 0].set(1.0)
        A = A.at[-1, :].set(0.0).at[-1, -1].set(1.0)

        def step(carry: Carry, n: int) -> Tuple[Carry, Tuple[jnp.ndarray, float]]:
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
            record : tuple (V_new, t_mid)
                t_mid is the mid-point time used for injected current.
            """
            V, gates = carry
            t_mid: float = dtype_local(n) * dt + dt / 2.0

            # half-step gating update (CN-exp style)
            gates_new: jnp.ndarray = Gating.half_step_gates(
                g_prev=gates,
                alpha_fun=axon.ion_channel.alpha_funcs,
                beta_fun=axon.ion_channel.beta_funcs,
                V=V,
                dt=dt,
                q10=axon.ion_channel.q10
            )

            # ionic currents computed at V (explicit) but using updated gates
            Iion: jnp.ndarray = Gating.compute_currents(
                V=V,
                gates=gates_new,
                g_bar=axon.ion_channel.g_bar,
                g_func=axon.ion_channel.g_funcs,
                E_rev=axon.ion_channel.E_rev
            )

            Iinj: jnp.ndarray = axon.Iinj_uAcm2(t_mid)

            rhs: jnp.ndarray = V + (dt / (2.0 * axon.Cm)) * (Iinj - Iion)

            # Enforce Dirichlet BC
            rhs = rhs.at[0].set(axon.Vinit).at[-1].set(axon.Vinit)

            # Solve linear system A x = rhs
            V_half: jnp.ndarray = jnp.linalg.solve(A, rhs)

            # Extrapolate to full step (explicit)
            V_new: jnp.ndarray = 2.0 * V_half - V
            V_new = V_new.at[0].set(axon.Vinit).at[-1].set(axon.Vinit)

            return (V_new, gates_new), (V_new, t_mid)

        (_, _), (V_all, _) = jax.lax.scan(step, (V0, gates0), jnp.arange(Nt))
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

    This implementation matches NEURON’s numerical strategy:
    - diffusion term treated implicitly using the Hines matrix,
    - membrane currents handled explicitly or semi-implicitly,
    - gating variables updated using a CN-exp (half-step exponential) rule.

    Implementation notes
    --------------------
    - The three coefficient vectors `dl, d, du` correspond to the lower,
      diagonal, and upper entries of the Hines matrix `(I - (dt/2) L)` and are
      constant in time for uniform compartments.
    - `half_step_gates` advances gating variables using a stable CN-exp update,
      supplying mid-step membrane state to the ionic current evaluation.
    - After solving for `V_half` with the tridiagonal solver, a standard
      CN extrapolation step computes the full step solution:

            V_{n+1} = 2 * V_half - V_n

    - Boundary conditions are applied by overwriting the first and last entries
      of the RHS and of the final voltage vector.
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

    def solve(self, axon: AxonBase, tsim: float, dt: float) -> SimResult:
        """
        Execute an optimized Crank–Nicolson simulation.

        Parameters
        ----------
        axon : AxonBase
            Axon model with ion channl model, geometry and stimulus.
        tsim : float
            Total simulation time in ms.
        dt : float
            Time step in ms.

        Returns
        -------
        SimResult
            Contains V_all (Nt × Nx) and t_vec (Nt).
        """
        Nx: int = axon.Nx
        Nt: int = int(jnp.ceil(tsim / dt))
        dtype_local = axon.ion_channel.dtype

        V0: jnp.ndarray = jnp.full((Nx,), axon.Vinit, dtype=dtype_local)
        gates0: jnp.ndarray = axon.ion_channel.init_gates(V0_mV=V0)

        dx2: float = axon.dx_cm ** 2
        alpha = axon.D * dt / (2.0 * dx2)

        # Tridiagonal diagonals (lower, diag, upper). First/last entries adjusted
        dl: jnp.ndarray = -alpha * jnp.ones((Nx,), dtype=dtype_local).at[0].set(0.0)
        d: jnp.ndarray = (1.0 + 2.0 * alpha) * jnp.ones((Nx,), dtype=dtype_local)
        du: jnp.ndarray = -alpha * jnp.ones((Nx,), dtype=dtype_local).at[-1].set(0.0)

        Cm: float = axon.Cm
        Vinit: float = axon.Vinit
        coef: float = dt / (2.0 * Cm)

        # Pre-extract references for speed / clarity
        alpha_fun = axon.ion_channel.alpha_funcs
        beta_fun = axon.ion_channel.beta_funcs
        g_bar = axon.ion_channel.g_bar
        g_func = axon.ion_channel.g_funcs
        E_rev = axon.ion_channel.E_rev
        q10 = axon.ion_channel.q10
        inj_fun = axon.Iinj_uAcm2

        @jax.jit
        def step(carry: Carry, n: int) -> Tuple[Carry, jnp.ndarray]:
            """
            Jitted inner step used by `lax.scan`.

            Parameters
            ----------
            carry : tuple (V, gates)
            n : int
                time index.

            Returns
            -------
            carry_out : tuple (Vm_new, gates_new)
            Vm_new : ndarray
                Voltage (Nx,) stored by scan as output.
            """
            V, gates = carry
            t_mid: float = dtype_local(n) * dt + dt / 2.0

            # Stable half-step gating update
            gates_new: jnp.ndarray = Gating.half_step_gates(
                g_prev=gates,
                alpha_fun=alpha_fun,
                beta_fun=beta_fun,
                V=V,
                dt=dt,
                q10=q10
            )

            # Ionic current at current voltage with updated gates
            Iion: jnp.ndarray = Gating.compute_currents(
                V=V,
                gates=gates_new,
                g_bar=g_bar,
                g_func=g_func,
                E_rev=E_rev
            )

            # Injected current (midpoint)
            Iinj: jnp.ndarray = inj_fun(t_mid)

            # Right-hand side for tridiagonal solve
            rhs: jnp.ndarray = V + coef * (Iinj - Iion)
            rhs = rhs.at[0].set(Vinit).at[-1].set(Vinit)

            # Solve tridiagonal system; note tridiagonal_solve expects (N, M)
            Vm_half: jnp.ndarray = jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:, None])[:, 0]
            Vm_new: jnp.ndarray = 2.0 * Vm_half - V
            Vm_new = Vm_new.at[0].set(Vinit).at[-1].set(Vinit)

            return (Vm_new, gates_new), Vm_new

        def run_scan(V_init: jnp.ndarray, gates_init: jnp.ndarray, Nt_local: int) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
            """
            Run the time loop using JAX scan.

            Parameters
            ----------
            V_init : ndarray
                Initial voltage vector (Nx,)
            gates_init : ndarray
                Initial gating variables (Nx, n_gates)
            Nt_local : int
                Number of time steps (compile-time Python int)

            Returns
            -------
            (final_carry, V_all) : tuple
                final_carry: (V_final, gates_final)
                V_all: stacked voltages (Nt, Nx)
            """
            return jax.lax.scan(step, (V_init, gates_init), jnp.arange(Nt_local))

        # Execute the scan (Nt must be a Python int for compilation stability)
        (_, _), V_all = run_scan(V0, gates0, Nt)
        t_vec: jnp.ndarray = jnp.arange(Nt, dtype=dtype_local) * dt

        return SimResult(axon, V_all, t_vec)
