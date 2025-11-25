from __future__ import annotations
import jax.numpy as jnp
from typing import Tuple, Callable

# ---------------------- Type aliases ----------------------
Array1D = jnp.ndarray  # shape (N,)
Array2D = jnp.ndarray  # shape (N, n_gates)
GFunc = Callable[[Array2D, Array1D], Array2D]
RateFunc = Callable[[Array1D], Array2D]


class Gating:
    """
    Utility class for gating variable dynamics and ionic currents.

    Provides static methods to compute:
    - Ionic currents from voltage and gating variables
    - Steady-state values and time constants of gates
    - Full-step CNEXP update
    - Half-step CNEXP update (for Crank-Nicholson solvers)

    All methods are vectorized for JAX arrays.
    """

    @staticmethod
    def compute_currents(
        V: Array1D,
        gates: Array2D,
        g_bar: Array1D,
        g_func: GFunc,
        E_rev: Array1D
    ) -> Array1D:
        """
        Compute total ionic currents for each compartment.

        Implements:
            I_ion = Σ_i g_i(V, gates) * (V - E_i)
        where g_i is computed from the gating variables.

        Parameters
        ----------
        V : Array1D, shape (N,)
            Membrane voltage (mV) for each compartment.
        gates : Array2D, shape (N, n_gates)
            Current gating variable values.
        g_bar : Array1D, shape (n_channels,)
            Maximum conductances for each channel.
        g_func : Callable
            Function computing conductances from gating variables: g_func(gates, g_bar).
        E_rev : Array1D, shape (n_channels,)
            Reversal potentials for each channel (mV).

        Returns
        -------
        I_tot : Array1D, shape (N,)
            Total ionic current for each compartment (µA/cm²).
        """
        V = jnp.atleast_1d(V)
        g_vals = g_func(gates, g_bar)          # shape (N, n_channels)
        delta_V = V[:, None] - E_rev[None, :]  # broadcasting
        I_tot = jnp.sum(g_vals * delta_V, axis=1)
        return I_tot

    @staticmethod
    def rates(
        V: Array1D,
        q10: float,
        alpha_fun: RateFunc,
        beta_fun: RateFunc
    ) -> Tuple[Array2D, Array2D]:
        """
        Compute steady-state values and time constants for gating variables.

        Gating dynamics follow:
            dg/dt = α(V) * (1 - g) - β(V) * g
        which can be rewritten as:
            dg/dt = (g_inf - g) / τ
        with:
            g_inf = α / (α + β)
            τ     = 1 / (q10 * (α + β))

        Parameters
        ----------
        V : Array1D, shape (N,)
            Membrane voltage (mV) for each compartment.
        q10 : float
            Temperature scaling factor.
        alpha_fun : Callable
            Function returning α(V), shape (N, n_gates)
        beta_fun : Callable
            Function returning β(V), shape (N, n_gates)

        Returns
        -------
        g_inf : Array2D, shape (N, n_gates)
            Steady-state gating values.
        tau : Array2D, shape (N, n_gates)
            Time constants (ms) for each gating variable.
        """
        alpha = alpha_fun(V)
        beta = beta_fun(V)
        sum_ab = jnp.maximum(alpha + beta, 1e-12)
        g_inf = alpha / sum_ab
        tau = 1.0 / (q10 * sum_ab)
        return g_inf, tau

    @staticmethod
    def update_gates(
        gates: Array2D,
        V: Array1D,
        dt: float,
        q10: float,
        alpha_fun: RateFunc,
        beta_fun: RateFunc
    ) -> Array2D:
        """
        CNEXP update for gating variables (full time step).

        Implements:
            g(t + dt) = g_inf - (g_inf - g) * exp(-dt / tau)

        Parameters
        ----------
        gates : Array2D, shape (N, n_gates)
            Current gating values.
        V : Array1D, shape (N,)
            Membrane voltage (mV).
        dt : float
            Time step (ms).
        q10 : float
            Temperature scaling factor.
        alpha_fun : Callable
            Function computing alpha rates α(V), shape (N, n_gates)
        beta_fun : Callable
            Function computing beta rates β(V), shape (N, n_gates)

        Returns
        -------
        gates_new : Array2D, shape (N, n_gates)
            Updated gating variables after dt.
        """
        g_inf, tau = Gating.rates(V, q10, alpha_fun, beta_fun)
        return g_inf - (g_inf - gates) * jnp.exp(-dt / tau)

    @staticmethod
    def half_step_gates(
        g_prev: Array2D,
        V: Array1D,
        dt: float,
        alpha_fun: RateFunc,
        beta_fun: RateFunc,
        q10: float = 1.0
    ) -> Array2D:
        """
        Half-step CNEXP update for gating variables.

        Useful for Crank-Nicholson-style solvers where ionic currents are
        evaluated at a half time-step.

        Implements:
            g_half = (α/d) + ((1/dt - 0.5*(α+β))/d) * g_prev
            with d = (1/dt) + 0.5*(α+β)

        Parameters
        ----------
        g_prev : Array2D, shape (N, n_gates)
            Previous gating values.
        V : Array1D, shape (N,)
            Membrane voltage (mV).
        dt : float
            Time step (ms).
        alpha_fun : Callable
            Function computing alpha rates α(V), shape (N, n_gates)
        beta_fun : Callable
            Function computing beta rates β(V), shape (N, n_gates)
        q10 : float, optional
            Temperature scaling factor for rates, default is 1.0

        Returns
        -------
        g_half : Array2D, shape (N, n_gates)
            Updated gating variables at half-step.
        """
        alpha = q10 * alpha_fun(V)
        beta = q10 * beta_fun(V)
        denom = jnp.maximum(1.0/dt + 0.5*(alpha + beta), 1e-12)
        term1 = alpha / denom
        term2 = ((1.0/dt) - 0.5*(alpha + beta)) / denom * g_prev
        return term1 + term2
