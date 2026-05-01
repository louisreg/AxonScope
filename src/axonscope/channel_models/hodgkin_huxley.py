from __future__ import annotations
import jax.numpy as jnp
from axonscope.utils.math_functions import vtrap_jax as vtrap
from axonscope.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.icm import Gating

class HodgkinHuxleyICM(IonChannelModelBase):
    """
    Hodgkin-Huxley ion channel model for the squid giant axon.

    This class implements the classical Hodgkin-Huxley (HH) ionic model, 
    including sodium (Na⁺), potassium (K⁺), and leak (L) currents, 
    following the original hh.hoc implementation from NEURON.

    The model is JAX-compatible: all parameters and outputs are stored 
    as `dtype` for vectorization and JIT compilation.

    Ionic currents are computed as:

        I_Na = g_Na * m³ * h * (V - E_Na)
        I_K  = g_K  * n⁴     * (V - E_K)
        I_L  = g_L           * (V - E_L)

    where m,h,n are gating variables with voltage-dependent kinetics:

        dm/dt = α_m(V) * (1 - m) - β_m(V) * m
        dh/dt = α_h(V) * (1 - h) - β_h(V) * h
        dn/dt = α_n(V) * (1 - n) - β_n(V) * n

    The rate functions α and β follow Hodgkin-Huxley empirical formulas.

    Parameters
    ----------
    gnabar : float, optional
        Maximum sodium conductance in S/cm² (default 0.12)
    gkbar : float, optional
        Maximum potassium conductance in S/cm² (default 0.036)
    gl : float, optional
        Leak conductance in S/cm² (default 0.0003)
    el : float, optional
        Leak reversal potential in mV (default -54.3)
    ena : float, optional
        Sodium reversal potential in mV (default 50.0)
    ek : float, optional
        Potassium reversal potential in mV (default -77.0)
    celsius : float, optional
        Temperature in °C for Q10 scaling of gating rates (default 6.3)

    Attributes
    ----------
    gnabar : float
        Maximum sodium conductance (S/cm²)
    gkbar : float
        Maximum potassium conductance (S/cm²)
    gl : float
        Leak conductance (S/cm²)
    el : float
        Leak reversal potential (mV)
    ena : float
        Sodium reversal potential (mV)
    ek : float
        Potassium reversal potential (mV)
    celsius : float
        Temperature (°C) used for Q10 gating scaling
    q10 : float
        Temperature factor for gating kinetics, Q10^((celsius-6.3)/10)
    """

    def __init__(
        self,
        gnabar: float = 0.12,
        gkbar: float = 0.036,
        gl: float = 0.0003,
        el: float = -54.3,
        ena: float = 50.0,
        ek: float = -77.0,
        celsius: float = 6.3
    ) -> None:
        """
        Initialize HH ion channel parameters.
        """
        super().__init__()
        self.gnabar: float = dtype(gnabar)
        self.gkbar: float = dtype(gkbar)
        self.gl: float = dtype(gl)
        self.el: float = dtype(el)
        self.ena: float = dtype(ena)
        self.ek: float = dtype(ek)
        self.celsius: float = dtype(celsius)
        self.q10: float = dtype(3.0 ** ((celsius - 6.3) / 10.0))

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        """
        Compute alpha rate constants for m, h, n gating variables.

        Parameters
        ----------
        V : jnp.ndarray, shape (N,)
            Membrane voltage in mV.

        Returns
        -------
        alpha : jnp.ndarray, shape (N,3)
            Alpha rates for [m, h, n] gates (ms⁻¹).

        Equations
        ---------
        α_m = 0.1 * (-(V+40)) / (exp(-(V+40)/10) - 1)
        α_h = 0.07 * exp(-(V+65)/20)
        α_n = 0.01 * (-(V+55)) / (exp(-(V+55)/10) - 1)
        """
        m = 0.1 * vtrap(-(V + 40.0), 10.0)
        h = 0.07 * jnp.exp(-(V + 65.0)/20.0)
        n = 0.01 * vtrap(-(V + 55.0), 10.0)
        return jnp.stack([m, h, n], axis=-1)

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        """
        Compute beta rate constants for m, h, n gating variables.

        Parameters
        ----------
        V : jnp.ndarray, shape (N,)
            Membrane voltage in mV.

        Returns
        -------
        beta : jnp.ndarray, shape (N,3)
            Beta rates for [m, h, n] gates (ms⁻¹).

        Equations
        ---------
        β_m = 4 * exp(-(V+65)/18)
        β_h = 1 / (exp(-(V+35)/10) + 1)
        β_n = 0.125 * exp(-(V+65)/80)
        """
        m = 4.0 * jnp.exp(-(V + 65.0)/18.0)
        h = 1.0 / (jnp.exp(-(V + 35.0)/10.0) + 1.0)
        n = 0.125 * jnp.exp(-(V + 65.0)/80.0)
        return jnp.stack([m, h, n], axis=-1)

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        """
        Compute conductances for HH channels.

        Parameters
        ----------
        gates : jnp.ndarray, shape (N,3)
            Gating variables [m, h, n].
        g_bar : jnp.ndarray, shape (3,)
            Maximum conductances [g_Na, g_K, g_L] (S/cm²).

        Returns
        -------
        g : jnp.ndarray, shape (N,3)
            Channel conductances at each node.

        Equations
        ---------
        g_Na = g_Na_bar * m³ * h
        g_K  = g_K_bar  * n⁴
        g_L  = g_L_bar
        """
        g_na = g_bar[0] * jnp.power(gates[:, 0], 3) * gates[:, 1]
        g_k  = g_bar[1] * jnp.power(gates[:, 2], 4)
        g_l  = jnp.full((gates.shape[0],), g_bar[2], dtype=dtype)
        return jnp.stack([g_na, g_k, g_l], axis=-1)

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        """
        Initialize gating variables at steady-state given V0.

        Parameters
        ----------
        V0_mV : jnp.ndarray, shape (N,)
            Initial membrane voltage (mV).

        Returns
        -------
        g_inf : jnp.ndarray, shape (N,3)
            Steady-state gating variables [m_inf, h_inf, n_inf] for each node.

        Notes
        -----
        Uses the `rates` helper function with Q10 scaling to compute
        steady-state gate values at the initial voltage.
        """
        g_inf, _ = Gating.rates(V0_mV, self.q10, self.alpha_funcs, self.beta_funcs)
        return g_inf

    def gate_names(self) -> tuple[str, ...]:
        return ("m", "h", "n")

    def conductance_names(self) -> tuple[str, ...]:
        return ("g_na", "g_k", "g_l")

    def current_names(self) -> tuple[str, ...]:
        return ("I_na", "I_k", "I_l")

    @property
    def g_bar(self) -> jnp.ndarray:
        """
        Maximum conductances for HH channels.

        Returns
        -------
        g_bar : jnp.ndarray, shape (3,)
            Maximum conductances [g_Na, g_K, g_L] in mS/cm².
        """
        return jnp.array([self.gnabar, self.gkbar, self.gl], dtype=dtype) * 1e3

    @property
    def E_rev(self) -> jnp.ndarray:
        """
        Reversal potentials for HH channels.

        Returns
        -------
        E_rev : jnp.ndarray, shape (3,)
            [E_Na, E_K, E_L] in mV.
        """
        return jnp.array([self.ena, self.ek, self.el], dtype=dtype)

