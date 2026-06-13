from __future__ import annotations
import jax.numpy as jnp
from axonscope.utils.math_functions import vtrap_jax as vtrap
from axonscope.utils.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.icm import Gating

class RattayAberhamICM(IonChannelModelBase):
    """
    Rattay-Aberham ion channel model (from RattayAberham.mod).

    This model describes the ionic currents of a mammalian axon using
    voltage-gated sodium and potassium channels plus a passive leak.

    The ionic currents are computed as:

        I_Na = g_Na * m³ * h * (V - E_Na)
        I_K  = g_K  * n⁴     * (V - E_K)
        I_L  = g_L           * (V - E_L)

    where m,h,n are gating variables updated via:

        dm/dt = α_m(V) * (1 - m) - β_m(V) * m
        dh/dt = α_h(V) * (1 - h) - β_h(V) * h
        dn/dt = α_n(V) * (1 - n) - β_n(V) * n

    Notes
    -----
    - Voltage `V` is in mV.
    - Ionic currents `I_ion` are in µA/cm².
    - Gating variables must be updated explicitly using a time-stepping
      function, e.g., `step_gates(dt_ms, V)`.
    - The Q10 temperature factor is used to scale the gating kinetics:

        q10 = 2.24659524757^((celsius - 6.3)/10)

    Parameters
    ----------
    gnabar : float, optional
        Maximum sodium conductance (S/cm²), default 0.12
    gkbar : float, optional
        Maximum potassium conductance (S/cm²), default 0.036
    gl : float, optional
        Leak conductance (S/cm²), default 0.0003
    el : float, optional
        Leak reversal potential (mV), default -59.4
    ena : float, optional
        Sodium reversal potential (mV), default 45.0
    ek : float, optional
        Potassium reversal potential (mV), default -82.0
    celsius : float, optional
        Temperature in °C for Q10 scaling, default 37.0

    Methods
    -------
    alpha_funcs(V)
        Compute voltage-dependent alpha rates for m,h,n gates.
    beta_funcs(V)
        Compute voltage-dependent beta rates for m,h,n gates.
    g_funcs(gates, g_bar)
        Compute conductances for each channel from gating variables.
    init_gates(V0_mV)
        Compute steady-state initial values for gating variables.
    
    Properties
    ----------
    g_bar
        Maximum conductances [g_Na, g_K, g_L] in mS/cm².
    E_rev
        Reversal potentials [E_Na, E_K, E_L] in mV.
    """

    def __init__(
        self,
        gnabar: float = 0.12,
        gkbar: float = 0.036,
        gl: float = 0.0003,
        el: float = -59.4,
        ena: float = 45.0,
        ek: float = -82.0,
        celsius: float = 37.0
    ) -> None:
        """
        Initialize ion channel parameters and Q10 temperature scaling.
        """
        super().__init__()
        self.gnabar: float = dtype(gnabar)
        self.gkbar: float = dtype(gkbar)
        self.gl: float = dtype(gl)
        self.el: float = dtype(el)
        self.ena: float = dtype(ena)
        self.ek: float = dtype(ek)
        self.celsius: float = dtype(celsius)
        self.q10: float = dtype(2.24659524757 ** ((celsius - 6.3) / 10))
        
    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        """
        Compute alpha rates for m, h, n gates.

        Parameters
        ----------
        V : jnp.ndarray, shape (N,)
            Membrane voltage in mV.

        Returns
        -------
        alpha : jnp.ndarray, shape (N,3)
            Alpha rates [α_m, α_h, α_n] in ms⁻¹.

        Equations
        ---------
        α_m = vtrap(2.5 - 0.1*(V+70), 1)
        α_h = 0.07 * exp(-(V+70)/20)
        α_n = 0.1 * vtrap(1.0 - 0.1*(V+70), 1)
        """
        m = vtrap(2.5 - 0.1 * (V + 70.0), 1.0)
        h = 0.07 * jnp.exp(-(V + 70.0) / 20.0)
        n = 0.1 * vtrap(1.0 - 0.1 * (V + 70.0), 1.0)
        return jnp.stack([m, h, n], axis=-1)

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        """
        Compute beta rates for m, h, n gates.

        Parameters
        ----------
        V : jnp.ndarray, shape (N,)
            Membrane voltage in mV.

        Returns
        -------
        beta : jnp.ndarray, shape (N,3)
            Beta rates [β_m, β_h, β_n] in ms⁻¹.

        Equations
        ---------
        β_m = 4 * exp(-(V+70)/18)
        β_h = 1 / (exp(3 - 0.1*(V+70)) + 1)
        β_n = 0.125 * exp(-(V+70)/80)
        """
        m = 4.0 * jnp.exp(-(V + 70.0) / 18.0)
        h = 1.0 / (jnp.exp(3.0 - 0.1 * (V + 70.0)) + 1.0)
        n = 0.125 * jnp.exp(-(V + 70.0) / 80.0)
        return jnp.stack([m, h, n], axis=-1)

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        """
        Compute channel conductances from gating variables.

        Parameters
        ----------
        gates : jnp.ndarray, shape (N,3)
            Gating variables [m, h, n].
        g_bar : jnp.ndarray, shape (3,)
            Maximum conductances [g_Na, g_K, g_L] in mS/cm².

        Returns
        -------
        g : jnp.ndarray, shape (N,3)
            Conductances for each channel in mS/cm².

        Equations
        ---------
        g_Na = g_Na_bar * m³ * h
        g_K  = g_K_bar  * n⁴
        g_L  = g_L_bar (constant)
        """
        g_na = g_bar[0] * jnp.power(gates[:, 0], 3) * gates[:, 1]
        g_k  = g_bar[1] * jnp.power(gates[:, 2], 4)
        g_l  = jnp.full((gates.shape[0],), g_bar[2], dtype=dtype)
        return jnp.stack([g_na, g_k, g_l], axis=-1).astype(dtype)

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        """
        Initialize gating variables at steady state.

        Parameters
        ----------
        V0_mV : jnp.ndarray, shape (N,)
            Initial membrane voltage in mV.

        Returns
        -------
        gates_init : jnp.ndarray, shape (N,3)
            Steady-state gating variables [m_inf, h_inf, n_inf].
        """
        V0 = jnp.atleast_1d(V0_mV)
        g_inf, _ = Gating.rates(V0, self.q10, self.alpha_funcs, self.beta_funcs)
        return g_inf

    def gate_names(self) -> tuple[str, ...]:
        return ("m", "h", "n")

    def conductance_names(self) -> tuple[str, ...]:
        return ("g_na", "g_k", "g_l")

    def current_names(self) -> tuple[str, ...]:
        return ("I_na", "I_k", "I_l")

    def final_gate_update(
        self,
        gates_prev: jnp.ndarray,
        V_mV_prev: jnp.ndarray,
        V_mV_new: jnp.ndarray,
        dt: float,
        gates_predictor: jnp.ndarray,
    ) -> jnp.ndarray:
        _ = V_mV_prev, gates_predictor
        return self.cn_gate_update(g_prev=gates_prev, V_mV=V_mV_new, dt=dt)

    @property
    def g_bar(self) -> jnp.ndarray:
        """
        Maximum channel conductances.

        Returns
        -------
        g_bar : jnp.ndarray, shape (3,)
            [g_Na, g_K, g_L] in mS/cm²
        """
        return jnp.array([self.gnabar, self.gkbar, self.gl], dtype=dtype) * 1e3

    @property
    def E_rev(self) -> jnp.ndarray:
        """
        Channel reversal potentials.

        Returns
        -------
        E_rev : jnp.ndarray, shape (3,)
            [E_Na, E_K, E_L] in mV
        """
        return jnp.array([self.ena, self.ek, self.el], dtype=dtype)
