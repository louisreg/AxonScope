from __future__ import annotations
import jax.numpy as jnp
from axonscope.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.icm import Gating


class KDRTigerICM(IonChannelModelBase):
    """K delayed-rectifier for Tigerholm C-fibers (Sheets 2007 / kdr_Tiger.mod).

    Gate: n (1 gate, power 4).
    Current: I = gbar * n^4 * (v - E_K)

    Temperature correction: Q10 = 3.3, ref 22°C, baked into alpha/beta.
    """

    def __init__(
        self,
        gbar: float = 0.018002,
        ek: float = -87.0,
        celsius: float = 37.0,
    ) -> None:
        super().__init__()
        self.gbar = dtype(gbar)
        self.ek = dtype(ek)
        self.celsius = dtype(celsius)
        self._qt = dtype(3.3 ** ((celsius - 22.0) / 10.0))
        self.q10 = dtype(1.0)
        self._k1 = dtype(15.4)
        self._Vh = dtype(35.0)

    def _inf_tau(self, V: jnp.ndarray):
        ninf = dtype(1.0) / (1.0 + jnp.exp((V + self._Vh - dtype(10.0)) / (-self._k1)))

        # piecewise tau — guard exp arguments against overflow in non-selected branch
        tau_high = dtype(0.16) + dtype(0.8) * jnp.exp(-dtype(0.0267) * (V + dtype(11.0)))
        exp1 = jnp.exp(jnp.clip((V + dtype(75.2)) / dtype(6.5), -50.0, 50.0))
        exp2 = jnp.exp(jnp.clip((V - dtype(131.5)) / dtype(-34.8), -50.0, 50.0))
        tau_low = dtype(1000.0) * (
            dtype(0.000688) + dtype(1.0) / jnp.maximum(exp1 + exp2, dtype(1e-12))
        )
        tau = jnp.where(V >= dtype(-31.0), tau_high, tau_low)
        return ninf, tau

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        ninf, tau = self._inf_tau(V)
        alpha_n = ninf / jnp.maximum(tau, dtype(1e-6))
        return alpha_n[:, None] * self._qt

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        ninf, tau = self._inf_tau(V)
        beta_n = (dtype(1.0) - ninf) / jnp.maximum(tau, dtype(1e-6))
        return beta_n[:, None] * self._qt

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        n = gates[:, 0]
        return (g_bar[0] * n ** 4)[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = Gating.rates(jnp.atleast_1d(V0_mV), self.q10, self.alpha_funcs, self.beta_funcs)
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self.gbar], dtype=dtype) * 1e3

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self.ek], dtype=dtype)
