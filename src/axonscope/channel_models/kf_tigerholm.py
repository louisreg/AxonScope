from __future__ import annotations
import jax.numpy as jnp
from axonscope.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.icm import Gating


class KFastICM(IonChannelModelBase):
    """K fast A-type channel (Winkelman 2005 / Gold 1996 / kf.mod).

    Gates: m (activation, stored as m_raw^4), h (inactivation).
    Current: I = gbar * m * h * (v - E_K)   [m is already 4th-power]

    Temperature correction: Q10 = 3.3, ref 23°C, baked into alpha/beta.
    """

    def __init__(
        self,
        gbar: float = 0.012756,
        ek: float = -87.0,
        celsius: float = 37.0,
        shift: float = -15.0,
        lj: float = 0.0,
    ) -> None:
        super().__init__()
        self.gbar = dtype(gbar)
        self.ek = dtype(ek)
        self.celsius = dtype(celsius)
        self.shift = dtype(shift)
        self.lj = dtype(lj)
        self._qt = dtype(3.3 ** ((celsius - 23.0) / 10.0))
        self.q10 = dtype(1.0)
        # channel parameters
        self._vhm = dtype(-5.4)
        self._vhh = dtype(-49.9)
        self._km  = dtype(16.4)
        self._kh  = dtype(4.6)

    def _inf_tau(self, V: jnp.ndarray):
        adj = self.lj + self.shift
        # minf = (sigmoid)^4 — 4th power baked in as per kf.mod
        minf = (dtype(1.0) / (
            1.0 + jnp.exp(-(dtype(1.0) / self._km) * (V - self._vhm + adj))
        )) ** 4
        tau_m = dtype(0.25) + dtype(10.04) * jnp.exp(
            -((V + dtype(24.67)) ** 2) / (dtype(2.0) * dtype(34.8) ** 2)
        )

        hinf = dtype(1.0) / (
            1.0 + jnp.exp((dtype(1.0) / self._kh) * (V - self._vhh + adj))
        )
        tau_h_raw = dtype(20.0) + dtype(50.0) * jnp.exp(
            -((V + dtype(40.0)) ** 2) / (dtype(2.0) * dtype(40.0) ** 2)
        )
        tau_h = jnp.maximum(tau_h_raw, dtype(5.0))

        return minf, tau_m, hinf, tau_h

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        minf, tau_m, hinf, tau_h = self._inf_tau(V)
        alpha_m = minf / jnp.maximum(tau_m, dtype(1e-6))
        alpha_h = hinf / jnp.maximum(tau_h, dtype(1e-6))
        return jnp.stack([alpha_m, alpha_h], axis=-1) * self._qt

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        minf, tau_m, hinf, tau_h = self._inf_tau(V)
        beta_m = (dtype(1.0) - minf) / jnp.maximum(tau_m, dtype(1e-6))
        beta_h = (dtype(1.0) - hinf) / jnp.maximum(tau_h, dtype(1e-6))
        return jnp.stack([beta_m, beta_h], axis=-1) * self._qt

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        m, h = gates[:, 0], gates[:, 1]
        return (g_bar[0] * m * h)[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = Gating.rates(jnp.atleast_1d(V0_mV), self.q10, self.alpha_funcs, self.beta_funcs)
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self.gbar], dtype=dtype) * 1e3

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self.ek], dtype=dtype)
