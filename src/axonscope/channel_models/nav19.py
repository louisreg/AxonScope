from __future__ import annotations
import jax.numpy as jnp
from axonscope.utils.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.icm import Gating


class NaV19ICM(IonChannelModelBase):
    """NaV1.9 TTX-resistant persistent Na channel (Herzog et al. 2001 / nav1p9.mod).

    Gates: m (activation, power 1), h (inactivation), s (ultra-slow inactivation).
    Current: I = gbar * m * h * s * (v - E_Na)

    Temperature correction: Q10 = 2.5, ref 21°C, baked into alpha/beta.
    """

    def __init__(
        self,
        gbar: float = 9.4779e-05,
        ena: float = 71.5,
        celsius: float = 37.0,
        NGFshift: float = 0.0,
    ) -> None:
        super().__init__()
        self.gbar = dtype(gbar)
        self.ena = dtype(ena)
        self.celsius = dtype(celsius)
        self.NGFshift = dtype(NGFshift)
        self._qt = dtype(2.5 ** ((celsius - 21.0) / 10.0))
        self.q10 = dtype(1.0)

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        Vs = V + self.NGFshift
        am = dtype(1.032) / (1.0 + jnp.exp((Vs + dtype(6.99)) / dtype(-14.87115)))
        ah = dtype(0.06435) / (1.0 + jnp.exp((Vs + dtype(73.26415)) / dtype(3.71928)))
        # gate=0, B_as9=0 → alpha_s = A_as9 * exp(-V/C_as9)
        as_ = dtype(0.00000016) * jnp.exp(-V / dtype(12.0))
        return jnp.stack([am, ah, as_], axis=-1) * self._qt

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        Vs = V + self.NGFshift
        bm = dtype(5.79) / (1.0 + jnp.exp((Vs + dtype(130.4)) / dtype(22.9)))
        bh = dtype(0.13496) / (1.0 + jnp.exp((Vs + dtype(10.27853)) / dtype(-9.09334)))
        bs = dtype(0.0005) / (1.0 + jnp.exp(-(V + dtype(32.0)) / dtype(23.0)))
        return jnp.stack([bm, bh, bs], axis=-1) * self._qt

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        # m^1 (not m^3) — persistent channel
        m, h, s = gates[:, 0], gates[:, 1], gates[:, 2]
        return (g_bar[0] * m * h * s)[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = Gating.rates(jnp.atleast_1d(V0_mV), self.q10, self.alpha_funcs, self.beta_funcs)
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self.gbar], dtype=dtype) * 1e3

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self.ena], dtype=dtype)
