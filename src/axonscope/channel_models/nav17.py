from __future__ import annotations
import jax.numpy as jnp
from axonscope.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.icm import Gating


class NaV17ICM(IonChannelModelBase):
    """NaV1.7 TTX-sensitive Na channel (Sheets et al. 2007 / nattxs.mod).

    Gates: m (activation), h (fast inactivation), s (slow inactivation).
    Current: I = gbar * m³ * h * s * (v - E_Na)

    Temperature correction: Q10 = 2.5, ref 21°C, baked into alpha/beta.
    """

    def __init__(
        self,
        gbar: float = 0.10664,
        ena: float = 71.5,
        celsius: float = 37.0,
        shift: float = 0.0,
        Tshift: float = 0.0,
    ) -> None:
        super().__init__()
        self.gbar = dtype(gbar)
        self.ena = dtype(ena)
        self.celsius = dtype(celsius)
        self.shift = dtype(shift)
        self.Tshift = dtype(Tshift)
        self._qt = dtype(2.5 ** ((celsius - 21.0) / 10.0))
        self.q10 = dtype(1.0)  # temp correction already inside alpha/beta

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        Vs = V + self.shift
        am = dtype(15.5) / (1.0 + jnp.exp((Vs + dtype(-5.0)) / dtype(-12.08)))
        ah = dtype(0.38685) / (1.0 + jnp.exp((Vs + dtype(122.35)) / dtype(15.29)))
        as_ = dtype(0.00003) + dtype(0.00092) / (
            1.0 + jnp.exp((Vs + dtype(93.9) + self.Tshift) / dtype(16.6))
        )
        return jnp.stack([am, ah, as_], axis=-1) * self._qt

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        Vs = V + self.shift
        bm = dtype(35.2) / (1.0 + jnp.exp((Vs + dtype(72.7)) / dtype(16.7)))
        bh = dtype(-0.00283) + dtype(2.00283) / (
            1.0 + jnp.exp((Vs + dtype(5.5266)) / dtype(-12.70195))
        )
        bh = jnp.maximum(bh, dtype(0.0))
        bs = dtype(132.05) + dtype(-132.05) / (
            1.0 + jnp.exp((Vs + dtype(-384.9) + self.Tshift) / dtype(28.5))
        )
        return jnp.stack([bm, bh, bs], axis=-1) * self._qt

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        m, h, s = gates[:, 0], gates[:, 1], gates[:, 2]
        return (g_bar[0] * m ** 3 * h * s)[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = Gating.rates(jnp.atleast_1d(V0_mV), self.q10, self.alpha_funcs, self.beta_funcs)
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self.gbar], dtype=dtype) * 1e3

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self.ena], dtype=dtype)
