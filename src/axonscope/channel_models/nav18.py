from __future__ import annotations
import jax.numpy as jnp
from axonscope.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.icm import Gating


class NaV18ICM(IonChannelModelBase):
    """NaV1.8 TTX-resistant Na channel (Sheets 2007 + Delmas / DNav18.mod).

    Gates: m (activation), h (fast inact.), s (slow inact.), u (ultra-slow inact.).
    Current: I = gbar * m³ * h * s * u * (v - E_Na)

    Temperature correction: Q10 = 2.5, ref 22°C, baked into alpha/beta.

    s and u gates: DNav18.mod defines sinf/uinf as independent sigmoids, distinct
    from alpha_s/(alpha_s+beta_s). The raw alpha/beta are used only for tau.
    Effective HH rates: alpha_eff = ginf*(alpha_raw+beta_raw), beta_eff = (1-ginf)*(...).
    """

    def __init__(
        self,
        gbar: float = 0.24271,
        ena: float = 71.5,
        celsius: float = 37.0,
    ) -> None:
        super().__init__()
        self.gbar = dtype(gbar)
        self.ena = dtype(ena)
        self.celsius = dtype(celsius)
        self._qt = dtype(2.5 ** ((celsius - 22.0) / 10.0))
        self.q10 = dtype(1.0)

    def _su_rates(self, V: jnp.ndarray):
        """Compute effective alpha/beta for s and u gates using sinf/uinf sigmoids."""
        sinf = dtype(1.0) / (1.0 + jnp.exp((V + dtype(45.0)) / dtype(8.0)))
        as_raw = dtype(0.001) * dtype(5.4203) / (
            1.0 + jnp.exp((V + dtype(79.816)) / dtype(16.269))
        )
        bs_raw = dtype(0.001) * dtype(5.0757) / (
            1.0 + jnp.exp(-(V + dtype(15.968)) / dtype(11.542))
        )
        sum_s = jnp.maximum(as_raw + bs_raw, dtype(1e-12))
        alpha_s = sinf * sum_s
        beta_s = (dtype(1.0) - sinf) * sum_s

        uinf = dtype(1.0) / (1.0 + jnp.exp((V + dtype(51.0)) / dtype(8.0)))
        au_raw = dtype(0.0002) * dtype(2.0434) / (
            1.0 + jnp.exp((V + dtype(67.499)) / dtype(19.51))
        )
        bu_raw = dtype(0.0002) * dtype(1.9952) / (
            1.0 + jnp.exp(-(V + dtype(30.963)) / dtype(14.792))
        )
        sum_u = jnp.maximum(au_raw + bu_raw, dtype(1e-12))
        alpha_u = uinf * sum_u
        beta_u = (dtype(1.0) - uinf) * sum_u

        return alpha_s, beta_s, alpha_u, beta_u

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        am = dtype(2.85) - dtype(2.839) / (1.0 + jnp.exp((V - dtype(1.159)) / dtype(13.95)))
        bm = dtype(7.6205) / (1.0 + jnp.exp((V + dtype(46.463)) / dtype(8.8289)))
        sum_m = jnp.maximum(am + bm, dtype(1e-12))
        alpha_m = am * sum_m / sum_m  # = am

        hinf = dtype(1.0) / (1.0 + jnp.exp((V + dtype(32.2)) / dtype(4.0)))
        tau_h = dtype(1.218) + dtype(42.043) * jnp.exp(
            -((V + dtype(38.1)) ** 2) / (dtype(2.0) * dtype(15.19) ** 2)
        )
        alpha_h = hinf / jnp.maximum(tau_h, dtype(1e-6))

        alpha_s, _, alpha_u, _ = self._su_rates(V)

        return jnp.stack([alpha_m, alpha_h, alpha_s, alpha_u], axis=-1) * self._qt

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        am = dtype(2.85) - dtype(2.839) / (1.0 + jnp.exp((V - dtype(1.159)) / dtype(13.95)))
        bm = dtype(7.6205) / (1.0 + jnp.exp((V + dtype(46.463)) / dtype(8.8289)))
        beta_m = bm

        hinf = dtype(1.0) / (1.0 + jnp.exp((V + dtype(32.2)) / dtype(4.0)))
        tau_h = dtype(1.218) + dtype(42.043) * jnp.exp(
            -((V + dtype(38.1)) ** 2) / (dtype(2.0) * dtype(15.19) ** 2)
        )
        beta_h = (dtype(1.0) - hinf) / jnp.maximum(tau_h, dtype(1e-6))

        _, beta_s, _, beta_u = self._su_rates(V)

        return jnp.stack([beta_m, beta_h, beta_s, beta_u], axis=-1) * self._qt

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        m, h, s, u = gates[:, 0], gates[:, 1], gates[:, 2], gates[:, 3]
        return (g_bar[0] * m ** 3 * h * s * u)[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = Gating.rates(jnp.atleast_1d(V0_mV), self.q10, self.alpha_funcs, self.beta_funcs)
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self.gbar], dtype=dtype) * 1e3

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self.ena], dtype=dtype)
