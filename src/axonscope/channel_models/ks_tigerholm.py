from __future__ import annotations
import jax.numpy as jnp
from axonscope.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.icm import Gating


class KSlowICM(IonChannelModelBase):
    """K slow channel Kv7.3 (Passmore 2003 / Maingret 2008 / ks.mod).

    Gates: ns (slow component), nf (fast component).
    Current: I = gbar * (0.25*ns + 0.75*nf) * (v - E_K)

    Temperature correction: Q10 = 3.3, ref 21°C, baked into alpha/beta.
    """

    def __init__(
        self,
        gbar: float = 0.0069733,
        ek: float = -87.0,
        celsius: float = 37.0,
    ) -> None:
        super().__init__()
        self.gbar = dtype(gbar)
        self.ek = dtype(ek)
        self.celsius = dtype(celsius)
        self._qt = dtype(3.3 ** ((celsius - 21.0) / 10.0))
        self.q10 = dtype(1.0)

    def _inf_tau(self, V: jnp.ndarray):
        ninf = dtype(1.0) / (1.0 + jnp.exp(-(V + dtype(30.0)) / dtype(6.0)))

        # tau_ns: piecewise — becomes negative below -77 mV without the clamp
        tau_ns_lin = dtype(13.0) * V + dtype(1000.0)
        tau_ns = jnp.where(V >= dtype(-60.0), tau_ns_lin, dtype(219.0))

        a = dtype(0.00395) * jnp.exp((V + dtype(30.0)) / dtype(40.0))
        b = dtype(0.00395) * jnp.exp(-(V + dtype(30.0)) / dtype(20.0))
        tau_nf = dtype(1.0) / jnp.maximum(a + b, dtype(1e-12))

        return ninf, tau_ns, tau_nf

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        ninf, tau_ns, tau_nf = self._inf_tau(V)
        alpha_ns = ninf / jnp.maximum(tau_ns, dtype(1e-6))
        alpha_nf = ninf / jnp.maximum(tau_nf, dtype(1e-6))
        return jnp.stack([alpha_ns, alpha_nf], axis=-1) * self._qt

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        ninf, tau_ns, tau_nf = self._inf_tau(V)
        beta_ns = (dtype(1.0) - ninf) / jnp.maximum(tau_ns, dtype(1e-6))
        beta_nf = (dtype(1.0) - ninf) / jnp.maximum(tau_nf, dtype(1e-6))
        return jnp.stack([beta_ns, beta_nf], axis=-1) * self._qt

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        ns, nf = gates[:, 0], gates[:, 1]
        return (g_bar[0] * (dtype(0.25) * ns + dtype(0.75) * nf))[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = Gating.rates(jnp.atleast_1d(V0_mV), self.q10, self.alpha_funcs, self.beta_funcs)
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self.gbar], dtype=dtype) * 1e3

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self.ek], dtype=dtype)
