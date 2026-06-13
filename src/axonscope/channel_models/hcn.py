from __future__ import annotations
import jax.numpy as jnp
from axonscope.utils.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.icm import Gating


class HCNICM(IonChannelModelBase):
    """Hyperpolarization-activated cyclic-nucleotide-gated (HCN) channel
    (Kouranova 2008 / h.mod).

    Gates: ns (slow), nf (fast). Both share the same steady-state.
    Current: I = 0.5*g*(v-E_Na) + 0.5*g*(v-E_K)  →  g*(v - E_h)
    where E_h = (E_Na + E_K) / 2 and g = gbar*(0.5*ns + 0.5*nf).

    Temperature correction: Q10 = 3.0, ref 22°C, baked into alpha/beta.
    """

    def __init__(
        self,
        gbar: float = 0.0025377,
        ena: float = 71.5,
        ek: float = -87.0,
        celsius: float = 37.0,
    ) -> None:
        super().__init__()
        self.gbar = dtype(gbar)
        self.ena = dtype(ena)
        self.ek = dtype(ek)
        self._eh = dtype((ena + ek) / 2.0)
        self.celsius = dtype(celsius)
        self._qt = dtype(3.0 ** ((celsius - 22.0) / 10.0))
        self.q10 = dtype(1.0)

    def _inf_tau(self, V: jnp.ndarray):
        ninf = dtype(1.0) / (1.0 + jnp.exp((V + dtype(87.2)) / dtype(9.7)))

        # tau_ns: piecewise
        tau_ns_high = dtype(300.0) + dtype(542.0) * jnp.exp(
            (V + dtype(25.0)) / dtype(-20.0)
        )
        tau_ns_low = dtype(2500.0) + dtype(100.0) * jnp.exp(
            (V + dtype(240.0)) / dtype(50.0)
        )
        tau_ns = jnp.where(V >= dtype(-70.0), tau_ns_high, tau_ns_low)

        # tau_nf: piecewise
        tau_nf_high = dtype(140.0) + dtype(50.0) * jnp.exp(
            (V + dtype(25.0)) / dtype(-20.0)
        )
        tau_nf_low = dtype(250.0) + dtype(12.0) * jnp.exp(
            (V + dtype(240.0)) / dtype(50.0)
        )
        tau_nf = jnp.where(V >= dtype(-70.0), tau_nf_high, tau_nf_low)

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
        return (g_bar[0] * (dtype(0.5) * ns + dtype(0.5) * nf))[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = Gating.rates(jnp.atleast_1d(V0_mV), self.q10, self.alpha_funcs, self.beta_funcs)
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self.gbar], dtype=dtype) * 1e3

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self._eh], dtype=dtype)
