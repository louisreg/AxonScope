"""
Test utility models: individual Hodgkin-Huxley channel components.

These classes are split-out versions of HodgkinHuxleyICM exposing Na, K, and
Leak channels individually. They exist solely to test that CompositeICM correctly
assembles sub-models into a combined channel — they are not part of the public API.
"""
from __future__ import annotations
import jax.numpy as jnp
from axonscope.utils.math_functions import vtrap_jax as vtrap
from axonscope.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.icm import Gating


class HHNaICM(IonChannelModelBase):
    """Mono-Na channel following exact Hodgkin-Huxley kinetics. Gates: m, h."""

    def __init__(self, gnabar=0.12, ena=50.0, celsius=6.3):
        super().__init__()
        self.gnabar = dtype(gnabar)
        self.ena = dtype(ena)
        self.celsius = dtype(celsius)
        self.q10 = dtype(3.0 ** ((celsius - 6.3) / 10.0))

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        m = 0.1 * vtrap(-(V + 40.0), 10.0)
        h = 0.07 * jnp.exp(-(V + 65.0) / 20.0)
        return jnp.stack([m, h], axis=-1)

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        m = 4.0 * jnp.exp(-(V + 65.0) / 18.0)
        h = 1.0 / (jnp.exp(-(V + 35.0) / 10.0) + 1.0)
        return jnp.stack([m, h], axis=-1)

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        inf, _ = Gating.rates(V0_mV, self.q10, self.alpha_funcs, self.beta_funcs)
        return inf

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        m, h = gates[:, 0], gates[:, 1]
        gna = g_bar[0] * m**3 * h
        return gna[:, None]

    @property
    def g_bar(self):
        return jnp.array([self.gnabar], dtype=dtype) * 1e3

    @property
    def E_rev(self):
        return jnp.array([self.ena], dtype=dtype)


class HHKICM(IonChannelModelBase):
    """Mono-K channel following exact Hodgkin-Huxley kinetics. Gate: n."""

    def __init__(self, gkbar=0.036, ek=-77.0, celsius=6.3):
        super().__init__()
        self.gkbar = dtype(gkbar)
        self.ek = dtype(ek)
        self.celsius = dtype(celsius)
        self.q10 = dtype(3.0 ** ((celsius - 6.3) / 10.0))

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        n = 0.01 * vtrap(-(V + 55.0), 10.0)
        return n[:, None]

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        n = 0.125 * jnp.exp(-(V + 65.0) / 80.0)
        return n[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        inf, _ = Gating.rates(V0_mV, self.q10, self.alpha_funcs, self.beta_funcs)
        return inf

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        n = gates[:, 0]
        gk = g_bar[0] * n**4
        return gk[:, None]

    @property
    def g_bar(self):
        return jnp.array([self.gkbar], dtype=dtype) * 1e3

    @property
    def E_rev(self):
        return jnp.array([self.ek], dtype=dtype)


class HHLeakICM(IonChannelModelBase):
    """Leak-only HH channel. No gates."""

    def __init__(self, gl=0.0003, el=-54.3):
        super().__init__()
        self.gl = dtype(gl)
        self.el = dtype(el)
        self.q10 = dtype(1.0)

    def alpha_funcs(self, V):
        return jnp.zeros((V.shape[0], 0), dtype=dtype)

    def beta_funcs(self, V):
        return jnp.zeros((V.shape[0], 0), dtype=dtype)

    def init_gates(self, V0_mV):
        return jnp.zeros((V0_mV.shape[0], 0), dtype=dtype)

    def g_funcs(self, gates, g_bar):
        N = gates.shape[0]
        return jnp.full((N, 1), g_bar[0], dtype=dtype)

    @property
    def g_bar(self):
        return jnp.array([self.gl], dtype=dtype) * 1e3

    @property
    def E_rev(self):
        return jnp.array([self.el], dtype=dtype)
