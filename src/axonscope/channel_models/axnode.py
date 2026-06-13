from __future__ import annotations

import jax.numpy as jnp

from axonscope.utils.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase


def _safe_exp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(x < -100.0, 0.0, jnp.exp(x))


def _vtrap(a: float, b: float, c: float, x: jnp.ndarray, flipped: bool = False) -> jnp.ndarray:
    z = (x + b) / c
    num = a * (-(x + b) if flipped else (x + b))
    den = 1.0 - _safe_exp((x + b) / c if flipped else (-(x + b) / c))
    small = jnp.abs(z) < 1e-6
    lim = a * c
    return jnp.where(small, lim, num / den)


class AxnodeICM(IonChannelModelBase):
    """MRG nodal membrane from AXNODE.mod (mp, m, h, s)."""

    def __init__(
        self,
        gnapbar_S_cm2: float = 0.01,
        gnabar_S_cm2: float = 3.0,
        gkbar_S_cm2: float = 0.08,
        gl_S_cm2: float = 0.007,
        ena_mV: float = 50.0,
        ek_mV: float = -90.0,
        el_mV: float = -90.0,
        celsius: float = 37.0,
    ) -> None:
        super().__init__()
        self.gnapbar = dtype(gnapbar_S_cm2 * 1e3)  # mS/cm2
        self.gnabar = dtype(gnabar_S_cm2 * 1e3)
        self.gkbar = dtype(gkbar_S_cm2 * 1e3)
        self.gl = dtype(gl_S_cm2 * 1e3)
        self.ena = dtype(ena_mV)
        self.ek = dtype(ek_mV)
        self.el = dtype(el_mV)
        self.celsius = dtype(celsius)
        self.q10 = dtype(1.0)

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self.gnapbar, self.gnabar, self.gkbar, self.gl], dtype=dtype)

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self.ena, self.ena, self.ek, self.el], dtype=dtype)

    def _rates(self, V: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        q10_1 = 2.2 ** ((self.celsius - 20.0) / 10.0)
        q10_2 = 2.9 ** ((self.celsius - 20.0) / 10.0)
        q10_3 = 3.0 ** ((self.celsius - 36.0) / 10.0)

        a_mp = q10_1 * _vtrap(0.01, 27.0, 10.2, V, flipped=False)
        b_mp = q10_1 * _vtrap(0.00025, 34.0, 10.0, V, flipped=True)

        a_m = q10_1 * _vtrap(1.86, 21.4, 10.3, V, flipped=False)
        b_m = q10_1 * _vtrap(0.086, 25.7, 9.16, V, flipped=True)

        a_h = q10_2 * _vtrap(0.062, 114.0, 11.0, V, flipped=True)
        b_h = q10_2 * 2.3 / (1.0 + _safe_exp(-(V + 31.8) / 13.4))

        v2 = V + 80.0
        a_s = q10_3 * 0.3 / (_safe_exp((v2 - 27.0) / -5.0) + 1.0)
        b_s = q10_3 * 0.03 / (_safe_exp((v2 + 10.0) / -1.0) + 1.0)

        alpha = jnp.stack([a_mp, a_m, a_h, a_s], axis=1)
        beta = jnp.stack([b_mp, b_m, b_h, b_s], axis=1)
        return alpha, beta

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        alpha, _ = self._rates(V)
        return alpha

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        _, beta = self._rates(V)
        return beta

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        V = jnp.atleast_1d(V0_mV)
        a, b = self._rates(V)
        return a / jnp.maximum(a + b, 1e-12)

    def gate_names(self) -> tuple[str, ...]:
        return ("mp", "m", "h", "s")

    def conductance_names(self) -> tuple[str, ...]:
        return ("g_nap", "g_na", "g_k", "g_l")

    def current_names(self) -> tuple[str, ...]:
        return ("I_nap", "I_na", "I_k", "I_l")

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        mp, m, h, s = gates[:, 0], gates[:, 1], gates[:, 2], gates[:, 3]
        gnap = g_bar[0] * mp ** 3
        gna = g_bar[1] * m ** 3 * h
        gk = g_bar[2] * s
        gl = jnp.full_like(gnap, g_bar[3])
        return jnp.stack([gnap, gna, gk, gl], axis=1)
