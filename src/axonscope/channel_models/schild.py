"""
Ion channel models for Schild 1994 / 1997 DRG C-fiber.

All temperature corrections are baked into alpha/beta at construction time
(q10=1.0 so CompositeICM applies no additional scaling).

References
----------
Schild, J.H. et al. (1994). J Neurophysiol 71(6): 2338–2358.
Schild, J.H. & Bhatt, D.L. (1997). J Neurophysiol 78(6): 3198–3209.
Catherall, D. (2016). NMODL implementations, Grill Lab, Duke University.
"""

from __future__ import annotations
import jax.numpy as jnp
from axonscope.utils.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gauss_tau(V, A, B, C, Vp):
    """tau = A*exp(-B²*(V-Vp)²) + C  [ms, raw at reference temperature]"""
    return A * jnp.exp(-(B ** 2) * (V - dtype(Vp)) ** 2) + dtype(C)


def _boltz(V, V0p5, S0p5, shift=0.0):
    """xinf = 1/(1+exp((V - V0p5 + shift)/(-S0p5)))  — rising sigmoid."""
    return dtype(1.0) / (dtype(1.0) + jnp.exp((V - dtype(V0p5) + dtype(shift)) / dtype(-S0p5)))


def _boltz97(V, V0p5, S0p5):
    """xinf = 1/(1+exp((V0p5 - V)/S0p5))  — naf97/nas97 convention (rising)."""
    return dtype(1.0) / (dtype(1.0) + jnp.exp((dtype(V0p5) - V) / dtype(S0p5)))


def _alpha_beta_from_tau_inf(V, tau, xinf):
    """Convert tau/inf into effective alpha/beta for the gate ODE."""
    tau_safe = jnp.maximum(tau, dtype(1e-9))
    alpha = xinf / tau_safe
    beta = (dtype(1.0) - xinf) / tau_safe
    return alpha, beta


def _stack_rate_pairs(*rates):
    """Pack interleaved alpha/beta vectors into two rate matrices."""
    alpha = jnp.stack(rates[0::2], axis=-1)
    beta = jnp.stack(rates[1::2], axis=-1)
    return alpha, beta


# ---------------------------------------------------------------------------
# Schild 94 — Fast Na  (naf.mod)
# ---------------------------------------------------------------------------

class NafSchildICM(IonChannelModelBase):
    """Fast TTX-sensitive Na, Schild 1994.

    g = gbar * m³ * h * j (j renamed l in NMODL to avoid keyword conflict).
    Shift -17.5 mV for m and h; j has no shift and no Q10.
    Q10m=2.30, Q10h=1.50, Tref=22.85°C.
    """

    def __init__(self, gbar: float = 0.068967142, ena: float = 76.2, celsius: float = 37.0):
        super().__init__()
        self.q10 = dtype(1.0)
        self._gbar = dtype(gbar)
        self._ena = dtype(ena)
        # temperature factors
        self._q10m = dtype(2.30 ** ((celsius - 22.85) / 10.0))
        self._q10h = dtype(1.50 ** ((celsius - 22.85) / 10.0))

    def _rates(self, V: jnp.ndarray):
        shift = dtype(-17.5)
        tau_m = _gauss_tau(V, 0.75, 0.0635, 0.12, -40.35) / self._q10m
        minf = _boltz(V, -41.35, 4.75, shift)
        am, bm = _alpha_beta_from_tau_inf(V, tau_m, minf)

        tau_h = _gauss_tau(V, 6.5, 0.0295, 0.55, -75.0) / self._q10h
        hinf = _boltz(V, -62.0, -4.50, shift)  # S0p5h=-4.5 → decreasing sigmoid
        ah, bh = _alpha_beta_from_tau_inf(V, tau_h, hinf)

        # j gate: no shift, no Q10
        tau_j = dtype(25.0) / (dtype(1.0) + jnp.exp((V + dtype(-20.0)) / dtype(4.50))) + dtype(0.01)
        jinf = _boltz(V, -40.0, -1.50, 0.0)  # S0p5j=-1.5 → decreasing sigmoid
        aj, bj = _alpha_beta_from_tau_inf(V, tau_j, jinf)
        return am, bm, ah, bh, aj, bj

    def exact_rate_constants(self, V: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _stack_rate_pairs(*self._rates(V))

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        alpha, _ = self.exact_rate_constants(V)
        return alpha

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        _, beta = self.exact_rate_constants(V)
        return beta

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        m, h, j = gates[:, 0], gates[:, 1], gates[:, 2]
        return (g_bar[0] * m ** 3 * h * j)[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = self.gating_inf_tau(jnp.atleast_1d(V0_mV))
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self._gbar * 1e3], dtype=dtype)  # S/cm² → mS/cm²

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self._ena], dtype=dtype)


# ---------------------------------------------------------------------------
# Schild 94 — Slow Na  (nas.mod)
# ---------------------------------------------------------------------------

class NasSchildICM(IonChannelModelBase):
    """Slow TTX-insensitive Na, Schild 1994.

    g = gbar * m³ * h. Shift -20 mV. Q10m=2.30, Q10h=1.50, Tref=22.85°C.
    """

    def __init__(self, gbar: float = 0.001043349, ena: float = 76.2, celsius: float = 37.0):
        super().__init__()
        self.q10 = dtype(1.0)
        self._gbar = dtype(gbar)
        self._ena = dtype(ena)
        self._q10m = dtype(2.30 ** ((celsius - 22.85) / 10.0))
        self._q10h = dtype(1.50 ** ((celsius - 22.85) / 10.0))

    def _rates(self, V: jnp.ndarray):
        shift = dtype(-20.0)
        tau_m = _gauss_tau(V, 1.50, 0.0595, 0.15, -20.35) / self._q10m
        minf = _boltz(V, -20.35, 4.45, shift)
        am, bm = _alpha_beta_from_tau_inf(V, tau_m, minf)

        tau_h = _gauss_tau(V, 4.95, 0.0335, 0.75, -20.0) / self._q10h
        hinf = _boltz(V, -18.0, -4.50, shift)  # S0p5h=-4.5 → decreasing sigmoid
        ah, bh = _alpha_beta_from_tau_inf(V, tau_h, hinf)
        return am, bm, ah, bh

    def exact_rate_constants(self, V: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _stack_rate_pairs(*self._rates(V))

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        alpha, _ = self.exact_rate_constants(V)
        return alpha

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        _, beta = self.exact_rate_constants(V)
        return beta

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        m, h = gates[:, 0], gates[:, 1]
        return (g_bar[0] * m ** 3 * h)[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = self.gating_inf_tau(jnp.atleast_1d(V0_mV))
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self._gbar * 1e3], dtype=dtype)

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self._ena], dtype=dtype)


# ---------------------------------------------------------------------------
# Schild 97 — Fast Na  (naf97mean.mod)
# ---------------------------------------------------------------------------

class Naf97ICM(IonChannelModelBase):
    """Fast Na, Schild 1997 mean. g = gbar*m³*h. Q10m=2.30, Q10h=1.50, Tref=22°C."""

    def __init__(self, gbar: float = 0.022434928, ena: float = 76.2, celsius: float = 37.0):
        super().__init__()
        self.q10 = dtype(1.0)
        self._gbar = dtype(gbar)
        self._ena = dtype(ena)
        self._q10m = dtype(2.30 ** ((celsius - 22.0) / 10.0))
        self._q10h = dtype(1.50 ** ((celsius - 22.0) / 10.0))

    def _rates(self, V: jnp.ndarray):
        tau_m = _gauss_tau(V, 1.15, 0.06, 0.21, -40.0) / self._q10m
        minf = _boltz97(V, -31.62, 6.98)
        am, bm = _alpha_beta_from_tau_inf(V, tau_m, minf)

        tau_h = _gauss_tau(V, 18.0, 0.043, 1.35, -62.5) / self._q10h
        hinf = _boltz97(V, -65.99, -5.97)  # S0p5h < 0 → decreasing sigmoid
        ah, bh = _alpha_beta_from_tau_inf(V, tau_h, hinf)
        return am, bm, ah, bh

    def exact_rate_constants(self, V: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _stack_rate_pairs(*self._rates(V))

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        alpha, _ = self.exact_rate_constants(V)
        return alpha

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        _, beta = self.exact_rate_constants(V)
        return beta

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        m, h = gates[:, 0], gates[:, 1]
        return (g_bar[0] * m ** 3 * h)[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = self.gating_inf_tau(jnp.atleast_1d(V0_mV))
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self._gbar * 1e3], dtype=dtype)

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self._ena], dtype=dtype)


# ---------------------------------------------------------------------------
# Schild 97 — Slow Na  (nas97mean.mod)
# ---------------------------------------------------------------------------

class Nas97ICM(IonChannelModelBase):
    """Slow Na, Schild 1997 mean. g = gbar*m³*h. Q10m=2.30, Q10h=1.50, Tref=22°C."""

    def __init__(self, gbar: float = 0.022434928, ena: float = 76.2, celsius: float = 37.0):
        super().__init__()
        self.q10 = dtype(1.0)
        self._gbar = dtype(gbar)
        self._ena = dtype(ena)
        self._q10m = dtype(2.30 ** ((celsius - 22.0) / 10.0))
        self._q10h = dtype(1.50 ** ((celsius - 22.0) / 10.0))

    def _rates(self, V: jnp.ndarray):
        tau_m = _gauss_tau(V, 1.45, 0.058, 0.26, -14.5) / self._q10m
        minf = _boltz97(V, -11.29, 5.54)
        am, bm = _alpha_beta_from_tau_inf(V, tau_m, minf)

        tau_h = _gauss_tau(V, 10.75, 0.067, 3.15, -13.5) / self._q10h
        hinf = _boltz97(V, -31.0, -5.20)  # S0p5h < 0 → decreasing sigmoid
        ah, bh = _alpha_beta_from_tau_inf(V, tau_h, hinf)
        return am, bm, ah, bh

    def exact_rate_constants(self, V: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _stack_rate_pairs(*self._rates(V))

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        alpha, _ = self.exact_rate_constants(V)
        return alpha

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        _, beta = self.exact_rate_constants(V)
        return beta

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        m, h = gates[:, 0], gates[:, 1]
        return (g_bar[0] * m ** 3 * h)[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = self.gating_inf_tau(jnp.atleast_1d(V0_mV))
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self._gbar * 1e3], dtype=dtype)

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self._ena], dtype=dtype)


# ---------------------------------------------------------------------------
# Schild 94 — Delayed rectifier K  (kd.mod)
# ---------------------------------------------------------------------------

class KdSchildICM(IonChannelModelBase):
    """Delayed rectifier K, Schild 1994. g = gbar*n. Shift +3 mV. Q10=1.40, Tref=22.85°C.

    Alpha uses the HH activation formula; tau includes a +1 ms offset.
    """

    def __init__(self, gbar: float = 0.000180376, ek: float = -87.9, celsius: float = 37.0):
        super().__init__()
        self.q10 = dtype(1.0)
        self._gbar = dtype(gbar)
        self._ek = dtype(ek)
        self._q10 = dtype(1.40 ** ((celsius - 22.85) / 10.0))
        self._shift = dtype(3.0)

    def _rates(self, V: jnp.ndarray):
        # kd.mod calls rates(v) with unshifted v — shift only applies to ninf
        x = V + dtype(14.273)
        alphan = jnp.where(
            jnp.abs(x) < dtype(1e-6),
            dtype(0.01265),
            dtype(0.001265) * x / (dtype(1.0) - jnp.exp(x / dtype(-10.0)))
        )
        betan = dtype(0.125) * jnp.exp((V + dtype(55.0)) / dtype(-2.5))
        ninf = _boltz(V, -14.62, 18.38, self._shift)  # shift only in ninf
        tau_n_raw = dtype(1.0) / jnp.maximum(alphan + betan, dtype(1e-9)) + dtype(1.0)
        tau_n = tau_n_raw / self._q10
        return _alpha_beta_from_tau_inf(V, tau_n, ninf)

    def exact_rate_constants(self, V: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        a, b = self._rates(V)
        return a[:, None], b[:, None]

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        alpha, _ = self.exact_rate_constants(V)
        return alpha

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        _, beta = self.exact_rate_constants(V)
        return beta

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        n = gates[:, 0]
        return (g_bar[0] * n)[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = self.gating_inf_tau(jnp.atleast_1d(V0_mV))
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self._gbar * 1e3], dtype=dtype)

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self._ek], dtype=dtype)


# ---------------------------------------------------------------------------
# Schild 94 — Transient outward K  (ka.mod)
# ---------------------------------------------------------------------------

class KaSchildICM(IonChannelModelBase):
    """Early transient outward K, Schild 1994. g = gbar*p³*q. Q10=1.93, Tref=22.85°C."""

    def __init__(self, gbar: float = 0.000141471, ek: float = -87.9, celsius: float = 37.0):
        super().__init__()
        self.q10 = dtype(1.0)
        self._gbar = dtype(gbar)
        self._ek = dtype(ek)
        self._q10 = dtype(1.93 ** ((celsius - 22.85) / 10.0))
        self._shift = dtype(3.0)

    def _rates(self, V: jnp.ndarray):
        tau_p = _gauss_tau(V, 5.0, 0.022, 2.5, -65.0) / self._q10
        pinf = _boltz(V, -28.0, 28.0, self._shift)
        ap, bp = _alpha_beta_from_tau_inf(V, tau_p, pinf)

        tau_q = _gauss_tau(V, 100.0, 0.035, 10.5, -30.0) / self._q10
        qinf = _boltz(V, -58.0, -7.0, self._shift)  # S0p5<0 → decreasing
        aq, bq = _alpha_beta_from_tau_inf(V, tau_q, qinf)
        return ap, bp, aq, bq

    def exact_rate_constants(self, V: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _stack_rate_pairs(*self._rates(V))

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        alpha, _ = self.exact_rate_constants(V)
        return alpha

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        _, beta = self.exact_rate_constants(V)
        return beta

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        p, q = gates[:, 0], gates[:, 1]
        return (g_bar[0] * p ** 3 * q)[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = self.gating_inf_tau(jnp.atleast_1d(V0_mV))
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self._gbar * 1e3], dtype=dtype)

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self._ek], dtype=dtype)


# ---------------------------------------------------------------------------
# Schild 94 — Slowly inactivating delay K  (kds.mod)
# ---------------------------------------------------------------------------

class KdsSchildICM(IonChannelModelBase):
    """Slowly inactivating delay K, Schild 1994. g = gbar*x³*y. Q10=1.93, Tref=22.85°C."""

    def __init__(self, gbar: float = 0.000106103, ek: float = -87.9, celsius: float = 37.0):
        super().__init__()
        self.q10 = dtype(1.0)
        self._gbar = dtype(gbar)
        self._ek = dtype(ek)
        self._q10 = dtype(1.93 ** ((celsius - 22.85) / 10.0))
        self._shift = dtype(3.0)
        # y gate: constant tau = 7500 ms at 22.85°C
        self._tau_y = dtype(7500.0 / (1.93 ** ((celsius - 22.85) / 10.0)))

    def _rates(self, V: jnp.ndarray):
        tau_x = _gauss_tau(V, 5.0, 0.022, 2.5, -65.0) / self._q10
        xinf = _boltz(V, -39.59, 14.68, self._shift)
        ax, bx = _alpha_beta_from_tau_inf(V, tau_x, xinf)

        yinf = _boltz(V, -48.0, -7.0, self._shift)  # S0p5<0 → decreasing
        ay, by = _alpha_beta_from_tau_inf(V, self._tau_y, yinf)
        return ax, bx, ay, by

    def exact_rate_constants(self, V: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _stack_rate_pairs(*self._rates(V))

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        alpha, _ = self.exact_rate_constants(V)
        return alpha

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        _, beta = self.exact_rate_constants(V)
        return beta

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        x, y = gates[:, 0], gates[:, 1]
        return (g_bar[0] * x ** 3 * y)[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = self.gating_inf_tau(jnp.atleast_1d(V0_mV))
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self._gbar * 1e3], dtype=dtype)

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self._ek], dtype=dtype)


# ---------------------------------------------------------------------------
# Schild 94 — High-threshold Ca  (can.mod)
# ---------------------------------------------------------------------------

class CaNICM(IonChannelModelBase):
    """High-threshold long-lasting Ca, Schild 1994.

    g = gbar * d * (0.55*f1 + 0.45*f2). Shift -7 mV. Q10=4.30, Tref=22.85°C.
    E_Ca uses Schild Nernst convention with 78.7 mV offset.
    """

    def __init__(self, gbar: float = 0.000106103, eca: float = 51.6, celsius: float = 37.0):
        super().__init__()
        self.q10 = dtype(1.0)
        self._gbar = dtype(gbar)
        self._eca = dtype(eca)
        self._q10 = dtype(4.30 ** ((celsius - 22.85) / 10.0))
        self._shift = dtype(-7.0)

    def _rates(self, V: jnp.ndarray):
        shift = self._shift
        tau_d = _gauss_tau(V, 3.25, 0.042, 0.395, -31.0) / self._q10
        dinf = _boltz(V, -20.0, 4.5, shift)
        ad, bd = _alpha_beta_from_tau_inf(V, tau_d, dinf)

        tau_f1 = _gauss_tau(V, 33.5, 0.0395, 5.0, -30.0) / self._q10
        f1inf = _boltz(V, -20.0, -25.0, shift)  # S0p5<0 → decreasing
        af1, bf1 = _alpha_beta_from_tau_inf(V, tau_f1, f1inf)

        tau_f2 = _gauss_tau(V, 225.0, 0.0275, 75.0, -40.0) / self._q10
        rn = dtype(0.2) / (dtype(1.0) + jnp.exp((V + dtype(5.0) + shift) / dtype(-10.0)))
        f2inf = rn + _boltz(V, -40.0, -10.0, shift)  # S0p5<0 + rn offset
        f2inf = jnp.clip(f2inf, dtype(0.0), dtype(1.0))
        af2, bf2 = _alpha_beta_from_tau_inf(V, tau_f2, f2inf)
        return ad, bd, af1, bf1, af2, bf2

    def exact_rate_constants(self, V: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _stack_rate_pairs(*self._rates(V))

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        alpha, _ = self.exact_rate_constants(V)
        return alpha

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        _, beta = self.exact_rate_constants(V)
        return beta

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        d, f1, f2 = gates[:, 0], gates[:, 1], gates[:, 2]
        return (g_bar[0] * d * (dtype(0.55) * f1 + dtype(0.45) * f2))[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = self.gating_inf_tau(jnp.atleast_1d(V0_mV))
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self._gbar * 1e3], dtype=dtype)

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self._eca], dtype=dtype)


# ---------------------------------------------------------------------------
# Schild 94 — Low-threshold transient Ca  (cat.mod)
# ---------------------------------------------------------------------------

class CaTICM(IonChannelModelBase):
    """Low-threshold transient Ca, Schild 1994.

    g = gbar * d * f. Shift -7 mV. Q10d=1.90, Q10f=2.20, Tref=22.85°C.
    """

    def __init__(self, gbar: float = 1.23787e-5, eca: float = 51.6, celsius: float = 37.0):
        super().__init__()
        self.q10 = dtype(1.0)
        self._gbar = dtype(gbar)
        self._eca = dtype(eca)
        self._q10d = dtype(1.90 ** ((celsius - 22.85) / 10.0))
        self._q10f = dtype(2.20 ** ((celsius - 22.85) / 10.0))
        self._shift = dtype(-7.0)

    def _rates(self, V: jnp.ndarray):
        shift = self._shift
        tau_d = _gauss_tau(V, 22.0, 0.052, 2.5, -68.0) / self._q10d
        dinf = _boltz(V, -54.0, 5.75, shift)
        ad, bd = _alpha_beta_from_tau_inf(V, tau_d, dinf)

        tau_f = _gauss_tau(V, 103.0, 0.050, 12.5, -58.0) / self._q10f
        finf = _boltz(V, -68.0, -6.0, shift)  # S0p5<0 → decreasing
        af, bf = _alpha_beta_from_tau_inf(V, tau_f, finf)
        return ad, bd, af, bf

    def exact_rate_constants(self, V: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _stack_rate_pairs(*self._rates(V))

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        alpha, _ = self.exact_rate_constants(V)
        return alpha

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        _, beta = self.exact_rate_constants(V)
        return beta

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        d, f = gates[:, 0], gates[:, 1]
        return (g_bar[0] * d * f)[:, None]

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        g_inf, _ = self.gating_inf_tau(jnp.atleast_1d(V0_mV))
        return g_inf

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self._gbar * 1e3], dtype=dtype)

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self._eca], dtype=dtype)


# ---------------------------------------------------------------------------
# Schild — Passive leak  (leakSchild.mod)
# ---------------------------------------------------------------------------

class LeakSchildICM(IonChannelModelBase):
    """Na + Ca passive leak, Schild 1994/1997.

    Two zero-gate channels: I = gbna*(V-ena) + gbca*(V-eca).
    """

    def __init__(
        self,
        gbna: float = 1.85681e-5,
        gbca: float = 3.00626e-6,
        ena: float = 76.2,
        eca: float = 51.6,
    ):
        super().__init__()
        self.q10 = dtype(1.0)
        self._gbna = dtype(gbna * 1e3)   # S/cm² → mS/cm²
        self._gbca = dtype(gbca * 1e3)
        self._ena = dtype(ena)
        self._eca = dtype(eca)

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((V.shape[0], 0), dtype=dtype)

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((V.shape[0], 0), dtype=dtype)

    def exact_rate_constants(self, V: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        empty = jnp.zeros((V.shape[0], 0), dtype=dtype)
        return empty, empty

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        N = gates.shape[0]
        gna = jnp.full((N, 1), g_bar[0], dtype=dtype)
        gca = jnp.full((N, 1), g_bar[1], dtype=dtype)
        return jnp.concatenate([gna, gca], axis=-1)

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((jnp.atleast_1d(V0_mV).shape[0], 0), dtype=dtype)

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self._gbna, self._gbca], dtype=dtype)

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self._ena, self._eca], dtype=dtype)
