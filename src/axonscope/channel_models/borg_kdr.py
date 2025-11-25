from __future__ import annotations
import jax.numpy as jnp
from typing import Tuple, Callable

# Assuming these imports are necessary for the environment
from axonscope.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.icm_compute import Gating

Array1D = jnp.ndarray
Array2D = jnp.ndarray


class BorgKDRICM(IonChannelModelBase):
    """
    Borg-Graham type K-DR (delayed rectifier potassium) channel (1987).

    Updated implementation to match the kinetic equations derived from the 
    NEURON .mod file, ensuring equivalence in steady-state (g_inf) and 
    time constants (tau).
    """

    def __init__(self,
                 gkdrbar: float = 0.003,
                 ek: float = -77.0,
                 celsius: float = 36.0,
                 vhalfn: float = -32.0,
                 vhalfl: float = -61.0,
                 a0n: float = 0.03,
                 a0l: float = 0.001,
                 zetan: float = -5.0,
                 zetal: float = 2.0,
                 gmn: float = 0.4,
                 gml: float = 1.0,
                 ) -> None:

        super().__init__()

        # main parameters
        self.gkdrbar = dtype(gkdrbar)
        self.ek = dtype(ek)
        self.celsius = dtype(celsius)
        self.q10 = dtype(3 ** ((celsius - 30.0) / 10.0)) 
        # Q10 factor calculation as in NEURON: 3^((celsius-30)/10)
        self._q10 = dtype(3 ** ((celsius - 30.0) / 10.0)) 
        self._q10 = 1

        # gating parameters
        self.vhalfn = dtype(vhalfn)
        self.vhalfl = dtype(vhalfl)
        self.a0n = dtype(a0n)
        self.a0l = dtype(a0l)
        self.zetan = dtype(zetan)
        self.zetal = dtype(zetal)
        self.gmn = dtype(gmn)
        self.gml = dtype(gml)

    # ------------------------------------------------------------------
    # INTERMEDIATE RATE TERMS (Equivalent to alpn, betn, alpl, betl in .mod)
    # These functions calculate the voltage-dependent exponent terms only.

    def _rate_term_a(self, V: Array1D) -> Array2D:
        """
        Intermediate 'alpha' term calculation (alpn, alpl in .mod).
        F, R, T are Faraday constant, Gas constant, and Absolute Temperature.
        """
        F = 9.648e4
        R = 8.315
        T = 273.16 + self.celsius

        # alpn
        term_an = jnp.exp(1e-3 * self.zetan * (V - self.vhalfn) * F / (R * T))
        # alpl
        term_al = jnp.exp(1e-3 * self.zetal * (V - self.vhalfl) * F / (R * T))

        return jnp.stack([term_an, term_al], axis=-1)

    def _rate_term_b(self, V: Array1D) -> Array2D:
        """
        Intermediate 'beta' term calculation (betn, betl in .mod).
        """
        F = 9.648e4
        R = 8.315
        T = 273.16 + self.celsius

        # betn
        term_bn = jnp.exp(1e-3 * self.zetan * self.gmn * (V - self.vhalfn) * F / (R * T))
        # betl
        term_bl = jnp.exp(1e-3 * self.zetal * self.gml * (V - self.vhalfl) * F / (R * T))

        return jnp.stack([term_bn, term_bl], axis=-1)

    # ------------------------------------------------------------------
    # TRUE HODGKIN-HUXLEY ALPHA/BETA RATES (alpha_funcs and beta_funcs)
    # These combine the intermediate terms with Q10 and a0 to calculate 
    # the true alpha and beta rates (where g_inf = alpha/(alpha+beta)).

    def alpha_funcs(self, V: Array1D) -> Array2D:
        """
        Compute true alpha rate constants for [n, l] gates.
        Equivalent to: alpha = Q10 * a0 * A / B
        """
        A = self._rate_term_a(V) # A = alpn, alpl
        B = self._rate_term_b(V) # B = betn, betl

        # alpha_n = Q10 * a0n * alpn / betn
        alpha_n = self._q10 * self.a0n * A[:, 0] / B[:, 0]
        # alpha_l = Q10 * a0l * alpl / betl
        alpha_l = self._q10 * self.a0l * A[:, 1] / B[:, 1]

        return jnp.stack([alpha_n, alpha_l], axis=-1)

    def beta_funcs(self, V: Array1D) -> Array2D:
        """
        Compute true beta rate constants for [n, l] gates.
        Equivalent to: beta = Q10 * a0 / B
        """
        B = self._rate_term_b(V) # B = betn, betl

        # beta_n = Q10 * a0n / betn
        beta_n = self._q10 * self.a0n / B[:, 0]
        # beta_l = Q10 * a0l / betl
        beta_l = self._q10 * self.a0l / B[:, 1]

        return jnp.stack([beta_n, beta_l], axis=-1)

    # ------------------------------------------------------------------
    # Standard Ion Channel Model Methods

    def init_gates(self, V0_mV: Array1D) -> Array2D:
        """Initialize gating variables at steady state (g_inf)."""
        # Gating.rates uses the true alpha and beta functions above
        g_inf, _ = Gating.rates(V0_mV, self.q10, self.alpha_funcs, self.beta_funcs)
        return g_inf

    # ------------------------------------------------------------------

    def g_funcs(self, gates: Array2D, g_bar: Array1D) -> Array2D:
        """Compute channel conductance: g = g_bar * n^3 * l^1."""
        g = g_bar[0] * gates[:, 0]**3 * gates[:, 1]
        return g[:, None]

    @property
    def g_bar(self) -> Array1D:
        # Returns maximal conductance in mS/cm^2 (S/cm^2 * 10^3)
        return jnp.array([self.gkdrbar], dtype=dtype)*1e3

    @property
    def E_rev(self) -> Array1D:
        # Returns reversal potential (E_K)
        return jnp.array([self.ek], dtype=dtype)