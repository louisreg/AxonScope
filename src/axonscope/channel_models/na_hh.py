from __future__ import annotations
import jax.numpy as jnp

from axonscope.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.math_functions import expM1_jax as expM1
from axonscope.icm_compute import Gating


class NaHHICM(IonChannelModelBase):
    """
    Traub & Miles Na⁺ channel (1991), adapted from Cummins 2007, Sheets 2007.

    Hodgkin-Huxley type sodium channel with activation (m) and inactivation (h)
    gates, including voltage and temperature shifts.

    Parameters
    ----------
    gnabar : float
        Maximal sodium conductance [S/cm²]
    ena : float
        Sodium reversal potential [mV]
    celsius : float
        Temperature [°C]
    mshift : float
        Voltage shift for the m activation gate (mV)
    hshift : float
        Voltage shift for the h inactivation gate (mV)
    ishift : float
        Additional voltage shift for the h inactivation (mV)
    """

    def __init__(
        self,
        gnabar: float = 0.30,
        ena: float = 50.0,
        celsius: float = 36.0,
        mshift: float = -6.0,
        hshift: float = 6.0,
        ishift: float = 0.0,
    ):
        super().__init__()

        self.gnabar = dtype(gnabar)
        self.ena = dtype(ena)
        self.celsius = dtype(celsius)
        self.q10 = dtype(1)
        self._q10 = dtype(3 ** ((self.celsius - 30.0) / 10.0))

        # Instance parameters (were class constants)
        self.mshift = dtype(mshift)
        self.hshift = dtype(hshift)
        self.ishift = dtype(ishift)

    # ------------------------------------------------------------------

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        """
        Compute alpha rate constants for m and h gating variables.
        """
        V_m = V + 65.0

        m = self._q10*0.32 * expM1(13.1 - (V_m + self.mshift), 4.0)
        h = self._q10*0.128 * jnp.exp((17.0 - (V_m + self.hshift) + self.ishift) / 18.0)

        return jnp.stack([m, h], axis=-1)

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        """
        Compute beta rate constants for m and h gating variables.
        """
        V_m = V + 65.0

        m = self._q10*0.28 * expM1((V_m + self.mshift) - 40.1, 5.0)
        h = self._q10*4.0 / (jnp.exp((40.0 - (V_m + self.hshift)) / 5.0) + 1.0)

        return jnp.stack([m, h], axis=-1)

    # ------------------------------------------------------------------

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        """Initialize gating variables at steady-state."""
        inf, _ = Gating.rates(V0_mV, self.q10, self.alpha_funcs, self.beta_funcs)
        return inf

    # ------------------------------------------------------------------

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        """Compute sodium conductance."""
        g_na = g_bar[0] * gates[:, 0] ** 3 * gates[:, 1]
        return g_na[:, None]

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.array([self.gnabar], dtype=dtype)*1e3

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.array([self.ena], dtype=dtype)
