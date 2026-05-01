from __future__ import annotations
import jax.numpy as jnp
from axonscope.settings import dtype
from axonscope.channel_models.passive import PassiveICM


class NaKPumpICM(PassiveICM):
    """Electrogenic Na/K ATPase pump (Chapman 1983 / Nakpump.mod).

    Pumps 3 Na+ out and 2 K+ in per cycle → net outward current.
    With fixed ion concentrations ([Na]_i, [K]_o) the current is
    voltage-independent and treated as a constant background term.

    Net pump current (µA/cm²):
        ikpump   = smalla / (1 + b1/ko)^2  *  f(nai)    [mA/cm²]
        inapump  = -1.5 * ikpump
        I_pump   = inapump + ikpump = -0.5 * ikpump      [mA/cm²]
        I_bg     = I_pump * 1e3                           [µA/cm²]

    Positive I_bg = outward current = hyperpolarizing (consistent with
    AxonScope's I_ion sign convention).

    The conductance contribution of this class is zero (Rm=1e12 Ω·cm²).
    """

    def __init__(
        self,
        smalla: float = -0.0047891,
        b1: float = 1.0,
        ko: float = 5.6,
        nai: float = 11.4,
    ) -> None:
        super().__init__(Rm=1e12, EL=0.0)  # zero conductance

        f_nai = (
            1.62 / (1.0 + (6.7 / (nai + 8.0)) ** 3)
            + 1.0 / (1.0 + (67.6 / (nai + 8.0)) ** 3)
        )
        ikpump_mA = smalla / (1.0 + b1 / ko) ** 2 * f_nai   # mA/cm², negative
        ipump_mA  = -0.5 * ikpump_mA                          # net outward (positive)
        self._I_bg_uA = dtype(ipump_mA * 1e3)                # µA/cm²

    def I_background(self, Nx: int) -> jnp.ndarray:
        return jnp.full(Nx, self._I_bg_uA, dtype=dtype)
