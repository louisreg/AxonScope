from __future__ import annotations
from typing import Optional
import jax.numpy as jnp


from axonscope.channel_models.passive import PassiveICM
from axonscope.channel_models.base_channel_model import CompositeICM
from axonscope.channel_models.hodgkin_huxley import HodgkinHuxleyICM
from axonscope.channel_models.rattay_aberham import RattayAberhamICM
from axonscope.channel_models.borg_kdr import BorgKDRICM
from axonscope.channel_models.na_hh import NaHHICM
from axonscope.channel_models.composite_models import (
    TigerholmCompositeICM,
    Schild94CompositeICM,
    Schild97CompositeICM,
)

from .base import AxonBase


def _apply_nrv_like_unmyelinated_extracellular_defaults(axon: AxonBase) -> None:
    """Match NRV's unmyelinated `extracellular` setup.

    NRV inserts `extracellular` on unmyelinated sections and sets:
    - `xg[0] = 1e10` (radial short-circuit, no myelin)
    - `xc[0] = 0`
    while leaving `xraxial` at NEURON's default. We keep `xraxial=1e9`
    explicitly here to mirror the effective default behavior.
    """
    dtype = axon.ion_channel.dtype
    axon.set_extracellular_layer(
        xraxial_MOhm_per_cm=jnp.full((axon.Nx,), 1e9, dtype=dtype),
        xg_S_per_cm2=jnp.full((axon.Nx,), 1e10, dtype=dtype),
        xc_uF_per_cm2=jnp.zeros((axon.Nx,), dtype=dtype),
        use_extracellular=False,
        Veinit=0.0,
    )
    axon.prefer_inline_extracellular_solver = True


class HodgkinHuxley(AxonBase):
    """
    Hodgkin-Huxley squid axon model.

    Parameters
    ----------
    include_passive_leak : bool, optional
        If True, add a passive `pas` conductance in parallel with the classical
        HH channels. This reproduces NRV's `model="HH"` configuration, which
        inserts both `hh` and `pas`.
    g_pas : float, optional
        Passive conductance density in S/cm² when `include_passive_leak=True`.
    e_pas : float, optional
        Passive reversal potential in mV when `include_passive_leak=True`.
    """
    def __init__(
        self,
        
        d: float, # Default diameter changed to reflect typical HH use case (optional)
        Nx: Optional[int],
        L: Optional[float] = None,
        x_vec: Optional[jnp.ndarray] = None, 
        Ra: float = 200.0,
        Cm: float = 1.0,
        Vinit: float = -67.5,
        gnabar: float = 0.12,
        gkbar: float = 0.036,
        gl: float = 0.0003,
        el: float = -54.3,
        ena: float = 50.0,
        ek: float = -77.0,
        celsius: float = 6.3,
        include_passive_leak: bool = False,
        g_pas: float = 0.001,
        e_pas: float = -70.0,
    ):
        hh_model = HodgkinHuxleyICM(
            gnabar=gnabar,
            gkbar=gkbar,
            gl=gl,
            el=el,
            ena=ena,
            ek=ek,
            celsius=celsius,
        )
        if include_passive_leak:
            if g_pas <= 0.0:
                raise ValueError("g_pas must be strictly positive when include_passive_leak=True.")
            passive_model = PassiveICM(Rm=1.0 / g_pas, EL=e_pas)
            ion_channel = CompositeICM([hh_model, passive_model])
        else:
            ion_channel = hh_model
        super().__init__(
            ion_channel=ion_channel, L=L, d=d, Nx=Nx, x_vec=x_vec, 
            Ra=Ra, Cm=Cm, Vinit=Vinit, Temp=celsius
        )
        _apply_nrv_like_unmyelinated_extracellular_defaults(self)


class RattayAberham(AxonBase):
    """Rattay-Aberham axon model for mammalian unmyelinated fibers."""
    def __init__(
        self,
        
        d: float,
        Nx: Optional[int],
        L: Optional[float] = None,
        x_vec: Optional[jnp.ndarray] = None, 
        Cm: float = 1.0,
        Ra: float = 100.0,
        Vinit: float = -70.0,
        gnabar: float = 0.12,
        gkbar: float = 0.036,
        gl: float = 0.0003,
        el: float = -59.4,
        ena: float = 50.0,
        ek: float = -82.0,
        celsius: float = 37.0,
        include_passive_leak: bool = True,
        g_pas: float = 0.001,
        e_pas: float = -70.0,
    ):
        rattay_model = RattayAberhamICM(
            gnabar=gnabar,
            gkbar=gkbar,
            gl=gl,
            el=el,
            ena=ena,
            ek=ek,
            celsius=celsius,
        )
        if include_passive_leak:
            if g_pas <= 0.0:
                raise ValueError("g_pas must be strictly positive when include_passive_leak=True.")
            passive_model = PassiveICM(Rm=1.0 / g_pas, EL=e_pas)
            ion_channel = CompositeICM([rattay_model, passive_model])
        else:
            ion_channel = rattay_model
        super().__init__(
            ion_channel=ion_channel, L=L, d=d, Nx=Nx, x_vec=x_vec,
            Ra=Ra, Cm=Cm, Vinit=Vinit, Temp=celsius
        )
        _apply_nrv_like_unmyelinated_extracellular_defaults(self)

class Sundt(AxonBase):
    """
    Sundt axon model combining Rattay-Aberham Na⁺ channels
    with Borg-Graham-type K-DR channels.

    This composite axon allows simulating both sodium and
    potassium currents together in unmyelinated mammalian fibers.

    Parameters
    ----------
    L : Optional[float]
        Axon length [µm]. Required if x_vec is None.
    d : float
        Axon diameter [µm].
    Nx : Optional[int]
        Number of compartments. Required if L is not None and x_vec is None.
    x_vec : Optional[jnp.ndarray]
        Custom array of compartment positions [µm] for a non-uniform mesh.
    Cm : float
        Membrane capacitance [µF/cm²]
    Ra : float
        Axial resistance [Ω·cm]
    Vinit : float
        Initial membrane voltage [mV]
    celsius : float
        Temperature [°C]
    gnabar : float
        Maximum Na+ conductance [S/cm²]
    gkdrbar : float
        Maximum K-DR conductance [S/cm²]
    ena : float
        Na+ reversal potential [mV]
        ek : float
        K+ reversal potential [mV]
    Rm : float
        Leak resistance [Ω·cm²]
    El : float
        Leak reversal potential [mV]. NRV's Sundt initialization gives `-70 mV`.
    """

    def __init__(
        self,
        d: float,
        Nx: Optional[int],
        L: Optional[float] = None,
        x_vec: Optional[jnp.ndarray] = None, 
        Cm: float = 1.0,
        Ra: float = 100.0,
        Vinit: float = -60.0,
        celsius: float = 37.0,
        gnabar: float = 0.04,
        gkdrbar: float = 0.04,
        ena: float = 45.0,
        ek : float = -90.0,
        Rm : float = 10000.0,
        El : float = -70.0
    ):
        # Create individual membrane models
        # NRV inserts `nahh` with explicit m/h shifts for the Sundt model.
        na_model = NaHHICM(
            gnabar=gnabar,
            ena=ena,
            celsius=celsius,
            mshift=-6.0,
            hshift=6.0,
        )
        kdr_model = BorgKDRICM(gkdrbar=gkdrbar, ek=ek, celsius=celsius)
        passive_model = PassiveICM(Rm = Rm, EL = El)

        # Combine into a composite membrane model
        composite_model = CompositeICM([na_model, kdr_model, passive_model])

        # Initialize the axon base class with the composite model, passing x_vec
        super().__init__(
            ion_channel=composite_model, L=L, d=d, Nx=Nx, x_vec=x_vec,
            Ra=Ra, Cm=Cm, Vinit=Vinit, Temp=celsius
        )
        _apply_nrv_like_unmyelinated_extracellular_defaults(self)


class Tigerholm(AxonBase):
    """Tigerholm et al. 2014 mammalian C-fiber model.

    Channels: NaV1.7 (nattxs), NaV1.8 (DNav18), NaV1.9 (nav1p9),
              K-slow/Kv7.3 (ks), K-fast/A-type (kf), K-DR (kdrTiger),
              HCN (h), K-Na-dependent (kna), Na/K ATPase pump (nakpump).

    Ion dynamics (mirroring NRV's naoi.mod / koi.mod):
    - [Na]_i tracked per-compartment; drives Kna weight and pump correction.
    - [K]_o tracked per-compartment (periaxonal shell, θ = 0.029 µm); shifts E_K.
    - CompositeICM uses static E_K / nai; dynamics_correction applies the delta.
    - Temperature correction is per-channel (each channel has its own Q10/ref T).
    """

    def __init__(
        self,
        d: float,
        Nx: Optional[int],
        L: Optional[float] = None,
        x_vec=None,
        Cm: float = 1.0,
        Ra: float = 35.5,
        Vinit: float = -62.0,
        celsius: float = 37.0,
        # reversal potentials
        ena: float = 71.5,
        ek: float = -87.0,
        # per-channel conductances (S/cm²)
        gbar_nav17: float = 0.10664,
        gbar_nav18: float = 0.24271,
        gbar_nav19: float = 9.4779e-05,
        gbar_ks: float = 0.0069733,
        gbar_kf: float = 0.012756,
        gbar_kdr: float = 0.018002,
        gbar_h: float = 0.0025377,
        gbar_kna: float = 0.00042,
        # Kna parameters (Na-dependent K, approximated at fixed [Na]_i)
        nai_fixed: float = 11.4,
        # Na/K pump parameters (Nakpump.mod defaults)
        pump_smalla: float = -0.0047891,
        pump_ko: float = 5.6,
    ):
        ion_channel = TigerholmCompositeICM(
            diameter_um=d,
            celsius=celsius,
            ena=ena,
            ek=ek,
            gbar_nav17=gbar_nav17,
            gbar_nav18=gbar_nav18,
            gbar_nav19=gbar_nav19,
            gbar_ks=gbar_ks,
            gbar_kf=gbar_kf,
            gbar_kdr=gbar_kdr,
            gbar_h=gbar_h,
            gbar_kna=gbar_kna,
            nai_fixed=nai_fixed,
            pump_smalla=pump_smalla,
            pump_ko=pump_ko,
        )

        super().__init__(
            ion_channel=ion_channel, L=L, d=d, Nx=Nx, x_vec=x_vec,
            Ra=Ra, Cm=Cm, Vinit=Vinit, Temp=celsius,
        )
        _apply_nrv_like_unmyelinated_extracellular_defaults(self)


# =============================================================================
# Schild 1994 / 1997 DRG C-fiber models
# =============================================================================


class _SchildBase(AxonBase):
    """Axon geometry wrapper for Schild membrane models."""

    def __init__(self, ion_channel, d, Nx=101, L=None, x_vec=None, Ra=100.0, Cm=1.326291192, Vinit=-48.0, Temp=37.0):
        super().__init__(
            ion_channel=ion_channel, L=L, d=d, Nx=Nx, x_vec=x_vec,
            Ra=Ra, Cm=Cm, Vinit=Vinit, Temp=Temp,
        )
        _apply_nrv_like_unmyelinated_extracellular_defaults(self)


class Schild94(_SchildBase):
    """DRG C-fiber model, Schild et al. 1994."""

    def __init__(
        self,
        d: float,
        Nx: Optional[int],
        L: Optional[float] = None,
        x_vec=None,
        Ra: float = 100.0,
        Cm: float = 1.326291192,
        Vinit: float = -48.0,
        Temp: float = 37.0,
    ):
        super().__init__(
            ion_channel=Schild94CompositeICM(diameter_um=d, temp_c=Temp, vinit_mV=Vinit),
            L=L,
            d=d,
            Nx=Nx,
            x_vec=x_vec,
            Ra=Ra,
            Cm=Cm,
            Vinit=Vinit,
            Temp=Temp,
        )


class Schild97(_SchildBase):
    """DRG C-fiber model, Schild & Bhatt 1997."""

    def __init__(
        self,
        d: float,
        Nx: Optional[int],
        L: Optional[float] = None,
        x_vec=None,
        Ra: float = 100.0,
        Cm: float = 1.326291192,
        Vinit: float = -48.0,
        Temp: float = 37.0,
    ):
        super().__init__(
            ion_channel=Schild97CompositeICM(diameter_um=d, temp_c=Temp, vinit_mV=Vinit),
            L=L,
            d=d,
            Nx=Nx,
            x_vec=x_vec,
            Ra=Ra,
            Cm=Cm,
            Vinit=Vinit,
            Temp=Temp,
        )
