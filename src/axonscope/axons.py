from __future__ import annotations
from typing import Optional
import jax.numpy as jnp

from axonscope.benchmark import Benchmark

from axonscope.channel_models.passive import PassiveICM
from axonscope.channel_models.base_channel_model import IonChannelModelBase, CompositeICM
from axonscope.channel_models.hodgkin_huxley import HodgkinHuxleyICM
from axonscope.channel_models.rattay_aberham import RattayAberhamICM
from axonscope.channel_models.borg_kdr import BorgKDRICM
from axonscope.channel_models.na_hh import NaHHICM

bench = Benchmark()


class AxonBase:
    """
    Abstract base class for an axon with a given Ion Channel model.

    Provides:
    - Compartmental geometry
    - Ion Channel voltage state
    - Stimulus handling (point current injection)
    - Physical properties for cable equation

    Attributes
    ----------
    ion_channel : IonChannelModelBase
        Ion Channel model instance (e.g., HH, Passive, Rattay-Aberham, Composite)
    L : float
        Axon length [µm]
    d : float
        Axon diameter [µm]
    Nx : int
        Number of compartments
    Ra : float
        Axial resistance [Ω·cm]
    Vinit : float
        Initial membrane potential [mV]
    Temp : float
        Temperature [°C]
    V : jnp.ndarray
        Membrane voltage array, shape (Nx,)
    x : jnp.ndarray
        Compartment positions along the axon [µm], shape (Nx,)
    D : float
        Diffusion coefficient for cable equation [cm²/ms]
    cm : float
        Membrane capacitance per unit length [F/cm]
    ra : float
        Axial resistance per unit length [Ω/cm]
    stim : bool
        Flag indicating if a stimulus is active
    idx_inj : Optional[int]
        Compartment index of current injection
    t_start_inj : Optional[float]
        Start time of injection [ms]
    t_stop_inj : Optional[float]
        Stop time of injection [ms]
    inj_uA_per_cm2 : Optional[float]
        Current injection amplitude [µA/cm²]
    """

    def __init__(
        self,
        ion_channel: IonChannelModelBase,
        L: float,
        d: float,
        Nx: int = 101,
        Ra: float = 100.0,
        Cm: float = 1.0,
        Vinit: float = -70.0,
        Temp: float = 37.0,
    ):
        self.L: float = L
        self.d: float = d
        self.Nx: int = Nx
        self.dx: float = L / (Nx - 1)
        self.a: float = d / 2.0
        self.Ra: float = Ra
        self.Vinit: float = Vinit
        self.Temp: float = Temp

        self.ion_channel: IonChannelModelBase = ion_channel

        self.Ra = Ra
        self.Cm = Cm

        # Physical conversion to cm
        self.a_cm: float = self.a * 1e-4
        self.dx_cm: float = self.dx * 1e-4
        self.L_cm: float = self.L * 1e-4

        # Compartment positions
        self.x: jnp.ndarray = jnp.linspace(0.0, L, Nx)

        # Derived cable properties
        self.cm: float = 2.0 * jnp.pi * self.a_cm * Cm * 1e-6  # [F/cm]
        self.ra: float = Ra / (jnp.pi * self.a_cm**2)                           # [Ω/cm]
        self.D: float = 1.0 / (self.ra * self.cm) / 1000.0                     # [cm²/ms]

        # Membrane voltage state
        self.V: jnp.ndarray = jnp.full((Nx,), Vinit, dtype=ion_channel.dtype)

        # Stimulation attributes
        self.stim: bool = False
        self.idx_inj: Optional[int] = None
        self.t_start_inj: Optional[float] = None
        self.t_stop_inj: Optional[float] = None
        self.inj_uA_per_cm2: Optional[float] = None

    # --------------------------
    # Stimulus handling
    # --------------------------
    def insert_I_Clamp(
        self, position: float, t_start: float, duration: float, amplitude: float
    ) -> None:
        """
        Insert a point current injection (I-Clamp) at a given position along the axon.

        Parameters
        ----------
        position : float
            Injection position along the axon [µm]
        t_start : float
            Start time [ms]
        duration : float
            Duration of current injection [ms]
        amplitude : float
            Amplitude of injected current [µA/cm²]
        """
        self.idx_inj = int(jnp.argmin(jnp.abs(self.x - position)))
        self.inj_uA_per_cm2 = amplitude * 1e-3 / (2.0 * jnp.pi * self.a_cm * self.dx_cm)
        self.t_start_inj = t_start
        self.t_stop_inj = t_start + duration
        self.stim = True

    def Iinj_uAcm2(self, t: float) -> jnp.ndarray:
        """
        Compute the injected current density at time t.

        Parameters
        ----------
        t : float
            Time [ms]

        Returns
        -------
        jnp.ndarray, shape (Nx,)
            Current density array [µA/cm²], zero if outside stimulus window.
        """
        if self.stim and self.idx_inj is not None:
            active = (self.t_start_inj <= t) & (t <= self.t_stop_inj)
            return jnp.where(
                active,
                jnp.eye(self.Nx, dtype=self.ion_channel.dtype)[self.idx_inj] * self.inj_uA_per_cm2,
                jnp.zeros(self.Nx, dtype=self.ion_channel.dtype),
            )
        return jnp.zeros(self.Nx, dtype=self.ion_channel.dtype)


# --------------------------
# Generic Axon Classes
# --------------------------
class GenericAxon(AxonBase):
    """Generic axon for cable simulations with a given ion channel model."""
    def __init__(
        self,
        ion_channel: IonChannelModelBase,
        L: float,
        d: float,
        Nx: int = 101,
        Ra: float = 100.0,
        Cm: float = 1.0,
        Vinit: float = -70.0,
        Temp: float = 37.0,
    ):
        super().__init__(ion_channel=ion_channel, L=L, d=d, Nx=Nx, Ra=Ra, Cm=Cm, Vinit=Vinit, Temp=Temp)


class Passive(AxonBase):
    """Axon with passive membrane properties (leak conductance)."""
    def __init__(
        self,
        L: float,
        d: float,
        Nx: int = 101,
        Rm: float = 1e4,
        Cm: float = 1.0,
        Ra: float = 100.0,
        EL: float = -70.0,
        Vinit: float = -70.0,
        Temp: float = 37.0,
    ):
        ion_channel = PassiveICM(Rm=Rm, EL=EL)
        super().__init__(ion_channel=ion_channel, L=L, d=d, Nx=Nx, Ra=Ra, Cm=Cm, Vinit=Vinit, Temp=Temp)


class HodgkinHuxley(AxonBase):
    """Hodgkin-Huxley squid axon model."""
    def __init__(
        self,
        L: float,
        d: float,
        Nx: int = 101,
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
    ):
        ion_channel = HodgkinHuxleyICM(gnabar=gnabar, gkbar=gkbar, gl=gl, el=el, ena=ena, ek=ek, celsius=celsius)
        super().__init__(ion_channel=ion_channel, L=L, d=d, Nx=Nx, Ra=Ra, Cm=Cm, Vinit=Vinit, Temp=celsius)


class RattayAberham(AxonBase):
    """Rattay-Aberham axon model for mammalian myelinated fibers."""
    def __init__(
        self,
        L: float,
        d: float,
        Nx: int = 101,
        Cm: float = 1.0,
        Ra: float = 100.0,
        Vinit: float = -70.0,
        gnabar: float = 0.12,
        gkbar: float = 0.036,
        gl: float = 0.0003,
        el: float = -59.4,
        ena: float = 45.0,
        ek: float = -82.0,
        celsius: float = 37.0,
    ):
        ion_channel = RattayAberhamICM(gnabar=gnabar, gkbar=gkbar, gl=gl, el=el, ena=ena, ek=ek, celsius=celsius)
        super().__init__(ion_channel=ion_channel, L=L, d=d, Nx=Nx, Ra=Ra, Cm=Cm, Vinit=Vinit, Temp=celsius)

class Sundt(AxonBase):
    """
    Sundt axon model combining Rattay-Aberham Na⁺ channels
    with Borg-Graham-type K-DR channels.

    This composite axon allows simulating both sodium and
    potassium currents together in myelinated mammalian fibers.

    Parameters
    ----------
    L : float
        Axon length [µm]
    d : float
        Axon diameter [µm]
    Nx : int
        Number of compartments
    Cm : float
        Membrane capacitance [µF/cm²]
    Ra : float
        Axial resistance [Ω·cm]
    Vinit : float
        Initial membrane voltage [mV]
    celsius : float
        Temperature [°C]
    """

    def __init__(
        self,
        L: float,
        d: float,
        Nx: int = 101,
        Cm: float = 1.0,
        Ra: float = 100.0,
        Vinit: float = -60.0,
        celsius: float = 37.0,
        gnabar: float = 0.04,
        gkdrbar: float = 0.04,
        ena: float = 45.0,          #! Not sure about this one
        ek : float = -90.0,
        Rm : float = 10000.0,
        El : float = -60.0          #! Not goog but for testing
    ):
        # Create individual membrane models
        na_model = NaHHICM(gnabar=gnabar, ena=ena, celsius=celsius)
        kdr_model = BorgKDRICM(gkdrbar=gkdrbar, ek=ek, celsius=celsius)
        passive_model = PassiveICM(Rm = Rm, EL = El)

        # Combine into a composite membrane model
        #composite_model = CompositeICM([na_model, kdr_model, passive_model])
        composite_model = CompositeICM([na_model, kdr_model,passive_model])
        #composite_model = CompositeMembraneModel([na_model, na_model])

        # Initialize the axon base class with the composite model
        super().__init__(ion_channel=composite_model, L=L, d=d, Nx=Nx, Ra=Ra, Cm=Cm, Vinit=Vinit, Temp=celsius)
