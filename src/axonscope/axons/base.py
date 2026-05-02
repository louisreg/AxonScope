from __future__ import annotations
from typing import Optional, Union
import warnings
import jax.numpy as jnp

from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.electrodes import Electrode
from axonscope.icm import ICMBackend, UniformICMBackend
from axonscope.stimulation import ExtracellularContext, IntracellularCurrentClamp
from axonscope.stimulus import Stimulus

# AxonBase remains unchanged, as it correctly handles L, Nx, and x_vec.
class AxonBase:
    """
    Abstract base class for an axon with a given Ion Channel model.

    Provides:
    - Compartmental geometry (uniform or non-uniform mesh).
    - Stimulus descriptions.
    - Physical properties for cable equation.

    Attributes
    ----------
    ion_channel : IonChannelModelBase
        Ion Channel model instance (e.g., HH, Passive, Rattay-Aberham, Composite).
    L : float
        Total axon length [µm].
    d : float
        Axon diameter [µm] (uniform diameter assumed in this base class).
    Nx : int
        Number of compartments.
    Ra : float
        Axial resistance [Ω·cm].
    Cm : float
        Membrane capacitance [µF/cm²] (per unit area).
    Vinit : float
        Initial membrane potential [mV].
    Temp : float
        Temperature [°C].
    x : jnp.ndarray
        Compartment positions along the axon [µm], shape (Nx,). **Defines the mesh.**
    h_um : jnp.ndarray
        Edge lengths vector: h_um[i] = x[i+1] - x[i] [µm], shape (Nx-1,).
    h_cm : jnp.ndarray
        Edge lengths in cm, shape (Nx-1,). Used by non-uniform diffusion operators.
    dx_cm : jnp.ndarray
        Control-volume length around each node [cm], shape (Nx,).
    a_cm : float
        Axon radius [cm].
    L_cm : float
        Total axon length [cm].
    
    # Uniform-mesh derived convenience properties
    ra : float
        Axial resistance per unit length [Ω/cm].
    cm : float
        Membrane capacitance per unit length [F/cm].
    D : float
        Diffusion coefficient for cable equation [cm²/ms] (based on uniform mesh).

    """

    def __init__(
        self,
        ion_channel: IonChannelModelBase,
        d: float,
        Nx: Optional[int],
        L: Optional[float] = None,
        x_vec: Optional[jnp.ndarray] = None, # Passed through to handle non-uniform mesh
        Ra: float = 100.0,
        Cm: float = 1.0,
        Vinit: float = -70.0,
        Temp: float = 37.0,
    ):
        
        # --- 1. Geometric & Physical Constants ---
        self.d: float = d
        self.Ra: float = Ra
        self.Cm: float = Cm
        self.Vinit: float = Vinit
        self.Temp: float = Temp
        self.ion_channel: IonChannelModelBase = ion_channel
        self.has_heterogeneous_cable_properties: bool = False

        # --- 2. Mesh Initialization ---
        if x_vec is not None:
            # Case A: User-defined non-uniform mesh (x_vec provided)
            self.x: jnp.ndarray = jnp.array(x_vec, dtype=ion_channel.dtype)
            self.Nx: int = self.x.shape[0]
            if self.Nx < 2:
                raise ValueError(f"Axon mesh requires at least 2 compartments, got {self.Nx}.")
            if L is None:
                self.L: float = self.x[-1].item()
            else:
                self.L: float = L
                if not jnp.isclose(self.x[-1], L):
                    warnings.warn(
                        f"Last position in x_vec ({self.x[-1].item():.2f}µm) does not match "
                        f"expected length L ({L:.2f}µm). Using x_vec length.",
                        UserWarning, stacklevel=2
                    )
                    self.L = self.x[-1].item()
            
        elif L is not None and Nx is not None:
            # Case B: Uniform mesh (L and Nx provided)
            self.L: float = L
            self.Nx: int = Nx
            if self.Nx < 2:
                raise ValueError(f"Axon mesh requires at least 2 compartments, got {self.Nx}.")
            self.x: jnp.ndarray = jnp.linspace(0.0, L, self.Nx, dtype=ion_channel.dtype)
        else:
            raise ValueError("AxonScope Error: Must provide either (L and Nx) for a uniform mesh, or 'x_vec' for a custom mesh.")

        
        self.a: float = self.d / 2.0

        # Physical conversion to cm (uniform properties)
        self.a_cm: float = self.a * 1e-4
        self.L_cm: float = self.L * 1e-4

        # --- 2. Derived Geometric Properties (for Non-Uniform Mesh Calculation) ---
        # Segment edge lengths and control-volume lengths around each node.
        dx: jnp.ndarray = jnp.diff(self.x) # dx[i] = x[i+1] - x[i] [µm]
        dx_avg = jnp.zeros_like(self.x) 
        # Compartment length for internal segments (average of left and right segments)
        dx_avg = dx_avg.at[1:-1].set(0.5 * (dx[:-1] + dx[1:])) 
        # Compartment length for boundaries (full first/last segment length)
        dx_avg = dx_avg.at[0].set(dx[0]) 
        dx_avg = dx_avg.at[-1].set(dx[-1]) 
        self.h_um: jnp.ndarray = dx
        self.h_cm: jnp.ndarray = dx * 1e-4
        self.dx_cm: jnp.ndarray = dx_avg * 1e-4 # [cm]

        # Derived cable properties (based on uniform Cm and Ra for simplicity)
        self.cm: float = 2.0 * jnp.pi * self.a_cm * Cm * 1e-6  # [F/cm]
        self.ra: float = Ra / (jnp.pi * self.a_cm**2)           # [Ω/cm]
        self.D: float = 1.0 / (self.ra * self.cm) / 1000.0     # [cm²/ms] # Uniform D

        # --- 4. Stimulation descriptions ---
        self.intracellular_clamps: list[IntracellularCurrentClamp] = []

        # --- 5. Per-compartment arrays (default: uniform axon) ---
        self.diam_vec: jnp.ndarray = jnp.full((self.Nx,), self.d, dtype=ion_channel.dtype)
        self.Ra_vec: jnp.ndarray = jnp.full((self.Nx,), self.Ra, dtype=ion_channel.dtype)
        self.Cm_vec: jnp.ndarray = jnp.full((self.Nx,), self.Cm, dtype=ion_channel.dtype)
        self.compartment_lengths_um: jnp.ndarray = dx_avg

        # --- 6. Extracellular one-layer mechanism defaults ---
        # Convention (matching NEURON extracellular one-layer units):
        # - xraxial: MOhm/cm
        # - xg: S/cm^2
        # - xc: uF/cm^2
        self.use_extracellular: bool = False
        self.Veinit: float = 0.0
        self.xraxial_vec: jnp.ndarray = jnp.full((self.Nx,), 1e9, dtype=ion_channel.dtype)
        self.xg_vec: jnp.ndarray = jnp.full((self.Nx,), 1e-6, dtype=ion_channel.dtype)
        self.xc_vec: jnp.ndarray = jnp.zeros((self.Nx,), dtype=ion_channel.dtype)
        self.extracellular_contexts: list[ExtracellularContext] = []

    def build_icm_backend(self) -> ICMBackend:
        """Return the compute backend associated with this axon description."""
        return UniformICMBackend.from_model(self.ion_channel, self.Nx)

    # --------------------------
    # Stimulus handling
    # --------------------------
    def insert_I_Clamp(
        self,
        position: float,
        stimulus: Optional[Stimulus] = None,
        *,
        t_start: Optional[float] = None,
        duration: Optional[float] = None,
        amplitude: Optional[float] = None,
    ) -> None:
        """
        Attach a descriptive point current injection at a given axon position.

        Parameters
        ----------
        position : float
            Injection position along the axon [µm]
        stimulus : Stimulus, optional
            Temporal waveform in nA.
        t_start, duration, amplitude :
            Convenience pulse constructor arguments, used only when `stimulus`
            is not provided.
        """
        if stimulus is None:
            if t_start is None or duration is None or amplitude is None:
                raise ValueError(
                    "Provide either a `stimulus`, or all of `t_start`, `duration`, and `amplitude`."
                )
            stimulus = Stimulus.pulse(start=t_start, duration=duration, amplitude=amplitude)
        elif t_start is not None or duration is not None or amplitude is not None:
            raise ValueError(
                "Do not mix `stimulus=` with pulse constructor arguments."
            )

        self.intracellular_clamps.append(
            IntracellularCurrentClamp(position_um=float(position), stimulus=stimulus)
        )

    def Iinj_uAcm2(self, t: float) -> jnp.ndarray:
        """Evaluate injected current density at time t."""
        from axonscope.solvers.stimulus_runtime import build_intracellular_current_density_fn

        return build_intracellular_current_density_fn(self)(t)

    def set_extracellular_layer(
        self,
        *,
        xraxial_MOhm_per_cm: Optional[jnp.ndarray] = None,
        xg_S_per_cm2: Optional[jnp.ndarray] = None,
        xc_uF_per_cm2: Optional[jnp.ndarray] = None,
        use_extracellular: Optional[bool] = None,
        Veinit: Optional[float] = None,
    ) -> None:
        """Set one-layer extracellular parameters on the current mesh."""
        if xraxial_MOhm_per_cm is not None:
            arr = jnp.asarray(xraxial_MOhm_per_cm, dtype=self.ion_channel.dtype)
            if arr.shape != (self.Nx,):
                raise ValueError(f"xraxial must have shape ({self.Nx},), got {arr.shape}")
            self.xraxial_vec = arr
        if xg_S_per_cm2 is not None:
            arr = jnp.asarray(xg_S_per_cm2, dtype=self.ion_channel.dtype)
            if arr.shape != (self.Nx,):
                raise ValueError(f"xg must have shape ({self.Nx},), got {arr.shape}")
            self.xg_vec = arr
        if xc_uF_per_cm2 is not None:
            arr = jnp.asarray(xc_uF_per_cm2, dtype=self.ion_channel.dtype)
            if arr.shape != (self.Nx,):
                raise ValueError(f"xc must have shape ({self.Nx},), got {arr.shape}")
            self.xc_vec = arr
        if use_extracellular is not None:
            self.use_extracellular = bool(use_extracellular)
        if Veinit is not None:
            self.Veinit = float(Veinit)

    def attach_extracellular_stimulus(self, stim: ExtracellularContext) -> None:
        """Attach one electrode/stimulus pair, replacing any previously attached extracellular context."""
        self.add_extracellular_ctx(stim, replace=True, enable=True)

    def add_extracellular_ctx(
        self,
        electrode_or_ctx: Union[Electrode, ExtracellularContext],
        stimulus: Optional[Stimulus] = None,
        *,
        replace: bool = False,
        enable: bool = True,
    ) -> None:
        """Add an extracellular context to this axon.

        Parameters
        ----------
        electrode_or_ctx :
            Either:
            - an `ExtracellularContext` already built via `electrode.attach_stimulus(stimulus)`, or
            - an `Electrode` object.
        stimulus :
            Required when `electrode_or_ctx` is an `Electrode`.
        replace :
            If True, clear previously attached extracellular contexts first.
        enable :
            If True, set `use_extracellular=True`.
        """
        if isinstance(electrode_or_ctx, ExtracellularContext):
            if stimulus is not None:
                raise ValueError("Do not provide `stimulus` when passing an `ExtracellularContext`.")
            ctx = electrode_or_ctx
        else:
            if stimulus is None:
                raise ValueError("`stimulus` is required when passing an `Electrode`.")
            ctx = electrode_or_ctx.attach_stimulus(stimulus)

        if replace:
            self.extracellular_contexts = [ctx]
        else:
            self.extracellular_contexts.append(ctx)
        if enable:
            self.use_extracellular = True

    def clear_extracellular_ctx(self) -> None:
        """Remove all attached extracellular contexts from this axon."""
        self.extracellular_contexts = []

    def Vext_mV(self, t_ms: float) -> jnp.ndarray:
        """Compatibility wrapper returning imposed extracellular potential in mV."""
        from axonscope.solvers.stimulus_runtime import build_extracellular_potential_fn

        return build_extracellular_potential_fn(self)(t_ms)
