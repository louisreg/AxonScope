from __future__ import annotations
import jax.numpy as jnp
from axonscope.utils.settings import dtype
from axonscope.channel_models.base_channel_model import IonChannelModelBase

class PassiveICM(IonChannelModelBase):
    """
    Passive (leak) ion channel model.

    This class implements a simple passive ion channel model with a constant leak 
    conductance. It is JAX-friendly: all attributes and outputs are 
    compatible with JAX arrays and dtypes, enabling vectorization and JIT 
    compilation.

    Parameters
    ----------
    Rm : float, optional
        Specific membrane resistance in MΩ·cm² (default 1e4). Higher Rm 
        corresponds to a smaller leak conductance.
    EL : float, optional
        Leak reversal potential in mV (default -70.0). This sets the resting 
        potential for the membrane in the absence of other currents.

    Attributes
    ----------
    Rm : float
        Membrane resistance stored as JAX `dtype`.
    EL : float
        Leak reversal potential stored as JAX `dtype`.
    g_leak : float
        Maximal leak conductance in mS/cm², computed as 1/Rm * 1e3.
        Constant over time and voltage.
    """

    def __init__(self, Rm: float = 1e4, EL: float = -70.0) -> None:
        """
        Initialize passive ion channel parameters.

        Notes
        -----
        - All attributes are converted to JAX `dtype` for compatibility 
          with vectorized computations.
        - Leak conductance g_leak is computed as g_leak = 1/Rm * 1e3 [mS/cm²].
        """
        super().__init__()
        self.Rm: float = dtype(Rm)
        self.EL: float = dtype(EL)
        self.g_leak: float = dtype(1.0 / Rm) * 1e3  # mS/cm²

    @property
    def E_rev(self) -> jnp.ndarray:
        """
        Return reversal potential array for leak channel.

        Returns
        -------
        E_rev : jnp.ndarray, shape (1,)
            Leak reversal potential (mV) for each channel.
        """
        return jnp.array([self.EL], dtype=dtype)

    @property
    def g_bar(self) -> jnp.ndarray:
        """
        Return maximum conductance array for leak channel.

        Returns
        -------
        g_bar : jnp.ndarray, shape (1,)
            Constant maximal conductance in mS/cm².
        """
        return jnp.array([self.g_leak], dtype=dtype)

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        """
        Compute conductances for a passive ion channel model.

        Parameters
        ----------
        gates : jnp.ndarray, shape (N,0)
            Gating variables (ignored for passive ion channel model).
        g_bar : jnp.ndarray, shape (1,)
            Maximum leak conductance.

        Returns
        -------
        g : jnp.ndarray, shape (N,1)
            Effective conductance at each node. Constant across nodes.

        Notes
        -----
        - Passive membrane has no voltage-dependent gating.
        """
        N = gates.shape[0]
        return jnp.full((N, 1), g_bar[0], dtype=dtype)

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        """
        Alpha rate constants for gating variables (passive has none).

        Parameters
        ----------
        V : jnp.ndarray, shape (N,)
            Membrane voltages (ignored).

        Returns
        -------
        alpha : jnp.ndarray, shape (N,0)
            Empty array, as there are no gating variables.
        """
        N = V.shape[0]
        return jnp.zeros((N, 0), dtype=dtype)

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        """
        Beta rate constants for gating variables (passive has none).

        Parameters
        ----------
        V : jnp.ndarray, shape (N,)
            Membrane voltages (ignored).

        Returns
        -------
        beta : jnp.ndarray, shape (N,0)
            Empty array, as there are no gating variables.
        """
        N = V.shape[0]
        return jnp.zeros((N, 0), dtype=dtype)

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        """
        Initialize gating variables for passive ion channel (empty).

        Parameters
        ----------
        V0_mV : jnp.ndarray, shape (N,)
            Initial membrane voltages (mV).

        Returns
        -------
        gates : jnp.ndarray, shape (N,0)
            Empty array, as passive membranes have no gating variables.
        """
        V0 = jnp.atleast_1d(V0_mV)
        return jnp.zeros((V0.shape[0], 0), dtype=dtype)

    def gate_names(self) -> tuple[str, ...]:
        return ()

    def conductance_names(self) -> tuple[str, ...]:
        return ("g_l",)

    def current_names(self) -> tuple[str, ...]:
        return ("I_l",)
