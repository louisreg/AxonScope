from __future__ import annotations
import jax.numpy as jnp
from abc import ABC, abstractmethod
from axonscope.settings import dtype
from typing import List, Callable

Array1D = jnp.ndarray
Array2D = jnp.ndarray
GFunc = Callable[[Array2D, Array1D], Array2D]
RateFunc = Callable[[Array1D], Array2D]

class IonChannelModelBase(ABC):
    """
    Abstract base class for ion channel models of excitable cells.

    This class defines the interface and core properties that any
    ion channel model must implement, including passive, Hodgkin-Huxley,
    and Markov-type channel models. Designed to be JAX-friendly: all
    dynamic states are passed explicitly and no in-place mutation occurs.
    
    Attributes
    ----------
    dtype : float
        Floating-point precision used for all arrays (from `axonscope.settings`).
    q10 : float
        Temperature scaling factor for gating kinetics.
    """

    def __init__(self) -> None:
        """
        Initialize base ion channel model attributes.

        Notes
        -----
        - q10 is set to 1 by default but should be overridden
          in concrete implementations.
        - dtype specifies the numerical type for all arrays.
        """
        super().__init__()
        self.dtype = dtype
        self.q10: float = dtype(1.0)

    @abstractmethod
    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        """
        Initialize gating variables based on the initial membrane voltage.

        Parameters
        ----------
        V0_mV : jnp.ndarray, shape (N,)
            Initial voltage at each compartment or node in millivolts.

        Returns
        -------
        gates : jnp.ndarray, shape (N, n_gates)
            Initialized gating variables at each node.

        Notes
        -----
        - Must return a JAX array with no in-place mutation.
        - For passive ion channel, n_gates = 0 (may return empty array).
        - For Hodgkin-Huxley models, this may include m, h, n, s, etc.
        - Enables JIT compilation by being a pure function.
        """
        pass

    @property
    @abstractmethod
    def g_bar(self) -> jnp.ndarray:
        """
        Maximum conductances for each channel.

        Returns
        -------
        g_bar : jnp.ndarray, shape (n_channels,)
            Maximum conductance values.
            Example:
                - Passive: [g_leak]
                - Hodgkin-Huxley: [g_na, g_k, g_l]

        Notes
        -----
        - Units in mho/cm².
        - Should be immutable once set.
        """
        pass

    @property
    @abstractmethod
    def E_rev(self) -> jnp.ndarray:
        """
        Reversal potentials for each channel.

        Returns
        -------
        E_rev : jnp.ndarray, shape (n_channels,)
            Reversal potential of each channel.
            Example:
                - Passive: [E_leak]
                - Hodgkin-Huxley: [E_na, E_k, E_l]

        Notes
        -----
        - Units in millivolts (mV).
        - Used to calculate ionic currents: I = g * (V - E_rev).
        """
        pass

    @abstractmethod
    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        """
        Compute channel conductances from gating variables.

        Parameters
        ----------
        gates : jnp.ndarray, shape (N, n_gates)
            Gating variable values at each node.
        g_bar : jnp.ndarray, shape (n_channels,)
            Maximum conductances for each channel.

        Returns
        -------
        g : jnp.ndarray, shape (N, n_channels)
            Effective conductance for each channel and node.

        Notes
        -----
        - Typically implements Hodgkin-Huxley style formulas,
          e.g., g_na = g_bar[0] * m**3 * h.
        - Must be vectorized over all nodes (N) for efficient JAX execution.
        """
        pass

    @abstractmethod
    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        """
        Compute alpha rate constants for gating variables.

        Parameters
        ----------
        V : jnp.ndarray, shape (N,)
            Membrane voltage at each node in millivolts.

        Returns
        -------
        alpha : jnp.ndarray, shape (N, n_gates)
            Forward rate constants for each gating variable.

        Notes
        -----
        - Used in gating dynamics: dx/dt = alpha * (1-x) - beta * x
        - Should be pure and vectorized for JAX JIT compatibility.
        """
        pass

    @abstractmethod
    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        """
        Compute beta rate constants for gating variables.

        Parameters
        ----------
        V : jnp.ndarray, shape (N,)
            Membrane voltage at each node in millivolts.

        Returns
        -------
        beta : jnp.ndarray, shape (N, n_gates)
            Backward rate constants for each gating variable.

        Notes
        -----
        - Must be consistent with the ordering of `alpha_funcs`.
        - Used in gating dynamics: dx/dt = alpha * (1-x) - beta * x.
        - Must be a pure function for JAX compatibility.
        """
        pass


class CompositeICM(IonChannelModelBase):


    def __init__(self, models: List[IonChannelModelBase]):
        super().__init__()
        self.models: List[IonChannelModelBase] = models  # must be static for JIT --> realy??

        # sizes per submodel (Python ints)
        sizes = [int(m.init_gates(jnp.array([0.0])).shape[-1]) for m in models]
        self.sizes: List[int] = sizes

        # cumulative boundaries for slicing gates
        cum = [0]
        for s in sizes:
            cum.append(cum[-1] + s)
        self.cum_sizes: List[int] = cum  # length = len(models) + 1


    @property
    def g_bar(self):
        # concatenation of channel maximal conductances (vector)
        return jnp.concatenate([m.g_bar for m in self.models])

    @property
    def E_rev(self):
        # concatenation of reversal potentials (vector)
        return jnp.concatenate([m.E_rev for m in self.models])

    # -------------------------
    # init_gates
    # -------------------------
    def init_gates(self, V0_mV):
        """
        Return initial gating variables for each submodel concatenated.
        Output shape: (batch, total_gates)
        """
        outs = []
        for m in self.models:
            g = m.init_gates(V0_mV)            # shape maybe (batch, n_i) or (n_i,)
            g = jnp.atleast_2d(g)          # ensure (batch, n_i)
            outs.append(g)
        if len(outs) == 0:
            return jnp.zeros((V0_mV.shape[0], 0))
        return jnp.concatenate(outs, axis=-1)


    def alpha_funcs(self, V):
        """
        Concatenate alpha(V) from each submodel.
        Output shape: (batch, total_gates)
        """
        outs = []
        for m in self.models:
            a = m.alpha_funcs(V)   # expected (batch, n_i) or (n_i,)
            a = jnp.atleast_2d(a)
            outs.append(a)
        if len(outs) == 0:
            return jnp.zeros((V.shape[0], 0))
        return jnp.concatenate(outs, axis=-1)


    def beta_funcs(self, V):
        """
        Concatenate beta(V) from each submodel.
        Output shape: (batch, total_gates)
        """
        outs = []
        for m in self.models:
            b = m.beta_funcs(V)
            b = jnp.atleast_2d(b)
            outs.append(b)
        if len(outs) == 0:
            return jnp.zeros((V.shape[0], 0))
        return jnp.concatenate(outs, axis=-1)


    def g_funcs(self, gates, g_bar=None):
        """
        Compute conductances from gating variables using each submodel's g_funcs.
        gates: (batch, total_gates) or (total_gates,)
        Returns concatenated g parts: shape (batch, n_channels_out)
        Note: assumes each submodel.g_funcs returns shape (batch, k_i) (commonly k_i=1).
        """
        _ : g_bar
        outs = []
        for i, m in enumerate(self.models):
            i0 = self.cum_sizes[i]
            i1 = self.cum_sizes[i + 1]
            sub = gates[..., i0:i1]           # (batch, n_i)
            g_part = m.g_funcs(sub, m.g_bar)  # (batch, k_i)
            g_part = jnp.atleast_2d(g_part)
            outs.append(g_part)
        if len(outs) == 0:
            return jnp.zeros((gates.shape[0], 0))
        return jnp.concatenate(outs, axis=-1)

class CompositeICM_nope(IonChannelModelBase):
    """
    Combines multiple ion channel models into a single composite model.

    This allows combining ionic channels (e.g., NaHHMM + BorgKDRMM) 
    into a single ion channel model for cable simulations. 
    The class provides concatenated gating variables, conductances, 
    reversal potentials, and alpha/beta rate functions suitable for JAX JIT.

    Attributes
    ----------
    models : list[IonChannelModelBase]
        Individual ion channel model instances to combine.
    gate_slices : list[slice]
        Precomputed slices indicating each model's gating variables 
        in the concatenated gating array.
    _g_bar : jnp.ndarray
        Concatenated maximum conductances for all sub-models.
    _E_rev : jnp.ndarray
        Concatenated reversal potentials for all sub-models.
    _alpha_funcs_list : list[RateFunc]
        List of alpha rate functions for each sub-model.
    _beta_funcs_list : list[RateFunc]
        List of beta rate functions for each sub-model.
    """

    def __init__(self, models: List[IonChannelModelBase]):
        """
        Initialize the composite ion channel model.

        Parameters
        ----------
        models : list[IonChannelModelBase]
            List of individual ion channel model instances to combine.
        """
        super().__init__()
        self.models: List[IonChannelModelBase] = models  # must be static for JIT --> realy??

        # sizes per submodel (Python ints)
        sizes = [int(m.init_gates(jnp.array([0.0])).shape[-1]) for m in models]
        self.sizes: List[int] = sizes

        # cumulative boundaries for slicing gates
        cum = [0]
        for s in sizes:
            cum.append(cum[-1] + s)
        self.cum_sizes: List[int] = cum  # length = len(models) + 1

        self._g_bar = jnp.concatenate([m.g_bar for m in self.models])
        self._E_rev = jnp.concatenate([m.E_rev for m in self.models])

    @property
    def g_bar(self) -> Array1D:
        """Return concatenated maximum conductances for all sub-models."""
        return self._g_bar

    @property
    def E_rev(self) -> Array1D:
        """Return concatenated reversal potentials for all sub-models."""
        return self._E_rev


    def alpha_funcs(self, V: Array1D) -> Array2D:           #TODO: maybe we can pre-compute this??
        """
        Instance method returning concatenated alpha rates for all sub-models.

        Parameters
        ----------
        V : Array1D
            Membrane voltages for each compartment (mV), shape (N,).

        Returns
        -------
        Array2D
            Concatenated alpha rates, shape (N, total_gates).
        """
        outs = []
        for m in self.models:
            a = m.alpha_funcs(V)   # expected (batch, n_i) or (n_i,)
            a = jnp.atleast_2d(a)
            outs.append(a)
        if len(outs) == 0:
            return jnp.zeros((V.shape[0], 0))
        return jnp.concatenate(outs, axis=-1)

    def beta_funcs(self, V: Array1D) -> Array2D:            #TODO: maybe we can pre-compute this??
        """
        Instance method returning concatenated beta rates for all sub-models.

        Parameters
        ----------
        V : Array1D
            Membrane voltages for each compartment (mV), shape (N,).

        Returns
        -------
        Array2D
            Concatenated beta rates, shape (N, total_gates).
        """
        outs = []
        for m in self.models:
            b = m.beta_funcs(V)
            b = jnp.atleast_2d(b)
            outs.append(b)
        if len(outs) == 0:
            return jnp.zeros((V.shape[0], 0))
        return jnp.concatenate(outs, axis=-1)

    def g_funcs(self, gates: Array2D, g_bar: Array1D) -> Array2D:         #TODO: maybe we can pre-compute this??  
        """
        Compute conductances for each sub-model and concatenate.

        Parameters
        ----------
        gates : Array2D
            Concatenated gating variables for all sub-models, shape (N, total_gates).
        g_bar : Array1D
            Concatenated maximum conductances, shape (total_channels,).

        Returns
        -------
        Array2D
            Concatenated conductances for all channels, shape (N, total_channels).
        """
        _ = g_bar
        outs = []
        for i, m in enumerate(self.models):
            i0 = self.cum_sizes[i]
            i1 = self.cum_sizes[i + 1]
            sub = gates[..., i0:i1]           # (batch, n_i)
            g_part = m.g_funcs(sub, m.g_bar)  # (batch, k_i)
            g_part = jnp.atleast_2d(g_part)
            outs.append(g_part)
        if len(outs) == 0:
            return jnp.zeros((gates.shape[0], 0))
        return jnp.concatenate(outs, axis=-1)

    def init_gates(self, V0_mV: Array1D) -> Array2D:        
        """
        Initialize gating variables to steady-state for each sub-model.

        Parameters
        ----------
        V0_mV : Array1D
            Initial membrane voltages for each compartment (mV), shape (N,).

        Returns
        -------
        Array2D
            Concatenated initial gating variables for all sub-models, shape (N, total_gates).
        """
        outs = []
        for m in self.models:
            g = m.init_gates(V0_mV)            # shape maybe (batch, n_i) or (n_i,)
            g = jnp.atleast_2d(g)          # ensure (batch, n_i)
            outs.append(g)
        if len(outs) == 0:
            return jnp.zeros((V0_mV.shape[0], 0))
        return jnp.concatenate(outs, axis=-1)


