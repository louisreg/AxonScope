from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from axonscope.utils.settings import dtype
from axonscope.channel_models.rate_tables import (
    RateTable,
    RateTableConfig,
    make_rate_table_config,
)
from dataclasses import fields, is_dataclass
from typing import Any, List, Callable, NamedTuple

Array1D = jnp.ndarray
Array2D = jnp.ndarray
GFunc = Callable[[Array2D, Array1D], Array2D]
RateFunc = Callable[[Array1D], Array2D]


class MembraneStepPlan(NamedTuple):
    state: tuple[jnp.ndarray, ...]
    linearization_gates: jnp.ndarray
    total_outward_current: jnp.ndarray
    explicit_outward_current: jnp.ndarray
    correction_current: jnp.ndarray


class MembraneStateSpec(NamedTuple):
    name: str


def _static_value_signature(value: Any) -> Any:
    """Return a hashable structural signature for model configuration values."""
    if isinstance(value, IonChannelModelBase):
        return value._static_signature()
    if is_dataclass(value) and not isinstance(value, type):
        return (
            value.__class__.__module__,
            value.__class__.__qualname__,
            tuple(
                (field.name, _static_value_signature(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    if isinstance(value, dict):
        items = (
            (_static_value_signature(key), _static_value_signature(val))
            for key, val in value.items()
        )
        return tuple(sorted(items, key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_static_value_signature(item) for item in value)
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        return ("array", str(arr.dtype), arr.shape, arr.tobytes())
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        arr = np.asarray(value)
        if arr.shape == ():
            return ("scalar", str(arr.dtype), arr.item())
        return ("array", str(arr.dtype), arr.shape, arr.tobytes())
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


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
        self.dtype = dtype
        self.q10: float = dtype(1.0)
        self._rate_table: RateTable | None = None

    def _static_signature(self) -> tuple[Any, ...]:
        """Structural identity used when channel models are JAX static args."""
        return (
            self.__class__.__module__,
            self.__class__.__qualname__,
            tuple(
                (name, _static_value_signature(value))
                for name, value in sorted(self.__dict__.items())
            ),
        )

    def __eq__(self, other: object) -> bool:
        if self.__class__ is not other.__class__:
            return False
        return self._static_signature() == other._static_signature()

    def __hash__(self) -> int:
        return hash(self._static_signature())

    def supports_stateless_vm_only_fast_path(self) -> bool:
        """Whether Vm-only solvers may skip post-solve membrane finalization."""
        cls = self.__class__
        base = IonChannelModelBase
        return (
            self.membrane_state_specs() == ()
            and cls.final_gate_update is base.final_gate_update
            and cls.prepare_membrane_step is base.prepare_membrane_step
            and cls.finalize_membrane_step is base.finalize_membrane_step
        )

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
        - Units in mS/cm².
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

    def exact_rate_constants(self, V_mV: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return exact alpha/beta rates.

        Subclasses with shared intermediate computations should override this
        method. The solver calls `rate_constants()` so exact and tabulated
        paths share one internal contract.
        """
        return self.alpha_funcs(V_mV), self.beta_funcs(V_mV)

    def rate_constants(self, V_mV: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return alpha/beta rates, using a voltage lookup table when enabled."""
        if self._rate_table is None:
            return self.exact_rate_constants(V_mV)
        return self._tabulated_rate_constants(V_mV)

    def gating_inf_tau(self, V_mV: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return steady-state gates and time constants from alpha/beta rates."""
        alpha, beta = self.rate_constants(V_mV)
        sum_ab = jnp.maximum(alpha + beta, dtype(1e-12))
        g_inf = alpha / sum_ab
        tau = dtype(1.0) / (self.q10 * sum_ab)
        return g_inf, tau

    def enable_rate_table(
        self,
        *,
        config: RateTableConfig | None = None,
        v_min_mV: float = -120.0,
        v_max_mV: float = 80.0,
        step_mV: float = 0.05,
        clamp: bool = True,
    ) -> "IonChannelModelBase":
        """Precompute a voltage lookup table for alpha/beta rate constants."""
        resolved = make_rate_table_config(
            config,
            v_min_mV=v_min_mV,
            v_max_mV=v_max_mV,
            step_mV=step_mV,
            clamp=clamp,
        )
        self._rate_table = RateTable.build(
            resolved,
            dtype_local=self.dtype,
            exact_rate_constants=self.exact_rate_constants,
        )
        return self

    def disable_rate_table(self) -> "IonChannelModelBase":
        self._rate_table = None
        return self

    @property
    def has_rate_table(self) -> bool:
        return self._rate_table is not None

    @property
    def rate_table_config(self) -> RateTableConfig | None:
        return None if self._rate_table is None else self._rate_table.config

    def _tabulated_rate_constants(self, V_mV: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self._rate_table is None:
            return self.exact_rate_constants(V_mV)
        return self._rate_table.interpolate(V_mV, dtype_local=self.dtype)

    def conductances(self, gates: jnp.ndarray) -> jnp.ndarray:
        """
        Compute per-channel open conductances from the current channel state.

        Subclasses can override this when conductance computation requires a
        more specialized physical law than the default Hodgkin-Huxley style
        `g_funcs(gates, g_bar)`.
        """
        return self.g_funcs(gates, self.g_bar)

    def gate_names(self) -> tuple[str, ...]:
        sample = self.init_gates(jnp.array([dtype(0.0)], dtype=dtype))
        n_gates = int(sample.shape[-1]) if sample.ndim > 1 else 0
        return tuple(f"gate_{i}" for i in range(n_gates))

    def conductance_names(self) -> tuple[str, ...]:
        n_channels = int(self.g_bar.shape[0])
        return tuple(f"g_{i}" for i in range(n_channels))

    def current_names(self) -> tuple[str, ...]:
        names = []
        for i, g_name in enumerate(self.conductance_names()):
            if g_name.startswith("g_"):
                names.append("I_" + g_name[2:])
            else:
                names.append(f"I_{i}")
        return tuple(names)

    def gate_trace_matrix(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        _ = state
        return gates

    def conductance_trace_matrix(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        _ = state
        return self.conductances(gates)

    def ionic_current_trace_matrix(
        self,
        V_mV: jnp.ndarray,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        _ = state
        g_open = self.conductances(gates)
        return g_open * (jnp.atleast_1d(V_mV)[:, None] - self.E_rev[None, :])

    def membrane_state_trace_matrix(
        self,
        state: tuple[jnp.ndarray, ...],
    ) -> jnp.ndarray:
        if not state:
            return jnp.zeros((0, 0), dtype=dtype)
        return jnp.stack(state, axis=1)

    def cn_gate_update(self, g_prev: jnp.ndarray, V_mV: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        Advance channel state by one frozen-voltage exponential gate update.

        This matches NEURON/NRV `cnexp` for first-order HH-style kinetics:
            dg/dt = alpha(V) * (1 - g) - beta(V) * g
        with rates frozen over the timestep.
        """
        alpha, beta = self.rate_constants(V_mV)
        alpha = self.q10 * alpha
        beta = self.q10 * beta
        sum_ab = jnp.maximum(alpha + beta, dtype(1e-12))
        g_inf = alpha / sum_ab
        tau = dtype(1.0) / sum_ab
        return g_inf - (g_inf - g_prev) * jnp.exp(-dtype(dt) / tau)

    def final_gate_update(
        self,
        gates_prev: jnp.ndarray,
        V_mV_prev: jnp.ndarray,
        V_mV_new: jnp.ndarray,
        dt: float,
        gates_predictor: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Final gate state committed after the voltage solve.

        By default we keep the predictor produced before the voltage solve.
        Models can override this when their NRV/NEURON reference behaves more
        like a post-voltage `cnexp` update.
        """
        _ = gates_prev, V_mV_prev, V_mV_new, dt
        return gates_predictor

    def currents(self, V_mV: jnp.ndarray, gates: jnp.ndarray) -> jnp.ndarray:
        """
        Compute total membrane current density carried by this channel model.

        Subclasses can override this when the physical current law is not
        reducible to `sum(g_i * (V - E_i))`.
        """
        g_open = self.conductances(gates)
        return jnp.sum(g_open * (jnp.atleast_1d(V_mV)[:, None] - self.E_rev[None, :]), axis=1)

    def total_conductance(self, gates: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the total open membrane conductance seen by the solver.
        """
        return jnp.sum(self.conductances(gates), axis=1)

    def membrane_conductance_terms(self, gates: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return the linearized membrane terms `(Gm, Gm * Erev)` used by implicit
        solvers and extracellular double-cable coupling.
        """
        g_open = self.conductances(gates)
        Gm = jnp.sum(g_open, axis=1)
        GE = jnp.sum(g_open * self.E_rev[None, :], axis=1)
        return Gm, GE

    def membrane_state_specs(self) -> tuple[MembraneStateSpec, ...]:
        """
        Optional membrane-state variables tracked in addition to gates.

        Examples include ion pools, buffers, pump states, or any auxiliary
        dynamical quantities specific to a membrane model.
        """
        return ()

    def membrane_state_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.membrane_state_specs())

    def membrane_state_dict(
        self,
        state: tuple[jnp.ndarray, ...],
    ) -> dict[str, jnp.ndarray]:
        names = self.membrane_state_names()
        if len(state) != len(names):
            raise ValueError(f"Expected {len(names)} membrane states, got {len(state)}.")
        return dict(zip(names, state, strict=True))

    def init_membrane_state(
        self, Nx: int, dtype_local: jnp.dtype, V0_mV: jnp.ndarray
    ) -> tuple[jnp.ndarray, ...]:
        _ = Nx, dtype_local, V0_mV
        return ()

    def prepare_membrane_step(
        self,
        V_mV: jnp.ndarray,
        gates_prev: jnp.ndarray,
        gates_new: jnp.ndarray,
        state: tuple[jnp.ndarray, ...],
        dt: float,
        I_ion: jnp.ndarray,
        I_background: jnp.ndarray,
    ) -> MembraneStepPlan:
        _ = gates_prev, dt
        return MembraneStepPlan(
            state=state,
            linearization_gates=gates_new,
            total_outward_current=I_background + I_ion,
            explicit_outward_current=I_background,
            correction_current=jnp.zeros_like(V_mV),
        )

    def finalize_membrane_step(
        self,
        V_mV_prev: jnp.ndarray,
        V_mV_new: jnp.ndarray,
        gates_prev: jnp.ndarray,
        gates_new: jnp.ndarray,
        state_prev: tuple[jnp.ndarray, ...],
        step_plan: MembraneStepPlan,
        dt: float,
    ) -> tuple[jnp.ndarray, ...]:
        _ = V_mV_prev, V_mV_new, gates_prev, gates_new, state_prev, dt
        return step_plan.state

    def diagnostic_names(self) -> tuple[str, ...]:
        return ()

    def compute_step_diagnostics(
        self,
        V_mV_prev: jnp.ndarray,
        V_mV_new: jnp.ndarray,
        gates_prev: jnp.ndarray,
        gates_new: jnp.ndarray,
        state_prev: tuple[jnp.ndarray, ...],
        state_new: tuple[jnp.ndarray, ...],
        step_plan: MembraneStepPlan,
        I_ion: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ...]:
        _ = V_mV_prev, V_mV_new, gates_prev, gates_new, state_prev, state_new, step_plan, I_ion
        return ()

    def I_background(self, Nx: int) -> jnp.ndarray:
        """Constant voltage-independent current density (µA/cm²), positive = outward.

        Override in subclasses that need a background current (e.g. electrogenic pumps).
        Default: zero everywhere.
        """
        return jnp.zeros(Nx, dtype=dtype)


class CompositeICM(IonChannelModelBase):


    def __init__(self, models: List[IonChannelModelBase]):
        super().__init__()
        self.models: List[IonChannelModelBase] = models  # must be static for JIT --> realy??
        if not models:
            raise ValueError("CompositeICM requires at least one submodel.")
        stateful = [
            model.__class__.__name__
            for model in models
            if model.membrane_state_specs()
        ]
        if stateful:
            names = ", ".join(stateful)
            raise NotImplementedError(
                "CompositeICM does not yet support stateful membrane components; "
                f"got: {names}."
            )

        # sizes per submodel (Python ints)
        sizes = [len(m.gate_names()) for m in models]
        self.sizes: List[int] = sizes

        # cumulative boundaries for slicing gates
        cum = [0]
        for s in sizes:
            cum.append(cum[-1] + s)
        self.cum_sizes: List[int] = cum  # length = len(models) + 1

        # Passive submodels expose q10 = 1.0 while active gate models usually
        # share a common temperature scaling. Keep that shared value so
        # CompositeICM behaves like the corresponding mono-model in solvers.
        active_q10 = [
            float(m.q10) for m, n_gates in zip(models, sizes) if n_gates > 0
        ]
        if active_q10:
            ref_q10 = active_q10[0]
            if not all(jnp.isclose(q, ref_q10) for q in active_q10[1:]):
                raise ValueError(
                    "CompositeICM requires a common q10 across gated submodels."
                )
            self.q10 = dtype(ref_q10)


    @property
    def g_bar(self):
        # Concatenate channel maximal conductances (mS/cm^2).
        return jnp.concatenate([m.g_bar for m in self.models])

    @property
    def E_rev(self):
        # concatenation of reversal potentials (vector)
        return jnp.concatenate([m.E_rev for m in self.models])

    @staticmethod
    def _dedupe_names(names: List[str]) -> tuple[str, ...]:
        counts: dict[str, int] = {}
        out: list[str] = []
        for name in names:
            counts[name] = counts.get(name, 0) + 1
            if counts[name] == 1:
                out.append(name)
            else:
                out.append(f"{name}_{counts[name]}")
        return tuple(out)

    @staticmethod
    def _group_name_indices(names: List[str]) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
        order: list[str] = []
        groups: dict[str, list[int]] = {}
        for i, name in enumerate(names):
            if name not in groups:
                order.append(name)
                groups[name] = []
            groups[name].append(i)
        return tuple(order), tuple(tuple(groups[name]) for name in order)

    @staticmethod
    def _aggregate_columns(
        values: jnp.ndarray,
        groups: tuple[tuple[int, ...], ...],
    ) -> jnp.ndarray:
        cols = []
        for indices in groups:
            if len(indices) == 1:
                cols.append(values[:, indices[0]])
            else:
                cols.append(jnp.sum(values[:, jnp.array(indices)], axis=1))
        if not cols:
            return jnp.zeros((values.shape[0], 0), dtype=values.dtype)
        return jnp.stack(cols, axis=1)

    def gate_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for model in self.models:
            names.extend(model.gate_names())
        return self._dedupe_names(names)

    def conductance_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for model in self.models:
            names.extend(model.conductance_names())
        unique_names, _ = self._group_name_indices(names)
        return unique_names

    def current_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for model in self.models:
            names.extend(model.current_names())
        unique_names, _ = self._group_name_indices(names)
        return unique_names

    # -------------------------
    # init_gates
    # -------------------------
    def init_gates(self, V0_mV):
        """
        Return initial gating variables for each submodel concatenated.
        Output shape: (batch, total_gates)
        """
        g_inf, _ = self.gating_inf_tau(V0_mV)
        return g_inf


    def alpha_funcs(self, V):
        """
        Concatenate alpha(V) from each submodel.
        Output shape: (batch, total_gates)
        """
        alpha, _ = self.rate_constants(V)
        return alpha


    def beta_funcs(self, V):
        """
        Concatenate beta(V) from each submodel.
        Output shape: (batch, total_gates)
        """
        _, beta = self.rate_constants(V)
        return beta

    def exact_rate_constants(self, V):
        alpha_parts = []
        beta_parts = []
        for m in self.models:
            alpha_i, beta_i = m.exact_rate_constants(V)
            alpha_parts.append(jnp.atleast_2d(alpha_i))
            beta_parts.append(jnp.atleast_2d(beta_i))
        if not alpha_parts:
            empty = jnp.zeros((V.shape[0], 0), dtype=dtype)
            return empty, empty
        return jnp.concatenate(alpha_parts, axis=-1), jnp.concatenate(beta_parts, axis=-1)


    def I_background(self, Nx: int) -> jnp.ndarray:
        total = jnp.zeros(Nx, dtype=dtype)
        for m in self.models:
            total = total + m.I_background(Nx)
        return total

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

    def conductance_trace_matrix(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        _ = state
        raw_names: list[str] = []
        for model in self.models:
            raw_names.extend(model.conductance_names())
        _, groups = self._group_name_indices(raw_names)
        raw_values = self.conductances(gates)
        return self._aggregate_columns(raw_values, groups)

    def ionic_current_trace_matrix(
        self,
        V_mV: jnp.ndarray,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        _ = state
        raw_names: list[str] = []
        for model in self.models:
            raw_names.extend(model.current_names())
        _, groups = self._group_name_indices(raw_names)
        g_open = self.conductances(gates)
        raw_currents = g_open * (jnp.atleast_1d(V_mV)[:, None] - self.E_rev[None, :])
        return self._aggregate_columns(raw_currents, groups)
