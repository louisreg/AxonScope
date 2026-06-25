from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Callable

from axonscope.channel_models.base_channel_model import IonChannelModelBase

# ---------------------- Type aliases ----------------------
Array1D = jnp.ndarray  # shape (N,)
Array2D = jnp.ndarray  # shape (N, n_gates)
GFunc = Callable[[Array2D, Array1D], Array2D]
RateFunc = Callable[[Array1D], Array2D]


class ICMBackend(Protocol):
    dtype: jnp.dtype
    n_gates_max: int
    n_channels_max: int

    @property
    def Nx(self) -> int: ...

    def init_gates(self, V0_mV: Array1D) -> Array2D: ...

    def alpha(self, V_mV: Array1D) -> Array2D: ...

    def beta(self, V_mV: Array1D) -> Array2D: ...

    def conductances(self, gates: Array2D) -> Array2D: ...

    def cn_gate_update(self, g_prev: Array2D, V_mV: Array1D, dt: float) -> Array2D: ...

    def currents(self, V_mV: Array1D, gates: Array2D) -> Array1D: ...

    def total_conductance(self, gates: Array2D) -> Array1D: ...

    def membrane_conductance_terms(self, gates: Array2D) -> tuple[Array1D, Array1D]: ...

    def background_current(self) -> Array1D: ...


class Gating:
    """
    Utility class for gating variable dynamics and ionic currents.

    Provides static methods to compute:
    - Ionic currents from voltage and gating variables
    - Steady-state values and time constants of gates
    - Full-step CNEXP update
    - Full-step Crank-Nicolson update for gating variables

    All methods are vectorized for JAX arrays.
    """

    @staticmethod
    def compute_currents(
        V: Array1D,
        gates: Array2D,
        g_bar: Array1D,
        g_func: GFunc,
        E_rev: Array1D
    ) -> Array1D:
        """
        Compute total ionic currents for each compartment.

        Implements:
            I_ion = Σ_i g_i(V, gates) * (V - E_i)
        where g_i is computed from the gating variables.

        Parameters
        ----------
        V : Array1D, shape (N,)
            Membrane voltage (mV) for each compartment.
        gates : Array2D, shape (N, n_gates)
            Current gating variable values.
        g_bar : Array1D, shape (n_channels,)
            Maximum conductances for each channel.
        g_func : Callable
            Function computing conductances from gating variables: g_func(gates, g_bar).
        E_rev : Array1D, shape (n_channels,)
            Reversal potentials for each channel (mV).

        Returns
        -------
        I_tot : Array1D, shape (N,)
            Total ionic current for each compartment (µA/cm²).
        """
        V = jnp.atleast_1d(V)
        g_vals = g_func(gates, g_bar)          # shape (N, n_channels)
        delta_V = V[:, None] - E_rev[None, :]  # broadcasting
        I_tot = jnp.sum(g_vals * delta_V, axis=1)
        return I_tot

    @staticmethod
    def rates(
        V: Array1D,
        q10: float,
        alpha_fun: RateFunc,
        beta_fun: RateFunc
    ) -> Tuple[Array2D, Array2D]:
        """
        Compute steady-state values and time constants for gating variables.

        Gating dynamics follow:
            dg/dt = α(V) * (1 - g) - β(V) * g
        which can be rewritten as:
            dg/dt = (g_inf - g) / τ
        with:
            g_inf = α / (α + β)
            τ     = 1 / (q10 * (α + β))

        Parameters
        ----------
        V : Array1D, shape (N,)
            Membrane voltage (mV) for each compartment.
        q10 : float
            Temperature scaling factor.
        alpha_fun : Callable
            Function returning α(V), shape (N, n_gates)
        beta_fun : Callable
            Function returning β(V), shape (N, n_gates)

        Returns
        -------
        g_inf : Array2D, shape (N, n_gates)
            Steady-state gating values.
        tau : Array2D, shape (N, n_gates)
            Time constants (ms) for each gating variable.
        """
        alpha = alpha_fun(V)
        beta = beta_fun(V)
        sum_ab = jnp.maximum(alpha + beta, 1e-12)
        g_inf = alpha / sum_ab
        tau = 1.0 / (q10 * sum_ab)
        return g_inf, tau

    @staticmethod
    def update_gates(
        gates: Array2D,
        V: Array1D,
        dt: float,
        q10: float,
        alpha_fun: RateFunc,
        beta_fun: RateFunc
    ) -> Array2D:
        """
        CNEXP update for gating variables (full time step).

        Implements:
            g(t + dt) = g_inf - (g_inf - g) * exp(-dt / tau)

        Parameters
        ----------
        gates : Array2D, shape (N, n_gates)
            Current gating values.
        V : Array1D, shape (N,)
            Membrane voltage (mV).
        dt : float
            Time step (ms).
        q10 : float
            Temperature scaling factor.
        alpha_fun : Callable
            Function computing alpha rates α(V), shape (N, n_gates)
        beta_fun : Callable
            Function computing beta rates β(V), shape (N, n_gates)

        Returns
        -------
        gates_new : Array2D, shape (N, n_gates)
            Updated gating variables after dt.
        """
        g_inf, tau = Gating.rates(V, q10, alpha_fun, beta_fun)
        return g_inf - (g_inf - gates) * jnp.exp(-dt / tau)

    @staticmethod
    def cn_gate_update(
        g_prev: Array2D,
        V: Array1D,
        dt: float,
        alpha_fun: RateFunc,
        beta_fun: RateFunc,
        q10: float = 1.0
    ) -> Array2D:
        """
        Crank-Nicolson update for gating variables over one full time step `dt`.

        This helper solves the linear scalar gating ODE
            dg/dt = α(V) * (1 - g) - β(V) * g
        with a Crank-Nicolson discretization evaluated at the provided voltage
        `V`. In the current solvers, rates are frozen over the step and gates
        are advanced directly from `t_n` to `t_{n+1}`.

        Implements the Crank-Nicolson discretization of dg/dt = α(1-g) - βg:
            g_new = (α/d) + ((1/dt - 0.5*(α+β))/d) * g_prev
            with d = (1/dt) + 0.5*(α+β)

        Parameters
        ----------
        g_prev : Array2D, shape (N, n_gates)
            Gating variables at the start of the step.
        V : Array1D, shape (N,)
            Membrane voltage (mV) used to evaluate α and β.
        dt : float
            Time step duration (ms).
        alpha_fun : Callable
            Function computing alpha rates α(V), shape (N, n_gates).
        beta_fun : Callable
            Function computing beta rates β(V), shape (N, n_gates).
        q10 : float, optional
            Temperature scaling factor applied to α and β (default 1.0).

        Returns
        -------
        g_new : Array2D, shape (N, n_gates)
            Updated gating variables after dt.
        """
        alpha = q10 * jax.lax.stop_gradient(alpha_fun(V))
        beta  = q10 * jax.lax.stop_gradient(beta_fun(V))
        denom = jnp.maximum(1.0/dt + 0.5*(alpha + beta), 1e-12)
        term1 = alpha / denom
        term2 = ((1.0/dt) - 0.5*(alpha + beta)) / denom * g_prev
        return term1 + term2


    @staticmethod
    def compute_total_conductance(gates, g_bar, g_func):
        g_open = g_func(gates, g_bar)   # shape (N, n_channels), already scaled by g_bar
        return jnp.sum(g_open, axis=1)  # shape (N,) – total conductance per compartment


@dataclass(frozen=True)
class UniformICMBackend:
    """Backend for a spatially uniform ion-channel model."""

    ion_channel: IonChannelModelBase
    nx: int
    n_gates_max: int
    n_channels_max: int
    dtype: jnp.dtype

    @classmethod
    def from_model(cls, ion_channel: IonChannelModelBase, nx: int) -> "UniformICMBackend":
        if nx < 1:
            raise ValueError(f"nx must be >= 1, got {nx}.")
        n_gates = len(ion_channel.gate_names())
        n_channels = int(ion_channel.g_bar.shape[0])
        return cls(
            ion_channel=ion_channel,
            nx=int(nx),
            n_gates_max=n_gates,
            n_channels_max=n_channels,
            dtype=ion_channel.dtype,
        )

    @property
    def Nx(self) -> int:
        return self.nx

    def init_gates(self, V0_mV: Array1D) -> Array2D:
        if V0_mV.shape[0] != self.Nx:
            raise ValueError(f"V0_mV must have shape ({self.Nx},), got {V0_mV.shape}.")
        return self.ion_channel.init_gates(V0_mV)

    def alpha(self, V_mV: Array1D) -> Array2D:
        return self.ion_channel.alpha_funcs(V_mV)

    def beta(self, V_mV: Array1D) -> Array2D:
        return self.ion_channel.beta_funcs(V_mV)

    def conductances(self, gates: Array2D) -> Array2D:
        return self.ion_channel.conductances(gates)

    def cn_gate_update(self, g_prev: Array2D, V_mV: Array1D, dt: float) -> Array2D:
        return self.ion_channel.cn_gate_update(g_prev=g_prev, V_mV=V_mV, dt=dt)

    def currents(self, V_mV: Array1D, gates: Array2D) -> Array1D:
        return self.ion_channel.currents(V_mV=V_mV, gates=gates)

    def total_conductance(self, gates: Array2D) -> Array1D:
        return self.ion_channel.total_conductance(gates)

    def membrane_conductance_terms(self, gates: Array2D) -> tuple[Array1D, Array1D]:
        return self.ion_channel.membrane_conductance_terms(gates)

    def background_current(self) -> Array1D:
        return self.ion_channel.I_background(self.Nx)


class HeterogeneousICMGroup(NamedTuple):
    model: IonChannelModelBase
    indices: tuple[int, ...]
    gate_size: int
    channel_size: int


@dataclass(frozen=True)
class HeterogeneousICMBackend:
    """Backend for one ion-channel model instance per compartment."""

    icm_vec: tuple[IonChannelModelBase, ...]
    groups: tuple[HeterogeneousICMGroup, ...]
    gate_sizes: tuple[int, ...]
    channel_sizes: tuple[int, ...]
    n_gates_max: int
    n_channels_max: int
    dtype: jnp.dtype

    @classmethod
    def from_icm_vec(cls, icm_vec: tuple[IonChannelModelBase, ...] | list[IonChannelModelBase]) -> "HeterogeneousICMBackend":
        if len(icm_vec) == 0:
            raise ValueError("icm_vec must contain at least one channel model.")

        frozen = tuple(icm_vec)
        grouped: list[tuple[IonChannelModelBase, list[int], int, int]] = []
        group_index_by_signature: dict[tuple[Any, ...], int] = {}
        signature_by_identity: dict[int, tuple[Any, ...]] = {}
        sizes_by_signature: dict[tuple[Any, ...], tuple[int, int]] = {}
        gate_sizes_list: list[int] = []
        channel_sizes_list: list[int] = []
        for i, model in enumerate(frozen):
            identity = id(model)
            signature = signature_by_identity.get(identity)
            if signature is None:
                signature = model._static_signature()
                signature_by_identity[identity] = signature
            sizes = sizes_by_signature.get(signature)
            if sizes is None:
                sizes = (
                    len(model.gate_names()),
                    int(model.g_bar.shape[0]),
                )
                sizes_by_signature[signature] = sizes
            gate_size, channel_size = sizes
            gate_sizes_list.append(gate_size)
            channel_sizes_list.append(channel_size)
            group_index = group_index_by_signature.get(signature)
            if group_index is None:
                group_index_by_signature[signature] = len(grouped)
                grouped.append((model, [i], gate_size, channel_size))
            else:
                grouped[group_index][1].append(i)
        gate_sizes = tuple(gate_sizes_list)
        channel_sizes = tuple(channel_sizes_list)
        n_gates_max = max(gate_sizes) if gate_sizes else 0
        n_channels_max = max(channel_sizes) if channel_sizes else 0
        groups = tuple(
            HeterogeneousICMGroup(
                model=model,
                indices=tuple(indices),
                gate_size=gate_size,
                channel_size=channel_size,
            )
            for model, indices, gate_size, channel_size in grouped
        )

        return cls(
            icm_vec=frozen,
            groups=groups,
            gate_sizes=gate_sizes,
            channel_sizes=channel_sizes,
            n_gates_max=n_gates_max,
            n_channels_max=n_channels_max,
            dtype=frozen[0].dtype,
        )

    @property
    def Nx(self) -> int:
        return len(self.icm_vec)

    def init_gates(self, V0_mV: Array1D) -> Array2D:
        if V0_mV.shape[0] != self.Nx:
            raise ValueError(f"V0_mV must have shape ({self.Nx},), got {V0_mV.shape}.")
        out = jnp.zeros((self.Nx, self.n_gates_max), dtype=self.dtype)
        for group in self.groups:
            model = group.model
            n_g = group.gate_size
            if n_g == 0:
                continue
            idx = jnp.asarray(group.indices, dtype=jnp.int32)
            gi = model.init_gates(V0_mV[idx])
            out = out.at[idx, :n_g].set(gi)
        return out

    def alpha(self, V_mV: Array1D) -> Array2D:
        out = jnp.zeros((self.Nx, self.n_gates_max), dtype=self.dtype)
        for group in self.groups:
            model = group.model
            n_g = group.gate_size
            if n_g == 0:
                continue
            idx = jnp.asarray(group.indices, dtype=jnp.int32)
            ai = model.alpha_funcs(V_mV[idx])
            out = out.at[idx, :n_g].set(ai)
        return out

    def beta(self, V_mV: Array1D) -> Array2D:
        out = jnp.zeros((self.Nx, self.n_gates_max), dtype=self.dtype)
        for group in self.groups:
            model = group.model
            n_g = group.gate_size
            if n_g == 0:
                continue
            idx = jnp.asarray(group.indices, dtype=jnp.int32)
            bi = model.beta_funcs(V_mV[idx])
            out = out.at[idx, :n_g].set(bi)
        return out

    def conductances(self, gates: Array2D) -> Array2D:
        if gates.shape != (self.Nx, self.n_gates_max):
            raise ValueError(
                f"gates must have shape ({self.Nx}, {self.n_gates_max}), got {gates.shape}."
            )
        out = jnp.zeros((self.Nx, self.n_channels_max), dtype=self.dtype)
        for group in self.groups:
            model = group.model
            n_g = group.gate_size
            n_c = group.channel_size
            idx = jnp.asarray(group.indices, dtype=jnp.int32)
            gi = gates[idx, :n_g] if n_g > 0 else jnp.zeros((len(group.indices), 0), dtype=self.dtype)
            gvals = model.g_funcs(gi, model.g_bar)
            out = out.at[idx, :n_c].set(gvals)
        return out

    def cn_gate_update(self, g_prev: Array2D, V_mV: Array1D, dt: float) -> Array2D:
        if g_prev.shape != (self.Nx, self.n_gates_max):
            raise ValueError(
                f"g_prev must have shape ({self.Nx}, {self.n_gates_max}), got {g_prev.shape}."
            )
        out = jnp.zeros_like(g_prev)
        for group in self.groups:
            model = group.model
            n_g = group.gate_size
            if n_g == 0:
                continue
            idx = jnp.asarray(group.indices, dtype=jnp.int32)
            g_new = model.cn_gate_update(
                g_prev=g_prev[idx, :n_g],
                V_mV=V_mV[idx],
                dt=dt,
            )
            out = out.at[idx, :n_g].set(g_new)
        return out

    def currents(self, V_mV: Array1D, gates: Array2D) -> Array1D:
        if V_mV.shape[0] != self.Nx:
            raise ValueError(f"V_mV must have shape ({self.Nx},), got {V_mV.shape}.")
        out = jnp.zeros((self.Nx,), dtype=self.dtype)
        for group in self.groups:
            model = group.model
            n_g = group.gate_size
            idx = jnp.asarray(group.indices, dtype=jnp.int32)
            gi = gates[idx, :n_g] if n_g > 0 else jnp.zeros((len(group.indices), 0), dtype=self.dtype)
            Ii = model.currents(V_mV=V_mV[idx], gates=gi)
            out = out.at[idx].set(Ii)
        return out

    def total_conductance(self, gates: Array2D) -> Array1D:
        out = jnp.zeros((self.Nx,), dtype=self.dtype)
        for group in self.groups:
            model = group.model
            n_g = group.gate_size
            idx = jnp.asarray(group.indices, dtype=jnp.int32)
            gi = gates[idx, :n_g] if n_g > 0 else jnp.zeros((len(group.indices), 0), dtype=self.dtype)
            out = out.at[idx].set(model.total_conductance(gi))
        return out

    def membrane_conductance_terms(self, gates: Array2D) -> tuple[Array1D, Array1D]:
        if gates.shape != (self.Nx, self.n_gates_max):
            raise ValueError(
                f"gates must have shape ({self.Nx}, {self.n_gates_max}), got {gates.shape}."
            )
        Gm = jnp.zeros((self.Nx,), dtype=self.dtype)
        GE = jnp.zeros((self.Nx,), dtype=self.dtype)
        for group in self.groups:
            model = group.model
            n_g = group.gate_size
            idx = jnp.asarray(group.indices, dtype=jnp.int32)
            gi = gates[idx, :n_g] if n_g > 0 else jnp.zeros((len(group.indices), 0), dtype=self.dtype)
            gm_i, ge_i = model.membrane_conductance_terms(gi)
            Gm = Gm.at[idx].set(gm_i)
            GE = GE.at[idx].set(ge_i)
        return Gm, GE

    def background_current(self) -> Array1D:
        out = jnp.zeros((self.Nx,), dtype=self.dtype)
        for group in self.groups:
            idx = jnp.asarray(group.indices, dtype=jnp.int32)
            out = out.at[idx].set(group.model.I_background(len(group.indices)))
        return out


@dataclass(frozen=True)
class PaddedICMBackend:
    """Adapter exposing a shorter backend at a padded spatial width."""

    backend: ICMBackend
    nx: int
    target_nx: int
    n_gates_max: int
    n_channels_max: int
    dtype: jnp.dtype

    @classmethod
    def from_backend(
        cls,
        backend: ICMBackend,
        *,
        target_nx: int,
        n_gates_max: int,
        n_channels_max: int,
    ) -> "PaddedICMBackend":
        if backend.Nx > target_nx:
            raise ValueError(
                f"backend Nx={backend.Nx} cannot be padded to target_nx={target_nx}."
            )
        return cls(
            backend=backend,
            nx=int(backend.Nx),
            target_nx=int(target_nx),
            n_gates_max=int(n_gates_max),
            n_channels_max=int(n_channels_max),
            dtype=backend.dtype,
        )

    @property
    def Nx(self) -> int:
        return self.target_nx

    def _pad_space(self, values: Array1D) -> Array1D:
        out = jnp.zeros((self.target_nx,), dtype=self.dtype)
        return out.at[: self.nx].set(values)

    def _pad_gates(self, values: Array2D) -> Array2D:
        out = jnp.zeros((self.target_nx, self.n_gates_max), dtype=self.dtype)
        return out.at[: self.nx, : values.shape[1]].set(values)

    def _pad_channels(self, values: Array2D) -> Array2D:
        out = jnp.zeros((self.target_nx, self.n_channels_max), dtype=self.dtype)
        return out.at[: self.nx, : values.shape[1]].set(values)

    def _local_gates(self, gates: Array2D) -> Array2D:
        return gates[: self.nx, : self.backend.n_gates_max]

    def init_gates(self, V0_mV: Array1D) -> Array2D:
        return self._pad_gates(self.backend.init_gates(V0_mV[: self.nx]))

    def alpha(self, V_mV: Array1D) -> Array2D:
        return self._pad_gates(self.backend.alpha(V_mV[: self.nx]))

    def beta(self, V_mV: Array1D) -> Array2D:
        return self._pad_gates(self.backend.beta(V_mV[: self.nx]))

    def conductances(self, gates: Array2D) -> Array2D:
        return self._pad_channels(self.backend.conductances(self._local_gates(gates)))

    def cn_gate_update(self, g_prev: Array2D, V_mV: Array1D, dt: float) -> Array2D:
        return self._pad_gates(
            self.backend.cn_gate_update(
                g_prev=self._local_gates(g_prev),
                V_mV=V_mV[: self.nx],
                dt=dt,
            )
        )

    def currents(self, V_mV: Array1D, gates: Array2D) -> Array1D:
        return self._pad_space(
            self.backend.currents(
                V_mV=V_mV[: self.nx],
                gates=self._local_gates(gates),
            )
        )

    def total_conductance(self, gates: Array2D) -> Array1D:
        return self._pad_space(self.backend.total_conductance(self._local_gates(gates)))

    def membrane_conductance_terms(self, gates: Array2D) -> tuple[Array1D, Array1D]:
        Gm, GE = self.backend.membrane_conductance_terms(self._local_gates(gates))
        return self._pad_space(Gm), self._pad_space(GE)

    def background_current(self) -> Array1D:
        return self._pad_space(self.backend.background_current())


@dataclass(frozen=True)
class _AxNodePassiveFamilyICMBackend:
    """Row-parametric backend for MRG-like AxNode/passive compartment layouts."""

    node_model: IonChannelModelBase
    target_nx: int
    dtype: jnp.dtype
    n_gates_max: int = 7
    n_channels_max: int = 4

    @property
    def Nx(self) -> int:
        return self.target_nx

    def init_gates(self, V0_mV: Array1D) -> Array2D:
        node_gates = self.node_model.init_gates(V0_mV)
        out = jnp.zeros((self.Nx, self.n_gates_max), dtype=self.dtype)
        return out.at[:, : node_gates.shape[1]].set(node_gates)

    def cn_gate_update_for_row(
        self,
        row_index,
        *,
        g_prev: Array2D,
        V_mV: Array1D,
        dt: float,
    ) -> Array2D:
        _ = row_index
        node_gates = self.node_model.cn_gate_update(
            g_prev=g_prev[:, :4],
            V_mV=V_mV,
            dt=dt,
        )
        return jnp.concatenate([node_gates, g_prev[:, 4:]], axis=1)

    def currents_for_row(
        self,
        row_index,
        *,
        V_mV: Array1D,
        gates: Array2D,
    ) -> Array1D:
        _ = row_index
        node_mask = gates[:, 6]
        node_current = self.node_model.currents(V_mV=V_mV, gates=gates[:, :4])
        passive_current = gates[:, 4] * V_mV - gates[:, 5]
        return node_mask * node_current + (1.0 - node_mask) * passive_current

    def membrane_conductance_terms_for_row(
        self,
        row_index,
        gates: Array2D,
    ) -> tuple[Array1D, Array1D]:
        _ = row_index
        node_mask = gates[:, 6]
        node_gm, node_ge = self.node_model.membrane_conductance_terms(gates[:, :4])
        passive_gm = gates[:, 4]
        passive_ge = gates[:, 5]
        return (
            node_mask * node_gm + (1.0 - node_mask) * passive_gm,
            node_mask * node_ge + (1.0 - node_mask) * passive_ge,
        )

    def conductances(self, gates: Array2D) -> Array2D:
        node_mask = gates[:, 6:7]
        return node_mask * self.node_model.conductances(gates[:, :4])

    def cn_gate_update(self, g_prev: Array2D, V_mV: Array1D, dt: float) -> Array2D:
        return self.cn_gate_update_for_row(0, g_prev=g_prev, V_mV=V_mV, dt=dt)

    def currents(self, V_mV: Array1D, gates: Array2D) -> Array1D:
        return self.currents_for_row(0, V_mV=V_mV, gates=gates)

    def total_conductance(self, gates: Array2D) -> Array1D:
        gm, _ = self.membrane_conductance_terms(gates)
        return gm

    def membrane_conductance_terms(self, gates: Array2D) -> tuple[Array1D, Array1D]:
        return self.membrane_conductance_terms_for_row(0, gates)

    def background_current(self) -> Array1D:
        return jnp.zeros((self.Nx,), dtype=self.dtype)


@dataclass(frozen=True)
class RowIndexedICMBackend:
    """Static backend multiplexer for parameter-batched membrane rows."""

    rows: tuple[PaddedICMBackend, ...]
    n_gates_max: int
    n_channels_max: int
    dtype: jnp.dtype

    @classmethod
    def from_backends(
        cls,
        backends: tuple[ICMBackend, ...],
        *,
        target_nx: int,
    ) -> "RowIndexedICMBackend":
        if not backends:
            raise ValueError("backends must contain at least one row.")
        n_gates_max = max(int(backend.n_gates_max) for backend in backends)
        n_channels_max = max(int(backend.n_channels_max) for backend in backends)
        rows = tuple(
            PaddedICMBackend.from_backend(
                backend,
                target_nx=target_nx,
                n_gates_max=n_gates_max,
                n_channels_max=n_channels_max,
            )
            for backend in backends
        )
        return cls(
            rows=rows,
            n_gates_max=n_gates_max,
            n_channels_max=n_channels_max,
            dtype=backends[0].dtype,
        )

    @property
    def Nx(self) -> int:
        return self.rows[0].Nx

    def _switch(self, row_index, method_name: str, *args):
        branches = [
            (lambda *branch_args, row=row: getattr(row, method_name)(*branch_args))
            for row in self.rows
        ]
        return jax.lax.switch(row_index, branches, *args)

    def init_gates_for_row(self, row_index, V0_mV: Array1D) -> Array2D:
        return self._switch(row_index, "init_gates", V0_mV)

    def cn_gate_update_for_row(
        self,
        row_index,
        *,
        g_prev: Array2D,
        V_mV: Array1D,
        dt: float,
    ) -> Array2D:
        return self._switch(row_index, "cn_gate_update", g_prev, V_mV, dt)

    def currents_for_row(
        self,
        row_index,
        *,
        V_mV: Array1D,
        gates: Array2D,
    ) -> Array1D:
        return self._switch(row_index, "currents", V_mV, gates)

    def membrane_conductance_terms_for_row(
        self,
        row_index,
        gates: Array2D,
    ) -> tuple[Array1D, Array1D]:
        return self._switch(row_index, "membrane_conductance_terms", gates)
