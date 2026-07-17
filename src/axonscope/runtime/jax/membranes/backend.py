from __future__ import annotations
from dataclasses import dataclass
from dataclasses import fields, is_dataclass
from typing import Any, ClassVar, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------- Type aliases ----------------------
Array1D = jnp.ndarray  # shape (N,)
Array2D = jnp.ndarray  # shape (N, n_gates)


class MembraneStepPlan(NamedTuple):
    state: tuple[jnp.ndarray, ...]
    linearization_gates: jnp.ndarray
    total_outward_current: jnp.ndarray
    explicit_outward_current: jnp.ndarray
    correction_current: jnp.ndarray


class MembraneStateSpec(NamedTuple):
    name: str


def _static_value_signature(value: Any) -> Any:
    """Return a hashable structural signature for runtime membrane values."""

    static_signature = getattr(value, "static_signature", None)
    if callable(static_signature):
        return static_signature()
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


def membrane_backend_model(model: Any) -> Any:
    """Return the executable model consumed by membrane backends."""

    return getattr(model, "runtime", model)


def membrane_static_signature(model: Any) -> tuple[Any, ...]:
    """Return a static signature for descriptors, layouts, and programs."""

    executable = membrane_backend_model(model)
    static_signature = getattr(executable, "static_signature", None)
    if callable(static_signature):
        return tuple(static_signature())
    descriptor_signature = getattr(executable, "_static_signature", None)
    if callable(descriptor_signature):
        return tuple(descriptor_signature())
    raise TypeError(
        f"Membrane executable {type(executable).__name__} has no static signature."
    )


class MembraneBackend(Protocol):
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


@dataclass(frozen=True)
class UniformMembraneBackend:
    """Backend for a spatially uniform compiled membrane model."""

    ion_channel: Any
    nx: int
    n_gates_max: int
    n_channels_max: int
    dtype: jnp.dtype

    @classmethod
    def from_model(cls, ion_channel: Any, nx: int) -> "UniformMembraneBackend":
        if nx < 1:
            raise ValueError(f"nx must be >= 1, got {nx}.")
        executable = membrane_backend_model(ion_channel)
        n_gates = len(executable.gate_names())
        n_channels = int(executable.g_bar.shape[0])
        return cls(
            ion_channel=executable,
            nx=int(nx),
            n_gates_max=n_gates,
            n_channels_max=n_channels,
            dtype=executable.dtype,
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


class HeterogeneousMembraneGroup(NamedTuple):
    model: Any
    indices: tuple[int, ...]
    gate_size: int
    channel_size: int


@dataclass(frozen=True)
class HeterogeneousMembraneBackend:
    """Backend for one compiled membrane model instance per compartment."""

    membrane_models: tuple[Any, ...]
    groups: tuple[HeterogeneousMembraneGroup, ...]
    gate_sizes: tuple[int, ...]
    channel_sizes: tuple[int, ...]
    n_gates_max: int
    n_channels_max: int
    dtype: jnp.dtype

    @classmethod
    def from_models(cls, membrane_models: tuple[Any, ...] | list[Any]) -> "HeterogeneousMembraneBackend":
        if len(membrane_models) == 0:
            raise ValueError("membrane_models must contain at least one membrane model.")

        frozen = tuple(membrane_backend_model(model) for model in membrane_models)
        grouped: list[tuple[Any, list[int], int, int]] = []
        group_index_by_signature: dict[tuple[Any, ...], int] = {}
        signature_by_identity: dict[int, tuple[Any, ...]] = {}
        sizes_by_signature: dict[tuple[Any, ...], tuple[int, int]] = {}
        gate_sizes_list: list[int] = []
        channel_sizes_list: list[int] = []
        for i, model in enumerate(frozen):
            identity = id(model)
            signature = signature_by_identity.get(identity)
            if signature is None:
                signature = membrane_static_signature(model)
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
            HeterogeneousMembraneGroup(
                model=model,
                indices=tuple(indices),
                gate_size=gate_size,
                channel_size=channel_size,
            )
            for model, indices, gate_size, channel_size in grouped
        )

        return cls(
            membrane_models=frozen,
            groups=groups,
            gate_sizes=gate_sizes,
            channel_sizes=channel_sizes,
            n_gates_max=n_gates_max,
            n_channels_max=n_channels_max,
            dtype=frozen[0].dtype,
        )

    @property
    def Nx(self) -> int:
        return len(self.membrane_models)

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
class PaddedMembraneBackend:
    """Adapter exposing a shorter backend at a padded spatial width."""

    backend: MembraneBackend
    nx: int
    target_nx: int
    n_gates_max: int
    n_channels_max: int
    dtype: jnp.dtype

    @classmethod
    def from_backend(
        cls,
        backend: MembraneBackend,
        *,
        target_nx: int,
        n_gates_max: int,
        n_channels_max: int,
    ) -> "PaddedMembraneBackend":
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
class GatedLeakStackMembraneBackend:
    """Row-parametric backend for structural gated/leak compartment layouts."""

    gated_model: Any
    target_nx: int
    dtype: jnp.dtype
    gated_gate_count: int
    gated_channel_count: int

    supports_node_first_batch: ClassVar[bool] = True

    @property
    def Nx(self) -> int:
        return self.target_nx

    @property
    def n_gates_max(self) -> int:
        return self.gated_gate_count + 3

    @property
    def n_channels_max(self) -> int:
        return self.gated_channel_count

    @property
    def _leak_g_col(self) -> int:
        return self.gated_gate_count

    @property
    def _leak_ge_col(self) -> int:
        return self.gated_gate_count + 1

    @property
    def _gated_mask_col(self) -> int:
        return self.gated_gate_count + 2

    def _gated_gates(self, gates: Array2D) -> Array2D:
        return gates[:, : self.gated_gate_count]

    def init_gates(self, V0_mV: Array1D) -> Array2D:
        gated_gates = self.gated_model.init_gates(V0_mV)
        out = jnp.zeros((self.Nx, self.n_gates_max), dtype=self.dtype)
        return out.at[:, : self.gated_gate_count].set(gated_gates)

    def cn_gate_update_for_row(
        self,
        row_index,
        *,
        g_prev: Array2D,
        V_mV: Array1D,
        dt: float,
    ) -> Array2D:
        _ = row_index
        gated_gates = self.gated_model.cn_gate_update(
            g_prev=self._gated_gates(g_prev),
            V_mV=V_mV,
            dt=dt,
        )
        return jnp.concatenate([gated_gates, g_prev[:, self.gated_gate_count :]], axis=1)

    def batch_cn_gate_update(
        self,
        *,
        g_prev: jnp.ndarray,
        V_mV: jnp.ndarray,
        dt: float,
    ) -> jnp.ndarray:
        """Update every compatible row through one flattened membrane call."""

        batch_shape = g_prev.shape[:-1]
        gated_gates = self.gated_model.cn_gate_update(
            g_prev=g_prev[..., : self.gated_gate_count].reshape(
                (-1, self.gated_gate_count)
            ),
            V_mV=V_mV.reshape((-1,)),
            dt=dt,
        ).reshape((*batch_shape, self.gated_gate_count))
        return jnp.concatenate(
            [gated_gates, g_prev[..., self.gated_gate_count :]],
            axis=-1,
        )

    def currents_for_row(
        self,
        row_index,
        *,
        V_mV: Array1D,
        gates: Array2D,
    ) -> Array1D:
        _ = row_index
        gated_mask = gates[:, self._gated_mask_col]
        gated_current = self.gated_model.currents(
            V_mV=V_mV,
            gates=self._gated_gates(gates),
        )
        leak_current = gates[:, self._leak_g_col] * V_mV - gates[:, self._leak_ge_col]
        return gated_mask * gated_current + (1.0 - gated_mask) * leak_current

    def membrane_conductance_terms_for_row(
        self,
        row_index,
        gates: Array2D,
    ) -> tuple[Array1D, Array1D]:
        _ = row_index
        gated_mask = gates[:, self._gated_mask_col]
        gated_gm, gated_ge = self.gated_model.membrane_conductance_terms(
            self._gated_gates(gates)
        )
        leak_gm = gates[:, self._leak_g_col]
        leak_ge = gates[:, self._leak_ge_col]
        return (
            gated_mask * gated_gm + (1.0 - gated_mask) * leak_gm,
            gated_mask * gated_ge + (1.0 - gated_mask) * leak_ge,
        )

    def batch_membrane_conductance_terms(
        self,
        gates: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate every compatible row through one flattened membrane call."""

        batch_shape = gates.shape[:-1]
        gated_gm, gated_ge = self.gated_model.membrane_conductance_terms(
            gates[..., : self.gated_gate_count].reshape(
                (-1, self.gated_gate_count)
            )
        )
        gated_gm = gated_gm.reshape(batch_shape)
        gated_ge = gated_ge.reshape(batch_shape)
        gated_mask = gates[..., self._gated_mask_col]
        leak_gm = gates[..., self._leak_g_col]
        leak_ge = gates[..., self._leak_ge_col]
        return (
            gated_mask * gated_gm + (1.0 - gated_mask) * leak_gm,
            gated_mask * gated_ge + (1.0 - gated_mask) * leak_ge,
        )

    def conductances(self, gates: Array2D) -> Array2D:
        gated_mask = gates[:, self._gated_mask_col : self._gated_mask_col + 1]
        return gated_mask * self.gated_model.conductances(self._gated_gates(gates))

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
class RowIndexedMembraneBackend:
    """Static backend multiplexer for parameter-batched membrane rows."""

    rows: tuple[PaddedMembraneBackend, ...]
    n_gates_max: int
    n_channels_max: int
    dtype: jnp.dtype

    @classmethod
    def from_backends(
        cls,
        backends: tuple[MembraneBackend, ...],
        *,
        target_nx: int,
    ) -> "RowIndexedMembraneBackend":
        if not backends:
            raise ValueError("backends must contain at least one row.")
        n_gates_max = max(int(backend.n_gates_max) for backend in backends)
        n_channels_max = max(int(backend.n_channels_max) for backend in backends)
        rows = tuple(
            PaddedMembraneBackend.from_backend(
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
