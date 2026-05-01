from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp

from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.icm.backends import HeterogeneousICMBackend, ICMBackend
from axonscope.settings import dtype


def _unique_names(models: Sequence[IonChannelModelBase], method_name: str) -> tuple[str, ...]:
    names: list[str] = []
    for model in models:
        for name in getattr(model, method_name)():
            if name not in names:
                names.append(name)
    return tuple(names)


@dataclass(frozen=True)
class CompartmentMembraneLayout:
    """Membrane model assignment for a one-dimensional compartment layout."""

    models: tuple[IonChannelModelBase, ...]

    def __init__(self, models: Sequence[IonChannelModelBase]) -> None:
        frozen = tuple(models)
        if not frozen:
            raise ValueError("CompartmentMembraneLayout requires at least one membrane model.")
        object.__setattr__(self, "models", frozen)

    @property
    def Nx(self) -> int:
        return len(self.models)

    @property
    def dtype(self) -> jnp.dtype:
        return self.models[0].dtype

    def build_backend(self) -> ICMBackend:
        return HeterogeneousICMBackend.from_icm_vec(self.models)

    def as_membrane_model(self) -> "HeterogeneousMembraneModel":
        return HeterogeneousMembraneModel(self)


class HeterogeneousMembraneModel(IonChannelModelBase):
    """Generic membrane-model facade for heterogeneous compartment layouts."""

    def __init__(self, layout: CompartmentMembraneLayout) -> None:
        super().__init__()
        self.layout = layout
        self.models = layout.models
        self.backend = layout.build_backend()
        self.dtype = self.backend.dtype
        self.q10 = self.models[0].q10
        self._gate_names = _unique_names(self.models, "gate_names")
        self._conductance_names = _unique_names(self.models, "conductance_names")
        self._current_names = _unique_names(self.models, "current_names")

    def build_backend(self) -> ICMBackend:
        return self.backend

    def build_icm_backend(self) -> ICMBackend:
        return self.backend

    @property
    def g_bar(self) -> jnp.ndarray:
        return jnp.zeros((self.backend.n_channels_max,), dtype=self.dtype)

    @property
    def E_rev(self) -> jnp.ndarray:
        return jnp.zeros((self.backend.n_channels_max,), dtype=self.dtype)

    def gate_names(self) -> tuple[str, ...]:
        return self._gate_names

    def conductance_names(self) -> tuple[str, ...]:
        return self._conductance_names

    def current_names(self) -> tuple[str, ...]:
        return self._current_names

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        return self.backend.init_gates(V0_mV)

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        return self.backend.alpha(V)

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        return self.backend.beta(V)

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        _ = g_bar
        return self.conductances(gates)

    def conductances(self, gates: jnp.ndarray) -> jnp.ndarray:
        return self.backend.conductances(gates)

    def cn_gate_update(self, g_prev: jnp.ndarray, V_mV: jnp.ndarray, dt: float) -> jnp.ndarray:
        return self.backend.cn_gate_update(g_prev=g_prev, V_mV=V_mV, dt=dt)

    def final_gate_update(
        self,
        gates_prev: jnp.ndarray,
        V_mV_prev: jnp.ndarray,
        V_mV_new: jnp.ndarray,
        dt: float,
        gates_predictor: jnp.ndarray,
    ) -> jnp.ndarray:
        out = jnp.zeros_like(gates_predictor)
        for i, model in enumerate(self.models):
            n_g = self.backend.gate_sizes[i]
            if n_g == 0:
                continue
            local = model.final_gate_update(
                gates_prev=gates_prev[i : i + 1, :n_g],
                V_mV_prev=V_mV_prev[i : i + 1],
                V_mV_new=V_mV_new[i : i + 1],
                dt=dt,
                gates_predictor=gates_predictor[i : i + 1, :n_g],
            )[0]
            out = out.at[i, :n_g].set(local)
        return out

    def currents(self, V_mV: jnp.ndarray, gates: jnp.ndarray) -> jnp.ndarray:
        return self.backend.currents(V_mV=V_mV, gates=gates)

    def total_conductance(self, gates: jnp.ndarray) -> jnp.ndarray:
        return self.backend.total_conductance(gates)

    def membrane_conductance_terms(self, gates: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.backend.membrane_conductance_terms(gates)

    def I_background(self, Nx: int) -> jnp.ndarray:
        if Nx != self.backend.Nx:
            raise ValueError(f"Nx must be {self.backend.Nx}, got {Nx}.")
        return self.backend.background_current()

    def _empty_trace(self, n_cols: int) -> jnp.ndarray:
        return jnp.zeros((self.backend.Nx, n_cols), dtype=self.dtype)

    def gate_trace_matrix(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        _ = state
        out = self._empty_trace(len(self._gate_names))
        name_to_col = {name: i for i, name in enumerate(self._gate_names)}
        for i, model in enumerate(self.models):
            n_g = self.backend.gate_sizes[i]
            if n_g == 0:
                continue
            for local_idx, name in enumerate(model.gate_names()):
                out = out.at[i, name_to_col[name]].set(gates[i, local_idx])
        return out

    def conductance_trace_matrix(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        _ = state
        out = self._empty_trace(len(self._conductance_names))
        name_to_col = {name: i for i, name in enumerate(self._conductance_names)}
        for i, model in enumerate(self.models):
            n_g = self.backend.gate_sizes[i]
            local_gates = (
                gates[i : i + 1, :n_g]
                if n_g > 0
                else jnp.zeros((1, 0), dtype=self.dtype)
            )
            local_values = model.conductance_trace_matrix(local_gates)[0]
            for local_idx, name in enumerate(model.conductance_names()):
                out = out.at[i, name_to_col[name]].add(local_values[local_idx])
        return out

    def ionic_current_trace_matrix(
        self,
        V_mV: jnp.ndarray,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        _ = state
        out = self._empty_trace(len(self._current_names))
        name_to_col = {name: i for i, name in enumerate(self._current_names)}
        for i, model in enumerate(self.models):
            n_g = self.backend.gate_sizes[i]
            local_gates = (
                gates[i : i + 1, :n_g]
                if n_g > 0
                else jnp.zeros((1, 0), dtype=self.dtype)
            )
            local_values = model.ionic_current_trace_matrix(V_mV[i : i + 1], local_gates)[0]
            for local_idx, name in enumerate(model.current_names()):
                out = out.at[i, name_to_col[name]].add(local_values[local_idx])
        return out

    def membrane_state_trace_matrix(self, state: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        _ = state
        return jnp.zeros((self.backend.Nx, 0), dtype=dtype)
