from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def membrane_observable_names(membrane: Any) -> dict[str, tuple[str, ...]]:
    return {
        "gates": membrane.gate_names(),
        "currents": membrane.current_names(),
        "conductances": membrane.conductance_names(),
        "states": membrane.membrane_state_names(),
    }


def observable_matrices(
    membrane: Any,
    V_mV: jnp.ndarray,
    gates: jnp.ndarray,
    state: tuple[jnp.ndarray, ...],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    Nx = V_mV.shape[0]
    gate_obs = membrane.gate_trace_matrix(gates, state)
    current_obs = membrane.ionic_current_trace_matrix(V_mV, gates, state)
    conductance_obs = membrane.conductance_trace_matrix(gates, state)
    if membrane.membrane_state_names():
        state_obs = membrane.membrane_state_trace_matrix(state)
    else:
        state_obs = jnp.zeros((Nx, 0), dtype=V_mV.dtype)
    return gate_obs, current_obs, conductance_obs, state_obs


def package_recordings(
    names: dict[str, tuple[str, ...]],
    gate_obs: jnp.ndarray,
    current_obs: jnp.ndarray,
    conductance_obs: jnp.ndarray,
    state_obs: jnp.ndarray,
) -> dict[str, dict[str, jnp.ndarray]]:
    recordings: dict[str, dict[str, jnp.ndarray]] = {}
    packed = {
        "gates": gate_obs,
        "currents": current_obs,
        "conductances": conductance_obs,
        "states": state_obs,
    }
    for group_name, group_names in names.items():
        if not group_names:
            continue
        values = packed[group_name]
        group_recordings: dict[str, jnp.ndarray] = {}
        sum_duplicates = group_name in {"currents", "conductances"}
        for i, name in enumerate(group_names):
            column = values[:, :, i]
            if sum_duplicates and name in group_recordings:
                group_recordings[name] = group_recordings[name] + column
            else:
                group_recordings[name] = column
        recordings[group_name] = group_recordings
    return recordings
