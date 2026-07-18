"""Shared JAX batch-solver membrane operations."""

from __future__ import annotations

from typing import Any

import jax

from ..cable_geometry import Array
from .double_cable_linear import (
    double_cable_space_to_xb,
    solve_double_cable_physical_system_jax_triton_loop_xb,
)


def batch_gate_update(
    gates: Array,
    Vm: Array,
    *,
    backend: Any,
    row_indices: Array,
    dt_ms: float,
) -> Array:
    """Update membrane gates for one batched voltage state."""

    batch_update = getattr(backend, "batch_cn_gate_update", None)
    if callable(batch_update):
        return batch_update(g_prev=gates, V_mV=Vm, dt=dt_ms)
    row_gate_update = getattr(backend, "cn_gate_update_for_row", None)
    if callable(row_gate_update):
        return jax.vmap(
            lambda row_index, gates_row, vm_row: row_gate_update(
                row_index,
                g_prev=gates_row,
                V_mV=vm_row,
                dt=dt_ms,
            )
        )(row_indices, gates, Vm)
    return jax.vmap(
        lambda gates_row, vm_row: backend.cn_gate_update(
            g_prev=gates_row,
            V_mV=vm_row,
            dt=dt_ms,
        )
    )(gates, Vm)


def batch_currents(
    Vm: Array,
    gates: Array,
    *,
    backend: Any,
    row_indices: Array,
) -> Array:
    """Evaluate membrane currents for one batched voltage/gate state."""

    row_currents = getattr(backend, "currents_for_row", None)
    if callable(row_currents):
        return jax.vmap(
            lambda row_index, vm_row, gates_row: row_currents(
                row_index,
                V_mV=vm_row,
                gates=gates_row,
            )
        )(row_indices, Vm, gates)
    return jax.vmap(
        lambda vm_row, gates_row: backend.currents(V_mV=vm_row, gates=gates_row)
    )(Vm, gates)


def batch_membrane_conductance_terms(
    gates: Array,
    *,
    backend: Any,
    row_indices: Array,
    static_gates: Array | None = None,
) -> tuple[Array, Array]:
    """Evaluate membrane conductance terms for one batched gate state."""

    batch_terms = getattr(backend, "batch_membrane_conductance_terms", None)
    if callable(batch_terms):
        if static_gates is not None:
            return batch_terms(gates, static_gates=static_gates)
        return batch_terms(gates)
    row_terms = getattr(backend, "membrane_conductance_terms_for_row", None)
    if callable(row_terms):
        return jax.vmap(
            lambda row_index, gates_row: row_terms(row_index, gates_row)
        )(row_indices, gates)
    return jax.vmap(backend.membrane_conductance_terms)(gates)


def batch_prepare_membrane_step(
    Vm: Array,
    gates_prev: Array,
    gates_new: Array,
    state: tuple[Array, ...],
    I_ion: Array,
    I_background_row: Array,
    *,
    membrane: Any,
    dt_ms: float,
) -> Any:
    """Build row-wise membrane step plans for a batched state."""

    state_axes = tuple(0 for _ in state)
    return jax.vmap(
        lambda vm_row, gates_prev_row, gates_new_row, state_row, iion_row, ibg_row: (
            membrane.prepare_membrane_step(
                V_mV=vm_row,
                gates_prev=gates_prev_row,
                gates_new=gates_new_row,
                state=state_row,
                dt=dt_ms,
                I_ion=iion_row,
                I_background=ibg_row,
            )
        ),
        in_axes=(0, 0, 0, state_axes, 0, 0),
    )(Vm, gates_prev, gates_new, state, I_ion, I_background_row)


def batch_final_gate_update(
    gates_prev: Array,
    Vm_prev: Array,
    Vm_new: Array,
    gates_predictor: Array,
    *,
    membrane: Any,
    dt_ms: float,
) -> Array:
    """Finalize membrane gates after solving one batched voltage step."""

    return jax.vmap(
        lambda gates_prev_row, vm_prev_row, vm_new_row, gates_predictor_row: (
            membrane.final_gate_update(
                gates_prev=gates_prev_row,
                V_mV_prev=vm_prev_row,
                V_mV_new=vm_new_row,
                dt=dt_ms,
                gates_predictor=gates_predictor_row,
            )
        )
    )(gates_prev, Vm_prev, Vm_new, gates_predictor)


def batch_finalize_membrane_step(
    Vm_prev: Array,
    Vm_new: Array,
    gates_prev: Array,
    gates_new: Array,
    state_prev: tuple[Array, ...],
    step_plan: Any,
    *,
    membrane: Any,
    dt_ms: float,
) -> tuple[Array, ...]:
    """Finalize row-wise membrane auxiliary state after one batched step."""

    state_axes = tuple(0 for _ in state_prev)
    return jax.vmap(
        lambda vm_prev_row, vm_new_row, gates_prev_row, gates_new_row, state_row, step_plan_row: (
            membrane.finalize_membrane_step(
                V_mV_prev=vm_prev_row,
                V_mV_new=vm_new_row,
                gates_prev=gates_prev_row,
                gates_new=gates_new_row,
                state_prev=state_row,
                step_plan=step_plan_row,
                dt=dt_ms,
            )
        ),
        in_axes=(0, 0, 0, 0, state_axes, 0),
    )(Vm_prev, Vm_new, gates_prev, gates_new, state_prev, step_plan)


def solve_double_cable_batch_step(
    Vi: Array,
    Ve: Array,
    gates_new: Array,
    Iinj_abs: Array,
    I_outward_abs: Array,
    I_corr_abs: Array,
    extracellular_drive_abs: Array,
    *,
    backend: Any,
    row_indices: Array,
    linear_static_xb: Any | None,
    batch_size: int,
    nx: int,
    double_cable_block_solver: str,
    tiled_thomas_block_b: int,
    return_node_first: bool = False,
    static_gates: Array | None = None,
    membrane_terms: tuple[Array, Array] | None = None,
) -> tuple[Array, Array]:
    """Solve one batched double-cable implicit step."""

    if membrane_terms is None:
        Gm_den, GE_den = batch_membrane_conductance_terms(
            gates_new,
            backend=backend,
            row_indices=row_indices,
            static_gates=static_gates,
        )
    else:
        Gm_den, GE_den = membrane_terms
    if double_cable_block_solver == "jax_triton_loop_xb":
        if linear_static_xb is None:
            raise ValueError("linear_static_xb is required for jax_triton_loop_xb.")
        return solve_double_cable_physical_system_jax_triton_loop_xb(
            static=linear_static_xb,
            Vi=double_cable_space_to_xb(Vi, batch_size=batch_size, nx=nx),
            Ve=double_cable_space_to_xb(Ve, batch_size=batch_size, nx=nx),
            Gm_density=double_cable_space_to_xb(
                Gm_den,
                batch_size=batch_size,
                nx=nx,
            ),
            GE_density=double_cable_space_to_xb(
                GE_den,
                batch_size=batch_size,
                nx=nx,
            ),
            Iinj_abs=double_cable_space_to_xb(
                Iinj_abs,
                batch_size=batch_size,
                nx=nx,
            ),
            I_outward_abs=double_cable_space_to_xb(
                I_outward_abs,
                batch_size=batch_size,
                nx=nx,
            ),
            I_corr_abs=double_cable_space_to_xb(
                I_corr_abs,
                batch_size=batch_size,
                nx=nx,
            ),
            extracellular_drive_abs=double_cable_space_to_xb(
                extracellular_drive_abs,
                batch_size=batch_size,
                nx=nx,
            ),
            block_b=tiled_thomas_block_b,
            return_node_first=return_node_first,
        )

    raise ValueError(
        f"Unsupported batch-native double-cable block solver: {double_cable_block_solver!r}"
    )


__all__ = [
    "batch_currents",
    "batch_final_gate_update",
    "batch_finalize_membrane_step",
    "batch_gate_update",
    "batch_membrane_conductance_terms",
    "batch_prepare_membrane_step",
    "solve_double_cable_batch_step",
]
