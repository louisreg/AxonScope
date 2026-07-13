"""JAX-specific membrane row stacking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from axonscope.dispatcher.plan import DispatchGroup, DispatchItem
from axonscope.model_ir.interpreter import NumpyModelInterpreter
from axonscope.runtime.jax.membranes.backend import (
    GatedLeakStackMembraneBackend,
    HeterogeneousMembraneBackend,
    membrane_backend_model,
    membrane_static_signature,
)
from axonscope.runtime.jax.membranes.compile import compile_membrane_model
from axonscope.runtime.jax.membranes.program import JaxMembraneProgram
from axonscope.runtime.jax.types import MembraneRuntime
from axonscope.solvers.options import SolverOptions


@dataclass(frozen=True)
class GatedLeakMembraneStack:
    backend: GatedLeakStackMembraneBackend
    gates0_rows: np.ndarray
    background_rows: np.ndarray
    membrane_static: Any
    gated_count: int
    leak_count: int
    source: str
    unique_rows: int = 0
    cache_hits: int = 0


@dataclass(frozen=True)
class _GatedLeakMember:
    role: str
    model: Any | None = None
    leak_g: float = 0.0
    leak_ge: float = 0.0


@dataclass(frozen=True)
class _EncodedGatedLeakRow:
    gates: np.ndarray
    background: np.ndarray
    gated_model: Any
    gated_signature: tuple[Any, ...]
    gated_count: int
    leak_count: int


def try_stack_gated_leak_membrane_from_group(
    group: DispatchGroup,
    *,
    target_nx: int,
    dtype_local: jnp.dtype,
    solver_options: SolverOptions | None,
) -> GatedLeakMembraneStack | None:
    """Fast-path gated/leak rows from structural membrane capabilities."""

    if not group.items:
        return None
    np_dtype = np.dtype(dtype_local)
    gated_model: Any | None = None
    gated_signature: tuple[Any, ...] | None = None
    gated_count = 0
    leak_count = 0
    gates_rows: list[np.ndarray] = []
    background_rows: list[np.ndarray] = []
    compiled_by_signature: dict[tuple[Any, ...], Any] = {}
    encoded_row_cache: dict[tuple[Any, ...], _EncodedGatedLeakRow] = {}
    row_cache_hits = 0

    for item in group.items:
        solver_axon = item.solver_axon
        row_nx = int(solver_axon.n_compartments)
        if row_nx > int(target_nx):
            return None
        row_key = _gated_leak_group_row_cache_key(
            item,
            target_nx=target_nx,
            np_dtype=np_dtype,
            solver_options=solver_options,
        )
        encoded = encoded_row_cache.get(row_key)
        if encoded is None:
            encoded = _encode_gated_leak_group_row(
                item,
                target_nx=target_nx,
                dtype_local=dtype_local,
                np_dtype=np_dtype,
                solver_options=solver_options,
                compiled_by_signature=compiled_by_signature,
            )
            if encoded is None:
                return None
            encoded_row_cache[row_key] = encoded
        else:
            row_cache_hits += 1
        if gated_signature is None:
            gated_signature = encoded.gated_signature
            gated_model = encoded.gated_model
        elif encoded.gated_signature != gated_signature:
            return None
        gated_count += int(encoded.gated_count)
        leak_count += int(encoded.leak_count)
        gates_rows.append(encoded.gates)
        background_rows.append(encoded.background)

    if gated_model is None or gated_count == 0 or leak_count == 0:
        return None
    gated_gate_count = len(gated_model.gate_names())
    gated_channel_count = int(gated_model.g_bar.shape[0])
    backend = GatedLeakStackMembraneBackend(
        gated_model=gated_model,
        target_nx=int(target_nx),
        dtype=dtype_local,
        gated_gate_count=gated_gate_count,
        gated_channel_count=gated_channel_count,
    )
    return GatedLeakMembraneStack(
        backend=backend,
        gates0_rows=np.stack(gates_rows, axis=0),
        background_rows=np.stack(background_rows, axis=0),
        membrane_static=gated_model,
        gated_count=gated_count,
        leak_count=leak_count,
        source="solver_axon_membrane_models",
        unique_rows=len(encoded_row_cache),
        cache_hits=row_cache_hits,
    )


def try_stack_gated_leak_membrane_from_runtime_rows(
    rows: tuple[MembraneRuntime, ...],
    *,
    target_nx: int,
    dtype_local: jnp.dtype,
) -> GatedLeakMembraneStack | None:
    """Encode gated/leak row layouts as dynamic row parameters."""

    if not rows:
        return None
    np_dtype = np.dtype(dtype_local)
    gated_model: Any | None = None
    gated_signature: tuple[Any, ...] | None = None
    gated_count = 0
    leak_count = 0
    members_by_row: list[list[_GatedLeakMember]] = []
    gates_by_row: list[np.ndarray] = []
    for row in rows:
        backend = row.backend
        if not isinstance(backend, HeterogeneousMembraneBackend):
            return None
        row_nx = int(backend.Nx)
        if row_nx > int(target_nx):
            return None
        row_background = np.asarray(row.background_current, dtype=np_dtype)
        if row_background.shape != (row_nx,) or not np.allclose(row_background, 0.0):
            return None
        row_gates = np.asarray(row.gates0, dtype=np_dtype)
        if row_gates.shape[0] != row_nx:
            return None
        row_members: list[_GatedLeakMember] = []
        for model in backend.membrane_models:
            member = _gated_leak_member(model, dtype=np_dtype)
            if member is None:
                return None
            if member.role == "gated":
                signature = membrane_static_signature(model)
                if gated_signature is None:
                    gated_signature = signature
                    gated_model = model
                elif signature != gated_signature:
                    return None
                gated_count += 1
            else:
                leak_count += 1
            row_members.append(member)
        members_by_row.append(row_members)
        gates_by_row.append(row_gates)
    if gated_model is None or gated_count == 0 or leak_count == 0:
        return None
    gated_gate_count = len(gated_model.gate_names())
    gated_channel_count = int(gated_model.g_bar.shape[0])
    encoded = _encode_gated_leak_runtime_members(
        members_by_row,
        gates_by_row,
        gated_gate_count=gated_gate_count,
        target_nx=target_nx,
        np_dtype=np_dtype,
    )
    if encoded is None:
        return None
    gates_rows, background_rows = encoded
    backend = GatedLeakStackMembraneBackend(
        gated_model=gated_model,
        target_nx=int(target_nx),
        dtype=dtype_local,
        gated_gate_count=gated_gate_count,
        gated_channel_count=gated_channel_count,
    )
    return GatedLeakMembraneStack(
        backend=backend,
        gates0_rows=gates_rows,
        background_rows=background_rows,
        membrane_static=gated_model,
        gated_count=gated_count,
        leak_count=leak_count,
        source="row_membrane_runtime",
    )


def _gated_leak_group_row_cache_key(
    item: DispatchItem,
    *,
    target_nx: int,
    np_dtype: np.dtype,
    solver_options: SolverOptions | None,
) -> tuple[Any, ...]:
    return (
        "gated_leak_group_row_v1",
        item.membrane_signature,
        int(item.solver_axon.n_compartments),
        int(target_nx),
        float(getattr(item.simulation, "v_init", 0.0)),
        np_dtype.str,
        repr(solver_options),
    )


def _encode_gated_leak_group_row(
    item: DispatchItem,
    *,
    target_nx: int,
    dtype_local: jnp.dtype,
    np_dtype: np.dtype,
    solver_options: SolverOptions | None,
    compiled_by_signature: dict[tuple[Any, ...], Any],
) -> _EncodedGatedLeakRow | None:
    solver_axon = item.solver_axon
    row_nx = int(solver_axon.n_compartments)
    membrane_models = tuple(solver_axon.membrane_models)
    if len(membrane_models) != row_nx:
        return None

    gated_model: Any | None = None
    gated_signature: tuple[Any, ...] | None = None
    gated_count = 0
    leak_count = 0
    members: list[_GatedLeakMember] = []
    for model in membrane_models:
        signature = membrane_static_signature(model)
        compiled = compiled_by_signature.get(signature)
        if compiled is None:
            compiled = compile_membrane_model(model, solver_options=solver_options)
            compiled_by_signature[signature] = compiled
        executable = membrane_backend_model(compiled)
        member = _gated_leak_member(executable, dtype=np_dtype)
        if member is None:
            return None
        if member.role == "gated":
            executable_signature = membrane_static_signature(executable)
            if gated_signature is None:
                gated_signature = executable_signature
                gated_model = executable
            elif executable_signature != gated_signature:
                return None
            gated_count += 1
        else:
            leak_count += 1
        members.append(member)

    if gated_model is None or gated_signature is None:
        return None
    gated_gate_count = len(gated_model.gate_names())
    leak_g_col = int(gated_gate_count)
    leak_ge_col = int(gated_gate_count) + 1
    gated_mask_col = int(gated_gate_count) + 2
    encoded_width = int(gated_gate_count) + 3
    gates = np.zeros((int(target_nx), encoded_width), dtype=np_dtype)
    vm0 = float(getattr(item.simulation, "v_init", 0.0))
    if isinstance(gated_model, JaxMembraneProgram):
        row_gated_gates = NumpyModelInterpreter(
            gated_model.model_ir,
            dtype=np_dtype,
        ).init_gates(np.asarray([vm0], dtype=np_dtype))[0]
    else:
        row_gated_gates = np.asarray(
            gated_model.init_gates(
                jnp.asarray([vm0], dtype=dtype_local),
            )[0],
            dtype=np_dtype,
        )
    for compartment_index, member in enumerate(members):
        if member.role == "gated":
            gates[compartment_index, :gated_gate_count] = row_gated_gates
            gates[compartment_index, gated_mask_col] = np_dtype.type(1.0)
        else:
            gates[compartment_index, leak_g_col] = np_dtype.type(member.leak_g)
            gates[compartment_index, leak_ge_col] = np_dtype.type(member.leak_ge)

    return _EncodedGatedLeakRow(
        gates=gates,
        background=np.zeros((int(target_nx),), dtype=np_dtype),
        gated_model=gated_model,
        gated_signature=gated_signature,
        gated_count=gated_count,
        leak_count=leak_count,
    )


def _gated_leak_member(model: Any, *, dtype: np.dtype) -> _GatedLeakMember | None:
    """Classify a compiled membrane by structural capabilities, not family name."""

    if model.membrane_state_specs():
        return None
    gate_count = len(model.gate_names())
    if gate_count > 0:
        if not model.supports_stateless_vm_only_fast_path():
            return None
        return _GatedLeakMember(role="gated", model=model)
    if int(model.g_bar.shape[0]) != 1:
        return None
    g = dtype.type(np.asarray(model.g_bar, dtype=dtype)[0])
    e_rev = dtype.type(np.asarray(model.E_rev, dtype=dtype)[0])
    return _GatedLeakMember(
        role="leak",
        leak_g=float(g),
        leak_ge=float(g * e_rev),
    )


def _encode_gated_leak_runtime_members(
    members_by_row: list[list[_GatedLeakMember]],
    gates_by_row: list[np.ndarray],
    *,
    gated_gate_count: int,
    target_nx: int,
    np_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray] | None:
    leak_g_col = int(gated_gate_count)
    leak_ge_col = int(gated_gate_count) + 1
    gated_mask_col = int(gated_gate_count) + 2
    encoded_width = int(gated_gate_count) + 3
    gates_rows: list[np.ndarray] = []
    background_rows: list[np.ndarray] = []
    for row_members, row_gates in zip(members_by_row, gates_by_row, strict=True):
        if int(row_gates.shape[1]) < int(gated_gate_count):
            return None
        encoded = np.zeros((int(target_nx), encoded_width), dtype=np_dtype)
        for compartment_index, member in enumerate(row_members):
            if member.role == "gated":
                encoded[compartment_index, :gated_gate_count] = row_gates[
                    compartment_index,
                    :gated_gate_count,
                ]
                encoded[compartment_index, gated_mask_col] = np_dtype.type(1.0)
            else:
                encoded[compartment_index, leak_g_col] = np_dtype.type(member.leak_g)
                encoded[compartment_index, leak_ge_col] = np_dtype.type(member.leak_ge)
        gates_rows.append(encoded)
        background_rows.append(np.zeros((int(target_nx),), dtype=np_dtype))
    return np.stack(gates_rows, axis=0), np.stack(background_rows, axis=0)


__all__ = [
    "GatedLeakMembraneStack",
    "try_stack_gated_leak_membrane_from_group",
    "try_stack_gated_leak_membrane_from_runtime_rows",
]
