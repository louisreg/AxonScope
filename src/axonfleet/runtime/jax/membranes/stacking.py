"""JAX-specific membrane row stacking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from axonfleet.benchmarking import benchmark_span
from axonfleet.dispatcher.plan import DispatchGroup, DispatchItem
from axonfleet.membranes.compiler import (
    lower_membrane_model_with_sources,
    membrane_source_path,
)
from axonfleet.membranes.model import ensure_membrane_model
from axonfleet.model_ir.source import load_generated_source_runtime
from axonfleet.preparation.membrane_rows import MembraneRowPlan
from axonfleet.runtime.jax.membranes.backend import (
    GatedLeakStackMembraneBackend,
    membrane_backend_model,
    membrane_static_signature,
)
from axonfleet.runtime.jax.membranes.compile import compile_membrane_model
from axonfleet.runtime.jax.membranes.generated_contract import (
    load_generated_membrane_contract,
)
from axonfleet.runtime.jax.membranes.program import JaxMembraneProgram


@dataclass(frozen=True)
class GatedLeakMembraneStack:
    backend: GatedLeakStackMembraneBackend
    gates0_rows: np.ndarray
    background_rows: np.ndarray
    membrane_static: Any
    parameter_rows: dict[str, np.ndarray]
    gated_count: int
    leak_count: int
    source: str
    unique_rows: int = 0
    cache_hits: int = 0
    host_leak_model_count: int = 0
    jax_compiled_model_count: int = 0


@dataclass(frozen=True)
class _GatedLeakMember:
    role: str
    model: Any | None = None
    gated_signature: tuple[Any, ...] | None = None
    parameter_values: dict[str, Any] | None = None
    leak_g: float = 0.0
    leak_ge: float = 0.0


@dataclass(frozen=True)
class _EncodedGatedLeakRow:
    gates: np.ndarray
    background: np.ndarray
    gated_model: Any
    gated_signature: tuple[Any, ...]
    parameter_values: dict[str, Any]
    gated_count: int
    leak_count: int


def try_stack_gated_leak_membrane_from_group(
    group: DispatchGroup,
    membrane_rows: MembraneRowPlan,
    *,
    target_nx: int,
    dtype_local: jnp.dtype,
    compiled_models_by_signature: dict[tuple[Any, ...], Any],
) -> GatedLeakMembraneStack | None:
    """Fast-path gated/leak rows from structural membrane capabilities."""

    if not group.items:
        return None
    np_dtype = np.dtype(dtype_local)
    gated_model: Any | None = None
    gated_execution_signature: tuple[Any, ...] | None = None
    gated_count = 0
    leak_count = 0
    encoded_rows: list[_EncodedGatedLeakRow] = []
    compiled_by_signature = dict(compiled_models_by_signature)
    seeded_compiled_count = len(compiled_by_signature)
    member_by_model_index: dict[int, _GatedLeakMember | None] = {}
    initial_gates_by_signature: dict[tuple[Any, ...], np.ndarray] = {}
    host_leak_signatures: set[tuple[Any, ...]] = set()

    with benchmark_span(
        "runtime.prepare.membrane_encode_unique_rows",
        unique_rows=membrane_rows.unique_count,
    ):
        for parameter_index, item_index in enumerate(
            membrane_rows.representative_item_indices
        ):
            item = group.items[int(item_index)]
            solver_axon = item.solver_axon
            row_nx = int(solver_axon.n_compartments)
            if row_nx > int(target_nx):
                return None
            encoded = _encode_gated_leak_group_row(
                item,
                target_nx=target_nx,
                dtype_local=dtype_local,
                np_dtype=np_dtype,
                compiled_by_signature=compiled_by_signature,
                model_signatures=membrane_rows.model_signatures,
                model_indices=membrane_rows.unique_row_model_indices[parameter_index],
                member_by_model_index=member_by_model_index,
                initial_gates_by_signature=initial_gates_by_signature,
                host_leak_signatures=host_leak_signatures,
                compatible_gated_model=gated_model,
            )
            if encoded is None:
                return None
            execution_signature = tuple(
                encoded.gated_model.execution_structure_signature()
            )
            if gated_execution_signature is None:
                gated_execution_signature = execution_signature
                gated_model = encoded.gated_model
            elif execution_signature != gated_execution_signature:
                return None
            encoded_rows.append(encoded)

    if gated_model is None or not encoded_rows:
        return None
    frequencies = np.bincount(
        membrane_rows.row_parameter_indices,
        minlength=membrane_rows.unique_count,
    )
    gated_count = sum(
        int(frequency) * int(encoded.gated_count)
        for frequency, encoded in zip(frequencies, encoded_rows, strict=True)
    )
    leak_count = sum(
        int(frequency) * int(encoded.leak_count)
        for frequency, encoded in zip(frequencies, encoded_rows, strict=True)
    )
    if gated_count == 0:
        return None
    with benchmark_span(
        "runtime.prepare.membrane_gather_rows",
        rows=membrane_rows.size,
        unique_rows=membrane_rows.unique_count,
    ):
        unique_gates = np.stack([encoded.gates for encoded in encoded_rows], axis=0)
        unique_background = np.stack(
            [encoded.background for encoded in encoded_rows],
            axis=0,
        )
        gates_rows = np.ascontiguousarray(
            unique_gates[membrane_rows.row_parameter_indices]
        )
        background_rows = np.ascontiguousarray(
            unique_background[membrane_rows.row_parameter_indices]
        )
    gated_gate_count = len(gated_model.gate_names())
    gated_channel_count = int(gated_model.g_bar.shape[0])
    parameter_names = tuple(sorted(encoded_rows[0].parameter_values))
    if any(
        tuple(sorted(encoded.parameter_values)) != parameter_names
        for encoded in encoded_rows
    ):
        return None
    unique_parameter_rows = {
        name: np.asarray(
            [encoded.parameter_values[name] for encoded in encoded_rows],
            dtype=np_dtype,
        )
        for name in parameter_names
    }
    parameter_rows = {
        name: np.ascontiguousarray(values[membrane_rows.row_parameter_indices])
        for name, values in unique_parameter_rows.items()
        if not np.array_equal(values, np.broadcast_to(values[:1], values.shape))
    }
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
        parameter_rows=parameter_rows,
        gated_count=gated_count,
        leak_count=leak_count,
        source="solver_axon_membrane_models",
        unique_rows=membrane_rows.unique_count,
        cache_hits=membrane_rows.cache_hits,
        host_leak_model_count=len(host_leak_signatures),
        jax_compiled_model_count=len(compiled_by_signature) - seeded_compiled_count,
    )


def _encode_gated_leak_group_row(
    item: DispatchItem,
    *,
    target_nx: int,
    dtype_local: jnp.dtype,
    np_dtype: np.dtype,
    compiled_by_signature: dict[tuple[Any, ...], Any],
    model_signatures: tuple[tuple[Any, ...], ...],
    model_indices: np.ndarray,
    member_by_model_index: dict[int, _GatedLeakMember | None],
    initial_gates_by_signature: dict[tuple[Any, ...], np.ndarray],
    host_leak_signatures: set[tuple[Any, ...]],
    compatible_gated_model: JaxMembraneProgram | None,
) -> _EncodedGatedLeakRow | None:
    solver_axon = item.solver_axon
    row_nx = int(solver_axon.n_compartments)
    membrane_models = tuple(solver_axon.membrane_models)
    if len(membrane_models) != row_nx:
        return None
    if int(model_indices.shape[0]) != row_nx:
        return None

    gated_model: Any | None = None
    gated_signature: tuple[Any, ...] | None = None
    gated_parameter_values: dict[str, Any] | None = None
    unique_model_indices, first_positions = np.unique(
        model_indices,
        return_index=True,
    )
    for model_index_value, first_position in zip(
        unique_model_indices,
        first_positions,
        strict=True,
    ):
        model_index = int(model_index_value)
        model = membrane_models[int(first_position)]
        if model_index in member_by_model_index:
            member = member_by_model_index[model_index]
        else:
            signature = model_signatures[model_index]
            compiled = compiled_by_signature.get(signature)
            if compiled is not None:
                executable = membrane_backend_model(compiled)
                if executable.gate_names():
                    member = _gated_leak_member(
                        executable,
                        dtype=np_dtype,
                        gated_signature=signature,
                    )
                else:
                    member = _try_host_stateless_leak_member(model, dtype=np_dtype)
                    if member is not None:
                        host_leak_signatures.add(signature)
                    else:
                        member = _gated_leak_member(executable, dtype=np_dtype)
            else:
                member = _try_host_stateless_leak_member(model, dtype=np_dtype)
                if member is not None:
                    host_leak_signatures.add(signature)
                elif compatible_gated_model is not None:
                    parameter_values = _compatible_generated_parameter_values(
                        compatible_gated_model,
                        model,
                    )
                    member = (
                        None
                        if parameter_values is None
                        else _GatedLeakMember(
                            role="gated",
                            model=compatible_gated_model,
                            gated_signature=signature,
                            parameter_values=parameter_values,
                        )
                    )
                if member is None:
                    compiled = compile_membrane_model(model)
                    compiled_by_signature[signature] = compiled
                    member = _gated_leak_member(
                        membrane_backend_model(compiled),
                        dtype=np_dtype,
                        gated_signature=signature,
                    )
            member_by_model_index[model_index] = member
        if member is None:
            return None
        if member.role == "gated":
            if member.model is None:
                return None
            executable_signature = member.gated_signature
            if executable_signature is None:
                return None
            if gated_signature is None:
                gated_signature = executable_signature
                gated_model = member.model
                gated_parameter_values = member.parameter_values
            elif executable_signature != gated_signature:
                return None

    if (
        gated_model is None
        or gated_signature is None
        or gated_parameter_values is None
    ):
        return None
    gated_gate_count = len(gated_model.gate_names())
    leak_g_col = int(gated_gate_count)
    leak_ge_col = int(gated_gate_count) + 1
    gated_mask_col = int(gated_gate_count) + 2
    encoded_width = int(gated_gate_count) + 3
    gates = np.zeros((int(target_nx), encoded_width), dtype=np_dtype)
    vm0 = float(getattr(item.simulation, "v_init", 0.0))
    initial_gate_key = (gated_signature, vm0, np_dtype.str)
    row_gated_gates = initial_gates_by_signature.get(initial_gate_key)
    if row_gated_gates is None:
        with benchmark_span("runtime.prepare.membrane_initial_gates"):
            if isinstance(gated_model, JaxMembraneProgram):
                row_gated_gates = gated_model.init_gates_host(
                    np.asarray([vm0], dtype=np_dtype),
                    dtype_local=np_dtype,
                    parameters=gated_parameter_values,
                )[0]
            else:
                row_gated_gates = np.asarray(
                    gated_model.init_gates(
                        jnp.asarray([vm0], dtype=dtype_local),
                    )[0],
                    dtype=np_dtype,
                )
        initial_gates_by_signature[initial_gate_key] = row_gated_gates

    model_count = len(model_signatures)
    gated_by_model = np.zeros((model_count,), dtype=bool)
    leak_g_by_model = np.zeros((model_count,), dtype=np_dtype)
    leak_ge_by_model = np.zeros((model_count,), dtype=np_dtype)
    for model_index_value in unique_model_indices:
        model_index = int(model_index_value)
        member = member_by_model_index[model_index]
        if member is None:
            return None
        gated_by_model[model_index] = member.role == "gated"
        leak_g_by_model[model_index] = member.leak_g
        leak_ge_by_model[model_index] = member.leak_ge
    gated_mask = gated_by_model[model_indices]
    gates[:row_nx, leak_g_col] = leak_g_by_model[model_indices]
    gates[:row_nx, leak_ge_col] = leak_ge_by_model[model_indices]
    gates[:row_nx, gated_mask_col] = gated_mask
    gated_indices = np.flatnonzero(gated_mask)
    gates[gated_indices, :gated_gate_count] = row_gated_gates
    gated_count = int(np.count_nonzero(gated_mask))
    leak_count = row_nx - gated_count

    return _EncodedGatedLeakRow(
        gates=gates,
        background=np.zeros((int(target_nx),), dtype=np_dtype),
        gated_model=gated_model,
        gated_signature=gated_signature,
        parameter_values=gated_parameter_values,
        gated_count=gated_count,
        leak_count=leak_count,
    )


def _try_host_stateless_leak_member(
    model: Any,
    *,
    dtype: np.dtype,
) -> _GatedLeakMember | None:
    """Encode a generic stateless one-current leak without building JAX state."""

    descriptor = ensure_membrane_model(model)
    with benchmark_span(
        "runtime.prepare.membrane_host_leak",
        membrane_kind=descriptor.kind,
    ):
        if descriptor.source_path is None:
            try:
                source_path = membrane_source_path(descriptor.kind)
            except ValueError:
                return None
        else:
            source_path = descriptor.source_path
        cached = load_generated_source_runtime(
            source_path,
            model_class_name=descriptor.source_class,
            targets=("numpy",),
        )
        if cached is None:
            try:
                lowered = lower_membrane_model_with_sources(
                    descriptor,
                    load_generated_modules=("numpy",),
                    generated_targets=("numpy",),
                )
            except (TypeError, ValueError):
                return None
            if len(lowered.source_results) != 1:
                return None
            module = lowered.source_results[0].cache.loaded_modules["numpy"]
        else:
            module = cached.cache.loaded_modules["numpy"]
        contract = load_generated_membrane_contract(module)
        if (
            contract.gate_state_names
            or contract.membrane_state_names
            or contract.has_step_program
            or contract.final_gate_update_mode == "post_solve_voltage"
            or len(contract.currents) != 1
        ):
            return None
        parameters = contract.parameter_values(descriptor.params)
        spec = contract.function("membrane_terms")
        env = {
            name: np.asarray(value, dtype=dtype)
            for name, value in parameters.items()
        }
        if any(name not in env for name in spec.args):
            return None
        try:
            raw = module.membrane_terms(*(env[name] for name in spec.args))
        except (KeyError, TypeError, ValueError):
            return None
        values = raw if isinstance(raw, tuple) else (raw,)
        if len(values) != 2:
            return None
        g_value = dtype.type(np.asarray(values[0], dtype=dtype).reshape(-1)[0])
        reversal = dtype.type(np.asarray(values[1], dtype=dtype).reshape(-1)[0])
        ge_value = dtype.type(g_value * reversal)
    if not np.isfinite(g_value) or not np.isfinite(ge_value):
        return None
    return _GatedLeakMember(
        role="leak",
        leak_g=float(g_value),
        leak_ge=float(ge_value),
    )


def _gated_leak_member(
    model: Any,
    *,
    dtype: np.dtype,
    gated_signature: tuple[Any, ...] | None = None,
) -> _GatedLeakMember | None:
    """Classify a compiled membrane by structural capabilities, not family name."""

    if model.membrane_state_specs():
        return None
    gate_count = len(model.gate_names())
    if gate_count > 0:
        if not model.supports_stateless_vm_only_fast_path():
            return None
        return _GatedLeakMember(
            role="gated",
            model=model,
            gated_signature=(
                membrane_static_signature(model)
                if gated_signature is None
                else gated_signature
            ),
            parameter_values=model.parameter_values,
        )
    if int(model.g_bar.shape[0]) != 1:
        return None
    g = dtype.type(np.asarray(model.g_bar, dtype=dtype)[0])
    e_rev = dtype.type(np.asarray(model.E_rev, dtype=dtype)[0])
    return _GatedLeakMember(
        role="leak",
        leak_g=float(g),
        leak_ge=float(g * e_rev),
    )


def _compatible_generated_parameter_values(
    program: JaxMembraneProgram,
    model: Any,
) -> dict[str, Any] | None:
    """Map one structurally compatible description onto generated parameters."""

    descriptor = ensure_membrane_model(model)
    expected = set(program.parameter_values)
    if descriptor.kind != "composite":
        values = {str(name): value for name, value in descriptor.params.items()}
        return values if set(values) == expected else None

    values: dict[str, Any] = {}
    for index, component in enumerate(descriptor.components):
        for name, value in component.params.items():
            prefixed = f"c{index}__{name}"
            runtime_name = prefixed if prefixed in expected else str(name)
            if runtime_name not in expected or runtime_name in values:
                return None
            values[runtime_name] = value
    return values if set(values) == expected else None
