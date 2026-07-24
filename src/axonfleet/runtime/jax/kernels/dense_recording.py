"""Shared dense membrane-recording helpers for JAX cable kernels."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from axonfleet.recording import RecordingPlan
from axonfleet.runtime.jax.cable_geometry import Array


class DenseRecordedTrace(NamedTuple):
    """One chunk of retained voltage and named membrane quantities."""

    Vm: Array
    recordings: dict[str, Array] | None


def recording_output_flags(plan: RecordingPlan | None) -> dict[str, bool]:
    """Return static scan flags for the membrane groups requested by ``plan``."""

    return {
        "gates": bool(plan is not None and plan.gates),
        "occupancies": bool(plan is not None and plan.markov_occupancies),
        "currents": bool(plan is not None and plan.currents),
        "conductances": bool(plan is not None and plan.conductances),
        "states": bool(plan is not None and plan.state_variables),
    }


def record_matrix(
    values: Array,
    record_indices: Array,
    *,
    record_full: bool,
) -> Array:
    """Retain all rows or the requested spatial rows of a quantity matrix."""

    if record_full:
        return values
    return jnp.take(values, record_indices, axis=0)


def record_matrix_batch(
    values: Array,
    record_indices: Array,
    *,
    record_full: bool,
) -> Array:
    """Retain spatial rows from a ``(batch, space, quantity)`` matrix."""

    if record_full:
        return values
    indices = jnp.asarray(record_indices, dtype=jnp.int32)
    if indices.ndim == 1:
        return jnp.take(values, indices, axis=1)
    if indices.ndim != 2:
        raise ValueError(
            "batch record_indices must have shape (width,) or (batch, width)."
        )
    expanded = jnp.broadcast_to(indices[..., None], (*indices.shape, values.shape[-1]))
    return jnp.take_along_axis(values, expanded, axis=1)


def empty_recording_matrix(vm: Array) -> Array:
    """Return a shape-stable empty quantity matrix for an unrequested group."""

    return jnp.zeros((vm.shape[0], 0), dtype=vm.dtype)


def empty_recording_matrix_batch(vm: Array) -> Array:
    """Return a shape-stable empty batch quantity matrix."""

    return jnp.zeros((*vm.shape, 0), dtype=vm.dtype)


def concat_recording_chunks(
    chunks: list[dict[str, Any]],
    *,
    concat: Any,
) -> dict[str, Any] | None:
    """Concatenate time chunks while preserving the fixed recording pytree."""

    if not chunks:
        return None
    return {
        name: concat([chunk[name] for chunk in chunks])
        for name in tuple(chunks[0])
    }


def recording_group(values: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """Expand the final quantity axis into stable public names."""

    return {
        name: values[..., index]
        for index, name in enumerate(names)
        if index < int(values.shape[-1])
    }


def recordings_for_plan(
    plan: RecordingPlan | None,
    trace: DenseRecordedTrace,
    *,
    observable_names: dict[str, tuple[str, ...]],
) -> dict[str, Any] | None:
    """Package one dense kernel trace according to the public recording plan."""

    if plan is None:
        return {"Vm": trace.Vm}
    recordings: dict[str, Any] = {}
    if plan.voltage:
        recordings["Vm"] = trace.Vm
    if trace.recordings is None:
        return recordings or None
    requested = {
        "gates": plan.gates,
        "occupancies": plan.markov_occupancies,
        "currents": plan.currents,
        "conductances": plan.conductances,
        "states": plan.state_variables,
    }
    for name, enabled in requested.items():
        if not enabled:
            continue
        group = recording_group(
            trace.recordings[name],
            observable_names.get(name, ()),
        )
        if group:
            recordings[name] = group
    return recordings or None
