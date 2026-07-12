"""Shared factorized-input helpers for JAX batch kernels."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from axonscope.benchmarking import benchmark_span
from axonscope.runtime.jax.inputs.payloads import FactorizedExtracellularPotentialBatch
from axonscope.runtime.jax.cable_geometry import Array
from axonscope.runtime.jax.preparation.caches import (
    get_single_cable_factorized_forcing,
    store_single_cable_factorized_forcing,
)


def _factorized_current_mid_rows(
    batch: FactorizedExtracellularPotentialBatch,
    *,
    dtype_local: Any,
    batch_size: int,
) -> Array:
    current = jnp.asarray(batch.current_mid_A, dtype=dtype_local)
    row_scales = (
        None
        if batch.current_row_scales is None
        else jnp.asarray(batch.current_row_scales, dtype=dtype_local)
    )
    footprint = jnp.asarray(batch.footprint_mV_per_A)
    if row_scales is not None:
        if current.ndim == 1:
            scales = row_scales.reshape((batch_size,))
            return current[None, None, :] * scales[:, None, None]
        if current.ndim == 2 and footprint.ndim == 3:
            drive_count = int(footprint.shape[1])
            scales = row_scales.reshape((batch_size, drive_count))
            return current[None, :, :] * scales[:, :, None]
        raise ValueError(
            "scaled factorized current_mid_A must have shape (Nt,) or (S, Nt), "
            f"got {current.shape}."
        )
    if current.ndim == 1:
        return jnp.broadcast_to(current[None, None, :], (batch_size, 1, current.shape[0]))
    if current.ndim == 2:
        if footprint.ndim == 3 and batch.current_row_indices is None:
            drive_count = int(footprint.shape[1])
            if int(current.shape[0]) == drive_count:
                return jnp.broadcast_to(
                    current[None, :, :],
                    (batch_size, drive_count, int(current.shape[1])),
                )
        if batch.current_row_indices is not None:
            row_indices = jnp.asarray(batch.current_row_indices, dtype=jnp.int32)
            current = jnp.take(current, row_indices, axis=0)
        return current[:, None, :]
    if current.ndim == 3:
        return current
    raise ValueError(
        "factorized current_mid_A must have shape (Nt,), (B, Nt), or (B, K, Nt), "
        f"got {current.shape}."
    )

def _factorized_current_initial_previous_rows(
    batch: FactorizedExtracellularPotentialBatch,
    *,
    dtype_local: Any,
    batch_size: int,
) -> Array:
    previous = batch.current_initial_previous_A
    if previous is None:
        raise ValueError("factorized current_initial_previous_A is required.")
    previous_arr = jnp.asarray(previous, dtype=dtype_local)
    row_scales = (
        None
        if batch.current_row_scales is None
        else jnp.asarray(batch.current_row_scales, dtype=dtype_local)
    )
    footprint = jnp.asarray(batch.footprint_mV_per_A)
    if row_scales is not None:
        if previous_arr.ndim != 0:
            raise ValueError(
                "scaled shared factorized previous current must be scalar, "
                f"got {previous_arr.shape}."
            )
        if footprint.ndim == 2:
            return previous_arr * row_scales.reshape((batch_size,))
        if footprint.ndim == 3:
            drive_count = int(footprint.shape[1])
            return previous_arr * row_scales.reshape((batch_size, drive_count))
        raise ValueError(
            "scaled factorized footprint must have shape (B, Nx) or (B, K, Nx), "
            f"got {footprint.shape}."
        )
    if batch.current_row_indices is not None:
        if previous_arr.ndim == 0:
            return previous_arr
        if previous_arr.ndim == 1 and int(previous_arr.shape[0]) == batch_size:
            return previous_arr
        row_indices = jnp.asarray(batch.current_row_indices, dtype=jnp.int32)
        return jnp.take(previous_arr, row_indices, axis=0)
    return previous_arr

def _double_cable_factorized_vext_can_stay_compact(
    batch: FactorizedExtracellularPotentialBatch,
) -> bool:
    previous = batch.current_initial_previous_A
    if previous is None or batch.drive_count != 1:
        return False
    previous_is_scalar = jnp.asarray(previous).ndim == 0
    if batch.shared_current:
        return bool(previous_is_scalar)
    if batch.current_row_scales is not None:
        return bool(previous_is_scalar)
    return False

def _single_cable_factorized_forcing_footprint_for_batch(
    batch: FactorizedExtracellularPotentialBatch,
    *,
    lower: Array,
    upper: Array,
    lower_cache_source: Array,
    upper_cache_source: Array,
    dtype_local: Any,
) -> Array:
    """Return a cached factorized single-cable forcing footprint when possible."""

    cache_key = _single_cable_factorized_forcing_cache_key(
        batch,
        lower_cache_source=lower_cache_source,
        upper_cache_source=upper_cache_source,
        dtype_local=dtype_local,
    )
    cached = (
        None
        if cache_key is None
        else get_single_cable_factorized_forcing(cache_key)
    )
    batch_cached = batch.single_cable_forcing_footprint_mV_per_A
    cache_state = (
        "batch"
        if batch_cached is not None
        else "hit" if cached is not None else "miss" if cache_key is not None else "disabled"
    )
    with benchmark_span(
        "kernel.prepare_factorized_forcing",
        mode="single",
        cache=cache_state,
        group_size=batch.batch_size,
        drive_count=batch.drive_count,
        footprint_rank=jnp.asarray(batch.footprint_mV_per_A).ndim,
    ):
        if batch_cached is not None:
            return jnp.asarray(batch_cached, dtype=dtype_local)
        if cached is not None:
            return jnp.asarray(cached, dtype=dtype_local)
        forcing = _compute_single_cable_factorized_forcing_footprint(
            batch.footprint_mV_per_A,
            lower=lower,
            upper=upper,
            dtype_local=dtype_local,
        )
        if cache_key is not None:
            store_single_cable_factorized_forcing(cache_key, forcing)
        return forcing

def _single_cable_factorized_forcing_cache_key(
    batch: FactorizedExtracellularPotentialBatch,
    *,
    lower_cache_source: Array,
    upper_cache_source: Array,
    dtype_local: Any,
) -> tuple[Any, ...] | None:
    footprint_key = batch.static_footprint_key
    if footprint_key is None:
        return None
    return (
        "single_cable_factorized_forcing_v1",
        footprint_key,
        _array_identity_cache_key(lower_cache_source),
        _array_identity_cache_key(upper_cache_source),
        str(dtype_local),
    )

def _array_identity_cache_key(values: Array) -> tuple[Any, ...]:
    arr = jnp.asarray(values)
    return (
        id(values),
        tuple(int(dim) for dim in arr.shape),
        str(arr.dtype),
    )

def _compute_single_cable_factorized_forcing_footprint(
    footprint_mV_per_A: Array,
    *,
    lower: Array,
    upper: Array,
    dtype_local: Any,
) -> Array:
    """Lower factorized Vstim footprints to diffusion forcing footprints once."""

    footprint = jnp.asarray(footprint_mV_per_A, dtype=dtype_local)
    lower_rows = jnp.asarray(lower, dtype=dtype_local)
    upper_rows = jnp.asarray(upper, dtype=dtype_local)
    if footprint.ndim == 3:
        batch_size, drive_count, nx = footprint.shape
        flattened = footprint.reshape((batch_size * drive_count, nx))
        lower_rows = jnp.broadcast_to(
            lower_rows[:, None, :],
            (batch_size, drive_count, nx),
        ).reshape((batch_size * drive_count, nx))
        upper_rows = jnp.broadcast_to(
            upper_rows[:, None, :],
            (batch_size, drive_count, nx),
        ).reshape((batch_size * drive_count, nx))
        forcing = _compute_single_cable_factorized_forcing_footprint(
            flattened,
            lower=lower_rows,
            upper=upper_rows,
            dtype_local=dtype_local,
        )
        return forcing.reshape((batch_size, drive_count, nx))
    if footprint.ndim != 2:
        raise ValueError(
            "factorized single-cable footprints must have shape (B, Nx) or (B, K, Nx), "
            f"got {footprint.shape}."
        )
    nx = int(footprint.shape[1])
    if nx < 2:
        return jnp.zeros_like(footprint)
    first = upper_rows[:, :1] * (footprint[:, 1:2] - footprint[:, :1])
    last = lower_rows[:, -1:] * (footprint[:, -2:-1] - footprint[:, -1:])
    if nx == 2:
        return jnp.concatenate((first, last), axis=1)
    middle = (
        lower_rows[:, 1:-1] * (footprint[:, :-2] - footprint[:, 1:-1])
        + upper_rows[:, 1:-1] * (footprint[:, 2:] - footprint[:, 1:-1])
    )
    return jnp.concatenate((first, middle, last), axis=1)

__all__ = [
    "_compute_single_cable_factorized_forcing_footprint",
    "_double_cable_factorized_vext_can_stay_compact",
    "_factorized_current_initial_previous_rows",
    "_factorized_current_mid_rows",
    "_single_cable_factorized_forcing_footprint_for_batch",
]
