"""Shared factorized-input helpers for JAX batch kernels."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from axonfleet.runtime.inputs.payloads import FactorizedExtracellularPotentialBatch
from axonfleet.runtime.jax.cable_geometry import Array


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
        if batch.current_row_indices is not None:
            row_indices = jnp.asarray(batch.current_row_indices, dtype=jnp.int32)
            return jnp.take(current, row_indices, axis=0)
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
        row_indices = jnp.asarray(batch.current_row_indices, dtype=jnp.int32)
        return jnp.take(previous_arr, row_indices, axis=0)
    return previous_arr

def _single_cable_factorized_forcing_footprint(
    batch: FactorizedExtracellularPotentialBatch,
    *,
    dtype_local: Any,
) -> Array:
    """Return the forcing footprint prepared by single-cable input lowering."""

    forcing = batch.single_cable_forcing_footprint_mV_per_A
    if forcing is None:
        raise ValueError(
            "single-cable factorized input requires a precomputed forcing footprint."
        )
    return jnp.asarray(forcing, dtype=dtype_local)
