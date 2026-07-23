"""JAX materializers for compact batched input payloads."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from axonfleet.runtime.inputs.payloads import (
    FactorizedExtracellularPotentialBatch,
    SparseIntracellularCurrentDensityBatch,
)

Array = Any


def materialize_sparse_intracellular_current_density_batch(
    batch: SparseIntracellularCurrentDensityBatch,
) -> Array:
    """Expand a sparse current-density batch to ``Iinj[B, Nt, Nx]``."""

    density_mid = jnp.asarray(batch.density_mid)
    indices = jnp.asarray(batch.indices, dtype=jnp.int32)
    mask = jnp.asarray(batch.mask, dtype=bool)
    target_nx = int(batch.target_nx)

    def one_row(row_values: Array, row_indices: Array, row_mask: Array) -> Array:
        safe_indices = jnp.where(row_mask, row_indices, 0)
        safe_values = jnp.where(row_mask[None, :], row_values, 0.0)

        def one_step(step_values: Array) -> Array:
            return jnp.zeros((target_nx,), dtype=density_mid.dtype).at[safe_indices].add(
                step_values
            )

        return jax.vmap(one_step)(safe_values)

    return jax.vmap(one_row)(density_mid, indices, mask)


def materialize_factorized_extracellular_potential_batch(
    batch: FactorizedExtracellularPotentialBatch,
) -> Array:
    """Expand a factorized extracellular batch to ``Vstim[B, Nt, Nx]``."""

    current_mid_A = jnp.asarray(batch.current_mid_A)
    footprint = jnp.asarray(batch.footprint_mV_per_A)
    row_scales = (
        None
        if batch.current_row_scales is None
        else jnp.asarray(batch.current_row_scales)
    )
    if row_scales is not None:
        if footprint.ndim == 2:
            scales = row_scales.reshape((footprint.shape[0],))
            current = current_mid_A[None, :, None] * scales[:, None, None]
            return current * footprint[:, None, :]
        scales = row_scales.reshape((footprint.shape[0], footprint.shape[1]))
        current = current_mid_A[None, :, :, None] * scales[:, :, None, None]
        return jnp.sum(current * footprint[:, :, None, :], axis=1)
    if footprint.ndim == 2:
        if current_mid_A.ndim == 1:
            current = current_mid_A[None, :, None]
        else:
            if batch.current_row_indices is not None:
                row_indices = jnp.asarray(batch.current_row_indices, dtype=jnp.int32)
                current_mid_A = jnp.take(current_mid_A, row_indices, axis=0)
            current = current_mid_A[:, :, None]
        return current * footprint[:, None, :]
    if current_mid_A.ndim == 2:
        return jnp.sum(
            current_mid_A[None, :, :, None] * footprint[:, :, None, :],
            axis=1,
        )
    if current_mid_A.ndim != 3:
        raise ValueError(
            "multi-drive factorized Vstim requires current_mid_A shape (S, Nt) "
            "or (B, S, Nt)."
        )
    if batch.current_row_indices is not None:
        row_indices = jnp.asarray(batch.current_row_indices, dtype=jnp.int32)
        current_mid_A = jnp.take(current_mid_A, row_indices, axis=0)
    return jnp.sum(current_mid_A[:, :, :, None] * footprint[:, :, None, :], axis=1)


def materialize_factorized_extracellular_potential_initial_previous(
    batch: FactorizedExtracellularPotentialBatch,
) -> Array:
    """Expand a factorized ``t=-dt/2`` extracellular sample to ``Vstim[B, Nx]``."""

    if batch.current_initial_previous_A is None:
        raise ValueError("current_initial_previous_A is required.")
    current_previous_A = jnp.asarray(batch.current_initial_previous_A)
    footprint = jnp.asarray(batch.footprint_mV_per_A)
    row_scales = (
        None
        if batch.current_row_scales is None
        else jnp.asarray(batch.current_row_scales)
    )
    if row_scales is not None:
        if footprint.ndim == 2:
            scales = row_scales.reshape((footprint.shape[0],))
            return (current_previous_A * scales)[:, None] * footprint
        scales = row_scales.reshape((footprint.shape[0], footprint.shape[1]))
        current = current_previous_A[None, :, None] * scales[:, :, None]
        return jnp.sum(current * footprint, axis=1)
    if footprint.ndim == 2:
        if current_previous_A.ndim == 0:
            current = current_previous_A
        else:
            if batch.current_row_indices is not None:
                row_indices = jnp.asarray(batch.current_row_indices, dtype=jnp.int32)
                if current_previous_A.shape[0] != footprint.shape[0]:
                    current_previous_A = jnp.take(
                        current_previous_A,
                        row_indices,
                        axis=0,
                    )
            current = current_previous_A[:, None]
        return current * footprint
    if current_previous_A.ndim == 1:
        return jnp.sum(current_previous_A[None, :, None] * footprint, axis=1)
    if current_previous_A.ndim != 2:
        raise ValueError(
            "rank-K factorized previous Vstim requires current_initial_previous_A "
            "shape (S,) or (B, S)."
        )
    if batch.current_row_indices is not None:
        row_indices = jnp.asarray(batch.current_row_indices, dtype=jnp.int32)
        current_previous_A = jnp.take(current_previous_A, row_indices, axis=0)
    return jnp.sum(current_previous_A[:, :, None] * footprint, axis=1)
