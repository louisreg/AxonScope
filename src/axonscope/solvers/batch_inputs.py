"""Internal solver input containers for batched kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

Array = Any


@dataclass(frozen=True)
class SparseIntracellularCurrentDensityBatch:
    """Point-clamp current density represented without the compartment axis.

    ``density_mid`` stores one sampled current-density waveform per sparse
    clamp slot with shape ``(B, Nt, K)``. ``indices`` and ``mask`` describe
    where each slot is injected in compartment space.
    """

    density_mid: Array
    indices: Array
    mask: Array
    target_nx: int

    def __post_init__(self) -> None:
        density_shape = tuple(int(dim) for dim in getattr(self.density_mid, "shape", ()))
        indices_shape = tuple(int(dim) for dim in getattr(self.indices, "shape", ()))
        mask_shape = tuple(int(dim) for dim in getattr(self.mask, "shape", ()))
        if len(density_shape) != 3:
            raise ValueError("density_mid must have shape (B, Nt, K).")
        expected_sparse_shape = (density_shape[0], density_shape[2])
        if indices_shape != expected_sparse_shape:
            raise ValueError(
                "indices must have shape (B, K) matching density_mid, "
                f"got {indices_shape} and {density_shape}."
            )
        if mask_shape != expected_sparse_shape:
            raise ValueError(
                "mask must have shape (B, K) matching density_mid, "
                f"got {mask_shape} and {density_shape}."
            )
        if int(self.target_nx) < 1:
            raise ValueError("target_nx must be >= 1.")
        object.__setattr__(self, "target_nx", int(self.target_nx))

    @property
    def batch_size(self) -> int:
        """Number of independent rows."""

        return int(self.density_mid.shape[0])

    @property
    def step_count(self) -> int:
        """Number of midpoint time samples."""

        return int(self.density_mid.shape[1])

    @property
    def max_sparse_entries(self) -> int:
        """Maximum number of clamp slots per row."""

        return int(self.density_mid.shape[2])


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


__all__ = [
    "SparseIntracellularCurrentDensityBatch",
    "materialize_sparse_intracellular_current_density_batch",
]
