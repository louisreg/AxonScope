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


@dataclass(frozen=True)
class FactorizedExtracellularPotentialBatch:
    """Static-footprint extracellular potential without the dense time-space tensor.

    ``current_mid_A`` stores the dynamic stimulus samples with shape ``(Nt,)``
    for a shared single-drive waveform, ``(B, Nt)`` for row-specific
    single-drive waveforms, or ``(B, K, Nt)`` for row-specific multi-drive
    waveforms.
    ``current_initial_previous_A`` optionally stores the ``t=-dt/2`` sample
    used by double-cable batches. ``footprint_mV_per_A`` stores the static
    spatial footprint with shape ``(B, Nx)`` or ``(B, K, Nx)``. The dense
    midpoint potential is their product, summed over the optional drive axis:
    ``Vstim[B, Nt, Nx] = sum_K current_mid_A * footprint_mV_per_A``.
    """

    current_mid_A: Array
    footprint_mV_per_A: Array
    target_nx: int
    current_initial_previous_A: Array | None = None

    def __post_init__(self) -> None:
        current_shape = tuple(int(dim) for dim in getattr(self.current_mid_A, "shape", ()))
        previous_shape = tuple(
            int(dim) for dim in getattr(self.current_initial_previous_A, "shape", ())
        )
        footprint_shape = tuple(
            int(dim) for dim in getattr(self.footprint_mV_per_A, "shape", ())
        )
        if len(footprint_shape) not in {2, 3}:
            raise ValueError("footprint_mV_per_A must have shape (B, Nx) or (B, K, Nx).")
        batch_size = footprint_shape[0]
        drive_count = 1 if len(footprint_shape) == 2 else footprint_shape[1]
        if len(current_shape) not in {1, 2, 3}:
            raise ValueError("current_mid_A must have shape (Nt,), (B, Nt), or (B, K, Nt).")
        if len(footprint_shape) == 2 and len(current_shape) == 3:
            raise ValueError("rank-K current_mid_A requires footprint_mV_per_A shape (B, K, Nx).")
        if len(footprint_shape) == 3 and len(current_shape) != 3:
            raise ValueError("rank-K footprint_mV_per_A requires current_mid_A shape (B, K, Nt).")
        if len(current_shape) == 2 and current_shape[0] != batch_size:
            raise ValueError(
                "current_mid_A batch size must match footprint_mV_per_A, "
                f"got {current_shape} and {footprint_shape}."
            )
        if len(current_shape) == 3 and current_shape[:2] != (batch_size, drive_count):
            raise ValueError(
                "current_mid_A batch/drive axes must match footprint_mV_per_A, "
                f"got {current_shape} and {footprint_shape}."
            )
        if self.current_initial_previous_A is not None:
            if len(footprint_shape) == 3:
                valid_previous_shapes = {(batch_size, drive_count)}
            else:
                valid_previous_shapes = {(), (batch_size,)}
            if previous_shape not in valid_previous_shapes:
                raise ValueError(
                    "current_initial_previous_A must be scalar or shape (B,) "
                    "for rank-1 batches, and shape (B, K) for rank-K batches; "
                    f"got {previous_shape} for footprint shape {footprint_shape}."
                )
        footprint_width = footprint_shape[-1]
        if int(self.target_nx) != footprint_width:
            raise ValueError(
                "target_nx must match footprint_mV_per_A width, "
                f"got target_nx={self.target_nx} and shape {footprint_shape}."
            )
        if int(self.target_nx) < 1:
            raise ValueError("target_nx must be >= 1.")
        object.__setattr__(self, "target_nx", int(self.target_nx))

    @property
    def batch_size(self) -> int:
        """Number of independent rows."""

        return int(self.footprint_mV_per_A.shape[0])

    @property
    def drive_count(self) -> int:
        """Maximum number of factorized drives per row."""

        shape = getattr(self.footprint_mV_per_A, "shape", ())
        return 1 if len(shape) == 2 else int(shape[1])

    @property
    def step_count(self) -> int:
        """Number of midpoint time samples."""

        current = self.current_mid_A
        if len(current.shape) == 1:
            return int(current.shape[0])
        if len(current.shape) == 2:
            return int(current.shape[1])
        return int(current.shape[2])

    @property
    def shared_current(self) -> bool:
        """Whether all rows share the same temporal waveform."""

        return len(getattr(self.current_mid_A, "shape", ())) == 1


def materialize_factorized_extracellular_potential_batch(
    batch: FactorizedExtracellularPotentialBatch,
) -> Array:
    """Expand a factorized extracellular batch to ``Vstim[B, Nt, Nx]``."""

    current_mid_A = jnp.asarray(batch.current_mid_A)
    footprint = jnp.asarray(batch.footprint_mV_per_A)
    if footprint.ndim == 2:
        if current_mid_A.ndim == 1:
            current = current_mid_A[None, :, None]
        else:
            current = current_mid_A[:, :, None]
        return current * footprint[:, None, :]
    if current_mid_A.ndim != 3:
        raise ValueError("rank-K factorized Vstim requires current_mid_A shape (B, K, Nt).")
    return jnp.sum(current_mid_A[:, :, :, None] * footprint[:, :, None, :], axis=1)


def materialize_factorized_extracellular_potential_initial_previous(
    batch: FactorizedExtracellularPotentialBatch,
) -> Array:
    """Expand a factorized ``t=-dt/2`` extracellular sample to ``Vstim[B, Nx]``."""

    if batch.current_initial_previous_A is None:
        raise ValueError("current_initial_previous_A is required.")
    current_previous_A = jnp.asarray(batch.current_initial_previous_A)
    footprint = jnp.asarray(batch.footprint_mV_per_A)
    if footprint.ndim == 2:
        if current_previous_A.ndim == 0:
            current = current_previous_A
        else:
            current = current_previous_A[:, None]
        return current * footprint
    if current_previous_A.ndim != 2:
        raise ValueError(
            "rank-K factorized previous Vstim requires current_initial_previous_A "
            "shape (B, K)."
        )
    return jnp.sum(current_previous_A[:, :, None] * footprint, axis=1)


__all__ = [
    "FactorizedExtracellularPotentialBatch",
    "SparseIntracellularCurrentDensityBatch",
    "materialize_factorized_extracellular_potential_initial_previous",
    "materialize_factorized_extracellular_potential_batch",
    "materialize_sparse_intracellular_current_density_batch",
]
