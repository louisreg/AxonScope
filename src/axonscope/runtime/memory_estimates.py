"""Runtime-neutral memory-estimate helpers for prepared batch execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from axonscope.runtime.input_contract import dense_nbytes_for_shape


@dataclass(frozen=True)
class RuntimeGroupMemoryEstimate:
    """Primitive memory estimate for one prepared batch group."""

    components_nbytes: dict[str, int]
    total_nbytes: int
    dtype: str
    shape: dict[str, int]
    vstim_dense_equivalent_nbytes: int | None = None
    factorized_rank: int | None = None

    @property
    def total_mib(self) -> float:
        """Total estimated memory in MiB."""

        return self.total_nbytes / (1024**2)

    def as_metadata(self) -> dict[str, Any]:
        """Return benchmark-friendly primitive metadata."""

        metadata: dict[str, Any] = {
            "memory_estimate_components_nbytes": self.components_nbytes,
            "memory_estimate_total_nbytes": self.total_nbytes,
            "memory_estimate_total_mib": self.total_mib,
            "memory_estimate_dtype": self.dtype,
            "memory_estimate_shape": self.shape,
        }
        if self.vstim_dense_equivalent_nbytes is not None:
            metadata["memory_estimate_vstim_dense_equivalent_nbytes"] = (
                self.vstim_dense_equivalent_nbytes
            )
        if self.factorized_rank is not None:
            metadata["memory_estimate_factorized_rank"] = self.factorized_rank
        return metadata


def estimate_runtime_group_memory(
    *,
    batch_size: int,
    nt: int,
    nx: int,
    dtype: np.dtype,
    positions_nbytes: int,
    recording_width: int,
    intracellular_format: str,
    extracellular_format: str,
    include_vstim_previous: bool,
    factorized_rank: int | None = None,
) -> RuntimeGroupMemoryEstimate:
    """Estimate memory implied by one runtime-neutral batch input contract."""

    dtype = np.dtype(dtype)
    itemsize = int(dtype.itemsize)
    batch_size = int(batch_size)
    nt = int(nt)
    nx = int(nx)
    recording_width = int(recording_width)
    positions_nbytes = int(positions_nbytes)
    dense_shape = (batch_size, nt, nx)
    dense_nbytes = dense_nbytes_for_shape(dense_shape, dtype=dtype)
    rank_for_metadata: int | None = None

    if extracellular_format == "zero_no_extracellular_stimulation":
        vstim_mid_nbytes = 0
        vstim_dense_equivalent_nbytes = None
    elif extracellular_format == "factorized_footprint":
        rank = _require_factorized_rank(factorized_rank)
        rank_for_metadata = rank
        vstim_mid_nbytes = (
            batch_size * rank * nt + batch_size * rank * nx
        ) * itemsize
        vstim_dense_equivalent_nbytes = dense_nbytes
    else:
        vstim_mid_nbytes = dense_nbytes
        vstim_dense_equivalent_nbytes = None

    if not include_vstim_previous:
        vstim_previous_nbytes = 0
    elif extracellular_format == "factorized_footprint":
        rank = _require_factorized_rank(factorized_rank)
        vstim_previous_nbytes = batch_size * rank * itemsize
    else:
        vstim_previous_nbytes = batch_size * nx * itemsize

    iinj_dense_nbytes = dense_nbytes if intracellular_format == "dense" else 0
    vm_output_nbytes = batch_size * nt * recording_width * itemsize
    components = {
        "positions": positions_nbytes,
        "vstim_mid": vstim_mid_nbytes,
        "vstim_previous": vstim_previous_nbytes,
        "iinj_dense": iinj_dense_nbytes,
        "vm_output": vm_output_nbytes,
    }
    return RuntimeGroupMemoryEstimate(
        components_nbytes=components,
        total_nbytes=int(sum(components.values())),
        dtype=str(dtype),
        shape={
            "batch_size": batch_size,
            "nt": nt,
            "nx": nx,
            "recording_width": recording_width,
        },
        vstim_dense_equivalent_nbytes=vstim_dense_equivalent_nbytes,
        factorized_rank=rank_for_metadata,
    )


def _require_factorized_rank(factorized_rank: int | None) -> int:
    if factorized_rank is None:
        raise ValueError("factorized_rank is required for factorized extracellular input.")
    rank = int(factorized_rank)
    if rank < 1:
        raise ValueError("factorized_rank must be >= 1.")
    return rank


__all__ = [
    "RuntimeGroupMemoryEstimate",
    "estimate_runtime_group_memory",
]
