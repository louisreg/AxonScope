"""Shape bucketing helpers for JAX batch execution."""

from __future__ import annotations

import os

from axonscope.benchmarking import record_benchmark_metadata
from axonscope.dispatcher.plan import DispatchGroup


_DOUBLE_CABLE_SHAPE_BUCKETING_ENV = "AXONSCOPE_EXPERIMENTAL_DOUBLE_CABLE_SHAPE_BUCKETING"
_DOUBLE_CABLE_BATCH_BUCKETS = (16, 32, 64, 128)
_DOUBLE_CABLE_NX_BUCKET_MULTIPLE = 128


def double_cable_kernel_group(group: DispatchGroup) -> DispatchGroup:
    """Return a backend-only group padded to stable double-cable JAX shapes."""

    if group.mode != "double" or not double_cable_shape_bucketing_enabled():
        return group
    target_size = _bucket_batch_size(group.size, buckets=_DOUBLE_CABLE_BATCH_BUCKETS)
    target_nx = _bucket_nx(
        group.nx,
        multiple=_DOUBLE_CABLE_NX_BUCKET_MULTIPLE,
    )
    if target_size == group.size and target_nx == group.nx:
        return group
    if not group.items:
        raise ValueError("cannot bucket an empty dispatch group.")
    padded_items = tuple(group.items) + (group.items[-1],) * (target_size - group.size)
    return DispatchGroup(
        group_id=group.group_id,
        items=padded_items,
        signature=(
            group.signature,
            "double_cable_kernel_bucket_v1",
            int(group.size),
            int(group.nx),
            int(target_size),
            int(target_nx),
        ),
        mode=group.mode,
        nx=int(target_nx),
        structure=group.structure.padded_with_last(
            group.items[-1],
            target_size - group.size,
        ),
        geometry_shared=False,
        numeric_axis=group.numeric_axis,
        numeric_axis_source_size=group.numeric_axis_source_size,
    )


def double_cable_shape_bucketing_enabled() -> bool:
    """Whether to run the experimental shape-bucketed double-cable kernel path."""

    return os.environ.get(_DOUBLE_CABLE_SHAPE_BUCKETING_ENV, "").strip() in {
        "1",
        "true",
        "yes",
    }


def record_kernel_bucket_metadata(
    *,
    group: DispatchGroup,
    kernel_group: DispatchGroup,
) -> None:
    """Record public-vs-kernel group shape metadata."""

    record_benchmark_metadata(
        public_group_size=int(group.size),
        public_nx=int(group.nx),
        kernel_group_size=int(kernel_group.size),
        kernel_nx=int(kernel_group.nx),
        kernel_batch_padding_rows=int(kernel_group.size) - int(group.size),
        kernel_spatial_padding=int(kernel_group.nx) - int(group.nx),
        kernel_shape_bucketed=(
            int(kernel_group.size) != int(group.size)
            or int(kernel_group.nx) != int(group.nx)
        ),
    )


def _bucket_batch_size(value: int, *, buckets: tuple[int, ...]) -> int:
    requested = int(value)
    if requested <= 0:
        raise ValueError("batch size must be positive.")
    for bucket in buckets:
        if requested <= int(bucket):
            return int(bucket)
    step = int(buckets[-1])
    return ((requested + step - 1) // step) * step


def _bucket_nx(value: int, *, multiple: int) -> int:
    requested = int(value)
    step = int(multiple)
    if requested <= 0 or step <= 0:
        raise ValueError("Nx and bucket multiple must be positive.")
    return ((requested + step - 1) // step) * step


__all__ = [
    "double_cable_kernel_group",
    "double_cable_shape_bucketing_enabled",
    "record_kernel_bucket_metadata",
]
