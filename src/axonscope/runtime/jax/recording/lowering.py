"""JAX VmRaster observer lowering for batch execution."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import numpy as np

from axonscope.benchmarking import record_benchmark_metadata
from axonscope.runtime.output_contract import observer_definition_signature
from axonscope.runtime.recording import cohort_original_indices


_VM_RASTER_PLAN_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_VM_RASTER_PLAN_IDENTITY_CACHE: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()
_RECORDING_LOWERING_CACHE_MAX_SIZE = 64


def lower_observers_for_cohort(
    observers: tuple[Any, ...] | None,
    *,
    cohort: Any,
    dtype: Any,
    prefer_vm_raster: bool = False,
) -> Any:
    """Lower public observers to a VmRaster plan when the kernel can consume one."""

    if observers is None or not prefer_vm_raster:
        return None
    identity_key = _vm_raster_plan_identity_cache_key(
        observers,
        cohort=cohort,
        dtype=dtype,
    )
    cached = _identity_cache_get(
        _VM_RASTER_PLAN_IDENTITY_CACHE,
        identity_key,
        cohort,
    )
    if cached is not None:
        record_benchmark_metadata(
            vm_raster_plan_identity_cache="hit",
            vm_raster_plan_cache="hit",
        )
        return cached

    cache_key = _vm_raster_plan_cache_key(
        observers,
        cohort=cohort,
        dtype=dtype,
    )
    cached = _cache_get(_VM_RASTER_PLAN_CACHE, cache_key)
    if cached is not None:
        _identity_cache_store(
            _VM_RASTER_PLAN_IDENTITY_CACHE,
            identity_key,
            cohort,
            cached,
        )
        record_benchmark_metadata(
            vm_raster_plan_identity_cache="miss",
            vm_raster_plan_cache="hit",
        )
        return cached

    from axonscope.runtime.jax.recording.observer import build_vm_raster_plan

    row_positions_um = np.asarray(cohort.x_positions_m, dtype=float) * 1e6
    plan = build_vm_raster_plan(
        observers,
        positions_um=row_positions_um,
        original_indices=cohort_original_indices(cohort),
        dtype=dtype,
    )
    _cache_store(_VM_RASTER_PLAN_CACHE, cache_key, plan)
    _identity_cache_store(
        _VM_RASTER_PLAN_IDENTITY_CACHE,
        identity_key,
        cohort,
        plan,
    )
    record_benchmark_metadata(
        vm_raster_plan_identity_cache="miss",
        vm_raster_plan_cache="miss",
        vm_raster_count=0 if plan is None else plan.raster_count,
        vm_raster_probe_count=0 if plan is None else plan.probe_count,
    )
    return plan


def _vm_raster_plan_cache_key(
    observers: tuple[Any, ...],
    *,
    cohort: Any,
    dtype: Any,
) -> tuple[Any, ...]:
    return (
        "vm_raster_plan_v1",
        str(np.dtype(dtype)),
        _prepared_cohort_signature(cohort),
        tuple(observer_definition_signature(observer) for observer in observers),
    )


def _vm_raster_plan_identity_cache_key(
    observers: tuple[Any, ...],
    *,
    cohort: Any,
    dtype: Any,
) -> tuple[Any, ...]:
    return (
        "vm_raster_plan_identity_v1",
        id(cohort),
        str(np.dtype(dtype)),
        tuple(observer_definition_signature(observer) for observer in observers),
    )


def _prepared_cohort_signature(cohort: Any) -> tuple[Any, ...]:
    return (
        "prepared_cohort_v1",
        int(cohort.group_id),
        str(cohort.mode),
        int(cohort.size),
        int(cohort.nx),
        bool(cohort.geometry_shared),
        bool(cohort.has_padding),
        tuple(id(axon) for axon in cohort.axons),
        tuple(id(solver_axon) for solver_axon in cohort.solver_axons),
        _array_shape_dtype_digest(cohort.x_positions_m),
        _array_shape_dtype_digest(cohort.axon_y_um),
        _array_shape_dtype_digest(cohort.axon_z_um),
    )


def _array_shape_dtype_digest(values: Any) -> tuple[Any, ...]:
    arr = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
    return (
        tuple(int(dim) for dim in arr.shape),
        arr.dtype.str,
        digest,
    )


def _cache_get(cache: OrderedDict[tuple[Any, ...], Any], key: tuple[Any, ...]) -> Any | None:
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_store(
    cache: OrderedDict[tuple[Any, ...], Any],
    key: tuple[Any, ...],
    value: Any,
) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _RECORDING_LOWERING_CACHE_MAX_SIZE:
        cache.popitem(last=False)


def _identity_cache_get(
    cache: OrderedDict[tuple[Any, ...], tuple[Any, Any]],
    key: tuple[Any, ...],
    cohort: Any,
) -> Any | None:
    entry = cache.get(key)
    if entry is None:
        return None
    cached_cohort, value = entry
    if cached_cohort is not cohort:
        cache.pop(key, None)
        return None
    cache.move_to_end(key)
    return value


def _identity_cache_store(
    cache: OrderedDict[tuple[Any, ...], tuple[Any, Any]],
    key: tuple[Any, ...],
    cohort: Any,
    value: Any,
) -> None:
    cache[key] = (cohort, value)
    cache.move_to_end(key)
    while len(cache) > _RECORDING_LOWERING_CACHE_MAX_SIZE:
        cache.popitem(last=False)


__all__ = [
    "lower_observers_for_cohort",
]
