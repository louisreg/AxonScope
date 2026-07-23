"""JAX threshold-observer lowering for batch execution."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import numpy as np

from axonfleet.benchmarking import record_benchmark_metadata
from axonfleet.runtime.outputs.contracts import observer_definition_signature
from axonfleet.runtime.recording import cohort_original_indices


_THRESHOLD_OBSERVER_PLAN_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_THRESHOLD_OBSERVER_PLAN_IDENTITY_CACHE: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()
_RECORDING_LOWERING_CACHE_MAX_SIZE = 64


def lower_observers_for_cohort(
    observers: tuple[Any, ...] | None,
    *,
    cohort: Any,
    dtype: Any,
) -> Any:
    """Lower public observers to the canonical threshold plan."""

    if observers is None:
        return None
    identity_key = _threshold_observer_plan_identity_cache_key(
        observers,
        cohort=cohort,
        dtype=dtype,
    )
    spatial_token = _cohort_spatial_cache_token(cohort)
    cached = _identity_cache_get(
        _THRESHOLD_OBSERVER_PLAN_IDENTITY_CACHE,
        identity_key,
        spatial_token,
    )
    if cached is not None:
        record_benchmark_metadata(
            threshold_observer_plan_identity_cache="hit",
            threshold_observer_plan_cache="hit",
        )
        return cached

    cache_key = _threshold_observer_plan_cache_key(
        observers,
        cohort=cohort,
        dtype=dtype,
    )
    cached = _cache_get(_THRESHOLD_OBSERVER_PLAN_CACHE, cache_key)
    if cached is not None:
        _identity_cache_store(
            _THRESHOLD_OBSERVER_PLAN_IDENTITY_CACHE,
            identity_key,
            spatial_token,
            cached,
        )
        record_benchmark_metadata(
            threshold_observer_plan_identity_cache="miss",
            threshold_observer_plan_cache="hit",
        )
        return cached

    from axonfleet.runtime.jax.recording.observer import build_threshold_observer_plan

    row_positions_um = np.asarray(cohort.x_positions_m, dtype=float) * 1e6
    plan = build_threshold_observer_plan(
        observers,
        positions_um=row_positions_um,
        original_indices=cohort_original_indices(cohort),
        dtype=dtype,
    )
    _cache_store(_THRESHOLD_OBSERVER_PLAN_CACHE, cache_key, plan)
    _identity_cache_store(
        _THRESHOLD_OBSERVER_PLAN_IDENTITY_CACHE,
        identity_key,
        spatial_token,
        plan,
    )
    record_benchmark_metadata(
        threshold_observer_plan_identity_cache="miss",
        threshold_observer_plan_cache="miss",
        threshold_observer_count=0 if plan is None else plan.definition_count,
        threshold_observer_probe_count=0 if plan is None else plan.probe_count,
    )
    return plan


def _threshold_observer_plan_cache_key(
    observers: tuple[Any, ...],
    *,
    cohort: Any,
    dtype: Any,
) -> tuple[Any, ...]:
    return (
        "threshold_observer_plan_v1",
        str(np.dtype(dtype)),
        _prepared_cohort_signature(cohort),
        tuple(observer_definition_signature(observer) for observer in observers),
    )


def _threshold_observer_plan_identity_cache_key(
    observers: tuple[Any, ...],
    *,
    cohort: Any,
    dtype: Any,
) -> tuple[Any, ...]:
    return (
        "threshold_observer_plan_identity_v1",
        id(_cohort_spatial_cache_token(cohort)),
        str(np.dtype(dtype)),
        tuple(observer_definition_signature(observer) for observer in observers),
    )


def _cohort_spatial_cache_token(cohort: Any) -> Any:
    """Return identity shared by cohort refreshes with unchanged spatial rows."""

    return getattr(cohort, "spatial_cache_token", cohort)


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
    spatial_token: Any,
) -> Any | None:
    entry = cache.get(key)
    if entry is None:
        return None
    cached_token, value = entry
    if cached_token is not spatial_token:
        cache.pop(key, None)
        return None
    cache.move_to_end(key)
    return value


def _identity_cache_store(
    cache: OrderedDict[tuple[Any, ...], tuple[Any, Any]],
    key: tuple[Any, ...],
    spatial_token: Any,
    value: Any,
) -> None:
    cache[key] = (spatial_token, value)
    cache.move_to_end(key)
    while len(cache) > _RECORDING_LOWERING_CACHE_MAX_SIZE:
        cache.popitem(last=False)
