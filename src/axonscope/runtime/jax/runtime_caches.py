"""Small bounded caches used by JAX batch runtime preparation."""

from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import Any

from axonscope.runtime.jax.runtime import SolverRuntime
from axonscope.preparation.cohort import PreparedCohort


_BATCH_RUNTIME_CACHE: OrderedDict[tuple[Any, ...], SolverRuntime] = OrderedDict()
_BATCH_STATIC_RUNTIME_CACHE: OrderedDict[tuple[Any, ...], SolverRuntime] = OrderedDict()
_PREPARED_COHORT_CACHE: OrderedDict[tuple[Any, ...], PreparedCohort] = OrderedDict()
_PREPARED_COHORT_IDENTITY_CACHE: OrderedDict[
    int,
    tuple[weakref.ReferenceType[Any], PreparedCohort],
] = OrderedDict()
_SINGLE_CABLE_FACTORIZED_FORCING_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_RUNTIME_CACHE_MAX_SIZE = 64


def get_batch_runtime(key: tuple[Any, ...]) -> SolverRuntime | None:
    """Return a full time-grid runtime cache entry."""

    return _cache_get(_BATCH_RUNTIME_CACHE, key)


def store_batch_runtime(key: tuple[Any, ...], runtime: SolverRuntime) -> None:
    """Store a full time-grid runtime cache entry."""

    _cache_store(_BATCH_RUNTIME_CACHE, key, runtime)


def get_batch_static_runtime(key: tuple[Any, ...]) -> SolverRuntime | None:
    """Return a stimulation-independent runtime cache entry."""

    return _cache_get(_BATCH_STATIC_RUNTIME_CACHE, key)


def store_batch_static_runtime(key: tuple[Any, ...], runtime: SolverRuntime) -> None:
    """Store a stimulation-independent runtime cache entry."""

    _cache_store(_BATCH_STATIC_RUNTIME_CACHE, key, runtime)


def get_prepared_cohort(key: tuple[Any, ...]) -> PreparedCohort | None:
    """Return a prepared-cohort cache entry."""

    return _cache_get(_PREPARED_COHORT_CACHE, key)


def store_prepared_cohort(key: tuple[Any, ...], cohort: PreparedCohort) -> None:
    """Store a prepared-cohort cache entry."""

    _cache_store(_PREPARED_COHORT_CACHE, key, cohort)


def get_prepared_cohort_identity(group: Any) -> PreparedCohort | None:
    """Return a prepared-cohort entry for this exact dispatch group object."""

    cache_key = id(group)
    cached = _PREPARED_COHORT_IDENTITY_CACHE.get(cache_key)
    if cached is None:
        return None
    ref, cohort = cached
    if ref() is group:
        _PREPARED_COHORT_IDENTITY_CACHE.move_to_end(cache_key)
        return cohort
    _PREPARED_COHORT_IDENTITY_CACHE.pop(cache_key, None)
    return None


def store_prepared_cohort_identity(group: Any, cohort: PreparedCohort) -> None:
    """Store a prepared cohort for this exact dispatch group object."""

    _PREPARED_COHORT_IDENTITY_CACHE[id(group)] = (weakref.ref(group), cohort)
    _PREPARED_COHORT_IDENTITY_CACHE.move_to_end(id(group))
    while len(_PREPARED_COHORT_IDENTITY_CACHE) > _RUNTIME_CACHE_MAX_SIZE:
        _PREPARED_COHORT_IDENTITY_CACHE.popitem(last=False)


def get_single_cable_factorized_forcing(key: tuple[Any, ...]) -> Any | None:
    """Return a cached single-cable forcing footprint."""

    return _cache_get(_SINGLE_CABLE_FACTORIZED_FORCING_CACHE, key)


def store_single_cable_factorized_forcing(key: tuple[Any, ...], forcing: Any) -> None:
    """Store a single-cable forcing footprint."""

    _cache_store(_SINGLE_CABLE_FACTORIZED_FORCING_CACHE, key, forcing)


def clear_batch_runtime_caches() -> None:
    """Clear dynamic and static runtime caches."""

    _BATCH_RUNTIME_CACHE.clear()
    _BATCH_STATIC_RUNTIME_CACHE.clear()
    _SINGLE_CABLE_FACTORIZED_FORCING_CACHE.clear()


def clear_prepared_cohort_cache() -> None:
    """Clear the prepared-cohort cache."""

    _PREPARED_COHORT_CACHE.clear()
    _PREPARED_COHORT_IDENTITY_CACHE.clear()


def clear_all_runtime_caches() -> None:
    """Clear every cache owned by this module."""

    clear_batch_runtime_caches()
    clear_prepared_cohort_cache()


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
    while len(cache) > _RUNTIME_CACHE_MAX_SIZE:
        cache.popitem(last=False)


__all__ = [
    "clear_all_runtime_caches",
    "clear_batch_runtime_caches",
    "clear_prepared_cohort_cache",
    "get_batch_runtime",
    "get_batch_static_runtime",
    "get_prepared_cohort",
    "get_single_cable_factorized_forcing",
    "store_batch_runtime",
    "store_batch_static_runtime",
    "store_prepared_cohort",
    "store_single_cable_factorized_forcing",
]
