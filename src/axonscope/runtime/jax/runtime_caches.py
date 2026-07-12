"""Small bounded caches used by JAX batch runtime preparation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from axonscope.runtime.jax.runtime import SolverRuntime


_BATCH_RUNTIME_CACHE: OrderedDict[tuple[Any, ...], SolverRuntime] = OrderedDict()
_BATCH_STATIC_RUNTIME_CACHE: OrderedDict[tuple[Any, ...], SolverRuntime] = OrderedDict()
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


def clear_all_runtime_caches() -> None:
    """Clear every cache owned by this module."""

    clear_batch_runtime_caches()
    from axonscope.runtime.group_preparation import (
        clear_group_signature_caches,
        clear_prepared_cohort_cache,
    )

    clear_group_signature_caches()
    clear_prepared_cohort_cache()
    from axonscope.runtime.jax.execution_policy import clear_jax_execution_caches

    clear_jax_execution_caches()


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
    "get_batch_runtime",
    "get_batch_static_runtime",
    "get_single_cable_factorized_forcing",
    "store_batch_runtime",
    "store_batch_static_runtime",
    "store_single_cable_factorized_forcing",
]
