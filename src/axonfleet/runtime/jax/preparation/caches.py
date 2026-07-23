"""Small bounded caches used by JAX batch runtime preparation."""

from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import Any, Callable

from axonfleet.runtime.jax.types import SolverRuntime


_BATCH_RUNTIME_CACHE: OrderedDict[tuple[Any, ...], SolverRuntime] = OrderedDict()
_BATCH_STATIC_RUNTIME_CACHE: OrderedDict[tuple[Any, ...], SolverRuntime] = OrderedDict()
_BATCHED_STATIC_ARRAY_CACHE: OrderedDict[
    tuple[Any, ...],
    tuple[tuple[Callable[[], Any], ...] | None, Any],
] = OrderedDict()
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


def get_batched_static_array(
    key: tuple[Any, ...],
    *,
    sources: tuple[Any, ...] | None = None,
) -> Any | None:
    """Return a cached batched static kernel array."""

    entry = _BATCHED_STATIC_ARRAY_CACHE.get(key)
    if entry is None:
        return None
    source_refs, values = entry
    if sources is not None and not _same_live_sources(source_refs, sources):
        _BATCHED_STATIC_ARRAY_CACHE.pop(key, None)
        return None
    _BATCHED_STATIC_ARRAY_CACHE.move_to_end(key)
    return values


def store_batched_static_array(
    key: tuple[Any, ...],
    values: Any,
    *,
    sources: tuple[Any, ...] | None = None,
) -> None:
    """Store a batched static kernel array."""

    source_refs = (
        None
        if sources is None
        else tuple(_identity_ref(source) for source in sources)
    )
    _cache_store(_BATCHED_STATIC_ARRAY_CACHE, key, (source_refs, values))


def _identity_ref(source: Any) -> Callable[[], Any]:
    try:
        return weakref.ref(source)
    except TypeError:
        return lambda: source


def _same_live_sources(
    refs: tuple[Callable[[], Any], ...] | None,
    sources: tuple[Any, ...],
) -> bool:
    return refs is not None and len(refs) == len(sources) and all(
        ref() is source for ref, source in zip(refs, sources, strict=True)
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
    while len(cache) > _RUNTIME_CACHE_MAX_SIZE:
        cache.popitem(last=False)


__all__ = [
    "get_batch_runtime",
    "get_batch_static_runtime",
    "get_batched_static_array",
    "store_batch_runtime",
    "store_batch_static_runtime",
    "store_batched_static_array",
]
