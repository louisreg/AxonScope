"""Runtime-neutral dispatch-group preparation helpers and caches."""

from __future__ import annotations

import hashlib
import weakref
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from axonscope.benchmarking import record_benchmark_metadata
from axonscope.dispatcher.plan import DispatchGroup, DispatchItem
from axonscope.preparation.cohort import PreparedCohort
from axonscope.preparation.runtime_batches import extracellular_stimulation_rows


_GROUP_SIGNATURE_CACHE_MAX_SIZE = 128
_PREPARED_COHORT_CACHE_MAX_SIZE = 64

_GROUP_STATIC_SIGNATURE_CACHE: OrderedDict[
    int,
    tuple[weakref.ReferenceType[DispatchGroup], tuple[Any, ...]],
] = OrderedDict()
_GROUP_RUNTIME_SIGNATURE_CACHE: OrderedDict[
    int,
    tuple[weakref.ReferenceType[DispatchGroup], tuple[Any, ...]],
] = OrderedDict()
_PREPARED_COHORT_CACHE: OrderedDict[tuple[Any, ...], PreparedCohort] = OrderedDict()
_PREPARED_COHORT_IDENTITY_CACHE: OrderedDict[
    int,
    tuple[weakref.ReferenceType[Any], PreparedCohort],
] = OrderedDict()


def representative_item(group: DispatchGroup) -> DispatchItem:
    """Return the row used to compile or inspect a shared group runtime."""

    for item in group.items:
        if int(item.solver_axon.n_compartments) == int(group.nx):
            return item
    return group.items[0]


def runtime_context_cache_key(context: Any | None) -> tuple[Any, ...] | None:
    """Return the runtime policy part of runtime cache identity."""

    if context is None:
        return None
    policy = getattr(context, "policy", None)
    runtime = getattr(policy, "runtime", None)
    device_request = getattr(policy, "device", None)
    precision = getattr(policy, "precision", None)
    solver_engine = getattr(context, "solver_engine", None)
    resolved_device = getattr(context, "device", None)
    resolved_device_key = None
    if resolved_device is not None:
        resolved_device_key = (
            getattr(resolved_device, "platform", None),
            getattr(resolved_device, "id", None),
            str(resolved_device),
        )
    return (
        "runtime_context_v1",
        getattr(runtime, "value", runtime),
        None
        if device_request is None
        else (
            getattr(device_request, "kind", None),
            getattr(device_request, "index", None),
        ),
        getattr(context, "platform", None),
        resolved_device_key,
        None
        if precision is None
        else (
            precision.state_dtype,
            precision.solver_dtype,
            precision.accumulation_dtype,
        ),
        None
        if solver_engine is None
        else (
            getattr(solver_engine, "name", None),
            getattr(solver_engine, "platform", None),
            getattr(solver_engine, "single_cable_solver", None),
            getattr(solver_engine, "double_cable_block_solver", None),
            getattr(solver_engine, "allow_internal_double_cable_block_solver", None),
            getattr(solver_engine, "tiled_thomas_block_b", None),
        ),
    )


def group_runtime_signature(group: DispatchGroup) -> tuple[Any, ...]:
    """Return a structural key for stimulation-independent solver runtimes."""

    return _cached_group_signature(
        group,
        cache=_GROUP_RUNTIME_SIGNATURE_CACHE,
        metadata_key="group_runtime_signature_cache",
        builder=_build_group_runtime_signature,
    )


def group_preparation_signature(group: DispatchGroup) -> tuple[Any, ...]:
    """Return the structural prepared-cohort cache key for a dispatch group."""

    return _cached_group_signature(
        group,
        cache=_GROUP_STATIC_SIGNATURE_CACHE,
        metadata_key="group_static_signature_cache",
        builder=_build_group_static_signature,
    )


def prepared_cohort_for_group(group: DispatchGroup) -> PreparedCohort:
    """Return a prepared cohort refreshed for the group's current stimulations."""

    cache_key = ("prepared_cohort_v1", group_preparation_signature(group))
    cached = _cache_get(_PREPARED_COHORT_CACHE, cache_key)
    if cached is not None:
        record_benchmark_metadata(prepared_cohort_cache="hit")
        return _with_current_stimulation_rows(cached, group)

    cohort = PreparedCohort.from_dispatch_group(group)
    _cache_store(_PREPARED_COHORT_CACHE, cache_key, cohort)
    record_benchmark_metadata(prepared_cohort_cache="miss")
    return cohort


def prepared_cohort_for_current_group(group: DispatchGroup) -> PreparedCohort:
    """Return a cohort for an unchanged dispatch group object."""

    cached = _get_prepared_cohort_identity(group)
    if cached is not None:
        record_benchmark_metadata(prepared_cohort_identity_cache="hit")
        refreshed = _with_current_stimulation_rows(cached, group)
        if refreshed is not cached:
            _store_prepared_cohort_identity(group, refreshed)
        return refreshed

    cohort = prepared_cohort_for_group(group)
    _store_prepared_cohort_identity(group, cohort)
    record_benchmark_metadata(prepared_cohort_identity_cache="miss")
    return cohort


def clear_prepared_cohort_cache() -> None:
    """Clear runtime-neutral prepared-cohort caches."""

    _PREPARED_COHORT_CACHE.clear()
    _PREPARED_COHORT_IDENTITY_CACHE.clear()


def clear_group_signature_caches() -> None:
    """Clear cached dispatch-group signatures."""

    _GROUP_STATIC_SIGNATURE_CACHE.clear()
    _GROUP_RUNTIME_SIGNATURE_CACHE.clear()


def _with_current_stimulation_rows(
    cohort: PreparedCohort,
    group: DispatchGroup,
) -> PreparedCohort:
    axons = tuple(item.simulation for item in group.items)
    stimulations = extracellular_stimulation_rows(axons)
    representative = representative_item(group).simulation
    if (
        _same_objects(cohort.axons, axons)
        and _same_stimulation_rows(cohort.stimulations, stimulations)
        and cohort.representative is representative
    ):
        return cohort
    return replace(
        cohort,
        representative=representative,
        axons=axons,
        stimulations=stimulations,
    )


def _same_objects(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return len(left) == len(right) and all(
        a is b for a, b in zip(left, right, strict=True)
    )


def _same_stimulation_rows(
    left: tuple[tuple[Any, ...], ...],
    right: tuple[tuple[Any, ...], ...],
) -> bool:
    if len(left) != len(right):
        return False
    return all(_same_objects(a, b) for a, b in zip(left, right, strict=True))


def _build_group_static_signature(group: DispatchGroup) -> tuple[Any, ...]:
    rows_digest = _digest_group_items(
        group.items,
        include_identity=True,
    )
    return (
        "dispatch_group_v3",
        group.mode,
        int(group.nx),
        bool(group.geometry_shared),
        bool(group.has_padding),
        int(group.size),
        _digest_signature_value(group.signature, cache={}),
        rows_digest,
    )


def _build_group_runtime_signature(group: DispatchGroup) -> tuple[Any, ...]:
    rows_digest = _digest_group_items(
        group.items,
        include_identity=False,
    )
    return (
        "dispatch_group_runtime_v3",
        group.mode,
        int(group.nx),
        bool(group.geometry_shared),
        bool(group.has_padding),
        int(group.size),
        _digest_signature_value(group.signature, cache={}),
        rows_digest,
    )


def _digest_group_items(
    items: tuple[DispatchItem, ...],
    *,
    include_identity: bool,
) -> str:
    token_cache: dict[int, str] = {}
    hasher = hashlib.blake2b(digest_size=16)
    for item in items:
        _update_digest_int(hasher, int(item.index))
        if include_identity:
            _update_digest_int(hasher, id(item.simulation))
            _update_digest_int(hasher, id(item.solver_axon))
        hasher.update(_digest_signature_value(item.membrane_signature, token_cache).encode())
        hasher.update(b"\0")
        hasher.update(_digest_signature_value(item.cable_signature, token_cache).encode())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _digest_signature_value(value: Any, cache: dict[int, str]) -> str:
    cache_key = id(value)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.blake2b(repr(value).encode("utf-8"), digest_size=16).hexdigest()
    cache[cache_key] = digest
    return digest


def _update_digest_int(hasher: Any, value: int) -> None:
    hasher.update(int(value).to_bytes(8, byteorder="little", signed=False))


def _cached_group_signature(
    group: DispatchGroup,
    *,
    cache: OrderedDict[int, tuple[weakref.ReferenceType[DispatchGroup], tuple[Any, ...]]],
    metadata_key: str,
    builder: Callable[[DispatchGroup], tuple[Any, ...]],
) -> tuple[Any, ...]:
    cache_key = id(group)
    cached = cache.get(cache_key)
    if cached is not None:
        ref, signature = cached
        if ref() is group:
            cache.move_to_end(cache_key)
            record_benchmark_metadata(**{metadata_key: "hit"})
            return signature
        cache.pop(cache_key, None)

    signature = builder(group)
    cache[cache_key] = (weakref.ref(group), signature)
    cache.move_to_end(cache_key)
    while len(cache) > _GROUP_SIGNATURE_CACHE_MAX_SIZE:
        cache.popitem(last=False)
    record_benchmark_metadata(**{metadata_key: "miss"})
    return signature


def _get_prepared_cohort_identity(group: Any) -> PreparedCohort | None:
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


def _store_prepared_cohort_identity(group: Any, cohort: PreparedCohort) -> None:
    _PREPARED_COHORT_IDENTITY_CACHE[id(group)] = (weakref.ref(group), cohort)
    _PREPARED_COHORT_IDENTITY_CACHE.move_to_end(id(group))
    while len(_PREPARED_COHORT_IDENTITY_CACHE) > _PREPARED_COHORT_CACHE_MAX_SIZE:
        _PREPARED_COHORT_IDENTITY_CACHE.popitem(last=False)


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
    while len(cache) > _PREPARED_COHORT_CACHE_MAX_SIZE:
        cache.popitem(last=False)


__all__ = [
    "clear_group_signature_caches",
    "clear_prepared_cohort_cache",
    "group_preparation_signature",
    "group_runtime_signature",
    "prepared_cohort_for_current_group",
    "prepared_cohort_for_group",
    "representative_item",
    "runtime_context_cache_key",
]
