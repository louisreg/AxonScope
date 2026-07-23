"""Runtime-neutral dispatch-group preparation helpers and caches."""

from __future__ import annotations

import weakref
from collections import OrderedDict
from dataclasses import replace
from typing import Any

from axonfleet.benchmarking import record_benchmark_metadata
from axonfleet.axon_instance import extracellular_topology_revision
from axonfleet.dispatcher.plan import DispatchGroup, DispatchItem
from axonfleet.preparation.cohort import PreparedCohort
from axonfleet.preparation.membrane_rows import MembraneRowPlan
from axonfleet.preparation.stimulation_rows import extracellular_stimulation_rows


_PREPARED_COHORT_CACHE_MAX_SIZE = 64

_PREPARED_COHORT_CACHE: OrderedDict[tuple[Any, ...], PreparedCohort] = OrderedDict()
_PREPARED_COHORT_IDENTITY_CACHE: OrderedDict[
    int,
    tuple[weakref.ReferenceType[Any], int, PreparedCohort],
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
            getattr(solver_engine, "tiled_thomas_block_b", None),
        ),
    )


def group_runtime_signature(group: DispatchGroup) -> tuple[Any, ...]:
    """Return a structural key for stimulation-independent solver runtimes."""

    record_benchmark_metadata(group_runtime_signature_source="dispatch_plan")
    return (
        "dispatch_group_runtime_v6",
        group.mode,
        int(group.nx),
        bool(group.geometry_shared),
        bool(group.has_padding),
        int(group.size),
        int(group.structure.schema_version),
        group.structure.runtime_rows,
    )


def group_preparation_signature(group: DispatchGroup) -> tuple[Any, ...]:
    """Return the structural prepared-cohort cache key for a dispatch group."""

    record_benchmark_metadata(group_static_signature_source="dispatch_plan")
    return (
        "dispatch_group_spatial_v5",
        group.mode,
        int(group.nx),
        bool(group.geometry_shared),
        bool(group.has_padding),
        int(group.size),
        int(group.structure.schema_version),
        group.structure.spatial_rows,
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
        cached_revision, cohort = cached
        current_revision = extracellular_topology_revision()
        if cached_revision == current_revision:
            record_benchmark_metadata(
                prepared_cohort_identity_cache="hit",
                prepared_cohort_topology_check="unchanged",
            )
            return cohort
        refreshed = _with_current_stimulation_rows(cohort, group)
        _store_prepared_cohort_identity(group, refreshed)
        record_benchmark_metadata(
            prepared_cohort_identity_cache="hit",
            prepared_cohort_topology_check="refreshed",
        )
        return refreshed

    cohort = prepared_cohort_for_group(group)
    _store_prepared_cohort_identity(group, cohort)
    record_benchmark_metadata(prepared_cohort_identity_cache="miss")
    return cohort


def _with_current_stimulation_rows(
    cohort: PreparedCohort,
    group: DispatchGroup,
) -> PreparedCohort:
    axons = tuple(item.simulation for item in group.items)
    solver_axons = tuple(item.solver_axon for item in group.items)
    stimulations = extracellular_stimulation_rows(axons)
    if (
        int(cohort.group_id) == int(group.group_id)
        and _same_objects(cohort.axons, axons)
        and _same_objects(cohort.solver_axons, solver_axons)
        and _same_stimulation_rows(cohort.stimulations, stimulations)
    ):
        return cohort
    return replace(
        cohort,
        group_id=int(group.group_id),
        axons=axons,
        solver_axons=solver_axons,
        membrane_rows=MembraneRowPlan.from_dispatch_items(group.items),
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


def _get_prepared_cohort_identity(
    group: Any,
) -> tuple[int, PreparedCohort] | None:
    cache_key = id(group)
    cached = _PREPARED_COHORT_IDENTITY_CACHE.get(cache_key)
    if cached is None:
        return None
    ref, revision, cohort = cached
    if ref() is group:
        _PREPARED_COHORT_IDENTITY_CACHE.move_to_end(cache_key)
        return revision, cohort
    _PREPARED_COHORT_IDENTITY_CACHE.pop(cache_key, None)
    return None


def _store_prepared_cohort_identity(group: Any, cohort: PreparedCohort) -> None:
    _PREPARED_COHORT_IDENTITY_CACHE[id(group)] = (
        weakref.ref(group),
        extracellular_topology_revision(),
        cohort,
    )
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
    "group_preparation_signature",
    "group_runtime_signature",
    "prepared_cohort_for_current_group",
    "prepared_cohort_for_group",
    "representative_item",
    "runtime_context_cache_key",
]
