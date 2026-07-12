"""Recording and observer lowering contracts for JAX batch execution."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import numpy as np

from axonscope.benchmarking import record_benchmark_metadata
from axonscope.runtime.output_contract import (
    observers_are_vm_raster_compatible,
    vm_raster_definitions,
)
from axonscope.solvers.options import BatchOptions, BatchRecording


_VM_RASTER_PLAN_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_VM_RASTER_PLAN_IDENTITY_CACHE: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()
_RECORDING_LOWERING_CACHE_MAX_SIZE = 64


def lower_batch_recording_options(
    group: Any,
    options: BatchOptions,
    *,
    observers: tuple[Any, ...] | None,
) -> BatchOptions:
    """Return kernel recording options after padding/observer lowering."""

    if not group.has_padding:
        return options
    if options.recording.mode == "none" and observers is not None:
        return options
    if row_recording_indices_for_group(group, options.recording) is not None:
        return options
    return options if options.recording.mode == "full" else _replace_full_recording(options)


def row_recording_indices_for_group(
    group: Any,
    recording: BatchRecording,
) -> np.ndarray | None:
    """Return row-aware retained Vm indices for padded batch groups.

    ``BatchRecording`` is shape-only and normally resolves indices against one
    ``Nx``. Padded mixed-diameter groups need the same number of retained
    columns per row, but those columns must be selected against each row's
    original compartment count.
    """

    if not getattr(group, "has_padding", False):
        return None
    if recording.mode not in {"center", "probes", "indices"}:
        return None

    rows: list[np.ndarray] = []
    width: int | None = None
    for item in group.items:
        row = np.asarray(
            recording.indices_for(int(item.solver_axon.n_compartments)),
            dtype=np.int32,
        )
        if row.ndim != 1:
            return None
        if width is None:
            width = int(row.shape[0])
        elif int(row.shape[0]) != width:
            return None
        rows.append(row)
    if not rows or width is None or width < 1:
        return None
    return np.stack(rows, axis=0)


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

    from axonscope.runtime.jax.observer_runtime import build_vm_raster_plan

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


def cohort_original_indices(cohort: Any) -> np.ndarray:
    """Return row-aware original compartment indices, with -1 for padding."""

    rows = np.full((cohort.size, cohort.nx), -1, dtype=np.int32)
    for row_index, solver_axon in enumerate(cohort.solver_axons):
        original_nx = int(solver_axon.n_compartments)
        rows[row_index, :original_nx] = np.arange(original_nx, dtype=np.int32)
    return rows


def _replace_full_recording(options: BatchOptions) -> BatchOptions:
    from dataclasses import replace

    return replace(options, recording=BatchRecording.full())


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
        tuple(_observer_definition_signature(observer) for observer in observers),
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
        tuple(_observer_definition_signature(observer) for observer in observers),
    )


def _observer_definition_signature(observer: Any) -> tuple[Any, ...]:
    signal = getattr(observer, "signal", None)
    signal_id = getattr(signal, "id", repr(signal))
    target = getattr(observer, "target", None)
    return (
        type(observer).__module__,
        type(observer).__qualname__,
        str(getattr(observer, "name", "")),
        str(signal_id),
        repr(target),
        _maybe_millivolt(getattr(observer, "threshold", None)),
        _maybe_millisecond(getattr(observer, "blanking", None)),
    )


def _maybe_millivolt(value: Any) -> float | None:
    if value is None:
        return None
    from axonscope.utils import units

    return float(units.to_mV(value))


def _maybe_millisecond(value: Any) -> float | None:
    if value is None:
        return None
    from axonscope.utils import units

    return float(units.to_ms(value))


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
    "cohort_original_indices",
    "lower_batch_recording_options",
    "lower_observers_for_cohort",
    "row_recording_indices_for_group",
]
