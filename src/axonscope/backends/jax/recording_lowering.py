"""Recording and observer lowering contracts for JAX batch execution."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import numpy as np

from axonscope.analysis.definitions import Activation, ConductionBlock, Latency
from axonscope.benchmarking.hotpaths import record_benchmark_metadata
from axonscope.solvers.options import BatchOptions, BatchRecording


_VM_RASTER_PLAN_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
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
    return options if options.recording.mode == "full" else _replace_full_recording(options)


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
    cache_key = _vm_raster_plan_cache_key(
        observers,
        cohort=cohort,
        dtype=dtype,
    )
    cached = _cache_get(_VM_RASTER_PLAN_CACHE, cache_key)
    if cached is not None:
        record_benchmark_metadata(vm_raster_plan_cache="hit")
        return cached

    from axonscope.backends.jax.observer_runtime import build_vm_raster_plan

    row_positions_um = np.asarray(cohort.x_positions_m, dtype=float) * 1e6
    plan = build_vm_raster_plan(
        observers,
        positions_um=row_positions_um,
        original_indices=cohort_original_indices(cohort),
        dtype=dtype,
    )
    _cache_store(_VM_RASTER_PLAN_CACHE, cache_key, plan)
    record_benchmark_metadata(
        vm_raster_plan_cache="miss",
        vm_raster_count=0 if plan is None else plan.raster_count,
        vm_raster_probe_count=0 if plan is None else plan.probe_count,
    )
    return plan


def observer_output_label(
    observers: tuple[Any, ...] | None,
    *,
    recording_mode: str,
) -> str:
    """Return the public output route selected for observer definitions."""

    if observers is None:
        return "none"
    if recording_mode == "none" and observers_are_vm_raster_compatible(observers):
        return "vm_raster"
    if recording_mode == "none":
        return "unsupported_observer_only"
    return "posthoc_from_recorded_vm"


def observers_are_vm_raster_compatible(observers: tuple[Any, ...] | None) -> bool:
    """Return whether all observer definitions can be lowered to VmRaster."""

    if observers is None:
        return False
    return bool(observers) and len(vm_raster_definitions(observers)) == len(observers)


def vm_raster_definitions(observers: tuple[Any, ...] | None) -> tuple[Any, ...]:
    """Return observer definitions supported by solver-side VmRaster."""

    if observers is None:
        return ()
    return tuple(
        observer
        for observer in observers
        if isinstance(observer, (Activation, Latency, ConductionBlock))
    )


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
        tuple(tuple(id(stimulation) for stimulation in row) for row in cohort.stimulations),
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


__all__ = [
    "cohort_original_indices",
    "lower_batch_recording_options",
    "lower_observers_for_cohort",
    "observer_output_label",
    "observers_are_vm_raster_compatible",
    "vm_raster_definitions",
]
