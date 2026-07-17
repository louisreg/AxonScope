"""Solver-side threshold observers.

One probe-and-threshold lowering supports bounded activation flags, first
crossing steps, and packed VmRaster retention. Higher-level velocity and
propagation analyses remain result-side concerns until they receive explicit
bounded contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from axonscope.benchmarking import benchmark_span, benchmark_wait
from axonscope.analysis.core import AnalysisResult, AnalysisStatus
from axonscope.analysis.definitions import (
    Activation,
    ConductionBlock,
    ConductionVelocity,
    Latency,
    SpikeCount,
    SpikeCountEvent,
    VmRaster,
)
from axonscope.positions import PositionSelector
from axonscope.results.vm_raster import VM_RASTER_OBSERVATION_KEY, VmRasterResult
from axonscope.runtime.jax.preparation.caches import (
    get_batched_static_array,
    store_batched_static_array,
)
from axonscope.signals import MEMBRANE_VOLTAGE, Signal
from axonscope.utils import units


ThresholdObserverState = Any
ObserverRetention = Literal[
    "activation",
    "first_crossing",
    "spike_summary",
    "spike_events",
    "vm_raster",
]


@dataclass(frozen=True)
class ThresholdObserverPlan:
    """Static probe/threshold plan with a bounded or raster retention policy."""

    definitions: tuple[Any, ...]
    names: tuple[str, ...]
    probe_indices: Any
    probe_mask: Any
    original_indices: Any
    positions_um: Any
    thresholds_mV: Any
    blanking_ms: Any
    reset_thresholds_mV: Any
    refractory_ms: Any
    temporal_stride: int
    max_spikes: int | None
    probe_indices_host: np.ndarray
    probe_mask_host: np.ndarray
    original_indices_host: np.ndarray
    positions_um_host: np.ndarray
    thresholds_mV_host: np.ndarray
    blanking_ms_host: np.ndarray
    retention: ObserverRetention = "vm_raster"
    row_aware: bool = False

    @property
    def definition_count(self) -> int:
        """Number of threshold/probe definitions carried by the plan."""

        return len(self.names)

    @property
    def probe_count(self) -> int:
        """Static number of probe slots per raster set."""

        return int(np.asarray(self.probe_mask).shape[-1])


@dataclass(frozen=True)
class PendingThresholdObservation:
    """Device-resident threshold output awaiting synchronization/finalization."""

    plan: ThresholdObserverPlan
    state: ThresholdObserverState
    nt: int
    dt_ms: float


def trim_pending_threshold_observation(
    pending: PendingThresholdObservation,
    *,
    batch_size: int,
) -> PendingThresholdObservation:
    """Drop backend-only padded rows from a pending threshold state."""

    return PendingThresholdObservation(
        plan=pending.plan,
        state=pending.state[: int(batch_size)],
        nt=pending.nt,
        dt_ms=pending.dt_ms,
    )


def _require_vm_signal(signal: Any) -> None:
    if not isinstance(signal, Signal) or signal.id != MEMBRANE_VOLTAGE.id:
        raise NotImplementedError("VmRaster observers support membrane voltage only.")


def _is_vm_raster_definition(definition: Any) -> bool:
    return isinstance(
        definition,
        (Activation, Latency, SpikeCount, VmRaster, ConductionBlock, ConductionVelocity),
    )


def _threshold_mV(definition: Any) -> float:
    threshold = getattr(definition, "threshold", None)
    return -20.0 if threshold is None else units.to_mV(threshold)


def _normalize_positions_and_indices(
    positions_um: Any,
    original_indices: Any | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    positions = np.asarray(positions_um, dtype=float)
    if positions.ndim not in {1, 2} or positions.size == 0:
        raise ValueError("VmRaster positions must be a non-empty 1D or 2D array.")

    if original_indices is None:
        if positions.ndim == 1:
            originals = np.arange(positions.shape[0], dtype=np.int32)
        else:
            originals = np.broadcast_to(
                np.arange(positions.shape[1], dtype=np.int32)[None, :],
                positions.shape,
            ).copy()
    else:
        originals = np.asarray(original_indices, dtype=np.int32)
        if positions.ndim == 2 and originals.ndim == 1:
            if originals.shape != (positions.shape[1],):
                raise ValueError("VmRaster original_indices must match positions.")
            originals = np.broadcast_to(originals[None, :], positions.shape).copy()
        elif originals.shape != positions.shape:
            raise ValueError("VmRaster original_indices must match positions.")

    return positions, originals, positions.ndim == 2


def _select_raster_probe_columns(
    target: PositionSelector,
    *,
    positions_um: np.ndarray,
    original_indices: np.ndarray,
) -> np.ndarray:
    valid_columns = np.flatnonzero(original_indices >= 0)
    if valid_columns.size == 0:
        raise ValueError("VmRaster row contains no valid probe positions.")
    selected_local = target.columns(
        positions_um=positions_um[valid_columns],
        original_indices=original_indices[valid_columns],
    )
    selected = valid_columns[np.asarray(selected_local, dtype=np.int32)]
    if selected.size == 0:
        raise ValueError("VmRaster target selects no positions.")
    return selected.astype(np.int32, copy=False)


def build_threshold_observer_plan(
    definitions: Any,
    *,
    positions_um: Any,
    original_indices: Any | None = None,
    dtype: Any = jnp.float32,
) -> ThresholdObserverPlan | None:
    """Lower threshold-style public observers to one packed VmRaster plan."""

    if definitions is None:
        return None
    raster_defs = tuple(definitions)
    if not raster_defs:
        return None

    positions, originals, row_aware = _normalize_positions_and_indices(
        positions_um,
        original_indices,
    )
    position_rows = positions if row_aware else positions[None, :]
    original_rows = originals if row_aware else originals[None, :]

    names: list[str] = []
    selected_by_row: list[list[np.ndarray]] = []
    thresholds: list[float] = []
    blanking_values: list[float] = []
    reset_thresholds: list[float] = []
    refractory_values: list[float] = []
    temporal_strides: list[int] = []
    spike_capacities: list[int | None] = []

    for definition in raster_defs:
        if not _is_vm_raster_definition(definition):
            raise NotImplementedError(
                "VmRaster observers currently support threshold-style Vm definitions only."
            )
        _require_vm_signal(getattr(definition, "signal", None))
        target = getattr(definition, "target", None)
        if not isinstance(target, PositionSelector):
            raise TypeError("VmRaster target must be an axonscope PositionSelector.")

        names.append(str(definition.name))
        thresholds.append(_threshold_mV(definition))
        blanking_values.append(units.to_ms(getattr(definition, "blanking", 0.0)))
        reset_thresholds.append(
            units.to_mV(getattr(definition, "reset_threshold", _threshold_mV(definition)))
        )
        refractory_values.append(units.to_ms(getattr(definition, "refractory", 0.0)))
        temporal_strides.append(int(getattr(definition, "every_n_steps", 1)))
        spike_capacities.append(getattr(definition, "max_spikes", None))
        selected_by_row.append(
            [
                _select_raster_probe_columns(
                    target,
                    positions_um=position_row,
                    original_indices=original_row,
                )
                for position_row, original_row in zip(position_rows, original_rows, strict=True)
            ]
        )

    if len(set(names)) != len(names):
        raise ValueError("VmRaster observer names must be unique.")

    width = max(int(selected.size) for rows in selected_by_row for selected in rows)
    if row_aware:
        index_table = np.zeros((position_rows.shape[0], len(raster_defs), width), dtype=np.int32)
        mask_table = np.zeros_like(index_table, dtype=bool)
        original_table = np.full_like(index_table, -1, dtype=np.int32)
        position_table = np.full(index_table.shape, np.nan, dtype=float)
        for raster_index, rows in enumerate(selected_by_row):
            for row_index, selected in enumerate(rows):
                count = int(selected.size)
                index_table[row_index, raster_index, :count] = selected
                mask_table[row_index, raster_index, :count] = True
                original_table[row_index, raster_index, :count] = original_rows[row_index, selected]
                position_table[row_index, raster_index, :count] = position_rows[row_index, selected]
    else:
        index_table = np.zeros((len(raster_defs), width), dtype=np.int32)
        mask_table = np.zeros_like(index_table, dtype=bool)
        original_table = np.full_like(index_table, -1, dtype=np.int32)
        position_table = np.full(index_table.shape, np.nan, dtype=float)
        for raster_index, rows in enumerate(selected_by_row):
            selected = rows[0]
            count = int(selected.size)
            index_table[raster_index, :count] = selected
            mask_table[raster_index, :count] = True
            original_table[raster_index, :count] = original_rows[0, selected]
            position_table[raster_index, :count] = position_rows[0, selected]

    thresholds_array = np.asarray(thresholds, dtype=float)
    blanking_array = np.asarray(blanking_values, dtype=float)
    reset_thresholds_array = np.asarray(reset_thresholds, dtype=float)
    refractory_array = np.asarray(refractory_values, dtype=float)
    temporal_stride = max(temporal_strides, default=1)
    if any(value != temporal_stride for value in temporal_strides):
        raise ValueError("VmRaster definitions in one plan must share every_n_steps.")
    if temporal_stride != 1 and not all(isinstance(value, VmRaster) for value in raster_defs):
        raise ValueError(
            "downsampling is only supported by explicit VmRaster definitions; "
            "exact analyses retain their dedicated compact state."
        )
    if all(isinstance(value, Activation) for value in raster_defs):
        retention: ObserverRetention = "activation"
    elif all(isinstance(value, Latency) for value in raster_defs):
        retention = "first_crossing"
    elif all(isinstance(value, SpikeCount) for value in raster_defs):
        capacity = spike_capacities[0]
        if any(value != capacity for value in spike_capacities):
            raise ValueError("SpikeCount definitions in one plan must share max_spikes.")
        retention = "spike_events" if capacity is not None else "spike_summary"
    else:
        retention = "vm_raster"
    return ThresholdObserverPlan(
        definitions=raster_defs,
        names=tuple(names),
        probe_indices=jnp.asarray(index_table, dtype=jnp.int32),
        probe_mask=jnp.asarray(mask_table, dtype=bool),
        original_indices=jnp.asarray(original_table, dtype=jnp.int32),
        positions_um=jnp.asarray(position_table, dtype=dtype),
        thresholds_mV=jnp.asarray(thresholds_array, dtype=dtype),
        blanking_ms=jnp.asarray(blanking_array, dtype=dtype),
        reset_thresholds_mV=jnp.asarray(reset_thresholds_array, dtype=dtype),
        refractory_ms=jnp.asarray(refractory_array, dtype=dtype),
        temporal_stride=int(temporal_stride),
        max_spikes=(None if retention != "spike_events" else int(spike_capacities[0])),
        probe_indices_host=_readonly_np_array(index_table, dtype=np.int32),
        probe_mask_host=_readonly_np_array(mask_table, dtype=bool),
        original_indices_host=_readonly_np_array(original_table, dtype=np.int32),
        positions_um_host=_readonly_np_array(position_table, dtype=float),
        thresholds_mV_host=_readonly_np_array(thresholds_array, dtype=float),
        blanking_ms_host=_readonly_np_array(blanking_array, dtype=float),
        retention=retention,
        row_aware=row_aware,
    )


def _readonly_np_array(values: Any, *, dtype: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    arr.setflags(write=False)
    return arr


def vm_raster_any_active_jax(
    row_words: Any,
    word_mask: Any,
    probe_mask: Any,
) -> Any:
    """Return row-wise VmRaster activity without materializing packed words."""

    word_mask_jax = jnp.asarray(word_mask, dtype=jnp.uint32)
    probe_mask_jax = jnp.asarray(probe_mask, dtype=bool)
    any_active_all_probes, any_active_masked = _vm_raster_any_active_jax_kernels()
    if bool(np.all(probe_mask)):
        return any_active_all_probes(row_words, word_mask_jax)
    return any_active_masked(row_words, word_mask_jax, probe_mask_jax)


@lru_cache(maxsize=1)
def _vm_raster_any_active_jax_kernels() -> tuple[Any, Any]:
    @jax.jit
    def any_active_all_probes(row_words: Any, word_mask: Any) -> Any:
        active_words = (row_words & word_mask[None, None, :]) != 0
        return jnp.any(active_words, axis=(1, 2))

    @jax.jit
    def any_active_masked(row_words: Any, word_mask: Any, probe_mask: Any) -> Any:
        active_words = (row_words & word_mask[None, None, :]) != 0
        return jnp.any(active_words & probe_mask[:, :, None], axis=(1, 2))

    return any_active_all_probes, any_active_masked


def init_threshold_observer_state(
    plan: ThresholdObserverPlan,
    *,
    batch_size: int,
    nt: int,
) -> ThresholdObserverState:
    """Return zeroed packed words carried by solver scans."""

    if plan.retention == "activation":
        shape = (int(batch_size), int(plan.definition_count))
        dtype = jnp.bool_
        fill_value = False
    elif plan.retention == "first_crossing":
        shape = (int(batch_size), int(plan.definition_count))
        dtype = jnp.int32
        fill_value = -1
    elif plan.retention in {"spike_summary", "spike_events"}:
        state_width = 4 if plan.max_spikes is None else 5 + int(plan.max_spikes)
        shape = (
            int(batch_size),
            int(plan.definition_count),
            int(plan.probe_count),
            state_width,
        )
        dtype = jnp.int32
        fill_value = None
    else:
        sampled_nt = (int(nt) + int(plan.temporal_stride) - 1) // int(
            plan.temporal_stride
        )
        word_count = (sampled_nt + 31) // 32
        shape = (
            int(batch_size),
            int(plan.definition_count),
            int(plan.probe_count),
            int(word_count),
        )
        dtype = jnp.uint32
        fill_value = 0
    key = (
        "threshold_observer_state_zeros_v2",
        plan.retention,
        shape,
        _current_jax_device_key(),
    )
    cached = get_batched_static_array(key)
    if cached is not None:
        return cached
    if plan.retention in {"spike_summary", "spike_events"}:
        out = jnp.zeros(shape, dtype=dtype)
        out = out.at[..., 1:3].set(-1)
        out = out.at[..., 3].set(1)
        if plan.retention == "spike_events":
            out = out.at[..., 5:].set(-1)
    else:
        out = jnp.full(shape, fill_value, dtype=dtype)
    store_batched_static_array(key, out)
    return out


def trim_threshold_observer_state(
    state: ThresholdObserverState,
    *,
    nt: int,
    retention: ObserverRetention = "vm_raster",
    temporal_stride: int = 1,
) -> ThresholdObserverState:
    """Trim packed observer storage and clear bits beyond the real duration."""

    if retention in {"activation", "first_crossing", "spike_summary", "spike_events"}:
        return state
    sampled_nt = (int(nt) + int(temporal_stride) - 1) // int(temporal_stride)
    word_count = (sampled_nt + 31) // 32
    trimmed = jnp.asarray(state)[..., :word_count]
    tail_bits = sampled_nt & 31
    if tail_bits and word_count:
        tail_mask = jnp.asarray((1 << tail_bits) - 1, dtype=jnp.uint32)
        trimmed = trimmed.at[..., -1].set(trimmed[..., -1] & tail_mask)
    return trimmed


def _current_jax_device_key() -> tuple[Any, ...]:
    device = getattr(jax.config, "jax_default_device", None)
    if device is None:
        try:
            devices = jax.devices(jax.default_backend())
        except Exception:
            devices = ()
        device = devices[0] if devices else None
    if device is None:
        return ("backend", jax.default_backend())
    return (
        "device",
        getattr(device, "platform", None),
        getattr(device, "id", None),
    )


def combine_threshold_observer_chunk_states(
    states: Sequence[ThresholdObserverState],
    *,
    starts: Sequence[int],
    lengths: Sequence[int] | None = None,
    nt: int,
    retention: ObserverRetention = "vm_raster",
) -> ThresholdObserverState:
    """Pack local chunk rasters back into one full-duration raster state."""

    if not states:
        raise ValueError("at least one VmRaster chunk state is required.")
    if len(states) != len(starts):
        raise ValueError("VmRaster chunk states and starts must have the same length.")
    if lengths is not None and len(states) != len(lengths):
        raise ValueError("VmRaster chunk states and lengths must have the same length.")

    if retention == "activation":
        return jnp.any(jnp.stack(tuple(jnp.asarray(state) for state in states)), axis=0)
    if retention == "first_crossing":
        candidates = []
        local_lengths = lengths if lengths is not None else (None,) * len(states)
        for state, start, length in zip(states, starts, local_lengths, strict=True):
            local = jnp.asarray(state, dtype=jnp.int32)
            valid = local >= 0
            if length is not None:
                valid = valid & (local < int(length))
            candidates.append(jnp.where(valid, local + int(start), -1))
        stacked = jnp.stack(tuple(candidates), axis=0)
        sentinel = jnp.asarray(int(nt), dtype=jnp.int32)
        earliest = jnp.min(jnp.where(stacked >= 0, stacked, sentinel), axis=0)
        return jnp.where(earliest < sentinel, earliest, -1)
    if retention in {"spike_summary", "spike_events"}:
        raise ValueError("spike-summary state must remain continuous across chunks")

    if lengths is None:
        chunk_lengths = [None] * len(states)
    else:
        chunk_lengths = [int(length) for length in lengths]
        if any(length < 0 for length in chunk_lengths):
            raise ValueError("VmRaster chunk lengths must be non-negative.")

    word_count = (int(nt) + 31) // 32

    if len(states) == 1 and int(starts[0]) == 0:
        only = jnp.asarray(states[0])
        if only.ndim < 1:
            raise ValueError("VmRaster chunk state must have at least one word axis.")
        local_capacity = int(only.shape[-1]) * 32
        local_count = min(
            local_capacity if chunk_lengths[0] is None else int(chunk_lengths[0]),
            local_capacity,
            int(nt),
        )
        if local_count >= int(nt) and int(only.shape[-1]) == word_count:
            return only

    with benchmark_span(
        "kernel.wait",
        observer="vm_raster",
        wait_scope="chunk_states",
        chunk_count=len(states),
        nt=int(nt),
    ):
        benchmark_wait(tuple(states))

    first = np.asarray(states[0], dtype=np.uint32)
    if first.ndim < 1:
        raise ValueError("VmRaster chunk state must have at least one word axis.")

    combined = np.zeros(first.shape[:-1] + (word_count,), dtype=np.uint32)
    static_shape = first.shape[:-1]

    for state, start, length in zip(states, starts, chunk_lengths, strict=True):
        chunk = np.asarray(state, dtype=np.uint32)
        if chunk.shape[:-1] != static_shape:
            raise ValueError("VmRaster chunk states must share static axes.")
        start_index = int(start)
        if start_index < 0:
            raise ValueError("VmRaster chunk start indices must be non-negative.")
        local_capacity = int(chunk.shape[-1]) * 32
        local_count = min(
            local_capacity if length is None else int(length),
            local_capacity,
            int(nt) - start_index,
        )
        if local_count <= 0:
            continue
        usable_words = min((local_count + 31) // 32, int(chunk.shape[-1]))
        if usable_words <= 0:
            continue
        values = chunk[..., :usable_words]
        valid_bits = local_count - (usable_words - 1) * 32
        if valid_bits < 32:
            values = values.copy()
            values[..., -1] &= np.uint32((1 << valid_bits) - 1)

        global_word = start_index // 32
        if global_word >= word_count:
            continue
        offset = start_index & 31
        low_words = min(usable_words, word_count - global_word)
        if low_words <= 0:
            continue

        if offset == 0:
            combined[..., global_word : global_word + low_words] |= values[
                ..., :low_words
            ]
            continue

        combined[..., global_word : global_word + low_words] |= (
            values[..., :low_words] << np.uint32(offset)
        )
        high_word = global_word + 1
        if high_word < word_count:
            high_words = min(usable_words, word_count - high_word)
            if high_words > 0:
                combined[..., high_word : high_word + high_words] |= (
                    values[..., :high_words] >> np.uint32(32 - offset)
                )

    return combined


def update_threshold_observer_state_batch_from_tables(
    state: ThresholdObserverState,
    *,
    vm_mV: Any,
    step_index: Any,
    probe_indices: Any,
    probe_mask: Any,
    thresholds_mV: Any,
    blanking_ms: Any = 0.0,
    reset_thresholds_mV: Any | None = None,
    refractory_ms: Any = 0.0,
    temporal_stride: int = 1,
    dt_ms: Any = 1.0,
    retention: ObserverRetention = "vm_raster",
) -> ThresholdObserverState:
    """Pack one batched Vm time step from pre-batched probe tables."""

    vm = jnp.asarray(vm_mV)
    indices = jnp.asarray(probe_indices)
    mask = jnp.asarray(probe_mask)
    if vm.ndim != 2:
        raise ValueError(f"VmRaster update expects Vm[B, Nx], got {vm.shape}.")
    if indices.ndim != 3 or mask.shape != indices.shape:
        raise ValueError("VmRaster probe tables must have shape (B, R, P).")

    selected = jnp.take_along_axis(vm[:, None, :], indices, axis=2)
    hit = mask & (selected >= jnp.asarray(thresholds_mV)[None, :, None])
    if retention in {"activation", "first_crossing"}:
        time_ms = (jnp.asarray(step_index) + 1) * jnp.asarray(dt_ms)
        after_blanking = time_ms >= jnp.asarray(blanking_ms)[None, :, None]
        crossed = jnp.any(hit & after_blanking, axis=-1)
        if retention == "activation":
            return jnp.asarray(state, dtype=bool) | crossed
        current = jnp.asarray(state, dtype=jnp.int32)
        step = jnp.asarray(step_index, dtype=jnp.int32)
        return jnp.where((current < 0) & crossed, step, current)
    if retention in {"spike_summary", "spike_events"}:
        return _update_spike_summary_state(
            state,
            selected=selected,
            step_index=step_index,
            probe_mask=mask,
            thresholds_mV=jnp.asarray(thresholds_mV)[None, :, None],
            reset_thresholds_mV=jnp.asarray(reset_thresholds_mV)[None, :, None],
            blanking_ms=jnp.asarray(blanking_ms)[None, :, None],
            refractory_ms=jnp.asarray(refractory_ms)[None, :, None],
            dt_ms=dt_ms,
        )
    raster_step = jnp.asarray(step_index) // jnp.asarray(
        temporal_stride,
        dtype=jnp.asarray(step_index).dtype,
    )
    return _write_raster_bits(state, hit, raster_step)


def update_threshold_observer_state_scalar_from_tables(
    state: ThresholdObserverState,
    *,
    vm_mV: Any,
    step_index: Any,
    probe_indices: Any,
    probe_mask: Any,
    thresholds_mV: Any,
    blanking_ms: Any = 0.0,
    reset_thresholds_mV: Any | None = None,
    refractory_ms: Any = 0.0,
    temporal_stride: int = 1,
    dt_ms: Any = 1.0,
    retention: ObserverRetention = "vm_raster",
) -> ThresholdObserverState:
    """Pack one scalar Vm time step from static probe tables."""

    vm = jnp.asarray(vm_mV)
    indices = jnp.asarray(probe_indices)
    mask = jnp.asarray(probe_mask)
    if vm.ndim != 1:
        raise ValueError(f"VmRaster scalar update expects Vm[Nx], got {vm.shape}.")
    if indices.ndim != 2 or mask.shape != indices.shape:
        raise ValueError("VmRaster scalar probe tables must have shape (R, P).")

    selected = jnp.take(vm, indices, axis=0)
    hit = mask & (selected >= jnp.asarray(thresholds_mV)[:, None])
    if retention in {"activation", "first_crossing"}:
        time_ms = (jnp.asarray(step_index) + 1) * jnp.asarray(dt_ms)
        after_blanking = time_ms >= jnp.asarray(blanking_ms)[:, None]
        crossed = jnp.any(hit & after_blanking, axis=-1)
        if retention == "activation":
            return jnp.asarray(state, dtype=bool) | crossed
        current = jnp.asarray(state, dtype=jnp.int32)
        step = jnp.asarray(step_index, dtype=jnp.int32)
        return jnp.where((current < 0) & crossed, step, current)
    if retention in {"spike_summary", "spike_events"}:
        return _update_spike_summary_state(
            state,
            selected=selected,
            step_index=step_index,
            probe_mask=mask,
            thresholds_mV=jnp.asarray(thresholds_mV)[:, None],
            reset_thresholds_mV=jnp.asarray(reset_thresholds_mV)[:, None],
            blanking_ms=jnp.asarray(blanking_ms)[:, None],
            refractory_ms=jnp.asarray(refractory_ms)[:, None],
            dt_ms=dt_ms,
        )
    raster_step = jnp.asarray(step_index) // jnp.asarray(
        temporal_stride,
        dtype=jnp.asarray(step_index).dtype,
    )
    return _write_raster_bits(state, hit, raster_step)


def _write_raster_bits(words: Any, hit: Any, step_index: Any) -> Any:
    step = jnp.asarray(step_index, dtype=jnp.int32)
    word_index = step // jnp.asarray(32, dtype=jnp.int32)
    bit_index = step & jnp.asarray(31, dtype=jnp.int32)
    bit = jnp.left_shift(jnp.asarray(1, dtype=jnp.uint32), bit_index.astype(jnp.uint32))
    bits = jnp.asarray(hit, dtype=jnp.uint32) * bit
    current = jnp.take(words, word_index, axis=-1)
    updated = current | bits
    zero = jnp.asarray(0, dtype=word_index.dtype)
    starts = (zero,) * (words.ndim - 1) + (word_index,)
    return jax.lax.dynamic_update_slice(words, updated[..., None], starts)


def _update_spike_summary_state(
    state: Any,
    *,
    selected: Any,
    step_index: Any,
    probe_mask: Any,
    thresholds_mV: Any,
    reset_thresholds_mV: Any,
    blanking_ms: Any,
    refractory_ms: Any,
    dt_ms: Any,
) -> Any:
    current = jnp.asarray(state, dtype=jnp.int32)
    step = jnp.asarray(step_index, dtype=jnp.int32)
    count = current[..., 0]
    first = current[..., 1]
    last = current[..., 2]
    armed = current[..., 3] != 0
    mask = jnp.asarray(probe_mask, dtype=bool)
    hit = mask & (selected >= thresholds_mV)
    crossing = armed & hit
    time_ms = (step + 1) * jnp.asarray(dt_ms)
    refractory_ready = (last < 0) | (
        (step - last) * jnp.asarray(dt_ms) >= refractory_ms
    )
    fire = crossing & (time_ms >= blanking_ms) & refractory_ready
    reset = mask & (selected <= reset_thresholds_mV)
    summary = (
        count + fire.astype(jnp.int32),
        jnp.where((first < 0) & fire, step, first),
        jnp.where(fire, step, last),
        ((armed & ~crossing) | reset).astype(jnp.int32),
    )
    if current.shape[-1] == 4:
        return jnp.stack(summary, axis=-1)

    overflow = current[..., 4] != 0
    event_steps = current[..., 5:]
    slots = jnp.arange(event_steps.shape[-1], dtype=jnp.int32)
    write = fire[..., None] & (count[..., None] == slots)
    updated_steps = jnp.where(write, step, event_steps)
    updated_overflow = overflow | (fire & (count >= event_steps.shape[-1]))
    return jnp.concatenate(
        (
            jnp.stack(summary, axis=-1),
            updated_overflow[..., None].astype(jnp.int32),
            updated_steps,
        ),
        axis=-1,
    )


def finalize_threshold_observer_state(
    plan: ThresholdObserverPlan,
    state: ThresholdObserverState,
    *,
    nt: int,
    dt_ms: float,
    synchronize: bool = True,
    materialize_words: bool = True,
) -> dict[str, object]:
    """Finalize bounded activation or packed VmRaster output."""

    if synchronize:
        with benchmark_span(
            "kernel.wait",
            observer=plan.retention,
            wait_scope="observer_state",
            observer_definition_count=plan.definition_count,
            probe_count=plan.probe_count,
            nt=int(nt),
            row_aware=plan.row_aware,
        ):
            benchmark_wait(state)

    if plan.retention == "activation":
        values = np.asarray(state, dtype=bool)
        statuses = (AnalysisStatus.VALID,) * int(values.shape[0])
        return {
            definition.name: AnalysisResult(
                name=definition.name,
                values=values[:, index],
                statuses=statuses,
                definition=definition,
            )
            for index, definition in enumerate(plan.definitions)
        }

    if plan.retention == "first_crossing":
        steps = np.asarray(state, dtype=np.int32)
        return {
            definition.name: _finalize_first_crossing_result(
                definition,
                steps[:, index],
                dt_ms=float(dt_ms),
            )
            for index, definition in enumerate(plan.definitions)
        }

    if plan.retention in {"spike_summary", "spike_events"}:
        values = np.asarray(state, dtype=np.int32)
        probe_mask = np.asarray(plan.probe_mask_host, dtype=bool)
        if probe_mask.ndim == 2:
            probe_mask = np.broadcast_to(probe_mask[None, ...], values.shape[:3])
        return {
            definition.name: _finalize_spike_summary_result(
                definition,
                values[:, index],
                probe_mask[:, index],
                dt_ms=float(dt_ms),
                nt=int(nt),
            )
            for index, definition in enumerate(plan.definitions)
        }

    span_name = (
        "kernel.finalize_observer.to_host"
        if bool(materialize_words)
        else "kernel.finalize_observer.package"
    )
    with benchmark_span(
        span_name,
        observer="vm_raster",
        raster_count=plan.definition_count,
        probe_count=plan.probe_count,
        nt=int(nt),
        row_aware=plan.row_aware,
        materialize_words=bool(materialize_words),
        synchronized_before_finalize=not synchronize,
    ):
        words = np.asarray(state, dtype=np.uint32) if materialize_words else state
        probe_indices = plan.probe_indices_host
        probe_mask = plan.probe_mask_host
        original_indices = plan.original_indices_host
        positions_um = plan.positions_um_host
        thresholds_mV = plan.thresholds_mV_host

    sampled_nt = (int(nt) + int(plan.temporal_stride) - 1) // int(
        plan.temporal_stride
    )
    return {
        VM_RASTER_OBSERVATION_KEY: VmRasterResult(
            words=words,
            nt=sampled_nt,
            dt_ms=float(dt_ms) * int(plan.temporal_stride),
            definitions=plan.definitions,
            names=plan.names,
            probe_indices=probe_indices,
            probe_mask=probe_mask,
            original_indices=original_indices,
            positions_um=positions_um,
            thresholds_mV=thresholds_mV,
            row_aware=plan.row_aware,
            _any_active_impl=None if materialize_words else vm_raster_any_active_jax,
        )
    }


def _finalize_first_crossing_result(
    definition: Any,
    steps: np.ndarray,
    *,
    dt_ms: float,
) -> AnalysisResult:
    crossed = steps >= 0
    values = np.where(crossed, (steps.astype(float) + 1.0) * dt_ms, np.nan)
    statuses = tuple(
        AnalysisStatus.VALID if value else AnalysisStatus.UNDETERMINED
        for value in crossed
    )
    messages = tuple(
        "" if value else "threshold was not crossed at the requested target."
        for value in crossed
    )
    return AnalysisResult(
        name=definition.name,
        values=values,
        statuses=statuses,
        messages=messages,
        unit="millisecond",
        definition=definition,
    )


def _finalize_spike_summary_result(
    definition: Any,
    state: np.ndarray,
    probe_mask: np.ndarray,
    *,
    dt_ms: float,
    nt: int,
) -> AnalysisResult:
    counts = np.sum(np.where(probe_mask, state[..., 0], 0), axis=-1, dtype=np.int64)
    first_steps = np.where(probe_mask & (state[..., 1] >= 0), state[..., 1], nt)
    last_steps = np.where(probe_mask, state[..., 2], -1)
    first = np.min(first_steps, axis=-1)
    last = np.max(last_steps, axis=-1)
    has_bounded_events = state.shape[-1] > 4
    events = tuple(
        SpikeCountEvent(
            count=int(count),
            first_time_ms=None if first_step >= nt else (float(first_step) + 1.0) * dt_ms,
            last_time_ms=None if last_step < 0 else (float(last_step) + 1.0) * dt_ms,
            probe_counts=tuple(
                int(value)
                for value, valid in zip(row_state[:, 0], row_mask, strict=True)
                if valid
            ),
            spike_times_ms=(
                tuple(
                    tuple((float(step) + 1.0) * dt_ms for step in probe_steps if step >= 0)
                    for probe_steps, valid in zip(
                        row_state[:, 5:], row_mask, strict=True
                    )
                    if valid
                )
                if has_bounded_events
                else ()
            ),
            overflow=(
                tuple(
                    bool(value)
                    for value, valid in zip(row_state[:, 4], row_mask, strict=True)
                    if valid
                )
                if has_bounded_events
                else ()
            ),
        )
        for count, first_step, last_step, row_state, row_mask in zip(
            counts, first, last, state, probe_mask, strict=True
        )
    )
    return AnalysisResult(
        name=definition.name,
        values=counts,
        statuses=(AnalysisStatus.VALID,) * int(counts.shape[0]),
        definition=definition,
        events=events,
    )


__all__ = [
    "PendingThresholdObservation",
    "ThresholdObserverPlan",
    "ThresholdObserverState",
    "build_threshold_observer_plan",
    "combine_threshold_observer_chunk_states",
    "finalize_threshold_observer_state",
    "init_threshold_observer_state",
    "trim_threshold_observer_state",
    "trim_pending_threshold_observation",
    "update_threshold_observer_state_batch_from_tables",
    "update_threshold_observer_state_scalar_from_tables",
]
