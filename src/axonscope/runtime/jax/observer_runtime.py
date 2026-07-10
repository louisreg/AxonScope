"""Solver-side Vm rasterization.

The solver-side observer path intentionally does one small, fixed operation:
threshold selected membrane-voltage probes at each time step and pack the
boolean raster into ``uint32`` words. Higher-level activation, latency, or
threshold-search analyses are post-processing concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from axonscope.benchmarking import benchmark_span, benchmark_wait
from axonscope.analysis.definitions import Activation, ConductionBlock, Latency
from axonscope.positions import PositionSelector
from axonscope.results.vm_raster import VM_RASTER_OBSERVATION_KEY, VmRasterResult
from axonscope.signals import MEMBRANE_VOLTAGE, Signal
from axonscope.utils import units


VmRasterState = Any


@dataclass(frozen=True)
class VmRasterPlan:
    """Static probe/threshold plan for packed Vm rasterization."""

    definitions: tuple[Any, ...]
    names: tuple[str, ...]
    probe_indices: Any
    probe_mask: Any
    original_indices: Any
    positions_um: Any
    thresholds_mV: Any
    row_aware: bool = False

    @property
    def raster_count(self) -> int:
        """Number of threshold/probe sets carried in the packed raster."""

        return len(self.names)

    @property
    def probe_count(self) -> int:
        """Static number of probe slots per raster set."""

        return int(np.asarray(self.probe_mask).shape[-1])


def _require_vm_signal(signal: Any) -> None:
    if not isinstance(signal, Signal) or signal.id != MEMBRANE_VOLTAGE.id:
        raise NotImplementedError("VmRaster observers support membrane voltage only.")


def _is_vm_raster_definition(definition: Any) -> bool:
    return isinstance(definition, (Activation, Latency, ConductionBlock))


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


def build_vm_raster_plan(
    definitions: Any,
    *,
    positions_um: Any,
    original_indices: Any | None = None,
    dtype: Any = jnp.float32,
) -> VmRasterPlan | None:
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
        thresholds.append(units.to_mV(definition.threshold))
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

    return VmRasterPlan(
        definitions=raster_defs,
        names=tuple(names),
        probe_indices=jnp.asarray(index_table, dtype=jnp.int32),
        probe_mask=jnp.asarray(mask_table, dtype=bool),
        original_indices=jnp.asarray(original_table, dtype=jnp.int32),
        positions_um=jnp.asarray(position_table, dtype=dtype),
        thresholds_mV=jnp.asarray(thresholds, dtype=dtype),
        row_aware=row_aware,
    )


def init_vm_raster_state(
    plan: VmRasterPlan,
    *,
    batch_size: int,
    nt: int,
) -> VmRasterState:
    """Return zeroed packed words carried by solver scans."""

    word_count = (int(nt) + 31) // 32
    shape = (
        int(batch_size),
        int(plan.raster_count),
        int(plan.probe_count),
        int(word_count),
    )
    return jnp.zeros(shape, dtype=jnp.uint32)


def combine_vm_raster_chunk_states(
    states: Sequence[VmRasterState],
    *,
    starts: Sequence[int],
    lengths: Sequence[int] | None = None,
    nt: int,
) -> VmRasterState:
    """Pack local chunk rasters back into one full-duration raster state."""

    if not states:
        raise ValueError("at least one VmRaster chunk state is required.")
    if len(states) != len(starts):
        raise ValueError("VmRaster chunk states and starts must have the same length.")
    if lengths is not None and len(states) != len(lengths):
        raise ValueError("VmRaster chunk states and lengths must have the same length.")

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


def _raster_probe_tables_for_batch(
    plan: VmRasterPlan,
    batch_size: int,
) -> tuple[Any, Any]:
    indices = jnp.asarray(plan.probe_indices)
    mask = jnp.asarray(plan.probe_mask)
    if indices.ndim == 2:
        target_shape = (int(batch_size),) + tuple(indices.shape)
        return (
            jnp.broadcast_to(indices[None, :, :], target_shape),
            jnp.broadcast_to(mask[None, :, :], target_shape),
        )
    if indices.ndim == 3:
        return indices, mask
    raise ValueError("VmRaster probe tables must be 2D or 3D.")


def update_vm_raster_state_batch(
    state: VmRasterState,
    *,
    vm_mV: Any,
    step_index: Any,
    plan: VmRasterPlan,
) -> VmRasterState:
    """Pack one batched Vm time step into the raster state."""

    vm = jnp.asarray(vm_mV)
    if vm.ndim != 2:
        raise ValueError(f"VmRaster update expects Vm[B, Nx], got {vm.shape}.")
    batch_size = int(vm.shape[0])
    indices, mask = _raster_probe_tables_for_batch(plan, batch_size)
    return update_vm_raster_state_batch_from_tables(
        state,
        vm_mV=vm,
        step_index=step_index,
        probe_indices=indices,
        probe_mask=mask,
        thresholds_mV=plan.thresholds_mV,
    )


def update_vm_raster_state_batch_from_tables(
    state: VmRasterState,
    *,
    vm_mV: Any,
    step_index: Any,
    probe_indices: Any,
    probe_mask: Any,
    thresholds_mV: Any,
) -> VmRasterState:
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
    return _write_raster_bits(state, hit, step_index)


def update_vm_raster_state_scalar_from_tables(
    state: VmRasterState,
    *,
    vm_mV: Any,
    step_index: Any,
    probe_indices: Any,
    probe_mask: Any,
    thresholds_mV: Any,
) -> VmRasterState:
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
    return _write_raster_bits(state, hit, step_index)


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


def finalize_vm_raster_state(
    plan: VmRasterPlan,
    state: VmRasterState,
    *,
    nt: int,
    dt_ms: float,
) -> dict[str, VmRasterResult]:
    """Package packed raster words as the single solver-side observation."""

    with benchmark_span(
        "kernel.wait",
        observer="vm_raster",
        wait_scope="observer_state",
        raster_count=plan.raster_count,
        probe_count=plan.probe_count,
        nt=int(nt),
        row_aware=plan.row_aware,
    ):
        benchmark_wait(state)

    with benchmark_span(
        "kernel.finalize_observer.to_host",
        observer="vm_raster",
        raster_count=plan.raster_count,
        probe_count=plan.probe_count,
        nt=int(nt),
        row_aware=plan.row_aware,
    ):
        words = np.asarray(state, dtype=np.uint32)
        probe_indices = np.asarray(plan.probe_indices)
        probe_mask = np.asarray(plan.probe_mask, dtype=bool)
        original_indices = np.asarray(plan.original_indices, dtype=np.int32)
        positions_um = np.asarray(plan.positions_um, dtype=float)
        thresholds_mV = np.asarray(plan.thresholds_mV, dtype=float)

    return {
        VM_RASTER_OBSERVATION_KEY: VmRasterResult(
            words=words,
            nt=int(nt),
            dt_ms=float(dt_ms),
            definitions=plan.definitions,
            names=plan.names,
            probe_indices=probe_indices,
            probe_mask=probe_mask,
            original_indices=original_indices,
            positions_um=positions_um,
            thresholds_mV=thresholds_mV,
            row_aware=plan.row_aware,
        )
    }


__all__ = [
    "VmRasterPlan",
    "VmRasterState",
    "build_vm_raster_plan",
    "combine_vm_raster_chunk_states",
    "finalize_vm_raster_state",
    "init_vm_raster_state",
    "update_vm_raster_state_batch",
    "update_vm_raster_state_batch_from_tables",
    "update_vm_raster_state_scalar_from_tables",
]
