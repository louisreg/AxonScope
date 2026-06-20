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

from axonscope.analysis.definitions import Activation, ConductionBlock, Latency
from axonscope.positions import PositionSelector
from axonscope.signals import MEMBRANE_VOLTAGE, Signal
from axonscope.utils import units


VM_RASTER_OBSERVATION_KEY = "vm_raster"
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


@dataclass(frozen=True)
class VmRasterResult:
    """Packed solver-side membrane-voltage raster.

    ``words`` has shape ``(B, R, P, W)`` where ``B`` is the run batch,
    ``R`` the number of requested raster definitions, ``P`` the fixed probe
    slots per definition, and ``W = ceil(nt / 32)``. Bit ``t % 32`` in word
    ``t // 32`` is true when ``Vm[t, probe] >= threshold``.
    """

    words: Any
    nt: int
    dt_ms: float
    definitions: tuple[Any, ...]
    names: tuple[str, ...]
    probe_indices: Any
    probe_mask: Any
    original_indices: Any
    positions_um: Any
    thresholds_mV: Any
    row_aware: bool = False

    @property
    def values(self) -> Any:
        """Alias for ``words`` for generic wait/metadata helpers."""

        return self.words

    @property
    def batch_size(self) -> int:
        return int(np.asarray(self.words).shape[0])

    @property
    def raster_count(self) -> int:
        return int(np.asarray(self.words).shape[1])

    @property
    def probe_count(self) -> int:
        return int(np.asarray(self.words).shape[2])

    @property
    def word_count(self) -> int:
        return int(np.asarray(self.words).shape[3])

    @property
    def packed_nbytes(self) -> int:
        return int(np.asarray(self.words).nbytes)

    def slice_batch(self, row: int) -> "VmRasterResult":
        """Return a one-row raster view with the batch axis preserved."""

        row = int(row)
        return VmRasterResult(
            words=np.asarray(self.words)[row : row + 1],
            nt=self.nt,
            dt_ms=self.dt_ms,
            definitions=self.definitions,
            names=self.names,
            probe_indices=_slice_row_aware_metadata(self.probe_indices, row, self.row_aware),
            probe_mask=_slice_row_aware_metadata(self.probe_mask, row, self.row_aware),
            original_indices=_slice_row_aware_metadata(
                self.original_indices,
                row,
                self.row_aware,
            ),
            positions_um=_slice_row_aware_metadata(self.positions_um, row, self.row_aware),
            thresholds_mV=np.asarray(self.thresholds_mV),
            row_aware=self.row_aware,
        )

    @classmethod
    def concat_batch(cls, results: Sequence["VmRasterResult"]) -> "VmRasterResult":
        """Concatenate one or more raster results along the batch axis."""

        if not results:
            raise ValueError("at least one VmRasterResult is required.")
        first = results[0]
        row_aware_metadata = (
            first.row_aware
            and np.asarray(first.probe_indices).ndim == 3
            and all(result.row_aware for result in results)
        )
        if row_aware_metadata:
            probe_indices = np.concatenate(
                [np.asarray(result.probe_indices) for result in results],
                axis=0,
            )
            probe_mask = np.concatenate(
                [np.asarray(result.probe_mask) for result in results],
                axis=0,
            )
            original_indices = np.concatenate(
                [np.asarray(result.original_indices) for result in results],
                axis=0,
            )
            positions_um = np.concatenate(
                [np.asarray(result.positions_um) for result in results],
                axis=0,
            )
        else:
            probe_indices = np.asarray(first.probe_indices)
            probe_mask = np.asarray(first.probe_mask)
            original_indices = np.asarray(first.original_indices)
            positions_um = np.asarray(first.positions_um)
        return cls(
            words=np.concatenate([np.asarray(result.words) for result in results], axis=0),
            nt=first.nt,
            dt_ms=first.dt_ms,
            definitions=first.definitions,
            names=first.names,
            probe_indices=probe_indices,
            probe_mask=probe_mask,
            original_indices=original_indices,
            positions_um=positions_um,
            thresholds_mV=np.asarray(first.thresholds_mV),
            row_aware=first.row_aware,
        )

    def unpack(self) -> np.ndarray:
        """Unpack words to a boolean array with shape ``(B, R, P, nt)``."""

        return unpack_vm_raster_words(self.words, nt=self.nt)


def _slice_row_aware_metadata(values: Any, row: int, row_aware: bool) -> Any:
    arr = np.asarray(values)
    if row_aware and arr.ndim == 3:
        return arr[row : row + 1]
    return arr


def unpack_vm_raster_words(words: Any, *, nt: int | None = None) -> np.ndarray:
    """Unpack little-endian ``uint32`` raster words to boolean time samples."""

    packed = np.asarray(words, dtype=np.uint32)
    bit_offsets = np.arange(32, dtype=np.uint32)
    unpacked = ((packed[..., :, None] >> bit_offsets) & np.uint32(1)).astype(bool)
    unpacked = unpacked.reshape(packed.shape[:-1] + (-1,))
    if nt is not None:
        unpacked = unpacked[..., : int(nt)]
    return unpacked


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

    return {
        VM_RASTER_OBSERVATION_KEY: VmRasterResult(
            words=np.asarray(state, dtype=np.uint32),
            nt=int(nt),
            dt_ms=float(dt_ms),
            definitions=plan.definitions,
            names=plan.names,
            probe_indices=np.asarray(plan.probe_indices),
            probe_mask=np.asarray(plan.probe_mask, dtype=bool),
            original_indices=np.asarray(plan.original_indices, dtype=np.int32),
            positions_um=np.asarray(plan.positions_um, dtype=float),
            thresholds_mV=np.asarray(plan.thresholds_mV, dtype=float),
            row_aware=plan.row_aware,
        )
    }


__all__ = [
    "VM_RASTER_OBSERVATION_KEY",
    "VmRasterPlan",
    "VmRasterResult",
    "VmRasterState",
    "build_vm_raster_plan",
    "finalize_vm_raster_state",
    "init_vm_raster_state",
    "unpack_vm_raster_words",
    "update_vm_raster_state_batch",
    "update_vm_raster_state_batch_from_tables",
    "update_vm_raster_state_scalar_from_tables",
]
