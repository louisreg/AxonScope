"""Public packed membrane-voltage raster result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


VM_RASTER_OBSERVATION_KEY = "vm_raster"


@dataclass(frozen=True)
class VmRasterResult:
    """Packed membrane-voltage threshold raster.

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
        for result in results[1:]:
            _check_concat_compatible(first, result)

        max_probe_count = max(result.probe_count for result in results)
        words = np.concatenate(
            [
                _pad_probe_axis(
                    np.asarray(result.words, dtype=np.uint32),
                    max_probe_count,
                    fill_value=0,
                )
                for result in results
            ],
            axis=0,
        )

        shared_metadata = _metadata_is_shared(results, max_probe_count=max_probe_count)
        if shared_metadata:
            probe_indices = np.asarray(first.probe_indices)
            probe_mask = np.asarray(first.probe_mask)
            original_indices = np.asarray(first.original_indices)
            positions_um = np.asarray(first.positions_um)
            row_aware = False
        else:
            probe_indices = np.concatenate(
                [
                    _metadata_as_row_aware(
                        result.probe_indices,
                        result=result,
                        max_probe_count=max_probe_count,
                        fill_value=0,
                    )
                    for result in results
                ],
                axis=0,
            )
            probe_mask = np.concatenate(
                [
                    _metadata_as_row_aware(
                        result.probe_mask,
                        result=result,
                        max_probe_count=max_probe_count,
                        fill_value=False,
                    )
                    for result in results
                ],
                axis=0,
            )
            original_indices = np.concatenate(
                [
                    _metadata_as_row_aware(
                        result.original_indices,
                        result=result,
                        max_probe_count=max_probe_count,
                        fill_value=-1,
                    )
                    for result in results
                ],
                axis=0,
            )
            positions_um = np.concatenate(
                [
                    _metadata_as_row_aware(
                        result.positions_um,
                        result=result,
                        max_probe_count=max_probe_count,
                        fill_value=np.nan,
                    )
                    for result in results
                ],
                axis=0,
            )
            row_aware = True
        return cls(
            words=words,
            nt=first.nt,
            dt_ms=first.dt_ms,
            definitions=first.definitions,
            names=first.names,
            probe_indices=probe_indices,
            probe_mask=probe_mask,
            original_indices=original_indices,
            positions_um=positions_um,
            thresholds_mV=np.asarray(first.thresholds_mV),
            row_aware=row_aware,
        )

    def unpack(self) -> np.ndarray:
        """Unpack words to a boolean array with shape ``(B, R, P, nt)``."""

        return unpack_vm_raster_words(self.words, nt=self.nt)


def _slice_row_aware_metadata(values: Any, row: int, row_aware: bool) -> Any:
    arr = np.asarray(values)
    if row_aware and arr.ndim == 3:
        return arr[row : row + 1]
    return arr


def _check_concat_compatible(first: VmRasterResult, result: VmRasterResult) -> None:
    """Validate static raster axes before batch concatenation."""

    if int(result.nt) != int(first.nt):
        raise ValueError("VmRaster results must share nt to concatenate.")
    if float(result.dt_ms) != float(first.dt_ms):
        raise ValueError("VmRaster results must share dt_ms to concatenate.")
    if int(result.raster_count) != int(first.raster_count):
        raise ValueError("VmRaster results must share raster_count to concatenate.")
    if int(result.word_count) != int(first.word_count):
        raise ValueError("VmRaster results must share word_count to concatenate.")
    if tuple(result.names) != tuple(first.names):
        raise ValueError("VmRaster results must share observer names to concatenate.")
    if not np.array_equal(np.asarray(result.thresholds_mV), np.asarray(first.thresholds_mV)):
        raise ValueError("VmRaster results must share thresholds to concatenate.")


def _pad_probe_axis(values: np.ndarray, max_probe_count: int, *, fill_value: Any) -> np.ndarray:
    """Pad axis 2, the probe axis, to a common width."""

    array = np.asarray(values)
    if array.ndim < 3:
        raise ValueError("VmRaster values must have a probe axis.")
    pad_count = int(max_probe_count) - int(array.shape[2])
    if pad_count < 0:
        raise ValueError("max_probe_count must be greater than or equal to the probe axis.")
    if pad_count == 0:
        return array
    pad_width = [(0, 0)] * array.ndim
    pad_width[2] = (0, pad_count)
    return np.pad(array, pad_width, mode="constant", constant_values=fill_value)


def _metadata_as_row_aware(
    values: Any,
    *,
    result: VmRasterResult,
    max_probe_count: int,
    fill_value: Any,
) -> np.ndarray:
    """Return metadata as ``(B, R, P)`` and pad probe slots when needed."""

    array = np.asarray(values)
    if array.ndim == 2:
        array = np.broadcast_to(array[None, :, :], (result.batch_size,) + array.shape).copy()
    elif array.ndim == 3:
        if array.shape[0] != result.batch_size:
            raise ValueError("row-aware VmRaster metadata must align with batch_size.")
    else:
        raise ValueError("VmRaster metadata must be 2D or 3D.")
    return _pad_probe_axis(array, int(max_probe_count), fill_value=fill_value)


def _metadata_is_shared(
    results: Sequence[VmRasterResult],
    *,
    max_probe_count: int,
) -> bool:
    """Return whether all results can safely reuse one non-row-aware metadata table."""

    first = results[0]
    first_metadata = (
        np.asarray(first.probe_indices),
        np.asarray(first.probe_mask),
        np.asarray(first.original_indices),
        np.asarray(first.positions_um),
    )
    if first.row_aware or any(array.ndim == 3 for array in first_metadata):
        return False
    if int(first.probe_count) != int(max_probe_count):
        return False
    for result in results[1:]:
        metadata = (
            np.asarray(result.probe_indices),
            np.asarray(result.probe_mask),
            np.asarray(result.original_indices),
            np.asarray(result.positions_um),
        )
        if result.row_aware or any(array.ndim == 3 for array in metadata):
            return False
        if int(result.probe_count) != int(max_probe_count):
            return False
        if not all(
            np.array_equal(current, reference, equal_nan=True)
            for current, reference in zip(metadata, first_metadata, strict=True)
        ):
            return False
    return True


def unpack_vm_raster_words(words: Any, *, nt: int | None = None) -> np.ndarray:
    """Unpack little-endian ``uint32`` raster words to boolean time samples."""

    packed = np.asarray(words, dtype=np.uint32)
    bit_offsets = np.arange(32, dtype=np.uint32)
    unpacked = ((packed[..., :, None] >> bit_offsets) & np.uint32(1)).astype(bool)
    unpacked = unpacked.reshape(packed.shape[:-1] + (-1,))
    if nt is not None:
        unpacked = unpacked[..., : int(nt)]
    return unpacked


__all__ = [
    "VM_RASTER_OBSERVATION_KEY",
    "VmRasterResult",
    "unpack_vm_raster_words",
]
