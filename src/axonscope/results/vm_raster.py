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


__all__ = [
    "VM_RASTER_OBSERVATION_KEY",
    "VmRasterResult",
    "unpack_vm_raster_words",
]
