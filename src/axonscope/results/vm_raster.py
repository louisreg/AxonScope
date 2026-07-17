"""Public packed membrane-voltage raster result containers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

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
    _any_active_impl: Any | None = field(default=None, repr=False, compare=False)

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

    @classmethod
    def from_result(cls, result: Any, *definitions: Any) -> "VmRasterResult":
        """Pack threshold-style Vm analyses from a dense recorded result.

        This is the post-hoc twin of the solver-side observer path: it uses the
        same definition names, probe selectors, thresholds, metadata tables, and
        little-endian ``uint32`` bit layout. The packed bits represent raw
        ``Vm >= threshold`` windows; analysis-specific blanking is applied by
        the analysis/protocol layer when those bits are interpreted.
        """

        raster_definitions = _normalize_vm_raster_definitions(definitions)
        rows = _result_rows_for_vm_raster(result)
        packed_rows = [
            _vm_raster_from_single_result(cls, row, raster_definitions)
            for row in rows
        ]
        if len(packed_rows) == 1:
            return packed_rows[0]
        return cls.concat_batch(packed_rows)

    @classmethod
    def from_vm(cls, result: Any, *definitions: Any) -> "VmRasterResult":
        """Alias for :meth:`from_result` for call sites emphasizing dense Vm."""

        return cls.from_result(result, *definitions)

    def unpack(self) -> np.ndarray:
        """Unpack words to a boolean array with shape ``(B, R, P, nt)``."""

        return unpack_vm_raster_words(self.words, nt=self.nt)

    def definition_index(self, name_or_definition: Any) -> int:
        """Return the raster-definition index for a name or definition object."""

        return vm_raster_definition_index(self, name_or_definition)

    def any_active(
        self,
        name_or_definition: Any,
        *,
        blanking: Any | None = None,
    ) -> np.ndarray:
        """Return per-batch activation flags for one raster definition."""

        return vm_raster_any_active(
            self,
            name_or_definition,
            blanking=blanking,
        )

    def rows(self) -> tuple[dict[str, Any], ...]:
        """Return row dictionaries for dataframe/text views."""

        from axonscope.results.views import vm_raster_rows

        return vm_raster_rows(self)

    def to_dataframe(self) -> Any:
        """Return this raster summary as a pandas DataFrame."""

        from axonscope.results.views import vm_raster_to_dataframe

        return vm_raster_to_dataframe(self)

    def format(self) -> str:
        """Return a compact text representation."""

        from axonscope.results.views import format_vm_raster

        return format_vm_raster(self)

    def print(self, file: TextIO | None = None) -> None:
        """Print a compact text representation."""

        from axonscope.results.views import print_vm_raster

        print_vm_raster(self, file=file)

    def plot(
        self,
        ax: Any | None = None,
        *,
        row: int = 0,
        time_unit: Any = "millisecond",
        title: str = "VmRaster threshold windows",
        grid: bool = True,
    ) -> Any:
        """Plot threshold-crossing windows stored in this packed raster."""

        from axonscope.results.views import plot_vm_raster

        return plot_vm_raster(
            self,
            ax=ax,
            row=row,
            time_unit=time_unit,
            title=title,
            grid=grid,
        )


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


def vm_raster_definition_index(raster: Any, name_or_definition: Any) -> int:
    """Return the definition index in a ``VmRasterResult``."""

    name = str(getattr(name_or_definition, "name", name_or_definition))
    names = tuple(getattr(raster, "names", ()))
    try:
        return names.index(name)
    except ValueError as exc:
        raise KeyError(f"VmRaster definition {name!r} is not present.") from exc


def vm_raster_any_active(
    raster: Any,
    name_or_definition: Any,
    *,
    blanking: Any | None = None,
) -> np.ndarray:
    """Return whether each batch row has any active sample for one definition."""

    raster_index = vm_raster_definition_index(raster, name_or_definition)
    raw_words = getattr(raster, "words")
    any_active_impl = getattr(raster, "_any_active_impl", None)
    if callable(any_active_impl):
        return _vm_raster_any_active_with_impl(
            raster,
            raster_index=raster_index,
            blanking=blanking,
            any_active_impl=any_active_impl,
        )

    words = np.asarray(raw_words, dtype=np.uint32)
    if words.ndim != 4:
        raise ValueError(f"VmRaster words must have shape (B, R, P, W), got {words.shape}.")
    if int(raster_index) < 0 or int(raster_index) >= words.shape[1]:
        raise ValueError(f"VmRaster definition index {raster_index} is out of range.")
    row_words = words[:, int(raster_index)]

    probe_mask = _probe_mask_for_definition(
        raster,
        raster_index=raster_index,
        batch_size=row_words.shape[0],
        probe_count=row_words.shape[1],
    )
    blanking_ms = None if blanking is None else _blanking_ms(blanking)
    word_mask = _vm_raster_time_word_mask(
        nt=int(raster.nt),
        word_count=row_words.shape[2],
        dt_ms=float(raster.dt_ms),
        blanking_ms=blanking_ms,
    )
    active_words = (row_words & word_mask[None, None, :]) != 0
    if bool(np.all(probe_mask)):
        return np.any(active_words, axis=(1, 2))
    return np.any(active_words & probe_mask[:, :, None], axis=(1, 2))


def _vm_raster_any_active_with_impl(
    raster: Any,
    *,
    raster_index: int,
    blanking: Any | None,
    any_active_impl: Any,
) -> np.ndarray:
    words = getattr(raster, "words")
    words_shape = tuple(int(value) for value in getattr(words, "shape", ()))
    if len(words_shape) != 4:
        raise ValueError(f"VmRaster words must have shape (B, R, P, W), got {words_shape}.")
    if int(raster_index) < 0 or int(raster_index) >= words_shape[1]:
        raise ValueError(f"VmRaster definition index {raster_index} is out of range.")
    row_words = words[:, int(raster_index)]

    probe_mask = _probe_mask_for_definition(
        raster,
        raster_index=raster_index,
        batch_size=words_shape[0],
        probe_count=words_shape[2],
    )
    blanking_ms = None if blanking is None else _blanking_ms(blanking)
    word_mask = _vm_raster_time_word_mask(
        nt=int(raster.nt),
        word_count=words_shape[3],
        dt_ms=float(raster.dt_ms),
        blanking_ms=blanking_ms,
    )
    activated = any_active_impl(row_words, word_mask, probe_mask)
    return np.asarray(activated, dtype=bool)


def activation_values_from_vm_raster(raster: Any, activation: Any) -> np.ndarray:
    """Return activation flags decoded from a named VmRaster definition."""

    try:
        return vm_raster_any_active(
            raster,
            activation,
            blanking=getattr(activation, "blanking", None),
        )
    except KeyError as exc:
        raise RuntimeError("activation observer result is missing from VmRaster output.") from exc


def conduction_velocity_values_from_vm_raster(
    raster: Any,
    definition: Any,
) -> np.ndarray:
    """Estimate conduction velocity from first VmRaster threshold crossings."""

    try:
        raster_index = vm_raster_definition_index(raster, definition)
    except KeyError as exc:
        raise RuntimeError(
            "conduction-velocity observer result is missing from VmRaster output."
        ) from exc

    words = np.asarray(getattr(raster, "words"), dtype=np.uint32)
    if words.ndim != 4:
        raise ValueError(f"VmRaster words must have shape (B, R, P, W), got {words.shape}.")
    if int(raster_index) < 0 or int(raster_index) >= words.shape[1]:
        raise ValueError(f"VmRaster definition index {raster_index} is out of range.")

    row_words = words[:, int(raster_index)]
    probe_mask = _probe_mask_for_definition(
        raster,
        raster_index=raster_index,
        batch_size=row_words.shape[0],
        probe_count=row_words.shape[1],
    )
    positions = _metadata_for_definition(
        getattr(raster, "positions_um"),
        raster_index=raster_index,
        batch_size=row_words.shape[0],
        probe_count=row_words.shape[1],
        fill_value=np.nan,
    )
    nt = int(getattr(raster, "nt"))
    dt_ms = float(getattr(raster, "dt_ms"))
    values = np.zeros((row_words.shape[0],), dtype=float)
    for row_index in range(row_words.shape[0]):
        first_samples = _first_active_samples(row_words[row_index], nt=nt)
        valid = (
            probe_mask[row_index]
            & (first_samples >= 0)
            & np.isfinite(positions[row_index])
        )
        if int(np.count_nonzero(valid)) < 2:
            continue
        values[row_index] = _distance_delay_velocity_from_samples(
            first_samples[valid],
            positions[row_index, valid],
            dt_ms=dt_ms,
        )
    return values


def _first_active_samples(row_words: np.ndarray, *, nt: int) -> np.ndarray:
    first = np.full((row_words.shape[0],), -1, dtype=np.int64)
    for probe_index, probe_words in enumerate(row_words):
        active_words = np.flatnonzero(probe_words != np.uint32(0))
        if active_words.size == 0:
            continue
        word_index = int(active_words[0])
        word = int(probe_words[word_index])
        bit_index = (word & -word).bit_length() - 1
        sample = word_index * 32 + bit_index
        if sample < int(nt):
            first[probe_index] = sample
    return first


def _distance_delay_velocity_from_samples(
    samples: np.ndarray,
    positions_um: np.ndarray,
    *,
    dt_ms: float,
) -> float:
    if float(dt_ms) <= 0.0:
        return 0.0
    order = np.argsort(samples)
    ordered_samples = np.asarray(samples, dtype=float)[order]
    ordered_positions = np.asarray(positions_um, dtype=float)[order]
    x_start = float(ordered_positions[0])
    t_start = float(ordered_samples[0]) * float(dt_ms)
    stop_index = int(np.argmax(np.abs(ordered_positions - x_start)))
    delay_ms = float(ordered_samples[stop_index]) * float(dt_ms) - t_start
    if delay_ms <= 0.0:
        return 0.0
    distance_um = abs(float(ordered_positions[stop_index]) - x_start)
    return float(distance_um / delay_ms * 1e-3)


def _metadata_for_definition(
    values: Any,
    *,
    raster_index: int,
    batch_size: int,
    probe_count: int,
    fill_value: Any,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 3:
        selected = arr[:, int(raster_index)]
    elif arr.ndim == 2:
        selected = np.broadcast_to(arr[int(raster_index)], (int(batch_size), arr.shape[1]))
    elif arr.ndim == 1:
        selected = np.broadcast_to(arr, (int(batch_size), arr.shape[0]))
    else:
        selected = np.full((int(batch_size), int(probe_count)), fill_value)
    if selected.shape[1] == int(probe_count):
        return np.asarray(selected)
    out = np.full((int(batch_size), int(probe_count)), fill_value, dtype=arr.dtype)
    width = min(int(probe_count), int(selected.shape[1]))
    out[:, :width] = selected[:, :width]
    return out


def _probe_mask_for_definition(
    raster: Any,
    *,
    raster_index: int,
    batch_size: int,
    probe_count: int,
) -> np.ndarray:
    mask = np.asarray(getattr(raster, "probe_mask", True), dtype=bool)
    if mask.ndim == 3:
        selected = mask[:, int(raster_index)]
    elif mask.ndim == 2:
        selected = np.broadcast_to(mask[int(raster_index)], (int(batch_size), int(probe_count)))
    elif mask.ndim == 1:
        selected = np.broadcast_to(mask, (int(batch_size), int(probe_count)))
    else:
        selected = np.broadcast_to(mask, (int(batch_size), int(probe_count)))
    return np.asarray(selected, dtype=bool)


def _blanking_ms(value: Any) -> float:
    from axonscope.utils import units

    return units.to_ms(value)


def _vm_raster_time_word_mask(
    *,
    nt: int,
    word_count: int,
    dt_ms: float,
    blanking_ms: float | None,
) -> np.ndarray:
    """Return packed-word masks matching the public VmRaster time window."""

    nt = int(nt)
    word_count = int(word_count)
    if nt < 0:
        raise ValueError("VmRaster nt must be non-negative.")
    if word_count < 1:
        raise ValueError("VmRaster word_count must be positive.")
    start = 0
    if blanking_ms is not None and float(blanking_ms) > 0.0:
        if float(dt_ms) <= 0.0:
            start = nt
        else:
            # VmRaster samples are indexed as (t + 1) * dt_ms, matching the
            # previous unpack-and-slice implementation.
            start = max(int(np.ceil(float(blanking_ms) / float(dt_ms))) - 1, 0)
    if start >= nt:
        return np.zeros(word_count, dtype=np.uint32)

    masks = np.full(word_count, np.uint32(0xFFFFFFFF), dtype=np.uint32)
    valid_word_count = min(word_count, max((nt + 31) // 32, 0))
    if valid_word_count < word_count:
        masks[valid_word_count:] = np.uint32(0)

    first_word = start // 32
    first_bit = start % 32
    if first_word > 0:
        masks[:first_word] = np.uint32(0)
    if first_bit:
        masks[first_word] &= np.uint32((0xFFFFFFFF << first_bit) & 0xFFFFFFFF)

    valid_tail_bits = nt % 32
    if valid_tail_bits and valid_word_count:
        tail_mask = (1 << valid_tail_bits) - 1
        masks[valid_word_count - 1] &= np.uint32(tail_mask)
    return masks


def _normalize_vm_raster_definitions(definitions: Sequence[Any]) -> tuple[Any, ...]:
    if (
        len(definitions) == 1
        and isinstance(definitions[0], Sequence)
        and not isinstance(definitions[0], (str, bytes))
    ):
        definitions = tuple(definitions[0])
    raster_definitions = tuple(definitions)
    if not raster_definitions:
        raise ValueError("VmRasterResult.from_result requires at least one definition.")

    names = tuple(str(getattr(definition, "name", "")) for definition in raster_definitions)
    if any(not name for name in names):
        raise ValueError("VmRaster definitions must expose a non-empty name.")
    if len(set(names)) != len(names):
        raise ValueError("VmRaster definition names must be unique.")
    return raster_definitions


def _result_rows_for_vm_raster(result: Any) -> tuple[Any, ...]:
    try:
        getattr(result, "Vm")
    except AttributeError:
        try:
            rows = tuple(result)
        except TypeError:
            rows = (result,)
        if not rows:
            raise ValueError("VmRasterResult.from_result requires at least one result row.")
        return rows
    return (result,)


def _vm_raster_from_single_result(
    cls: type[VmRasterResult],
    row: Any,
    definitions: tuple[Any, ...],
) -> VmRasterResult:
    vm = _dense_vm_mV(row)
    time_ms = _dense_time_ms(row)
    positions_um = _dense_positions_um(row, expected_width=vm.shape[1])
    original_indices = _dense_original_indices(row, expected_width=vm.shape[1])
    dt_ms = _uniform_dt_ms(time_ms)

    selected_by_definition: list[np.ndarray] = []
    thresholds_mV: list[float] = []
    temporal_strides: list[int] = []
    for definition in definitions:
        _require_vm_raster_definition(definition)
        target = getattr(definition, "target")
        selected = _select_vm_raster_columns(
            target,
            positions_um=positions_um,
            original_indices=original_indices,
        )
        selected_by_definition.append(selected)
        thresholds_mV.append(_threshold_mV(definition))
        temporal_strides.append(int(getattr(definition, "every_n_steps", 1)))

    temporal_stride = max(temporal_strides, default=1)
    if any(value != temporal_stride for value in temporal_strides):
        raise ValueError("VmRaster definitions must share every_n_steps.")

    nt = int(vm.shape[0])
    probe_width = max(int(selected.size) for selected in selected_by_definition)
    bits = np.zeros((len(definitions), probe_width, nt), dtype=bool)
    probe_indices = np.zeros((len(definitions), probe_width), dtype=np.int32)
    probe_mask = np.zeros((len(definitions), probe_width), dtype=bool)
    selected_original_indices = np.full((len(definitions), probe_width), -1, dtype=np.int32)
    selected_positions_um = np.full((len(definitions), probe_width), np.nan, dtype=float)

    for definition_index, (selected, threshold_mV) in enumerate(
        zip(selected_by_definition, thresholds_mV, strict=True)
    ):
        count = int(selected.size)
        probe_indices[definition_index, :count] = selected
        probe_mask[definition_index, :count] = True
        selected_original_indices[definition_index, :count] = original_indices[selected]
        selected_positions_um[definition_index, :count] = positions_um[selected]
        bits[definition_index, :count, :] = vm[:, selected].T >= float(threshold_mV)

    if temporal_stride > 1:
        sampled_nt = (nt + temporal_stride - 1) // temporal_stride
        padded = np.zeros(
            bits.shape[:-1] + (sampled_nt * temporal_stride,),
            dtype=bool,
        )
        padded[..., :nt] = bits
        bits = padded.reshape(
            bits.shape[:-1] + (sampled_nt, temporal_stride)
        ).any(axis=-1)
        nt = sampled_nt

    return cls(
        words=_pack_vm_raster_bits(bits)[None, ...],
        nt=nt,
        dt_ms=dt_ms * temporal_stride,
        definitions=definitions,
        names=tuple(str(definition.name) for definition in definitions),
        probe_indices=probe_indices,
        probe_mask=probe_mask,
        original_indices=selected_original_indices,
        positions_um=selected_positions_um,
        thresholds_mV=np.asarray(thresholds_mV, dtype=float),
        row_aware=False,
    )


def _dense_vm_mV(row: Any) -> np.ndarray:
    try:
        values = row.voltage_values(unit="millivolt")
    except AttributeError:
        try:
            values = row.Vm
        except AttributeError as exc:
            raise ValueError("VmRasterResult.from_result requires dense Vm recordings.") from exc
    vm = np.asarray(values, dtype=float)
    if vm.ndim != 2:
        raise ValueError(f"dense Vm must be 2D (time, position), got shape {vm.shape}.")
    if vm.shape[0] == 0 or vm.shape[1] == 0:
        raise ValueError("dense Vm must include at least one time and position sample.")
    return vm


def _dense_time_ms(row: Any) -> np.ndarray:
    try:
        values = row.time_values(unit="millisecond")
    except AttributeError:
        try:
            values = row.t
        except AttributeError as exc:
            raise ValueError("VmRasterResult.from_result requires a result time vector.") from exc
    time_ms = np.asarray(values, dtype=float)
    if time_ms.ndim != 1 or time_ms.size == 0:
        raise ValueError("result time vector must be a non-empty 1D array.")
    return time_ms


def _dense_positions_um(row: Any, *, expected_width: int) -> np.ndarray:
    try:
        values = row.position_values(unit="micrometer")
    except AttributeError as exc:
        raise ValueError(
            "VmRasterResult.from_result requires recorded position metadata."
        ) from exc
    positions_um = np.asarray(values, dtype=float)
    if positions_um.shape != (int(expected_width),):
        raise ValueError("recorded positions must match dense Vm columns.")
    return positions_um


def _dense_original_indices(row: Any, *, expected_width: int) -> np.ndarray:
    try:
        values = row.recorded_axis.index_values()
    except AttributeError:
        values = getattr(row, "record_indices", None)
        if values is None:
            values = np.arange(int(expected_width), dtype=np.int32)
    original_indices = np.asarray(values, dtype=np.int32)
    if original_indices.shape != (int(expected_width),):
        raise ValueError("recorded original indices must match dense Vm columns.")
    return original_indices


def _uniform_dt_ms(time_ms: np.ndarray) -> float:
    if time_ms.shape[0] == 1:
        return 0.0
    steps = np.diff(time_ms)
    dt_ms = float(np.median(steps))
    if dt_ms <= 0.0:
        raise ValueError("VmRasterResult.from_result requires increasing result times.")
    if not np.allclose(steps, dt_ms, rtol=1e-4, atol=max(1e-9, abs(dt_ms) * 1e-6)):
        raise ValueError("VmRasterResult.from_result requires a uniform result time step.")
    return dt_ms


def _require_vm_raster_definition(definition: Any) -> None:
    from axonscope.positions import PositionSelector
    from axonscope.signals import MEMBRANE_VOLTAGE, Signal

    if not hasattr(definition, "threshold") or not hasattr(definition, "target"):
        raise NotImplementedError(
            "VmRasterResult.from_result supports threshold-style Vm definitions only."
        )
    signal = getattr(definition, "signal", MEMBRANE_VOLTAGE)
    if not isinstance(signal, Signal) or signal.id != MEMBRANE_VOLTAGE.id:
        raise NotImplementedError("VmRasterResult.from_result supports membrane voltage only.")
    if not isinstance(getattr(definition, "target"), PositionSelector):
        raise TypeError("VmRaster definitions must use axonscope PositionSelector targets.")


def _threshold_mV(definition: Any) -> float:
    from axonscope.utils import units

    threshold = getattr(definition, "threshold", None)
    return -20.0 if threshold is None else units.to_mV(threshold)


def _select_vm_raster_columns(
    target: Any,
    *,
    positions_um: np.ndarray,
    original_indices: np.ndarray,
) -> np.ndarray:
    valid_columns = np.flatnonzero(original_indices >= 0)
    if valid_columns.size == 0:
        raise ValueError("VmRasterResult.from_result found no valid recorded positions.")
    selected_local = target.columns(
        positions_um=positions_um[valid_columns],
        original_indices=original_indices[valid_columns],
    )
    selected = valid_columns[np.asarray(selected_local, dtype=np.int32)]
    if selected.size == 0:
        raise ValueError("VmRaster target selects no positions.")
    return selected.astype(np.int32, copy=False)


def _pack_vm_raster_bits(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=bool)
    nt = int(values.shape[-1])
    word_count = (nt + 31) // 32
    words = np.zeros(values.shape[:-1] + (word_count,), dtype=np.uint32)
    for word_index in range(word_count):
        start = word_index * 32
        stop = min(start + 32, nt)
        block = values[..., start:stop].astype(np.uint32)
        weights = np.left_shift(
            np.uint32(1),
            np.arange(stop - start, dtype=np.uint32),
        )
        words[..., word_index] = np.sum(block * weights, axis=-1, dtype=np.uint32)
    return words


__all__ = [
    "VM_RASTER_OBSERVATION_KEY",
    "VmRasterResult",
    "activation_values_from_vm_raster",
    "conduction_velocity_values_from_vm_raster",
    "unpack_vm_raster_words",
    "vm_raster_any_active",
    "vm_raster_definition_index",
]
