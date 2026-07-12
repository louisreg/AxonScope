"""Runtime-neutral planning helpers for prepared input rows."""

from __future__ import annotations

import hashlib
import weakref
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from axonscope.runtime.input_contract import ExtracellularLoweringMode


ArraySignatureFn = Callable[[np.ndarray], tuple[Any, ...]]

_ARRAY_CONTENT_KEY_CACHE: OrderedDict[
    int,
    tuple[
        weakref.ReferenceType[np.ndarray],
        tuple[int, ...],
        str,
        tuple[int, ...],
        tuple[tuple[int, ...], str, str],
    ],
] = OrderedDict()
_ARRAY_CONTENT_KEY_CACHE_MAX_SIZE = 128

_STIMULUS_SCALED_WAVEFORM_CACHE: OrderedDict[
    tuple[Any, ...], "_ScaledWaveformSignatureCacheEntry"
] = OrderedDict()
_STIMULUS_SCALED_WAVEFORM_CACHE_MAX_SIZE = 4096


@dataclass(frozen=True)
class Rank1CurrentRows:
    current_mid_A: np.ndarray
    current_initial_previous_A: Any | None
    current_row_indices: np.ndarray | None
    shared_current: bool
    temporal_mid_cache_hits: int
    temporal_mid_cache_misses: int
    temporal_previous_cache_hits: int
    temporal_previous_cache_misses: int


@dataclass(frozen=True)
class ScaledSharedWaveformRows:
    current_mid_A: np.ndarray
    current_initial_previous_A: Any | None
    current_row_scales: np.ndarray | None
    shared_current: bool
    temporal_mid_cache_hits: int
    temporal_mid_cache_misses: int
    temporal_previous_cache_hits: int
    temporal_previous_cache_misses: int


@dataclass(frozen=True)
class _ScaledWaveformSignatureCacheEntry:
    t_ref: weakref.ReferenceType[np.ndarray]
    y_ref: weakref.ReferenceType[np.ndarray]
    t_shape: tuple[int, ...]
    y_shape: tuple[int, ...]
    t_dtype: str
    y_dtype: str
    t_strides: tuple[int, ...]
    y_strides: tuple[int, ...]
    signature: tuple[Any, ...]
    scale: float


def cached_stimulus_current_A(
    cache: dict[Any, np.ndarray],
    stimulus: Any,
    t_ms: np.ndarray,
    *,
    np_dtype: np.dtype[Any],
) -> tuple[np.ndarray, bool]:
    """Evaluate stimulus current in amperes with identity and content caching."""

    identity_key = ("stimulus_identity", id(stimulus))
    values = cache.get(identity_key)
    if values is not None:
        return values, True
    cache_key = stimulus_temporal_cache_key(stimulus)
    values = cache.get(cache_key)
    if values is not None:
        cache[identity_key] = values
        return values, True
    values = np.asarray(stimulus.evaluate(t_ms, unit="ampere"), dtype=np_dtype)
    cache[cache_key] = values
    cache[identity_key] = values
    return values, False


def stimulus_temporal_cache_key(stimulus: Any) -> tuple[Any, ...]:
    """Return a stable semantic cache key for a sampled stimulus waveform."""

    return (
        "stimulus_temporal_v1",
        type(stimulus),
        stimulus.mode,
        stimulus.y_unit,
        cached_array_content_signature(np.asarray(stimulus.t)),
        cached_array_content_signature(np.asarray(stimulus.y)),
    )


def build_rank1_current_rows_from_unique_stimuli(
    drive_rows: Sequence[Sequence[tuple[Any, Any, Any]]],
    t_ms: np.ndarray,
    *,
    t_initial_previous_ms: np.ndarray | None,
    np_dtype: np.dtype[Any],
) -> Rank1CurrentRows | None:
    """Lower one-drive rows to shared or unique-indexed temporal currents."""

    if not drive_rows or not all(len(row) == 1 for row in drive_rows):
        return None

    inverse_indices = np.empty((len(drive_rows),), dtype=np.intp)
    key_to_unique_index: dict[tuple[Any, ...], int] = {}
    unique_stimuli: list[Any] = []
    for row_index, row in enumerate(drive_rows):
        stimulus = row[0][2]
        key = stimulus_temporal_cache_key(stimulus)
        unique_index = key_to_unique_index.get(key)
        if unique_index is None:
            unique_index = len(unique_stimuli)
            key_to_unique_index[key] = unique_index
            unique_stimuli.append(stimulus)
        inverse_indices[row_index] = unique_index

    unique_current_mid_A = np.stack(
        [
            np.asarray(stimulus.evaluate(t_ms, unit="ampere"), dtype=np_dtype)
            for stimulus in unique_stimuli
        ],
        axis=0,
    )
    unique_count = int(unique_current_mid_A.shape[0])
    temporal_hits = int(len(drive_rows) - unique_count)
    temporal_misses = unique_count

    unique_previous_A = None
    if t_initial_previous_ms is not None:
        unique_previous_A = np.asarray(
            [
                np.asarray(
                    stimulus.evaluate(t_initial_previous_ms, unit="ampere"),
                    dtype=np_dtype,
                ).reshape(-1)[0]
                for stimulus in unique_stimuli
            ],
            dtype=np_dtype,
        )

    if unique_count == 1:
        current_mid_A = unique_current_mid_A[0]
        current_previous_A = (
            None if unique_previous_A is None else np.asarray(unique_previous_A[0])
        )
        current_row_indices = None
        shared_current = True
    else:
        current_mid_A = np.ascontiguousarray(unique_current_mid_A, dtype=np_dtype)
        current_previous_A = (
            None if unique_previous_A is None else np.ascontiguousarray(unique_previous_A)
        )
        current_row_indices = np.ascontiguousarray(inverse_indices, dtype=np.int32)
        shared_current = False

    return Rank1CurrentRows(
        current_mid_A=current_mid_A,
        current_initial_previous_A=current_previous_A,
        current_row_indices=current_row_indices,
        shared_current=shared_current,
        temporal_mid_cache_hits=temporal_hits,
        temporal_mid_cache_misses=temporal_misses,
        temporal_previous_cache_hits=(
            0 if t_initial_previous_ms is None else temporal_hits
        ),
        temporal_previous_cache_misses=(
            0 if t_initial_previous_ms is None else temporal_misses
        ),
    )


def build_scaled_shared_waveform_rows(
    drive_rows: Sequence[Sequence[tuple[Any, Any, Any]]],
    t_ms: np.ndarray,
    *,
    t_initial_previous_ms: np.ndarray | None,
    np_dtype: np.dtype[Any],
) -> ScaledSharedWaveformRows | None:
    """Return shared waveform samples plus per-row scales when possible."""

    if not drive_rows:
        return None
    drive_count = len(drive_rows[0])
    if drive_count < 1 or not all(len(row) == drive_count for row in drive_rows):
        return None

    row_scales = np.zeros((len(drive_rows), drive_count), dtype=np_dtype)
    base_stimuli: list[Any] = []
    base_scales: list[float] = []
    for drive_index in range(drive_count):
        first_stimulus = drive_rows[0][drive_index][2]
        first = cached_stimulus_scaled_waveform_signature_and_scale(first_stimulus)
        if first is None:
            return None
        first_signature, first_scale = first
        base_stimuli.append(first_stimulus)
        base_scales.append(first_scale)
        row_scales[0, drive_index] = np.asarray(first_scale, dtype=np_dtype)
        for row_index, row in enumerate(drive_rows[1:], start=1):
            candidate = cached_stimulus_scaled_waveform_signature_and_scale(
                row[drive_index][2]
            )
            if candidate is None:
                return None
            signature, scale = candidate
            if signature != first_signature:
                return None
            row_scales[row_index, drive_index] = np.asarray(scale, dtype=np_dtype)

    shared_scales = bool(np.all(row_scales == row_scales[0:1, :]))

    current_mid_A = np.stack(
        [
            _scaled_waveform_base_current_A(
                stimulus,
                scale,
                t_ms,
                np_dtype=np_dtype,
            )
            for stimulus, scale in zip(base_stimuli, base_scales, strict=True)
        ],
        axis=0,
    )
    current_initial_previous_A = None
    if t_initial_previous_ms is not None:
        current_initial_previous_A = np.asarray(
            [
                _scaled_waveform_base_current_A(
                    stimulus,
                    scale,
                    t_initial_previous_ms,
                    np_dtype=np_dtype,
                ).reshape(-1)[0]
                for stimulus, scale in zip(base_stimuli, base_scales, strict=True)
            ],
            dtype=np_dtype,
        )

    if drive_count == 1:
        current_mid_A = np.ascontiguousarray(current_mid_A[0], dtype=np_dtype)
        if current_initial_previous_A is not None:
            current_initial_previous_A = np.asarray(
                current_initial_previous_A[0],
                dtype=np_dtype,
            )
        if shared_scales:
            shared_scale = np.asarray(row_scales[0, 0], dtype=np_dtype)
            current_mid_A = np.ascontiguousarray(
                current_mid_A * shared_scale,
                dtype=np_dtype,
            )
            if current_initial_previous_A is not None:
                current_initial_previous_A = np.asarray(
                    current_initial_previous_A * shared_scale,
                    dtype=np_dtype,
                )
            current_row_scales = None
        else:
            current_row_scales = np.ascontiguousarray(row_scales[:, 0], dtype=np_dtype)
    else:
        if shared_scales:
            shared_scale_rows = np.asarray(row_scales[0], dtype=np_dtype)
            current_mid_A = np.ascontiguousarray(
                current_mid_A * shared_scale_rows[:, None],
                dtype=np_dtype,
            )
            if current_initial_previous_A is not None:
                current_initial_previous_A = np.ascontiguousarray(
                    current_initial_previous_A * shared_scale_rows,
                    dtype=np_dtype,
                )
            current_row_scales = None
        else:
            current_mid_A = np.ascontiguousarray(current_mid_A, dtype=np_dtype)
            current_row_scales = np.ascontiguousarray(row_scales, dtype=np_dtype)

    unique_waveforms = drive_count
    sample_count = len(drive_rows) * drive_count
    return ScaledSharedWaveformRows(
        current_mid_A=current_mid_A,
        current_initial_previous_A=current_initial_previous_A,
        current_row_scales=current_row_scales,
        shared_current=shared_scales,
        temporal_mid_cache_hits=max(sample_count - unique_waveforms, 0),
        temporal_mid_cache_misses=unique_waveforms,
        temporal_previous_cache_hits=(
            0 if t_initial_previous_ms is None else max(sample_count - unique_waveforms, 0)
        ),
        temporal_previous_cache_misses=(
            0 if t_initial_previous_ms is None else unique_waveforms
        ),
    )


def _scaled_waveform_base_current_A(
    stimulus: Any,
    scale: float,
    t_ms: np.ndarray,
    *,
    np_dtype: np.dtype[Any],
) -> np.ndarray:
    current = np.asarray(stimulus.evaluate(t_ms, unit="ampere"), dtype=np_dtype)
    if scale == 0.0:
        return np.zeros_like(current, dtype=np_dtype)
    return np.asarray(current / np.asarray(scale, dtype=np_dtype), dtype=np_dtype)


def cached_stimulus_scaled_waveform_signature_and_scale(
    stimulus: Any,
) -> tuple[tuple[Any, ...], float] | None:
    """Cached variant of :func:`stimulus_scaled_waveform_signature_and_scale`."""

    try:
        t = np.asarray(stimulus.t)
        y = np.asarray(stimulus.y, dtype=float)
    except (TypeError, ValueError):
        return None
    cache_key = _scaled_waveform_signature_cache_key(stimulus, t, y)
    if cache_key is not None:
        cached = _STIMULUS_SCALED_WAVEFORM_CACHE.get(cache_key)
        if cached is not None and _scaled_waveform_cache_entry_matches(cached, t, y):
            _STIMULUS_SCALED_WAVEFORM_CACHE.move_to_end(cache_key)
            return cached.signature, cached.scale
        _STIMULUS_SCALED_WAVEFORM_CACHE.pop(cache_key, None)
    computed = stimulus_scaled_waveform_signature_and_scale(
        stimulus,
        array_signature=cached_array_content_signature,
        t=t,
        y=y,
    )
    if computed is None:
        return None
    signature, scale = computed
    if cache_key is not None:
        _STIMULUS_SCALED_WAVEFORM_CACHE[cache_key] = (
            _ScaledWaveformSignatureCacheEntry(
                t_ref=weakref.ref(t),
                y_ref=weakref.ref(y),
                t_shape=tuple(int(dim) for dim in t.shape),
                y_shape=tuple(int(dim) for dim in y.shape),
                t_dtype=t.dtype.str,
                y_dtype=y.dtype.str,
                t_strides=tuple(int(stride) for stride in t.strides),
                y_strides=tuple(int(stride) for stride in y.strides),
                signature=signature,
                scale=scale,
            )
        )
        _STIMULUS_SCALED_WAVEFORM_CACHE.move_to_end(cache_key)
        while (
            len(_STIMULUS_SCALED_WAVEFORM_CACHE)
            > _STIMULUS_SCALED_WAVEFORM_CACHE_MAX_SIZE
        ):
            _STIMULUS_SCALED_WAVEFORM_CACHE.popitem(last=False)
    return signature, scale


def _scaled_waveform_signature_cache_key(
    stimulus: Any,
    t: np.ndarray,
    y: np.ndarray,
) -> tuple[Any, ...] | None:
    if not (_can_cache_array_content_key(t) and _can_cache_array_content_key(y)):
        return None
    return (
        "stimulus_scaled_waveform_identity_v1",
        type(stimulus),
        getattr(stimulus, "mode", None),
        getattr(stimulus, "y_unit", None),
        id(t),
        id(y),
    )


def _scaled_waveform_cache_entry_matches(
    entry: _ScaledWaveformSignatureCacheEntry,
    t: np.ndarray,
    y: np.ndarray,
) -> bool:
    return (
        entry.t_ref() is t
        and entry.y_ref() is y
        and entry.t_shape == tuple(int(dim) for dim in t.shape)
        and entry.y_shape == tuple(int(dim) for dim in y.shape)
        and entry.t_dtype == t.dtype.str
        and entry.y_dtype == y.dtype.str
        and entry.t_strides == tuple(int(stride) for stride in t.strides)
        and entry.y_strides == tuple(int(stride) for stride in y.strides)
    )


def can_factorize_footprint_rows(
    rows: Sequence[tuple[Any, ...]],
) -> bool:
    """Return whether rows use sampled footprint/drive stimulation objects."""

    if not rows or not any(rows):
        return False
    for row in rows:
        for stimulation in row:
            drives = tuple(getattr(stimulation, "drives", ()))
            if not drives:
                return False
            for drive in drives:
                if getattr(drive, "stimulus", None) is None:
                    return False
                if getattr(drive, "footprint", None) is None:
                    return False
    return True


def factorized_drive_count_from_rows(rows: Sequence[tuple[Any, ...]]) -> int:
    """Return the maximum factorized drive count per row."""

    max_count = 1
    for row in rows:
        row_count = 0
        for stimulation in row:
            row_count += len(tuple(getattr(stimulation, "drives", ())))
        max_count = max(max_count, row_count)
    return int(max_count)


def can_plan_compact_double_cable_factorized_rows(
    rows: Sequence[tuple[Any, ...]],
) -> bool:
    """Conservatively predict the current double-cable compact factorized path."""

    return planned_factorized_extracellular_mode_from_rows(rows) in {
        ExtracellularLoweringMode.SHARED_CURRENT,
        ExtracellularLoweringMode.SCALED_SHARED_WAVEFORM,
    }


def planned_factorized_extracellular_mode_from_rows(
    rows: Sequence[tuple[Any, ...]],
) -> ExtracellularLoweringMode | None:
    """Predict the semantic factorized extracellular mode without arrays."""

    if not can_factorize_footprint_rows(rows):
        return None
    if factorized_drive_count_from_rows(rows) != 1:
        return ExtracellularLoweringMode.CURRENT_TABLE

    row_stimuli: list[Any] = []
    for row in rows:
        stimuli = [
            getattr(drive, "stimulus", None)
            for stimulation in row
            for drive in tuple(getattr(stimulation, "drives", ()))
        ]
        if len(stimuli) != 1 or stimuli[0] is None:
            return ExtracellularLoweringMode.CURRENT_TABLE
        row_stimuli.append(stimuli[0])
    if not row_stimuli:
        return None

    first = row_stimuli[0]
    if all(stimulus is first for stimulus in row_stimuli[1:]):
        return ExtracellularLoweringMode.SHARED_CURRENT

    scaled = [
        stimulus_scaled_waveform_signature_and_scale(stimulus)
        for stimulus in row_stimuli
    ]
    if any(item is None for item in scaled):
        return ExtracellularLoweringMode.CURRENT_TABLE
    first_signature, first_scale = scaled[0]  # type: ignore[index]
    if not all(item[0] == first_signature for item in scaled[1:] if item is not None):
        return ExtracellularLoweringMode.CURRENT_TABLE
    if all(item is not None and item[1] == first_scale for item in scaled[1:]):
        return ExtracellularLoweringMode.SHARED_CURRENT
    return ExtracellularLoweringMode.SCALED_SHARED_WAVEFORM


def extracellular_stimulation_count(rows: Sequence[tuple[Any, ...]]) -> int:
    """Return the number of attached extracellular stimulation objects."""

    return sum(len(tuple(row)) for row in rows)


def stimulus_scaled_waveform_signature_and_scale(
    stimulus: Any,
    *,
    array_signature: ArraySignatureFn | None = None,
    t: Any | None = None,
    y: Any | None = None,
) -> tuple[tuple[Any, ...], float] | None:
    """Return a normalized-waveform signature plus its amplitude scale.

    Runtimes may pass a cached ``array_signature`` implementation, but the
    semantic result is independent from the concrete array backend.
    """

    if array_signature is None:
        array_signature = array_content_signature
    try:
        t_array = np.asarray(stimulus.t if t is None else t)
        y_array = np.asarray(stimulus.y if y is None else y, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None
    if y_array.ndim != 1 or not np.all(np.isfinite(y_array)):
        return None
    nonzero = np.flatnonzero(np.abs(y_array) > 0.0)
    if len(nonzero) == 0:
        scale = 0.0
        normalized = np.zeros_like(y_array, dtype=float)
    else:
        scale = float(y_array[int(nonzero[0])])
        if scale == 0.0:
            return None
        normalized = np.asarray(y_array / scale, dtype=float)
    signature = (
        "stimulus_scaled_waveform_v1",
        type(stimulus),
        getattr(stimulus, "mode", None),
        getattr(stimulus, "y_unit", None),
        array_signature(t_array),
        array_signature(normalized),
    )
    return signature, scale


def array_content_signature(values: np.ndarray) -> tuple[Any, ...]:
    """Return a deterministic shape/dtype/content signature for an array."""

    arr = np.ascontiguousarray(np.asarray(values))
    return (
        tuple(int(dim) for dim in arr.shape),
        arr.dtype.str,
        hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest(),
    )


def cached_array_content_signature(
    values: np.ndarray,
) -> tuple[tuple[int, ...], str, str]:
    """Return a deterministic array signature with identity-cache reuse."""

    arr = np.asarray(values)
    cached = _cached_array_content_key(arr)
    if cached is not None:
        return cached

    arr = np.ascontiguousarray(arr)
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
    key = tuple(int(dim) for dim in arr.shape), arr.dtype.str, digest
    if _can_cache_array_content_key(arr):
        _store_array_content_key(arr, key)
    return key


def _cached_array_content_key(
    arr: np.ndarray,
) -> tuple[tuple[int, ...], str, str] | None:
    if not _can_cache_array_content_key(arr):
        return None
    cache_key = id(arr)
    cached = _ARRAY_CONTENT_KEY_CACHE.get(cache_key)
    if cached is None:
        return None
    ref, shape, dtype_str, strides, key = cached
    if (
        ref() is arr
        and shape == tuple(int(dim) for dim in arr.shape)
        and dtype_str == arr.dtype.str
        and strides == tuple(int(stride) for stride in arr.strides)
    ):
        _ARRAY_CONTENT_KEY_CACHE.move_to_end(cache_key)
        return key
    _ARRAY_CONTENT_KEY_CACHE.pop(cache_key, None)
    return None


def _store_array_content_key(
    arr: np.ndarray,
    key: tuple[tuple[int, ...], str, str],
) -> None:
    _ARRAY_CONTENT_KEY_CACHE[id(arr)] = (
        weakref.ref(arr),
        tuple(int(dim) for dim in arr.shape),
        arr.dtype.str,
        tuple(int(stride) for stride in arr.strides),
        key,
    )
    _ARRAY_CONTENT_KEY_CACHE.move_to_end(id(arr))
    while len(_ARRAY_CONTENT_KEY_CACHE) > _ARRAY_CONTENT_KEY_CACHE_MAX_SIZE:
        _ARRAY_CONTENT_KEY_CACHE.popitem(last=False)


def _can_cache_array_content_key(arr: np.ndarray) -> bool:
    return (
        isinstance(arr, np.ndarray)
        and arr.flags.c_contiguous
        and arr.flags.owndata
        and not arr.flags.writeable
        and not arr.dtype.hasobject
    )


__all__ = [
    "ArraySignatureFn",
    "Rank1CurrentRows",
    "ScaledSharedWaveformRows",
    "array_content_signature",
    "build_rank1_current_rows_from_unique_stimuli",
    "build_scaled_shared_waveform_rows",
    "cached_array_content_signature",
    "cached_stimulus_current_A",
    "cached_stimulus_scaled_waveform_signature_and_scale",
    "can_factorize_footprint_rows",
    "can_plan_compact_double_cable_factorized_rows",
    "extracellular_stimulation_count",
    "factorized_drive_count_from_rows",
    "planned_factorized_extracellular_mode_from_rows",
    "stimulus_scaled_waveform_signature_and_scale",
    "stimulus_temporal_cache_key",
]
