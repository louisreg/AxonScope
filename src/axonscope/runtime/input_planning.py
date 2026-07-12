"""Runtime-neutral planning helpers for prepared input rows."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, Sequence

import numpy as np

from axonscope.runtime.input_contract import ExtracellularLoweringMode


ArraySignatureFn = Callable[[np.ndarray], tuple[Any, ...]]


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
) -> tuple[tuple[Any, ...], float] | None:
    """Return a normalized-waveform signature plus its amplitude scale.

    Runtimes may pass a cached ``array_signature`` implementation, but the
    semantic result is independent from the concrete array backend.
    """

    if array_signature is None:
        array_signature = array_content_signature
    try:
        t = np.asarray(stimulus.t)
        y = np.asarray(stimulus.y, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        return None
    nonzero = np.flatnonzero(np.abs(y) > 0.0)
    if len(nonzero) == 0:
        scale = 0.0
        normalized = np.zeros_like(y, dtype=float)
    else:
        scale = float(y[int(nonzero[0])])
        if scale == 0.0:
            return None
        normalized = np.asarray(y / scale, dtype=float)
    signature = (
        "stimulus_scaled_waveform_v1",
        type(stimulus),
        getattr(stimulus, "mode", None),
        getattr(stimulus, "y_unit", None),
        array_signature(t),
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


__all__ = [
    "ArraySignatureFn",
    "array_content_signature",
    "can_factorize_footprint_rows",
    "can_plan_compact_double_cable_factorized_rows",
    "extracellular_stimulation_count",
    "factorized_drive_count_from_rows",
    "planned_factorized_extracellular_mode_from_rows",
    "stimulus_scaled_waveform_signature_and_scale",
]
