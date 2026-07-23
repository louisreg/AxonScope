"""JAX materialization of extracellular input tensors.

The solver kernels operate on JAX arrays. This module turns extracellular
stimulation rows and static footprints into backend arrays for batch execution.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np

from axonfleet.benchmarking import benchmark_span, record_benchmark_metadata
from axonfleet.stimulation import (
    ExtracellularStimulation,
    Stimulus,
)
from axonfleet.runtime.inputs.planning import (
    build_rank1_current_rows_from_unique_stimuli as _build_rank1_current_rows_from_unique_stimuli,
    build_scaled_shared_waveform_rows as _build_scaled_shared_waveform_rows,
    cached_array_content_signature as _array_content_key,
    cached_stimulus_current_A as _cached_stimulus_current_A,
    stimulus_temporal_cache_key as _stimulus_temporal_cache_key,
)
from axonfleet.runtime.inputs.payloads import (
    FactorizedExtracellularPotentialBatch,
)
from axonfleet.runtime.timebase import simulation_step_count

Array = Any
StimulationBatchRow = ExtracellularStimulation | Sequence[ExtracellularStimulation] | None
_FOOTPRINT_CACHE: dict[tuple[Any, ...], np.ndarray] = {}
_FOOTPRINT_MV_CACHE: dict[tuple[Any, ...], np.ndarray] = {}
_SINGLE_CABLE_FORCING_MV_CACHE: dict[tuple[Any, ...], np.ndarray] = {}
_FOOTPRINT_JAX_CACHE: OrderedDict[tuple[Any, ...], Array] = OrderedDict()
_FOOTPRINT_JAX_CACHE_MAX_SIZE = 32


@dataclass(frozen=True)
class _FactorizedRowsIdentityPlan:
    rows: tuple[tuple[ExtracellularStimulation, ...], ...]
    x_rows: np.ndarray
    np_dtype_str: str
    drive_rows: tuple[
        tuple[tuple[ExtracellularStimulation, Any, Stimulus], ...], ...
    ]
    max_drive_count: int
    shared_rank1_stimulus: Stimulus | None
    shared_rank1_detection: str
    footprint_cache_key: tuple[Any, ...]
    footprint_mV_cache_key: tuple[Any, ...]
    footprint_V_per_A: np.ndarray
    footprint_mV_per_A: np.ndarray


_FACTORIZED_ROWS_IDENTITY_CACHE: OrderedDict[
    tuple[Any, ...], _FactorizedRowsIdentityPlan
] = OrderedDict()
_FACTORIZED_ROWS_IDENTITY_CACHE_MAX_SIZE = 64


def build_vstim_midpoint_batch(
    axon: object,
    stimulations_batch: Sequence[StimulationBatchRow],
    *,
    tsim_ms: float,
    dt_ms: float,
    x_positions_m: Array | None = None,
    dtype_local: jnp.dtype | None = None,
) -> Array:
    """Build imposed extracellular potentials at solver midpoints.

    Returns ``Vstim[B, Nt, Nx]`` in millivolts. Each row can contain one
    ``ExtracellularStimulation``, several stimulations that are summed, or
    ``None`` for a zero-field control row.
    """

    dtype = _resolve_dtype(axon, dtype_local)
    nt = simulation_step_count(tsim_ms, dt_ms)
    t_mid_ms = (
        jnp.arange(nt, dtype=dtype) + jnp.asarray(0.5, dtype=dtype)
    ) * jnp.asarray(dt_ms, dtype=dtype)
    return build_vstim_batch(
        axon,
        stimulations_batch,
        t_ms=t_mid_ms,
        x_positions_m=x_positions_m,
        dtype_local=dtype,
    )


def build_factorized_vstim_midpoint_batch(
    axon: object,
    stimulations_batch: Sequence[StimulationBatchRow],
    *,
    tsim_ms: float,
    dt_ms: float,
    x_positions_m: Array | None = None,
    dtype_local: jnp.dtype | None = None,
    include_initial_previous: bool = False,
    single_cable_lower: Array | None = None,
    single_cable_upper: Array | None = None,
    numeric_axis_shape: tuple[int, int] | None = None,
) -> FactorizedExtracellularPotentialBatch | None:
    """Build a factorized midpoint ``Vstim`` batch when stimulations allow it.

    Supported stimulation rows keep static spatial footprints separate from
    temporal stimuli so observer-only kernels can avoid materializing
    ``Vstim[B, Nt, Nx]``.
    """

    rows = _normalize_stimulation_rows(stimulations_batch)
    if not rows or not any(rows):
        return None
    if not _can_build_stimulation_rows_from_footprints(rows):
        return None

    dtype = _resolve_dtype(axon, dtype_local)
    np_dtype = np.dtype(dtype)
    nt = simulation_step_count(tsim_ms, dt_ms)
    t_mid_ms = (
        np.arange(nt, dtype=np_dtype) + np.asarray(0.5, dtype=np_dtype)
    ) * np.asarray(dt_ms, dtype=np_dtype)
    t_initial_previous_ms = (
        np.asarray([-0.5 * dt_ms], dtype=np_dtype) if include_initial_previous else None
    )
    x_rows_np = _resolve_x_positions_m_numpy(
        axon,
        x_positions_m,
        batch_size=len(rows),
    )
    return _try_build_factorized_footprint_vstim_batch(
        rows,
        t_mid_ms,
        t_initial_previous_ms=t_initial_previous_ms,
        x_rows=x_rows_np,
        np_dtype=np_dtype,
        dtype_local=dtype,
        single_cable_lower=single_cable_lower,
        single_cable_upper=single_cable_upper,
        numeric_axis_shape=numeric_axis_shape,
    )


def with_extracellular_waveform_axis(
    payload: FactorizedExtracellularPotentialBatch,
    axis_input: Any,
    *,
    source_size: int,
    tsim_ms: float,
    dt_ms: float,
    dtype_local: Any,
    include_initial_previous: bool,
) -> FactorizedExtracellularPotentialBatch:
    """Replace factorized currents with an amplitude-major waveform table."""

    waveform_tuple = tuple(axis_input.waveforms)
    axis_size = len(waveform_tuple)
    if axis_size < 1:
        raise ValueError("waveform axis must contain at least one waveform.")
    source_count = int(source_size)
    if source_count < 1:
        raise ValueError("waveform source size must be >= 1.")
    if int(axis_input.source_size) != source_count:
        raise ValueError(
            "waveform-axis source rows do not match the dispatch group: "
            f"got {axis_input.source_size}, expected {source_count}."
        )
    footprint_shape = tuple(int(value) for value in payload.footprint_mV_per_A.shape)
    if len(footprint_shape) not in {2, 3}:
        raise ValueError(
            "compact waveform-axis lowering requires rank-1 or rank-K footprints."
        )
    logical_size = axis_size * source_count
    batch_size = footprint_shape[0]
    if batch_size < logical_size:
        raise ValueError(
            "waveform-axis runtime batch is smaller than its logical axis: "
            f"got {batch_size}, expected at least {logical_size}."
        )

    np_dtype = np.dtype(dtype_local)
    nt = simulation_step_count(tsim_ms, dt_ms)
    t_mid_ms = (
        np.arange(nt, dtype=np_dtype) + np.asarray(0.5, dtype=np_dtype)
    ) * np.asarray(dt_ms, dtype=np_dtype)
    temporal_mid_cache: dict[Any, np.ndarray] = {}
    temporal_mid_hits = 0
    temporal_mid_misses = 0
    selected_current_rows: list[np.ndarray] = []
    for waveform in waveform_tuple:
        values, cache_hit = _cached_stimulus_current_A(
            temporal_mid_cache,
            waveform,
            t_mid_ms,
            np_dtype=np_dtype,
        )
        selected_current_rows.append(values)
        temporal_mid_hits += int(cache_hit)
        temporal_mid_misses += int(not cache_hit)
    selected_current_mid_A = np.stack(selected_current_rows, axis=0)
    row_indices = None
    if len(footprint_shape) == 2:
        if int(axis_input.drive_count) != 1:
            raise ValueError(
                "rank-1 footprint payload cannot lower a multi-drive numeric axis."
            )
        current_mid_A = selected_current_mid_A
        row_indices = np.repeat(np.arange(axis_size, dtype=np.int32), source_count)
        if batch_size > logical_size:
            row_indices = np.concatenate(
                [
                    row_indices,
                    np.full(
                        (batch_size - logical_size,),
                        axis_size - 1,
                        dtype=np.int32,
                    ),
                ]
            )
    else:
        drive_count = int(footprint_shape[1])
        if int(axis_input.drive_count) != drive_count:
            raise ValueError(
                "waveform-axis drive count does not match factorized footprints: "
                f"got {axis_input.drive_count}, expected {drive_count}."
            )
        selected_indices = np.asarray(axis_input.selected_drive_indices, dtype=np.intp)
        unique_pattern_rows: list[tuple[Any, ...]] = []
        pattern_to_index: dict[tuple[Any, ...], int] = {}
        logical_row_indices = np.empty((logical_size,), dtype=np.int32)
        logical_index = 0
        for selected_waveform in waveform_tuple:
            for source_index, source_row in enumerate(axis_input.source_drive_waveforms):
                pattern_row = list(source_row)
                pattern_row[int(selected_indices[source_index])] = selected_waveform
                pattern_tuple = tuple(pattern_row)
                pattern_key = tuple(
                    _stimulus_temporal_cache_key(waveform)
                    for waveform in pattern_tuple
                )
                pattern_index = pattern_to_index.get(pattern_key)
                if pattern_index is None:
                    pattern_index = len(unique_pattern_rows)
                    pattern_to_index[pattern_key] = pattern_index
                    unique_pattern_rows.append(pattern_tuple)
                logical_row_indices[logical_index] = pattern_index
                logical_index += 1

        unique_current_rows: list[np.ndarray] = []
        for pattern_row in unique_pattern_rows:
            drive_values: list[np.ndarray] = []
            for waveform in pattern_row:
                values, cache_hit = _cached_stimulus_current_A(
                    temporal_mid_cache,
                    waveform,
                    t_mid_ms,
                    np_dtype=np_dtype,
                )
                drive_values.append(values)
                temporal_mid_hits += int(cache_hit)
                temporal_mid_misses += int(not cache_hit)
            unique_current_rows.append(np.stack(drive_values, axis=0))
        unique_current_mid_A = np.stack(unique_current_rows, axis=0)
        unique_pattern_count = len(unique_pattern_rows)
        if unique_pattern_count == 1:
            current_mid_A = unique_current_mid_A[0]
        else:
            current_mid_A = unique_current_mid_A
            row_indices = logical_row_indices
        if batch_size > logical_size:
            padding_count = batch_size - logical_size
            if row_indices is None:
                row_indices = np.zeros((logical_size,), dtype=np.int32)
                current_mid_A = unique_current_mid_A
            row_indices = np.concatenate(
                [
                    row_indices,
                    np.full((padding_count,), int(row_indices[-1]), dtype=np.int32),
                ],
            )

    previous_A = None
    if include_initial_previous:
        t_previous_ms = np.asarray([-0.5 * dt_ms], dtype=np_dtype)
        if len(footprint_shape) == 2:
            temporal_previous_cache: dict[Any, np.ndarray] = {}
            previous_A = np.asarray(
                [
                    _cached_stimulus_current_A(
                        temporal_previous_cache,
                        waveform,
                        t_previous_ms,
                        np_dtype=np_dtype,
                    )[0].reshape(-1)[0]
                    for waveform in waveform_tuple
                ],
                dtype=np_dtype,
            )
        else:
            temporal_previous_cache: dict[Any, np.ndarray] = {}
            unique_previous_rows: list[np.ndarray] = []
            for pattern_row in unique_pattern_rows:
                previous_values: list[Any] = []
                for waveform in pattern_row:
                    values, _cache_hit = _cached_stimulus_current_A(
                        temporal_previous_cache,
                        waveform,
                        t_previous_ms,
                        np_dtype=np_dtype,
                    )
                    previous_values.append(values.reshape(-1)[0])
                unique_previous_rows.append(
                    np.asarray(previous_values, dtype=np_dtype)
                )
            unique_previous_A = np.stack(unique_previous_rows, axis=0)
            previous_A = (
                unique_previous_A[0]
                if len(unique_pattern_rows) == 1 and row_indices is None
                else unique_previous_A
            )
    record_benchmark_metadata(
        extracellular_waveform_axis_size=axis_size,
        extracellular_waveform_source_size=source_count,
        extracellular_waveform_logical_batch_size=logical_size,
        extracellular_waveform_kernel_batch_size=batch_size,
        extracellular_waveform_drive_count=int(axis_input.drive_count),
        extracellular_waveform_unique_pattern_count=(
            axis_size if len(footprint_shape) == 2 else len(unique_pattern_rows)
        ),
        extracellular_waveform_temporal_mid_cache_hits=temporal_mid_hits,
        extracellular_waveform_temporal_mid_cache_misses=temporal_mid_misses,
        extracellular_waveform_current_nbytes=int(np.asarray(current_mid_A).nbytes),
        extracellular_waveform_row_indices_nbytes=(
            0 if row_indices is None else int(row_indices.nbytes)
        ),
    )
    return replace(
        payload,
        current_mid_A=jnp.asarray(current_mid_A, dtype=dtype_local),
        current_initial_previous_A=(
            None if previous_A is None else jnp.asarray(previous_A, dtype=dtype_local)
        ),
        current_row_indices=(
            None if row_indices is None else jnp.asarray(row_indices, dtype=jnp.int32)
        ),
        current_row_scales=None,
    )


def build_vstim_midpoint_and_initial_previous_batch(
    axon: object,
    stimulations_batch: Sequence[StimulationBatchRow],
    *,
    tsim_ms: float,
    dt_ms: float,
    x_positions_m: Array | None = None,
    dtype_local: jnp.dtype | None = None,
) -> tuple[Array, Array]:
    """Build double-cable midpoint and previous imposed fields together."""

    dtype = _resolve_dtype(axon, dtype_local)
    nt = simulation_step_count(tsim_ms, dt_ms)
    t_mid_ms = (
        jnp.arange(nt, dtype=dtype) + jnp.asarray(0.5, dtype=dtype)
    ) * jnp.asarray(dt_ms, dtype=dtype)
    t_all_ms = jnp.concatenate(
        [jnp.asarray([-0.5 * dt_ms], dtype=dtype), t_mid_ms],
        axis=0,
    )
    samples = build_vstim_batch(
        axon,
        stimulations_batch,
        t_ms=t_all_ms,
        x_positions_m=x_positions_m,
        dtype_local=dtype,
    )
    return samples[:, 1:, :], samples[:, 0, :]


def build_vstim_batch(
    axon: object,
    stimulations_batch: Sequence[StimulationBatchRow],
    *,
    t_ms: Array,
    x_positions_m: Array | None = None,
    dtype_local: jnp.dtype | None = None,
) -> Array:
    """Build imposed extracellular samples for a batch of stimulation rows.

    ``x_positions_m`` can have shape ``(Nx,)`` for shared positions or
    ``(B, Nx)`` when each row is spatially shifted relative to the electrode.
    """

    rows = tuple(_normalize_stimulation_row(row) for row in stimulations_batch)
    if not rows:
        raise ValueError("stimulations_batch must contain at least one row.")

    dtype = _resolve_dtype(axon, dtype_local)
    t = jnp.asarray(t_ms, dtype=dtype)
    if t.ndim != 1:
        raise ValueError(f"t_ms must have shape (Nt,), got {t.shape}.")

    if not any(rows):
        nx = _resolve_x_positions_width(axon, x_positions_m, batch_size=len(rows))
        return jnp.zeros(
            (len(rows), int(t.shape[0]), nx),
            dtype=dtype,
        )

    if _can_build_stimulation_rows_from_footprints(rows):
        x_rows_np = _resolve_x_positions_m_numpy(
            axon,
            x_positions_m,
            batch_size=len(rows),
        )
        return _build_vstim_batch_from_footprints(
            rows,
            t,
            x_rows=x_rows_np,
            dtype_local=dtype,
        )

    x_rows = _resolve_x_positions_m(
        axon,
        x_positions_m,
        batch_size=len(rows),
        dtype_local=dtype,
    )
    vstim_rows = [
        _build_vstim_row(
            row,
            t,
            x_positions_row_m=x_rows[i],
            dtype_local=dtype,
        )
        for i, row in enumerate(rows)
    ]
    return jnp.stack(vstim_rows, axis=0)


def _can_build_stimulation_rows_from_footprints(
    rows: Sequence[tuple[ExtracellularStimulation, ...]],
) -> bool:
    for row in rows:
        for stimulation in row:
            if not isinstance(stimulation, ExtracellularStimulation):
                return False
            for drive in stimulation.drives:
                if getattr(drive, "stimulus", None) is None:
                    return False
    return True


def _build_vstim_batch_from_footprints(
    rows: Sequence[tuple[ExtracellularStimulation, ...]],
    t_ms: Array,
    *,
    x_rows: Array,
    dtype_local: jnp.dtype,
) -> Array:
    np_dtype = np.dtype(dtype_local)
    t = np.asarray(t_ms, dtype=np_dtype)
    x = np.asarray(x_rows, dtype=float)
    fast_footprint = _try_build_footprint_vstim_batch(
        rows,
        t,
        x_rows=x,
        np_dtype=np_dtype,
        dtype_local=dtype_local,
    )
    if fast_footprint is not None:
        return fast_footprint

    values = np.zeros((len(rows), int(t.shape[0]), int(x.shape[1])), dtype=np_dtype)
    current_cache: dict[int, np.ndarray] = {}
    mV_per_V = np.asarray(1e3, dtype=np_dtype)

    for row_index, row in enumerate(rows):
        for stimulation in row:
            for drive in stimulation.drives:
                stimulus = getattr(drive, "stimulus", None)
                if stimulus is None:
                    raise ValueError(
                        "Each extracellular drive must have an attached stimulus."
                    )
                current_A, _ = _cached_stimulus_current_A(
                    current_cache,
                    stimulus,
                    t,
                    np_dtype=np_dtype,
                )
                footprint = _drive_footprint_for_positions(
                    drive,
                    x[row_index],
                    np_dtype=np_dtype,
                )
                values[row_index] += current_A[:, None] * footprint[None, :] * mV_per_V
    return jnp.asarray(values, dtype=dtype_local)


def _try_build_footprint_vstim_batch(
    rows: Sequence[tuple[ExtracellularStimulation, ...]],
    t_ms: np.ndarray,
    *,
    x_rows: np.ndarray,
    np_dtype: np.dtype[Any],
    dtype_local: jnp.dtype,
) -> Array | None:
    source = _compatible_footprint_rows(rows)
    if source is None:
        return None
    stimulations, drives, row_stimuli = source

    current_cache: dict[int, np.ndarray] = {}
    temporal_cache_hits = 0
    temporal_cache_misses = 0
    current_rows: list[np.ndarray] = []
    for stimulus in row_stimuli:
        current_A, cache_hit = _cached_stimulus_current_A(
            current_cache,
            stimulus,
            t_ms,
            np_dtype=np_dtype,
        )
        temporal_cache_hits += int(cache_hit)
        temporal_cache_misses += int(not cache_hit)
        current_rows.append(current_A)
    current_rows_A = np.stack(current_rows, axis=0)
    cache_key = _footprint_rows_cache_key(
        stimulations,
        drives,
        x_rows=x_rows,
        np_dtype=np_dtype,
    )
    footprint = _FOOTPRINT_CACHE.get(cache_key)
    if footprint is None:
        footprint = _compute_footprint_rows(
            stimulations,
            drives,
            x_rows=x_rows,
            np_dtype=np_dtype,
        )
        _FOOTPRINT_CACHE[cache_key] = footprint
        footprint_cache_status = "miss"
    else:
        footprint_cache_status = "hit"
    record_benchmark_metadata(
        vstim_footprint_cache=footprint_cache_status,
        vstim_footprint_cache_nbytes=int(footprint.nbytes),
        vstim_temporal_cache_hits=int(temporal_cache_hits),
        vstim_temporal_cache_misses=int(temporal_cache_misses),
        vstim_temporal_unique_stimuli=int(temporal_cache_misses),
    )
    values = (
        current_rows_A[:, :, None]
        * footprint[:, None, :]
        * np.asarray(1e3, dtype=np_dtype)
    )
    return jnp.asarray(values, dtype=dtype_local)


def _try_build_factorized_footprint_vstim_batch(
    rows: Sequence[tuple[ExtracellularStimulation, ...]],
    t_ms: np.ndarray,
    *,
    t_initial_previous_ms: np.ndarray | None = None,
    x_rows: np.ndarray,
    np_dtype: np.dtype[Any],
    dtype_local: jnp.dtype,
    single_cable_lower: Array | None = None,
    single_cable_upper: Array | None = None,
    numeric_axis_shape: tuple[int, int] | None = None,
) -> FactorizedExtracellularPotentialBatch | None:
    with benchmark_span("inputs.extracellular.normalize_rows"):
        rows_tuple = _normalize_stimulation_rows(rows)
    with benchmark_span("inputs.extracellular.plan_cache"):
        plan = _factorized_rows_identity_cache_get(
            rows_tuple,
            x_rows=x_rows,
            np_dtype=np_dtype,
        )
        record_benchmark_metadata(
            vstim_factorized_identity_cache="hit" if plan is not None else "miss",
        )
    row_cache_metadata: dict[str, int] = {}
    if plan is not None:
        drive_rows = plan.drive_rows
        max_drive_count = plan.max_drive_count
        footprint_cache_key = plan.footprint_cache_key
        footprint_mV_cache_key = plan.footprint_mV_cache_key
        footprint_V_per_A = plan.footprint_V_per_A
        footprint_mV_per_A = plan.footprint_mV_per_A
        footprint_cache_status = "hit"
        footprint_mV_cache_status = "hit"
    else:
        with benchmark_span("inputs.extracellular.scan_rows"):
            source = _factorizable_footprint_rows(rows_tuple)
        if source is None:
            return None
        drive_rows, max_drive_count, row_cache_hits, row_cache_misses = source
        row_cache_metadata = {
            "vstim_factorized_row_cache_hits": int(row_cache_hits),
            "vstim_factorized_row_cache_misses": int(row_cache_misses),
        }
        with benchmark_span("inputs.extracellular.footprint_key"):
            footprint_cache_key = _factorized_footprint_rows_cache_key(
                drive_rows,
                max_drive_count=max_drive_count,
                x_rows=x_rows,
                np_dtype=np_dtype,
                numeric_axis_shape=numeric_axis_shape,
            )
        with benchmark_span("inputs.extracellular.footprint_cache"):
            footprint_V_per_A = _FOOTPRINT_CACHE.get(footprint_cache_key)
        if footprint_V_per_A is None:
            with benchmark_span("inputs.extracellular.footprint_compute"):
                footprint_V_per_A = _compute_factorized_footprint_rows(
                    drive_rows,
                    max_drive_count=max_drive_count,
                    x_rows=x_rows,
                    np_dtype=np_dtype,
                    numeric_axis_shape=numeric_axis_shape,
                )
            _FOOTPRINT_CACHE[footprint_cache_key] = footprint_V_per_A
            footprint_cache_status = "miss"
        else:
            footprint_cache_status = "hit"

        with benchmark_span("inputs.extracellular.footprint_mv"):
            footprint_mV_cache_key = (
                "factorized_footprint_mV_per_A_v1",
                footprint_cache_key,
                int(max_drive_count),
                str(np_dtype),
            )
            footprint_mV_per_A = _FOOTPRINT_MV_CACHE.get(footprint_mV_cache_key)
            if footprint_mV_per_A is None:
                footprint_mV_per_A = footprint_V_per_A * np.asarray(1e3, dtype=np_dtype)
                if max_drive_count == 1:
                    footprint_mV_per_A = footprint_mV_per_A[:, 0, :]
                footprint_mV_per_A.setflags(write=False)
                _FOOTPRINT_MV_CACHE[footprint_mV_cache_key] = footprint_mV_per_A
                footprint_mV_cache_status = "miss"
            else:
                footprint_mV_cache_status = "hit"

        with benchmark_span("inputs.extracellular.store_plan"):
            _factorized_rows_identity_cache_store(
                rows_tuple,
                x_rows=x_rows,
                np_dtype=np_dtype,
                drive_rows=drive_rows,
                max_drive_count=max_drive_count,
                footprint_cache_key=footprint_cache_key,
                footprint_mV_cache_key=footprint_mV_cache_key,
                footprint_V_per_A=footprint_V_per_A,
                footprint_mV_per_A=footprint_mV_per_A,
            )
    batch_size = len(drive_rows)
    with benchmark_span("inputs.extracellular.shared_rank1_detection"):
        rank1_stimulus_key: tuple[Any, ...] | None = None
        shared_rank1_detection = (
            plan.shared_rank1_detection if plan is not None else "none"
        )
        shared_rank1_stimulus_obj = (
            plan.shared_rank1_stimulus if plan is not None else None
        )
        first_stimulus = drive_rows[0][0][2] if drive_rows and drive_rows[0] else None
        if shared_rank1_stimulus_obj is not None and first_stimulus is not None:
            rank1_stimulus_key = _stimulus_temporal_cache_key(shared_rank1_stimulus_obj)
        elif max_drive_count == 1 and bool(drive_rows) and all(
            len(row) == 1 for row in drive_rows
        ):
            first_stimulus = drive_rows[0][0][2]
            if all(row[0][2] is first_stimulus for row in drive_rows[1:]):
                rank1_stimulus_key = _stimulus_temporal_cache_key(first_stimulus)
                shared_rank1_detection = "identity"
                shared_rank1_stimulus_obj = first_stimulus
        has_shared_rank1_stimulus = rank1_stimulus_key is not None
        if has_shared_rank1_stimulus and shared_rank1_detection == "none":
            shared_rank1_detection = "content"
    temporal_mid_cache_hits = 0
    temporal_mid_cache_misses = 0
    temporal_previous_cache_hits = 0
    temporal_previous_cache_misses = 0
    current_rows_lowering = "none"
    current_row_indices = None
    current_row_scales = None

    if numeric_axis_shape is not None:
        current_A = np.zeros(
            (1,) if max_drive_count == 1 else (max_drive_count, 1),
            dtype=np_dtype,
        )
        current_initial_previous_A = None
        if t_initial_previous_ms is not None:
            current_initial_previous_A = np.zeros(
                () if max_drive_count == 1 else (max_drive_count,),
                dtype=np_dtype,
            )
        shared_current = True
        current_rows_lowering = "deferred_numeric_axis"
    elif has_shared_rank1_stimulus:
        with benchmark_span("inputs.extracellular.current_shared_rank1"):
            stimulus = drive_rows[0][0][2]
            current_A = np.asarray(
                stimulus.evaluate(t_ms, unit="ampere"), dtype=np_dtype
            )
            temporal_mid_cache_hits = max(batch_size - 1, 0)
            temporal_mid_cache_misses = 1
            current_initial_previous_A = None
            if t_initial_previous_ms is not None:
                current_initial_previous_A = np.asarray(
                    stimulus.evaluate(t_initial_previous_ms, unit="ampere"),
                    dtype=np_dtype,
                ).reshape(-1)[0]
                temporal_previous_cache_hits = max(batch_size - 1, 0)
                temporal_previous_cache_misses = 1
            shared_current = True
            current_rows_lowering = "shared_rank1"
    else:
        with benchmark_span("inputs.extracellular.current_scaled_shared_waveform"):
            scaled_waveform_rows = _build_scaled_shared_waveform_rows(
                drive_rows,
                t_ms,
                t_initial_previous_ms=t_initial_previous_ms,
                np_dtype=np_dtype,
            )
        if scaled_waveform_rows is not None:
            current_A = scaled_waveform_rows.current_mid_A
            current_initial_previous_A = (
                scaled_waveform_rows.current_initial_previous_A
            )
            current_row_scales = scaled_waveform_rows.current_row_scales
            shared_current = scaled_waveform_rows.shared_current
            current_rows_lowering = (
                "shared_rank1" if shared_current else "scaled_shared_waveform"
            )
            temporal_mid_cache_hits = scaled_waveform_rows.temporal_mid_cache_hits
            temporal_mid_cache_misses = (
                scaled_waveform_rows.temporal_mid_cache_misses
            )
            temporal_previous_cache_hits = (
                scaled_waveform_rows.temporal_previous_cache_hits
            )
            temporal_previous_cache_misses = (
                scaled_waveform_rows.temporal_previous_cache_misses
            )
        elif max_drive_count == 1:
            with benchmark_span("inputs.extracellular.current_unique_index"):
                rank1_current_rows = _build_rank1_current_rows_from_unique_stimuli(
                    drive_rows,
                    t_ms,
                    t_initial_previous_ms=t_initial_previous_ms,
                    np_dtype=np_dtype,
                )
            if rank1_current_rows is None:
                return None
            current_A = rank1_current_rows.current_mid_A
            current_initial_previous_A = rank1_current_rows.current_initial_previous_A
            current_row_indices = rank1_current_rows.current_row_indices
            shared_current = rank1_current_rows.shared_current
            temporal_mid_cache_hits = rank1_current_rows.temporal_mid_cache_hits
            temporal_mid_cache_misses = rank1_current_rows.temporal_mid_cache_misses
            temporal_previous_cache_hits = (
                rank1_current_rows.temporal_previous_cache_hits
            )
            temporal_previous_cache_misses = (
                rank1_current_rows.temporal_previous_cache_misses
            )
            if shared_current and shared_rank1_detection == "none":
                shared_rank1_detection = "content"
            current_rows_lowering = "shared_rank1" if shared_current else "unique_index"
        else:
            with benchmark_span("inputs.extracellular.current_row_loop"):
                current_rows_A = np.zeros(
                    (batch_size, max_drive_count, int(t_ms.shape[0])),
                    dtype=np_dtype,
                )
                previous_rows_A = (
                    None
                    if t_initial_previous_ms is None
                    else np.zeros((batch_size, max_drive_count), dtype=np_dtype)
                )
                mid_cache: dict[Any, np.ndarray] = {}
                previous_cache: dict[Any, np.ndarray] = {}
                for row_index, row in enumerate(drive_rows):
                    for drive_index, (_stimulation, _drive, stimulus) in enumerate(row):
                        current_values_A, cache_hit = _cached_stimulus_current_A(
                            mid_cache,
                            stimulus,
                            t_ms,
                            np_dtype=np_dtype,
                        )
                        temporal_mid_cache_hits += int(cache_hit)
                        temporal_mid_cache_misses += int(not cache_hit)
                        current_rows_A[row_index, drive_index] = current_values_A
                        if previous_rows_A is not None and t_initial_previous_ms is not None:
                            previous_values_A, cache_hit = _cached_stimulus_current_A(
                                previous_cache,
                                stimulus,
                                t_initial_previous_ms,
                                np_dtype=np_dtype,
                            )
                            temporal_previous_cache_hits += int(cache_hit)
                            temporal_previous_cache_misses += int(not cache_hit)
                            previous_rows_A[row_index, drive_index] = (
                                previous_values_A.reshape(-1)[0]
                            )

                shared_current = False
                current_A = current_rows_A
                current_initial_previous_A = previous_rows_A
                current_rows_lowering = "row_loop"

    dense_equivalent_nbytes = (
        int(len(rows_tuple))
        * int(t_ms.shape[0])
        * int(x_rows.shape[1])
        * int(np_dtype.itemsize)
    )
    previous_nbytes = (
        0
        if current_initial_previous_A is None
        else int(np.asarray(current_initial_previous_A).nbytes)
    )
    current_indices_nbytes = (
        0 if current_row_indices is None else int(current_row_indices.nbytes)
    )
    current_scales_nbytes = (
        0 if current_row_scales is None else int(current_row_scales.nbytes)
    )
    forcing_footprint_mV_per_A = None
    forcing_nbytes = 0
    forcing_cache_status = "disabled"
    if single_cable_lower is not None and single_cable_upper is not None:
        with benchmark_span("inputs.extracellular.single_cable_forcing_cache"):
            forcing_cache_key = _single_cable_forcing_footprint_cache_key(
                footprint_mV_cache_key,
                single_cable_lower,
                single_cable_upper,
                np_dtype=np_dtype,
            )
            forcing_footprint_mV_per_A = _SINGLE_CABLE_FORCING_MV_CACHE.get(
                forcing_cache_key
            )
            forcing_cache_status = (
                "hit" if forcing_footprint_mV_per_A is not None else "miss"
            )
        if forcing_footprint_mV_per_A is None:
            with benchmark_span("inputs.extracellular.single_cable_forcing_compute"):
                forcing_footprint_mV_per_A = (
                    _compute_single_cable_forcing_footprint_numpy(
                        footprint_mV_per_A,
                        lower=single_cable_lower,
                        upper=single_cable_upper,
                        np_dtype=np_dtype,
                    )
                )
                forcing_footprint_mV_per_A.setflags(write=False)
                _SINGLE_CABLE_FORCING_MV_CACHE[forcing_cache_key] = (
                    forcing_footprint_mV_per_A
                )
    forcing_nbytes = (
        0
        if forcing_footprint_mV_per_A is None
        else int(forcing_footprint_mV_per_A.nbytes)
    )
    factorized_nbytes = (
        int(current_A.nbytes)
        + int(footprint_mV_per_A.nbytes)
        + forcing_nbytes
        + previous_nbytes
        + current_indices_nbytes
        + current_scales_nbytes
    )
    record_benchmark_metadata(
        vstim_footprint_cache=footprint_cache_status,
        vstim_footprint_cache_nbytes=int(footprint_V_per_A.nbytes),
        vstim_input_format="factorized_footprint",
        vstim_factorized_rank=int(max_drive_count),
        vstim_factorized_nstim=int(max_drive_count),
        vstim_factorized_current_nbytes=int(current_A.nbytes),
        vstim_factorized_initial_previous_nbytes=previous_nbytes,
        vstim_factorized_current_indices_nbytes=current_indices_nbytes,
        vstim_factorized_current_scales_nbytes=current_scales_nbytes,
        vstim_factorized_footprint_nbytes=int(footprint_mV_per_A.nbytes),
        vstim_single_cable_forcing_footprint_cache=forcing_cache_status,
        vstim_single_cable_forcing_footprint_nbytes=forcing_nbytes,
        vstim_factorized_total_nbytes=factorized_nbytes,
        vstim_dense_equivalent_nbytes=dense_equivalent_nbytes,
        shared_current=bool(shared_current),
        scaled_shared_waveform=current_row_scales is not None,
        vstim_shared_current_detection=shared_rank1_detection,
        vstim_temporal_cache_hits=int(temporal_mid_cache_hits),
        vstim_temporal_cache_misses=int(temporal_mid_cache_misses),
        vstim_temporal_previous_cache_hits=int(temporal_previous_cache_hits),
        vstim_temporal_previous_cache_misses=int(temporal_previous_cache_misses),
        vstim_temporal_unique_stimuli=int(temporal_mid_cache_misses),
        vstim_temporal_unique_patterns=int(temporal_mid_cache_misses),
        vstim_current_rows_lowering=current_rows_lowering,
        vstim_footprint_mv_cache=footprint_mV_cache_status,
        vstim_factorized_identity_cache="hit" if plan is not None else "miss",
        vstim_factorized_dense_ratio=(
            factorized_nbytes / float(dense_equivalent_nbytes)
            if dense_equivalent_nbytes
            else 0.0
        ),
        **row_cache_metadata,
    )
    with benchmark_span("inputs.extracellular.footprint_to_device"):
        footprint_jax, footprint_jax_cache_status = _cached_jax_footprint_array(
            footprint_mV_per_A,
            cache_key=footprint_mV_cache_key,
            dtype_local=dtype_local,
        )
    record_benchmark_metadata(vstim_footprint_jax_cache=footprint_jax_cache_status)
    with benchmark_span("inputs.extracellular.current_to_device"):
        current_mid_A = jnp.asarray(current_A, dtype=dtype_local)
        current_initial_previous_device = (
            None
            if current_initial_previous_A is None
            else jnp.asarray(current_initial_previous_A, dtype=dtype_local)
        )
        current_row_indices_device = (
            None
            if current_row_indices is None
            else jnp.asarray(current_row_indices, dtype=jnp.int32)
        )
        current_row_scales_device = (
            None
            if current_row_scales is None
            else jnp.asarray(current_row_scales, dtype=dtype_local)
        )
    with benchmark_span("inputs.extracellular.single_cable_forcing_to_device"):
        forcing_footprint_device = (
            None
            if forcing_footprint_mV_per_A is None
            else jnp.asarray(forcing_footprint_mV_per_A, dtype=dtype_local)
        )
    return FactorizedExtracellularPotentialBatch(
        current_mid_A=current_mid_A,
        footprint_mV_per_A=footprint_jax,
        target_nx=int(x_rows.shape[1]),
        current_initial_previous_A=current_initial_previous_device,
        single_cable_forcing_footprint_mV_per_A=forcing_footprint_device,
        current_row_indices=current_row_indices_device,
        current_row_scales=current_row_scales_device,
    )


def _single_cable_forcing_footprint_cache_key(
    footprint_mV_cache_key: tuple[Any, ...],
    lower: Array,
    upper: Array,
    *,
    np_dtype: np.dtype[Any],
) -> tuple[Any, ...]:
    lower_np = np.asarray(lower, dtype=np_dtype)
    upper_np = np.asarray(upper, dtype=np_dtype)
    return (
        "single_cable_forcing_footprint_mV_per_A_v1",
        footprint_mV_cache_key,
        _array_content_key(lower_np),
        _array_content_key(upper_np),
        str(np_dtype),
    )


def _compute_single_cable_forcing_footprint_numpy(
    footprint_mV_per_A: np.ndarray,
    *,
    lower: Array,
    upper: Array,
    np_dtype: np.dtype[Any],
) -> np.ndarray:
    footprint = np.asarray(footprint_mV_per_A, dtype=np_dtype)
    lower_rows = _as_single_cable_operator_rows_numpy(
        lower,
        batch_size=int(footprint.shape[0]),
        nx=int(footprint.shape[-1]),
        np_dtype=np_dtype,
    )
    upper_rows = _as_single_cable_operator_rows_numpy(
        upper,
        batch_size=int(footprint.shape[0]),
        nx=int(footprint.shape[-1]),
        np_dtype=np_dtype,
    )
    if footprint.ndim == 3:
        batch_size, drive_count, nx = footprint.shape
        flattened = footprint.reshape((batch_size * drive_count, nx))
        lower_rows = np.broadcast_to(
            lower_rows[:, None, :],
            (batch_size, drive_count, nx),
        ).reshape((batch_size * drive_count, nx))
        upper_rows = np.broadcast_to(
            upper_rows[:, None, :],
            (batch_size, drive_count, nx),
        ).reshape((batch_size * drive_count, nx))
        forcing = _compute_single_cable_forcing_footprint_numpy(
            flattened,
            lower=lower_rows,
            upper=upper_rows,
            np_dtype=np_dtype,
        )
        return forcing.reshape((batch_size, drive_count, nx))
    if footprint.ndim != 2:
        raise ValueError(
            "factorized single-cable footprints must have shape (B, Nx) or (B, K, Nx), "
            f"got {footprint.shape}."
        )
    nx = int(footprint.shape[1])
    if nx < 2:
        return np.zeros_like(footprint)
    forcing = np.empty_like(footprint)
    forcing[:, :1] = upper_rows[:, :1] * (footprint[:, 1:2] - footprint[:, :1])
    forcing[:, -1:] = lower_rows[:, -1:] * (footprint[:, -2:-1] - footprint[:, -1:])
    if nx > 2:
        forcing[:, 1:-1] = (
            lower_rows[:, 1:-1] * (footprint[:, :-2] - footprint[:, 1:-1])
            + upper_rows[:, 1:-1] * (footprint[:, 2:] - footprint[:, 1:-1])
        )
    return forcing


def _as_single_cable_operator_rows_numpy(
    values: Array,
    *,
    batch_size: int,
    nx: int,
    np_dtype: np.dtype[Any],
) -> np.ndarray:
    arr = np.asarray(values, dtype=np_dtype)
    if arr.ndim == 1:
        if arr.shape != (nx,):
            raise ValueError(
                f"single-cable operator must have shape (Nx,)=({nx},) or (B, Nx), "
                f"got {arr.shape}."
            )
        return np.broadcast_to(arr[None, :], (batch_size, nx))
    if arr.ndim == 2:
        if arr.shape[1:] != (nx,):
            raise ValueError(
                f"single-cable operator must have trailing shape (Nx,)=({nx},), "
                f"got {arr.shape}."
            )
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            return np.broadcast_to(arr, (batch_size, nx))
        raise ValueError(
            f"single-cable operator batch size must be 1 or {batch_size}, "
            f"got {arr.shape[0]}."
        )
    raise ValueError(
        f"single-cable operator must have shape (Nx,) or (B, Nx), got {arr.shape}."
    )


def _cached_jax_footprint_array(
    values: np.ndarray,
    *,
    cache_key: tuple[Any, ...],
    dtype_local: jnp.dtype,
) -> tuple[Array, str]:
    """Return a device-local JAX footprint for a cached static footprint."""

    device_key = _current_jax_device_key()
    key = (
        "factorized_footprint_jax_array_v1",
        cache_key,
        str(np.dtype(dtype_local)),
        device_key,
    )
    cached = _FOOTPRINT_JAX_CACHE.get(key)
    if cached is not None:
        _FOOTPRINT_JAX_CACHE.move_to_end(key)
        record_benchmark_metadata(vstim_footprint_jax_cache="hit")
        return cached, "hit"

    array = jnp.asarray(values, dtype=dtype_local)
    _FOOTPRINT_JAX_CACHE[key] = array
    _FOOTPRINT_JAX_CACHE.move_to_end(key)
    while len(_FOOTPRINT_JAX_CACHE) > _FOOTPRINT_JAX_CACHE_MAX_SIZE:
        _FOOTPRINT_JAX_CACHE.popitem(last=False)
    record_benchmark_metadata(vstim_footprint_jax_cache="miss")
    return array, "miss"


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
        str(device),
    )


def _factorizable_footprint_rows(
    rows: Sequence[tuple[ExtracellularStimulation, ...]],
) -> (
    tuple[
        tuple[tuple[tuple[ExtracellularStimulation, Any, Stimulus], ...], ...],
        int,
        int,
        int,
    ]
    | None
):
    if not rows:
        return None
    drive_rows: list[tuple[tuple[ExtracellularStimulation, Any, Stimulus], ...]] = []
    max_drive_count = 0
    row_cache_hits = 0
    row_cache_misses = 0
    local_row_cache: dict[
        tuple[int, ...],
        tuple[tuple[ExtracellularStimulation, Any, Stimulus], ...],
    ] = {}
    for row in rows:
        row_key = tuple(id(stimulation) for stimulation in row)
        cached_row = local_row_cache.get(row_key)
        if cached_row is None:
            row_cache_misses += 1
            row_drives: list[tuple[ExtracellularStimulation, Any, Stimulus]] = []
            for stimulation in row:
                if not isinstance(stimulation, ExtracellularStimulation):
                    return None
                for drive in stimulation.drives:
                    stimulus = getattr(drive, "stimulus", None)
                    if stimulus is None:
                        return None
                    row_drives.append((stimulation, drive, stimulus))
            row_drive_tuple = tuple(row_drives)
            local_row_cache[row_key] = row_drive_tuple
        else:
            row_cache_hits += 1
            row_drive_tuple = cached_row
        max_drive_count = max(max_drive_count, len(row_drive_tuple))
        drive_rows.append(row_drive_tuple)
    if max_drive_count == 0:
        return None
    record_benchmark_metadata(
        vstim_factorized_row_cache_hits=int(row_cache_hits),
        vstim_factorized_row_cache_misses=int(row_cache_misses),
    )
    return (
        tuple(drive_rows),
        int(max_drive_count),
        int(row_cache_hits),
        int(row_cache_misses),
    )


def _compute_factorized_footprint_rows(
    drive_rows: Sequence[Sequence[tuple[ExtracellularStimulation, Any, Stimulus]]],
    *,
    max_drive_count: int,
    x_rows: np.ndarray,
    np_dtype: np.dtype[Any],
    numeric_axis_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    source_size, axis_size, logical_size = _numeric_axis_expansion(
        len(drive_rows),
        numeric_axis_shape,
    )
    compute_rows = drive_rows if source_size is None else drive_rows[:source_size]
    compute_x_rows = x_rows if source_size is None else x_rows[:source_size]
    footprint = np.zeros(
        (len(compute_rows), int(max_drive_count), int(x_rows.shape[1])),
        dtype=np_dtype,
    )
    for row_index, row in enumerate(compute_rows):
        for drive_index, (_stimulation, drive, _stimulus) in enumerate(row):
            footprint[row_index, drive_index] = _drive_footprint_for_positions(
                drive,
                compute_x_rows[row_index],
                np_dtype=np_dtype,
            )
    if source_size is not None and axis_size is not None:
        footprint = np.tile(footprint, (axis_size, 1, 1))
        padding_count = len(drive_rows) - logical_size
        if padding_count:
            footprint = np.concatenate(
                [footprint, np.repeat(footprint[-1:], padding_count, axis=0)],
                axis=0,
            )
        record_benchmark_metadata(
            vstim_footprint_source_rows=source_size,
            vstim_footprint_expanded_rows=len(drive_rows),
            vstim_footprint_axis_size=axis_size,
        )
    return footprint


def _factorized_footprint_rows_cache_key(
    drive_rows: Sequence[Sequence[tuple[ExtracellularStimulation, Any, Stimulus]]],
    *,
    max_drive_count: int,
    x_rows: np.ndarray,
    np_dtype: np.dtype[Any],
    numeric_axis_shape: tuple[int, int] | None = None,
) -> tuple[Any, ...]:
    source_size, axis_size, _logical_size = _numeric_axis_expansion(
        len(drive_rows),
        numeric_axis_shape,
    )
    key_drive_rows = drive_rows if source_size is None else drive_rows[:source_size]
    key_x_rows = x_rows if source_size is None else x_rows[:source_size]
    rows_digest = _digest_cache_rows(
        tuple(
            _drive_static_footprint_key(drive)
            for _stimulation, drive, _stimulus in row
        )
        for row in key_drive_rows
    )
    return (
        "factorized_footprint_rows_v3",
        str(np_dtype),
        int(max_drive_count),
        len(drive_rows),
        source_size,
        axis_size,
        rows_digest,
        _array_content_key(key_x_rows),
    )


def _numeric_axis_expansion(
    row_count: int,
    numeric_axis_shape: tuple[int, int] | None,
) -> tuple[int | None, int | None, int]:
    if numeric_axis_shape is None:
        return None, None, int(row_count)
    source_size, axis_size = (int(value) for value in numeric_axis_shape)
    if source_size < 1 or axis_size < 1:
        raise ValueError("numeric-axis source and axis sizes must be positive.")
    logical_size = source_size * axis_size
    if logical_size > int(row_count):
        raise ValueError(
            "numeric-axis footprint expansion exceeds the runtime batch: "
            f"got {logical_size} logical rows for {row_count} runtime rows."
        )
    return source_size, axis_size, logical_size


def _compatible_footprint_rows(
    rows: Sequence[tuple[ExtracellularStimulation, ...]],
) -> tuple[
    tuple[ExtracellularStimulation, ...],
    tuple[Any, ...],
    tuple[Stimulus, ...],
] | None:
    if not rows or any(len(row) != 1 for row in rows):
        return None
    stimulations: list[ExtracellularStimulation] = []
    drives: list[Any] = []
    stimuli = []
    for row in rows:
        stimulation = row[0]
        if len(stimulation.drives) != 1:
            return None
        drive = stimulation.drives[0]
        stimulus = getattr(drive, "stimulus", None)
        if stimulus is None:
            return None
        stimulations.append(stimulation)
        drives.append(drive)
        stimuli.append(stimulus)
    return tuple(stimulations), tuple(drives), tuple(stimuli)


def _compute_footprint_rows(
    stimulations: Sequence[ExtracellularStimulation],
    drives: Sequence[Any],
    *,
    x_rows: np.ndarray,
    np_dtype: np.dtype[Any],
) -> np.ndarray:
    return np.stack(
        [
            _drive_footprint_for_positions(
                drive,
                x_rows[row_index],
                np_dtype=np_dtype,
            )
            for row_index, (_stimulation, drive) in enumerate(
                zip(stimulations, drives, strict=True)
            )
        ],
        axis=0,
    )


def _footprint_rows_cache_key(
    stimulations: Sequence[ExtracellularStimulation],
    drives: Sequence[Any],
    *,
    x_rows: np.ndarray,
    np_dtype: np.dtype[Any],
) -> tuple[Any, ...]:
    rows_digest = _digest_cache_rows(
        _drive_static_footprint_key(drive)
        for _stimulation, drive in zip(stimulations, drives, strict=True)
    )
    return (
        "footprint_rows_v2",
        str(np_dtype),
        len(drives),
        rows_digest,
        _array_content_key(x_rows),
    )


def _drive_static_footprint_key(drive: Any) -> tuple[Any, ...]:
    footprint = getattr(drive, "footprint", None)
    if footprint is None:
        return ("drive_identity", id(drive))
    return (
        "static_footprint_v1",
        getattr(drive, "id", None),
        id(footprint),
        getattr(footprint, "interpolation", None),
        getattr(footprint, "source_id", None),
        getattr(footprint, "reference", None),
    )


def _digest_cache_rows(rows: Iterable[Any]) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for row in rows:
        hasher.update(repr(row).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _factorized_rows_identity_cache_get(
    rows: tuple[tuple[ExtracellularStimulation, ...], ...],
    *,
    x_rows: np.ndarray,
    np_dtype: np.dtype[Any],
) -> _FactorizedRowsIdentityPlan | None:
    key = _factorized_rows_identity_cache_key(
        rows,
        x_rows=x_rows,
        np_dtype=np_dtype,
    )
    cached = _FACTORIZED_ROWS_IDENTITY_CACHE.get(key)
    if cached is None:
        return None
    if (
        cached.rows is rows
        and cached.x_rows is x_rows
        and cached.np_dtype_str == str(np_dtype)
    ):
        _FACTORIZED_ROWS_IDENTITY_CACHE.move_to_end(key)
        return cached
    _FACTORIZED_ROWS_IDENTITY_CACHE.pop(key, None)
    return None


def _factorized_rows_identity_cache_store(
    rows: tuple[tuple[ExtracellularStimulation, ...], ...],
    *,
    x_rows: np.ndarray,
    np_dtype: np.dtype[Any],
    drive_rows: tuple[
        tuple[tuple[ExtracellularStimulation, Any, Stimulus], ...], ...
    ],
    max_drive_count: int,
    footprint_cache_key: tuple[Any, ...],
    footprint_mV_cache_key: tuple[Any, ...],
    footprint_V_per_A: np.ndarray,
    footprint_mV_per_A: np.ndarray,
) -> None:
    shared_stimulus, shared_detection = _shared_rank1_stimulus_identity(
        drive_rows,
        max_drive_count=max_drive_count,
    )
    key = _factorized_rows_identity_cache_key(
        rows,
        x_rows=x_rows,
        np_dtype=np_dtype,
    )
    _FACTORIZED_ROWS_IDENTITY_CACHE[key] = _FactorizedRowsIdentityPlan(
        rows=rows,
        x_rows=x_rows,
        np_dtype_str=str(np_dtype),
        drive_rows=drive_rows,
        max_drive_count=int(max_drive_count),
        shared_rank1_stimulus=shared_stimulus,
        shared_rank1_detection=shared_detection,
        footprint_cache_key=footprint_cache_key,
        footprint_mV_cache_key=footprint_mV_cache_key,
        footprint_V_per_A=footprint_V_per_A,
        footprint_mV_per_A=footprint_mV_per_A,
    )
    _FACTORIZED_ROWS_IDENTITY_CACHE.move_to_end(key)
    while len(_FACTORIZED_ROWS_IDENTITY_CACHE) > _FACTORIZED_ROWS_IDENTITY_CACHE_MAX_SIZE:
        _FACTORIZED_ROWS_IDENTITY_CACHE.popitem(last=False)


def _factorized_rows_identity_cache_key(
    rows: tuple[tuple[ExtracellularStimulation, ...], ...],
    *,
    x_rows: np.ndarray,
    np_dtype: np.dtype[Any],
) -> tuple[Any, ...]:
    return (
        "factorized_rows_identity_v1",
        id(rows),
        len(rows),
        id(x_rows),
        tuple(int(dim) for dim in x_rows.shape),
        x_rows.dtype.str,
        tuple(int(stride) for stride in x_rows.strides),
        str(np_dtype),
    )


def _shared_rank1_stimulus_identity(
    drive_rows: Sequence[Sequence[tuple[ExtracellularStimulation, Any, Stimulus]]],
    *,
    max_drive_count: int,
) -> tuple[Stimulus | None, str]:
    if max_drive_count != 1 or not drive_rows:
        return None, "none"
    if not all(len(row) == 1 for row in drive_rows):
        return None, "none"
    first = drive_rows[0][0][2]
    if all(row[0][2] is first for row in drive_rows[1:]):
        return first, "identity"
    return None, "none"


def _drive_footprint_for_positions(
    drive: Any,
    x_positions_m: Array,
    *,
    np_dtype: np.dtype[Any],
) -> np.ndarray:
    footprint = drive.footprint
    values = np.asarray(footprint.values_for_axon(), dtype=np_dtype)
    x_um = np.asarray(x_positions_m, dtype=float) * 1e6
    support_um = np.asarray(footprint.positions_um, dtype=float)
    if x_um.shape == support_um.shape and np.allclose(x_um, support_um):
        return values
    if footprint.interpolation not in {"sampled", "linear"}:
        raise NotImplementedError(
            "Only sampled/linear footprint interpolation is supported by "
            "batch extracellular lowering."
        )
    if np.any(np.diff(support_um) < 0.0):
        raise ValueError("Footprint positions must be sorted for interpolation.")
    return np.asarray(np.interp(x_um, support_um, values), dtype=np_dtype)


def _build_vstim_row(
    stimulations: tuple[ExtracellularStimulation, ...],
    t_ms: Array,
    *,
    x_positions_row_m: Array,
    dtype_local: jnp.dtype,
) -> Array:
    nt = int(t_ms.shape[0])
    nx = int(x_positions_row_m.shape[0])
    np_dtype = np.dtype(dtype_local)
    t = np.asarray(t_ms, dtype=np_dtype)
    vstim = np.zeros((nt, nx), dtype=np_dtype)
    for stimulation in stimulations:
        for drive in stimulation.drives:
            footprint = _drive_footprint_for_positions(
                drive,
                np.asarray(x_positions_row_m, dtype=float),
                np_dtype=np_dtype,
            )
            current = np.asarray(
                drive.stimulus.evaluate(t, unit="ampere"),
                dtype=np_dtype,
            )
            vstim = vstim + current[:, None] * footprint[None, :]
    return jnp.asarray(vstim * np.asarray(1e3, dtype=np_dtype), dtype=dtype_local)


def _normalize_stimulation_row(
    row: StimulationBatchRow,
) -> tuple[ExtracellularStimulation, ...]:
    if row is None:
        return ()
    if isinstance(row, ExtracellularStimulation):
        return (row,)
    return tuple(row)


def _normalize_stimulation_rows(
    rows: Sequence[StimulationBatchRow],
) -> tuple[tuple[ExtracellularStimulation, ...], ...]:
    if isinstance(rows, tuple) and all(isinstance(row, tuple) for row in rows):
        return cast(tuple[tuple[ExtracellularStimulation, ...], ...], rows)
    return tuple(_normalize_stimulation_row(row) for row in rows)


def _resolve_dtype(axon: object, dtype_local: jnp.dtype | None) -> jnp.dtype:
    if dtype_local is not None:
        return dtype_local
    for attr in ("dtype",):
        dtype = getattr(axon, attr, None)
        if dtype is not None:
            return _jax_scalar_dtype(dtype)
    membrane_models = getattr(axon, "membrane_models", None)
    if membrane_models:
        return _jax_scalar_dtype(membrane_models[0].dtype)
    return jnp.float32


def _jax_scalar_dtype(dtype_like: object) -> jnp.dtype:
    return jnp.float64 if np.dtype(cast(Any, dtype_like)).name == "float64" else jnp.float32


def _resolve_x_positions_m(
    axon: object,
    x_positions_m: Array | None,
    *,
    batch_size: int,
    dtype_local: jnp.dtype,
) -> Array:
    if x_positions_m is None:
        layout = getattr(axon, "layout", None)
        if layout is None:
            raise AttributeError("axon must expose a layout for spatial sampling.")
        x_um = layout.position_values(unit="micrometer")
        x = jnp.asarray(x_um, dtype=dtype_local) * jnp.asarray(1e-6, dtype=dtype_local)
    else:
        x = jnp.asarray(x_positions_m, dtype=dtype_local)

    if x.ndim == 1:
        return jnp.broadcast_to(x[jnp.newaxis, :], (batch_size, x.shape[0]))
    if x.ndim == 2:
        if x.shape[0] != batch_size:
            raise ValueError(
                f"x_positions_m has batch size {x.shape[0]}, expected {batch_size}."
            )
        return x
    raise ValueError(f"x_positions_m must have shape (Nx,) or (B, Nx), got {x.shape}.")


def _resolve_x_positions_width(
    axon: object,
    x_positions_m: Array | None,
    *,
    batch_size: int,
) -> int:
    if x_positions_m is None:
        layout = getattr(axon, "layout", None)
        if layout is None:
            raise AttributeError("axon must expose a layout for spatial sampling.")
        return int(np.asarray(layout.position_values(unit="micrometer")).shape[0])
    shape, ndim = _shape_and_ndim(x_positions_m)
    if ndim == 1:
        return int(shape[0])
    if ndim == 2:
        if int(shape[0]) != batch_size:
            raise ValueError(
                f"x_positions_m has batch size {shape[0]}, expected {batch_size}."
            )
        return int(shape[1])
    raise ValueError(f"x_positions_m must have shape (Nx,) or (B, Nx), got {shape}.")


def _resolve_x_positions_m_numpy(
    axon: object,
    x_positions_m: Array | None,
    *,
    batch_size: int,
) -> np.ndarray:
    if x_positions_m is None:
        layout = getattr(axon, "layout", None)
        if layout is None:
            raise AttributeError("axon must expose a layout for spatial sampling.")
        x_um = layout.position_values(unit="micrometer")
        x = np.asarray(x_um, dtype=float) * 1e-6
    else:
        x = np.asarray(x_positions_m, dtype=float)

    if x.ndim == 1:
        return np.broadcast_to(x[None, :], (batch_size, x.shape[0]))
    if x.ndim == 2:
        if x.shape[0] != batch_size:
            raise ValueError(
                f"x_positions_m has batch size {x.shape[0]}, expected {batch_size}."
            )
        return x
    raise ValueError(f"x_positions_m must have shape (Nx,) or (B, Nx), got {x.shape}.")


def _shape_and_ndim(values: object) -> tuple[tuple[int, ...], int]:
    shape = getattr(values, "shape", None)
    ndim = getattr(values, "ndim", None)
    if shape is not None and ndim is not None:
        return tuple(int(dim) for dim in shape), int(ndim)
    arr = np.asarray(values)
    return arr.shape, arr.ndim
