"""Recruitment sweep protocol."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from axonscope.analysis import ActivationCriterion
from axonscope.axon_instance import AxonInstance
from axonscope.protocols.observer_path import (
    _can_use_activation_observer,
    _evaluate_activation_observer_pool,
)
from axonscope.protocols.progress import (
    _OneShotProgress,
    _SweepProgress,
    _activation_progress_summary,
)
from axonscope.protocols.results import PoolSweepResult, RecruitmentCurve
from axonscope.protocols.sweep import _apply_pool_update, pool_sweep
from axonscope.protocols.types import PoolUpdate, SimulationCandidate
from axonscope.protocols.values import _normalize_sweep_values, _require_current_array_uA
from axonscope.recording import Recording
from axonscope.runtime import ExecutionPolicy
from axonscope.runtime.benchmarking import benchmark_span
from axonscope.solvers import BatchOptions
from axonscope.utils import units


def recruitment_sweep(
    pool: Sequence[SimulationCandidate],
    *,
    update: PoolUpdate,
    values: Any,
    duration: Any,
    dt: Any,
    criterion: ActivationCriterion,
    recording: Recording | None = None,
    batch_options: BatchOptions | None = None,
    execution_policy: ExecutionPolicy | None = None,
    progress: bool | str = False,
    solver_progress: bool | str = False,
    batch_amplitudes: bool = False,
    amplitude_batch_size: int | None = None,
) -> RecruitmentCurve:
    """Evaluate pool recruitment over sampled current values.

    ``pool`` contains the simulations to evaluate. ``update(simulation,
    amplitude)`` is called before each run to change the swept parameter,
    usually an electrode current. The update function may mutate the row and
    return ``None``, or return a replacement simulation. ``batch_options``
    forwards solver-side execution knobs such as observer time chunking.
    Amplitudes are evaluated sequentially by default so memory scales with
    ``n_rows``, not ``n_rows * n_amplitudes``. Pass ``batch_amplitudes=True``
    to build native expanded pools for observer-only activation sweeps.
    ``amplitude_batch_size`` controls how many amplitude values are packed into
    each expanded pool; ``None`` means all values at once.
    ``solver_progress`` is forwarded only to the first solver call so cold-start
    compilation remains visible without logging every sampled amplitude.
    """

    amplitudes_uA = _require_current_array_uA(values, name="values")
    if amplitudes_uA.ndim != 1:
        raise ValueError("values must be a 1D current array.")
    if _can_use_activation_observer(recording, criterion):
        sweep = _activation_pool_sweep(
            pool,
            update=update,
            values=units.Q_(amplitudes_uA, "microampere"),
            duration=duration,
            dt=dt,
            criterion=criterion,
            progress=progress,
            batch_options=batch_options,
            execution_policy=execution_policy,
            solver_progress=solver_progress,
            batch_amplitudes=batch_amplitudes,
            amplitude_batch_size=amplitude_batch_size,
        )
        return RecruitmentCurve(
            amplitudes_uA=np.asarray(amplitudes_uA, dtype=float),
            activated=np.asarray(sweep.observations, dtype=bool),
        )
    if batch_amplitudes:
        raise ValueError(
            "batch_amplitudes=True requires an observer-only recruitment path; "
            "use recording=None or Recording.none() with a compatible activation criterion."
        )
    if amplitude_batch_size is not None:
        raise ValueError(
            "amplitude_batch_size requires an observer-only recruitment path; "
            "use recording=None or Recording.none() with a compatible activation criterion."
        )
    sweep = pool_sweep(
        pool,
        update=update,
        values=units.Q_(amplitudes_uA, "microampere"),
        observe=lambda result: criterion.evaluate(result).activated,
        duration=duration,
        dt=dt,
        recording=recording,
        batch_options=batch_options,
        execution_policy=execution_policy,
        progress=progress,
        progress_summary=_activation_progress_summary,
        solver_progress=solver_progress,
    )
    return RecruitmentCurve(
        amplitudes_uA=np.asarray(amplitudes_uA, dtype=float),
        activated=np.asarray(sweep.observations, dtype=bool),
    )


def _activation_pool_sweep(
    pool: Sequence[SimulationCandidate],
    *,
    update: PoolUpdate,
    values: Sequence[Any],
    duration: Any,
    dt: Any,
    criterion: ActivationCriterion,
    progress: bool | str = False,
    batch_options: BatchOptions | None = None,
    execution_policy: ExecutionPolicy | None = None,
    solver_progress: bool | str = False,
    batch_amplitudes: bool = False,
    amplitude_batch_size: int | None = None,
) -> PoolSweepResult:
    """Sweep activation with solver-side observers instead of stored Vm traces."""

    base_pool = tuple(pool)
    value_tuple = _normalize_sweep_values(values)
    if len(base_pool) == 0:
        return PoolSweepResult(
            values=value_tuple,
            observations=np.zeros((len(value_tuple), 0), dtype=bool),
        )
    if batch_amplitudes:
        return _activation_pool_sweep_batched_amplitudes(
            base_pool,
            update=update,
            values=value_tuple,
            duration=duration,
            dt=dt,
            criterion=criterion,
            progress=progress,
            batch_options=batch_options,
            execution_policy=execution_policy,
            solver_progress=solver_progress,
            amplitude_batch_size=amplitude_batch_size,
        )
    if amplitude_batch_size is not None:
        raise ValueError("amplitude_batch_size requires batch_amplitudes=True.")

    progress_display = _SweepProgress(progress)
    solver_progress_gate = _OneShotProgress(solver_progress)
    observation_rows: list[np.ndarray] = []
    try:
        for index, value in enumerate(value_tuple):
            progress_display.begin(
                label="Pool sweep",
                current_index=index,
                values=value_tuple,
                completed_rows=observation_rows,
                progress_summary=_activation_progress_summary,
            )
            with benchmark_span(
                "protocol.sweep.value",
                index=index,
                value=str(value),
                pool_size=len(base_pool),
            ):
                updated_pool = tuple(
                    _apply_pool_update(row, update, value) for row in base_pool
                )
                observations = _evaluate_activation_observer_pool(
                    updated_pool,
                    criterion=criterion,
                    duration=duration,
                    dt=dt,
                    progress=solver_progress_gate.consume(),
                    batch_options=batch_options,
                    execution_policy=execution_policy,
                )
            observation_rows.append(observations)
            progress_display.update(
                label="Pool sweep",
                current_index=index,
                values=value_tuple,
                completed_rows=observation_rows,
                progress_summary=_activation_progress_summary,
            )
    finally:
        progress_display.close()

    if not observation_rows:
        return PoolSweepResult(
            values=value_tuple,
            observations=np.zeros((0, len(base_pool)), dtype=bool),
        )
    width = observation_rows[0].shape[0]
    if any(row.shape[0] != width for row in observation_rows):
        raise ValueError("pool/update must keep the same number of rows each time.")
    return PoolSweepResult(
        values=value_tuple,
        observations=np.stack(observation_rows, axis=0),
    )


def _activation_pool_sweep_batched_amplitudes(
    pool: tuple[SimulationCandidate, ...],
    *,
    update: PoolUpdate,
    values: tuple[Any, ...],
    duration: Any,
    dt: Any,
    criterion: ActivationCriterion,
    progress: bool | str = False,
    batch_options: BatchOptions | None = None,
    execution_policy: ExecutionPolicy | None = None,
    solver_progress: bool | str = False,
    amplitude_batch_size: int | None = None,
) -> PoolSweepResult:
    """Evaluate all amplitudes in one native expanded observer-only pool."""

    _validate_batched_amplitude_pool(pool)
    chunk_size = _normalize_amplitude_batch_size(amplitude_batch_size, len(values))
    progress_display = _SweepProgress(progress)
    solver_progress_gate = _OneShotProgress(solver_progress)
    observation_rows: list[np.ndarray] = []
    try:
        progress_display.begin(
            label="Pool sweep",
            current_index=0,
            values=values,
            completed_rows=observation_rows,
            progress_summary=_activation_progress_summary,
        )
        for chunk_start in range(0, len(values), chunk_size):
            chunk_values = values[chunk_start : chunk_start + chunk_size]
            started_s = time.perf_counter()
            with benchmark_span(
                "protocol.sweep.batched_values",
                value_count=len(chunk_values),
                pool_size=len(pool),
                expanded_pool_size=len(chunk_values) * len(pool),
                amplitude_batch_size=chunk_size,
                chunk_start=chunk_start,
            ):
                expanded_pool = _build_native_amplitude_pool(pool, update, chunk_values)
                flat_observations = _evaluate_activation_observer_pool(
                    expanded_pool,
                    criterion=criterion,
                    duration=duration,
                    dt=dt,
                    progress=solver_progress_gate.consume(),
                    batch_options=batch_options,
                    execution_policy=execution_policy,
                )
            elapsed_s = time.perf_counter() - started_s
            progress_display.note_batched_solver(
                elapsed_s=elapsed_s,
                value_count=len(chunk_values),
            )
            observations = np.asarray(flat_observations, dtype=bool).reshape(
                (len(chunk_values), len(pool))
            )
            for chunk_offset, row in enumerate(observations):
                observation_rows.append(row)
                progress_display.update(
                    label="Pool sweep",
                    current_index=chunk_start + chunk_offset,
                    values=values,
                    completed_rows=observation_rows,
                    progress_summary=_activation_progress_summary,
                    elapsed_s=elapsed_s / max(len(chunk_values), 1),
                )
    finally:
        progress_display.close()

    return PoolSweepResult(
        values=values,
        observations=np.asarray(observation_rows, dtype=bool),
    )


def _build_native_amplitude_pool(
    pool: tuple[SimulationCandidate, ...],
    update: PoolUpdate,
    values: tuple[Any, ...],
) -> tuple[SimulationCandidate, ...]:
    rows: list[SimulationCandidate] = []
    with benchmark_span(
        "protocol.sweep.build_amplitude_pool",
        value_count=len(values),
        pool_size=len(pool),
        expanded_pool_size=len(values) * len(pool),
    ):
        for value in values:
            for row in pool:
                native_row = _clone_native_pool_row(row)
                rows.append(_apply_pool_update(native_row, update, value))
    return tuple(rows)


def _normalize_amplitude_batch_size(
    amplitude_batch_size: int | None,
    value_count: int,
) -> int:
    if amplitude_batch_size is None:
        return max(value_count, 1)
    chunk_size = int(amplitude_batch_size)
    if chunk_size < 1:
        raise ValueError("amplitude_batch_size must be a positive integer or None.")
    return chunk_size


def _clone_native_pool_row(row: SimulationCandidate) -> SimulationCandidate:
    if not isinstance(row, AxonInstance):
        raise TypeError(
            "batch_amplitudes=True currently requires AxonInstance pool rows."
        )
    clone = AxonInstance(row.axon)
    clone.intracellular_contexts = list(row.intracellular_contexts)
    clone.extracellular_stimulation = row.extracellular_stimulation
    clone.Veinit = row.Veinit
    clone._use_extracellular_override = row._use_extracellular_override
    clone._xraxial_override = _copy_optional_array(row._xraxial_override)
    clone._xg_override = _copy_optional_array(row._xg_override)
    clone._xc_override = _copy_optional_array(row._xc_override)
    return clone


def _validate_batched_amplitude_pool(pool: tuple[SimulationCandidate, ...]) -> None:
    for row in pool:
        if not isinstance(row, AxonInstance):
            raise TypeError(
                "batch_amplitudes=True currently requires AxonInstance pool rows."
            )


def _copy_optional_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.array(value, copy=True)


__all__ = [
    "recruitment_sweep",
]
