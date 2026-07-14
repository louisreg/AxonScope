"""Recruitment sweep protocol."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from axonscope.analysis import ActivationCriterion
from axonscope.axon_instance import AxonInstance
from axonscope.protocols.observer_path import (
    _build_activation_observer_simulation,
    _can_use_activation_observer,
    _evaluate_activation_observer_pool,
    _evaluate_activation_observer_simulation,
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


@dataclass(frozen=True)
class _NativeAmplitudeBatch:
    values: tuple[Any, ...]


@dataclass(frozen=True)
class _NativeAmplitudeBatchPlan:
    source_pool: tuple[SimulationCandidate, ...]
    update: PoolUpdate
    source_pool_size: int
    batches: tuple[_NativeAmplitudeBatch, ...]


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
    """Evaluate native amplitude batches with the current sequential executor."""

    _validate_batched_amplitude_pool(pool)
    plan = _plan_native_amplitude_batches(
        pool,
        update=update,
        values=values,
        amplitude_batch_size=amplitude_batch_size,
    )
    return _execute_activation_observer_batch_plan(
        plan,
        all_values=values,
        duration=duration,
        dt=dt,
        criterion=criterion,
        progress=progress,
        batch_options=batch_options,
        execution_policy=execution_policy,
        solver_progress=solver_progress,
    )


def _execute_activation_observer_batch_plan(
    plan: _NativeAmplitudeBatchPlan,
    *,
    all_values: tuple[Any, ...],
    duration: Any,
    dt: Any,
    criterion: ActivationCriterion,
    progress: bool | str = False,
    batch_options: BatchOptions | None = None,
    execution_policy: ExecutionPolicy | None = None,
    solver_progress: bool | str = False,
) -> PoolSweepResult:
    """Execute a planned set of native amplitude batches sequentially."""

    progress_display = _SweepProgress(progress)
    solver_progress_gate = _OneShotProgress(solver_progress)
    observation_rows: list[np.ndarray] = []
    work_pools: dict[int, tuple[SimulationCandidate, ...]] = {}
    work_simulations: dict[int, tuple[Any, Any]] = {}
    completed_value_count = 0
    try:
        progress_display.begin(
            label="Pool sweep",
            current_index=0,
            values=all_values,
            completed_rows=observation_rows,
            progress_summary=_activation_progress_summary,
        )
        for batch_index, batch in enumerate(plan.batches):
            value_count = len(batch.values)
            work_pool = work_pools.get(value_count)
            if work_pool is None:
                work_pool = _build_native_amplitude_pool(
                    plan.source_pool,
                    plan.update,
                    batch.values,
                )
            else:
                work_pool = _refresh_native_amplitude_pool(
                    work_pool,
                    source_pool=plan.source_pool,
                    update=plan.update,
                    values=batch.values,
                )
            work_pools[value_count] = work_pool
            reusable = work_simulations.get(value_count)
            if reusable is None:
                reusable = _build_activation_observer_simulation(
                    work_pool,
                    criterion=criterion,
                    duration=duration,
                    dt=dt,
                    progress=solver_progress_gate.consume(),
                    batch_options=batch_options,
                    execution_policy=execution_policy,
                )
                work_simulations[value_count] = reusable
            started_s = time.perf_counter()
            with benchmark_span(
                "protocol.sweep.batched_values",
                batch_index=batch_index,
                value_count=value_count,
                pool_size=plan.source_pool_size,
                expanded_pool_size=len(work_pool),
            ):
                flat_observations = _evaluate_activation_observer_simulation(
                    *reusable,
                )
            elapsed_s = time.perf_counter() - started_s
            progress_display.note_batched_solver(
                elapsed_s=elapsed_s,
                value_count=value_count,
            )
            observations = np.asarray(flat_observations, dtype=bool).reshape(
                (value_count, plan.source_pool_size)
            )
            for chunk_offset, row in enumerate(observations):
                observation_rows.append(row)
                progress_display.update(
                    label="Pool sweep",
                    current_index=completed_value_count + chunk_offset,
                    values=all_values,
                    completed_rows=observation_rows,
                    progress_summary=_activation_progress_summary,
                    elapsed_s=elapsed_s / max(value_count, 1),
                )
            completed_value_count += value_count
    finally:
        progress_display.close()

    return PoolSweepResult(
        values=all_values,
        observations=np.asarray(observation_rows, dtype=bool),
    )


def _plan_native_amplitude_batches(
    pool: tuple[SimulationCandidate, ...],
    *,
    update: PoolUpdate,
    values: tuple[Any, ...],
    amplitude_batch_size: int | None,
) -> _NativeAmplitudeBatchPlan:
    """Build native amplitude batches without deciding how to schedule them."""

    chunk_size = _normalize_amplitude_batch_size(amplitude_batch_size, len(values))
    batches: list[_NativeAmplitudeBatch] = []
    for chunk_start in range(0, len(values), chunk_size):
        chunk_values = values[chunk_start : chunk_start + chunk_size]
        batches.append(
            _NativeAmplitudeBatch(
                values=chunk_values,
            )
        )
    return _NativeAmplitudeBatchPlan(
        source_pool=pool,
        update=update,
        source_pool_size=len(pool),
        batches=tuple(batches),
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


def _refresh_native_amplitude_pool(
    work_pool: tuple[SimulationCandidate, ...],
    *,
    source_pool: tuple[SimulationCandidate, ...],
    update: PoolUpdate,
    values: tuple[Any, ...],
) -> tuple[SimulationCandidate, ...]:
    """Reset and update a stable native pool for another amplitude chunk."""

    expected_size = len(source_pool) * len(values)
    if len(work_pool) != expected_size:
        raise ValueError(
            "native amplitude work pool has the wrong size; "
            f"expected {expected_size}, got {len(work_pool)}."
        )

    refreshed: list[SimulationCandidate] = []
    with benchmark_span(
        "protocol.sweep.refresh_amplitude_pool",
        value_count=len(values),
        pool_size=len(source_pool),
        expanded_pool_size=expected_size,
    ):
        for value_index, value in enumerate(values):
            row_start = value_index * len(source_pool)
            for row_index, source_row in enumerate(source_pool):
                work_row = work_pool[row_start + row_index]
                reusable_row = _reset_native_pool_row(work_row, source_row)
                refreshed.append(_apply_pool_update(reusable_row, update, value))

    if all(
        refreshed_row is work_row
        for refreshed_row, work_row in zip(refreshed, work_pool, strict=True)
    ):
        return work_pool
    return tuple(refreshed)


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
    return _reset_native_pool_row(AxonInstance(row.axon), row)


def _reset_native_pool_row(
    work_row: SimulationCandidate,
    source_row: SimulationCandidate,
) -> AxonInstance:
    if not isinstance(source_row, AxonInstance):
        raise TypeError(
            "batch_amplitudes=True currently requires AxonInstance pool rows."
        )
    if (
        not isinstance(work_row, AxonInstance)
        or work_row.axon is not source_row.axon
    ):
        work_row = AxonInstance(source_row.axon)

    # Remove state left by the previous update, then recreate exactly the same
    # baseline that a fresh native clone would have received.
    work_row.__dict__.clear()
    work_row.axon = source_row.axon
    work_row.intracellular_contexts = list(source_row.intracellular_contexts)
    work_row.extracellular_stimulation = source_row.extracellular_stimulation
    work_row.Veinit = source_row.Veinit
    work_row._use_extracellular_override = source_row._use_extracellular_override
    work_row._xraxial_override = _copy_optional_array(source_row._xraxial_override)
    work_row._xg_override = _copy_optional_array(source_row._xg_override)
    work_row._xc_override = _copy_optional_array(source_row._xc_override)
    return work_row


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
