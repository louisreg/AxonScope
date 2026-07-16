"""Generic pool sweep protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from axonscope.protocols.progress import _OneShotProgress, _SweepProgress
from axonscope.protocols.results import PoolSweepResult
from axonscope.protocols.types import (
    PoolObserver,
    PoolUpdate,
    NumericAxisInputBuilder,
    NumericAxisUpdate,
    ProgressSummary,
    SimulationCandidate,
)
from axonscope.protocols.values import _normalize_sweep_values
from axonscope.recording import Recording
from axonscope.runtime import ExecutionPolicy
from axonscope.runtime.benchmarking import benchmark_span
from axonscope.simulation import AxonSimulation
from axonscope.solvers import BatchOptions


@dataclass(frozen=True)
class _NumericAxisValueBatch:
    """One ordered, bounded slice of values in a compact sweep plan."""

    start_index: int
    values: tuple[Any, ...]


@dataclass(frozen=True)
class _NumericPoolSweepPlan:
    """Protocol-neutral numeric-axis plan over one stable source population."""

    source_pool: tuple[SimulationCandidate, ...]
    values: tuple[Any, ...]
    update: NumericAxisUpdate
    input_builder: NumericAxisInputBuilder
    batches: tuple[_NumericAxisValueBatch, ...]

    @property
    def source_pool_size(self) -> int:
        return len(self.source_pool)


def pool_sweep(
    pool: Sequence[SimulationCandidate],
    *,
    update: PoolUpdate,
    values: Sequence[Any],
    observe: PoolObserver,
    duration: Any,
    dt: Any,
    recording: Recording | None = None,
    batch_options: BatchOptions | None = None,
    execution_policy: ExecutionPolicy | None = None,
    progress: bool | str = False,
    progress_summary: ProgressSummary | None = None,
    solver_progress: bool | str = False,
) -> PoolSweepResult:
    """Sweep a parameter over a stable simulation pool.

    Parameters
    ----------
    pool:
        Stable sequence of simulations or axons.
    update:
        Called as ``update(row, value)`` before each run. It may mutate the row
        in place and return ``None``, or return a replacement candidate.
    values:
        Parameter values to test. Unit-bearing arrays are accepted and each row
        receives one scalar quantity from the array.
    observe:
        Called on each per-axon result view to produce one per-row observation.
    duration, dt:
        Simulation duration and timestep.
    recording:
        Recording policy used when pool entries must be simulated.
    batch_options:
        Optional solver-side batch execution knobs, forwarded to
        ``AxonSimulation``.
    execution_policy:
        Optional typed runtime/device/solver policy forwarded to each
        ``AxonSimulation`` call.
    progress:
        If true, display a Rich live progress table when Rich is available.
    progress_summary:
        Optional formatter for one completed observation row.
    solver_progress:
        Optional progress flag forwarded only to the first ``AxonSimulation``
        call, which is normally the cold solver run.
    """

    base_pool = tuple(pool)
    value_tuple = _normalize_sweep_values(values)
    if len(base_pool) == 0:
        return PoolSweepResult(
            values=value_tuple,
            observations=np.zeros((len(value_tuple), 0), dtype=object),
        )

    progress_display = _SweepProgress(progress)
    solver_progress_gate = _OneShotProgress(solver_progress)
    observation_rows: list[np.ndarray] = []
    reusable_simulation: AxonSimulation | None = None
    reusable_axis_builder: NumericAxisInputBuilder | None = None
    try:
        for index, value in enumerate(value_tuple):
            progress_display.begin(
                label="Pool sweep",
                current_index=index,
                values=value_tuple,
                completed_rows=observation_rows,
                progress_summary=progress_summary,
            )
            with benchmark_span(
                "protocol.sweep.value",
                index=index,
                value=str(value),
                pool_size=len(base_pool),
            ):
                if isinstance(update, NumericAxisUpdate):
                    if reusable_simulation is None:
                        reusable_axis_builder = update.prepare_numeric_axis(base_pool)
                        reusable_simulation = AxonSimulation(
                            axons=base_pool,
                            duration=duration,
                            dt=dt,
                            recording=recording or Recording.voltage(),
                            batch_options=batch_options,
                            execution_policy=execution_policy,
                            progress=solver_progress_gate.consume(),
                        )
                    if reusable_axis_builder is None:
                        raise RuntimeError("numeric-axis input builder was not prepared.")
                    axis_input = reusable_axis_builder.numeric_axis_input((value,))
                    results = tuple(reusable_simulation._run_numeric_axis(axis_input))
                    reusable_simulation.progress = False
                else:
                    results = _run_updated_pool(
                        base_pool,
                        update,
                        tuple(value for _ in base_pool),
                        duration=duration,
                        dt=dt,
                        recording=recording,
                        batch_options=batch_options,
                        execution_policy=execution_policy,
                        progress=solver_progress_gate.consume(),
                    )
                observations = np.asarray([observe(result) for result in results])
            observation_rows.append(observations)
            progress_display.update(
                label="Pool sweep",
                current_index=index,
                values=value_tuple,
                completed_rows=observation_rows,
                progress_summary=progress_summary,
            )
    finally:
        progress_display.close()

    if not observation_rows:
        return PoolSweepResult(
            values=value_tuple,
            observations=np.zeros((0, len(base_pool)), dtype=object),
        )
    width = observation_rows[0].shape[0]
    if any(row.shape[0] != width for row in observation_rows):
        raise ValueError("pool/update must keep the same number of rows each time.")
    return PoolSweepResult(
        values=value_tuple,
        observations=np.stack(observation_rows, axis=0),
    )


def _run_updated_pool(
    pool: tuple[SimulationCandidate, ...],
    update: PoolUpdate,
    values: tuple[Any, ...],
    *,
    duration: Any,
    dt: Any,
    recording: Recording | None,
    batch_options: BatchOptions | None,
    execution_policy: ExecutionPolicy | None,
    progress: bool | str,
) -> tuple[Any, ...]:
    if len(values) != len(pool):
        raise ValueError(
            f"values must contain one value per row; got {len(values)} for {len(pool)} rows."
        )
    updated_pool = tuple(
        _apply_pool_update(row, update, value)
        for row, value in zip(pool, values, strict=True)
    )
    pool_result = AxonSimulation(
        axons=updated_pool,
        duration=duration,
        dt=dt,
        recording=recording or Recording.voltage(),
        batch_options=batch_options,
        execution_policy=execution_policy,
        progress=progress,
    ).run()
    return tuple(pool_result)


def _apply_pool_update(
    row: SimulationCandidate,
    update: PoolUpdate,
    value: Any,
) -> SimulationCandidate:
    updated = update(row, value)
    return row if updated is None else updated


def _plan_numeric_pool_sweep(
    pool: tuple[SimulationCandidate, ...],
    *,
    update: PoolUpdate,
    values: tuple[Any, ...],
    value_batch_size: int | None,
) -> _NumericPoolSweepPlan:
    """Plan ordered value chunks without expanding value-by-population rows."""

    if not isinstance(update, NumericAxisUpdate):
        raise ValueError(
            "compact value batches require a typed NumericAxisUpdate; arbitrary "
            "row callbacks cannot prove a stable execution contract."
        )
    input_builder = update.prepare_numeric_axis(pool)
    chunk_size = _normalize_value_batch_size(value_batch_size, len(values))
    batches = tuple(
        _NumericAxisValueBatch(
            start_index=start,
            values=values[start : start + chunk_size],
        )
        for start in range(0, len(values), chunk_size)
    )
    return _NumericPoolSweepPlan(
        source_pool=pool,
        values=values,
        update=update,
        input_builder=input_builder,
        batches=batches,
    )


def _normalize_value_batch_size(
    value_batch_size: int | None,
    value_count: int,
) -> int:
    if value_batch_size is None:
        return max(value_count, 1)
    chunk_size = int(value_batch_size)
    if chunk_size < 1:
        raise ValueError("amplitude_batch_size must be a positive integer or None.")
    return chunk_size


__all__ = [
    "pool_sweep",
]
