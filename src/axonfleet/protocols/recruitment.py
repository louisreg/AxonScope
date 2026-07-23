"""Recruitment sweep protocol."""

from __future__ import annotations

from functools import wraps
from typing import Any, Sequence

import numpy as np

from axonfleet.analysis import Activation
from axonfleet.protocols.observer_path import (
    _can_use_activation_observer,
    _evaluate_activation_observer_pool,
    _execute_activation_observer_sweep_plan,
)
from axonfleet.protocols.progress import (
    _OneShotProgress,
    _SweepProgress,
    _activation_progress_summary,
)
from axonfleet.protocols.results import PoolSweepResult, RecruitmentCurve
from axonfleet.protocols.sweep import (
    _apply_pool_update,
    _plan_numeric_pool_sweep,
    pool_sweep,
)
from axonfleet.protocols.types import NumericAxisUpdate, PoolUpdate, SimulationCandidate
from axonfleet.protocols.values import _normalize_sweep_values, _require_current_array_uA
from axonfleet.recording import Recording
from axonfleet.runtime import ExecutionPolicy
from axonfleet.runtime.benchmarking import benchmark_span
from axonfleet.solvers import BatchOptions
from axonfleet.utils import units


def _recruitment_stage(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with benchmark_span("protocol.recruitment_sweep"):
            return function(*args, **kwargs)

    return wrapped


@_recruitment_stage
def recruitment_sweep(
    pool: Sequence[SimulationCandidate],
    *,
    update: PoolUpdate,
    values: Any,
    duration: Any,
    dt: Any,
    criterion: Activation,
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
    Amplitudes are evaluated over one stable source population so memory scales
    with ``n_rows``, not ``n_rows * n_amplitudes``. For typed extracellular
    waveform updates, ``batch_amplitudes=True`` creates compact ordered value
    chunks without cloning axons or stimulation graphs. ``amplitude_batch_size``
    bounds those chunks; ``None`` means all values in one plan chunk.
    ``solver_progress`` is forwarded only to the first solver call so cold-start
    compilation remains visible without logging every sampled amplitude.
    """

    amplitudes_uA = _require_current_array_uA(values, name="values")
    if amplitudes_uA.ndim != 1:
        raise ValueError("values must be a 1D current array.")
    if _can_use_activation_observer(recording):
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
        observe=lambda result: criterion.detect(result).activated,
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
    criterion: Activation,
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
    if batch_amplitudes and not isinstance(update, NumericAxisUpdate):
        raise ValueError(
            "batch_amplitudes=True requires a typed NumericAxisUpdate so the "
            "runner can preserve one stable population and execution contract."
        )
    if amplitude_batch_size is not None:
        if not batch_amplitudes:
            raise ValueError("amplitude_batch_size requires batch_amplitudes=True.")

    if isinstance(update, NumericAxisUpdate):
        plan = _plan_numeric_pool_sweep(
            base_pool,
            update=update,
            values=value_tuple,
            value_batch_size=amplitude_batch_size if batch_amplitudes else 1,
        )
        return _execute_activation_observer_sweep_plan(
            plan,
            criterion=criterion,
            duration=duration,
            dt=dt,
            progress=progress,
            batch_options=batch_options,
            execution_policy=execution_policy,
            solver_progress=solver_progress,
        )

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


__all__ = [
    "recruitment_sweep",
]
