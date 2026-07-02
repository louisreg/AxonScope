"""Recruitment sweep protocol."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from axonscope.analysis import ActivationCriterion
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
    progress: bool | str = False,
    solver_progress: bool | str = False,
) -> RecruitmentCurve:
    """Evaluate pool recruitment over sampled current values.

    ``pool`` contains the simulations to evaluate. ``update(simulation,
    amplitude)`` is called before each run to change the swept parameter,
    usually an electrode current. The update function may mutate the row and
    return ``None``, or return a replacement simulation. ``batch_options``
    forwards solver-side execution knobs such as observer time chunking.
    Amplitudes are evaluated sequentially so memory scales with ``n_rows``, not
    ``n_rows * n_amplitudes``.
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
            solver_progress=solver_progress,
        )
        return RecruitmentCurve(
            amplitudes_uA=np.asarray(amplitudes_uA, dtype=float),
            activated=np.asarray(sweep.observations, dtype=bool),
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
    solver_progress: bool | str = False,
) -> PoolSweepResult:
    """Sweep activation with solver-side observers instead of stored Vm traces."""

    base_pool = tuple(pool)
    value_tuple = _normalize_sweep_values(values)
    if len(base_pool) == 0:
        return PoolSweepResult(
            values=value_tuple,
            observations=np.zeros((len(value_tuple), 0), dtype=bool),
        )

    progress_display = _SweepProgress(progress)
    solver_progress_gate = _OneShotProgress(solver_progress)
    observation_rows: list[np.ndarray] = []
    try:
        for index, value in enumerate(value_tuple):
            updated_pool = tuple(
                _apply_pool_update(row, update, value) for row in base_pool
            )
            progress_display.begin(
                label="Pool sweep",
                current_index=index,
                values=value_tuple,
                completed_rows=observation_rows,
                progress_summary=_activation_progress_summary,
            )
            observations = _evaluate_activation_observer_pool(
                updated_pool,
                criterion=criterion,
                duration=duration,
                dt=dt,
                progress=solver_progress_gate.consume(),
                batch_options=batch_options,
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
