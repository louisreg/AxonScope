"""Recruitment sweep protocol."""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, Sequence

import numpy as np

from axonfleet.analysis import Activation
from axonfleet.protocols.observer_path import (
    _activation_observer_sweep_plan,
    _can_use_activation_observer,
)
from axonfleet.plans import SweepPlan
from axonfleet.protocols.progress import _activation_progress_summary
from axonfleet.protocols.results import PoolSweepResult, RecruitmentCurve
from axonfleet.protocols.sweep import (
    _normalize_value_batch_size,
    pool_sweep,
)
from axonfleet.protocols.types import NumericAxisUpdate, PoolUpdate, SimulationCandidate
from axonfleet.protocols.values import _normalize_sweep_values, _require_current_array_uA
from axonfleet.recording import Recording
from axonfleet.runtime import ExecutionPolicy
from axonfleet.solvers import BatchOptions
from axonfleet.utils import units


@dataclass(frozen=True)
class _ActivationRowDecoder:
    criterion: Activation

    def __call__(self, result: Any) -> bool:
        return bool(self.criterion.detect(result).activated)


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
) -> SweepPlan:
    """Describe a lazy recruitment sweep over sampled current values.

    Execute the returned plan with :meth:`axonfleet.Runner.run`.

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
    base_pool = tuple(pool)
    if not base_pool:
        raise ValueError("recruitment_sweep requires at least one source row.")
    value_tuple = _normalize_sweep_values(units.Q_(amplitudes_uA, "microampere"))
    observer_path = _can_use_activation_observer(recording)
    if batch_amplitudes and not isinstance(update, NumericAxisUpdate):
        raise ValueError(
            "batch_amplitudes=True requires a typed NumericAxisUpdate so the "
            "runner can preserve one stable population and execution contract."
        )
    if amplitude_batch_size is not None and not batch_amplitudes:
        raise ValueError("amplitude_batch_size requires batch_amplitudes=True.")
    if not observer_path and (batch_amplitudes or amplitude_batch_size is not None):
        raise ValueError(
            "amplitude batching requires an observer-only recruitment path; "
            "use recording=None or Recording.none()."
        )

    value_batch_size = (
        _normalize_value_batch_size(amplitude_batch_size, len(value_tuple))
        if batch_amplitudes
        else 1
    )
    if observer_path:
        plan = _activation_observer_sweep_plan(
            base_pool,
            update=update,
            values=value_tuple,
            value_batch_size=value_batch_size,
            criterion=criterion,
            duration=duration,
            dt=dt,
            progress=progress,
            batch_options=batch_options,
            execution_policy=execution_policy,
            solver_progress=solver_progress,
        )
    else:
        plan = pool_sweep(
            base_pool,
            update=update,
            values=value_tuple,
            observe=_ActivationRowDecoder(criterion),
            duration=duration,
            dt=dt,
            recording=recording,
            batch_options=batch_options,
            execution_policy=execution_policy,
            progress=progress,
            progress_summary=_activation_progress_summary,
            solver_progress=solver_progress,
        )
    return dataclass_replace(plan, result_factory=_RecruitmentResultFactory())


@dataclass(frozen=True)
class _RecruitmentResultFactory:
    def __call__(self, sweep: PoolSweepResult) -> RecruitmentCurve:
        amplitudes_uA = _require_current_array_uA(sweep.values, name="values")
        return RecruitmentCurve(
            amplitudes_uA=np.asarray(amplitudes_uA, dtype=float),
            activated=np.asarray(sweep.observations, dtype=bool),
            row_labels=sweep.row_labels,
        )


__all__ = [
    "recruitment_sweep",
]
