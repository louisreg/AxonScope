"""Generic pool sweep protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from axonfleet.plans import SweepPlan
from axonfleet.protocols.types import (
    PoolObserver,
    PoolUpdate,
    ProgressSummary,
    SimulationCandidate,
)
from axonfleet.protocols.values import _normalize_sweep_values
from axonfleet.recording import Recording
from axonfleet.runtime import ExecutionPolicy
from axonfleet.simulation import AxonSimulation
from axonfleet.solvers import BatchOptions


@dataclass(frozen=True)
class _RowResultDecoder:
    observe: PoolObserver

    def __call__(self, result: Any) -> np.ndarray:
        return np.asarray([self.observe(row) for row in result])


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
    value_batch_size: int = 1,
) -> SweepPlan:
    """Describe a lazy parameter sweep over a stable simulation pool.

    Execute the returned plan with :meth:`axonfleet.Runner.run`.

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
    if not base_pool:
        raise ValueError("pool_sweep requires at least one source row.")
    value_tuple = _normalize_sweep_values(values)
    source = AxonSimulation(
        axons=base_pool,
        duration=duration,
        dt=dt,
        recording=recording or Recording.voltage(),
        batch_options=batch_options,
        execution_policy=execution_policy,
        progress=solver_progress,
    ).plan()
    return SweepPlan(
        source=source,
        values=value_tuple,
        update=update,
        decode=_RowResultDecoder(observe),
        value_batch_size=value_batch_size,
        progress=progress,
        progress_summary=progress_summary,
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
