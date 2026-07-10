"""Shared dispatch-route eligibility helpers."""

from __future__ import annotations

from typing import Any

from axonscope.dispatcher.plan import DispatchGroup
from axonscope.solvers import BatchOptions


def can_use_batch_route(
    group: DispatchGroup,
    *,
    batch_options: BatchOptions,
    observers: tuple[Any, ...] | None,
    record_observables: bool = False,
) -> bool:
    """Return whether a dispatch group can use the current batch backend.

    A one-row group is still a valid batch of size one. The explicit fallback is
    dense observable recording, because the active batch kernels retain Vm and
    VmRaster-style observations, not full gates/currents/conductances payloads.
    """

    del batch_options, observers
    if group.mode not in {"single", "double"}:
        return False
    return not bool(record_observables)


__all__ = ["can_use_batch_route"]
