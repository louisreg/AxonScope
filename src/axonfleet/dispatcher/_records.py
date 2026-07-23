"""Internal dispatcher records passed between execution and result assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

from axonfleet.axon_instance import AxonInstance
from axonfleet.axons.axon import Axon


@dataclass(frozen=True)
class DispatchRowRecord:
    """Raw execution record for one axon before public result assembly."""

    index: int
    axon: Axon
    simulation: AxonInstance
    Vm: Any | None
    t: Any
    group_id: int
    method: str
    record_indices: tuple[int, ...] | None = None
    recordings: dict[str, Any] | None = None
    observations: dict[str, Any] | None = None
    final_state: Any | None = None
    observations_are_batched: bool = False
    group_size: int = 1
    batch_kind: str = "unknown"
    geometry_shared: bool = True
    has_padding: bool = False


@dataclass(frozen=True)
class DispatchCohortRecord:
    """Raw execution record for one already-batched axon cohort.

    This is used when the backend can keep a compact population-level payload,
    typically observer-only simulations where no per-axon Vm trace needs to be
    materialized between the solver and the public result layer.
    """

    indices: tuple[int, ...]
    axons: tuple[Axon, ...]
    simulations: tuple[AxonInstance, ...]
    Vm: Any | None
    t: Any
    group_id: int
    method: str
    record_indices: tuple[tuple[int, ...] | None, ...]
    recordings: dict[str, Any] | None = None
    observations: dict[str, Any] | None = None
    final_states: tuple[Any | None, ...] | None = None
    group_size: int = 1
    batch_kind: str = "unknown"
    geometry_shared: bool = True
    has_padding: bool = False


DispatchRecord: TypeAlias = DispatchRowRecord | DispatchCohortRecord


__all__ = ["DispatchCohortRecord", "DispatchRecord", "DispatchRowRecord"]
