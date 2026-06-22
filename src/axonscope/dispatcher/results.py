"""Internal dispatcher result records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon


@dataclass(frozen=True)
class DispatchResult:
    """Raw execution result for one axon before public result assembly."""

    index: int
    axon: Axon
    simulation: AxonInstance
    Vm: Any | None
    t: Any
    group_id: int
    method: str
    record_indices: tuple[int, ...] | None = None
    observations: dict[str, Any] | None = None
    observations_are_batched: bool = False
    group_size: int = 1
    batch_kind: str = "scalar"
    geometry_shared: bool = True
    has_padding: bool = False


@dataclass(frozen=True)
class DispatchCohortResult:
    """Raw execution result for one already-batched axon cohort.

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
    observations: dict[str, Any] | None = None
    group_size: int = 1
    batch_kind: str = "scalar"
    geometry_shared: bool = True
    has_padding: bool = False


DispatchRecord: TypeAlias = DispatchResult | DispatchCohortResult


__all__ = ["DispatchCohortResult", "DispatchRecord", "DispatchResult"]
