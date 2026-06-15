"""Internal dispatcher result records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon


@dataclass(frozen=True)
class DispatchResult:
    """Raw execution result for one axon before conversion to ``SimResult``."""

    index: int
    axon: Axon
    simulation: AxonInstance
    Vm: Any | None
    t: Any
    group_id: int
    method: str
    record_indices: tuple[int, ...] | None = None
    observations: dict[str, Any] | None = None
    group_size: int = 1
    batch_kind: str = "scalar"
    geometry_shared: bool = True
    has_padding: bool = False


__all__ = ["DispatchResult"]
