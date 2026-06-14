"""Catalog for Phase 2.5 hotpath diagnostic workloads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HotpathWorkload:
    """One named diagnostic workload family."""

    name: str
    description: str


HOTPATH_WORKLOADS: dict[str, HotpathWorkload] = {
    "intracellular_only": HotpathWorkload(
        name="intracellular_only",
        description=(
            "Homogeneous HH pool with intracellular clamps only; useful to separate "
            "dispatch/runtime/kernel cost from extracellular field construction."
        ),
    ),
    "point_source_extracellular": HotpathWorkload(
        name="point_source_extracellular",
        description=(
            "Homogeneous HH pool driven by analytical point-source extracellular "
            "contexts; useful to expose generic Vstim preprocessing cost."
        ),
    ),
}


HOTPATH_PRESETS: dict[str, tuple[int, ...]] = {
    "smoke": (5,),
    "scale": (5, 50, 500),
}


__all__ = ["HOTPATH_PRESETS", "HOTPATH_WORKLOADS", "HotpathWorkload"]
