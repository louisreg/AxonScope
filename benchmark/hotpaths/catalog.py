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
    "double_cable_extracellular": HotpathWorkload(
        name="double_cable_extracellular",
        description=(
            "Homogeneous MRG double-cable pool driven by analytical point-source "
            "extracellular contexts; useful to expose the priority myelinated "
            "extracellular hotpath."
        ),
    ),
    "footprint_reuse_sweep": HotpathWorkload(
        name="footprint_reuse_sweep",
        description=(
            "Repeated point-source pool runs with fixed geometry and changing "
            "stimulus amplitude; useful to measure missing footprint/stimulus-only reuse."
        ),
    ),
    "observer_only": HotpathWorkload(
        name="observer_only",
        description=(
            "Homogeneous HH pool with solver-side PeakVoltage/Activation observers "
            "and no retained Vm trace; useful to verify Phase 7.5 memory behavior."
        ),
    ),
    "realistic_mixed_population": HotpathWorkload(
        name="realistic_mixed_population",
        description=(
            "Mixed HH/Rattay-Aberham population with varied diameters, compartment "
            "counts, intracellular clamps, and some extracellular rows; useful to "
            "expose heterogeneous dispatch and preparation stalls."
        ),
    ),
    "hotpath_matrix": HotpathWorkload(
        name="hotpath_matrix",
        description=(
            "Compact matrix covering homogeneous center/probes recording, "
            "observer-only retention, point-source extracellular input, and a "
            "mixed-population path for Phase 7.6 coverage."
        ),
    ),
}


HOTPATH_PRESETS: dict[str, tuple[int, ...]] = {
    "smoke": (5,),
    "scale": (5, 50, 500),
}


__all__ = ["HOTPATH_PRESETS", "HOTPATH_WORKLOADS", "HotpathWorkload"]
