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
            "stimulations; useful to expose generic Vstim preprocessing cost."
        ),
    ),
    "double_cable_extracellular": HotpathWorkload(
        name="double_cable_extracellular",
        description=(
            "Homogeneous MRG double-cable pool driven by analytical point-source "
            "extracellular stimulations; useful to expose the priority myelinated "
            "extracellular hotpath."
        ),
    ),
    "double_cable_observer": HotpathWorkload(
        name="double_cable_observer",
        description=(
            "Homogeneous MRG double-cable extracellular pool with compact "
            "VmRaster threshold probes and no retained Vm trace."
        ),
    ),
    "footprint_reuse_sweep": HotpathWorkload(
        name="footprint_reuse_sweep",
        description=(
            "Repeated point-source pool runs with fixed geometry and changing "
            "stimulus amplitude; useful to measure missing footprint/stimulus-only reuse."
        ),
    ),
    "solver_only_precomputed": HotpathWorkload(
        name="solver_only_precomputed",
        description=(
            "Direct backend workload with runtime and inputs prepared before timing; "
            "useful to separate kernel throughput from dispatch and input materialization."
        ),
    ),
    "typed_footprint_drive_matrix": HotpathWorkload(
        name="typed_footprint_drive_matrix",
        description=(
            "Direct backend workload comparing typed stimulation lowering against "
            "typed ExtracellularFootprint/ExtracellularDrive lowering."
        ),
    ),
    "observer_only": HotpathWorkload(
        name="observer_only",
        description=(
            "Homogeneous HH pool with compact VmRaster threshold probes and no "
            "retained Vm trace; useful to verify Phase 7.5 memory behavior."
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
    "path_comparison_matrix": HotpathWorkload(
        name="path_comparison_matrix",
        description=(
            "Controlled matrix for intra vs extra and single-cable vs double-cable "
            "comparisons across center/probes/full/observer retention policies."
        ),
    ),
}


HOTPATH_PRESETS: dict[str, tuple[int, ...]] = {
    "smoke": (5,),
    "scale": (5, 50, 500),
}


__all__ = ["HOTPATH_PRESETS", "HOTPATH_WORKLOADS", "HotpathWorkload"]
