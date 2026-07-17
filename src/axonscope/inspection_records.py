"""Structured records produced by AxonScope simulation inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TextIO

from axonscope.runtime import ExecutionPolicy
from axonscope.runtime.execution import CableSolverRoute


@dataclass(frozen=True)
class PlanningInspection:
    """High-level simulation planning summary."""

    axon_count: int
    duration_ms: float
    dt_ms: float
    step_count: int
    execution_policy: ExecutionPolicy | None


@dataclass(frozen=True)
class DispatchGroupInspection:
    """One dispatch/batch group selected by planning."""

    group_id: int
    pool_indices: tuple[int, ...]
    mode: str
    size: int
    nx: int
    batch_kind: str
    geometry_shared: bool
    has_padding: bool
    will_batch: bool


@dataclass(frozen=True)
class PaddingInspection:
    """Spatial padding expected for one dispatch group."""

    group_id: int
    row_nx: tuple[int, ...]
    padded_nx: int
    padded_compartments: int
    padded_fraction: float


@dataclass(frozen=True)
class PreparationInspection:
    """Host-side prepared cohort summary for one dispatch group."""

    group_id: int
    mode: str
    size: int
    nx: int
    extracellular_stimulation_count: int
    x_positions_shape: tuple[int, ...]
    y_shape: tuple[int, ...]
    z_shape: tuple[int, ...]
    representative_index: int | None


@dataclass(frozen=True)
class MembraneSourceInspection:
    """Membrane source/compiler provenance summary for one dispatch group."""

    group_id: int
    unique_membrane_count: int
    kinds: tuple[str, ...]
    source_count: int
    cache_statuses: tuple[str, ...]
    cache_reasons: tuple[str, ...]
    cache_keys: tuple[str, ...]
    source_hashes: tuple[str, ...]
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class LoweringInspection:
    """Predicted host/input lowering for one dispatch group."""

    group_id: int
    route: str
    intracellular_format: str
    extracellular_format: str
    observer_format: str
    recording_mode: str
    kernel_recording_mode: str
    retained_vm_width: int
    dense_iinj_shape: tuple[int, ...] | None
    dense_vstim_shape: tuple[int, ...] | None
    materializes_dense_vstim: bool


@dataclass(frozen=True)
class ProbeInspection:
    """Predicted threshold-observer probe lowering for one dispatch group."""

    group_id: int
    observer_names: tuple[str, ...]
    thresholds_mV: tuple[float, ...]
    row_aware: bool
    probe_indices_by_row: tuple[tuple[tuple[int, ...], ...], ...]
    row_probe_counts: tuple[tuple[int, ...], ...]
    max_probe_count: int
    retained_shape: tuple[int, ...] | None
    retained_bytes: int


@dataclass(frozen=True)
class MemoryInspection:
    """Estimated array pressure for one dispatch group."""

    group_id: int
    dtype: str
    state_bytes: int
    prepared_position_bytes: int
    dense_iinj_bytes: int
    dense_vstim_bytes: int
    retained_vm_bytes: int
    observer_bytes: int
    total_estimated_bytes: int
    retained_public_bytes: int

    @property
    def total_estimated_mib(self) -> float:
        return self.total_estimated_bytes / (1024**2)

    @property
    def retained_public_mib(self) -> float:
        return self.retained_public_bytes / (1024**2)


@dataclass(frozen=True)
class KernelInspection:
    """Predicted backend kernel route for one dispatch group."""

    group_id: int
    route: str
    kernel: str
    cable_mode: str
    solver: CableSolverRoute
    time_chunk_steps: int | None


@dataclass(frozen=True)
class ResultAssemblyInspection:
    """Predicted solver-output to public-result assembly for one group."""

    group_id: int
    record_kind: str
    vm_output: str
    observation_output: str
    public_result: str


@dataclass(frozen=True)
class AssemblyDetailInspection:
    """Shape-level public result assembly details for one group."""

    group_id: int
    row_count: int
    vm_shape: tuple[int, ...] | None
    observation_shape: tuple[int, ...] | None
    observations_are_batched: bool
    public_rows: int


@dataclass(frozen=True)
class SimulationInspection:
    """Structured inspection report for the solver pipeline."""

    planning: PlanningInspection
    dispatch_groups: tuple[DispatchGroupInspection, ...]
    padding: tuple[PaddingInspection, ...]
    preparations: tuple[PreparationInspection, ...]
    membrane_sources: tuple[MembraneSourceInspection, ...]
    lowerings: tuple[LoweringInspection, ...]
    probes: tuple[ProbeInspection, ...]
    memory: tuple[MemoryInspection, ...]
    kernels: tuple[KernelInspection, ...]
    result_assembly: tuple[ResultAssemblyInspection, ...]
    assembly_details: tuple[AssemblyDetailInspection, ...]

    def format(self) -> str:
        """Return a compact multiline text report."""

        from axonscope.inspection_views import format_simulation_inspection

        return format_simulation_inspection(self)

    def print(self, file: TextIO | None = None, *, rich: bool | None = None) -> None:
        """Print the inspection report."""

        from axonscope.inspection_views import print_simulation_inspection

        print_simulation_inspection(self, file=file, rich=rich)

    def plot(self, ax: Any | None = None) -> Any:
        """Plot dispatch groups, spatial widths, retained Vm, and observations."""

        from axonscope.inspection_views import plot_simulation_inspection

        return plot_simulation_inspection(self, ax=ax)

    def plot_details(self, axes: Sequence[Any] | None = None) -> tuple[Any, ...]:
        """Plot padding, memory, VmRaster probes, and result assembly."""

        from axonscope.inspection_views import plot_simulation_inspection_details

        return plot_simulation_inspection_details(self, axes=axes)


__all__ = [
    "AssemblyDetailInspection",
    "DispatchGroupInspection",
    "KernelInspection",
    "LoweringInspection",
    "MemoryInspection",
    "MembraneSourceInspection",
    "PaddingInspection",
    "PlanningInspection",
    "PreparationInspection",
    "ProbeInspection",
    "ResultAssemblyInspection",
    "SimulationInspection",
]
