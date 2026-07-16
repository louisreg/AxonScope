"""Internal prepared row data for one dispatch group."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.preparation.axon_rows import MaterializedAxonRows
from axonscope.preparation.membrane_rows import MembraneRowPlan
from axonscope.preparation.runtime_batches import (
    extracellular_stimulation_rows,
)
from axonscope.runtime.solver_axon import SolverAxon
from axonscope.stimulation import ExtracellularStimulation


@dataclass(frozen=True)
class PreparedCohort:
    """Prepared host-side inputs shared by batch execution paths."""

    group_id: int
    mode: str
    size: int
    nx: int
    geometry_shared: bool
    has_padding: bool
    representative: AxonInstance
    axons: tuple[AxonInstance, ...]
    solver_axons: tuple[SolverAxon, ...]
    materialized_axons: MaterializedAxonRows
    membrane_rows: MembraneRowPlan
    stimulations: tuple[tuple[ExtracellularStimulation, ...], ...]
    axon_y_um: np.ndarray
    axon_z_um: np.ndarray
    spatial_cache_token: object = field(
        default_factory=object,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_dispatch_group(cls, group: Any) -> "PreparedCohort":
        """Prepare the row data needed by solver input builders."""

        items = tuple(group.items)
        axons = tuple(item.simulation for item in items)
        solver_axons = tuple(item.solver_axon for item in items)
        representative = _representative_simulation(items, int(group.nx))
        stimulations = extracellular_stimulation_rows(axons)
        materialized_axons = MaterializedAxonRows.from_solver_axons(
            solver_axons,
            target_nx=int(group.nx),
        )
        membrane_rows = MembraneRowPlan.from_dispatch_items(items)
        axon_y_um = np.zeros((len(axons),), dtype=float)
        axon_z_um = np.zeros((len(axons),), dtype=float)
        return cls(
            group_id=int(group.group_id),
            mode=str(group.mode),
            size=len(items),
            nx=int(group.nx),
            geometry_shared=bool(group.geometry_shared),
            has_padding=bool(group.has_padding),
            representative=representative,
            axons=axons,
            solver_axons=solver_axons,
            materialized_axons=materialized_axons,
            membrane_rows=membrane_rows,
            stimulations=stimulations,
            axon_y_um=_readonly_array(axon_y_um),
            axon_z_um=_readonly_array(axon_z_um),
        )

    @property
    def extracellular_stimulation_count(self) -> int:
        """Number of enabled extracellular stimulations in the cohort."""

        return sum(len(row) for row in self.stimulations)

    @property
    def x_positions_m(self) -> np.ndarray:
        """Population-major padded intrinsic positions in meters."""

        return self.materialized_axons.x_positions_m


def _representative_simulation(items: tuple[Any, ...], nx: int) -> AxonInstance:
    for item in items:
        if int(item.solver_axon.n_compartments) == nx:
            return item.simulation
    return items[0].simulation


def _readonly_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    arr.setflags(write=False)
    return arr


__all__ = ["PreparedCohort"]
