"""Internal prepared row data for one dispatch group."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from axonfleet.axon_instance import AxonInstance
from axonfleet.preparation.axon_rows import MaterializedAxonRows
from axonfleet.preparation.membrane_rows import MembraneRowPlan
from axonfleet.preparation.stimulation_rows import (
    extracellular_stimulation_rows,
)
from axonfleet.runtime.solver_axon import SolverAxon
from axonfleet.stimulation import ExtracellularStimulation


@dataclass(frozen=True)
class PreparedCohort:
    """Prepared host-side inputs shared by batch execution paths."""

    group_id: int
    geometry_shared: bool
    axons: tuple[AxonInstance, ...]
    solver_axons: tuple[SolverAxon, ...]
    materialized_axons: MaterializedAxonRows
    membrane_rows: MembraneRowPlan
    stimulations: tuple[tuple[ExtracellularStimulation, ...], ...]
    spatial_cache_token: object = field(
        default_factory=object,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        size = len(self.axons)
        if size < 1:
            raise ValueError("PreparedCohort requires at least one axon row.")
        if len(self.solver_axons) != size or len(self.stimulations) != size:
            raise ValueError(
                "PreparedCohort axons, solver_axons, and stimulations must align."
            )
        if self.materialized_axons.size != size or self.membrane_rows.size != size:
            raise ValueError(
                "PreparedCohort materialized and membrane row plans must match its rows."
            )
        formulations = {axon.formulation for axon in self.solver_axons}
        if len(formulations) != 1:
            raise ValueError("PreparedCohort requires one cable formulation.")

    @classmethod
    def from_dispatch_group(cls, group: Any) -> "PreparedCohort":
        """Prepare the row data needed by solver input builders."""

        items = tuple(group.items)
        axons = tuple(item.simulation for item in items)
        solver_axons = tuple(item.solver_axon for item in items)
        stimulations = extracellular_stimulation_rows(axons)
        materialized_axons = MaterializedAxonRows.from_solver_axons(
            solver_axons,
            target_nx=int(group.nx),
        )
        membrane_rows = MembraneRowPlan.from_dispatch_items(items)
        return cls(
            group_id=int(group.group_id),
            geometry_shared=bool(group.geometry_shared),
            axons=axons,
            solver_axons=solver_axons,
            materialized_axons=materialized_axons,
            membrane_rows=membrane_rows,
            stimulations=stimulations,
        )

    @property
    def size(self) -> int:
        """Number of aligned simulation rows in this cohort."""

        return len(self.axons)

    @property
    def nx(self) -> int:
        """Padded spatial width of the numerical row table."""

        return self.materialized_axons.nx

    @property
    def mode(self) -> str:
        """Cable mode shared by all numerical rows."""

        return "double" if self.solver_axons[0].is_double_cable else "single"

    @property
    def has_padding(self) -> bool:
        """Whether any row is shorter than the numerical row width."""

        return any(axon.n_compartments != self.nx for axon in self.solver_axons)

    @property
    def representative(self) -> AxonInstance:
        """Simulation row whose spatial width matches the prepared width."""

        for axon, solver_axon in zip(self.axons, self.solver_axons, strict=True):
            if solver_axon.n_compartments == self.nx:
                return axon
        return self.axons[0]

    @property
    def extracellular_stimulation_count(self) -> int:
        """Number of enabled extracellular stimulations in the cohort."""

        return sum(len(row) for row in self.stimulations)

    @property
    def x_positions_m(self) -> np.ndarray:
        """Population-major padded intrinsic positions in meters."""

        return self.materialized_axons.x_positions_m
