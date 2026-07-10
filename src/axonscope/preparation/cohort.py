"""Internal prepared row data for one dispatch group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.preparation.runtime_batches import (
    extracellular_stimulation_rows,
)
from axonscope.solvers.axon_runtime import SolverAxon
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
    stimulations: tuple[tuple[ExtracellularStimulation, ...], ...]
    x_positions_m: np.ndarray
    axon_y_um: np.ndarray
    axon_z_um: np.ndarray

    @classmethod
    def from_dispatch_group(cls, group: Any) -> "PreparedCohort":
        """Prepare the row data needed by solver input builders."""

        items = tuple(group.items)
        axons = tuple(item.simulation for item in items)
        solver_axons = tuple(item.solver_axon for item in items)
        representative = _representative_simulation(items, int(group.nx))
        stimulations = extracellular_stimulation_rows(axons)
        x_positions = _x_positions_from_solver_axons_m(
            axons,
            solver_axons,
            target_nx=int(group.nx),
        )
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
            stimulations=stimulations,
            x_positions_m=_readonly_array(x_positions),
            axon_y_um=_readonly_array(axon_y_um),
            axon_z_um=_readonly_array(axon_z_um),
        )

    @property
    def extracellular_stimulation_count(self) -> int:
        """Number of enabled extracellular stimulations in the cohort."""

        return sum(len(row) for row in self.stimulations)


def _representative_simulation(items: tuple[Any, ...], nx: int) -> AxonInstance:
    for item in items:
        if int(item.solver_axon.n_compartments) == nx:
            return item.simulation
    return items[0].simulation


def _x_positions_from_solver_axons_m(
    axons: tuple[AxonInstance, ...],
    solver_axons: tuple[SolverAxon, ...],
    *,
    target_nx: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    row_cache: dict[tuple[int, int], np.ndarray] = {}
    for _axon, solver_axon in zip(axons, solver_axons, strict=True):
        cache_key = (id(solver_axon), int(target_nx))
        row = row_cache.get(cache_key)
        if row is None:
            row = np.asarray(solver_axon.x_um, dtype=float) * 1e-6
            row = _pad_position_row(row, target_nx=int(target_nx))
            row_cache[cache_key] = row
        rows.append(row)
    return np.stack(rows, axis=0)


def _pad_position_row(values: np.ndarray, *, target_nx: int) -> np.ndarray:
    pad_count = int(target_nx) - int(values.shape[-1])
    if pad_count < 0:
        raise ValueError(
            f"target_nx must be >= array width, got target_nx={target_nx}, "
            f"width={values.shape[-1]}."
        )
    if pad_count == 0:
        return values
    if values.shape[-1] == 0:
        raise ValueError("cannot pad an empty spatial row.")
    return np.pad(values, (0, pad_count), mode="edge")


def _readonly_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    arr.setflags(write=False)
    return arr


__all__ = ["PreparedCohort"]
