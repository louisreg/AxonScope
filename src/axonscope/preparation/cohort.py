"""Internal prepared row data for one dispatch group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.dispatcher.runtime_batches import (
    axon_transverse_positions_um,
    extracellular_context_rows,
)
from axonscope.solvers.axon_runtime import SolverAxon
from axonscope.stimulation import ExtracellularContext


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
    contexts: tuple[tuple[ExtracellularContext, ...], ...]
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
        contexts = extracellular_context_rows(axons)
        x_positions = _x_positions_from_solver_axons_m(
            axons,
            solver_axons,
            target_nx=int(group.nx),
        )
        axon_y_um, axon_z_um = axon_transverse_positions_um(axons)
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
            contexts=contexts,
            x_positions_m=x_positions,
            axon_y_um=axon_y_um,
            axon_z_um=axon_z_um,
        )

    @property
    def context_count(self) -> int:
        """Number of enabled extracellular contexts in the cohort."""

        return sum(len(row) for row in self.contexts)


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
    row_cache: dict[tuple[int, float, int], np.ndarray] = {}
    for axon, solver_axon in zip(axons, solver_axons, strict=True):
        x_offset_um = float(getattr(axon, "x_offset_um", 0.0))
        cache_key = (id(solver_axon), x_offset_um, int(target_nx))
        row = row_cache.get(cache_key)
        if row is None:
            row = np.asarray(solver_axon.x_um, dtype=float) * 1e-6
            row = row + x_offset_um * 1e-6
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


__all__ = ["PreparedCohort"]
