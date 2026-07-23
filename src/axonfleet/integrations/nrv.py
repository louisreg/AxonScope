"""NRV-to-AxonFleet handoff helpers.

This module intentionally does not import NRV. NRV still owns external nerve
geometry, fiber placement, FEM fields, and its result objects; these helpers
only turn already-built NRV objects into AxonFleet fiber metadata, sampled
footprints, stimulation objects, and compact comparison rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from axonfleet.axon_instance import AxonInstance
from axonfleet.axons import (
    Axon,
    HodgkinHuxley,
    MRG,
    RattayAberham,
)
from axonfleet.axons.templates.mrg_like_double_cable import (
    mrg_like_node_spacing,
    mrg_like_nodes_from_length,
)
from axonfleet.benchmarking import benchmark_span, record_benchmark_metadata
from axonfleet.identifiers import DriveId
from axonfleet.population import AxonPopulation
from axonfleet.stimulation import (
    ExtracellularDrive,
    ExtracellularFootprint,
    ExtracellularStimulation,
    Stimulus,
)
from axonfleet.utils.units import Q_, ureg


_FiberKind = Literal["hh", "rattay", "mrg"]


def _nrv_bridge_stage(name: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with benchmark_span(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


@dataclass(frozen=True)
class NRVFiberRow:
    """One NRV fiber after conversion to AxonFleet intrinsic metadata."""

    fascicle_id: str
    fiber_index: int
    kind: _FiberKind
    diameter_um: float
    y_um: float
    z_um: float
    node_shift: float = 0.0
    x_shift_um: float = 0.0


@dataclass(frozen=True)
class NRVAxonPopulation:
    """AxonFleet population built from an NRV fiber population."""

    population: AxonPopulation
    rows: tuple[NRVFiberRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.population, AxonPopulation):
            raise TypeError("population must be an AxonPopulation.")
        rows = tuple(self.rows)
        if len(self.population) != len(rows):
            raise ValueError("population and rows must have the same length.")
        object.__setattr__(self, "rows", rows)

    def __len__(self) -> int:
        return len(self.population)


@dataclass(frozen=True)
class NRVFootprints:
    """Footprints sampled from NRV geometry for an AxonFleet population."""

    population: AxonPopulation
    rows: tuple[NRVFiberRow, ...]
    footprints: tuple[tuple[ExtracellularFootprint, ...], ...]
    electrode_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.population, AxonPopulation):
            raise TypeError("population must be an AxonPopulation.")
        rows = tuple(self.rows)
        footprints = tuple(tuple(item) for item in self.footprints)
        electrode_ids = tuple(str(item) for item in self.electrode_ids)
        if len(self.population) != len(rows) or len(rows) != len(footprints):
            raise ValueError("population, rows, and footprints must have the same length.")
        if footprints:
            electrode_count = len(footprints[0])
            if any(len(item) != electrode_count for item in footprints):
                raise ValueError("all rows must have the same number of electrode footprints.")
            if len(electrode_ids) != electrode_count:
                raise ValueError("electrode_ids must match the electrode footprint count.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "footprints", footprints)
        object.__setattr__(self, "electrode_ids", electrode_ids)

    def __len__(self) -> int:
        return len(self.rows)

    @_nrv_bridge_stage("nrv_bridge.stimulated_population")
    def stimulated_population(
        self,
        *,
        electrode_index: int = 0,
        stimulus: Stimulus,
        drive_id_prefix: str = "nrv_electrode",
    ) -> AxonPopulation:
        """Attach one sampled NRV electrode footprint to each population row."""

        electrode_index = int(electrode_index)
        instances = []
        for row_index, (instance, footprint) in enumerate(
            zip(
                self.population.instances,
                (row[electrode_index] for row in self.footprints),
                strict=True,
            )
        ):
            simulation = AxonInstance(instance.axon)
            simulation.add_extracellular_stimulation(
                stimulation=_stimulation_from_footprint(
                    footprint,
                    stimulus=stimulus,
                    drive_id=DriveId(f"{drive_id_prefix}_{int(electrode_index)}"),
                    metadata={"nrv_row": row_index},
                )
            )
            instances.append(simulation)
        return AxonPopulation(instances, name=self.population.name)

def _fiber_kind_from_nrv(nrv_type: int, *, include_mrg: bool) -> _FiberKind:
    """Map NRV's fiber type code to the AxonFleet template used by examples."""

    if int(nrv_type) == 1:
        return "mrg" if include_mrg else "rattay"
    return "rattay"


def _nrv_node_shift_to_x_shift_um(
    node_shift: float,
    diameter_um: float,
    *,
    kind: _FiberKind,
) -> float:
    """Convert NRV's fractional MRG node shift to AxonFleet MRG phase."""

    if kind != "mrg":
        return 0.0
    node_spacing_um = mrg_like_node_spacing(Q_(float(diameter_um), "micrometer"))
    return float(node_shift) * node_spacing_um


def _fiber_rows_from_nrv(
    nrv_population: Any,
    *,
    include_unmyelinated: bool,
    include_mrg: bool = True,
) -> list[NRVFiberRow]:
    """Map NRV fascicle populations to AxonFleet fiber rows.

    The returned rows keep NRV cross-section coordinates as metadata for
    external footprint sampling. They do not assign real-world coordinates to
    `AxonInstance`; AxonFleet still sees intrinsic x positions only.
    """

    rows: list[NRVFiberRow] = []
    for fascicle_id, fascicle in _iter_nrv_fascicles(nrv_population):
        table = fascicle.axons.axon_pop
        for fiber_index, table_row in table.iterrows():
            if not _nrv_fiber_is_simulated(fascicle, int(fiber_index)):
                continue
            nrv_type = int(float(table_row.get("types", 0)))
            kind = _fiber_kind_from_nrv(nrv_type, include_mrg=include_mrg)
            if kind != "mrg" and not include_unmyelinated:
                continue
            diameter_um = float(table_row.get("diameters", 1.0))
            node_shift = float(table_row.get("node_shift", 0.0))
            rows.append(
                NRVFiberRow(
                    fascicle_id=str(fascicle_id),
                    fiber_index=int(fiber_index),
                    kind=kind,
                    diameter_um=diameter_um,
                    y_um=float(table_row.get("y", 0.0)),
                    z_um=float(table_row.get("z", 0.0)),
                    node_shift=node_shift,
                    x_shift_um=_nrv_node_shift_to_x_shift_um(
                        node_shift,
                        diameter_um,
                        kind=kind,
                    ),
                )
            )
    rows.sort(key=lambda item: (item.fascicle_id, item.fiber_index))
    return rows


def _nrv_fiber_is_simulated(fascicle: Any, fiber_index: int) -> bool:
    """Return whether NRV masks keep this fiber in the simulation set."""

    for mask_label in getattr(fascicle, "sim_mask", ()):
        try:
            if not bool(fascicle.axons[mask_label].iloc[int(fiber_index)]):
                return False
        except Exception:
            continue
    return True


@_nrv_bridge_stage("nrv_bridge.population_from_nrv")
def population_from_nrv(
    nrv_population: Any,
    *,
    nerve_length_um: float | None = None,
    include_unmyelinated: bool = True,
    include_mrg: bool = True,
    unmyelinated_compartments: int = 0,
    name: str | None = "nrv",
) -> NRVAxonPopulation:
    """Build an AxonFleet population from an NRV nerve or fascicle population.

    This is the first canonical bridge: NRV defines fiber placement and
    diameters; AxonFleet receives a one-dimensional axon population. No
    extracellular footprint is sampled here.
    """

    length_um = _resolve_nerve_length_um(nrv_population, nerve_length_um)
    rows = _fiber_rows_from_nrv(
        nrv_population,
        include_unmyelinated=include_unmyelinated,
        include_mrg=include_mrg,
    )
    templates: dict[tuple[Any, ...], Axon] = {}
    instances: list[AxonInstance] = []
    for row in rows:
        template_key = _fiber_axon_template_key(
            row,
            nerve_length_um=length_um,
            unmyelinated_compartments=unmyelinated_compartments,
        )
        axon = templates.get(template_key)
        if axon is None:
            axon = _axon_from_fiber_row(
                row,
                nerve_length_um=length_um,
                unmyelinated_compartments=unmyelinated_compartments,
            )
            templates[template_key] = axon
        instances.append(AxonInstance(axon))
    record_benchmark_metadata(
        nrv_population_rows=len(rows),
        nrv_population_unique_axon_templates=len(templates),
        nrv_population_template_cache_hits=len(rows) - len(templates),
    )
    return NRVAxonPopulation(
        population=AxonPopulation(instances, name=name),
        rows=tuple(rows),
    )


@_nrv_bridge_stage("nrv_bridge.footprints_from_nrv")
def footprints_from_nrv(
    nrv_geometry: Any,
    axons: NRVAxonPopulation | AxonPopulation | Sequence[AxonInstance],
    *,
    rows: Sequence[NRVFiberRow] | None = None,
    source_id: str = "nrv_fem",
    clear: bool = True,
) -> NRVFootprints:
    """Sample all NRV electrode footprints on an AxonFleet population.

    This is the second canonical bridge. `nrv_geometry` may be a nerve/fascicle
    carrying `.extra_stim` or an NRV stimulation object itself. Every electrode
    exposed by `extra_stim.electrodes` is sampled for every AxonFleet row.
    """

    population, row_tuple = _resolve_population_and_rows(axons, rows)
    extra_stim = _resolve_extra_stim(nrv_geometry)
    electrodes = tuple(getattr(extra_stim, "electrodes", ()))
    if not electrodes:
        raise ValueError("NRV geometry has no electrodes to sample.")

    all_footprints: list[tuple[ExtracellularFootprint, ...]] = []
    for instance, row in zip(population.instances, row_tuple, strict=True):
        positions_um = np.asarray(
            instance.axon.layout.position_values(unit=ureg.micrometer),
            dtype=float,
        )
        extra_stim.compute_electrodes_footprints(
            positions_um,
            float(row.y_um),
            float(row.z_um),
            _nrv_row_id(row),
        )
        row_footprints = []
        for electrode_index, electrode in enumerate(electrodes):
            values_mV_per_mA = np.asarray(
                electrode.get_footprint(),
                dtype=float,
            ).copy()
            row_footprints.append(
                ExtracellularFootprint.shared(
                    values=values_mV_per_mA,
                    positions=positions_um * ureg.micrometer,
                    voltage_unit=ureg.millivolt,
                    current_unit=ureg.milliampere,
                    source_id=f"{source_id}:{electrode_index}",
                    reference=(
                        "NRV electrode footprint sampled on AxonFleet intrinsic positions"
                    ),
                    metadata={
                        "source": "nrv",
                        "electrode_index": int(electrode_index),
                        "electrode_id": _electrode_id(electrode, electrode_index),
                        "fascicle_id": row.fascicle_id,
                        "fiber_index": int(row.fiber_index),
                        "fiber_y_um": float(row.y_um),
                        "fiber_z_um": float(row.z_um),
                        "nrv_footprint_unit": "mV/mA",
                    },
                )
            )
        all_footprints.append(tuple(row_footprints))
        if clear and hasattr(extra_stim, "clear_electrodes_footprints"):
            extra_stim.clear_electrodes_footprints()

    return NRVFootprints(
        population=population,
        rows=row_tuple,
        footprints=tuple(all_footprints),
        electrode_ids=tuple(
            _electrode_id(electrode, electrode_index)
            for electrode_index, electrode in enumerate(electrodes)
        ),
    )


def _axon_from_fiber_row(
    row: NRVFiberRow,
    *,
    nerve_length_um: float,
    unmyelinated_compartments: int = 0,
) -> Axon:
    """Build the AxonFleet axon template matching one NRV fiber row."""

    diameter = max(float(row.diameter_um), 0.2) * ureg.micrometer
    length = float(nerve_length_um) * ureg.micrometer
    if row.kind == "mrg":
        nodes = max(
            2,
            mrg_like_nodes_from_length(
                diameter,
                length,
                x_shift=float(row.x_shift_um) * ureg.micrometer,
            ),
        )
        return MRG(
            diameter=diameter,
            nodes=nodes,
            length=length,
            x_shift=float(row.x_shift_um) * ureg.micrometer,
        )
    if row.kind == "rattay":
        compartments = int(unmyelinated_compartments)
        if compartments <= 0:
            compartments = max(3, int(float(nerve_length_um) // 25))
        return RattayAberham(
            length=length,
            diameter=diameter,
            compartments=compartments,
            celsius=37.0 * ureg.degree_Celsius,
        )
    return HodgkinHuxley(
        length=length,
        diameter=diameter,
        compartments=max(3, int(float(nerve_length_um) // 25)),
        celsius=6.3 * ureg.degree_Celsius,
    )


def _fiber_axon_template_key(
    row: NRVFiberRow,
    *,
    nerve_length_um: float,
    unmyelinated_compartments: int,
) -> tuple[Any, ...]:
    """Return the complete constructor key for an NRV-derived axon template."""

    diameter_um = max(float(row.diameter_um), 0.2)
    if row.kind == "mrg":
        return (
            row.kind,
            diameter_um,
            float(nerve_length_um),
            float(row.x_shift_um),
        )
    compartments = int(unmyelinated_compartments)
    if compartments <= 0:
        compartments = max(3, int(float(nerve_length_um) // 25))
    return (
        row.kind,
        diameter_um,
        float(nerve_length_um),
        compartments,
    )


def _stimulation_from_footprint(
    footprint: ExtracellularFootprint,
    *,
    stimulus: Stimulus,
    drive_id: DriveId | str = DriveId("nrv_electrode_0"),
    metadata: Mapping[str, Any] | None = None,
) -> ExtracellularStimulation:
    """Wrap one sampled NRV footprint and one AxonFleet stimulus."""

    drive = ExtracellularDrive(
        id=_drive_id(drive_id),
        footprint=footprint,
        stimulus=stimulus,
        metadata={"source": "nrv", **dict(metadata or {})},
    )
    return ExtracellularStimulation([drive])


def _nrv_row_id(row: NRVFiberRow) -> int:
    """Return a stable integer ID for single-fiber NRV footprint sampling."""

    try:
        fascicle_id = int(row.fascicle_id)
    except ValueError:
        fascicle_id = 0
    return fascicle_id * 1_000_000 + int(row.fiber_index)


def _iter_nrv_fascicles(nrv_population: Any) -> tuple[tuple[Any, Any], ...]:
    if hasattr(nrv_population, "fascicles"):
        return tuple(nrv_population.fascicles.items())
    if hasattr(nrv_population, "axons") and hasattr(nrv_population.axons, "axon_pop"):
        fascicle_id = getattr(nrv_population, "ID", 0)
        return ((fascicle_id, nrv_population),)
    raise TypeError("nrv_population must be an NRV nerve or fascicle-like object.")


def _resolve_nerve_length_um(nrv_population: Any, nerve_length_um: float | None) -> float:
    if nerve_length_um is not None:
        return float(nerve_length_um)
    for candidate in (nrv_population, *_fascicle_values(nrv_population)):
        for attr in ("length", "L", "L_ax", "axon_length"):
            if hasattr(candidate, attr):
                try:
                    return _as_um(getattr(candidate, attr))
                except (TypeError, ValueError):
                    continue
    raise ValueError("nerve_length_um is required when the NRV object has no readable length.")


def _fascicle_values(nrv_population: Any) -> tuple[Any, ...]:
    if hasattr(nrv_population, "fascicles"):
        return tuple(nrv_population.fascicles.values())
    return ()


def _as_um(value: Any) -> float:
    if hasattr(value, "to"):
        return float(value.to(ureg.micrometer).magnitude)
    return float(value)


def _resolve_population_and_rows(
    axons: NRVAxonPopulation | AxonPopulation | Sequence[AxonInstance],
    rows: Sequence[NRVFiberRow] | None,
) -> tuple[AxonPopulation, tuple[NRVFiberRow, ...]]:
    if isinstance(axons, NRVAxonPopulation):
        if rows is not None and tuple(rows) != axons.rows:
            raise ValueError("rows must not conflict with the NRVAxonPopulation rows.")
        return axons.population, axons.rows
    population = axons if isinstance(axons, AxonPopulation) else AxonPopulation(axons)
    if rows is None:
        raise ValueError("rows are required when axons is not an NRVAxonPopulation.")
    row_tuple = tuple(rows)
    if len(population) != len(row_tuple):
        raise ValueError("population and rows must have the same length.")
    return population, row_tuple


def _resolve_extra_stim(nrv_geometry: Any) -> Any:
    if hasattr(nrv_geometry, "extra_stim"):
        extra_stim = nrv_geometry.extra_stim
    else:
        extra_stim = nrv_geometry
    if not hasattr(extra_stim, "compute_electrodes_footprints"):
        raise TypeError(
            "nrv_geometry must be an NRV object with extra_stim or an NRV stimulation object."
        )
    return extra_stim


def _electrode_id(electrode: Any, index: int) -> str:
    for attr in ("ID", "id", "label", "name"):
        if hasattr(electrode, attr):
            value = getattr(electrode, attr)
            if value is not None:
                return str(value)
    return f"electrode{int(index)}"


def _drive_id(value: DriveId | str) -> DriveId:
    if isinstance(value, DriveId):
        return value
    return DriveId(str(value))


__all__ = [
    "NRVAxonPopulation",
    "NRVFiberRow",
    "NRVFootprints",
    "footprints_from_nrv",
    "population_from_nrv",
]
