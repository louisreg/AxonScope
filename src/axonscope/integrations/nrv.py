"""NRV-to-AxonScope handoff helpers.

This module intentionally does not import NRV. NRV still owns external nerve
geometry, fiber placement, FEM fields, and its result objects; these helpers
only turn already-built NRV objects into AxonScope fiber metadata, sampled
footprints, stimulation objects, and compact comparison rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.axons import (
    Axon,
    HodgkinHuxley,
    MRG,
    RattayAberham,
    mrg_like_nodes_from_length,
)
from axonscope.axons.templates import mrg_like_node_spacing
from axonscope.identifiers import DriveId
from axonscope.stimulation import (
    ExtracellularDrive,
    ExtracellularFootprint,
    ExtracellularStimulation,
    Stimulus,
)
from axonscope.utils.units import Q_, ureg


FiberKind = Literal["hh", "rattay", "mrg"]


@dataclass(frozen=True)
class NRVFiberRow:
    """One NRV fiber after conversion to AxonScope intrinsic metadata."""

    fascicle_id: str
    fiber_index: int
    kind: FiberKind
    diameter_um: float
    y_um: float
    z_um: float
    node_shift: float = 0.0
    x_shift_um: float = 0.0


@dataclass(frozen=True)
class NRVLifeElectrodeSetup:
    """NRV LIFE/FEM object plus electrode placement metadata."""

    extra_stim: Any
    diameter_um: float
    length_um: float
    x_offset_um: float
    y_um: float
    z_um: float


@dataclass(frozen=True)
class NRVFiberContext:
    """One AxonScope axon plus its current-independent NRV LIFE footprint."""

    row: NRVFiberRow
    axon: Any
    positions_um: np.ndarray
    footprint: ExtracellularFootprint


@dataclass(frozen=True)
class NRVActivationComparison:
    """Fiber-by-fiber agreement between NRV and AxonScope activation."""

    row: NRVFiberRow
    nrv_activated: bool
    axonscope_activated: bool

    @property
    def matched(self) -> bool:
        """Return whether NRV and AxonScope agree for this fiber."""

        return bool(self.nrv_activated == self.axonscope_activated)


def fiber_kind_from_nrv(nrv_type: int, *, include_mrg: bool) -> FiberKind:
    """Map NRV's fiber type code to the AxonScope template used by examples."""

    if int(nrv_type) == 1:
        return "mrg" if include_mrg else "rattay"
    return "rattay"


def nrv_node_shift_to_x_shift_um(
    node_shift: float,
    diameter_um: float,
    *,
    kind: FiberKind,
) -> float:
    """Convert NRV's fractional MRG node shift to AxonScope MRG phase."""

    if kind != "mrg":
        return 0.0
    node_spacing_um = mrg_like_node_spacing(Q_(float(diameter_um), "micrometer"))
    return float(node_shift) * node_spacing_um


def build_synthetic_4_fascicle_nerve(
    nrv_module: Any,
    *,
    nerve_length_um: float,
    nerve_diameter_um: float,
    fascicle_diameter_um: float,
    fascicle_offset_um: float,
    axons_per_fascicle: int,
    percent_unmyelinated: float,
    delta_trace_um: float,
) -> Any:
    """Create one NRV nerve with four circular fascicles.

    This is the reproducible geometry used by the with-NRV example and Kaggle
    performance presets. NRV owns the geometry and fiber placement; AxonScope
    only receives the extracted rows and sampled footprints later.
    """

    nerve = nrv_module.nerve(
        diameter=nrv_numeric(nerve_diameter_um),
        length=nrv_numeric(nerve_length_um),
    )
    for fascicle_id, center in enumerate(synthetic_fascicle_centers(fascicle_offset_um)):
        fascicle = nrv_module.fascicle(ID=fascicle_id)
        fascicle.set_geometry(
            geometry=nrv_module.create_cshape(
                center=tuple(float(value) for value in center),
                diameter=nrv_numeric(fascicle_diameter_um),
            )
        )
        nerve.add_fascicle(fascicle)

    for fascicle in nerve.fascicles.values():
        fascicle.fill(
            n_ax=int(axons_per_fascicle),
            percent_unmyel=float(percent_unmyelinated),
            delta_trace=float(delta_trace_um),
            with_node_shift=True,
        )
    return nerve


def build_histology_contour_nerve(
    nrv_module: Any,
    *,
    nerve_length_um: float,
    nerve_diameter_um: float,
    axons_per_fascicle: int,
    percent_unmyelinated: float,
    delta_trace_um: float,
    fascicle_epsilon_fraction: float = 0.002,
) -> Any:
    """Create NRV's bundled histology-contour nerve used by comparison benches."""

    _, fascicle_contours = load_histology_contours(
        nrv_module,
        nerve_diameter_um=nerve_diameter_um,
        fascicle_epsilon_fraction=fascicle_epsilon_fraction,
    )
    nerve = nrv_module.nerve(
        diameter=nrv_numeric(nerve_diameter_um),
        length=nrv_numeric(nerve_length_um),
    )
    for fascicle_id, points in enumerate(fascicle_contours):
        geometry = nrv_module.create_cshape(vertices=np.asarray(points, dtype=float))
        fascicle = nrv_module.fascicle(ID=fascicle_id)
        fascicle.set_geometry(geometry=geometry)
        nerve.add_fascicle(fascicle)

    for fascicle in nerve.fascicles.values():
        fascicle.fill(
            n_ax=int(axons_per_fascicle),
            percent_unmyel=float(percent_unmyelinated),
            delta_trace=float(delta_trace_um),
            with_node_shift=True,
        )
    return nerve


def build_nerve_from_mode(
    nrv_module: Any,
    *,
    geometry_mode: str,
    nerve_length_um: float,
    nerve_diameter_um: float,
    axons_per_fascicle: int,
    percent_unmyelinated: float,
    delta_trace_um: float,
    synthetic_fascicle_diameter_um: float = 250.0,
    synthetic_fascicle_offset_um: float = 250.0,
    fascicle_epsilon_fraction: float = 0.002,
) -> Any:
    """Build an NRV nerve from the named geometry mode used by examples/benches."""

    if geometry_mode == "synthetic_4_fascicles":
        return build_synthetic_4_fascicle_nerve(
            nrv_module,
            nerve_length_um=nerve_length_um,
            nerve_diameter_um=nerve_diameter_um,
            fascicle_diameter_um=synthetic_fascicle_diameter_um,
            fascicle_offset_um=synthetic_fascicle_offset_um,
            axons_per_fascicle=axons_per_fascicle,
            percent_unmyelinated=percent_unmyelinated,
            delta_trace_um=delta_trace_um,
        )
    if geometry_mode == "histology":
        return build_histology_contour_nerve(
            nrv_module,
            nerve_length_um=nerve_length_um,
            nerve_diameter_um=nerve_diameter_um,
            axons_per_fascicle=axons_per_fascicle,
            percent_unmyelinated=percent_unmyelinated,
            delta_trace_um=delta_trace_um,
            fascicle_epsilon_fraction=fascicle_epsilon_fraction,
        )
    raise ValueError(f"Unsupported NRV geometry mode: {geometry_mode!r}")


def load_histology_contours(
    nrv_module: Any,
    *,
    nerve_diameter_um: float,
    fascicle_epsilon_fraction: float = 0.002,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load NRV's bundled histology image and return rescaled contours."""

    import cv2

    image_path = (
        Path(nrv_module.__path__[0])
        / "_misc"
        / "geom"
        / "smoothed_edges_white.png"
    )
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read NRV geometry image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise RuntimeError("NRV geometry image did not yield contours.")

    hierarchy = hierarchy.squeeze()
    nerve_id = 2
    nerve_points_pix = contours[nerve_id].squeeze()
    center_pix = np.mean(nerve_points_pix, axis=0)
    nerve_points = nerve_points_pix - center_pix
    radius_pix = np.max(np.abs(nerve_points))
    scale = float(nerve_diameter_um) / (2.0 * radius_pix)
    scale_xy = scale * np.asarray([1.0, -1.0])
    nerve_points = nerve_points * scale_xy

    fascicle_points = []
    for index, contour in enumerate(contours):
        if hierarchy[index, -1] == nerve_id:
            points = simplify_cv2_contour(
                cv2,
                contour,
                epsilon_fraction=fascicle_epsilon_fraction,
            )
            fascicle_points.append((points - center_pix) * scale_xy)

    return np.asarray(nerve_points, dtype=float), fascicle_points


def simplify_cv2_contour(
    cv2_module: Any,
    contour: Any,
    *,
    epsilon_fraction: float,
) -> np.ndarray:
    """Return a valid open contour suitable for NRV/Gmsh polygon geometry."""

    points = np.asarray(contour).squeeze()
    if float(epsilon_fraction) > 0.0:
        epsilon = float(epsilon_fraction) * cv2_module.arcLength(contour, True)
        simplified = cv2_module.approxPolyDP(contour, epsilon, True).squeeze()
        if np.asarray(simplified).ndim == 2 and len(simplified) >= 3:
            points = np.asarray(simplified)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("NRV contour simplification produced an invalid polygon.")
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    return np.asarray(points, dtype=float)


def synthetic_fascicle_centers(offset_um: float) -> tuple[tuple[float, float], ...]:
    """Return four fascicle centers inside the synthetic nerve cross-section."""

    offset = float(offset_um)
    return (
        (offset, 0.0),
        (0.0, offset),
        (-offset, 0.0),
        (0.0, -offset),
    )


def nrv_numeric(value: float) -> int | float:
    """Preserve NRV example integer parameters when configured as floats."""

    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def attach_life_fem_electrode(
    nrv_module: Any,
    nerve: Any,
    *,
    nerve_length_um: float,
    life_fascicle_id: str = "0",
    life_diameter_um: float = 25.0,
    life_length_um: float = 1_000.0,
    stimulus_start_ms: float = 0.1,
    pulse_duration_ms: float = 0.1,
    validation_current_uA: float = 60.0,
    fem_n_proc: int | None = None,
    gmsh_n_core: int | None = 1,
) -> NRVLifeElectrodeSetup:
    """Attach one NRV LIFE/FEM electrode and return its AxonScope handoff data."""

    fascicle_key: Any = life_fascicle_id
    if fascicle_key not in nerve.fascicles:
        fascicle_key = int(life_fascicle_id)
    fascicle = nerve.fascicles[fascicle_key]
    life_y_um, life_z_um = fascicle.center
    life_x_offset_um = (float(nerve_length_um) - float(life_length_um)) / 2.0

    extra_stim = nrv_module.FEM_stimulation(
        endo_mat="endoneurium_ranck",
        peri_mat="perineurium",
        epi_mat="epineurium",
        ext_mat="saline",
        n_proc=fem_n_proc,
    )
    electrode = nrv_module.LIFE_electrode(
        "LIFE_2",
        float(life_diameter_um),
        float(life_length_um),
        life_x_offset_um,
        life_y_um,
        life_z_um,
    )
    pulse_stim = nrv_module.stimulus()
    pulse_stim.pulse(
        float(stimulus_start_ms),
        -float(validation_current_uA),
        float(pulse_duration_ms),
    )
    extra_stim.add_electrode(electrode, pulse_stim)
    nerve.attach_extracellular_stimulation(extra_stim)
    if fem_n_proc is not None:
        nerve.extra_stim.set_n_proc(int(fem_n_proc))
    if gmsh_n_core is not None:
        nerve.extra_stim.model.mesh.n_core = int(gmsh_n_core)

    return NRVLifeElectrodeSetup(
        extra_stim=nerve.extra_stim,
        diameter_um=float(life_diameter_um),
        length_um=float(life_length_um),
        x_offset_um=float(life_x_offset_um),
        y_um=float(life_y_um),
        z_um=float(life_z_um),
    )


def extract_fiber_rows(
    nerve: Any,
    *,
    include_unmyelinated: bool,
    include_mrg: bool = True,
) -> list[NRVFiberRow]:
    """Map NRV fascicle populations to AxonScope fiber rows.

    The returned rows keep NRV cross-section coordinates as metadata for
    external footprint sampling. They do not assign real-world coordinates to
    `AxonInstance`; AxonScope still sees intrinsic x positions only.
    """

    rows: list[NRVFiberRow] = []
    for fascicle_id, fascicle in nerve.fascicles.items():
        table = fascicle.axons.axon_pop
        for fiber_index, table_row in table.iterrows():
            if not nrv_fiber_is_simulated(fascicle, int(fiber_index)):
                continue
            nrv_type = int(float(table_row.get("types", 0)))
            kind = fiber_kind_from_nrv(nrv_type, include_mrg=include_mrg)
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
                    x_shift_um=nrv_node_shift_to_x_shift_um(
                        node_shift,
                        diameter_um,
                        kind=kind,
                    ),
                )
            )
    rows.sort(key=lambda item: (item.fascicle_id, item.fiber_index))
    return rows


def nrv_fiber_is_simulated(fascicle: Any, fiber_index: int) -> bool:
    """Return whether NRV masks keep this fiber in the simulation set."""

    for mask_label in getattr(fascicle, "sim_mask", ()):
        try:
            if not bool(fascicle.axons[mask_label].iloc[int(fiber_index)]):
                return False
        except Exception:
            continue
    return True


def select_rows(rows: Sequence[NRVFiberRow], *, limit: int) -> list[NRVFiberRow]:
    """Return all rows when `limit <= 0`, otherwise the first requested rows."""

    if int(limit) <= 0:
        return list(rows)
    return list(rows[: int(limit)])


def sample_life_footprint(
    life_setup: NRVLifeElectrodeSetup,
    *,
    positions_um: Sequence[float],
    row: NRVFiberRow,
    source_id: str = "nrv_life_fem",
) -> ExtracellularFootprint:
    """Sample NRV's current-independent LIFE/FEM footprint for one fiber row."""

    life_setup.extra_stim.compute_electrodes_footprints(
        np.asarray(positions_um, dtype=float),
        float(row.y_um),
        float(row.z_um),
        nrv_row_id(row),
    )
    values_mV_per_mA = np.asarray(
        life_setup.extra_stim.electrodes[0].get_footprint(),
        dtype=float,
    ).copy()
    life_setup.extra_stim.clear_electrodes_footprints()

    mesh = getattr(getattr(life_setup.extra_stim, "model", None), "mesh", None)
    return ExtracellularFootprint.shared(
        values=values_mV_per_mA,
        positions=np.asarray(positions_um, dtype=float) * ureg.micrometer,
        voltage_unit=ureg.millivolt,
        current_unit=ureg.milliampere,
        source_id=source_id,
        reference="NRV FEM LIFE footprint sampled on AxonScope intrinsic positions",
        metadata={
            "source": "nrv.FEM_stimulation/LIFE_electrode",
            "life_diameter_um": float(life_setup.diameter_um),
            "life_length_um": float(life_setup.length_um),
            "life_x_offset_um": float(life_setup.x_offset_um),
            "life_y_um": float(life_setup.y_um),
            "life_z_um": float(life_setup.z_um),
            "gmsh_n_core": None if mesh is None else getattr(mesh, "n_core", None),
            "nrv_footprint_unit": "mV/mA",
        },
    )


def sample_life_context(
    row: NRVFiberRow,
    *,
    axon: Any,
    life_setup: NRVLifeElectrodeSetup,
) -> NRVFiberContext:
    """Build one AxonScope fiber context by sampling its NRV LIFE footprint."""

    positions_um = np.asarray(axon.layout.position_values(unit=ureg.micrometer), dtype=float)
    return NRVFiberContext(
        row=row,
        axon=axon,
        positions_um=positions_um,
        footprint=sample_life_footprint(
            life_setup,
            positions_um=positions_um,
            row=row,
        ),
    )


def axon_from_fiber_row(
    row: NRVFiberRow,
    *,
    nerve_length_um: float,
    unmyelinated_compartments: int = 0,
) -> Axon:
    """Build the AxonScope axon template matching one NRV fiber row."""

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


def life_context_from_fiber_row(
    row: NRVFiberRow,
    *,
    life_setup: NRVLifeElectrodeSetup,
    nerve_length_um: float,
    unmyelinated_compartments: int = 0,
) -> NRVFiberContext:
    """Build one AxonScope row and sample its NRV LIFE footprint."""

    axon = axon_from_fiber_row(
        row,
        nerve_length_um=nerve_length_um,
        unmyelinated_compartments=unmyelinated_compartments,
    )
    return sample_life_context(row, axon=axon, life_setup=life_setup)


def life_stimulation_from_footprint(
    footprint: ExtracellularFootprint,
    *,
    current: Any,
    start_ms: float,
    pulse_duration_ms: float,
    drive_id: DriveId | str = DriveId("nrv_life"),
    cathodic: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> ExtracellularStimulation:
    """Wrap a sampled NRV LIFE footprint in AxonScope stimulation objects."""

    stimulus = life_pulse_stimulus(
        current=current,
        start_ms=start_ms,
        pulse_duration_ms=pulse_duration_ms,
        cathodic=cathodic,
    )
    drive = ExtracellularDrive(
        id=_drive_id(drive_id),
        footprint=footprint,
        stimulus=stimulus,
        metadata={"source": "nrv_life_fem", **dict(metadata or {})},
    )
    return ExtracellularStimulation([drive])


def life_simulation_from_context(
    context: NRVFiberContext,
    *,
    current_uA: float,
    start_ms: float,
    pulse_duration_ms: float,
) -> AxonInstance:
    """Attach one sampled NRV LIFE footprint to a fresh AxonScope simulation."""

    simulation = AxonInstance(context.axon)
    simulation.add_extracellular_stimulation(
        stimulation=life_stimulation_from_footprint(
            context.footprint,
            current=float(current_uA) * ureg.microampere,
            start_ms=start_ms,
            pulse_duration_ms=pulse_duration_ms,
        )
    )
    return simulation


def replace_life_current(
    simulation: AxonInstance,
    current: Any,
    *,
    start_ms: float,
    pulse_duration_ms: float,
    drive_id: DriveId | str = DriveId("nrv_life"),
    cathodic: bool = True,
) -> None:
    """Replace one NRV LIFE drive stimulus on an existing AxonScope simulation."""

    if simulation.extracellular_stimulation is None:
        raise ValueError("simulation has no extracellular stimulation to update.")
    drive_key = _drive_id(drive_id)
    updated = simulation.extracellular_stimulation.replace_drive(
        drive_key,
        stimulus=life_pulse_stimulus(
            current=current,
            start_ms=start_ms,
            pulse_duration_ms=pulse_duration_ms,
            cathodic=cathodic,
        ),
    )
    simulation.add_extracellular_stimulation(stimulation=updated, replace=True)


def life_pulse_stimulus(
    *,
    current: Any,
    start_ms: float,
    pulse_duration_ms: float,
    cathodic: bool = True,
) -> Stimulus:
    """Create the LIFE pulse used by NRV/AxonScope recruitment comparisons."""

    amplitude = _current_quantity(current)
    if cathodic:
        amplitude = -amplitude
    return Stimulus.pulse(
        start=float(start_ms) * ureg.millisecond,
        duration=float(pulse_duration_ms) * ureg.millisecond,
        amplitude=amplitude,
    )


def nrv_activation_by_row(
    nrv_result: Any,
    nerve: Any,
    rows: Sequence[NRVFiberRow],
    *,
    t_start_ms: float,
) -> dict[tuple[str, int], bool]:
    """Return NRV recruitment flags keyed by `(fascicle_id, fiber_index)`."""

    activated: dict[tuple[str, int], bool] = {}
    sim_index_by_fascicle: dict[str, dict[int, int]] = {}
    for row in rows:
        fascicle_key = f"fascicle{row.fascicle_id}"
        fascicle_result = nrv_result[fascicle_key]
        if fascicle_key not in sim_index_by_fascicle:
            fascicle = nrv_fascicle_by_id(nerve, row.fascicle_id)
            sim_list = list(getattr(fascicle, "sim_list", ()))
            sim_index_by_fascicle[fascicle_key] = {
                int(fiber_index): int(sim_index)
                for sim_index, fiber_index in enumerate(sim_list)
            }
        try:
            sim_index = sim_index_by_fascicle[fascicle_key][int(row.fiber_index)]
        except KeyError as exc:
            raise RuntimeError(
                f"NRV did not simulate fascicle={row.fascicle_id} fiber={row.fiber_index}; "
                "check the NRV simulation masks."
            ) from exc
        axon_key = f"axon{sim_index}"
        try:
            axon_result = fascicle_result[axon_key]
        except KeyError as exc:
            raise RuntimeError(
                f"NRV result for fascicle={row.fascicle_id} has no {axon_key} entry."
            ) from exc
        if "recruited" in axon_result:
            activated[row_key(row)] = bool(axon_result["recruited"])
        else:
            activated[row_key(row)] = bool(
                axon_result.is_recruited(vm_key="V_mem", t_start=float(t_start_ms))
            )
    return activated


def activation_comparisons(
    rows: Sequence[NRVFiberRow],
    *,
    nrv_activated: Mapping[tuple[str, int], bool],
    axonscope_activated: Sequence[bool],
) -> list[NRVActivationComparison]:
    """Pair NRV and AxonScope activation flags for the same fiber rows."""

    return [
        NRVActivationComparison(
            row=row,
            nrv_activated=bool(nrv_activated.get(row_key(row), False)),
            axonscope_activated=bool(axonscope_active),
        )
        for row, axonscope_active in zip(rows, axonscope_activated, strict=True)
    ]


def nrv_fascicle_by_id(nerve: Any, fascicle_id: str) -> Any:
    """Return an NRV fascicle by string or integer id."""

    if fascicle_id in nerve.fascicles:
        return nerve.fascicles[fascicle_id]
    try:
        return nerve.fascicles[int(fascicle_id)]
    except (KeyError, ValueError) as exc:
        raise KeyError(f"Unknown NRV fascicle id {fascicle_id!r}.") from exc


def nrv_row_id(row: NRVFiberRow) -> int:
    """Return a stable integer ID for single-fiber NRV footprint sampling."""

    try:
        fascicle_id = int(row.fascicle_id)
    except ValueError:
        fascicle_id = 0
    return fascicle_id * 1_000_000 + int(row.fiber_index)


def row_key(row: NRVFiberRow) -> tuple[str, int]:
    """Return the stable fiber comparison key."""

    return (str(row.fascicle_id), int(row.fiber_index))


def _drive_id(value: DriveId | str) -> DriveId:
    if isinstance(value, DriveId):
        return value
    return DriveId(str(value))


def _current_quantity(value: Any) -> Any:
    if hasattr(value, "to"):
        return value
    return float(value) * ureg.microampere


__all__ = [
    "FiberKind",
    "NRVActivationComparison",
    "NRVFiberContext",
    "NRVFiberRow",
    "NRVLifeElectrodeSetup",
    "attach_life_fem_electrode",
    "axon_from_fiber_row",
    "activation_comparisons",
    "build_histology_contour_nerve",
    "build_nerve_from_mode",
    "build_synthetic_4_fascicle_nerve",
    "extract_fiber_rows",
    "fiber_kind_from_nrv",
    "life_context_from_fiber_row",
    "life_pulse_stimulus",
    "life_simulation_from_context",
    "life_stimulation_from_footprint",
    "load_histology_contours",
    "nrv_numeric",
    "nrv_activation_by_row",
    "nrv_fascicle_by_id",
    "nrv_fiber_is_simulated",
    "nrv_node_shift_to_x_shift_um",
    "nrv_row_id",
    "replace_life_current",
    "row_key",
    "sample_life_context",
    "sample_life_footprint",
    "select_rows",
    "simplify_cv2_contour",
    "synthetic_fascicle_centers",
]
