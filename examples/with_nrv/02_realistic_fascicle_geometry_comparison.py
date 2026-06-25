"""Compare realistic NRV fascicle activation with AxonScope.

Run:
    python examples/with_nrv/02_realistic_fascicle_geometry_comparison.py

This example mirrors NRV's `25_test_fit_fasc.py` workflow: contours come from
NRV's bundled fascicle image, NRV fills the fascicles, a LIFE electrode is
placed in the first fascicle, and AxonScope receives the resulting fiber table.
The important handoffs are:

- NRV `node_shift`, a fraction of the MRG internode spacing, becomes
  unit-bearing `MRG(..., x_shift=...)`.
- NRV's FEM/LIFE footprint is sampled on AxonScope's intrinsic positions and
  wrapped as `ExtracellularFootprint -> ExtracellularDrive`.
- By default, all NRV-simulated fibers are run in AxonScope with the same
  pulse, duration, and dt as the NRV example. Use `--axons-per-fascicle` for a
  smaller debug nerve; `--max-fibers` and `--simulate-fibers` only limit plotted
  or compared AxonScope rows after NRV placement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

import axonscope as axs
from axonscope.integrations.nrv import (
    FiberKind,
    fiber_kind_from_nrv,
    nrv_node_shift_to_x_shift_um,
)


@dataclass(frozen=True)
class RealisticFiberRow:
    """One NRV fiber row after conversion to AxonScope layout metadata."""

    fascicle_id: str
    fiber_index: int
    kind: FiberKind
    diameter_um: float
    y_um: float
    z_um: float
    node_shift: float
    x_shift_um: float


@dataclass(frozen=True)
class LayoutComparison:
    """Geometry values used to compare NRV node-shift semantics to AxonScope."""

    row: RealisticFiberRow
    node_spacing_um: float
    axonscope_nodes_um: np.ndarray
    expected_nodes_um: np.ndarray


@dataclass(frozen=True)
class LifeElectrodeSetup:
    """NRV LIFE electrode context used by both NRV and AxonScope simulations."""

    extra_stim: Any
    diameter_um: float
    length_um: float
    x_offset_um: float
    y_um: float
    z_um: float


@dataclass(frozen=True)
class SimulationComparison:
    """One single-fiber NRV/AxonScope simulation comparison."""

    row: RealisticFiberRow
    t_ms: np.ndarray
    x_nodes_um: np.ndarray
    vm_axonscope_mV: np.ndarray
    vm_nrv_mV: np.ndarray
    rmse_mV: float
    max_abs_mV: float
    aligned_rmse_mV: float
    aligned_max_abs_mV: float
    peak_delta_mV: float
    center_peak_delta_mV: float
    center_peak_lag_ms: float


@dataclass(frozen=True)
class ActivationComparison:
    """One row of fiber-by-fiber recruitment comparison."""

    row: RealisticFiberRow
    nrv_activated: bool
    axonscope_activated: bool

    @property
    def matched(self) -> bool:
        """Return whether NRV and AxonScope agree for this fiber."""

        return bool(self.nrv_activated == self.axonscope_activated)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    console = Console(width=120)

    import nrv

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    nerve_contour, fascicle_contours = load_nrv_contours(
        nrv,
        nerve_diameter_um=args.nerve_diameter_um,
    )
    nerve = build_nrv_nerve(
        nrv,
        fascicle_contours,
        nerve_length_um=args.nerve_length_um,
        nerve_diameter_um=args.nerve_diameter_um,
        axons_per_fascicle=args.axons_per_fascicle,
        percent_unmyelinated=args.percent_unmyelinated,
        delta_trace_um=args.delta_trace_um,
    )
    life_setup = attach_life_electrode(nrv, nerve, args)
    rows = extract_fiber_rows(nerve, include_unmyelinated=args.include_unmyelinated)
    rows = select_rows(rows, limit=args.max_fibers)
    comparisons = compare_mrg_layouts(rows, nerve_length_um=args.nerve_length_um)

    print_geometry_summary(console, nerve, rows, comparisons)
    plot_fiber_map(axes[0], nerve_contour, fascicle_contours, rows, life_setup)
    plot_node_alignment(
        axes[1],
        comparisons,
        max_lanes=min(args.plot_fibers, len(comparisons)),
    )

    if not args.no_simulation:
        activation_comparisons = compare_population_activation(
            console,
            nerve,
            rows=select_rows(rows, limit=args.simulate_fibers),
            args=args,
            life_setup=life_setup,
        )
        print_activation_summary(
            console,
            activation_comparisons,
            print_limit=args.print_fiber_limit,
        )
        plot_activation_comparison(axes[0], activation_comparisons)
        if args.trace_fibers > 0:
            simulation_comparisons = compare_single_fiber_simulations(
                console,
                select_rows(rows, limit=args.trace_fibers),
                args,
                life_setup=life_setup,
            )
            print_simulation_summary(console, simulation_comparisons)
            plot_simulation_comparisons(simulation_comparisons)

    plt.show()


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nerve-diameter-um", type=float, default=1_000.0)
    parser.add_argument("--nerve-length-um", type=float, default=10_000.0)
    parser.add_argument("--axons-per-fascicle", type=int, default=100)
    parser.add_argument("--percent-unmyelinated", type=float, default=0.7)
    parser.add_argument("--delta-trace-um", type=float, default=10.0)
    parser.add_argument("--max-fibers", type=int, default=0)
    parser.add_argument("--plot-fibers", type=int, default=8)
    parser.set_defaults(include_unmyelinated=True)
    parser.add_argument("--include-unmyelinated", dest="include_unmyelinated", action="store_true")
    parser.add_argument("--mrg-only", dest="include_unmyelinated", action="store_false")
    parser.add_argument("--no-simulation", action="store_true")
    parser.add_argument("--simulate-fibers", type=int, default=0)
    parser.add_argument("--trace-fibers", type=int, default=0)
    parser.add_argument("--duration-ms", type=float, default=3.0)
    parser.add_argument("--dt-ms", type=float, default=0.001)
    parser.add_argument("--stimulus-start-ms", type=float, default=0.1)
    parser.add_argument("--pulse-duration-ms", type=float, default=0.1)
    parser.add_argument("--stimulus-current-uA", type=float, default=60.0)
    parser.add_argument("--activation-threshold-mV", type=float, default=0.0)
    parser.add_argument("--print-fiber-limit", type=int, default=40)
    parser.add_argument("--unmyelinated-compartments", type=int, default=0)
    parser.add_argument("--life-diameter-um", type=float, default=25.0)
    parser.add_argument("--life-length-um", type=float, default=1_000.0)
    parser.add_argument("--life-fascicle-id", type=str, default="0")
    parser.add_argument("--fem-n-proc", type=int, default=None)
    return parser.parse_args(argv)


def load_nrv_contours(
    nrv_module: Any,
    *,
    nerve_diameter_um: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load NRV's example histology image and return rescaled contours."""

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
    scale = nerve_diameter_um / (2.0 * radius_pix)
    scale_xy = scale * np.asarray([1.0, -1.0])
    nerve_points = nerve_points * scale_xy

    fascicle_points = []
    for index, contour in enumerate(contours):
        if hierarchy[index, -1] == nerve_id:
            fascicle_points.append((contour.squeeze() - center_pix) * scale_xy)

    return nerve_points, fascicle_points


def build_nrv_nerve(
    nrv_module: Any,
    fascicle_contours: Sequence[np.ndarray],
    *,
    nerve_length_um: float,
    nerve_diameter_um: float,
    axons_per_fascicle: int,
    percent_unmyelinated: float,
    delta_trace_um: float,
) -> Any:
    """Create the same realistic NRV nerve object used to generate fiber rows."""

    nerve = nrv_module.nerve(
        diameter=nrv_numeric(nerve_diameter_um),
        length=nrv_numeric(nerve_length_um),
    )
    for fascicle_id, points in enumerate(fascicle_contours):
        vertex_count = 50
        indices = np.arange(vertex_count + 1) * points.shape[0] // vertex_count
        indices[-1] -= 1
        geometry = nrv_module.create_cshape(vertices=points[indices])
        fascicle = nrv_module.fascicle(ID=fascicle_id)
        fascicle.set_geometry(geometry=geometry)
        nerve.add_fascicle(fascicle)

    for fascicle in nerve.fascicles.values():
        fascicle.fill(
            n_ax=axons_per_fascicle,
            percent_unmyel=percent_unmyelinated,
            delta_trace=delta_trace_um,
            with_node_shift=True,
        )
    return nerve


def nrv_numeric(value: float) -> int | float:
    """Preserve NRV example integer parameters when argparse parsed them as floats."""

    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def attach_life_electrode(
    nrv_module: Any,
    nerve: Any,
    args: argparse.Namespace,
) -> LifeElectrodeSetup:
    """Attach the NRV-example LIFE electrode and return the configured context."""

    fascicle_key: Any = args.life_fascicle_id
    if fascicle_key not in nerve.fascicles:
        fascicle_key = int(args.life_fascicle_id)
    fascicle = nerve.fascicles[fascicle_key]
    life_y_um, life_z_um = fascicle.center
    life_x_offset_um = (args.nerve_length_um - args.life_length_um) / 2.0

    extra_stim = nrv_module.FEM_stimulation(
        endo_mat="endoneurium_ranck",
        peri_mat="perineurium",
        epi_mat="epineurium",
        ext_mat="saline",
        n_proc=args.fem_n_proc,
    )
    electrode = nrv_module.LIFE_electrode(
        "LIFE_2",
        args.life_diameter_um,
        args.life_length_um,
        life_x_offset_um,
        life_y_um,
        life_z_um,
    )
    pulse_stim = nrv_module.stimulus()
    pulse_stim.pulse(
        args.stimulus_start_ms,
        -args.stimulus_current_uA,
        args.pulse_duration_ms,
    )
    extra_stim.add_electrode(electrode, pulse_stim)
    nerve.attach_extracellular_stimulation(extra_stim)
    if args.fem_n_proc is not None:
        nerve.extra_stim.set_n_proc(args.fem_n_proc)

    return LifeElectrodeSetup(
        extra_stim=nerve.extra_stim,
        diameter_um=float(args.life_diameter_um),
        length_um=float(args.life_length_um),
        x_offset_um=float(life_x_offset_um),
        y_um=float(life_y_um),
        z_um=float(life_z_um),
    )


def extract_fiber_rows(
    nerve: Any,
    *,
    include_unmyelinated: bool,
) -> list[RealisticFiberRow]:
    """Map NRV fascicle populations to rows with AxonScope MRG node phases."""

    rows: list[RealisticFiberRow] = []
    for fascicle_id, fascicle in nerve.fascicles.items():
        table = fascicle.axons.axon_pop
        for fiber_index, row in table.iterrows():
            if not nrv_fiber_is_simulated(fascicle, int(fiber_index)):
                continue
            nrv_type = int(float(row.get("types", 0)))
            kind = fiber_kind_from_nrv(nrv_type, include_mrg=True)
            if kind != "mrg" and not include_unmyelinated:
                continue
            diameter_um = float(row.get("diameters", 1.0))
            node_shift = float(row.get("node_shift", 0.0))
            rows.append(
                RealisticFiberRow(
                    fascicle_id=str(fascicle_id),
                    fiber_index=int(fiber_index),
                    kind=kind,
                    diameter_um=diameter_um,
                    y_um=float(row.get("y", 0.0)),
                    z_um=float(row.get("z", 0.0)),
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


def select_rows(rows: Sequence[RealisticFiberRow], *, limit: int) -> list[RealisticFiberRow]:
    """Return all rows when limit <= 0, otherwise the first requested rows."""

    if int(limit) <= 0:
        return list(rows)
    return list(rows[: int(limit)])


def compare_mrg_layouts(
    rows: Sequence[RealisticFiberRow],
    *,
    nerve_length_um: float,
) -> list[LayoutComparison]:
    """Build AxonScope MRG layouts and compare their phased node positions."""

    comparisons: list[LayoutComparison] = []
    for row in rows:
        if row.kind != "mrg":
            continue
        diameter = row.diameter_um * axs.um
        nodes = max(
            2,
            axs.axons.mrg_like_nodes_from_length(
                diameter,
                nerve_length_um * axs.um,
                x_shift=row.x_shift_um * axs.um,
            ),
        )
        axon = axs.axons.MRG(
            diameter=diameter,
            nodes=nodes,
            x_shift=row.x_shift_um * axs.um,
        )
        node_spacing_um = axs.axons.mrg_like_node_spacing(diameter)
        expected = 0.5 + row.x_shift_um + node_spacing_um * np.arange(axon.nodes)
        comparisons.append(
            LayoutComparison(
                row=row,
                node_spacing_um=node_spacing_um,
                axonscope_nodes_um=axon.node_position_values(unit=axs.um),
                expected_nodes_um=expected,
            )
        )
    return comparisons


def compare_population_activation(
    console: Console,
    nerve: Any,
    *,
    rows: Sequence[RealisticFiberRow],
    args: argparse.Namespace,
    life_setup: LifeElectrodeSetup,
) -> list[ActivationComparison]:
    """Simulate NRV and AxonScope populations and compare activation per fiber."""

    row_list = list(rows)
    if not row_list:
        return []

    console.print(
        f"[bold]Population activation[/bold]: simulating {len(row_list)} AxonScope rows "
        f"and NRV nerve with postproc_script='is_recruited'."
    )
    nrv_result = nerve.simulate(
        t_sim=args.duration_ms,
        postproc_script="is_recruited",
        dt=args.dt_ms,
        unmyelinated_nseg=max(3, int(args.nerve_length_um // 25)),
        myelinated_nseg_per_sec=3,
    )
    nrv_activated = nrv_activation_by_row(
        nrv_result,
        nerve,
        row_list,
        t_start_ms=args.stimulus_start_ms,
    )

    simulations = [
        build_axonscope_simulation_from_row(row, args=args, life_setup=life_setup)
        for row in row_list
    ]
    activation = axs.Activation(
        threshold=args.activation_threshold_mV * axs.mV,
        blanking=args.stimulus_start_ms * axs.ms,
        target=axs.positions.ALL,
        name="activation",
    )
    axonscope_result = axs.simulate_pool(
        simulations,
        duration=args.duration_ms * axs.ms,
        dt=args.dt_ms * axs.ms,
        recording=axs.Recording.none(),
        observers=[activation],
        progress=True,
    )
    axonscope_activated = [
        axonscope_activation_from_observer(row_result, activation)
        for row_result in axonscope_result
    ]

    return [
        ActivationComparison(
            row=row,
            nrv_activated=bool(nrv_activated.get(row_key(row), False)),
            axonscope_activated=bool(axonscope_active),
        )
        for row, axonscope_active in zip(row_list, axonscope_activated, strict=True)
    ]


def build_axonscope_simulation_from_row(
    row: RealisticFiberRow,
    *,
    args: argparse.Namespace,
    life_setup: LifeElectrodeSetup,
) -> axs.AxonInstance:
    """Build one AxonScope simulation row from one NRV fiber row."""

    axon = build_axonscope_axon(row, args=args)
    positions_um = axon.layout.position_values(unit=axs.um)
    stimulus = axs.Stimulus.pulse(
        start=args.stimulus_start_ms * axs.ms,
        duration=args.pulse_duration_ms * axs.ms,
        amplitude=-args.stimulus_current_uA * axs.uA,
    )
    simulation = axs.AxonInstance(axon)
    simulation.add_extracellular_stimulation(
        stimulation=axonscope_stimulation_from_nrv_life(
            life_setup,
            positions_um=positions_um,
            row=row,
            stimulus=stimulus,
        )
    )
    return simulation


def build_axonscope_axon(row: RealisticFiberRow, *, args: argparse.Namespace) -> axs.axons.Axon:
    """Build the AxonScope axon template matching one NRV row."""

    diameter = max(float(row.diameter_um), 0.2) * axs.um
    if row.kind == "mrg":
        nodes = max(
            2,
            axs.axons.mrg_like_nodes_from_length(
                diameter,
                args.nerve_length_um * axs.um,
                x_shift=row.x_shift_um * axs.um,
            ),
        )
        return axs.axons.MRG(
            diameter=diameter,
            nodes=nodes,
            length=args.nerve_length_um * axs.um,
            x_shift=row.x_shift_um * axs.um,
        )
    if row.kind == "rattay":
        compartments = int(args.unmyelinated_compartments)
        if compartments <= 0:
            compartments = max(3, int(args.nerve_length_um // 25))
        return axs.axons.RattayAberham(
            length=args.nerve_length_um * axs.um,
            diameter=diameter,
            compartments=compartments,
            celsius=37.0 * axs.degC,
        )
    return axs.axons.HodgkinHuxley(
        length=args.nerve_length_um * axs.um,
        diameter=diameter,
        compartments=max(3, int(args.nerve_length_um // 25)),
        celsius=6.3 * axs.degC,
    )


def nrv_activation_by_row(
    nrv_result: Any,
    nerve: Any,
    rows: Sequence[RealisticFiberRow],
    *,
    t_start_ms: float,
) -> dict[tuple[str, int], bool]:
    """Return NRV recruitment flags keyed by (fascicle_id, fiber_index)."""

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


def nrv_fascicle_by_id(nerve: Any, fascicle_id: str) -> Any:
    """Return an NRV fascicle by string or integer id."""

    if fascicle_id in nerve.fascicles:
        return nerve.fascicles[fascicle_id]
    try:
        return nerve.fascicles[int(fascicle_id)]
    except (KeyError, ValueError) as exc:
        raise KeyError(f"Unknown NRV fascicle id {fascicle_id!r}.") from exc


def axonscope_activation_from_observer(row_result: Any, activation: axs.Activation) -> bool:
    """Decode one AxonScope observer-only activation result."""

    observations = getattr(row_result, "observations", None)
    if observations is None:
        raise RuntimeError("AxonScope observer-only row has no observations.")
    if activation.name in observations:
        return bool(np.asarray(observations[activation.name].values).ravel()[0])
    raster = observations[axs.VM_RASTER_OBSERVATION_KEY]
    names = tuple(getattr(raster, "names", ()))
    raster_index = names.index(activation.name)
    bits = np.asarray(raster.unpack(), dtype=bool)
    row_bits = bits[0, raster_index]
    mask = np.asarray(getattr(raster, "probe_mask", True), dtype=bool)
    if mask.ndim == 3:
        probe_mask = mask[0, raster_index]
    elif mask.ndim == 2:
        probe_mask = mask[raster_index]
    else:
        probe_mask = np.broadcast_to(mask, row_bits.shape[:1])
    blanking_ms = axs.units.to_ms(activation.blanking)
    if blanking_ms > 0.0:
        times_ms = (np.arange(int(raster.nt), dtype=float) + 1.0) * float(raster.dt_ms)
        row_bits = row_bits[:, times_ms >= blanking_ms]
    return bool(np.any(row_bits[np.asarray(probe_mask, dtype=bool)]))


def row_key(row: RealisticFiberRow) -> tuple[str, int]:
    """Return the stable fiber comparison key."""

    return (str(row.fascicle_id), int(row.fiber_index))


def print_activation_summary(
    console: Console,
    comparisons: Sequence[ActivationComparison],
    *,
    print_limit: int,
) -> None:
    """Print population-level and row-level activation agreement."""

    if not comparisons:
        console.print("[yellow]No fibers selected for activation comparison.[/yellow]")
        return
    matched = sum(item.matched for item in comparisons)
    nrv_count = sum(item.nrv_activated for item in comparisons)
    axonscope_count = sum(item.axonscope_activated for item in comparisons)
    console.print(
        f"\n[bold]Fiber activation agreement[/bold]: {matched}/{len(comparisons)} matched "
        f"({100.0 * matched / len(comparisons):.1f}%). "
        f"NRV active={nrv_count}, AxonScope active={axonscope_count}."
    )

    table = Table(title="Fiber-by-fiber activation")
    for column in ("fasc", "fiber", "kind", "diam [um]", "NRV", "AxonScope", "match"):
        table.add_column(column, justify="right")
    limit = len(comparisons) if int(print_limit) <= 0 else min(int(print_limit), len(comparisons))
    for item in comparisons[:limit]:
        row = item.row
        table.add_row(
            row.fascicle_id,
            str(row.fiber_index),
            row.kind,
            f"{row.diameter_um:.2f}",
            "yes" if item.nrv_activated else "no",
            "yes" if item.axonscope_activated else "no",
            "yes" if item.matched else "NO",
            style=None if item.matched else "bold red",
        )
    console.print(table)
    if limit < len(comparisons):
        console.print(f"[dim]Showing {limit}/{len(comparisons)} fibers. Use --print-fiber-limit 0 for all.[/dim]")


def plot_activation_comparison(
    ax: Any,
    comparisons: Sequence[ActivationComparison],
) -> None:
    """Overlay activation agreement on the existing fiber map."""

    if not comparisons:
        return
    y = np.asarray([item.row.y_um for item in comparisons], dtype=float)
    z = np.asarray([item.row.z_um for item in comparisons], dtype=float)
    both = np.asarray(
        [item.nrv_activated and item.axonscope_activated for item in comparisons],
        dtype=bool,
    )
    mismatch = np.asarray([not item.matched for item in comparisons], dtype=bool)
    ax.scatter(
        y[both],
        z[both],
        s=95,
        facecolors="none",
        edgecolors="#2ca02c",
        linewidth=1.4,
        label="active both",
    )
    ax.scatter(
        y[mismatch],
        z[mismatch],
        s=95,
        marker="x",
        color="#d62728",
        linewidth=1.6,
        label="activation mismatch",
    )
    ax.set_title("NRV fibers -> AxonScope rows, activation overlay")
    ax.legend(loc="upper right")


def print_geometry_summary(
    console: Console,
    nerve: Any,
    rows: Sequence[RealisticFiberRow],
    comparisons: Sequence[LayoutComparison],
) -> None:
    """Print a compact geometry handoff report."""

    console.print(
        f"\n[bold]NRV realistic nerve[/bold]: {len(nerve.fascicles)} fascicles, "
        f"{len(rows)} selected fibers, {len(comparisons)} MRG comparisons."
    )
    table = Table(title="NRV node_shift -> AxonScope layout x_shift")
    for column in (
        "fasc",
        "fiber",
        "diam [um]",
        "node_shift",
        "x_shift [um]",
        "spacing [um]",
        "first node [um]",
        "max error [um]",
    ):
        table.add_column(column, justify="right")
    for comparison in comparisons[:12]:
        row = comparison.row
        error = np.max(np.abs(comparison.axonscope_nodes_um - comparison.expected_nodes_um))
        table.add_row(
            row.fascicle_id,
            str(row.fiber_index),
            f"{row.diameter_um:.2f}",
            f"{row.node_shift:.3f}",
            f"{row.x_shift_um:.2f}",
            f"{comparison.node_spacing_um:.2f}",
            f"{comparison.axonscope_nodes_um[0]:.2f}",
            f"{error:.3g}",
        )
    console.print(table)


def plot_fiber_map(
    ax: Any,
    nerve_contour: np.ndarray,
    fascicle_contours: Sequence[np.ndarray],
    rows: Sequence[RealisticFiberRow],
    life_setup: LifeElectrodeSetup,
) -> None:
    """Plot NRV contours, selected fibers, and the LIFE electrode."""

    ax.clear()
    ax.plot(nerve_contour[:, 0], nerve_contour[:, 1], color="black", linewidth=1.2)
    for contour in fascicle_contours:
        ax.plot(contour[:, 0], contour[:, 1], color="0.35", linewidth=0.9)

    y = np.asarray([row.y_um for row in rows], dtype=float)
    z = np.asarray([row.z_um for row in rows], dtype=float)
    is_mrg = np.asarray([row.kind == "mrg" for row in rows], dtype=bool)
    ax.scatter(y[~is_mrg], z[~is_mrg], s=20, color="#4c78a8", label="unmyelinated")
    scatter = ax.scatter(
        y[is_mrg],
        z[is_mrg],
        c=np.asarray([row.x_shift_um for row in rows if row.kind == "mrg"], dtype=float),
        s=42,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.35,
        label="MRG",
    )
    if np.any(is_mrg):
        plt.colorbar(scatter, ax=ax, label="MRG x_shift [um]")
    ax.add_patch(
        plt.Circle(
            (life_setup.y_um, life_setup.z_um),
            life_setup.diameter_um / 2.0,
            color="#eeca3b",
            alpha=0.75,
            label="LIFE",
        )
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("NRV fascicles -> AxonScope rows")
    ax.set_xlabel("y [um]")
    ax.set_ylabel("z [um]")
    ax.legend(loc="upper right")


def plot_node_alignment(
    ax: Any,
    comparisons: Sequence[LayoutComparison],
    *,
    max_lanes: int,
) -> None:
    """Plot expected NRV-phased nodes and AxonScope layout nodes."""

    ax.clear()
    for lane, comparison in enumerate(comparisons[:max_lanes]):
        node_count = min(8, comparison.axonscope_nodes_um.size)
        expected = comparison.expected_nodes_um[:node_count]
        axonscope_nodes = comparison.axonscope_nodes_um[:node_count]
        y_expected = np.full(node_count, lane - 0.12)
        y_axonscope = np.full(node_count, lane + 0.12)
        ax.scatter(expected, y_expected, marker="|", s=160, color="#f58518")
        ax.scatter(axonscope_nodes, y_axonscope, marker="o", s=28, color="#54a24b")
        ax.text(
            float(axonscope_nodes[0]),
            lane + 0.32,
            f"f{comparison.row.fascicle_id}/#{comparison.row.fiber_index}",
            fontsize=8,
        )
    ax.set_title("MRG node alignment")
    ax.set_xlabel("x [um]")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.25)
    ax.scatter([], [], marker="|", s=160, color="#f58518", label="NRV expected")
    ax.scatter([], [], marker="o", s=28, color="#54a24b", label="AxonScope layout")
    ax.legend(loc="upper right")


def compare_single_fiber_simulations(
    console: Console,
    rows: Sequence[RealisticFiberRow],
    args: argparse.Namespace,
    *,
    life_setup: LifeElectrodeSetup,
) -> list[SimulationComparison]:
    """Run matched NRV and AxonScope simulations with the NRV LIFE electrode."""

    import nrv

    reports: list[SimulationComparison] = []
    for row in rows:
        if row.kind != "mrg":
            continue
        diameter = row.diameter_um * axs.um
        nodes = max(
            2,
            axs.axons.mrg_like_nodes_from_length(
                diameter,
                args.nerve_length_um * axs.um,
                x_shift=row.x_shift_um * axs.um,
            ),
        )
        axon = axs.axons.MRG(
            diameter=diameter,
            nodes=nodes,
            length=args.nerve_length_um * axs.um,
            x_shift=row.x_shift_um * axs.um,
        )
        positions_um = axon.layout.position_values(unit=axs.um)
        stimulus = axs.Stimulus.pulse(
            start=args.stimulus_start_ms * axs.ms,
            duration=args.pulse_duration_ms * axs.ms,
            amplitude=-args.stimulus_current_uA * axs.uA,
        )
        simulation = axs.AxonInstance(axon)
        simulation.add_extracellular_stimulation(
            stimulation=axonscope_stimulation_from_nrv_life(
                life_setup,
                positions_um=positions_um,
                row=row,
                stimulus=stimulus,
            )
        )

        console.print(
            f"[dim]simulate fiber fasc={row.fascicle_id} id={row.fiber_index} "
            f"d={row.diameter_um:.2f} um nodes={nodes} electrode=LIFE[/dim]"
        )
        result_as = axs.simulate(
            simulation,
            duration=args.duration_ms * axs.ms,
            dt=args.dt_ms * axs.ms,
        ).single
        t_as = result_as.time_values(unit=axs.ms)
        vm_as = result_as.voltage_values(unit=axs.mV)[:, np.asarray(axon.node_indices, dtype=int)]
        x_nodes_as = axon.node_position_values(unit=axs.um)

        axon_nrv = nrv.myelinated(
            row.y_um,
            row.z_um,
            row.diameter_um,
            args.nerve_length_um,
            model="MRG",
            dt=args.dt_ms,
            node_shift=row.node_shift,
            Nseg_per_sec=1,
            rec="nodes",
            T=37.0,
            v_init=-80.0,
        )
        axon_nrv.attach_extracellular_stimulation(life_setup.extra_stim)
        result_nrv = axon_nrv.simulate(t_sim=args.duration_ms)

        t_nrv = np.asarray(result_nrv["t"], dtype=float).ravel()
        x_nodes_nrv = np.asarray(result_nrv["x_rec"], dtype=float).ravel()
        vm_nrv = normalize_space_time_matrix(result_nrv["V_mem"], t_nrv, x_nodes_nrv)
        vm_nrv_i = interpolate_rows(vm_nrv, t_nrv, t_as)
        n_nodes = min(vm_as.shape[1], vm_nrv_i.shape[0], x_nodes_as.size)
        vm_as_nodes = vm_as[:, :n_nodes].T
        vm_nrv_nodes = vm_nrv_i[:n_nodes]
        diff = vm_as_nodes - vm_nrv_nodes
        center = int(n_nodes // 2)
        as_peak_index = int(np.argmax(vm_as_nodes[center]))
        nrv_peak_index = int(np.argmax(vm_nrv_nodes[center]))
        center_peak_lag_ms = float(t_as[as_peak_index] - t_as[nrv_peak_index])
        vm_nrv_aligned = interpolate_rows_with_nan(
            vm_nrv_nodes,
            t_as,
            t_as - center_peak_lag_ms,
        )
        aligned_diff = vm_as_nodes - vm_nrv_aligned
        reports.append(
            SimulationComparison(
                row=row,
                t_ms=t_as,
                x_nodes_um=x_nodes_as[:n_nodes],
                vm_axonscope_mV=vm_as_nodes,
                vm_nrv_mV=vm_nrv_nodes,
                rmse_mV=finite_rmse(diff),
                max_abs_mV=float(np.max(np.abs(diff))),
                aligned_rmse_mV=finite_rmse(aligned_diff),
                aligned_max_abs_mV=finite_max_abs(aligned_diff),
                peak_delta_mV=float(np.max(vm_as_nodes) - np.max(vm_nrv_nodes)),
                center_peak_delta_mV=float(
                    vm_as_nodes[center, as_peak_index] - vm_nrv_nodes[center, nrv_peak_index]
                ),
                center_peak_lag_ms=center_peak_lag_ms,
            )
        )
    return reports


def axonscope_stimulation_from_nrv_life(
    life_setup: LifeElectrodeSetup,
    *,
    positions_um: np.ndarray,
    row: RealisticFiberRow,
    stimulus: axs.Stimulus,
) -> axs.ExtracellularStimulation:
    """Sample NRV's LIFE/FEM footprint and wrap it in AxonScope stimulation."""

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

    footprint = axs.ExtracellularFootprint.shared(
        values=values_mV_per_mA,
        positions=np.asarray(positions_um, dtype=float) * axs.um,
        voltage_unit=axs.mV,
        current_unit=axs.mA,
        source_id="nrv_life_fem",
        reference="NRV FEM LIFE footprint sampled on AxonScope intrinsic positions",
        metadata={
            "source": "nrv.FEM_stimulation/LIFE_electrode",
            "life_diameter_um": life_setup.diameter_um,
            "life_length_um": life_setup.length_um,
            "life_x_offset_um": life_setup.x_offset_um,
            "life_y_um": life_setup.y_um,
            "life_z_um": life_setup.z_um,
            "nrv_footprint_unit": "mV/mA",
        },
    )
    drive = axs.ExtracellularDrive(
        id=axs.DriveId("nrv_life"),
        footprint=footprint,
        stimulus=stimulus,
        metadata={"source": "nrv_life_fem"},
    )
    return axs.ExtracellularStimulation([drive])


def nrv_row_id(row: RealisticFiberRow) -> int:
    """Return a stable integer ID for single-fiber NRV footprint sampling."""

    try:
        fascicle_id = int(row.fascicle_id)
    except ValueError:
        fascicle_id = 0
    return fascicle_id * 1_000_000 + int(row.fiber_index)


def normalize_space_time_matrix(values: Any, t_ms: np.ndarray, x_um: np.ndarray) -> np.ndarray:
    """Return a matrix shaped as space x time."""

    matrix = np.asarray(values, dtype=float)
    if matrix.shape == (x_um.size, t_ms.size):
        return matrix
    if matrix.shape == (t_ms.size, x_um.size):
        return matrix.T
    if matrix.shape[0] == x_um.size:
        return matrix
    if matrix.shape[1] == x_um.size:
        return matrix.T
    raise ValueError(f"Cannot align matrix shape {matrix.shape} with x={x_um.size}, t={t_ms.size}.")


def interpolate_rows(values: np.ndarray, t_src_ms: np.ndarray, t_dst_ms: np.ndarray) -> np.ndarray:
    """Interpolate a space x time matrix onto another time axis."""

    out = np.empty((values.shape[0], t_dst_ms.size), dtype=float)
    for index in range(values.shape[0]):
        out[index] = np.interp(t_dst_ms, t_src_ms, values[index])
    return out


def interpolate_rows_with_nan(
    values: np.ndarray,
    t_src_ms: np.ndarray,
    t_dst_ms: np.ndarray,
) -> np.ndarray:
    """Interpolate rows and mark samples outside the source time range as NaN."""

    out = np.full((values.shape[0], t_dst_ms.size), np.nan, dtype=float)
    valid = (t_dst_ms >= float(t_src_ms[0])) & (t_dst_ms <= float(t_src_ms[-1]))
    for index in range(values.shape[0]):
        out[index, valid] = np.interp(t_dst_ms[valid], t_src_ms, values[index])
    return out


def finite_rmse(values: np.ndarray) -> float:
    """Return RMSE over finite entries only."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


def finite_max_abs(values: np.ndarray) -> float:
    """Return max absolute value over finite entries only."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(np.abs(finite)))


def print_simulation_summary(
    console: Console,
    reports: Sequence[SimulationComparison],
) -> None:
    """Print simulation comparison metrics."""

    if not reports:
        console.print("[yellow]No MRG rows selected for simulation comparison.[/yellow]")
        return
    table = Table(title="Single-fiber simulation: NRV versus AxonScope")
    for column in (
        "fasc",
        "fiber",
        "diam [um]",
        "nodes",
        "raw RMSE [mV]",
        "aligned RMSE [mV]",
        "raw max |err| [mV]",
        "center lag [us]",
        "center peak dV [mV]",
    ):
        table.add_column(column, justify="right")
    for report in reports:
        row = report.row
        table.add_row(
            row.fascicle_id,
            str(row.fiber_index),
            f"{row.diameter_um:.2f}",
            str(report.x_nodes_um.size),
            f"{report.rmse_mV:.3f}",
            f"{report.aligned_rmse_mV:.3f}",
            f"{report.max_abs_mV:.3f}",
            f"{report.center_peak_lag_ms * 1000.0:.1f}",
            f"{report.center_peak_delta_mV:.3f}",
        )
    console.print(table)


def plot_simulation_comparisons(reports: Sequence[SimulationComparison]) -> None:
    """Plot trace, heatmaps, and difference for the first simulation report."""

    if not reports:
        return
    report = reports[0]
    center = int(report.x_nodes_um.size // 2)
    diff = report.vm_axonscope_mV - report.vm_nrv_mV
    as_peak_t = float(report.t_ms[int(np.argmax(report.vm_axonscope_mV[center]))])
    nrv_peak_t = float(report.t_ms[int(np.argmax(report.vm_nrv_mV[center]))])
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.0), constrained_layout=True)
    axes[0, 0].plot(report.t_ms, report.vm_axonscope_mV[center], label="AxonScope")
    axes[0, 0].plot(report.t_ms, report.vm_nrv_mV[center], "--", label="NRV")
    axes[0, 0].axvline(as_peak_t, color="C0", alpha=0.35, linewidth=1.0)
    axes[0, 0].axvline(nrv_peak_t, color="C1", alpha=0.35, linewidth=1.0)
    axes[0, 0].set_title(
        f"Center node trace, x={report.x_nodes_um[center]:.1f} um, "
        f"lag={report.center_peak_lag_ms * 1000.0:.1f} us"
    )
    axes[0, 0].set_xlabel("time [ms]")
    axes[0, 0].set_ylabel("Vm [mV]")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend()

    extent = [
        float(report.t_ms[0]),
        float(report.t_ms[-1]),
        0,
        int(report.x_nodes_um.size - 1),
    ]
    im_as = axes[0, 1].imshow(
        report.vm_axonscope_mV,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    axes[0, 1].set_title("AxonScope Vm at nodes")
    fig.colorbar(im_as, ax=axes[0, 1], label="Vm [mV]")

    im_nrv = axes[1, 0].imshow(
        report.vm_nrv_mV,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    axes[1, 0].set_title("NRV Vm at nodes")
    axes[1, 0].set_xlabel("time [ms]")
    axes[1, 0].set_ylabel("node index")
    fig.colorbar(im_nrv, ax=axes[1, 0], label="Vm [mV]")

    vmax = max(float(np.max(np.abs(diff))), 1e-9)
    im_diff = axes[1, 1].imshow(
        diff,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[1, 1].set_title("AxonScope - NRV")
    axes[1, 1].set_xlabel("time [ms]")
    fig.colorbar(im_diff, ax=axes[1, 1], label="delta Vm [mV]")

    row = report.row
    fig.suptitle(
        f"Realistic row comparison: fasc={row.fascicle_id}, fiber={row.fiber_index}, "
        f"d={row.diameter_um:.2f} um"
    )


if __name__ == "__main__":
    main()
