"""Map optional NRV fiber geometry into an AxonScope pool.

Run:
    python examples/with_nrv/01_nrv_pool_geometry.py --fibers 12 --include-mrg

With NRV installed, NRV can be used only to generate the fiber table:
    python examples/with_nrv/01_nrv_pool_geometry.py --source nrv --fibers 24

The simulation stays on the public AxonScope API:
`AxonInstance rows -> dispatcher plan -> simulate_pool -> AxonSimulationResult`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs
from axonscope.integrations.nrv import (
    FiberKind,
    fiber_kind_from_nrv,
    nrv_node_shift_to_x_shift_um,
)


@dataclass(frozen=True)
class FiberRow:
    """One positioned fiber before it is turned into an `AxonInstance`."""

    kind: FiberKind
    diameter_um: float
    y_um: float
    z_um: float
    compartments: int
    x_shift_um: float = 0.0


def main(argv: Sequence[str] | None = None) -> None:
    """Build a mixed pool, print/plot dispatch, and simulate center Vm."""

    args = parse_args(argv)
    length = args.length_um * axs.um
    rows = build_pool_rows(
        source=args.source,
        fibers=args.fibers,
        radius=args.radius_um * axs.um,
        include_mrg=args.include_mrg,
    )
    electrode = make_point_source_electrode(length=length)
    simulations = tuple(
        build_simulation(
            row,
            length=length,
            mrg_nodes=args.mrg_nodes,
            electrode=electrode,
            index=index,
        )
        for index, row in enumerate(rows)
    )

    plan = axs.dispatcher.build_dispatch_plan(simulations)
    axs.dispatcher.print_dispatch_plan(plan)

    results = axs.simulate_pool(
        simulations,
        duration=args.duration_ms * axs.ms,
        dt=args.dt_ms * axs.ms,
        batch_options=(
            None
            if args.time_chunk_steps is None
            else axs.BatchOptions.full(time_chunk_steps=args.time_chunk_steps)
        ),
        recording=axs.Recording.center(axs.signals.Vm),
        progress=args.progress,
    )
    print_summary(rows, results)
    if not args.no_plot:
        plot_summary(rows, results, plan)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line options for the example."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "nrv"), default="synthetic")
    parser.add_argument("--fibers", type=int, default=12)
    parser.add_argument("--radius-um", type=float, default=120.0)
    parser.add_argument("--length-um", type=float, default=500.0)
    parser.add_argument("--duration-ms", type=float, default=0.5)
    parser.add_argument("--dt-ms", type=float, default=0.02)
    parser.add_argument("--include-mrg", action="store_true")
    parser.add_argument("--mrg-nodes", type=int, default=3)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--time-chunk-steps", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def make_point_source_electrode(*, length: Any) -> axs.analytical.PointSourceElectrode:
    """Create one global analytical source to localize per fiber."""

    electrode = axs.analytical.PointSourceElectrode(
        x=length / 2.0,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
    )
    stimulus = axs.Stimulus.pulse(
        start=0.10 * axs.ms,
        duration=0.08 * axs.ms,
        amplitude=20.0 * axs.uA,
    )
    return electrode.with_stimulus(stimulus)


def build_pool_rows(
    *,
    source: Literal["synthetic", "nrv"],
    fibers: int,
    radius: Any,
    include_mrg: bool,
) -> list[FiberRow]:
    """Build pool rows from NRV when available, otherwise synthetically."""

    if fibers < 1:
        raise ValueError("--fibers must be >= 1.")
    if source == "nrv":
        try:
            return build_pool_rows_from_nrv(
                fibers=fibers,
                radius=radius,
                include_mrg=include_mrg,
            )
        except Exception as exc:
            print(f"NRV pool generation unavailable ({exc}); using synthetic fallback.")
    return build_synthetic_pool_rows(
        fibers=fibers,
        radius=radius,
        include_mrg=include_mrg,
    )


def build_pool_rows_from_nrv(
    *,
    fibers: int,
    radius: Any,
    include_mrg: bool,
) -> list[FiberRow]:
    """Use NRV's pool placement helper and map its table to `FiberRow`."""

    from nrv.nmod._axon_pool import axon_pool

    pool = axon_pool()
    pool.set_geometry(center=(0, 0), radius=axs.units.to_um(radius))
    pool.create_pool_from_stat(n_ax=fibers)
    pool.place_pool(delta=2)

    table = pool.axon_pop
    iter_rows = table.iterrows() if hasattr(table, "iterrows") else enumerate(table)
    rows: list[FiberRow] = []
    for index, row in iter_rows:
        get = row.get if hasattr(row, "get") else lambda key, default=None: row[key]
        nrv_type = int(float(get("types", index % 2)))
        diameter_um = float(get("diameters", 1.0 if nrv_type == 0 else 6.0))
        kind = fiber_kind_from_nrv(nrv_type, include_mrg=include_mrg)
        node_shift = float(get("node_shift", 0.0))
        rows.append(
            FiberRow(
                kind=kind,
                diameter_um=diameter_um,
                y_um=float(get("y", 0.0)),
                z_um=float(get("z", 0.0)),
                compartments=compartments_for(diameter_um, kind),
                x_shift_um=nrv_node_shift_to_x_shift_um(
                    node_shift,
                    diameter_um,
                    kind=kind,
                ),
            )
        )
    return rows


def build_synthetic_pool_rows(
    *,
    fibers: int,
    radius: Any,
    include_mrg: bool,
) -> list[FiberRow]:
    """Create a deterministic mixed pool for machines without NRV."""

    rng = np.random.default_rng(1234)
    radius_um = axs.units.to_um(radius)
    angles = np.linspace(0.0, 2.0 * np.pi, fibers, endpoint=False)
    radii = radius_um * np.sqrt(np.linspace(0.15, 0.95, fibers))
    rows: list[FiberRow] = []
    for index, (angle, local_radius) in enumerate(zip(angles, radii, strict=True)):
        if include_mrg and index % 4 == 3:
            kind: FiberKind = "mrg"
            diameter_um = float(rng.choice([6.0, 8.7, 10.0, 12.0]))
        elif index % 3 == 2:
            kind = "rattay"
            diameter_um = 0.8 if index % 2 == 0 else 1.2
        else:
            kind = "hh"
            diameter_um = 0.5 if index % 2 == 0 else 0.7
        rows.append(
            FiberRow(
                kind=kind,
                diameter_um=diameter_um,
                y_um=float(local_radius * np.cos(angle)),
                z_um=float(local_radius * np.sin(angle)),
                compartments=compartments_for(diameter_um, kind),
                x_shift_um=float(rng.uniform(0.0, 0.5)) * (
                    axs.axons.mrg_like_node_spacing(diameter_um * axs.um)
                    if kind == "mrg"
                    else 0.0
                ),
            )
        )
    return rows


def build_simulation(
    row: FiberRow,
    *,
    length: Any,
    mrg_nodes: int,
    electrode: axs.analytical.PointSourceElectrode,
    index: int,
) -> axs.AxonInstance:
    """Create one local axon simulation from a geometry-owned pool row."""

    axon = build_axon(row, length=length, mrg_nodes=mrg_nodes)
    positions = axon.layout.position_values(unit=axs.um) * axs.um
    extracellular = axs.analytical.point_source_stimulation(
        electrode,
        positions,
        sigma=0.3 * axs.S_per_m,
        axon_y=row.y_um * axs.um,
        axon_z=row.z_um * axs.um,
    )
    sim = axs.AxonInstance(axon)
    sim.add_extracellular_stimulation(stimulation=extracellular)
    sim.add_current_clamp(
        position=stimulation_position(axon, length=length),
        current=axs.Stimulus.pulse(
            start=(0.05 + 0.005 * (index % 3)) * axs.ms,
            duration=0.04 * axs.ms,
            amplitude=(0.25 + 0.05 * (index % 4)) * axs.nA,
        ),
    )
    sim.label = f"{index}:{row.kind}:{row.diameter_um:.2f}um"
    return sim


def build_axon(row: FiberRow, *, length: Any, mrg_nodes: int) -> axs.axons.Axon:
    """Instantiate the descriptive axon model for one row."""

    diameter = max(row.diameter_um, 0.2) * axs.um
    if row.kind == "mrg":
        return axs.axons.MRG(
            diameter=max(row.diameter_um, 2.0) * axs.um,
            nodes=mrg_nodes,
            x_shift=row.x_shift_um * axs.um,
            membranes=mrg_batch_membranes(),
        )
    if row.kind == "rattay":
        return axs.axons.RattayAberham(
            length=length,
            diameter=diameter,
            compartments=row.compartments,
            celsius=37.0 * axs.degC,
        )
    return axs.axons.HodgkinHuxley(
        length=length,
        diameter=diameter,
        compartments=row.compartments,
        celsius=6.3 * axs.degC,
    )


def mrg_batch_membranes() -> axs.membranes.SectionLayout:
    """Return shared MRG-like membranes for diameter-batched MRG examples."""

    return axs.membranes.SectionLayout(
        node=axs.membranes.AxNode(celsius=37.0 * axs.degC),
        mysa=axs.membranes.Passive(Rm=3_000.0 * axs.ohm_cm2, EL=-80.0 * axs.mV),
        flut=axs.membranes.Passive(Rm=15_000.0 * axs.ohm_cm2, EL=-80.0 * axs.mV),
        stin=axs.membranes.Passive(Rm=15_000.0 * axs.ohm_cm2, EL=-80.0 * axs.mV),
    )


def stimulation_position(axon: axs.axons.Axon, *, length: Any) -> Any:
    """Return a central position compatible with uniform and MRG layouts."""

    if axon.n_compartments > 0:
        x_um = np.asarray(axon.layout.position_values(unit=axs.um), dtype=float)
        return float(x_um[axon.n_compartments // 2]) * axs.um
    return length / 2.0


def compartments_for(diameter_um: float, kind: FiberKind) -> int:
    """Choose a compact compartment count that keeps the example fast."""

    if kind == "mrg":
        return 0
    if kind == "rattay":
        return 41 if diameter_um >= 1.0 else 31
    return 41 if diameter_um >= 0.6 else 31


def print_summary(rows: Sequence[FiberRow], results: axs.AxonSimulationResult) -> None:
    """Print row-level dispatch and voltage summary."""

    print("\n=== Row summary ===")
    print("idx  kind    diam[um]  y[um]    z[um]    method                         peak[mV]")
    print("---  ------  --------  -------  -------  -----------------------------  --------")
    for index, (row, result) in enumerate(zip(rows, results, strict=True)):
        diagnostics = result.diagnostics or {}
        peak_mV = float(result.peak_voltage_values(unit=axs.mV)[0])
        print(
            f"{index:>3}  {row.kind:<6}  {row.diameter_um:>8.2f}  "
            f"{row.y_um:>7.1f}  {row.z_um:>7.1f}  "
            f"{str(diagnostics.get('dispatch_method', '?')):<29}  "
            f"{peak_mV:>8.2f}"
        )


def plot_summary(
    rows: Sequence[FiberRow],
    results: axs.AxonSimulationResult,
    plan: axs.dispatcher.DispatchPlan,
) -> None:
    """Plot dispatch grouping and the transverse fiber map."""

    fig, (ax_dispatch, ax_map) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.4),
        constrained_layout=True,
    )
    axs.dispatcher.plot_dispatch_plan(plan, ax=ax_dispatch)

    y = np.asarray([row.y_um for row in rows], dtype=float)
    z = np.asarray([row.z_um for row in rows], dtype=float)
    peaks = np.asarray(
        [float(result.peak_voltage_values(unit=axs.mV)[0]) for result in results],
        dtype=float,
    )
    scatter = ax_map.scatter(y, z, c=peaks, s=70, cmap="viridis", edgecolor="black")
    for index, row in enumerate(rows):
        ax_map.text(row.y_um, row.z_um, str(index), ha="center", va="center", fontsize=7)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("y [um]")
    ax_map.set_ylabel("z [um]")
    ax_map.set_title("Pool map colored by peak center Vm")
    ax_map.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax_map, label="peak center Vm [mV]")
    plt.show()


if __name__ == "__main__":
    main()
