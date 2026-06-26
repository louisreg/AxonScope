"""Build AxonScope recruitment curves from realistic NRV fascicle geometry.

Run:
    python examples/with_nrv/01_realistic_fascicle_geometry_comparison.py

By default this example mirrors NRV's `25_test_fit_fasc.py` workflow: contours
come from NRV's bundled fascicle image, NRV fills the fascicles, a LIFE
electrode is placed in the first fascicle, and AxonScope receives the resulting
fiber table. Set `ExampleConfig.geometry_mode = "synthetic_4_fascicles"` for the
reproducible four-fascicle geometry used by Kaggle performance benchmarks. Edit
`ExampleConfig` below to change the population size, timing, or current grid;
the default sweep uses 21 amplitudes from 0 to 300 uA.
The important handoffs are:

- NRV `node_shift`, a fraction of the MRG internode spacing, becomes
  unit-bearing `MRG(..., x_shift=...)`.
- NRV's FEM/LIFE footprint is sampled on AxonScope's intrinsic positions and
  wrapped as `ExtracellularFootprint -> ExtracellularDrive`.
- AxonScope sweeps LIFE current and prints per-fascicle recruitment as the
  sweep progresses.
- NRV is run once at the middle recruitment amplitude to validate fiber-by-fiber
  activation without turning the example into a full NRV benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
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
from axonscope.timebase import simulation_step_count
from axonscope.utils.progress_reporting import format_duration


@dataclass(frozen=True)
class ExampleConfig:
    """Editable constants for the realistic NRV/AxonScope recruitment example."""

    nerve_diameter_um: float = 1_000.0
    nerve_length_um: float = 10_000.0
    geometry_mode: str = "histology"
    axons_per_fascicle: int = 100
    percent_unmyelinated: float = 0.7
    delta_trace_um: float = 10.0
    synthetic_fascicle_diameter_um: float = 250.0
    synthetic_fascicle_offset_um: float = 250.0
    include_unmyelinated: bool = True
    max_fibers: int = 0
    simulate_fibers: int = 0
    run_simulation: bool = True
    duration_ms: float = 3.0
    dt_ms: float = 0.001
    stimulus_start_ms: float = 0.1
    pulse_duration_ms: float = 0.1
    observer_time_chunk_steps: int | None = 1000
    solver_progress: bool | str = False
    recruitment_amplitudes_uA: tuple[float, ...] = tuple(
        float(value) for value in np.linspace(0.0, 300.0, 21)
    )
    nrv_validation_current_uA: float = 60.0
    activation_threshold_mV: float = 0.0
    print_fiber_limit: int = 40
    unmyelinated_compartments: int = 0
    life_diameter_um: float = 25.0
    life_length_um: float = 1_000.0
    life_fascicle_id: str = "0"
    fem_n_proc: int | None = None
    gmsh_n_core: int | None = 1
    fascicle_contour_epsilon_fraction: float = 0.002


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
class AxonScopeFiberContext:
    """One AxonScope row with its current-independent NRV LIFE footprint."""

    row: RealisticFiberRow
    axon: Any
    positions_um: np.ndarray
    footprint: axs.ExtracellularFootprint


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


def main(config: ExampleConfig | None = None) -> None:
    if config is None:
        config = ExampleConfig()
    console = Console(width=120)

    import nrv

    amplitudes_uA = np.asarray(config.recruitment_amplitudes_uA, dtype=float)
    validation_current_uA = float(config.nrv_validation_current_uA)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    nerve_contour, fascicle_contours, nerve = build_nrv_nerve_from_config(nrv, config)
    life_setup = attach_life_electrode(nrv, nerve, config)
    rows = extract_fiber_rows(nerve, include_unmyelinated=config.include_unmyelinated)
    rows = select_rows(rows, limit=config.max_fibers)
    comparisons = compare_mrg_layouts(rows, nerve_length_um=config.nerve_length_um)

    print_geometry_summary(console, nerve, rows, comparisons)
    plot_fiber_map(axes[0], nerve_contour, fascicle_contours, rows, life_setup)

    if config.run_simulation:
        simulated_rows = select_rows(rows, limit=config.simulate_fibers)
        recruitment_curve, activation_comparisons = compare_recruitment_curve(
            console,
            nerve,
            rows=simulated_rows,
            config=config,
            life_setup=life_setup,
            amplitudes_uA=amplitudes_uA,
            validation_current_uA=validation_current_uA,
        )
        print_activation_summary(
            console,
            activation_comparisons,
            validation_current_uA=validation_current_uA,
            print_limit=config.print_fiber_limit,
        )
        plot_activation_comparison(axes[0], activation_comparisons)
        plot_recruitment_curve(
            axes[1],
            recruitment_curve,
            rows=simulated_rows,
            validation_current_uA=validation_current_uA,
        )
    else:
        axes[1].axis("off")
        axes[1].set_title("Recruitment curve skipped")

    plt.show()


def build_nrv_nerve_from_config(
    nrv_module: Any,
    config: ExampleConfig,
) -> tuple[np.ndarray, list[np.ndarray], Any]:
    """Build the configured NRV nerve and matching plotting contours."""

    if config.geometry_mode == "histology":
        nerve_contour, fascicle_contours = load_nrv_contours(
            nrv_module,
            nerve_diameter_um=config.nerve_diameter_um,
            fascicle_epsilon_fraction=config.fascicle_contour_epsilon_fraction,
        )
        nerve = build_nrv_nerve(
            nrv_module,
            fascicle_contours,
            nerve_length_um=config.nerve_length_um,
            nerve_diameter_um=config.nerve_diameter_um,
            axons_per_fascicle=config.axons_per_fascicle,
            percent_unmyelinated=config.percent_unmyelinated,
            delta_trace_um=config.delta_trace_um,
        )
        return nerve_contour, fascicle_contours, nerve

    if config.geometry_mode == "synthetic_4_fascicles":
        nerve_contour = circular_contour(
            center=(0.0, 0.0),
            diameter_um=config.nerve_diameter_um,
            n_points=160,
        )
        fascicle_contours = [
            circular_contour(
                center=center,
                diameter_um=config.synthetic_fascicle_diameter_um,
                n_points=96,
            )
            for center in synthetic_fascicle_centers(config.synthetic_fascicle_offset_um)
        ]
        nerve = build_synthetic_nrv_nerve(
            nrv_module,
            nerve_length_um=config.nerve_length_um,
            nerve_diameter_um=config.nerve_diameter_um,
            fascicle_diameter_um=config.synthetic_fascicle_diameter_um,
            fascicle_offset_um=config.synthetic_fascicle_offset_um,
            axons_per_fascicle=config.axons_per_fascicle,
            percent_unmyelinated=config.percent_unmyelinated,
            delta_trace_um=config.delta_trace_um,
        )
        return nerve_contour, fascicle_contours, nerve

    raise ValueError(f"Unsupported NRV geometry mode: {config.geometry_mode!r}")


def load_nrv_contours(
    nrv_module: Any,
    *,
    nerve_diameter_um: float,
    fascicle_epsilon_fraction: float = 0.002,
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
            points = simplify_cv2_contour(
                cv2,
                contour,
                epsilon_fraction=fascicle_epsilon_fraction,
            )
            fascicle_points.append((points - center_pix) * scale_xy)

    return nerve_points, fascicle_points


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
        geometry = nrv_module.create_cshape(vertices=np.asarray(points, dtype=float))
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


def build_synthetic_nrv_nerve(
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
    """Create one NRV nerve with four simple circular fascicles."""

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
            n_ax=axons_per_fascicle,
            percent_unmyel=percent_unmyelinated,
            delta_trace=delta_trace_um,
            with_node_shift=True,
        )
    return nerve


def synthetic_fascicle_centers(offset_um: float) -> tuple[tuple[float, float], ...]:
    """Return four fascicle centers inside the synthetic nerve cross-section."""

    offset = float(offset_um)
    return (
        (offset, 0.0),
        (0.0, offset),
        (-offset, 0.0),
        (0.0, -offset),
    )


def circular_contour(
    *,
    center: tuple[float, float],
    diameter_um: float,
    n_points: int,
) -> np.ndarray:
    """Return y/z contour points for plotting a circular NRV geometry."""

    theta = np.linspace(0.0, 2.0 * np.pi, int(n_points), endpoint=False)
    radius = float(diameter_um) / 2.0
    return np.column_stack(
        (
            float(center[0]) + radius * np.cos(theta),
            float(center[1]) + radius * np.sin(theta),
        )
    )


def nrv_numeric(value: float) -> int | float:
    """Preserve NRV example integer parameters when configured as floats."""

    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def attach_life_electrode(
    nrv_module: Any,
    nerve: Any,
    config: ExampleConfig,
) -> LifeElectrodeSetup:
    """Attach the NRV-example LIFE electrode and return the configured context."""

    fascicle_key: Any = config.life_fascicle_id
    if fascicle_key not in nerve.fascicles:
        fascicle_key = int(config.life_fascicle_id)
    fascicle = nerve.fascicles[fascicle_key]
    life_y_um, life_z_um = fascicle.center
    life_x_offset_um = (config.nerve_length_um - config.life_length_um) / 2.0

    extra_stim = nrv_module.FEM_stimulation(
        endo_mat="endoneurium_ranck",
        peri_mat="perineurium",
        epi_mat="epineurium",
        ext_mat="saline",
        n_proc=config.fem_n_proc,
    )
    electrode = nrv_module.LIFE_electrode(
        "LIFE_2",
        config.life_diameter_um,
        config.life_length_um,
        life_x_offset_um,
        life_y_um,
        life_z_um,
    )
    pulse_stim = nrv_module.stimulus()
    pulse_stim.pulse(
        config.stimulus_start_ms,
        -config.nrv_validation_current_uA,
        config.pulse_duration_ms,
    )
    extra_stim.add_electrode(electrode, pulse_stim)
    nerve.attach_extracellular_stimulation(extra_stim)
    if config.fem_n_proc is not None:
        nerve.extra_stim.set_n_proc(config.fem_n_proc)
    if config.gmsh_n_core is not None:
        nerve.extra_stim.model.mesh.n_core = int(config.gmsh_n_core)

    return LifeElectrodeSetup(
        extra_stim=nerve.extra_stim,
        diameter_um=float(config.life_diameter_um),
        length_um=float(config.life_length_um),
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


def compare_recruitment_curve(
    console: Console,
    nerve: Any,
    *,
    rows: Sequence[RealisticFiberRow],
    config: ExampleConfig,
    life_setup: LifeElectrodeSetup,
    amplitudes_uA: np.ndarray,
    validation_current_uA: float,
) -> tuple[axs.protocols.RecruitmentCurve, list[ActivationComparison]]:
    """Build an AxonScope recruitment curve and validate one amplitude with NRV."""

    row_list = list(rows)
    if not row_list:
        empty_curve = axs.protocols.RecruitmentCurve(
            amplitudes_uA=np.asarray(amplitudes_uA, dtype=float),
            activated=np.zeros((len(amplitudes_uA), 0), dtype=bool),
        )
        return empty_curve, []

    console.print(
        f"[bold]AxonScope recruitment[/bold]: {len(row_list)} rows, "
        f"{len(amplitudes_uA)} amplitudes. NRV validates "
        f"{validation_current_uA:.2f} uA only."
    )
    print_recruitment_workload_summary(
        console,
        row_list,
        amplitudes_uA=amplitudes_uA,
        config=config,
    )
    console.print("[dim]Sampling NRV LIFE/FEM footprints once per AxonScope row.[/dim]")
    contexts, footprint_timings = build_axonscope_contexts(
        row_list,
        config=config,
        life_setup=life_setup,
    )
    print_footprint_sampling_summary(console, contexts, timings=footprint_timings)
    pool = tuple(
        build_axonscope_simulation_from_context(
            context,
            config=config,
            current_uA=0.0,
        )
        for context in contexts
    )
    criterion = axs.analysis.ActivationCriterion(
        threshold=config.activation_threshold_mV * axs.mV,
        blanking=config.stimulus_start_ms * axs.ms,
        target=axs.positions.ALL,
    )

    def update_life_current(
        simulation: axs.AxonInstance,
        current_magnitude: Any,
    ) -> None:
        stimulation = simulation.extracellular_stimulation
        if stimulation is None:
            raise ValueError("simulation has no extracellular stimulation to update.")
        drive = stimulation.drives[0]
        updated = stimulation.replace_drive(
            drive.id,
            stimulus=axs.Stimulus.pulse(
                start=config.stimulus_start_ms * axs.ms,
                duration=config.pulse_duration_ms * axs.ms,
                amplitude=-current_magnitude,
            ),
        )
        simulation.add_extracellular_stimulation(stimulation=updated, replace=True)

    axonscope_start_s = time.perf_counter()
    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update_life_current,
        amplitudes=np.asarray(amplitudes_uA, dtype=float) * axs.uA,
        duration=config.duration_ms * axs.ms,
        dt=config.dt_ms * axs.ms,
        criterion=criterion,
        recording=axs.Recording.none(),
        batch_options=axs.BatchOptions.none(
            time_chunk_steps=config.observer_time_chunk_steps
        ),
        progress=True,
        solver_progress=config.solver_progress,
    )
    axonscope_sweep_s = time.perf_counter() - axonscope_start_s

    console.print(
        f"[bold]NRV validation[/bold]: simulating one nerve at "
        f"{validation_current_uA:.2f} uA with postproc_script='is_recruited'."
    )
    nrv_start_s = time.perf_counter()
    nrv_result = nerve.simulate(
        t_sim=config.duration_ms,
        postproc_script="is_recruited",
        dt=config.dt_ms,
        unmyelinated_nseg=max(3, int(config.nerve_length_um // 25)),
        myelinated_nseg_per_sec=3,
    )
    nrv_validation_s = time.perf_counter() - nrv_start_s
    print_simulation_timing_comparison(
        console,
        amplitude_count=len(amplitudes_uA),
        axonscope_sweep_s=axonscope_sweep_s,
        footprint_handoff_s=footprint_timings["total_s"],
        nrv_validation_s=nrv_validation_s,
    )
    nrv_activated = nrv_activation_by_row(
        nrv_result,
        nerve,
        row_list,
        t_start_ms=config.stimulus_start_ms,
    )
    validation_index = int(
        np.argmin(np.abs(np.asarray(curve.amplitudes_uA, dtype=float) - validation_current_uA))
    )
    axonscope_validation = curve.activated[validation_index]
    return curve, [
        ActivationComparison(
            row=row,
            nrv_activated=bool(nrv_activated.get(row_key(row), False)),
            axonscope_activated=bool(axonscope_active),
        )
        for row, axonscope_active in zip(row_list, axonscope_validation, strict=True)
    ]


def print_recruitment_workload_summary(
    console: Console,
    rows: Sequence[RealisticFiberRow],
    *,
    amplitudes_uA: np.ndarray,
    config: ExampleConfig,
) -> None:
    """Print the expanded AxonScope recruitment workload before solving."""

    row_list = list(rows)
    amplitude_count = int(len(amplitudes_uA))
    mrg_rows = sum(row.kind == "mrg" for row in row_list)
    single_rows = len(row_list) - mrg_rows
    nt = simulation_step_count(config.duration_ms, config.dt_ms)
    chunk_steps = config.observer_time_chunk_steps
    if chunk_steps is None or int(chunk_steps) >= nt:
        chunk_count = 1
        chunk_label = "unchunked"
    else:
        chunk_count = int(np.ceil(nt / int(chunk_steps)))
        chunk_label = f"{int(chunk_steps)} steps"
    solver_progress = "on" if bool(config.solver_progress) else "off"

    table = Table(title="AxonScope recruitment workload")
    table.add_column("item")
    table.add_column("value", justify="right")
    table.add_row("fibers", str(len(row_list)))
    table.add_row("amplitudes", str(amplitude_count))
    table.add_row("rows per solver step", str(len(row_list)))
    table.add_row("sequential amplitude steps", str(amplitude_count))
    table.add_row("total row evaluations", str(len(row_list) * amplitude_count))
    table.add_row("MRG rows per step", str(mrg_rows))
    table.add_row("single-cable rows per step", str(single_rows))
    table.add_row("time steps", str(nt))
    table.add_row("observer chunk", chunk_label)
    table.add_row("chunks per group", str(chunk_count))
    table.add_row("solver progress", solver_progress)
    console.print(table)
    if not config.solver_progress:
        console.print(
            "[dim]Set ExampleConfig.solver_progress = 'plain' to debug dispatch, "
            "compilation, and chunk progress.[/dim]"
        )


def build_axonscope_context(
    row: RealisticFiberRow,
    *,
    config: ExampleConfig,
    life_setup: LifeElectrodeSetup,
) -> AxonScopeFiberContext:
    """Build one AxonScope row and sample its NRV LIFE footprint once."""

    axon = build_axonscope_axon(row, config=config)
    positions_um = axon.layout.position_values(unit=axs.um)
    return AxonScopeFiberContext(
        row=row,
        axon=axon,
        positions_um=positions_um,
        footprint=nrv_life_footprint(
            life_setup,
            positions_um=positions_um,
            row=row,
        ),
    )


def build_axonscope_contexts(
    rows: Sequence[RealisticFiberRow],
    *,
    config: ExampleConfig,
    life_setup: LifeElectrodeSetup,
) -> tuple[list[AxonScopeFiberContext], dict[str, float]]:
    """Build AxonScope rows while separating first FEM solve from cached sampling."""

    contexts: list[AxonScopeFiberContext] = []
    first_elapsed_s = 0.0
    cached_elapsed_s = 0.0
    total_start_s = time.perf_counter()
    for index, row in enumerate(rows):
        start_s = time.perf_counter()
        contexts.append(
            build_axonscope_context(row, config=config, life_setup=life_setup)
        )
        elapsed_s = time.perf_counter() - start_s
        if index == 0:
            first_elapsed_s = elapsed_s
        else:
            cached_elapsed_s += elapsed_s
    return contexts, {
        "first_s": first_elapsed_s,
        "cached_s": cached_elapsed_s,
        "total_s": time.perf_counter() - total_start_s,
    }


def print_footprint_sampling_summary(
    console: Console,
    contexts: Sequence[AxonScopeFiberContext],
    *,
    timings: dict[str, float],
) -> None:
    """Print the cost of the one-time NRV footprint handoff."""

    footprint_bytes = sum(context.footprint.values_V_per_A.nbytes for context in contexts)
    table = Table(title="NRV LIFE/FEM footprint sampling")
    table.add_column("item")
    table.add_column("value", justify="right")
    table.add_row("footprints", str(len(contexts)))
    table.add_row("first footprint / FEM solve", format_duration(timings["first_s"]))
    table.add_row("cached footprint sampling", format_duration(timings["cached_s"]))
    table.add_row("total", format_duration(timings["total_s"]))
    table.add_row("stored footprint arrays", f"{footprint_bytes / 1024.0:.1f} KiB")
    console.print(table)


def print_simulation_timing_comparison(
    console: Console,
    *,
    amplitude_count: int,
    axonscope_sweep_s: float,
    footprint_handoff_s: float,
    nrv_validation_s: float,
) -> None:
    """Print AxonScope sweep timing next to a one-amplitude NRV estimate."""

    n_amplitudes = max(int(amplitude_count), 1)
    nrv_estimated_s = float(nrv_validation_s) * n_amplitudes
    axonscope_with_handoff_s = float(axonscope_sweep_s) + float(footprint_handoff_s)
    table = Table(title="AxonScope versus NRV timing estimate")
    table.add_column("item")
    table.add_column("time", justify="right")
    table.add_column("note", justify="right")
    table.add_row(
        "AxonScope sweep",
        format_duration(axonscope_sweep_s),
        f"{n_amplitudes} amplitudes",
    )
    table.add_row(
        "AxonScope + footprint handoff",
        format_duration(axonscope_with_handoff_s),
        "one FEM/LIFE handoff",
    )
    table.add_row(
        "NRV validation simulation",
        format_duration(nrv_validation_s),
        "one amplitude",
    )
    table.add_row(
        "NRV full sweep estimate",
        format_duration(nrv_estimated_s),
        f"{n_amplitudes} x validation",
    )
    table.add_row(
        "NRV estimate / AxonScope sweep",
        _format_ratio(nrv_estimated_s, axonscope_sweep_s),
        "higher favors AxonScope",
    )
    table.add_row(
        "NRV estimate / AxonScope + handoff",
        _format_ratio(nrv_estimated_s, axonscope_with_handoff_s),
        "includes footprint sampling",
    )
    console.print(table)


def build_axonscope_simulation_from_context(
    context: AxonScopeFiberContext,
    *,
    config: ExampleConfig,
    current_uA: float,
) -> axs.AxonInstance:
    """Build one AxonScope simulation row by attaching a new temporal stimulus."""

    stimulus = axs.Stimulus.pulse(
        start=config.stimulus_start_ms * axs.ms,
        duration=config.pulse_duration_ms * axs.ms,
        amplitude=-float(current_uA) * axs.uA,
    )
    drive = axs.ExtracellularDrive(
        id=axs.DriveId("nrv_life"),
        footprint=context.footprint,
        stimulus=stimulus,
        metadata={"source": "nrv_life_fem"},
    )
    simulation = axs.AxonInstance(context.axon)
    simulation.add_extracellular_stimulation(
        stimulation=axs.ExtracellularStimulation([drive])
    )
    return simulation


def build_axonscope_axon(row: RealisticFiberRow, *, config: ExampleConfig) -> axs.axons.Axon:
    """Build the AxonScope axon template matching one NRV row."""

    diameter = max(float(row.diameter_um), 0.2) * axs.um
    if row.kind == "mrg":
        nodes = max(
            2,
            axs.axons.mrg_like_nodes_from_length(
                diameter,
                config.nerve_length_um * axs.um,
                x_shift=row.x_shift_um * axs.um,
            ),
        )
        return axs.axons.MRG(
            diameter=diameter,
            nodes=nodes,
            length=config.nerve_length_um * axs.um,
            x_shift=row.x_shift_um * axs.um,
        )
    if row.kind == "rattay":
        compartments = int(config.unmyelinated_compartments)
        if compartments <= 0:
            compartments = max(3, int(config.nerve_length_um // 25))
        return axs.axons.RattayAberham(
            length=config.nerve_length_um * axs.um,
            diameter=diameter,
            compartments=compartments,
            celsius=37.0 * axs.degC,
        )
    return axs.axons.HodgkinHuxley(
        length=config.nerve_length_um * axs.um,
        diameter=diameter,
        compartments=max(3, int(config.nerve_length_um // 25)),
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


def row_key(row: RealisticFiberRow) -> tuple[str, int]:
    """Return the stable fiber comparison key."""

    return (str(row.fascicle_id), int(row.fiber_index))


def fascicle_sort_key(fascicle_id: str) -> tuple[int, int | str]:
    """Sort numeric NRV fascicle identifiers naturally."""

    try:
        return (0, int(fascicle_id))
    except ValueError:
        return (1, str(fascicle_id))


def print_activation_summary(
    console: Console,
    comparisons: Sequence[ActivationComparison],
    *,
    validation_current_uA: float,
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
        f"\n[bold]Fiber activation agreement at {validation_current_uA:.2f} uA[/bold]: "
        f"{matched}/{len(comparisons)} matched "
        f"({100.0 * matched / len(comparisons):.1f}%). "
        f"NRV active={nrv_count}, AxonScope active={axonscope_count}."
    )

    table = Table(title="Fiber-by-fiber activation")
    for column in ("fasc", "fiber", "kind", "diam (um)", "NRV", "AxonScope", "match"):
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
        console.print(
            f"[dim]Showing {limit}/{len(comparisons)} fibers. "
            "Set ExampleConfig.print_fiber_limit = 0 for all.[/dim]"
        )


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


def plot_recruitment_curve(
    ax: Any,
    curve: axs.protocols.RecruitmentCurve,
    *,
    rows: Sequence[RealisticFiberRow],
    validation_current_uA: float,
) -> None:
    """Plot one AxonScope recruitment curve per fascicle."""

    ax.clear()
    row_list = list(rows)
    if not row_list:
        ax.set_title("AxonScope recruitment by fascicle")
        ax.text(0.5, 0.5, "no simulated fibers", ha="center", va="center")
        ax.set_axis_off()
        return
    amplitudes = np.asarray(curve.amplitudes_uA, dtype=float)
    activated = np.asarray(curve.activated, dtype=bool)
    row_fascicles = np.asarray([row.fascicle_id for row in row_list], dtype=object)
    fascicle_ids = sorted(set(row_fascicles.tolist()), key=fascicle_sort_key)
    for fascicle_id in fascicle_ids:
        mask = row_fascicles == fascicle_id
        if not np.any(mask):
            continue
        fractions = np.sum(activated[:, mask], axis=1) / float(np.sum(mask))
        ax.plot(amplitudes, fractions, marker="o", linewidth=1.6, label=f"fasc {fascicle_id}")
    ax.axvline(
        validation_current_uA,
        color="#f58518",
        linestyle="--",
        linewidth=1.4,
        label="NRV validation amplitude",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("LIFE current amplitude [uA]")
    ax.set_ylabel("Recruitment fraction")
    ax.set_title("AxonScope recruitment by fascicle")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)


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
    if comparisons:
        max_error = max(
            float(np.max(np.abs(item.axonscope_nodes_um - item.expected_nodes_um)))
            for item in comparisons
        )
        console.print(
            f"[dim]MRG node phase conversion checked internally; "
            f"max node-position error={max_error:.3g} um.[/dim]"
        )


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
    ax.scatter(
        y[is_mrg],
        z[is_mrg],
        color="#54a24b",
        s=42,
        edgecolor="black",
        linewidth=0.35,
        label="MRG",
    )
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


def nrv_life_footprint(
    life_setup: LifeElectrodeSetup,
    *,
    positions_um: np.ndarray,
    row: RealisticFiberRow,
) -> axs.ExtracellularFootprint:
    """Sample NRV's current-independent LIFE/FEM footprint for one AxonScope row."""

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

    return axs.ExtracellularFootprint.shared(
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
            "gmsh_n_core": getattr(life_setup.extra_stim.model.mesh, "n_core", None),
            "nrv_footprint_unit": "mV/mA",
        },
    )


def nrv_row_id(row: RealisticFiberRow) -> int:
    """Return a stable integer ID for single-fiber NRV footprint sampling."""

    try:
        fascicle_id = int(row.fascicle_id)
    except ValueError:
        fascicle_id = 0
    return fascicle_id * 1_000_000 + int(row.fiber_index)


def _format_ratio(numerator: float, denominator: float) -> str:
    if denominator <= 0.0:
        return "n/a"
    return f"{numerator / denominator:.2f}x"


if __name__ == "__main__":
    main()
