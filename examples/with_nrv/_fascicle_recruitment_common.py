"""Shared NRV geometry-to-AxonScope recruitment example runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Patch, Polygon
from rich.console import Console

import axonscope as axs
from axonscope.integrations import nrv as axs_nrv


@dataclass(frozen=True)
class ExampleConfig:
    """Editable constants for the NRV-to-AxonScope recruitment example."""

    nerve_diameter_um: float = 1_000.0
    nerve_length_um: float = 10_000.0
    axons_per_fascicle: int = 100
    percent_unmyelinated: float = 0.7
    delta_trace_um: float = 10.0
    fascicle_vertices: int = 50
    include_unmyelinated: bool = True
    duration_ms: float = 3.0
    dt_ms: float = 0.001
    stimulus_start_ms: float = 0.1
    pulse_duration_ms: float = 0.1
    observer_time_chunk_steps: int | None = axs.DEFAULT_OBSERVER_TIME_CHUNK_STEPS
    solver_progress: bool | str = False
    recruitment_amplitudes_uA: tuple[float, ...] = tuple(
        float(value) for value in np.linspace(0.0, 300.0, 21)
    )
    activation_threshold_mV: float = 0.0
    unmyelinated_compartments: int = 0
    life_diameter_um: float = 25.0
    life_length_um: float = 1_000.0
    life_fascicle_id: str = "0"
    fem_n_proc: int | None = None
    gmsh_n_core: int | None = 1


@dataclass(frozen=True)
class NrvGeometry:
    """NRV-owned nerve geometry plus contours for the AxonScope plot."""

    nerve: Any
    nerve_contour: np.ndarray
    fascicle_contours: tuple[np.ndarray, ...]
    life_fascicle_id: str


GeometryBuilder = Callable[[Any, ExampleConfig], NrvGeometry]


def run_fascicle_recruitment_example(
    *,
    config: ExampleConfig,
    build_geometry: GeometryBuilder,
    geometry_label: str,
) -> None:
    console = Console(width=110)

    import nrv

    console.print(f"[bold]1. Build NRV {geometry_label} geometry[/bold]")
    middle_amplitude_uA = config.recruitment_amplitudes_uA[
        len(config.recruitment_amplitudes_uA) // 2
    ]
    geometry = build_geometry(nrv, config)
    nerve = geometry.nerve

    for fascicle in nerve.fascicles.values():
        fascicle.fill(
            n_ax=config.axons_per_fascicle,
            percent_unmyel=config.percent_unmyelinated,
            delta_trace=config.delta_trace_um,
            with_node_shift=True,
        )

    fascicle_key: object = geometry.life_fascicle_id
    if fascicle_key not in nerve.fascicles:
        fascicle_key = int(geometry.life_fascicle_id)
    life_y_um, life_z_um = nerve.fascicles[fascicle_key].center
    life_x_offset_um = (config.nerve_length_um - config.life_length_um) / 2.0
    extra_stim = nrv.FEM_stimulation(
        endo_mat="endoneurium_ranck",
        peri_mat="perineurium",
        epi_mat="epineurium",
        ext_mat="saline",
        n_proc=config.fem_n_proc,
    )
    electrode = nrv.LIFE_electrode(
        "LIFE_2",
        config.life_diameter_um,
        config.life_length_um,
        life_x_offset_um,
        life_y_um,
        life_z_um,
    )
    nrv_stimulus = nrv.stimulus()
    nrv_stimulus.pulse(
        config.stimulus_start_ms,
        -float(middle_amplitude_uA),
        config.pulse_duration_ms,
    )
    extra_stim.add_electrode(electrode, nrv_stimulus)
    nerve.attach_extracellular_stimulation(extra_stim)
    if config.fem_n_proc is not None:
        nerve.extra_stim.set_n_proc(int(config.fem_n_proc))
    if config.gmsh_n_core is not None:
        nerve.extra_stim.model.mesh.n_core = int(config.gmsh_n_core)

    axons = axs_nrv.population_from_nrv(
        nerve,
        nerve_length_um=config.nerve_length_um,
        include_unmyelinated=config.include_unmyelinated,
        unmyelinated_compartments=config.unmyelinated_compartments,
    )
    console.print(
        f"NRV generated {len(nerve.fascicles)} fascicles and {len(axons)} AxonScope axons."
    )

    console.print("[bold]2. Sample NRV LIFE/FEM footprints on AxonScope axons[/bold]")
    footprints = axs_nrv.footprints_from_nrv(nerve, axons)
    pool = footprints.stimulated_population(
        electrode_index=0,
        stimulus=_life_pulse(
            current=0.0 * axs.uA,
            start_ms=config.stimulus_start_ms,
            pulse_duration_ms=config.pulse_duration_ms,
        ),
        drive_id_prefix="nrv_life",
    )

    def update_life_current(simulation: axs.AxonInstance, current: object) -> None:
        if simulation.extracellular_stimulation is None:
            raise ValueError("simulation has no extracellular stimulation.")
        updated = simulation.extracellular_stimulation.replace_drive(
            axs.DriveId("nrv_life_0"),
            stimulus=_life_pulse(
                current=current,
                start_ms=config.stimulus_start_ms,
                pulse_duration_ms=config.pulse_duration_ms,
            ),
        )
        simulation.add_extracellular_stimulation(stimulation=updated, replace=True)

    console.print("[bold]3. Run AxonScope recruitment sweep[/bold]")
    activation = axs.analysis.ActivationCriterion(
        threshold=config.activation_threshold_mV * axs.mV,
        blanking=config.stimulus_start_ms * axs.ms,
        target=axs.positions.ALL,
    )
    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update_life_current,
        values=np.asarray(config.recruitment_amplitudes_uA, dtype=float) * axs.uA,
        duration=config.duration_ms * axs.ms,
        dt=config.dt_ms * axs.ms,
        criterion=activation,
        recording=axs.Recording.none(),
        batch_options=axs.BatchOptions.none(
            time_chunk_steps=config.observer_time_chunk_steps
        ),
        progress=True,
        solver_progress=config.solver_progress,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    row_fascicles = np.asarray([row.fascicle_id for row in footprints.rows], dtype=object)
    activated = np.asarray(curve.activated, dtype=bool)
    curve.plot_groups(
        np.asarray([f"fasc {value}" for value in row_fascicles], dtype=object),
        ax=ax,
        unit=axs.uA,
        include_total=False,
    )
    ax.set_title(f"AxonScope recruitment on NRV {geometry_label} fibers")

    snapshot_index = len(curve.amplitudes_uA) // 2
    snapshot_amplitude_uA = float(curve.amplitudes_uA[snapshot_index])
    snapshot_active = activated[snapshot_index]
    fig_snapshot, ax_snapshot = plt.subplots(
        figsize=(6.2, 6.0),
        constrained_layout=True,
    )
    ax_snapshot.add_patch(
        Polygon(
            geometry.nerve_contour,
            closed=True,
            fill=False,
            linewidth=1.4,
            color="0.25",
        )
    )
    for contour in geometry.fascicle_contours:
        ax_snapshot.add_patch(
            Polygon(
                contour,
                closed=True,
                fill=False,
                linewidth=1.0,
                linestyle="--",
                color="0.55",
            )
        )

    fiber_patches = [
        Circle((row.y_um, row.z_um), max(row.diameter_um / 2.0, 0.1))
        for row in footprints.rows
    ]
    fiber_colors = [
        ("#2868b0" if row.kind == "mrg" else "#d97627")
        if bool(is_active)
        else "0.78"
        for row, is_active in zip(footprints.rows, snapshot_active, strict=True)
    ]
    collection = PatchCollection(
        fiber_patches,
        facecolor=fiber_colors,
        edgecolor="0.12",
        linewidth=0.25,
        alpha=0.92,
    )
    ax_snapshot.add_collection(collection)
    ax_snapshot.scatter(
        [life_y_um],
        [life_z_um],
        marker="+",
        s=140,
        color="crimson",
        linewidths=2.2,
        label="LIFE center",
    )
    ax_snapshot.set_aspect("equal", adjustable="box")
    margin_um = config.nerve_diameter_um * 0.08
    ax_snapshot.set_xlim(
        float(np.min(geometry.nerve_contour[:, 0])) - margin_um,
        float(np.max(geometry.nerve_contour[:, 0])) + margin_um,
    )
    ax_snapshot.set_ylim(
        float(np.min(geometry.nerve_contour[:, 1])) - margin_um,
        float(np.max(geometry.nerve_contour[:, 1])) + margin_um,
    )
    ax_snapshot.set_xlabel("y [um]")
    ax_snapshot.set_ylabel("z [um]")
    ax_snapshot.set_title(
        f"Activated fibers at {snapshot_amplitude_uA:.1f} uA"
    )
    ax_snapshot.grid(True, alpha=0.22)
    ax_snapshot.legend(
        handles=[
            Patch(facecolor="#2868b0", edgecolor="0.12", label="activated myelinated"),
            Patch(facecolor="#d97627", edgecolor="0.12", label="activated unmyelinated"),
            Patch(facecolor="0.78", edgecolor="0.12", label="not activated"),
        ],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    plt.show()


def build_realistic_histology_geometry(nrv_module: Any, config: ExampleConfig) -> NrvGeometry:
    nerve_contour, fascicle_contours = _load_histology_contours(
        nrv_module,
        nerve_diameter_um=config.nerve_diameter_um,
        fascicle_vertices=config.fascicle_vertices,
    )
    nerve = nrv_module.nerve(
        diameter=_nrv_numeric(config.nerve_diameter_um),
        length=_nrv_numeric(config.nerve_length_um),
    )
    for fascicle_id, contour in enumerate(fascicle_contours):
        fascicle = nrv_module.fascicle(ID=fascicle_id)
        fascicle.set_geometry(
            geometry=nrv_module.create_cshape(vertices=np.asarray(contour, dtype=float))
        )
        nerve.add_fascicle(fascicle)
    return NrvGeometry(
        nerve=nerve,
        nerve_contour=nerve_contour,
        fascicle_contours=tuple(fascicle_contours),
        life_fascicle_id=config.life_fascicle_id,
    )


def _nrv_numeric(value: float) -> int | float:
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _life_pulse(*, current: object, start_ms: float, pulse_duration_ms: float) -> axs.Stimulus:
    amplitude = current if hasattr(current, "to") else float(current) * axs.uA
    return axs.Stimulus.pulse(
        start=float(start_ms) * axs.ms,
        duration=float(pulse_duration_ms) * axs.ms,
        amplitude=-amplitude,
    )


def _load_histology_contours(
    nrv_module: object,
    *,
    nerve_diameter_um: float,
    fascicle_vertices: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
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
    contours, hierarchy = cv2.findContours(
        threshold,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        raise RuntimeError("NRV geometry image did not yield contours.")

    hierarchy = hierarchy.squeeze()
    nerve_id = 2
    nerve_points_pix = contours[nerve_id].squeeze()
    center_pix = np.mean(nerve_points_pix, axis=0)
    centered_nerve = nerve_points_pix - center_pix
    radius_pix = np.max(np.abs(centered_nerve))
    scale = float(nerve_diameter_um) / (2.0 * radius_pix)
    scale_yz = scale * np.asarray([1.0, -1.0])
    nerve_contour = centered_nerve * scale_yz

    fascicle_contours = []
    for index, contour in enumerate(contours):
        if hierarchy[index, -1] != nerve_id:
            continue
        points = _undersample_cv2_contour(contour, vertices=fascicle_vertices)
        fascicle_contours.append((points - center_pix) * scale_yz)
    if not fascicle_contours:
        raise RuntimeError("NRV geometry image did not yield fascicle contours.")
    return np.asarray(nerve_contour, dtype=float), fascicle_contours


def _undersample_cv2_contour(contour: object, *, vertices: int) -> np.ndarray:
    """Keep the mesh-friendly contour sampling used by NRV's reference example."""

    points = np.asarray(contour).squeeze()
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("NRV contour extraction produced an invalid polygon.")
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    target = max(3, min(int(vertices), len(points)))
    indices = np.arange(target + 1) * len(points) // target
    indices[-1] -= 1
    points = points[indices]
    return np.asarray(points, dtype=float)
