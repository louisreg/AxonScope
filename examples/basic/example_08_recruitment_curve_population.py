"""Example 08: recruitment curve for a mixed fiber population.

Run:
    python examples/basic/example_08_recruitment_curve_population.py

The population contains 50 unmyelinated and 50 myelinated fibers with random
model-compatible diameters, placed randomly in a 250 um diameter circular
cross-section. One point-source electrode is placed at the center. The
recruitment protocol keeps the population and extracellular context fixed, and
only updates the electrode current between sampled amplitudes.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


RNG_SEED = 7
FIBERS_PER_FAMILY = 50
CIRCLE_RADIUS = 125.0 * axs.um
FIBER_LENGTH = 1500.0 * axs.um
STIM_START = 0.20 * axs.ms
PULSE_WIDTH = 0.10 * axs.ms
SIGMA = 0.3 * axs.S_per_m
UNMYELINATED_DIAMETER_RANGE_UM = (0.4, 1.2)
MRG_DIAMETER_CHOICES_UM = np.asarray([7.3, 10.0, 12.8])
CURRENT_STEPS = np.asarray([5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 120.0, 160.0]) * axs.uA


def random_positions_in_disk(
    count: int,
    *,
    radius: Any,
    rng: np.random.Generator,
) -> tuple[Any, Any]:
    """Return random y/z positions uniformly sampled in a disk."""

    radius_um = radius.to(axs.um).magnitude
    angles = rng.uniform(0.0, 2.0 * np.pi, count)
    radii = radius_um * np.sqrt(rng.uniform(0.0, 1.0, count))
    y = radii * np.cos(angles) * axs.um
    z = radii * np.sin(angles) * axs.um
    return y, z


def make_shared_context() -> axs.AnalyticalExtracellularContext:
    """Create the central point-source extracellular context."""

    electrode = axs.PointSourceElectrode(
        x=FIBER_LENGTH / 2.0,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
        min_distance=5.0 * axs.um,
    )
    electrode.set_stimulus(
        axs.Stimulus.pulse(
            start=STIM_START,
            duration=PULSE_WIDTH,
            amplitude=0.0 * axs.uA,
        )
    )
    return axs.AnalyticalExtracellularContext(electrodes=[electrode], sigma=SIGMA)


def make_unmyelinated_simulation(
    *,
    diameter: Any,
    y: Any,
    z: Any,
    context: axs.AnalyticalExtracellularContext,
) -> axs.AxonInstance:
    """Create one unmyelinated Rattay-Aberham simulation."""

    axon = axs.axons.RattayAberham(
        length=FIBER_LENGTH,
        diameter=diameter,
        compartments=61,
        celsius=37.0 * axs.degC,
    )
    sim = axs.AxonInstance(axon, y=y, z=z)
    sim.add_extracellular_context(context=context)
    return sim


def make_myelinated_simulation(
    *,
    diameter: Any,
    y: Any,
    z: Any,
    context: axs.AnalyticalExtracellularContext,
) -> axs.AxonInstance:
    """Create one myelinated MRG simulation."""

    axon = axs.axons.MRG(
        diameter=diameter,
        nodes=4,
        length=FIBER_LENGTH,
        compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
    )
    sim = axs.AxonInstance(axon, y=y, z=z)
    sim.add_extracellular_context(context=context)
    return sim


def make_population() -> tuple[tuple[axs.AxonInstance, ...], np.ndarray, Any]:
    """Create the mixed population and its family labels."""

    rng = np.random.default_rng(RNG_SEED)
    context = make_shared_context()
    unmyelinated_diameters = rng.uniform(
        UNMYELINATED_DIAMETER_RANGE_UM[0],
        UNMYELINATED_DIAMETER_RANGE_UM[1],
        FIBERS_PER_FAMILY,
    ) * axs.um
    myelinated_diameters = rng.choice(
        MRG_DIAMETER_CHOICES_UM,
        size=FIBERS_PER_FAMILY,
        replace=True,
    ) * axs.um

    y_unmyelinated, z_unmyelinated = random_positions_in_disk(
        FIBERS_PER_FAMILY,
        radius=CIRCLE_RADIUS,
        rng=rng,
    )
    y_myelinated, z_myelinated = random_positions_in_disk(
        FIBERS_PER_FAMILY,
        radius=CIRCLE_RADIUS,
        rng=rng,
    )

    simulations: list[axs.AxonInstance] = []
    for diameter, y, z in zip(
        unmyelinated_diameters,
        y_unmyelinated,
        z_unmyelinated,
        strict=True,
    ):
        simulations.append(
            make_unmyelinated_simulation(
                diameter=diameter,
                y=y,
                z=z,
                context=context,
            )
        )
    for diameter, y, z in zip(
        myelinated_diameters,
        y_myelinated,
        z_myelinated,
        strict=True,
    ):
        simulations.append(
            make_myelinated_simulation(
                diameter=diameter,
                y=y,
                z=z,
                context=context,
            )
        )

    families = np.asarray(
        ["unmyelinated"] * FIBERS_PER_FAMILY
        + ["myelinated"] * FIBERS_PER_FAMILY,
        dtype=object,
    )
    diameters = np.concatenate(
        [
            unmyelinated_diameters.to(axs.um).magnitude,
            myelinated_diameters.to(axs.um).magnitude,
        ]
    ) * axs.um
    return tuple(simulations), families, diameters


def update_point_source_current(
    sim: axs.AxonInstance,
    current_magnitude: Any,
) -> None:
    """Change only the point-source current for one simulation row."""

    context = sim.extracellular_context
    if context is None:
        raise ValueError("simulation has no extracellular context to update.")
    electrode = context.electrodes[0]
    electrode.set_stimulus(
        axs.Stimulus.pulse(
            start=STIM_START,
            duration=PULSE_WIDTH,
            amplitude=-current_magnitude,
        )
    )


def print_summary(
    curve: axs.protocols.RecruitmentCurve,
    families: np.ndarray,
    diameters: Any,
) -> None:
    """Print total and family-wise recruitment counts."""

    diameter_um = diameters.to(axs.um).magnitude
    print("=== Population ===")
    print(
        "unmyelinated diameter range: "
        f"{diameter_um[families == 'unmyelinated'].min():.2f}-"
        f"{diameter_um[families == 'unmyelinated'].max():.2f} um"
    )
    print(
        "myelinated diameter range:   "
        f"{diameter_um[families == 'myelinated'].min():.2f}-"
        f"{diameter_um[families == 'myelinated'].max():.2f} um"
    )
    print()
    print("=== Recruitment curve ===")
    total_fibers = int(families.shape[0])
    unmyelinated_total = int(np.sum(families == "unmyelinated"))
    myelinated_total = int(np.sum(families == "myelinated"))
    for amplitude_uA, activated in zip(
        curve.amplitudes.to(axs.uA).magnitude,
        curve.activated,
        strict=True,
    ):
        total = int(np.sum(activated))
        unmyelinated = int(np.sum(activated[families == "unmyelinated"]))
        myelinated = int(np.sum(activated[families == "myelinated"]))
        print(
            f"{amplitude_uA:>6.1f} uA: "
            f"{total:>3d}/{total_fibers:<3d} total, "
            f"{unmyelinated:>2d}/{unmyelinated_total:<2d} unmyelinated, "
            f"{myelinated:>2d}/{myelinated_total:<2d} myelinated"
        )


def plot_population_and_recruitment(
    pool: tuple[axs.AxonInstance, ...],
    families: np.ndarray,
    diameters: Any,
    curve: axs.protocols.RecruitmentCurve,
) -> None:
    """Plot fiber placement and recruitment fractions."""

    y_um = np.asarray([sim.y_um for sim in pool], dtype=float)
    z_um = np.asarray([sim.z_um for sim in pool], dtype=float)
    diameter_um = diameters.to(axs.um).magnitude
    threshold_like_uA = curve.threshold_like_uA
    finite_thresholds = threshold_like_uA[np.isfinite(threshold_like_uA)]
    if finite_thresholds.size:
        color_min = float(np.min(finite_thresholds))
        color_max = float(np.max(finite_thresholds))
        if color_max <= color_min:
            color_max = color_min + 1.0
    else:
        color_min = 0.0
        color_max = 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")

    fig, (ax_population, ax_curve) = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.2),
        constrained_layout=True,
    )

    boundary = plt.Circle(
        (0.0, 0.0),
        CIRCLE_RADIUS.to(axs.um).magnitude,
        fill=False,
        linestyle="--",
        linewidth=1.0,
        color="0.35",
    )
    ax_population.add_patch(boundary)
    for family, marker, label in (
        ("unmyelinated", "o", "unmyelinated"),
        ("myelinated", "s", "myelinated"),
    ):
        mask = families == family
        marker_size = 20.0 + 8.0 * diameter_um[mask]
        scatter = ax_population.scatter(
            y_um[mask],
            z_um[mask],
            c=threshold_like_uA[mask],
            marker=marker,
            s=marker_size,
            cmap=cmap,
            vmin=color_min,
            vmax=color_max,
            label=label,
            edgecolors="black",
            linewidths=0.3,
        )
    ax_population.scatter(
        [0.0],
        [0.0],
        marker="+",
        s=120,
        color="crimson",
        linewidths=2.0,
        label="electrode",
    )
    ax_population.set_aspect("equal", adjustable="box")
    ax_population.set_xlabel("y [um]")
    ax_population.set_ylabel("z [um]")
    ax_population.set_title("Fiber positions")
    ax_population.grid(True, alpha=0.25)
    ax_population.legend(loc="upper right", fontsize=8)
    fig.colorbar(scatter, ax=ax_population, label="first activating sample [uA]")

    amplitudes_uA = curve.amplitudes.to(axs.uA).magnitude
    ax_curve.plot(
        amplitudes_uA,
        curve.fraction,
        marker="o",
        linewidth=2.0,
        label="all fibers",
    )
    for family, linestyle in (
        ("unmyelinated", "--"),
        ("myelinated", ":"),
    ):
        mask = families == family
        fraction = np.mean(curve.activated[:, mask], axis=1)
        ax_curve.plot(
            amplitudes_uA,
            fraction,
            marker="o",
            linestyle=linestyle,
            label=family,
        )
    ax_curve.set_xlabel("point-source current magnitude [uA]")
    ax_curve.set_ylabel("recruited fraction")
    ax_curve.set_ylim(-0.05, 1.05)
    ax_curve.set_title("Recruitment")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend()
    plt.show()


def main() -> None:
    pool, families, diameters = make_population()
    criterion = axs.results.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=STIM_START,
        target=axs.positions.ALL,
    )

    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update_point_source_current,
        amplitudes=CURRENT_STEPS,
        duration=4.0 * axs.ms,
        dt=0.025 * axs.ms,
        criterion=criterion,
        recording=axs.Recording.voltage(),
        progress=True,
    )

    print_summary(curve, families, diameters)
    plot_population_and_recruitment(pool, families, diameters, curve)


if __name__ == "__main__":
    main()
