"""Compute a recruitment curve for a mixed fiber population.

Run:
    python examples/basic/08_recruitment_curve_population.py

The population contains unmyelinated and myelinated fibers with random
model-compatible diameters, placed randomly in a circular cross-section. One
point-source electrode is placed at the center. The recruitment protocol keeps
the population and geometry fixed, and only updates the electrode current
between sampled amplitudes.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # Keep the example reproducible. Changing only the seed gives a different
    # synthetic population with the same workflow.
    rng = np.random.default_rng(7)

    fibers_per_family = 100
    circle_radius = 125.0 * axs.um
    fiber_length = 1500.0 * axs.um
    stim_start = 0.20 * axs.ms
    pulse_width = 0.10 * axs.ms
    sigma = 0.3 * axs.S_per_m
    current_steps = (
        np.asarray([5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 120.0, 160.0])
        * axs.uA
    )

    # The central electrode is shared by every simulation row. The current is
    # initialized at zero because `recruitment_sweep(...)` will update it for
    # each tested amplitude.
    electrode = axs.analytical.PointSourceElectrode(
        x=fiber_length / 2.0,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
        min_distance=5.0 * axs.um,
    )
    zero_current = axs.Stimulus.pulse(
        start=stim_start,
        duration=pulse_width,
        amplitude=0.0 * axs.uA,
    )

    # Draw positions uniformly in a disk. The square root on the random radius
    # is the standard trick that avoids over-sampling the center.
    radius_um = circle_radius.to(axs.um).magnitude
    unmyelinated_angles = rng.uniform(0.0, 2.0 * np.pi, fibers_per_family)
    unmyelinated_radii = radius_um * np.sqrt(rng.uniform(0.0, 1.0, fibers_per_family))
    unmyelinated_y = unmyelinated_radii * np.cos(unmyelinated_angles) * axs.um
    unmyelinated_z = unmyelinated_radii * np.sin(unmyelinated_angles) * axs.um

    myelinated_angles = rng.uniform(0.0, 2.0 * np.pi, fibers_per_family)
    myelinated_radii = radius_um * np.sqrt(rng.uniform(0.0, 1.0, fibers_per_family))
    myelinated_y = myelinated_radii * np.cos(myelinated_angles) * axs.um
    myelinated_z = myelinated_radii * np.sin(myelinated_angles) * axs.um

    # Pick model-compatible diameters. The unmyelinated family uses a continuous
    # range; the MRG family uses a small set of valid template diameters.
    unmyelinated_diameters = (
        rng.uniform(0.4, 1.2, fibers_per_family)
        * axs.um
    )
    myelinated_diameters = (
        rng.choice(np.asarray([7.3, 10.0, 12.8]), size=fibers_per_family)
        * axs.um
    )

    # Build the population row by row. The sampled y/z coordinates stay in this
    # example as geometry inputs for the analytical point-source helper; the
    # AxonInstance rows themselves remain local.
    pool: list[axs.AxonInstance] = []
    families: list[str] = []
    diameter_values_um: list[float] = []

    for diameter, y, z in zip(
        unmyelinated_diameters,
        unmyelinated_y,
        unmyelinated_z,
        strict=True,
    ):
        axon = axs.axons.RattayAberham(
            length=fiber_length,
            diameter=diameter,
            compartments=61,
            celsius=37.0 * axs.degC,
        )
        positions = axon.layout.position_values(unit=axs.um) * axs.um
        extracellular = axs.analytical.point_source_stimulation(
            electrode,
            positions,
            sigma=sigma,
            stimulus=zero_current,
            axon_y=y,
            axon_z=z,
        )
        sim = axs.AxonInstance(axon)
        sim.add_extracellular_stimulation(stimulation=extracellular)
        pool.append(sim)
        families.append("unmyelinated")
        diameter_values_um.append(float(diameter.to(axs.um).magnitude))

    for diameter, y, z in zip(
        myelinated_diameters,
        myelinated_y,
        myelinated_z,
        strict=True,
    ):
        axon = axs.axons.MRG(
            diameter=diameter,
            nodes=4,
            length=fiber_length,
            compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
        )
        positions = axon.layout.position_values(unit=axs.um) * axs.um
        extracellular = axs.analytical.point_source_stimulation(
            electrode,
            positions,
            sigma=sigma,
            stimulus=zero_current,
            axon_y=y,
            axon_z=z,
        )
        sim = axs.AxonInstance(axon)
        sim.add_extracellular_stimulation(stimulation=extracellular)
        pool.append(sim)
        families.append("myelinated")
        diameter_values_um.append(float(diameter.to(axs.um).magnitude))

    families_arr = np.asarray(families, dtype=object)
    diameter_um = np.asarray(diameter_values_um, dtype=float)

    # Recruitment is a row-wise activation test repeated for each current.
    # `target=ALL` means any recorded position may count as activation.
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=stim_start,
        target=axs.positions.ALL,
    )

    # The protocol callback mutates only the stimulus amplitude. That is what
    # lets the protocol reuse the same population while scanning currents.
    def update_point_source_current(
        sim: axs.AxonInstance,
        current_magnitude: Any,
    ) -> None:
        stimulation = sim.extracellular_stimulation
        if stimulation is None:
            raise ValueError("simulation has no extracellular stimulation to update.")
        drive = stimulation.drives[0]
        updated = stimulation.replace_drive(
            drive.id,
            stimulus=axs.Stimulus.pulse(
                start=stim_start,
                duration=pulse_width,
                amplitude=-current_magnitude,
            ),
        )
        sim.add_extracellular_stimulation(stimulation=updated, replace=True)

    curve = axs.protocols.recruitment_sweep(
        tuple(pool),
        update=update_point_source_current,
        values=current_steps,
        duration=4.0 * axs.ms,
        dt=0.025 * axs.ms,
        criterion=criterion,
        recording=axs.Recording.none(),
        batch_amplitudes=True,
        progress=True,
        solver_progress="plain",
    )

    print("=== Population ===")
    print(
        "unmyelinated diameter range: "
        f"{diameter_um[families_arr == 'unmyelinated'].min():.2f}-"
        f"{diameter_um[families_arr == 'unmyelinated'].max():.2f} um"
    )
    print(
        "myelinated diameter range:   "
        f"{diameter_um[families_arr == 'myelinated'].min():.2f}-"
        f"{diameter_um[families_arr == 'myelinated'].max():.2f} um"
    )

    print("\n=== Recruitment curve ===")
    print(curve.to_dataframe(unit=axs.uA).to_string(index=False))

    # The population plot colors each fiber by the first sampled current that
    # activated it. Non-activated fibers are gray.
    y_um = np.concatenate(
        [
            np.asarray(unmyelinated_y.to(axs.um).magnitude, dtype=float),
            np.asarray(myelinated_y.to(axs.um).magnitude, dtype=float),
        ]
    )
    z_um = np.concatenate(
        [
            np.asarray(unmyelinated_z.to(axs.um).magnitude, dtype=float),
            np.asarray(myelinated_z.to(axs.um).magnitude, dtype=float),
        ]
    )
    first_activation_uA = curve.first_activation_uA
    finite_thresholds = first_activation_uA[np.isfinite(first_activation_uA)]
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
        circle_radius.to(axs.um).magnitude,
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
        mask = families_arr == family
        marker_size = 20.0 + 8.0 * diameter_um[mask]
        scatter = ax_population.scatter(
            y_um[mask],
            z_um[mask],
            c=first_activation_uA[mask],
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

    curve.plot_groups(
        families_arr,
        ax=ax_curve,
        unit=axs.uA,
    )
    ax_curve.set_title("Recruitment")
    plt.show()


if __name__ == "__main__":
    main()
