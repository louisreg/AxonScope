"""Reuse sampled extracellular drives and combine several sources.

Run:
    python examples/advanced/stimulation/01_stimulation_contexts.py

The important API pattern is:

    helper geometry -> ExtracellularFootprint -> ExtracellularDrive

Point-source geometry is only a helper used to generate footprints. The solver
receives the typed `ExtracellularStimulation`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


def main() -> None:
    # Step 1: choose the axon and the intrinsic positions used to sample fields.
    axon = axs.axons.MRG(diameter=10.0 * axs.um, nodes=5)
    positions = axon.layout.position_values(unit=axs.um) * axs.um
    center_x = positions[axon.n_compartments // 2]
    t = np.linspace(0.0, 1.6, 161) * axs.ms

    # Step 2: one helper geometry can generate several drives by pairing the
    # same sampled footprint with different temporal stimuli.
    base_electrode = axs.analytical.PointSourceElectrode(
        x=center_x,
        z=100.0 * axs.um,
    )
    cathodic = axs.Stimulus.pulse(
        start=0.45 * axs.ms,
        duration=0.08 * axs.ms,
        amplitude=-80.0 * axs.uA,
    )
    anodic = axs.Stimulus.pulse(
        start=0.45 * axs.ms,
        duration=0.08 * axs.ms,
        amplitude=80.0 * axs.uA,
    )

    cathodic_stimulation = axs.analytical.point_source_stimulation(
        base_electrode,
        positions,
        sigma=0.3 * axs.S_per_m,
        stimulus=cathodic,
        drive_id="source",
    )
    anodic_stimulation = cathodic_stimulation.replace_drive(
        axs.DriveId("source"),
        stimulus=anodic,
    )

    # Step 3: combine several helpers by building several drives on the same
    # intrinsic position support.
    left_electrode = axs.analytical.PointSourceElectrode(
        x=center_x - 250.0 * axs.um,
        z=500.0 * axs.um,
    )
    right_electrode = axs.analytical.PointSourceElectrode(
        x=center_x + 250.0 * axs.um,
        z=500.0 * axs.um,
    )
    left_drive = axs.analytical.point_source_drive(
        left_electrode,
        positions,
        sigma=0.3 * axs.S_per_m,
        stimulus=cathodic,
        drive_id="left",
    )
    right_drive = axs.analytical.point_source_drive(
        right_electrode,
        positions,
        sigma=0.3 * axs.S_per_m,
        stimulus=anodic,
        drive_id="right",
    )
    bipolar_stimulation = axs.ExtracellularStimulation([left_drive, right_drive])

    cases = [
        ("same footprint, cathodic", cathodic_stimulation),
        ("same footprint, anodic", anodic_stimulation),
        ("two sampled drives", bipolar_stimulation),
    ]

    # Step 4: solve each case explicitly. The axon description is reused, but
    # each AxonInstance receives one typed stimulation for that run.
    results = []
    for label, stimulation in cases:
        simulation = axs.AxonInstance(axon)
        simulation.add_extracellular_stimulation(stimulation=stimulation)
        run = axs.AxonSimulation(
            simulation,
            duration=1.6 * axs.ms,
            dt=0.02 * axs.ms,
        ).run()
        result = run.single
        results.append((label, stimulation, result))

    # Step 5: plot the imposed extracellular field and the membrane response at
    # the center compartment.
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    for col, (label, stimulation, result) in enumerate(results):
        stimulation.plot_potential(
            t,
            ax=axes[0, col],
            time_unit=axs.ms,
            position_unit=axs.um,
            voltage_unit=axs.mV,
            title=label,
        )

        result.plot_trace(
            ax=axes[1, col],
            position=center_x,
            voltage_unit=axs.mV,
            title=f"Vm at x={center_x.to(axs.um).magnitude:g} um",
        )

    plt.show()


if __name__ == "__main__":
    main()
