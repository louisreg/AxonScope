"""Reuse electrodes with different stimuli and combine multiple electrodes.

Run:
    python examples/advanced/stimulation/01_stimulation_contexts.py

The important API pattern is:

    electrode.with_stimulus(stimulus)

It returns a stimulated copy of the same electrode geometry, so a base electrode
can be reused safely across simulations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # Step 1: build one MRG axon and locate its center compartment. The electrode
    # geometry below is defined in the same physical coordinate system.
    axon = axs.axons.MRG(diameter=10.0 * axs.um, nodes=5)
    x = axon.layout.position_values(unit=axs.um) * axs.um
    center_x = x[axon.n_compartments // 2]
    t = np.linspace(0.0, 1.6, 161) * axs.ms

    # Step 2: create one reusable electrode geometry, then attach different
    # temporal stimuli to stimulated copies of it.
    base_electrode = axs.PointSourceElectrode(
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

    cathodic_context = axs.AnalyticalExtracellularContext(
        electrodes=[base_electrode.with_stimulus(cathodic)],
        sigma=0.3 * axs.S_per_m,
    )
    anodic_context = axs.AnalyticalExtracellularContext(
        electrodes=[base_electrode.with_stimulus(anodic)],
        sigma=0.3 * axs.S_per_m,
    )

    # Step 3: a context can also combine several electrodes. Here the two copies
    # use opposite pulse polarities and sit on opposite sides of the axon center.
    left_electrode = axs.PointSourceElectrode(
        x=center_x - 250.0 * axs.um,
        z=500.0 * axs.um,
    )
    right_electrode = axs.PointSourceElectrode(
        x=center_x + 250.0 * axs.um,
        z=500.0 * axs.um,
    )
    bipolar_context = axs.AnalyticalExtracellularContext(
        electrodes=[
            left_electrode.with_stimulus(cathodic),
            right_electrode.with_stimulus(anodic),
        ],
        sigma=0.3 * axs.S_per_m,
    )

    cases = [
        ("same electrode, cathodic", cathodic_context),
        ("same electrode, anodic", anodic_context),
        ("two electrodes", bipolar_context),
    ]

    # Step 4: solve each case explicitly. The axon description is reused, but
    # each AxonInstance receives one context for that run.
    results = []
    for label, context in cases:
        simulation = axs.AxonInstance(axon)
        simulation.add_extracellular_context(context=context)
        result = axs.simulate(
            simulation,
            duration=1.6 * axs.ms,
            dt=0.02 * axs.ms,
        )
        results.append((label, context, result))

    # Step 5: plot both sides of the concept: the imposed extracellular field and
    # the membrane response at the center compartment.
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    for col, (label, context, result) in enumerate(results):
        context.plot_evaluation(
            x,
            t,
            ax=axes[0, col],
            voltage_unit=axs.mV,
            title=label,
            colorbar=False,
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
