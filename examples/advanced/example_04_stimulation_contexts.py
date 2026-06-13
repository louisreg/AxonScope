"""Reuse electrodes with different stimuli and combine multiple electrodes.

Run:
    python examples/advanced/example_04_stimulation_contexts.py

The important API pattern is:

    electrode.with_stimulus(stimulus)

It returns a stimulated copy of the same electrode geometry, so a base electrode
can be reused safely across simulations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs
from axonscope.solvers import CrankNicholson


def _solve(axon: axs.axons.Axon, context: axs.ExtracellularContext):
    sim = axs.AxonSimulation(axon)
    sim.add_extracellular_context(context=context)
    return CrankNicholson().solve(sim, tsim=1.6 * axs.ms, dt=0.02 * axs.ms)


def main() -> None:
    axon = axs.axons.MRG(diameter=10.0 * axs.um, nodes=5)
    x = axon.layout.position_values(unit=axs.um) * axs.um
    center_x = x[axon.n_compartments // 2]
    t = np.linspace(0.0, 1.6, 161) * axs.ms

    base_electrode = axs.PointSourceElectrode(
        x_um=center_x,
        z_um=100.0 * axs.um,
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

    left_electrode = axs.PointSourceElectrode(
        x_um=center_x - 250.0 * axs.um,
        z_um=500.0 * axs.um,
    )
    right_electrode = axs.PointSourceElectrode(
        x_um=center_x + 250.0 * axs.um,
        z_um=500.0 * axs.um,
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
    results = [(label, context, _solve(axon, context)) for label, context in cases]

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
