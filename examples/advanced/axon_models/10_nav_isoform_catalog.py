"""Run all nine public Nav1.x isoforms through one cable composition.

Run:
    python examples/advanced/axon_models/10_nav_isoform_catalog.py

The shared potassium/leak support makes the runtime comparison consistent but
is not a physiological model for every isoform. The independent ModelDB I-V,
G-V, availability, and recovery validation remains under
``benchmark/curves/nav_isoform_voltage_clamp.py``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


NAV_MODELS = (
    ("Nav1.1", axs.membranes.Nav11),
    ("Nav1.2", axs.membranes.Nav12),
    ("Nav1.3", axs.membranes.Nav13),
    ("Nav1.4", axs.membranes.Nav14),
    ("Nav1.5", axs.membranes.Nav15),
    ("Nav1.6", axs.membranes.Nav16),
    ("Nav1.7", axs.membranes.Nav17),
    ("Nav1.8", axs.membranes.Nav18),
    ("Nav1.9", axs.membranes.Nav19),
)


def build_simulation(nav_model):
    membrane = axs.membranes.Composite(
        {
            "sodium": nav_model(
                celsius=22.0 * axs.degC,
                gbar=3_000.0 * axs.mS_per_cm2,
            ),
            "potassium_leak": axs.membranes.HodgkinHuxley(
                celsius=22.0 * axs.degC,
                gnabar=0.0 * axs.mS_per_cm2,
                gkbar=80.0 * axs.mS_per_cm2,
                gl=7.0 * axs.mS_per_cm2,
                ek=-90.0 * axs.mV,
                el=-73.2 * axs.mV,
            ),
        }
    )
    axon = axs.axons.Unmyelinated(
        membrane=membrane,
        length=2_000.0 * axs.um,
        diameter=1.0 * axs.um,
        compartments=201,
        v_init=-70.0 * axs.mV,
        temperature=22.0 * axs.degC,
    )
    simulation = axs.AxonInstance(axon)
    simulation.add_intracellular_context(
        context=axs.IntracellularCurrentClamp(
            position=1_000.0 * axs.um,
            current=axs.Stimulus.pulse(
                start=1.0 * axs.ms,
                duration=0.1 * axs.ms,
                amplitude=5.0 * axs.nA,
            ),
        )
    )
    return simulation


def main() -> None:
    simulations = tuple(build_simulation(model) for _, model in NAV_MODELS)
    results = axs.AxonSimulation(
        simulations,
        duration=10.0 * axs.ms,
        dt=0.005 * axs.ms,
    ).run()

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for axis, (name, _), result in zip(axes.ravel(), NAV_MODELS, results, strict=True):
        _, center_vm = result.trace_values(
            position=1_000.0 * axs.um,
            voltage_unit=axs.mV,
        )
        _, probe_vm = result.trace_values(
            position=1_500.0 * axs.um,
            voltage_unit=axs.mV,
        )
        print(
            f"{name}: center peak {np.max(center_vm):7.2f} mV, "
            f"distal peak {np.max(probe_vm):7.2f} mV"
        )
        result.plot_traces(
            ax=axis,
            positions=(1_000.0 * axs.um, 1_500.0 * axs.um),
            labels=("stimulus", "distal probe"),
            voltage_unit=axs.mV,
            title=name,
        )
    plt.show()


if __name__ == "__main__":
    main()
