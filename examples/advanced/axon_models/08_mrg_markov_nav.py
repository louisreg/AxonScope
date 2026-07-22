"""Compose an MRG axon with Markov Nav1.1 and Nav1.6 nodal channels.

Run:
    python examples/advanced/axon_models/08_mrg_markov_nav.py

The MRG potassium and leak currents are retained while its original sodium
conductances are disabled. Nav1.1 and Nav1.6 are then composed at every node
through the same public membrane compiler and runtime used by other models.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import axonscope as axs


def stimulated(axon):
    center_node = len(axon.node_indices) // 2
    center = axon.node_position(center_node, unit=axs.um)
    simulation = axs.AxonInstance(axon)
    simulation.add_intracellular_context(
        context=axs.IntracellularCurrentClamp(
            position=center,
            current=axs.Stimulus.pulse(
                start=1.0 * axs.ms,
                duration=0.1 * axs.ms,
                amplitude=5.0 * axs.nA,
            ),
        )
    )
    return simulation


def main() -> None:
    template = axs.axons.MRGLikeDoubleCableTemplate(
        diameter=10.0 * axs.um,
        nodes=11,
    )
    mrg_membranes = template.default_membranes()
    markov_node = axs.membranes.Composite(
        {
            "mrg_k_leak": axs.membranes.AxNode(
                gnapbar=0.0 * axs.mS_per_cm2,
                gnabar=0.0 * axs.mS_per_cm2,
            ),
            "nav11": axs.membranes.Nav11(
                gbar=11_900.0 * axs.mS_per_cm2,
                ena=50.0 * axs.mV,
            ),
            "nav16": axs.membranes.Nav16(
                gbar=10.0 * axs.mS_per_cm2,
                ena=50.0 * axs.mV,
            ),
        }
    )
    markov_membranes = axs.membranes.SectionLayout(
        node=markov_node,
        mysa=mrg_membranes.membrane_for("MYSA"),
        flut=mrg_membranes.membrane_for("FLUT"),
        stin=mrg_membranes.membrane_for("STIN"),
    )

    reference = axs.axons.MRG(diameter=10.0 * axs.um, nodes=11)
    markov = axs.axons.MRG(
        diameter=10.0 * axs.um,
        nodes=11,
        membranes=markov_membranes,
    )
    reference_result = axs.AxonSimulation(
        stimulated(reference),
        duration=4.0 * axs.ms,
        dt=0.005 * axs.ms,
    ).run().single
    markov_result = axs.AxonSimulation(
        stimulated(markov),
        duration=4.0 * axs.ms,
        dt=0.005 * axs.ms,
    ).run().single

    distal = reference.node_position(-2, unit=axs.um)
    print(markov_node.explain().format())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    reference_result.plot_trace(
        ax=axes[0],
        position=distal,
        voltage_unit=axs.mV,
        title="MRG distal node",
    )
    markov_result.plot_trace(
        ax=axes[1],
        position=distal,
        voltage_unit=axs.mV,
        title="MRG + Nav1.1/Nav1.6",
    )
    markov_result.plot_map(
        ax=axes[2],
        position_unit=axs.mm,
        voltage_unit=axs.mV,
        title="Markov Nav propagation",
    )
    plt.show()


if __name__ == "__main__":
    main()
