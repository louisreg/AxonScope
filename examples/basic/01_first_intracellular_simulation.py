"""Simulate one Hodgkin-Huxley axon with Pint quantities.

Run:
    python examples/basic/01_first_intracellular_simulation.py
"""

import matplotlib.pyplot as plt

import axonscope as axs
from axonscope.axons import HodgkinHuxley


def main() -> None:
    # Start with one unmyelinated cable, one current clamp, and one retained Vm
    # result. Plain numbers are always paired with units at the public boundary.
    length = 500.0 * axs.um
    dt = 0.01 * axs.ms
    duration = 5.0 * axs.ms
    clamp_x = length / 2.0

    # Hodgkin-Huxley is a compact introductory model: one cable, one membrane
    # family, and a small number of compartments.
    axon = HodgkinHuxley(
        length=length,
        diameter=0.5 * axs.um,
        compartments=41,
        celsius=6.3 * axs.degC,
    )

    # AxonInstance is the executable occurrence of the axon. Stimulation,
    # offsets, and per-run settings attach to the instance rather than the
    # reusable axon description.
    sim = axs.AxonInstance(axon)
    clamp = axs.IntracellularCurrentClamp(
        position=clamp_x,
        current=axs.Stimulus.pulse(
            start=1.0 * axs.ms,
            duration=0.5 * axs.ms,
            amplitude=2.0 * axs.nA,
        ),
    )
    sim.add_intracellular_context(context=clamp)

    # `simulate(...)` returns a single SimResult with time, Vm, metadata, and
    # convenience plotting helpers.
    result = axs.simulate(sim, duration=duration, dt=dt)

    print(f"Recorded Vm shape: {result.Vm.shape} (time steps x compartments)")

    # The three panels connect the physical model to the numerical output:
    # layout, one local trace, and the full Vm time/space map.
    fig, (ax_layout, ax_trace, ax_map) = plt.subplots(1, 3, figsize=(13, 3.5))
    axon.layout.plot(
        ax=ax_layout,
        position_unit=axs.um,
        title="HH layout",
        compartment_labels="auto",
        max_compartment_labels=50,
    )
    ax_layout.axvline(
        clamp.position_um,
        color="C3",
        linestyle="--",
        linewidth=1.2,
        label="clamp",
    )
    ax_layout.legend(frameon=False)
    result.plot_trace(
        ax=ax_trace,
        position=clamp_x,
        voltage_unit=axs.mV,
        title="Clamp trace",
    )
    result.plot_map(ax=ax_map, voltage_unit=axs.mV)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
