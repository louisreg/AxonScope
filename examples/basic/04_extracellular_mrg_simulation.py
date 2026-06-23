"""Stimulate one MRG axon with Pint-aware extracellular units.

Run:
    python examples/basic/04_extracellular_mrg_simulation.py
"""

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs
from axonscope.axons import MRG


def main() -> None:
    # MRG is a myelinated axon model. Here we keep only five nodes so the
    # example runs quickly while still showing nodal activation.
    axon = MRG(diameter=10.0 * axs.um, nodes=5)
    node_indices = np.asarray(axon.node_indices, dtype=int)
    node_positions = axon.node_position_values(unit=axs.um)

    # Put the electrode next to the middle node. The x coordinate is intrinsic
    # axon position; z is the distance from the axon.
    electrode_x = node_positions[len(node_positions) // 2] * axs.um
    electrode_z = 500.0 * axs.um

    electrode = axs.analytical.PointSourceElectrode(
        x=electrode_x,
        z=electrode_z,
    )
    stimulus = axs.Stimulus.biphasic(
        start=0.5 * axs.ms,
        cathodic_amplitude=150.0 * axs.uA,
        cathodic_duration=0.05 * axs.ms,
        interphase=0.02 * axs.ms,
    )

    positions = axon.layout.position_values(unit=axs.um) * axs.um
    extracellular = axs.analytical.point_source_stimulation(
        electrode,
        positions,
        sigma=0.3 * axs.S_per_m,
        stimulus=stimulus,
    )

    # The sampled extracellular stimulation is attached to the instance, not
    # the reusable MRG axon description.
    sim = axs.AxonInstance(axon)
    sim.add_extracellular_stimulation(stimulation=extracellular)

    # The default recording keeps full Vm, which is useful here because we want
    # to inspect several nodal traces after the run.
    run = axs.simulate(sim, duration=2.0 * axs.ms, dt=0.01 * axs.ms)
    result = run.single

    t_ms = result.time_values(unit=axs.ms)
    vm_mV = result.voltage_values(unit=axs.mV)
    x_um = axon.layout.position_values(unit=axs.um)

    print(f"MRG nodes recorded: {len(node_indices)}")

    fig, (ax_layout, ax_trace) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=False)
    axon.layout.plot(
        ax=ax_layout,
        position_unit=axs.um,
        title="MRG layout",
        compartment_labels="auto",
    )
    ax_layout.axvline(electrode.x_um, color="C2", linestyle="--", linewidth=1.5, label="electrode x")
    ax_layout.legend(ncol=2, frameon=False)

    for idx in node_indices[:3]:
        ax_trace.plot(t_ms, vm_mV[:, idx], label=f"node x={x_um[idx]:.0f} um")
    ax_trace.set_xlabel("Time [ms]")
    ax_trace.set_ylabel("Vm [mV]")
    ax_trace.set_title("Nodal voltage traces")
    ax_trace.grid(True, alpha=0.3)
    ax_trace.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
