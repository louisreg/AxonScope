"""Explore velocity versus diameter with automatic batching.

Run:
    python examples/basic/06_activation_velocity.py

Axons are built explicitly, then AxonScope automatically batches compatible
simulations when ``axs.AxonSimulation(...).run()`` is called.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # We compare two model families:
    # - Hodgkin-Huxley, unmyelinated, over small diameters;
    # - MRG, myelinated, over larger fiber diameters.
    hh_diameters = np.linspace(0.1, 2.0, 20) * axs.um
    mrg_diameters = np.linspace(2.0, 20.0, 20) * axs.um

    clamp_start = 1.0 * axs.ms
    clamp_duration = 0.1 * axs.ms
    clamp_current = 5.0 * axs.nA

    # First build the unmyelinated pool. The clamp is placed at x=0 so the
    # spike travels along the cable and a velocity can be estimated from Vm.
    hh_simulations = []
    for diameter in hh_diameters:
        axon = axs.axons.HodgkinHuxley(
            length=5000.0 * axs.um,
            diameter=diameter,
            compartments=501,
            celsius=32.0 * axs.degC,
            v_init=-67.5 * axs.mV,
            include_passive_leak=True,
            g_pas=0.001,
            e_pas=-70.0,
        )
        sim = axs.AxonInstance(axon)
        sim.add_current_clamp(
            position=0.0 * axs.um,
            current=axs.Stimulus.pulse(
                start=clamp_start,
                duration=clamp_duration,
                amplitude=clamp_current,
            ),
        )
        hh_simulations.append(sim)

    # Then build the myelinated pool. For MRG we stimulate the first Ranvier node
    # explicitly instead of indexing raw compartment-position arrays.
    mrg_simulations = []
    for diameter in mrg_diameters:
        axon = axs.axons.MRG(
            diameter=diameter,
            nodes=21,
            compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
        )
        sim = axs.AxonInstance(axon)
        sim.add_current_clamp(
            position=axon.node_position("proximal", unit=axs.um),
            current=axs.Stimulus.pulse(
                start=clamp_start,
                duration=clamp_duration,
                amplitude=clamp_current,
            ),
        )
        mrg_simulations.append(sim)

    # Pool execution keeps the public code simple. Compatible rows are batched
    # internally; the user still receives one result view per input axon.
    hh_results = axs.AxonSimulation(
        hh_simulations,
        duration=10.0 * axs.ms,
        dt=0.001 * axs.ms,
        progress="plain",
    ).run()
    mrg_results = axs.AxonSimulation(
        mrg_simulations,
        duration=5.0 * axs.ms,
        dt=0.001 * axs.ms,
        progress="plain",
    ).run()

    # The structured analysis layer reads recorded Vm, keeps statuses, and
    # exposes the same text/dataframe/plot surface used by reports.
    velocity = axs.analysis.ConductionVelocity()
    hh_velocity = hh_results.analyze(velocity)
    mrg_velocity = mrg_results.analyze(velocity)
    hh_speeds = hh_velocity.values.astype(float)
    mrg_speeds = mrg_velocity.values.astype(float)

    print("\n=== HH unmyelinated ===")
    for diameter, speed in zip(hh_diameters, hh_speeds, strict=True):
        print(f"d={diameter.to(axs.um).magnitude:>5.2f} um: {speed:>8.3f} m/s")

    print("\n=== MRG myelinated ===")
    for diameter, speed in zip(mrg_diameters, mrg_speeds, strict=True):
        print(f"d={diameter.to(axs.um).magnitude:>5.2f} um: {speed:>8.3f} m/s")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    hh_velocity.plot(
        ax=ax,
        x=hh_diameters.to(axs.um).magnitude,
        x_label="diameter [um]",
        y_label="velocity [m/s]",
        label="HH",
    )
    mrg_velocity.plot(
        ax=ax,
        x=mrg_diameters.to(axs.um).magnitude,
        x_label="diameter [um]",
        y_label="velocity [m/s]",
        label="MRG",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Conduction velocity versus diameter")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
