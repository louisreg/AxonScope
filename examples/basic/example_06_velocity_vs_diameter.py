"""Example 06: velocity versus diameter with automatic batching.

Run:
    python examples/basic/example_06_velocity_vs_diameter.py

Axons are built explicitly, then AxonScope automatically batches compatible
simulations when ``axs.simulate_pool`` is called.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


HH_DIAMETERS = np.linspace(0.1, 2.0, 10) * axs.um
MRG_DIAMETERS = np.linspace(2.0, 20.0, 10) * axs.um

CLAMP_START = 1.0 * axs.ms
CLAMP_DURATION = 0.1 * axs.ms
CLAMP_CURRENT = 5.0 * axs.nA


def first_compartment_position(axon: axs.axons.Axon) -> object:
    """Return the first compartment position."""

    return float(axon.layout.position_values(unit=axs.um)[0]) * axs.um


def print_curve(label: str, diameters: object, speeds_m_s: list[float]) -> None:
    print(f"\n=== {label} ===")
    for diameter, speed in zip(diameters, speeds_m_s, strict=True):
        print(f"d={diameter.to(axs.um).magnitude:>5.2f} um: {speed:>8.3f} m/s")


def main() -> None:
    hh_simulations = []
    for diameter in HH_DIAMETERS:
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
                start=CLAMP_START,
                duration=CLAMP_DURATION,
                amplitude=CLAMP_CURRENT,
            ),
        )
        hh_simulations.append(sim)

    mrg_simulations = []
    for diameter in MRG_DIAMETERS:
        axon = axs.axons.MRG(
            diameter=diameter,
            nodes=21,
            compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
        )
        sim = axs.AxonInstance(axon)
        sim.add_current_clamp(
            position=first_compartment_position(axon),
            current=axs.Stimulus.pulse(
                start=CLAMP_START,
                duration=CLAMP_DURATION,
                amplitude=CLAMP_CURRENT,
            ),
        )
        mrg_simulations.append(sim)

    hh_results = axs.simulate_pool(
        hh_simulations,
        duration=10.0 * axs.ms,
        dt=0.001 * axs.ms,
        progress=True,
    )
    mrg_results = axs.simulate_pool(
        mrg_simulations,
        duration=5.0 * axs.ms,
        dt=0.001 * axs.ms,
        progress=True,
    )

    hh_speeds = [axs.analysis.conduction_velocity(result) for result in hh_results]
    mrg_speeds = [axs.analysis.conduction_velocity(result) for result in mrg_results]

    print_curve("HH unmyelinated", HH_DIAMETERS, hh_speeds)
    print_curve("MRG myelinated", MRG_DIAMETERS, mrg_speeds)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.loglog(HH_DIAMETERS.to(axs.um).magnitude, hh_speeds, "o-", label="HH")
    ax.loglog(MRG_DIAMETERS.to(axs.um).magnitude, mrg_speeds, "o-", label="MRG")
    ax.set_xlabel("diameter (um)")
    ax.set_ylabel("velocity (m/s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
