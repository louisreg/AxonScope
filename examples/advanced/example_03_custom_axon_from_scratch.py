"""Create and simulate a custom axon from first principles.

Run:
    python examples/advanced/example_03_custom_axon_from_scratch.py

This example intentionally does not use `axs.axons.HodgkinHuxley` or `MRG`.
It builds a reusable axon class from:

- membrane descriptions;
- local sections;
- explicit layout elements;
- the base descriptive `Axon`;
- a simulation protocol with one current clamp.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


class CustomHeterogeneousAxon(axs.axons.Axon):
    """Small custom single-cable axon assembled from local sections."""

    def __init__(
        self,
        *,
        temperature=6.3 * axs.degC,
        v_init=-70.0 * axs.mV,
    ) -> None:
        hh_standard = axs.membranes.HodgkinHuxley(celsius=temperature)
        hh_hotspot = axs.membranes.HodgkinHuxley(
            celsius=temperature,
            gnabar=0.18 * axs.S_per_cm2,
            gkbar=0.045 * axs.S_per_cm2,
            gl=0.0003 * axs.S_per_cm2,
        )

        proximal = axs.axons.Section(
            "proximal HH",
            membrane=hh_standard,
            diameter=0.45 * axs.um,
            Ra=180.0 * axs.ohm_cm,
            Cm=1.0 * axs.uF_per_cm2,
            tags=("custom", "thin"),
        )
        hotspot = axs.axons.Section(
            "hotspot HH",
            membrane=hh_hotspot,
            diameter=0.75 * axs.um,
            Ra=120.0 * axs.ohm_cm,
            Cm=1.0 * axs.uF_per_cm2,
            tags=("custom", "active"),
        )
        distal = axs.axons.Section(
            "distal HH",
            membrane=hh_standard,
            diameter=1.0 * axs.um,
            Ra=100.0 * axs.ohm_cm,
            Cm=1.0 * axs.uF_per_cm2,
            tags=("custom", "wide"),
        )

        layout = axs.axons.Layout(
            [
                axs.axons.LayoutElement(
                    proximal,
                    length=300.0 * axs.um,
                    compartments=31,
                ),
                axs.axons.LayoutElement(
                    hotspot,
                    length=100.0 * axs.um,
                    compartments=11,
                ),
                axs.axons.LayoutElement(
                    distal,
                    length=600.0 * axs.um,
                    compartments=61,
                ),
            ]
        )

        super().__init__(
            layout=layout,
            formulation="single-cable",
            v_init=v_init,
            temperature=temperature,
        )

    @property
    def hotspot_center(self):
        """Center of the custom active segment."""

        return 350.0 * axs.um


def main() -> None:
    axon = CustomHeterogeneousAxon()
    sim = axs.AxonSimulation(axon)
    clamp = axs.IntracellularCurrentClamp(
        position_um=axon.hotspot_center,
        current=axs.Stimulus.pulse(
            start=1.0 * axs.ms,
            duration=0.5 * axs.ms,
            amplitude=2.0 * axs.nA,
        ),
    )
    sim.add_intracellular_context(context=clamp)

    result = axs.solvers.CrankNicholson().solve(
        sim,
        tsim=6.0 * axs.ms,
        dt=0.01 * axs.ms,
    )

    probe_positions = np.asarray([150.0, 350.0, 750.0]) * axs.um
    peak_mV = axs.results.analysis.peak_voltage(result)
    x_um = result.position_values(unit=axs.um)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax_layout, ax_traces, ax_map, ax_peak = axes.ravel()

    axon.layout.plot(
        ax=ax_layout,
        position_unit=axs.um,
        title="Custom layout",
        compartment_labels="auto",
        max_compartment_labels=60,
    )
    ax_layout.axvline(
        clamp.position_um,
        color="C3",
        linestyle="--",
        linewidth=1.2,
        label="clamp",
    )
    ax_layout.legend(frameon=False)

    for position in probe_positions:
        result.plot_trace(
            ax=ax_traces,
            position=position,
            voltage_unit=axs.mV,
            label=f"x={position.to(axs.um).magnitude:g} um",
        )
    ax_traces.set_title("Probe traces")
    ax_traces.legend(frameon=False)

    result.plot_map(
        ax=ax_map,
        voltage_unit=axs.mV,
        position_unit=axs.um,
        title="Custom axon Vm",
    )

    ax_peak.plot(x_um, peak_mV, color="C2", linewidth=2.0)
    ax_peak.set_title("Peak voltage by compartment")
    ax_peak.set_xlabel("Axon position x [um]")
    ax_peak.set_ylabel("Peak Vm [mV]")
    ax_peak.grid(True, alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()
