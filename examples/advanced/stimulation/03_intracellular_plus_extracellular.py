"""Combine intracellular and extracellular stimulation on the same axon.

Run:
    python examples/advanced/stimulation/03_intracellular_plus_extracellular.py

An `AxonInstance` may carry local intracellular clamps and sampled
extracellular stimulation at the same time. The two mechanisms remain separate
public concepts: the clamp injects current at one intrinsic position, while the
extracellular drive imposes a sampled field along the whole axon.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


def main() -> None:
    # Step 1: one descriptive axon is reused to build three concrete cases.
    axon = axs.axons.RattayAberham(
        length=700.0 * axs.um,
        diameter=0.9 * axs.um,
        compartments=71,
        celsius=37.0 * axs.degC,
    )
    positions = axon.layout.position_values(unit=axs.um) * axs.um
    center = 350.0 * axs.um

    intracellular = axs.Stimulus.pulse(
        start=0.35 * axs.ms,
        duration=0.12 * axs.ms,
        amplitude=0.45 * axs.nA,
    )
    extracellular_stimulus = axs.Stimulus.biphasic(
        start=0.30 * axs.ms,
        cathodic_duration=0.12 * axs.ms,
        cathodic_amplitude=70.0 * axs.uA,
        interphase=0.04 * axs.ms,
    )
    electrode = axs.analytical.PointSourceElectrode(
        x=center,
        z=120.0 * axs.um,
    )
    extracellular = axs.analytical.point_source_stimulation(
        electrode,
        positions,
        sigma=0.3 * axs.S_per_m,
        stimulus=extracellular_stimulus,
    )

    def build_case(
        *,
        use_intracellular: bool,
        use_extracellular: bool,
    ) -> axs.AxonInstance:
        instance = axs.AxonInstance(axon)
        if use_intracellular:
            instance.add_current_clamp(position=center, current=intracellular)
        if use_extracellular:
            instance.add_extracellular_stimulation(stimulation=extracellular)
        return instance

    cases = {
        "intracellular only": build_case(
            use_intracellular=True,
            use_extracellular=False,
        ),
        "extracellular only": build_case(
            use_intracellular=False,
            use_extracellular=True,
        ),
        "combined": build_case(
            use_intracellular=True,
            use_extracellular=True,
        ),
    }

    # Step 2: execute the three cases as one population. The public result
    # surface is the same regardless of which stimulation mechanisms are present
    # on each row.
    results = axs.AxonSimulation(
        tuple(cases.values()),
        duration=1.4 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.voltage(),
    ).run()

    activation = axs.analysis.Activation(
        threshold=-20.0 * axs.mV,
        blanking=0.20 * axs.ms,
        target=axs.positions.ALL,
    )
    report = results.report(activation)
    print("=== Mixed stimulation cases ===")
    for label, value in zip(cases, report["activation"].values, strict=True):
        print(f"{label:>20}: activated={bool(value)}")

    # Step 3: plot the intracellular waveform, the extracellular field, and the
    # membrane response. The combined case does not require a special execution
    # path; it is just one AxonInstance carrying both public stimulation types.
    t = np.linspace(0.0, 1.4, 281) * axs.ms

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), constrained_layout=True)
    ax_inputs, ax_vext, ax_traces, ax_peak = axes.ravel()

    intracellular.plot(
        t,
        ax=ax_inputs,
        time_unit=axs.ms,
        amplitude_unit=axs.nA,
        label="intracellular clamp",
    )
    extracellular_stimulus.plot(
        t,
        ax=ax_inputs,
        time_unit=axs.ms,
        amplitude_unit=axs.uA,
        label="extracellular drive",
    )
    ax_inputs.set_title("Temporal inputs")
    ax_inputs.legend(frameon=False, fontsize=8)

    extracellular.plot_potential(
        t,
        ax=ax_vext,
        time_unit=axs.ms,
        position_unit=axs.um,
        voltage_unit=axs.mV,
        title="Sampled extracellular field",
    )
    ax_vext.axhline(float(center.to(axs.um).magnitude), color="black", linewidth=1.0)

    peak_values = []
    center_labels = tuple(cases)
    for result in results:
        center_index = result.nearest_position_index(center)
        peak_values.append(float(result.peak_voltage_values(unit=axs.mV)[center_index]))
    results.plot_traces(
        ax=ax_traces,
        position=center,
        labels=center_labels,
        voltage_unit=axs.mV,
        title="Center membrane response",
    )
    ax_traces.axhline(-20.0, color="0.3", linestyle="--", linewidth=1.0)

    ax_peak.bar(tuple(cases), peak_values, color=["C0", "C1", "C2"])
    ax_peak.set_title("Peak center Vm")
    ax_peak.set_ylabel("Vm [mV]")
    ax_peak.tick_params(axis="x", labelrotation=20)
    ax_peak.grid(True, axis="y", alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
