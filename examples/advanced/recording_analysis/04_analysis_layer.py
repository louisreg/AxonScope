"""Build a structured post-hoc analysis report.

Run:
    python examples/advanced/recording_analysis/04_analysis_layer.py

`axs.analysis` definitions declare the recorded signals and positions they need,
then return per-fiber values, statuses, messages, and population denominators.
This example keeps Vm traces for fibers with several diameters and evaluates a
shared set of analyses after the solve.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


def main() -> None:
    # Step 1: create a small fiber population. Each row is a real fiber with
    # its own diameter; the stimulation is kept fixed so the analysis report is
    # indexed by fiber properties instead of by protocol sweep values.
    diameters_um = np.asarray([0.5, 0.8, 1.2])
    pool: list[axs.AxonInstance] = []
    for diameter_um in diameters_um:
        axon_model = axs.axons.HodgkinHuxley(
            length=180.0 * axs.um,
            diameter=float(diameter_um) * axs.um,
            compartments=31,
            celsius=6.3 * axs.degC,
        )
        simulation = axs.AxonInstance(axon_model)
        simulation.add_current_clamp(
            position=90.0 * axs.um,
            current=axs.Stimulus.pulse(
                start=0.20 * axs.ms,
                duration=0.30 * axs.ms,
                amplitude=0.90 * axs.nA,
            ),
        )
        pool.append(simulation)

    # Step 2: post-hoc analyses need recorded Vm at the requested positions.
    # Full Vm is the most flexible choice while designing an analysis workflow.
    results = axs.AxonSimulation(
        pool,
        duration=2.0 * axs.ms,
        dt=0.02 * axs.ms,
        recording=axs.Recording.voltage(),
    ).run()

    # Step 3: define analyses with explicit scientific targets. Activation is a
    # boolean, latency returns the first crossing time, and peak voltage reports a
    # scalar maximum over all recorded positions.
    activation = axs.analysis.Activation(
        threshold=-20.0 * axs.mV,
        blanking=0.15 * axs.ms,
        target=axs.positions.DISTAL,
    )
    latency = axs.analysis.Latency(
        threshold=-20.0 * axs.mV,
        blanking=0.15 * axs.ms,
        target=axs.positions.DISTAL,
    )
    peak = axs.analysis.PeakVoltage(target=axs.positions.ALL)

    # Step 4: requirements are inspectable before evaluation, which is useful
    # when a notebook switches between retained Vm and compact observer output.
    required = [signal.id.value for signal in activation.requirements.required_signals]
    print("=== Analysis requirements ===")
    print(f"activation requires signals: {required}")
    print(f"activation positions: {activation.requirements.required_positions}")
    print(f"activation algorithm: {activation.requirements.algorithm_version}")

    # Step 5: one report keeps related metrics aligned by row.
    report = results.report(activation, latency, peak)

    print("\n=== Analysis report ===")
    print(report.format())

    # Step 6: plot center traces and the analysis values side by side.
    fig, (ax_traces, ax_metrics) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.0),
        constrained_layout=True,
    )
    results.plot_traces(
        ax=ax_traces,
        position=90.0 * axs.um,
        labels=tuple(f"d={diameter:.1f} um" for diameter in diameters_um),
        voltage_unit=axs.mV,
        title="Center Vm traces by fiber diameter",
    )
    ax_traces.axhline(-20.0, color="0.3", linestyle="--", linewidth=1.0)

    report.plot(ax=ax_metrics)
    ax_metrics.set_title("Report values by fiber diameter")
    ax_metrics.set_xticks(
        np.arange(len(diameters_um)),
        [f"{value:.1f}" for value in diameters_um],
    )
    ax_metrics.set_xlabel("Fiber diameter [um]")
    plt.show()


if __name__ == "__main__":
    main()
