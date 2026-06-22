"""Build a structured post-hoc analysis report.

Run:
    python examples/advanced/recording_analysis/04_analysis_layer.py

`axs.analysis` definitions declare the recorded signals and positions they need,
then return per-row values, statuses, messages, and population denominators.
This example keeps Vm traces and evaluates several analyses after the solve.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # Step 1: create a small dose-response population. The rows share the same
    # axon geometry; only the intracellular pulse amplitude changes.
    axon_model = axs.axons.HodgkinHuxley(
        length=180.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=31,
        celsius=6.3 * axs.degC,
    )
    amplitudes_nA = np.asarray([0.05, 0.35, 0.90])
    pool: list[axs.AxonInstance] = []
    for amplitude_nA in amplitudes_nA:
        simulation = axs.AxonInstance(axon_model)
        simulation.add_current_clamp(
            position=90.0 * axs.um,
            current=axs.Stimulus.pulse(
                start=0.20 * axs.ms,
                duration=0.30 * axs.ms,
                amplitude=float(amplitude_nA) * axs.nA,
            ),
        )
        pool.append(simulation)

    # Step 2: post-hoc analyses need recorded Vm at the requested positions.
    # Full Vm is the most flexible choice while designing an analysis workflow.
    result = axs.simulate_pool(
        pool,
        duration=2.0 * axs.ms,
        dt=0.02 * axs.ms,
        recording=axs.Recording.voltage(),
    )

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
    report = result.report(activation, latency, peak)
    activation_result = report["activation"]
    latency_result = report["latency"]
    peak_result = report["peak_voltage"]

    print("\n=== Population denominators ===")
    for analysis_result in report:
        population = analysis_result.population
        print(
            f"{analysis_result.name:>12}: total={population.n_total}, "
            f"valid={population.n_valid}, failed={population.n_failed}"
        )

    print("\n=== Per-row analysis report ===")
    for row_index, amplitude_nA in enumerate(amplitudes_nA):
        activated = bool(activation_result.values[row_index])
        latency_ms = float(latency_result.values[row_index])
        latency_text = "n/a" if np.isnan(latency_ms) else f"{latency_ms:.3f} ms"
        peak_mV = float(peak_result.values[row_index])
        print(
            f"row {row_index}, I={amplitude_nA:.2f} nA: "
            f"activated={activated}, latency={latency_text}, peak={peak_mV:.2f} mV, "
            f"latency_status={latency_result.statuses[row_index].value}"
        )

    # Step 6: plot center traces and the analysis values side by side.
    fig, (ax_traces, ax_metrics) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.0),
        constrained_layout=True,
    )
    for row_index, row in enumerate(result):
        center_index = row.nearest_position_index(90.0 * axs.um)
        t_ms, vm_mV = row.trace_values(
            index=center_index,
            time_unit=axs.ms,
            voltage_unit=axs.mV,
        )
        ax_traces.plot(t_ms, vm_mV, label=f"{amplitudes_nA[row_index]:.2f} nA")
    ax_traces.axhline(-20.0, color="0.3", linestyle="--", linewidth=1.0)
    ax_traces.set_title("Center Vm traces")
    ax_traces.set_xlabel("Time [ms]")
    ax_traces.set_ylabel("Vm [mV]")
    ax_traces.grid(True, alpha=0.3)
    ax_traces.legend(title="Pulse")

    x = np.arange(len(amplitudes_nA))
    ax_metrics.bar(
        x - 0.18,
        peak_result.values.astype(float),
        width=0.36,
        label="peak Vm [mV]",
    )
    latency_plot = np.nan_to_num(latency_result.values.astype(float), nan=0.0)
    ax_metrics.bar(
        x + 0.18,
        latency_plot,
        width=0.36,
        label="latency [ms]",
    )
    for row_index, status in enumerate(latency_result.statuses):
        if status.value != "VALID":
            ax_metrics.text(row_index + 0.18, 0.05, status.value, rotation=90, fontsize=8)
    ax_metrics.set_title("Report values")
    ax_metrics.set_xticks(x, [f"{value:.2f}" for value in amplitudes_nA])
    ax_metrics.set_xlabel("Current [nA]")
    ax_metrics.grid(True, axis="y", alpha=0.3)
    ax_metrics.legend(frameon=False)
    plt.show()


if __name__ == "__main__":
    main()
