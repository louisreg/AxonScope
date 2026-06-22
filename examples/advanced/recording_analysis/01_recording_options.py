"""Compare recording retention policies and typed signal descriptors.

Run:
    python examples/advanced/recording_analysis/01_recording_options.py

This script covers the two public recording questions that belong together:
which typed signals are requested, and how much of each signal is retained.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # Step 1: inspect the public signal descriptors used below. Recording APIs
    # accept descriptors such as `axs.signals.Vm`, not raw result-key strings.
    requested_signals = (
        axs.signals.Vm,
        axs.signals.GATES,
        axs.signals.CURRENTS,
    )
    print("=== Signal descriptors ===")
    for signal in requested_signals:
        print(
            f"{signal.id.value:>16}: result_key={signal.result_key!r}, "
            f"unit={signal.unit}"
        )

    # Step 2: build one single-axon run with the full observable recording. Full
    # recording is useful while exploring a model because it keeps Vm and the
    # membrane observable groups exposed by the backend.
    full_axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )
    full_instance = axs.AxonInstance(full_axon)
    full_instance.add_current_clamp(
        position=50.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )
    full_result = axs.simulate(
        full_instance,
        duration=2.0 * axs.ms,
        dt=0.001 * axs.ms,
        recording=axs.Recording.full(),
    )

    # Step 3: run a shorter solve that explicitly asks for Vm and gates only.
    gates_axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )
    gates_instance = axs.AxonInstance(gates_axon)
    gates_instance.add_current_clamp(
        position=50.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )
    gates_only_result = axs.simulate(
        gates_instance,
        duration=0.2 * axs.ms,
        dt=0.001 * axs.ms,
        recording=axs.Recording.only(axs.signals.Vm, axs.signals.GATES),
    )

    # Step 4: build a small pool. Pool recording policies currently control the
    # retained Vm columns rather than HH gates/currents.
    pool_axon_0 = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )
    pool_axon_1 = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )
    pool_axon_2 = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )

    pool_row_0 = axs.AxonInstance(pool_axon_0)
    pool_row_0.add_current_clamp(
        position=50.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )
    pool_row_1 = axs.AxonInstance(pool_axon_1)
    pool_row_1.add_current_clamp(
        position=50.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )
    pool_row_2 = axs.AxonInstance(pool_axon_2)
    pool_row_2.add_current_clamp(
        position=50.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )
    pool = [pool_row_0, pool_row_1, pool_row_2]

    # Step 5: create four public Recording objects and inspect their plans before
    # solving. The plan is backend-neutral and answers "which columns are kept?".
    recording_modes = {
        "full": axs.Recording.voltage(),
        "center": axs.Recording.center(axs.signals.Vm),
        "probes": axs.Recording.probes(axs.signals.Vm, count=5),
        "indices": axs.Recording.indices([0, 10, 20], axs.signals.Vm),
    }
    recording_plans = {
        label: recording.to_plan()
        for label, recording in recording_modes.items()
    }

    # Step 6: run the same pool with each policy. This makes retained Vm width
    # differences visible while keeping the scientific setup fixed.
    pool_results = {
        label: axs.simulate_pool(
            pool,
            duration=0.2 * axs.ms,
            dt=0.001 * axs.ms,
            recording=recording,
        )
        for label, recording in recording_modes.items()
    }

    # Step 7: print the single-axon observable groups. Nested dictionaries are
    # observable families such as gates or currents.
    full_groups: dict[str, str | tuple[str, ...]] = {}
    for name, value in (full_result.recordings or {}).items():
        if isinstance(value, dict):
            full_groups[name] = tuple(value)
        else:
            full_groups[name] = str(np.asarray(value).shape)

    gate_groups: dict[str, str | tuple[str, ...]] = {}
    for name, value in (gates_only_result.recordings or {}).items():
        if isinstance(value, dict):
            gate_groups[name] = tuple(value)
        else:
            gate_groups[name] = str(np.asarray(value).shape)

    print("=== Single-axon observable groups ===")
    print(f"Recording.full(): {full_groups}")
    print(f"Recording.only(axs.signals.Vm, axs.signals.GATES): {gate_groups}")

    # Step 8: print how each pool recording policy lowers to retained indices.
    print("=== Backend-neutral pool RecordingPlans ===")
    for label, plan in recording_plans.items():
        print(
            f"{label:>7}: indices={plan.indices_for(21)} "
            f"width={plan.width_for(21)}"
        )

    print("=== Pool Vm recording widths ===")
    for label, results in pool_results.items():
        first = results[0]
        print(
            f"{label:>7}: Vm shape={np.asarray(first.Vm).shape} "
            f"record_indices={first.record_indices}"
        )

    # Step 9: plot Vm, observable groups, and the retained pool widths together.
    center_index = full_result.nearest_position_index(50.0 * axs.um)
    t_ms = full_result.time_values(unit=axs.ms)
    recordings = full_result.recordings or {}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax_vm, ax_gates, ax_currents, ax_widths = axes.ravel()

    full_result.plot_trace(
        ax=ax_vm,
        index=center_index,
        voltage_unit=axs.mV,
        title="Center Vm",
    )

    for name, values in recordings.get("gates", {}).items():
        ax_gates.plot(t_ms, np.asarray(values)[:, center_index], label=name)
    ax_gates.set_title("Center gates")
    ax_gates.set_xlabel("Time [ms]")
    ax_gates.set_ylabel("Gate value")
    ax_gates.grid(True, alpha=0.3)
    ax_gates.legend(frameon=False)

    for name, values in recordings.get("currents", {}).items():
        ax_currents.plot(t_ms, np.asarray(values)[:, center_index], label=name)
    ax_currents.set_title("Center current densities")
    ax_currents.set_xlabel("Time [ms]")
    ax_currents.set_ylabel("Current density [mA/cm2]")
    ax_currents.grid(True, alpha=0.3)
    ax_currents.legend(frameon=False)

    labels = tuple(pool_results)
    widths = [np.asarray(results[0].Vm).shape[1] for results in pool_results.values()]
    ax_widths.bar(labels, widths, color=["0.45", "C0", "C2", "C3"])
    ax_widths.set_title("Pool retained Vm columns")
    ax_widths.set_ylabel("Recorded compartments")
    ax_widths.set_ylim(0, max(widths) + 2)
    ax_widths.grid(True, axis="y", alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
