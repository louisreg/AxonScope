"""Compare recording retention policies and typed signal descriptors.

Run:
    python examples/advanced/recording_analysis/01_recording_options.py

This script covers the public recording questions that belong together: which
typed signals are requested, which spatial columns are retained, and where
probe/index recordings sit on the fiber.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # Step 1: inspect the public signal descriptors used below. Recording APIs
    # accept descriptors such as `axs.signals.Vm`, not raw result-key strings.
    requested_signals = (axs.signals.Vm,)
    print("=== Signal descriptors ===")
    for signal in requested_signals:
        print(
            f"{signal.id.value:>16}: result_key={signal.result_key!r}, "
            f"unit={signal.unit}"
        )

    # Step 2: build one single-axon run with retained Vm. One-row simulations
    # use the same batch route as populations, so Vm recording policies behave
    # consistently for B=1 and B>1.
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
    voltage_run = axs.AxonSimulation(
        full_instance,
        duration=2.0 * axs.ms,
        dt=0.001 * axs.ms,
        recording=axs.Recording.voltage(),
    ).run()
    voltage_result = voltage_run.single

    # Step 3: build a small pool. Pool recording policies control the retained
    # Vm columns.
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

    # Step 4: create four public Recording objects and inspect their plans before
    # solving. The plan is backend-neutral and answers "which columns are kept?".
    # Recording.none() is shown in ../observers/01_vmraster_observer_only.py
    # because it is
    # useful when solver-side observers replace stored Vm traces.
    recording_modes = {
        "voltage": axs.Recording.voltage(),
        "center": axs.Recording.center(axs.signals.Vm),
        "probes": axs.Recording.probes(axs.signals.Vm, count=5),
        "indices": axs.Recording.indices([0, 10, 20], axs.signals.Vm),
    }
    recording_plans = {
        label: recording.to_plan()
        for label, recording in recording_modes.items()
    }

    # Step 5: run the same pool with each policy. This makes retained Vm width
    # differences visible while keeping the scientific setup fixed.
    pool_results = {
        label: axs.AxonSimulation(
            pool,
            duration=0.2 * axs.ms,
            dt=0.001 * axs.ms,
            recording=recording,
        ).run()
        for label, recording in recording_modes.items()
    }

    # Step 6: print the single-axon retained outputs.
    voltage_groups: dict[str, str | tuple[str, ...]] = {}
    for name, value in (voltage_result.recordings or {}).items():
        if isinstance(value, dict):
            voltage_groups[name] = tuple(value)
        else:
            voltage_groups[name] = str(np.asarray(value).shape)

    print("=== Single-axon retained outputs ===")
    print(f"Recording.voltage(): {voltage_groups}")

    # Step 7: print how each pool recording policy lowers to retained indices.
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

    # Step 8: plot Vm, retained widths, and retained positions together. The
    # probe plot is the important check: "probes" means
    # evenly spaced compartment indices, while "indices" means the exact
    # original compartment indices supplied by the user.
    center_index = voltage_result.nearest_position_index(50.0 * axs.um)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax_vm, ax_widths, ax_locations, ax_pool_trace = axes.ravel()

    voltage_result.plot_trace(
        ax=ax_vm,
        index=center_index,
        voltage_unit=axs.mV,
        title="Center Vm",
    )

    labels = tuple(pool_results)
    widths = [np.asarray(results[0].Vm).shape[1] for results in pool_results.values()]
    ax_widths.bar(labels, widths, color=["0.45", "C0", "C2", "C3"])
    ax_widths.set_title("Pool retained Vm columns")
    ax_widths.set_ylabel("Recorded compartments")
    ax_widths.set_ylim(0, max(widths) + 2)
    ax_widths.grid(True, axis="y", alpha=0.3)

    axs.results.plot_recorded_axes(
        pool_results,
        ax=ax_locations,
        position_unit=axs.um,
        title="Where Vm is retained",
    )

    for label, results in pool_results.items():
        first = results[0]
        first.plot_trace(
            ax=ax_pool_trace,
            index=np.asarray(first.Vm).shape[1] // 2,
            voltage_unit=axs.mV,
            label=label,
            title="First pool row, retained trace",
        )
    ax_pool_trace.legend(frameon=False, fontsize=8)
    plt.show()


if __name__ == "__main__":
    main()
