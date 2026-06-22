"""Evaluate post-hoc activation criteria.

Run:
    python examples/advanced/recording_analysis/03_activation_criterion.py

This example uses full Vm recording and evaluates activation after the solve.
It is the CPU/post-hoc companion to the solver-side `Activation` observer shown
in the VmRaster example.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import axonscope as axs


def main() -> None:
    # Step 1: record a full Vm map so post-hoc criteria can inspect any
    # compartment after the solve.
    length = 500.0 * axs.um
    clamp_x = length / 2.0
    axon = axs.axons.HodgkinHuxley(
        length=length,
        diameter=0.5 * axs.um,
        compartments=41,
        celsius=6.3 * axs.degC,
    )
    simulation = axs.AxonInstance(axon)
    simulation.add_intracellular_context(
        context=axs.IntracellularCurrentClamp(
            position=clamp_x,
            current=axs.Stimulus.pulse(
                start=1.0 * axs.ms,
                duration=0.5 * axs.ms,
                amplitude=2.0 * axs.nA,
            ),
        )
    )

    result = axs.simulate(
        simulation,
        duration=5.0 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.voltage(),
    )

    # Step 2: define criteria with the same threshold but different typed
    # position selectors.
    criteria = {
        "any compartment": axs.analysis.ActivationCriterion(
            threshold=0.0 * axs.mV,
            blanking=0.5 * axs.ms,
            target=axs.positions.ALL,
        ),
        "proximal end": axs.analysis.ActivationCriterion(
            threshold=0.0 * axs.mV,
            blanking=0.5 * axs.ms,
            target=axs.positions.PROXIMAL,
        ),
        "distal end": axs.analysis.ActivationCriterion(
            threshold=0.0 * axs.mV,
            blanking=0.5 * axs.ms,
            target=axs.positions.DISTAL,
        ),
        "clamp center": axs.analysis.ActivationCriterion(
            threshold=0.0 * axs.mV,
            blanking=0.5 * axs.ms,
            target=axs.positions.At(clamp_x),
        ),
    }

    # Step 3: evaluate each criterion on the same recorded result.
    events = {
        label: criterion.evaluate(result)
        for label, criterion in criteria.items()
    }

    print("=== Activation criteria ===")
    for label, event in events.items():
        print(
            f"{label:>15}: activated={event.activated} "
            f"first_time={event.first_time_ms} ms "
            f"first_index={event.first_index} peak={event.peak_mV:.2f} mV"
        )

    # Step 4: plot the Vm map and mark each criterion's first activation point.
    fig, (ax_map, ax_trace) = plt.subplots(
        1,
        2,
        figsize=(11, 3.8),
        constrained_layout=True,
    )
    result.plot_map(ax=ax_map, voltage_unit=axs.mV, title="Vm used by criterion")
    for label, event in events.items():
        if event.activated:
            ax_map.scatter(
                [event.first_time_ms],
                [event.first_position_um],
                s=45,
                label=label,
            )
    ax_map.legend(frameon=False)

    # Step 5: plot the traces that triggered each criterion. Criteria that did
    # not activate are skipped because they have no first index to display.
    for label, event in events.items():
        if event.first_index is None:
            continue
        t_ms, vm_mV = result.trace_values(
            index=event.first_index,
            time_unit=axs.ms,
            voltage_unit=axs.mV,
        )
        ax_trace.plot(t_ms, vm_mV, label=label)
    ax_trace.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0)
    ax_trace.set_title("Detected traces")
    ax_trace.set_xlabel("Time [ms]")
    ax_trace.set_ylabel("Vm [mV]")
    ax_trace.grid(True, alpha=0.3)
    ax_trace.legend(frameon=False)
    plt.show()


if __name__ == "__main__":
    main()
