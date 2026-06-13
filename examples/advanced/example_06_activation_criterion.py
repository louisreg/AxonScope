"""Advanced example 06: post-hoc activation criteria.

Run:
    python examples/advanced/example_06_activation_criterion.py

This example uses full Vm recording and evaluates activation after the solve.
It is the CPU/post-hoc companion to the future GPU on-the-fly
`ActivationObserver`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    length = 500.0 * axs.um
    clamp_x = length / 2.0
    axon = axs.axons.HodgkinHuxley(
        length=length,
        diameter=0.5 * axs.um,
        compartments=41,
        celsius=6.3 * axs.degC,
    )
    sim = axs.AxonSimulation(axon)
    sim.add_intracellular_context(
        context=axs.IntracellularCurrentClamp(
            position_um=clamp_x,
            current=axs.Stimulus.pulse(
                start=1.0 * axs.ms,
                duration=0.5 * axs.ms,
                amplitude=2.0 * axs.nA,
            ),
        )
    )

    result = axs.simulate(
        sim,
        duration_ms=5.0 * axs.ms,
        dt_ms=0.01 * axs.ms,
        recording=axs.Recording.voltage(),
    )

    criteria = {
        "any compartment": axs.results.ActivationCriterion(
            threshold=0.0 * axs.mV,
            blanking=0.5 * axs.ms,
            positions="all",
        ),
        "distal end": axs.results.ActivationCriterion(
            threshold=0.0 * axs.mV,
            blanking=0.5 * axs.ms,
            positions="distal",
        ),
        "clamp center": axs.results.ActivationCriterion(
            threshold=0.0 * axs.mV,
            blanking=0.5 * axs.ms,
            positions=[clamp_x],
        ),
    }
    events = {label: criterion.evaluate(result) for label, criterion in criteria.items()}
    print_activation_summary(events)
    plot_activation_summary(result, events)
    plt.show()


def print_activation_summary(events: dict[str, axs.results.ActivationEvent]) -> None:
    """Print compact activation events."""

    print("=== Activation criteria ===")
    for label, event in events.items():
        print(
            f"{label:>15}: activated={event.activated} "
            f"first_time={event.first_time_ms} ms "
            f"first_index={event.first_index} peak={event.peak_mV:.2f} mV"
        )


def plot_activation_summary(
    result: axs.SimResult,
    events: dict[str, axs.results.ActivationEvent],
) -> None:
    """Plot Vm map and mark first activation events."""

    fig, (ax_map, ax_trace) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
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


if __name__ == "__main__":
    main()
