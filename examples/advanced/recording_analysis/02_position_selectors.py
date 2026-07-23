"""Use typed position selectors.

Run:
    python examples/advanced/recording_analysis/02_position_selectors.py

Position selectors are public value objects. Use named selectors such as
`axs.positions.PROXIMAL`, `axs.positions.CENTER`, and `axs.positions.DISTAL`,
or explicit physical/index selectors when an analysis needs a precise target.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


def main() -> None:
    # Step 1: build one axon and stimulate near the middle.
    axon = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )
    instance = axs.AxonInstance(axon)
    instance.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.9 * axs.nA,
        ),
    )

    # Step 2: record Vm everywhere so post-hoc position selectors can compare.
    run = axs.AxonSimulation(
        instance,
        duration=1.0 * axs.ms,
        dt=0.02 * axs.ms,
        recording=axs.Recording.voltage(),
    ).run()
    result = run.single

    # Step 3: evaluate the same threshold at different typed targets.
    criteria = {
        "anywhere": axs.analysis.Activation(
            threshold=-20.0 * axs.mV,
            blanking=0.1 * axs.ms,
            target=axs.positions.ALL,
        ),
        "proximal": axs.analysis.Activation(
            threshold=-20.0 * axs.mV,
            blanking=0.1 * axs.ms,
            target=axs.positions.PROXIMAL,
        ),
        "center": axs.analysis.Activation(
            threshold=-20.0 * axs.mV,
            blanking=0.1 * axs.ms,
            target=axs.positions.CENTER,
        ),
        "distal": axs.analysis.Activation(
            threshold=-20.0 * axs.mV,
            blanking=0.1 * axs.ms,
            target=axs.positions.DISTAL,
        ),
        "near clamp": axs.analysis.Activation(
            threshold=-20.0 * axs.mV,
            blanking=0.1 * axs.ms,
            target=axs.positions.At(60.0 * axs.um),
        ),
    }

    # Step 4: print compact event summaries.
    for label, criterion in criteria.items():
        event = criterion.detect(result)
        print(
            f"{label}: activated={event.activated}, "
            f"index={event.first_index}, "
            f"position={event.first_position_um} um"
        )

    # Step 5: draw the selector locations on the same one-dimensional fiber.
    # This is the mental model: selectors resolve against recorded result
    # columns, not against hidden solver arrays.
    positions_um = result.position_values(unit=axs.um)
    original_indices = (
        np.arange(positions_um.shape[0], dtype=int)
        if result.record_indices is None
        else np.asarray(result.record_indices, dtype=int)
    )
    selected_columns = {
        label: criterion.target.columns(
            positions_um=positions_um,
            original_indices=original_indices,
        )
        for label, criterion in criteria.items()
    }

    fig, (ax_fiber, ax_traces) = plt.subplots(
        2,
        1,
        figsize=(8.5, 5.8),
        constrained_layout=True,
        sharex=False,
    )
    result.plot_recorded_axis(
        ax=ax_fiber,
        selectors={label: criterion.target for label, criterion in criteria.items()},
        markers={"clamp": 60.0 * axs.um},
        position_unit=axs.um,
        title="Selector targets on recorded Vm columns",
    )

    trace_labels = ("proximal", "center", "distal", "near clamp")
    result.plot_traces(
        ax=ax_traces,
        indices=tuple(int(selected_columns[label][0]) for label in trace_labels),
        labels=trace_labels,
        voltage_unit=axs.mV,
        title="Recorded traces at selected positions",
    )
    ax_traces.axhline(-20.0, color="0.3", linestyle="--", linewidth=1.0)
    plt.show()


if __name__ == "__main__":
    main()
