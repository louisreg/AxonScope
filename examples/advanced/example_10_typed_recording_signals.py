"""Advanced example 10: typed recording signals.

Run:
    python examples/advanced/example_10_typed_recording_signals.py

Phase 2 replaces raw recording strings such as ``"Vm"`` with typed public
selectors from ``axs.signals``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # Step 1: build one descriptive axon.
    axon = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )

    # Step 2: attach stimulation to the concrete instance.
    instance = axs.AxonInstance(axon)
    instance.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )

    # Step 3: request signal groups with typed selectors, not strings.
    recording = axs.Recording(
        signals=[axs.signals.Vm, axs.signals.GATES],
    )

    # Step 4: run the single axon and inspect the groups that came back.
    result = axs.AxonSimulation(
        instance,
        duration=1.0 * axs.ms,
        dt=0.02 * axs.ms,
        recording=recording,
    ).run()

    print(f"recording signals: {[signal.result_key for signal in recording.signals]}")
    print(f"result groups: {tuple(result.recordings or {})}")

    # Step 5: population retention uses the same typed Vm selector.
    population = axs.AxonPopulation([instance])
    center_results = axs.AxonSimulation(
        population,
        duration=0.5 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    ).run()

    center = center_results[0]
    print(f"population Vm shape: {np.asarray(center.Vm).shape}")
    print(f"recorded indices: {center.record_indices}")

    # Step 6: plot the single-run center trace.
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    center_index = result.nearest_position_index(60.0 * axs.um)
    result.plot_trace(
        ax=ax,
        index=center_index,
        voltage_unit=axs.mV,
        title="Typed recording signals",
    )
    plt.show()


if __name__ == "__main__":
    main()
