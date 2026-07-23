"""Use AxonSimulation as an executable root object.

Run:
    python examples/advanced/simulation_workflow/01_axon_simulation_root.py

`AxonInstance` describes one concrete axon occurrence. `AxonSimulation` is the
executable root object: it binds one or more axons/instances to duration, time
step, recording policy, and run options.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


def main() -> None:
    # Step 1: build one descriptive Hodgkin-Huxley axon. It has geometry and
    # membrane dynamics, but no position, stimulus, or runtime settings yet.
    single_axon = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )

    # Step 2: wrap the description in an AxonInstance and attach stimulation to
    # that concrete occurrence.
    single_instance = axs.AxonInstance(single_axon)
    single_instance.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )

    # Step 3: AxonSimulation is the executable root for a single row too. It is
    # where duration, dt, recording, and later runtime policy live.
    single_simulation = axs.AxonSimulation(
        single_instance,
        duration=1.0 * axs.ms,
        dt=0.02 * axs.ms,
        recording=axs.Recording.voltage(),
    )
    single_run = single_simulation.run()
    single_result = single_run.single

    # Step 4: build three rows for a population run. They share the
    # same public model class, but each row owns its own stimulus amplitude.
    population_axon_0 = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )
    population_axon_1 = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )
    population_axon_2 = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )

    population_row_0 = axs.AxonInstance(population_axon_0)
    population_row_0.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.7 * axs.nA,
        ),
    )
    population_row_1 = axs.AxonInstance(population_axon_1)
    population_row_1.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )
    population_row_2 = axs.AxonInstance(population_axon_2)
    population_row_2.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.9 * axs.nA,
        ),
    )

    # Step 5: the same AxonSimulation root can execute several rows. Here the
    # recording policy keeps only the center Vm trace for each row.
    population_simulation = axs.AxonSimulation(
        [population_row_0, population_row_1, population_row_2],
        duration=0.5 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )
    population_results = population_simulation.run()

    # Step 6: inspect what the two roots returned. Both roots return
    # AxonSimulationResult; `.single` extracts the one-row view.
    print("=== Single executable root ===")
    print(f"Vm shape: {np.asarray(single_result.Vm).shape}")
    print(f"recording groups: {tuple(single_result.recordings or {})}")

    print("=== Population executable root ===")
    for index, result in enumerate(population_results):
        print(
            f"{index}: Vm shape={np.asarray(result.Vm).shape}, "
            f"method={result.diagnostics['dispatch_method']}, "
            f"record_indices={result.record_indices}"
        )

    # Step 7: plot the full single-row trace next to the retained center traces
    # from the population run.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    single_ax, population_ax = axes

    center_index = single_result.nearest_position_index(60.0 * axs.um)
    single_result.plot_trace(
        ax=single_ax,
        index=center_index,
        voltage_unit=axs.mV,
        title="Single AxonSimulation",
    )

    population_results.plot_traces(
        ax=population_ax,
        index=0,
        labels=tuple(f"axon {index}" for index in range(len(population_results))),
        voltage_unit=axs.mV,
        title="Population center recordings",
    )
    plt.show()


if __name__ == "__main__":
    main()
