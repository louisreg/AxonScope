"""Advanced example 09: AxonPopulation as a first-class cohort.

Run:
    python examples/advanced/example_09_axon_population.py

`AxonPopulation` is the explicit public container for cohorts. It stores
`AxonInstance` rows, preserves order, and can contain one row when a workflow
should still use population execution.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # Step 1: build descriptive axons. They do not own stimulation.
    axon_a = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )
    axon_b = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.7 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )
    axon_c = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.9 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )

    # Step 2: wrap each axon in an instance when it needs position or stimulus.
    instance_a = axs.AxonInstance(axon_a, y=0.0 * axs.um)
    instance_b = axs.AxonInstance(axon_b, y=30.0 * axs.um)
    instance_c = axs.AxonInstance(axon_c, y=60.0 * axs.um)

    # Step 3: attach per-instance stimulation. The descriptive axons stay pure.
    instance_a.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.7 * axs.nA,
        ),
    )
    instance_b.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )
    instance_c.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.9 * axs.nA,
        ),
    )

    # Step 4: create the cohort explicitly. Order is result order.
    population = axs.AxonPopulation(
        [instance_a, instance_b, instance_c],
        name="diameter sweep",
    )

    print(f"population: {population}")
    for row, instance in enumerate(population):
        print(
            f"row {row}: diameter={instance.axon.diameter:.2f} um, "
            f"y={instance.y_um:.1f} um"
        )

    # Step 5: bind the cohort to execution settings with AxonSimulation.
    simulation = axs.AxonSimulation(
        population,
        duration=0.5 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    # Step 6: running a population returns one SimResult per population row.
    results = simulation.run()

    print("dispatch:")
    for row, result in enumerate(results):
        print(
            f"row {row}: Vm shape={np.asarray(result.Vm).shape}, "
            f"method={result.diagnostics['dispatch_method']}, "
            f"indices={result.record_indices}"
        )

    # Step 7: plot the center recording from each row.
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for row, result in enumerate(results):
        t_ms = result.time_values(unit=axs.ms)
        ax.plot(t_ms, np.asarray(result.Vm)[:, 0], label=f"row {row}")
    ax.set_title("AxonPopulation center recordings")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Vm [mV]")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.show()


if __name__ == "__main__":
    main()
