"""Advanced example 08: executable AxonSimulation root object.

Run:
    python examples/advanced/example_08_root_axon_simulation.py

`AxonInstance` describes one concrete axon occurrence. `AxonSimulation` is the
executable root object: it binds one or more axons/instances to duration, time
step, recording policy, and run options.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def make_instance(*, y=0.0 * axs.um, amplitude=0.8 * axs.nA) -> axs.AxonInstance:
    """Build one positioned Hodgkin-Huxley instance with a current clamp."""

    axon = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )
    instance = axs.AxonInstance(axon, y=y)
    instance.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=amplitude,
        ),
    )
    return instance


def main() -> None:
    single = axs.AxonSimulation(
        make_instance(),
        duration=1.0 * axs.ms,
        dt=0.02 * axs.ms,
        recording=axs.Recording.full(),
    )
    single_result = single.run()

    population = axs.AxonSimulation(
        [
            make_instance(y=0.0 * axs.um, amplitude=0.7 * axs.nA),
            make_instance(y=30.0 * axs.um, amplitude=0.8 * axs.nA),
            make_instance(y=60.0 * axs.um, amplitude=0.9 * axs.nA),
        ],
        duration=0.5 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )
    population_results = population.run()

    print_summary(single_result, population_results)
    plot_summary(single_result, population_results)
    plt.show()


def print_summary(
    single_result: axs.SimResult,
    population_results: list[axs.SimResult],
) -> None:
    """Print compact shape and dispatch information."""

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


def plot_summary(
    single_result: axs.SimResult,
    population_results: list[axs.SimResult],
) -> None:
    """Plot one full single trace and center traces from the population run."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    single_ax, population_ax = axes

    center_index = single_result.nearest_position_index(60.0 * axs.um)
    single_result.plot_trace(
        ax=single_ax,
        index=center_index,
        voltage_unit=axs.mV,
        title="Single AxonSimulation",
    )

    for index, result in enumerate(population_results):
        t_ms = result.time_values(unit=axs.ms)
        population_ax.plot(t_ms, np.asarray(result.Vm)[:, 0], label=f"axon {index}")
    population_ax.set_title("Population center recordings")
    population_ax.set_xlabel("Time [ms]")
    population_ax.set_ylabel("Vm [mV]")
    population_ax.grid(True, alpha=0.3)
    population_ax.legend(frameon=False)


if __name__ == "__main__":
    main()
