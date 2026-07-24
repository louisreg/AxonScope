"""Inspect and execute a named study through one reusable runner."""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt

import axonfleet as axs


def main() -> None:
    axon = axs.axons.HodgkinHuxley(
        length=500.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=41,
    )
    instance = axs.AxonInstance(axon)
    instance.add_current_clamp(
        position=250.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.2 * axs.ms,
            duration=0.2 * axs.ms,
            amplitude=2.0 * axs.nA,
        ),
    )

    baseline = axs.AxonSimulation(
        instance,
        duration=1.0 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    ).plan()
    extended = replace(baseline, duration=1.5 * axs.ms)
    study = axs.StudyPlan(
        name="duration_comparison",
        tasks=(
            axs.StudyTask("baseline", baseline),
            axs.StudyTask("extended", extended, depends_on=("baseline",)),
        ),
    )

    runner = axs.Runner()
    print(runner.estimate(study).format())
    print(runner.inspect(study).format())
    results = runner.run(study)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for key in results.keys:
        result = results[key].single
        result.plot_trace(
            ax=ax,
            index=0,
            time_unit=axs.ms,
            voltage_unit=axs.mV,
            label=key,
        )
    ax.set(xlabel="Time (ms)", ylabel="Center Vm (mV)")
    ax.legend()
    ax.grid(alpha=0.25)

    cancellation = axs.CancellationToken()
    cancellation.cancel()
    try:
        runner.run(study, cancellation=cancellation)
    except axs.PlanCancelledError as error:
        print(f"cancelled with pending tasks: {error.pending_keys}")

    plt.show()


if __name__ == "__main__":
    main()
