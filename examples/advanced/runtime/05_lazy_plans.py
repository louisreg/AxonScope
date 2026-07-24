"""Compose lazy simulation plans into a named study with one runner."""

from __future__ import annotations

from dataclasses import replace

import axonfleet as axs


def main() -> None:
    axon = axs.axons.HodgkinHuxley(
        length=200.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=21,
    )
    instance = axs.AxonInstance(axon)
    instance.add_current_clamp(
        position=100.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.1 * axs.ms,
            duration=0.1 * axs.ms,
            amplitude=1.0 * axs.nA,
        ),
    )

    short = axs.AxonSimulation(
        instance,
        duration=0.4 * axs.ms,
        dt=0.02 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    ).plan()
    long = replace(short, duration=0.6 * axs.ms)
    assert isinstance(short.population, axs.PopulationPlan)

    runner = axs.Runner()
    estimates = tuple(runner.estimate(plan) for plan in (short, long))
    study = axs.StudyPlan(
        name="duration_comparison",
        tasks=(
            axs.StudyTask("short", short),
            axs.StudyTask("long", long, depends_on=("short",)),
        ),
    )
    study_estimate = runner.estimate(study)
    study_inspection = runner.inspect(study)
    assert study_estimate.simulation_executions_min == 2
    assert study_estimate.peak_bytes == max(
        estimate.total_bytes for estimate in estimates
    )
    assert study_inspection.components[1].depends_on == ("short",)
    print(study_estimate.format())
    print(study_inspection.format())

    cancellation = axs.CancellationToken()
    study_result = runner.run(study, cancellation=cancellation)
    assert study_result.name == "duration_comparison"
    results = study_result.values

    for plan, estimate, result in zip(
        (short, long),
        estimates,
        results,
        strict=True,
    ):
        print(
            f"{plan.duration}: rows={plan.expected_rows}, "
            f"estimated={estimate.total_mib:.3f} MiB, Vm={result.single.Vm.shape}"
        )


if __name__ == "__main__":
    main()
