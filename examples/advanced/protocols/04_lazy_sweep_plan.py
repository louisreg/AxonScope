"""Build a generic sweep plan, then execute it through one runner."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


def main() -> None:
    axon = axs.axons.HodgkinHuxley(
        length=500.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=41,
    )
    amplitudes = np.linspace(0.0, 4.0, 9) * axs.nA

    def with_current(source: axs.AxonInstance, amplitude):
        row = axs.AxonInstance(source.axon)
        row.add_current_clamp(
            position=250.0 * axs.um,
            current=axs.Stimulus.pulse(
                start=0.2 * axs.ms,
                duration=0.2 * axs.ms,
                amplitude=amplitude,
            ),
        )
        return row

    plan = axs.protocols.pool_sweep_plan(
        (axon,),
        update=with_current,
        values=amplitudes,
        observe=lambda result: float(np.max(result.voltage_values(unit=axs.mV))),
        duration=1.0 * axs.ms,
        dt=0.02 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
        batch_options=None,
        execution_policy=None,
    )

    runner = axs.Runner()
    estimate = runner.estimate(plan.source)
    sweep = runner.run(plan)

    print(
        f"kind={plan.plan_kind}, rows={plan.expected_rows}, "
        f"source_estimate={estimate.total_mib:.3f} MiB"
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    sweep.plot(ax=ax, value_unit=axs.nA, marker="o")
    ax.set(xlabel="Current (nA)", ylabel="Peak center Vm (mV)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
