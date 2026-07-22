"""Compare the validated Sundt, Tigerholm, and Schild axon families.

Run:
    python examples/advanced/axon_models/09_validated_unmyelinated_families.py

These configurations mirror the focused intracellular NRV validation cases.
Each model keeps its own published membrane dynamics, initial voltage, cable
geometry, stimulus, time step, and simulation duration.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


@dataclass(frozen=True)
class ModelCase:
    name: str
    axon: object
    clamp_start: object
    clamp_duration: object
    clamp_amplitude: object
    duration: object
    dt: object


def cases() -> tuple[ModelCase, ...]:
    return (
        ModelCase(
            "Sundt",
            axs.axons.Sundt(
                length=1_000.0 * axs.um,
                diameter=0.5 * axs.um,
                compartments=101,
                celsius=37.0 * axs.degC,
            ),
            1.0 * axs.ms,
            1.0 * axs.ms,
            2.0 * axs.nA,
            10.0 * axs.ms,
            0.001 * axs.ms,
        ),
        ModelCase(
            "Tigerholm",
            axs.axons.Tigerholm(
                length=5_000.0 * axs.um,
                diameter=1.0 * axs.um,
                compartments=101,
                celsius=37.0 * axs.degC,
            ),
            5.0 * axs.ms,
            1.0 * axs.ms,
            2.0 * axs.nA,
            30.0 * axs.ms,
            0.025 * axs.ms,
        ),
        ModelCase(
            "Schild 1994",
            axs.axons.Schild94(
                length=3_000.0 * axs.um,
                diameter=0.8 * axs.um,
                compartments=51,
            ),
            2.0 * axs.ms,
            1.0 * axs.ms,
            1.0 * axs.nA,
            20.0 * axs.ms,
            0.01 * axs.ms,
        ),
        ModelCase(
            "Schild 1997",
            axs.axons.Schild97(
                length=3_000.0 * axs.um,
                diameter=0.8 * axs.um,
                compartments=51,
            ),
            2.0 * axs.ms,
            1.0 * axs.ms,
            1.0 * axs.nA,
            20.0 * axs.ms,
            0.01 * axs.ms,
        ),
    )


def run(case: ModelCase):
    center = 0.5 * case.axon.length * axs.um
    simulation = axs.AxonInstance(case.axon)
    simulation.add_intracellular_context(
        context=axs.IntracellularCurrentClamp(
            position=center,
            current=axs.Stimulus.pulse(
                start=case.clamp_start,
                duration=case.clamp_duration,
                amplitude=case.clamp_amplitude,
            ),
        )
    )
    result = axs.AxonSimulation(
        simulation,
        duration=case.duration,
        dt=case.dt,
    ).run().single
    return center, result


def main() -> None:
    model_cases = cases()
    completed = tuple((case, *run(case)) for case in model_cases)

    fig, axes = plt.subplots(
        len(completed),
        2,
        figsize=(12, 10),
        constrained_layout=True,
    )
    for row, (case, center, result) in enumerate(completed):
        _, center_vm = result.trace_values(position=center, voltage_unit=axs.mV)
        print(f"{case.name:12s} center peak: {np.max(center_vm):7.2f} mV")
        result.plot_trace(
            ax=axes[row, 0],
            position=center,
            voltage_unit=axs.mV,
            title=f"{case.name}: center trace",
        )
        result.plot_map(
            ax=axes[row, 1],
            position_unit=axs.mm,
            voltage_unit=axs.mV,
            title=f"{case.name}: propagation",
        )
    plt.show()


if __name__ == "__main__":
    main()
