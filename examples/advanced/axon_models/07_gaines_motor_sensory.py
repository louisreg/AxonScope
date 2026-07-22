"""Compare the Gaines motor and sensory myelinated axon families.

Run:
    python examples/advanced/axon_models/07_gaines_motor_sensory.py

Both models reuse the MRG-like double-cable geometry. Their nodal and
internodal membrane parameters differ, producing distinct action-potential
shapes and conduction velocities under the same intracellular pulse.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import axonscope as axs


def run_axon(axon):
    proximal = axon.node_position("proximal", unit=axs.um)
    simulation = axs.AxonInstance(axon)
    simulation.add_intracellular_context(
        context=axs.IntracellularCurrentClamp(
            position=proximal,
            current=axs.Stimulus.pulse(
                start=1.0 * axs.ms,
                duration=0.1 * axs.ms,
                amplitude=5.0 * axs.nA,
            ),
        )
    )
    return axs.AxonSimulation(
        simulation,
        duration=4.0 * axs.ms,
        dt=0.005 * axs.ms,
    ).run().single


def main() -> None:
    motor = axs.axons.GainesMotor(diameter=10.0 * axs.um, nodes=21)
    sensory = axs.axons.GainesSensory(diameter=10.0 * axs.um, nodes=21)
    motor_result = run_axon(motor)
    sensory_result = run_axon(sensory)

    distal_motor = motor.node_position(-2, unit=axs.um)
    distal_sensory = sensory.node_position(-2, unit=axs.um)
    velocity = axs.analysis.ConductionVelocity(threshold=0.0 * axs.mV)
    motor_velocity = float(motor_result.analyze(velocity).values[0])
    sensory_velocity = float(sensory_result.analyze(velocity).values[0])
    print(f"Gaines motor velocity:   {motor_velocity:.2f} m/s")
    print(f"Gaines sensory velocity: {sensory_velocity:.2f} m/s")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    motor_result.plot_trace(
        ax=axes[0, 0],
        position=distal_motor,
        voltage_unit=axs.mV,
        title="Motor distal node",
    )
    sensory_result.plot_trace(
        ax=axes[0, 1],
        position=distal_sensory,
        voltage_unit=axs.mV,
        title="Sensory distal node",
    )
    motor_result.plot_map(
        ax=axes[1, 0],
        position_unit=axs.mm,
        voltage_unit=axs.mV,
        title="Motor propagation",
    )
    sensory_result.plot_map(
        ax=axes[1, 1],
        position_unit=axs.mm,
        voltage_unit=axs.mV,
        title="Sensory propagation",
    )
    plt.show()


if __name__ == "__main__":
    main()
