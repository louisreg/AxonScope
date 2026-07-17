"""Retain one activation flag per axon without storing a Vm time raster.

Run:
    python examples/advanced/recording_analysis/06_compact_activation.py
"""

from __future__ import annotations

import numpy as np

import axonscope as axs


def _axon(amplitude_na: float) -> axs.AxonInstance:
    instance = axs.AxonInstance(
        axs.axons.HodgkinHuxley(
            length=120.0 * axs.um,
            diameter=0.8 * axs.um,
            compartments=31,
            celsius=6.3 * axs.degC,
        )
    )
    instance.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.05 * axs.ms,
            duration=0.10 * axs.ms,
            amplitude=amplitude_na * axs.nA,
        ),
    )
    return instance


def main() -> None:
    population = [_axon(0.0), _axon(4.0)]
    activation = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=0.20 * axs.ms,
        target=axs.positions.CENTER,
    )

    compact_simulation = axs.AxonSimulation(
        population,
        duration=2.0 * axs.ms,
        dt=0.005 * axs.ms,
        recording=axs.Recording.none(),
        observers=(activation,),
    )
    inspection = compact_simulation.inspect()
    compact = compact_simulation.run()
    if compact.observations is None:
        raise RuntimeError("compact activation simulation returned no observations")
    flags = np.asarray(compact.observations["activation"].values, dtype=bool)

    recorded = axs.AxonSimulation(
        population,
        duration=2.0 * axs.ms,
        dt=0.005 * axs.ms,
        recording=axs.Recording.voltage(),
    ).run()
    reference = np.asarray(recorded.analyze(activation).values, dtype=bool)
    np.testing.assert_array_equal(flags, reference)

    print("=== Compact activation ===")
    print(f"flags: {flags.tolist()}")
    print(f"retained shape: {inspection.probes[0].retained_shape}")
    print(f"retained bytes: {inspection.probes[0].retained_bytes}")
    print(f"Vm retained: {inspection.lowerings[0].retained_vm_width > 0}")


if __name__ == "__main__":
    main()
