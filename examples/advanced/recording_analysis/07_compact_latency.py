"""Retain first-crossing latency without storing a Vm time raster.

Run:
    python examples/advanced/recording_analysis/07_compact_latency.py
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
    latency = axs.analysis.Latency(
        threshold=0.0 * axs.mV,
        blanking=0.20 * axs.ms,
        target=axs.positions.CENTER,
    )

    compact_simulation = axs.AxonSimulation(
        population,
        duration=2.0 * axs.ms,
        dt=0.005 * axs.ms,
        recording=axs.Recording.none(),
        observers=(latency,),
    )
    inspection = compact_simulation.inspect()
    compact = compact_simulation.run()
    if compact.observations is None:
        raise RuntimeError("compact latency simulation returned no observations")
    values = np.asarray(compact.observations["latency"].values, dtype=float)

    recorded = axs.AxonSimulation(
        population,
        duration=2.0 * axs.ms,
        dt=0.005 * axs.ms,
        recording=axs.Recording.voltage(),
    ).run()
    reference = np.asarray(recorded.analyze(latency).values, dtype=float)
    np.testing.assert_allclose(values, reference, equal_nan=True)

    print("=== Compact latency ===")
    print(f"latency (ms): {values.tolist()}")
    print(f"retained shape: {inspection.probes[0].retained_shape}")
    print(f"retained bytes: {inspection.probes[0].retained_bytes}")
    print(f"Vm retained: {inspection.lowerings[0].retained_vm_width > 0}")


if __name__ == "__main__":
    main()
