"""Run VmRaster observer-only output without stored Vm traces.

Run:
    python examples/advanced/observers/01_vmraster_observer_only.py

This example uses `Recording.none()` plus solver-side threshold observers, so
the result keeps a packed VmRaster instead of a dense Vm recording. The
recording-analysis examples show the complementary post-hoc path.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


def main() -> None:
    # Step 1: build one small axon and one stimulus. The same simulation object
    # is reused for the recorded reference and the compact observer-only run.
    axon = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=31,
        celsius=6.3 * axs.degC,
    )
    simulation = axs.AxonInstance(axon)
    simulation.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.05 * axs.ms,
            duration=0.10 * axs.ms,
            amplitude=2.0 * axs.nA,
        ),
    )

    # Step 2: describe two threshold observers. Both are VmRaster-compatible and
    # therefore can run without retaining the full membrane-voltage matrix.
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        blanking=0.0 * axs.ms,
        target=axs.positions.CENTER,
    )
    latency = axs.analysis.Latency(
        threshold=-80.0 * axs.mV,
        blanking=0.0 * axs.ms,
        target=axs.positions.DISTAL,
        name="latency_distal",
    )

    # Step 3: run a normal recording as a reference.
    recorded_run = axs.AxonSimulation(
        simulation,
        duration=0.30 * axs.ms,
        dt=0.001 * axs.ms,
        recording=axs.Recording.voltage(),
    ).run()
    recorded = recorded_run.single

    # Step 4: run the compact path. This result intentionally has no `Vm`
    # recording, but it does have `observations["vm_raster"]`.
    compact_run = axs.AxonSimulation(
        simulation,
        duration=0.30 * axs.ms,
        dt=0.001 * axs.ms,
        recording=axs.Recording.none(),
        observers=[activation, latency],
    ).run()
    compact = compact_run.single
    if compact.observations is None:
        raise RuntimeError("observer-only simulation returned no observations.")
    raster = compact.observations[axs.VM_RASTER_OBSERVATION_KEY]

    # Step 5: derive the same packed raster from the recorded Vm reference.
    # The observer-only and post-hoc paths now agree on one result object and
    # one bit layout.
    derived_raster = axs.VmRasterResult.from_result(recorded, activation, latency)
    unpacked = raster.unpack()
    derived_unpacked = derived_raster.unpack()
    center_bits = unpacked[0, 0, 0, :]
    probe_indices = np.asarray(raster.probe_indices)
    if probe_indices.ndim == 3:
        probe_indices = probe_indices[0]
    center_probe_index = int(probe_indices[0, 0])
    posthoc_activation = recorded.analyze(activation)

    print("=== Recorded versus compact output ===")
    print(f"recorded Vm shape: {recorded.Vm.shape}")
    print(f"compact recordings: {compact.recordings}")
    print(f"VmRaster words shape: {np.asarray(raster.words).shape}")
    print(f"VmRaster unpacked shape: {unpacked.shape}")
    print(f"post-hoc activated: {bool(posthoc_activation.value)}")
    print(f"raster activated: {bool(np.any(center_bits))}")
    print()
    print(raster.format())

    np.testing.assert_array_equal(unpacked, derived_unpacked)

    # Step 6: plot the dense reference and the compact crossing windows together.
    # Each bar means "this observer probe was above threshold at these times".
    fig, (ax_trace, ax_raster) = plt.subplots(
        2,
        1,
        figsize=(8.0, 5.6),
        constrained_layout=True,
        sharex=True,
    )
    recorded.plot_trace(
        ax=ax_trace,
        index=center_probe_index,
        voltage_unit=axs.mV,
        label="recorded center Vm",
        title="Recorded reference",
    )
    ax_trace.axhline(-80.0, color="0.3", linestyle="--", linewidth=1.0)
    raster.plot(
        ax=ax_raster,
        time_unit=axs.ms,
        title="Threshold-crossing windows retained in VmRaster",
    )
    plt.show()


if __name__ == "__main__":
    main()
