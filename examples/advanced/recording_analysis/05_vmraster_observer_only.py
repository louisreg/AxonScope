"""Run VmRaster observer-only output without stored Vm traces.

Run:
    python examples/advanced/recording_analysis/05_vmraster_observer_only.py

The previous example keeps Vm and evaluates analyses post-hoc. This one uses
`Recording.none()` plus solver-side threshold observers, so the result keeps a
packed VmRaster instead of a dense Vm recording.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


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
            amplitude=0.8 * axs.nA,
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
    recorded = axs.simulate(
        simulation,
        duration=0.30 * axs.ms,
        dt=0.001 * axs.ms,
        recording=axs.Recording.voltage(),
    )

    # Step 4: run the compact path. This result intentionally has no `Vm`
    # recording, but it does have `observations["vm_raster"]`.
    compact = axs.simulate(
        simulation,
        duration=0.30 * axs.ms,
        dt=0.001 * axs.ms,
        recording=axs.Recording.none(),
        observers=[activation, latency],
    )
    if compact.observations is None:
        raise RuntimeError("observer-only simulation returned no observations.")
    raster = compact.observations[axs.VM_RASTER_OBSERVATION_KEY]

    # Step 5: compare the first observer slot with the equivalent post-hoc
    # threshold on the recorded reference.
    unpacked = raster.unpack()
    center_bits = unpacked[0, 0, 0, :]
    center_probe_index = int(np.asarray(raster.original_indices)[0, 0])
    reference_bits = np.asarray(recorded.Vm)[:, center_probe_index] >= -80.0
    posthoc_activation = recorded.analyze(activation)

    print("=== Recorded versus compact output ===")
    print(f"recorded Vm shape: {recorded.Vm.shape}")
    print(f"compact recordings: {compact.recordings}")
    print(f"VmRaster words shape: {np.asarray(raster.words).shape}")
    print(f"VmRaster unpacked shape: {unpacked.shape}")
    print(f"post-hoc activated: {bool(posthoc_activation.value)}")
    print(f"raster activated: {bool(np.any(center_bits))}")

    np.testing.assert_array_equal(center_bits, reference_bits)

    # Step 6: plot the dense reference and the compact crossing windows together.
    # Each bar means "this observer probe was above threshold at these times".
    t_ms = recorded.time_values(unit=axs.ms)
    center_trace = np.asarray(recorded.Vm)[:, center_probe_index]
    raster_rows = unpacked[0].reshape(unpacked.shape[1] * unpacked.shape[2], raster.nt)
    original_indices = np.asarray(raster.original_indices)
    if original_indices.ndim == 3:
        original_indices = original_indices[0]
    raster_labels = []
    for definition_index, name in enumerate(raster.names):
        for probe_index in range(raster.probe_count):
            original_index = int(original_indices[definition_index, probe_index])
            raster_labels.append(f"{name} @ compartment {original_index}")

    fig, (ax_trace, ax_raster) = plt.subplots(
        2,
        1,
        figsize=(8.0, 5.6),
        constrained_layout=True,
        sharex=True,
    )
    ax_trace.plot(t_ms, center_trace, label="recorded center Vm")
    ax_trace.axhline(-80.0, color="0.3", linestyle="--", linewidth=1.0)
    ax_trace.set_title("Recorded reference")
    ax_trace.set_ylabel("Vm [mV]")
    ax_trace.grid(True, alpha=0.3)
    ax_trace.legend(frameon=False)

    for row_index, bits in enumerate(raster_rows):
        padded = np.concatenate(([False], np.asarray(bits, dtype=bool), [False]))
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        starts = transitions[0::2]
        stops = transitions[1::2]
        spans = [
            (float(t_ms[start]), max(float((stop - start) * raster.dt_ms), raster.dt_ms))
            for start, stop in zip(starts, stops, strict=True)
        ]
        if spans:
            ax_raster.broken_barh(
                spans,
                (row_index - 0.35, 0.70),
                facecolors=f"C{row_index}",
                alpha=0.85,
            )
        else:
            ax_raster.plot([], [], color=f"C{row_index}", label=raster_labels[row_index])
    ax_raster.set_title("Threshold-crossing windows retained in VmRaster")
    ax_raster.set_yticks(np.arange(len(raster_labels)), raster_labels)
    ax_raster.set_ylim(len(raster_labels) - 0.5, -0.5)
    ax_raster.set_xlim(float(t_ms[0]), float(t_ms[-1] + raster.dt_ms))
    ax_raster.set_xlabel("Time [ms]")
    ax_raster.set_ylabel("Observer probe")
    ax_raster.grid(True, axis="x", alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
