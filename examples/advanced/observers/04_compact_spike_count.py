"""Count repeated spikes without retaining a Vm time raster.

Run:
    python examples/advanced/observers/04_compact_spike_count.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
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
    for start_ms in (0.50, 7.00):
        instance.add_current_clamp(
            position=60.0 * axs.um,
            current=axs.Stimulus.pulse(
                start=start_ms * axs.ms,
                duration=0.10 * axs.ms,
                amplitude=amplitude_na * axs.nA,
            ),
        )
    return instance


def main() -> None:
    population = [_axon(0.0), _axon(4.0)]
    spike_count = axs.analysis.SpikeCount(
        threshold=0.0 * axs.mV,
        reset_threshold=-20.0 * axs.mV,
        blanking=0.20 * axs.ms,
        refractory=1.0 * axs.ms,
        target=axs.positions.CENTER,
        max_spikes=1,
    )

    compact_simulation = axs.AxonSimulation(
        population,
        duration=12.0 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.none(),
        observers=(spike_count,),
    )
    inspection = compact_simulation.inspect()
    compact = compact_simulation.run()
    if compact.observations is None:
        raise RuntimeError("compact spike simulation returned no observations")
    summary = compact.observations["spike_count"]
    counts = np.asarray(summary.values, dtype=int)

    recorded = axs.AxonSimulation(
        population,
        duration=12.0 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.voltage(),
    ).run()
    reference = recorded.analyze(spike_count)
    np.testing.assert_array_equal(counts, reference.values)
    for compact_event, reference_event in zip(
        summary.events, reference.events, strict=True
    ):
        assert compact_event is not None and reference_event is not None
        assert compact_event.count == reference_event.count
        assert compact_event.probe_counts == reference_event.probe_counts
        if compact_event.first_time_ms is None:
            assert reference_event.first_time_ms is None
            assert compact_event.last_time_ms is None
            assert reference_event.last_time_ms is None
            continue
        np.testing.assert_allclose(
            (compact_event.first_time_ms, compact_event.last_time_ms),
            (reference_event.first_time_ms, reference_event.last_time_ms),
            equal_nan=True,
        )
        assert compact_event.overflow == reference_event.overflow
        for compact_times, reference_times in zip(
            compact_event.spike_times_ms,
            reference_event.spike_times_ms,
            strict=True,
        ):
            np.testing.assert_allclose(compact_times, reference_times)
    reference_raster = axs.VmRasterResult.from_result(recorded, spike_count)

    print("=== Compact spike count ===")
    print(f"counts: {counts.tolist()}")
    print(f"events: {summary.events}")
    print(f"retained shape: {inspection.probes[0].retained_shape}")
    print(f"retained bytes: {inspection.probes[0].retained_bytes}")
    print(f"Vm retained: {inspection.lowerings[0].retained_vm_width > 0}")

    fig, (ax_trace, ax_raster, ax_summary) = plt.subplots(
        3,
        1,
        figsize=(8.0, 8.2),
        constrained_layout=True,
    )
    labels = ("0 nA", "4 nA")
    for row, label in zip(recorded, labels, strict=True):
        vm_mV = row.voltage_values(unit=axs.mV)
        row.plot_trace(
            ax=ax_trace,
            index=vm_mV.shape[1] // 2,
            time_unit=axs.ms,
            voltage_unit=axs.mV,
            label=label,
        )
    ax_trace.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0)
    ax_trace.axvline(0.20, color="0.5", linestyle=":", linewidth=1.0)
    ax_trace.set(title="Recorded Vm reference", xlabel="Time (ms)", ylabel="Vm (mV)")
    ax_trace.legend()

    reference_raster.plot(
        ax=ax_raster,
        row=1,
        time_unit=axs.ms,
        title="Raster derived from dense Vm (4 nA reference)",
    )

    indices = np.arange(counts.size)
    ax_summary.bar(indices, counts, color=("0.65", "C0"), alpha=0.8)
    for index, event in enumerate(summary.events):
        if event is None or event.first_time_ms is None:
            continue
        ax_summary.text(
            index,
            counts[index] + 0.08,
            f"first {event.first_time_ms:.2f} ms\nlast {event.last_time_ms:.2f} ms",
            ha="center",
            va="bottom",
        )
    ax_summary.set(
        title="Bounded K=1 timestamp storage (second spike overflows)",
        ylabel="Spike count",
        xticks=indices,
        xticklabels=labels,
        ylim=(0.0, max(1.5, float(np.max(counts)) + 0.8)),
    )
    plt.show()


if __name__ == "__main__":
    main()
