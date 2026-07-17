"""Retain first-crossing latency without storing a Vm time raster.

Run:
    python examples/advanced/observers/03_compact_latency.py
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
    reference_raster = axs.VmRasterResult.from_result(recorded, latency)

    print("=== Compact latency ===")
    print(f"latency (ms): {values.tolist()}")
    print(f"retained shape: {inspection.probes[0].retained_shape}")
    print(f"retained bytes: {inspection.probes[0].retained_bytes}")
    print(f"Vm retained: {inspection.lowerings[0].retained_vm_width > 0}")

    fig, (ax_trace, ax_raster, ax_latency) = plt.subplots(
        3,
        1,
        figsize=(8.0, 7.8),
        constrained_layout=True,
    )
    labels = ("0 nA", "4 nA")
    for row, label, value in zip(recorded, labels, values, strict=True):
        vm_mV = row.voltage_values(unit=axs.mV)
        center = vm_mV.shape[1] // 2
        row.plot_trace(
            ax=ax_trace,
            index=center,
            time_unit=axs.ms,
            voltage_unit=axs.mV,
            label=label,
        )
        if np.isfinite(value):
            ax_trace.axvline(value, color="C3", linestyle="--", linewidth=1.0)
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

    finite = np.isfinite(values)
    indices = np.arange(values.size)
    ax_latency.scatter(indices[finite], values[finite], color="C3", s=60, label="detected")
    ax_latency.scatter(
        indices[~finite],
        np.zeros(np.count_nonzero(~finite)),
        color="0.4",
        marker="x",
        s=60,
        label="not detected",
    )
    ax_latency.set(
        title="One retained first-crossing timestep per axon",
        ylabel="Latency (ms)",
        xticks=indices,
        xticklabels=labels,
    )
    ax_latency.legend()
    plt.show()


if __name__ == "__main__":
    main()
