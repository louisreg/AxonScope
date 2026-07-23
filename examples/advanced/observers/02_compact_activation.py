"""Retain one activation flag per axon without storing a Vm time raster.

Run:
    python examples/advanced/observers/02_compact_activation.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


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
    reference_raster = axs.VmRasterResult.from_result(recorded, activation)

    print("=== Compact activation ===")
    print(f"flags: {flags.tolist()}")
    print(f"retained shape: {inspection.probes[0].retained_shape}")
    print(f"retained bytes: {inspection.probes[0].retained_bytes}")
    print(f"Vm retained: {inspection.lowerings[0].retained_vm_width > 0}")

    fig, (ax_trace, ax_raster, ax_flags) = plt.subplots(
        3,
        1,
        figsize=(8.0, 7.8),
        constrained_layout=True,
    )
    labels = ("0 nA", "4 nA")
    for row, label in zip(recorded, labels, strict=True):
        vm_mV = row.voltage_values(unit=axs.mV)
        center = vm_mV.shape[1] // 2
        row.plot_trace(
            ax=ax_trace,
            index=center,
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

    ax_flags.bar(labels, flags.astype(int), color=("0.65", "C0"))
    ax_flags.set(
        title="One retained boolean per axon",
        ylabel="Activated",
        ylim=(0.0, 1.15),
        yticks=(0, 1),
    )
    for index, flag in enumerate(flags):
        ax_flags.text(index, 0.05 if flag else 0.03, str(bool(flag)), ha="center")
    plt.show()


if __name__ == "__main__":
    main()
