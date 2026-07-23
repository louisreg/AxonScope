"""Retain an event-preserving raster at sparse spatial and temporal probes.

Run:
    python examples/advanced/observers/05_downsampled_vm_raster.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


def main() -> None:
    axon = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=31,
        celsius=6.3 * axs.degC,
    )
    instance = axs.AxonInstance(axon)
    instance.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.05 * axs.ms,
            duration=0.10 * axs.ms,
            amplitude=4.0 * axs.nA,
        ),
    )
    raster_definition = axs.analysis.VmRaster(
        threshold=0.0 * axs.mV,
        target=axs.positions.At((20.0 * axs.um, 60.0 * axs.um, 100.0 * axs.um)),
        every_n_steps=10,
    )

    compact_simulation = axs.AxonSimulation(
        instance,
        duration=2.0 * axs.ms,
        dt=0.005 * axs.ms,
        recording=axs.Recording.none(),
        observers=(raster_definition,),
    )
    inspection = compact_simulation.inspect()
    compact = compact_simulation.run()
    if compact.observations is None:
        raise RuntimeError("downsampled raster simulation returned no observations")
    raster = compact.observations[axs.VM_RASTER_OBSERVATION_KEY]

    recorded = axs.AxonSimulation(
        instance,
        duration=2.0 * axs.ms,
        dt=0.005 * axs.ms,
        recording=axs.Recording.voltage(),
    ).run().single
    reference = axs.VmRasterResult.from_result(recorded, raster_definition)
    np.testing.assert_array_equal(raster.unpack(), reference.unpack())

    full_definition = axs.analysis.VmRaster(
        threshold=0.0 * axs.mV,
        target=raster_definition.target,
    )
    full_raster = axs.VmRasterResult.from_result(recorded, full_definition)
    print("=== Spatial and downsampled VmRaster ===")
    print(f"dense Vm shape: {recorded.Vm.shape}")
    print(f"compact raster shape: {raster.unpack().shape}")
    print(f"full raster bytes: {np.asarray(full_raster.words).nbytes}")
    print(f"compact raster bytes: {np.asarray(raster.words).nbytes}")
    print(f"retained shape: {inspection.probes[0].retained_shape}")

    fig, (ax_trace, ax_raster) = plt.subplots(
        2,
        1,
        figsize=(8.0, 5.8),
        constrained_layout=True,
        sharex=True,
    )
    recorded.plot_trace(
        ax=ax_trace,
        index=recorded.Vm.shape[1] // 2,
        time_unit=axs.ms,
        voltage_unit=axs.mV,
        label="center Vm",
    )
    ax_trace.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0)
    ax_trace.set(title="Dense Vm reference", ylabel="Vm (mV)")
    ax_trace.legend()

    raster.plot(
        ax=ax_raster,
        time_unit=axs.ms,
        title="Three probes, 10-step event-preserving windows",
    )
    plt.show()


if __name__ == "__main__":
    main()
