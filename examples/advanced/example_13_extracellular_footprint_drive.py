"""Advanced example 13: factorized extracellular stimulation.

Run:
    python examples/advanced/example_13_extracellular_footprint_drive.py

Phase 2 separates static spatial transfer from temporal stimulation:

    ExtracellularFootprint  -> where current becomes voltage
    ExtracellularDrive      -> one footprint times one stimulus
    ExtracellularStimulation -> the sum of several drives
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # Step 1: choose the intrinsic axon positions where the field is sampled.
    positions = np.linspace(0.0, 1000.0, 201) * axs.um

    # Step 2: describe two analytical point sources in global coordinates.
    cathode = axs.PointSourceElectrode(
        x=500.0 * axs.um,
        z=120.0 * axs.um,
    )
    anode = axs.PointSourceElectrode(
        x=700.0 * axs.um,
        z=180.0 * axs.um,
    )

    # Step 3: keep the medium on the analytical context.
    context = axs.AnalyticalExtracellularContext(
        electrodes=[
            cathode.with_stimulus(axs.Stimulus.constant(0.0 * axs.uA)),
            anode.with_stimulus(axs.Stimulus.constant(0.0 * axs.uA)),
        ],
        sigma=0.3 * axs.S_per_m,
    )

    # Step 4: build static footprints. These arrays contain V/A, not time.
    cathode_footprint = context.build_footprint(
        cathode,
        positions,
        source_id="cathode",
    )
    anode_footprint = context.build_footprint(
        anode,
        positions,
        source_id="anode",
    )

    # Step 5: define temporal waveforms separately from the spatial fields.
    cathode_stimulus = axs.Stimulus.pulse(
        start=1.0 * axs.ms,
        duration=0.20 * axs.ms,
        amplitude=-80.0 * axs.uA,
    )
    anode_stimulus = axs.Stimulus.pulse(
        start=1.0 * axs.ms,
        duration=0.20 * axs.ms,
        amplitude=40.0 * axs.uA,
    )

    # Step 6: pair one footprint with one stimulus to form each drive.
    cathode_drive = axs.ExtracellularDrive(
        id=axs.DriveId("cathode"),
        footprint=cathode_footprint,
        stimulus=cathode_stimulus,
    )
    anode_drive = axs.ExtracellularDrive(
        id=axs.DriveId("anode"),
        footprint=anode_footprint,
        stimulus=anode_stimulus,
    )

    # Step 7: group the drives. Evaluation sums them without duplicating geometry.
    extracellular = axs.ExtracellularStimulation([cathode_drive, anode_drive])
    t = np.linspace(0.0, 2.0, 301) * axs.ms
    vext_mV = extracellular.evaluate(t, voltage_unit=axs.mV)

    print(f"drives: {extracellular.names}")
    print(f"positions: {extracellular.positions_um.size}")
    print(f"Vext shape: {vext_mV.shape}")
    print(f"peak |Vext|: {np.max(np.abs(vext_mV)):.3f} mV")

    # Step 8: materialize a dense potential only when it is useful to inspect.
    potential = extracellular.potential(t, voltage_unit=axs.mV)
    print(f"dense potential: {potential.value_values(voltage_unit=axs.mV).shape}")

    # Step 9: plot the static fields and the final summed time-space map.
    x_um = cathode_footprint.position_values(unit=axs.um)
    fig, (ax_static, ax_dynamic) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax_static.plot(x_um, cathode_footprint.value_values(voltage_unit=axs.mV, current_unit=axs.uA), label="cathode")
    ax_static.plot(x_um, anode_footprint.value_values(voltage_unit=axs.mV, current_unit=axs.uA), label="anode")
    ax_static.set_title("Static footprints")
    ax_static.set_xlabel("Position [um]")
    ax_static.set_ylabel("Footprint [mV/uA]")
    ax_static.grid(True, alpha=0.3)
    ax_static.legend(frameon=False)

    ax_dynamic.imshow(
        vext_mV.T,
        aspect="auto",
        origin="lower",
        extent=[0.0, 2.0, x_um[0], x_um[-1]],
        cmap="coolwarm",
    )
    ax_dynamic.set_title("Summed Vext")
    ax_dynamic.set_xlabel("Time [ms]")
    ax_dynamic.set_ylabel("Position [um]")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
