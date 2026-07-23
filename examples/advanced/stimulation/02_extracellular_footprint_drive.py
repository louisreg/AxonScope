"""Separate extracellular footprints from temporal drives.

Run:
    python examples/advanced/stimulation/02_extracellular_footprint_drive.py

AxonFleet separates static spatial transfer from temporal stimulation:

    ExtracellularFootprint  -> where current becomes voltage
    ExtracellularDrive      -> one footprint times one stimulus
    ExtracellularStimulation -> the sum of several drives
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


def main() -> None:
    # Step 1: choose the intrinsic axon positions where the field is sampled.
    positions = np.linspace(0.0, 1000.0, 201) * axs.um

    # Step 2: describe two analytical point sources in the local axon frame.
    cathode = axs.analytical.PointSourceElectrode(
        x=500.0 * axs.um,
        z=120.0 * axs.um,
    )
    anode = axs.analytical.PointSourceElectrode(
        x=700.0 * axs.um,
        z=180.0 * axs.um,
    )

    # Step 3: build static footprints. These arrays contain V/A, not time.
    cathode_footprint = axs.analytical.point_source_footprint(
        cathode,
        positions,
        sigma=0.3 * axs.S_per_m,
        source_id="cathode",
    )
    anode_footprint = axs.analytical.point_source_footprint(
        anode,
        positions,
        sigma=0.3 * axs.S_per_m,
        source_id="anode",
    )

    # Step 4: define temporal waveforms separately from the spatial fields.
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

    # Step 5: pair one footprint with one stimulus to form each drive.
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

    # Step 6: group the drives. Evaluation sums them without duplicating geometry.
    extracellular = axs.ExtracellularStimulation([cathode_drive, anode_drive])
    t = np.linspace(0.0, 2.0, 301) * axs.ms
    vext_mV = extracellular.evaluate(t, voltage_unit=axs.mV)

    print(f"drives: {extracellular.names}")
    print(f"positions: {extracellular.positions_um.size}")
    print(f"Vext shape: {vext_mV.shape}")
    print(f"peak |Vext|: {np.max(np.abs(vext_mV)):.3f} mV")

    # Step 7: materialize a dense potential only when it is useful to inspect.
    potential = extracellular.potential(t, voltage_unit=axs.mV)
    print(f"dense potential: {potential.value_values(voltage_unit=axs.mV).shape}")

    # Step 8: plot the static fields and the final summed time-space map.
    fig, (ax_static, ax_dynamic) = plt.subplots(1, 2, figsize=(10, 3.5))
    extracellular.plot_footprints(
        ax=ax_static,
        position_unit=axs.um,
        voltage_unit=axs.mV,
        current_unit=axs.uA,
        title="Static footprints",
    )
    potential.plot(
        ax=ax_dynamic,
        time_unit=axs.ms,
        position_unit=axs.um,
        voltage_unit=axs.mV,
        title="Summed Vext",
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
