"""Advanced example 15: preparation signatures.

Run:
    python examples/advanced/example_15_preparation_signatures.py

Phase 3 starts by giving preparation inputs stable signatures. Later, these
signatures can decide when prepared cohorts, footprints, or runtime arrays are
reusable instead of being rebuilt.
"""

from __future__ import annotations

import numpy as np

import axonscope as axs


def main() -> None:
    # Step 1: define the static spatial support for a footprint.
    positions = np.linspace(0.0, 800.0, 9) * axs.um

    # Step 2: build a simple shared footprint in canonical V/A units.
    footprint = axs.ExtracellularFootprint.shared(
        values=np.linspace(1.0, 2.0, 9),
        positions=positions,
        source_id="teaching-electrode",
    )

    # Step 3: define a temporal drive that will be part of the preparation key.
    stimulus = axs.Stimulus.pulse(
        start=0.50 * axs.ms,
        duration=0.20 * axs.ms,
        amplitude=20.0 * axs.uA,
    )

    # Step 4: combine identifier, footprint, and stimulus into one drive.
    drive = axs.ExtracellularDrive(
        id=axs.DriveId("source-a"),
        footprint=footprint,
        stimulus=stimulus,
    )

    # Step 5: sign the full extracellular stimulation definition.
    stimulation = axs.ExtracellularStimulation([drive])
    signature = axs.preparation.extracellular_stimulation_signature(stimulation)

    # Step 6: the same scientific inputs produce the same signature.
    same_footprint = axs.ExtracellularFootprint.shared(
        values=np.linspace(1.0, 2.0, 9),
        positions=positions,
        source_id="teaching-electrode",
    )
    same_drive = axs.ExtracellularDrive(
        id=axs.DriveId("source-a"),
        footprint=same_footprint,
        stimulus=stimulus,
    )
    same_signature = axs.preparation.extracellular_stimulation_signature(
        axs.ExtracellularStimulation([same_drive])
    )

    # Step 7: changing the input data changes the preparation signature.
    changed_footprint = axs.ExtracellularFootprint.shared(
        values=np.linspace(1.0, 2.1, 9),
        positions=positions,
        source_id="teaching-electrode",
    )
    changed_drive = axs.ExtracellularDrive(
        id=axs.DriveId("source-a"),
        footprint=changed_footprint,
        stimulus=stimulus,
    )
    changed_signature = axs.preparation.extracellular_stimulation_signature(
        axs.ExtracellularStimulation([changed_drive])
    )

    print(f"drive ids: {[drive_signature.id for drive_signature in signature.drives]}")
    print(f"same signature: {signature == same_signature}")
    print(f"changed signature: {signature == changed_signature}")
    print(f"footprint digest: {signature.drives[0].footprint.values_V_per_A.digest}")


if __name__ == "__main__":
    main()
