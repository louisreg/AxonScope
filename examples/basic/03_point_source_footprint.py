"""Sample one analytical point-source helper into typed extracellular objects.

Run:
    python examples/basic/03_point_source_footprint.py
"""

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # This script does not run an axon simulation. It samples a helper into the
    # footprint/drive/stimulation objects that normal simulations consume.
    positions = np.linspace(0.0, 1000.0, 201) * axs.um
    t = np.linspace(0.0, 3.0, 600) * axs.ms

    # PointSourceElectrode lives in `axs.analytical` because it is a helper for
    # homogeneous-media examples, not a solver/runtime concept.
    electrode = axs.analytical.PointSourceElectrode(
        x=500.0 * axs.um,
        z=100.0 * axs.um,
    )
    stimulus = axs.Stimulus.biphasic(
        start=1.0 * axs.ms,
        cathodic_amplitude=100.0 * axs.uA,
        cathodic_duration=0.1 * axs.ms,
        interphase=0.05 * axs.ms,
    )

    stimulation = axs.analytical.point_source_stimulation(
        electrode,
        positions,
        sigma=0.3 * axs.S_per_m,
        stimulus=stimulus,
    )
    drive = stimulation.drives[0]
    footprint = drive.footprint

    positions_um = footprint.position_values(unit=axs.um)
    footprint_mV_per_uA = footprint.value_values(
        voltage_unit=axs.mV,
        current_unit=axs.uA,
    )
    vext_mV = stimulation.evaluate(t, voltage_unit=axs.mV)
    t_ms = np.asarray(t.to(axs.ms).magnitude, dtype=float)
    activation_proxy = np.gradient(
        np.gradient(footprint_mV_per_uA, positions_um),
        positions_um,
    )

    fig, (ax_footprint, ax_map, ax_activation) = plt.subplots(1, 3, figsize=(13, 3.5))

    ax_footprint.plot(positions_um, footprint_mV_per_uA)
    ax_footprint.axvline(electrode.x_um, color="black", linestyle="--", linewidth=1.0)
    ax_footprint.set_xlabel("Intrinsic axon position [um]")
    ax_footprint.set_ylabel("footprint [mV/uA]")
    ax_footprint.set_title("Sampled footprint")
    ax_footprint.grid(True, alpha=0.3)

    image = ax_map.imshow(
        vext_mV.T,
        origin="lower",
        aspect="auto",
        extent=[t_ms[0], t_ms[-1], positions_um[0], positions_um[-1]],
        cmap="coolwarm",
    )
    ax_map.set_xlabel("Time [ms]")
    ax_map.set_ylabel("Intrinsic axon position [um]")
    ax_map.set_title("Typed stimulation Vext")
    fig.colorbar(image, ax=ax_map, label="Vext [mV]")

    ax_activation.plot(positions_um, activation_proxy)
    ax_activation.axvline(electrode.x_um, color="black", linestyle="--", linewidth=1.0)
    ax_activation.set_xlabel("Intrinsic axon position [um]")
    ax_activation.set_ylabel("d2 footprint / dx2")
    ax_activation.set_title("Activation-function proxy")
    ax_activation.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
