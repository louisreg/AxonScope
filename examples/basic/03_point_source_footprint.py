"""Evaluate one Pint-aware point-source electrode.

Run:
    python examples/basic/03_point_source_footprint.py
"""

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # This script does not run an axon simulation. It only visualizes the
    # extracellular field object that will later be attached to axons.
    x = np.linspace(0.0, 1000.0, 201) * axs.um
    t = np.linspace(0.0, 3.0, 600) * axs.ms
    electrode_x = 500.0 * axs.um
    electrode_z = 100.0 * axs.um

    # A bare point-source electrode describes geometry. Adding a stimulus turns
    # it into one contribution to an extracellular context.
    electrode = axs.PointSourceElectrode(
        x=electrode_x,
        z=electrode_z,
    )
    stimulus = axs.Stimulus.biphasic(
        start=1.0 * axs.ms,
        cathodic_amplitude=100.0 * axs.uA,
        cathodic_duration=0.1 * axs.ms,
        interphase=0.05 * axs.ms,
    )
    extracellular = axs.AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(stimulus)],
        sigma=0.3 * axs.S_per_m,
    )

    # Read the panels left to right:
    # spatial footprint per current, time-varying Vext, and activation function.
    fig, (ax_footprint, ax_map, ax_activation) = plt.subplots(1, 3, figsize=(13, 3.5))
    extracellular.plot_footprint(
        x,
        ax=ax_footprint,
        voltage_unit=axs.mV,
        current_unit=axs.uA,
    )
    extracellular.plot_evaluation(
        x,
        t,
        ax=ax_map,
        voltage_unit=axs.mV,
    )
    extracellular.plot_activation_function(
        x,
        ax=ax_activation,
        voltage_unit=axs.mV,
        current_unit=axs.uA,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
