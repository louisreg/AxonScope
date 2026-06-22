"""Build and plot stimulus waveforms with units.

Run:
    python examples/basic/02_stimuli_and_units.py
"""

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # Stimulus objects describe time-dependent amplitudes. They can be plotted,
    # added together, and reused later by clamps or electrodes.
    current_unit = axs.uA

    # Keep the three primitive shapes visible. The parameters are unit-bearing,
    # so the script reads like the experiment it represents.
    stimuli = {
        "pulse": axs.Stimulus.pulse(
            start=1.0 * axs.ms,
            duration=1.0 * axs.ms,
            amplitude=5.0 * current_unit,
        ),
        "biphasic": axs.Stimulus.biphasic(
            start=3.0 * axs.ms,
            cathodic_amplitude=10.0 * current_unit,
            cathodic_duration=0.2 * axs.ms,
            interphase=0.05 * axs.ms,
        ),
        "sinus": axs.Stimulus.sinus(
            start=5.0 * axs.ms,
            duration=3.0 * axs.ms,
            amplitude=2.0 * current_unit,
            frequency_khz=1.0 * axs.kHz,
        ),
    }

    # Stimuli are composable: this combined waveform can be attached to a single
    # electrode or clamp exactly like the primitive waveforms.
    stimuli["combined"] = stimuli["pulse"] + stimuli["biphasic"] + stimuli["sinus"]

    # The time axis is also unit-bearing. The plot helper converts it to the
    # requested display units at the plotting boundary.
    t = np.linspace(0.0, 10.0, 2000) * axs.ms
    fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
    colors = {
        "pulse": "tab:blue",
        "biphasic": "tab:red",
        "sinus": "tab:green",
        "combined": "tab:purple",
    }
    for ax, (label, stimulus) in zip(axes, stimuli.items(), strict=True):
        stimulus.plot(
            t,
            ax=ax,
            amplitude_unit=current_unit,
            ylabel="Current [uA]",
            color=colors[label],
        )
        ax.set_title(label)

    axes[-1].set_xlabel("Time [ms]")
    fig.suptitle("Temporal stimulus waveforms")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
