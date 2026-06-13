# examples/basic/stimulus_demo.py

"""
Small demo of the AxonScope Stimulus API.

Run:
    python examples/basic/stimulus_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

from axonscope.stimulus import Stimulus
from axonscope.stimulus_eval import evaluate_stimulus_numpy


# ==========================================================
# Helper
# ==========================================================

def plot_stimulus(ax, stim, t_end=10.0, label=None):
    t = np.linspace(0.0, t_end, 2000)
    y = evaluate_stimulus_numpy(stim, t)
    ax.plot(t, y, linewidth=2, label=label)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.grid(True)


# ==========================================================
# Main demo
# ==========================================================

def main():

    # ------------------------------------------------------
    # 1. Monophasic pulse
    # ------------------------------------------------------
    pulse = Stimulus.pulse(
        start=1.0,
        amplitude=5.0,
        duration=1.0,
    )

    # ------------------------------------------------------
    # 2. Biphasic pulse
    # ------------------------------------------------------
    biphasic = Stimulus.biphasic(
        start=3.0,
        cathodic_amplitude=10.0,
        cathodic_duration=0.2,
        interphase=0.05,
    )

    # ------------------------------------------------------
    # 3. Sinusoidal waveform
    # ------------------------------------------------------
    sinus = Stimulus.sinus(
        start=5.0,
        duration=3.0,
        amplitude=2.0,
        frequency_khz=1.0,
    )

    # ------------------------------------------------------
    # 4. Ramp
    # ------------------------------------------------------
    ramp = Stimulus.ramp(
        start=0.0,
        duration=4.0,
        start_value=0.0,
        stop_value=3.0,
        dt=0.05,
    )

    # ------------------------------------------------------
    # 5. Composite waveform
    # ------------------------------------------------------
    composite = pulse + biphasic + sinus + 0.5 * ramp

    # ------------------------------------------------------
    # Plot all
    # ------------------------------------------------------
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    plot_stimulus(axs[0], pulse, label="Pulse")
    axs[0].legend()

    plot_stimulus(axs[1], biphasic, label="Biphasic")
    axs[1].legend()

    plot_stimulus(axs[2], sinus, label="Sinus")
    axs[2].legend()

    plot_stimulus(axs[3], ramp, label="Ramp")
    axs[3].legend()

    plot_stimulus(axs[4], composite, label="Composite")
    axs[4].legend()

    axs[-1].set_xlim(0.0, 10.0)

    fig.suptitle("AxonScope Stimulus Demo", fontsize=16)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
