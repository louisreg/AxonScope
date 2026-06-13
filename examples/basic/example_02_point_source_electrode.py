# examples/basic/point_source_electrode_demo.py

"""
Small demo of the AxonScope PSA stimulation API.

Run:
    python examples/basic/point_source_electrode_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

from axonscope.stimulus import Stimulus
from axonscope.electrodes import PointSourceElectrode
from axonscope.solvers.stimulus_runtime import compile_extracellular_context
from axonscope.stimulus_eval import (
    evaluate_extracellular_context_numpy,
    evaluate_stimulus_numpy,
)


def main():
    # ------------------------------------------------------
    # Axon geometry
    # ------------------------------------------------------
    L_um = 1000.0
    Nx = 201

    x_um = np.linspace(0.0, L_um, Nx)
    x_m = x_um * 1e-6

    # ------------------------------------------------------
    # Electrode
    # ------------------------------------------------------
    electrode = PointSourceElectrode(
        x0_m=500e-6,       # centered along the axon
        y0_m=0.0,
        z0_m=100e-6,       # 100 µm away from the axon
        sigma_S_m=0.3,     # extracellular conductivity
    )

    # ------------------------------------------------------
    # Stimulus
    # ------------------------------------------------------
    stim = Stimulus.biphasic(
        start=1.0,
        cathodic_amplitude=100e-6,   # A
        cathodic_duration=0.1,       # ms
        interphase=0.05,
    )

    extra = electrode.attach_stimulus(stim)

    # ------------------------------------------------------
    # Evaluate extracellular potential
    # ------------------------------------------------------
    t_ms = np.linspace(0.0, 3.0, 1000)
    Vext = evaluate_extracellular_context_numpy(extra, x_m, t_ms)  # shape (Nt, Nx), in volts

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    # Stimulus current
    axs[0].plot(t_ms, evaluate_stimulus_numpy(stim, t_ms) * 1e6)
    axs[0].set_ylabel("Current [µA]")
    axs[0].set_title("Biphasic point-source stimulation")
    axs[0].grid(True)

    # Spatial footprint
    footprint = electrode.footprint(x_m)
    axs[1].plot(x_um, footprint * 1e-3)
    axs[1].set_xlabel("Position along axon [µm]")
    axs[1].set_ylabel("Footprint [mV/µA]")
    axs[1].grid(True)

    # Space-time extracellular potential
    im = axs[2].imshow(
        Vext.T * 1e3,
        aspect="auto",
        extent=[t_ms[0], t_ms[-1], x_um[0], x_um[-1]],
        origin="lower",
        cmap="coolwarm",
    )
    axs[2].set_xlabel("Time [ms]")
    axs[2].set_ylabel("Position along axon [µm]")
    axs[2].set_title("Extracellular potential Vext(x, t)")
    cbar = fig.colorbar(im, ax=axs[2])
    cbar.set_label("Vext [mV]")

    fig.tight_layout()
    plt.show()

    # ------------------------------------------------------
    # JAX-ready object for solvers
    # ------------------------------------------------------
    compiled_extra = compile_extracellular_context(extra, x_m)

    example_time = 1.05
    Vext_jax = compiled_extra(example_time)

    print(f"Compiled extracellular stimulus evaluated at t = {example_time} ms")
    print(f"Shape: {Vext_jax.shape}")
    print(f"Max Vext: {float(np.max(np.asarray(Vext_jax))) * 1e3:.3f} mV")


if __name__ == "__main__":
    main()
