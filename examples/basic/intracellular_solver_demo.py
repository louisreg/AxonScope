"""Small intracellular stimulation demo.

Run:
    python examples/basic/intracellular_solver_demo.py
"""

import matplotlib.pyplot as plt
import numpy as np

from axonscope.axons import HodgkinHuxley
from axonscope.solvers import CrankNicholson
from axonscope.stimulus import Stimulus


def main() -> None:
    length_um = 500.0
    axon = HodgkinHuxley(L=length_um, d=0.5, Nx=41, celsius=6.3)
    axon.insert_I_Clamp(
        position=length_um / 2.0,
        stimulus=Stimulus.pulse(start=1.0, duration=0.5, amplitude=2.0),
    )

    res = CrankNicholson().solve(axon, tsim=5.0, dt=0.01)

    center_idx = int(np.argmin(np.abs(np.asarray(axon.x) - length_um / 2.0)))
    plt.figure(figsize=(7, 3))
    plt.plot(np.asarray(res.t), np.asarray(res.Vm)[:, center_idx])
    plt.xlabel("Time [ms]")
    plt.ylabel("Vm [mV]")
    plt.title("Hodgkin-Huxley intracellular stimulation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
