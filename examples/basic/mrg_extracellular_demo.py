"""Small MRG extracellular stimulation demo.

Run:
    python examples/basic/mrg_extracellular_demo.py
"""

import matplotlib.pyplot as plt
import numpy as np

from axonscope.axons import MRG
from axonscope.electrodes import PointSourceElectrode
from axonscope.solvers import CrankNicholson
from axonscope.stimulus import Stimulus


def main() -> None:
    axon = MRG(d=10.0, nodes=5)
    electrode = PointSourceElectrode(
        x0_m=float(np.asarray(axon.x)[axon.Nx // 2]) * 1e-6,
        z0_m=500e-6,
        sigma_S_m=0.3,
    )
    stim = Stimulus.biphasic(
        start=0.5,
        cathodic_amplitude=80e-6,
        cathodic_duration=0.05,
        interphase=0.02,
    )
    axon.add_extracellular_ctx(electrode, stim)

    res = CrankNicholson().solve(axon, tsim=2.0, dt=0.01)

    node_indices = np.asarray(axon.node_indices, dtype=int)
    plt.figure(figsize=(7, 3))
    for idx in node_indices[:3]:
        plt.plot(np.asarray(res.t), np.asarray(res.Vm)[:, idx], label=f"node {idx}")
    plt.xlabel("Time [ms]")
    plt.ylabel("Vm [mV]")
    plt.title("MRG extracellular point-source stimulation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
