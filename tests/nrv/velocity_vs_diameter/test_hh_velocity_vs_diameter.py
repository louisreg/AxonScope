"""
Velocity vs diameter — HodgkinHuxley compared to NRV.

Diameters: 0.4 → 1.0 µm (7 points, below 0.4 µm HH doesn't propagate reliably).
Tolerance: 15 % relative.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from axonscope import AxonInstance, degC, mV, ms, um
from axonscope.axons.unmyelinated import HodgkinHuxley
from axonscope.analysis import conduction_velocity
from axonscope.solvers.crank_nicholson import CrankNicholson
from axonscope.stimulation import Stimulus
import nrv

DIAMETERS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
L, Nx = 1000, 101
T_START, DURATION, AMP = 1.0, 0.5, 1.0
TSIM, DT = 10.0, 0.001
RTOL = 0.15


@pytest.mark.nrv_velocity
def test_hh_velocity_vs_diameter(save_dir="figures/nrv_tests"):
    vel_as, vel_nrv = [], []

    for d in DIAMETERS:
        axon = HodgkinHuxley(length=L * um, diameter=d * um, compartments=Nx, celsius=6.3 * degC, v_init=-70.0 * mV,
                             include_passive_leak=True, g_pas=0.001, e_pas=-70.0)
        sim = AxonInstance(axon)
        sim.add_current_clamp(position=(L / 2) * um, current=Stimulus.pulse(start=T_START * ms, duration=DURATION * ms, amplitude=AMP))
        res = CrankNicholson().solve(sim, tsim=TSIM, dt=DT)

        axon_nrv = nrv.unmyelinated(y=0, z=0, d=d, L=L, Nsec=Nx, dt=DT, v_init=-70, T=6.3, model="HH")
        axon_nrv.insert_I_Clamp(0.5, T_START, DURATION, AMP)
        results_nrv = axon_nrv.simulate(t_sim=TSIM)
        del axon_nrv
        results_nrv.rasterize()

        v_nrv = results_nrv.get_avg_AP_speed()
        v_as = conduction_velocity(res)
        print(f"d={d} µm — AS={np.round(v_as,3)} NRV={np.round(v_nrv,3)} m/s")
        vel_as.append(v_as)
        vel_nrv.append(v_nrv)

    import os; os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(DIAMETERS, vel_nrv, label="NRV")
    ax.plot(DIAMETERS, vel_as, label="AxonScope")
    ax.set_xlabel("Diameter (µm)")
    ax.set_ylabel("AP propagation speed (m/s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{save_dir}/hh_velocity_vs_diameter.png")
    plt.close(fig)

    assert np.allclose(np.array(vel_as), np.array(vel_nrv), rtol=RTOL), (
        f"Velocity mismatch > {RTOL*100:.0f}%: AS={vel_as}, NRV={vel_nrv}"
    )
