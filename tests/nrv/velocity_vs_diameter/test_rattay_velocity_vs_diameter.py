"""
Velocity vs diameter — RattayAberham compared to NRV.

Diameters: 0.2 → 1.0 µm (9 points).
Tolerance: 10 % relative.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from axonscope.axons.unmyelinated import RattayAberham
from axonscope.solvers.Euler import Euler
import nrv

DIAMETERS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
L, Nx = 1000, 101
T_START, DURATION, AMP = 1.0, 0.5, 1.0
TSIM, DT = 10.0, 0.001
RTOL = 0.10


@pytest.mark.nrv_velocity
def test_rattay_velocity_vs_diameter(save_dir="figures/nrv_tests"):
    vel_as, vel_nrv = [], []

    for d in DIAMETERS:
        axon = RattayAberham(L=L, d=d, Nx=Nx, celsius=37)
        axon.insert_I_Clamp(position=L/2, t_start=T_START, duration=DURATION, amplitude=AMP)
        res = Euler().solve(axon, tsim=TSIM, dt=DT)

        axon_nrv = nrv.unmyelinated(y=0, z=0, d=d, L=L, Nsec=Nx, dt=DT, v_init=-70)
        axon_nrv.insert_I_Clamp(0.5, T_START, DURATION, AMP)
        results_nrv = axon_nrv.simulate(t_sim=TSIM)
        del axon_nrv
        results_nrv.rasterize()

        v_nrv = results_nrv.get_avg_AP_speed()
        v_as = res.average_velocity()
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
    fig.savefig(f"{save_dir}/rattay_velocity_vs_diameter.png")
    plt.close(fig)

    assert np.allclose(np.array(vel_as), np.array(vel_nrv), rtol=RTOL), (
        f"Velocity mismatch > {RTOL*100:.0f}%: AS={vel_as}, NRV={vel_nrv}"
    )
