"""
Numerics — CN solver variants vs NRV.

test_cn_solvers_vs_euler:  visual comparison of CN vs Euler on 3 axon types.
test_cn_fine_mesh_vs_nrv:  fine-mesh stability check (Nx=501, dx≈2µm) vs NRV.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pytest

from axonscope.axons.unmyelinated import RattayAberham, HodgkinHuxley
from axonscope.axons.generic import Passive
from axonscope.solvers.CrankNicholson import CrankNicholson
from axonscope.solvers.Euler import Euler
import nrv

pytestmark = pytest.mark.nrv_numerics


def test_cn_solvers_vs_euler(save_dir="figures/nrv_tests"):
    L, d, Nx = 1000, 1, 101
    tsim, dt = 25.0, 0.001
    t_start, duration, amplitude = 1.0, 1.0, 5

    axons = {
        "Passive": Passive(L=L, d=d, Nx=Nx),
        "RattayAberham": RattayAberham(L=L, d=d, Nx=Nx),
        "HodgkinHuxley": HodgkinHuxley(L=L, d=d, Nx=Nx),
    }
    x_positions = [L/4, L/3, L/2, 2*L/3, 3*L/4]

    for axon in axons.values():
        axon.insert_I_Clamp(position=L/2, t_start=t_start, duration=duration, amplitude=amplitude)

    results = {}
    for name, axon in axons.items():
        results[name] = {
            "CN":    CrankNicholson().solve(axon, tsim=tsim, dt=dt),
            "Euler": Euler().solve(axon, tsim=tsim, dt=dt),
        }

    fig, axs = plt.subplots(len(axons), 2, figsize=(14, 12), sharex="col")
    for i, (name, res_dict) in enumerate(results.items()):
        res_cn, res_euler = res_dict["CN"], res_dict["Euler"]
        x_arr = np.linspace(0, L, Nx)
        indices = [np.argmin(np.abs(x_arr - xp)) for xp in x_positions]
        for idx, xp in zip(indices, x_positions):
            axs[i, 0].plot(res_cn.t, res_cn.Vm[:, idx], label=f"x={xp:.0f}µm CN")
            axs[i, 0].plot(res_euler.t, res_euler.Vm[:, idx], "--", label=f"Euler")
        axs[i, 0].set_title(f"{name} — traces")
        axs[i, 0].set_ylabel("Vm [mV]")
        axs[i, 1].imshow(res_cn.Vm.T, aspect="auto", origin="lower",
                         extent=[0, tsim, 0, L], cmap="viridis")
        axs[i, 1].set_title(f"{name} — space-time (CN)")
    axs[-1, 0].set_xlabel("Time [ms]")
    axs[-1, 1].set_xlabel("Time [ms]")
    fig.tight_layout()
    import os; os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/compare_three_axons_CN_vs_Euler.png")
    plt.close(fig)


def _is_unstable(Vm: np.ndarray) -> bool:
    return bool(np.any(Vm < -150.0) or np.any(Vm > 100.0))


@pytest.mark.nrv_numerics
def test_cn_fine_mesh_vs_nrv(save_dir="figures/nrv_tests"):
    """Fine-mesh stability check (Nx=501, dx≈2µm) — AxonScope vs NRV."""
    L, d, Nx, tsim, dt = 1000, 1.0, 501, 10.0, 0.001
    t_start, duration, amplitude = 1.0, 1.0, 5.0
    x_positions = [L/4, L/3, L/2, 2*L/3, 3*L/4]

    axon_ra = RattayAberham(L=L, d=d, Nx=Nx, celsius=37.0)
    axon_hh = HodgkinHuxley(L=L, d=d, Nx=Nx, celsius=6.3, Vinit=-70.0,
                             include_passive_leak=True, g_pas=0.001, e_pas=-70.0)
    for ax in (axon_ra, axon_hh):
        ax.insert_I_Clamp(position=L/2, t_start=t_start, duration=duration, amplitude=amplitude)

    res_ra = CrankNicholson().solve(axon_ra, tsim=tsim, dt=dt)
    res_hh = CrankNicholson().solve(axon_hh, tsim=tsim, dt=dt)

    nrv_ra = nrv.unmyelinated(0, 0, d, L, dt=dt, Nsec=Nx, V_init=axon_ra.Vinit, T=axon_ra.Temp)
    nrv_ra.insert_I_Clamp(0.5, t_start, duration, amplitude)
    res_nrv_ra = nrv_ra.simulate(t_sim=tsim)

    nrv_hh = nrv.unmyelinated(0, 0, d, L, dt=dt, Nsec=Nx, model="HH",
                               v_init=axon_hh.Vinit, T=axon_hh.Temp)
    nrv_hh.insert_I_Clamp(0.5, t_start, duration, amplitude)
    res_nrv_hh = nrv_hh.simulate(t_sim=tsim)

    checks = {
        "AxonScope RattayAberham": _is_unstable(np.array(res_ra.Vm)),
        "AxonScope HodgkinHuxley": _is_unstable(np.array(res_hh.Vm)),
        "NRV RattayAberham":       _is_unstable(res_nrv_ra["V_mem"].T),
        "NRV HodgkinHuxley":       _is_unstable(res_nrv_hh["V_mem"].T),
    }
    for label, unstable in checks.items():
        print(f"  {label}: {'UNSTABLE' if unstable else 'stable'}")
