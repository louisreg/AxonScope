import numpy as np
import matplotlib.pyplot as plt

from axonscope.axons import Passive, RattayAberham, HodgkinHuxley
from axonscope.solvers import CrankNicholson, Euler

import sys
sys.path.append("./external/")


def test_compare_three_axons(save_dir="figures/physics_tests"):

    # --- Axon parameters ---
    L = 1000    # µm
    d = 1       # µm
    Nx = 101
    tsim = 25.0
    dt_cn = 0.01
    dt_euler = 0.001

    # --- Axons ---
    axons = {
        "Passive": Passive(L=L, d=d, Nx=Nx),
        "RattayAberham": RattayAberham(L=L, d=d, Nx=Nx),
        "HodgkinHuxley": HodgkinHuxley(L=L, d=d, Nx=Nx)
    }

    # --- Inject current ---
    t_start = 1.0
    duration = 1.0
    amplitude = 5
    for axon in axons.values():
        axon.insert_I_Clamp(position=L/2, t_start=t_start, duration=duration, amplitude=amplitude)

    # --- Solve with CN and Euler ---
    results = {}
    for name, axon in axons.items():
        solver_cn = CrankNicholson()
        solver_euler = Euler()
        res_cn = solver_cn.solve(axon, tsim=tsim, dt=dt_cn)
        res_euler = solver_euler.solve(axon, tsim=tsim, dt=dt_euler)
        results[name] = {"CN": res_cn, "Euler": res_euler}

    # --- Positions to track ---
    x_positions = [L/4, L/3, L/2, 2*L/3, 3*L/4]

    # --- Create figure: stacked vertically ---
    fig, axs = plt.subplots(len(axons), 2, figsize=(14, 12), sharex='col')

    for i, (name, res_dict) in enumerate(results.items()):
        res_cn = res_dict["CN"]
        res_euler = res_dict["Euler"]
        indices = [np.argmin(np.abs(res_cn.Vm.shape[1]*0 + res_cn.Vm.shape[1]*0 + np.linspace(0,L,Nx) - xp)) for xp in x_positions]

        # Left: Vm vs time at different positions
        for idx, xp in zip(indices, x_positions):
            axs[i, 0].plot(res_cn.t, res_cn.Vm[:, idx], label=f'x={xp:.1f}µm - CN')
            axs[i, 0].plot(res_euler.t, res_euler.Vm[:, idx], '--', label=f'x={xp:.1f}µm - Euler')

        axs[i, 0].set_ylabel('Vm [mV]')
        axs[i, 0].set_title(f'{name} axon - Vm vs Time')
        axs[i, 0].legend(fontsize=8)
        axs[i, 0].grid(True)

        # Right: 2D space-time map (CN)
        extent = [0, tsim, 0, L]
        im = axs[i, 1].imshow(
            res_cn.Vm.T,
            aspect='auto',
            extent=extent,
            origin='lower',
            cmap='viridis'
        )
        axs[i, 1].set_ylabel('Position [µm]')
        axs[i, 1].set_title(f'{name} axon - Space-time map (CN)')
        cbar = fig.colorbar(im, ax=axs[i, 1])
        cbar.set_label('Vm [mV]')

    axs[-1, 0].set_xlabel('Time [ms]')
    axs[-1, 1].set_xlabel('Time [ms]')

    fig.tight_layout()
    filename = save_dir + "/compare_three_axons_CN_vs_Euler.png"
    fig.savefig(filename)
    plt.close(fig)
