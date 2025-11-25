# tests/physic/test_rattay_euler_plot.py
import numpy as np
import matplotlib.pyplot as plt

from axonscope.axons import Sundt
from axonscope.solvers import CrankNicholson

import nrv


def test_Sundt_vs_NRV(save_dir="figures/physics_tests"):
    """
    Physical test of the HodgkinHuxley axon model with Euler solver.
    A current pulse is injected and the membrane potential is recorded.
    We compare result with NRV
    The resulting figure is saved to disk.
    """

    # --- axon parameters
    L = 1000    #in µm
    d = 0.5       # diameter in µm
    Nx = 101
    axon = Sundt(L=L, d=d, Nx=Nx, celsius=37)

    #Inject current 1ms pulse 
    t_start = 1.0
    duration = 1.0
    amplitude = 2
    axon.insert_I_Clamp(position=L / 2, t_start=t_start, duration=duration, amplitude=amplitude)

    # --- solver setup
    solver = CrankNicholson()
    tsim = 10.0         # total simulation time [ms]
    dt = 0.001          # time step [ms]

    res = solver.solve(axon, tsim=tsim, dt=dt)


     # ---- NRV Axon ----
    axon_nrv = nrv.unmyelinated(0,0,d,L,dt=dt,Nsec=Nx, model = "Sundt", v_init = axon.Vinit, T = axon.Temp)
                        
    axon_nrv.insert_I_Clamp(0.5, t_start, duration, amplitude)
    results_NRV = axon_nrv.simulate(t_sim=tsim)

    # ---- Choose positions along the axon ----
    x_positions = [L/4, L/3, L/2, 2*L/3, 3*L/4]
    indices = [np.argmin(np.abs(axon.x - xp)) for xp in x_positions]

    fig, axs = plt.subplots(1,3, figsize=(12,5))
    for idx, xp in zip(indices, x_positions):
        axs[0].plot(res.t, res.Vm[:, idx], label=f'x = {xp:.1f} µm')

    t = results_NRV['t'].ravel()          # s'assure que t est 1D
    x_rec = results_NRV['x_rec']          # positions [µm]
    Vm = results_NRV['V_mem']             # shape (Nt, Nx)

    indices = [np.argmin(np.abs(x_rec - xp)) for xp in x_positions]

    for idx, xp in zip(indices, x_positions):
        axs[0].plot(t, Vm[idx, :],'--', label=f"x = {xp:.1f} µm - NRV")
    axs[0].legend()

    axs[0].set_xlabel('Time [ms]')
    axs[0].set_ylabel('Vm [mV]')
    axs[0].legend()
    axs[0].grid(True)

    # --- Plot 2: 2D space–time voltage map
    extent = [0, tsim, 0, L]  # [time(ms), position(cm)]
    map = axs[1].imshow(
        res.Vm.T,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="viridis"
    )
    
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Position along axon (µm)")
    cbar = fig.colorbar(map)
    cbar.set_label('membrane voltage - AxonScope (mV)')


    map = axs[2].pcolormesh(results_NRV['t'], results_NRV['x_rec'], results_NRV['V_mem'] ,shading='auto')
    axs[2].set_xlabel('time (ms)')
    axs[2].set_ylabel('position (µm)')
    cbar = fig.colorbar(map)
    cbar.set_label('membrane voltage - NRV (mV)')

    fig.tight_layout()
    fig_path = save_dir + "/Sundt_CN_vs_NRV.png"
    fig.savefig(fig_path)




