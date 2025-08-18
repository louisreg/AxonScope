# tests/physic/test_rattay_euler_plot.py
import numpy as np
import matplotlib.pyplot as plt

from axonscope.axons import RattayAberham as RA
from axonscope.solvers import Euler


def test_rasterplot(save_dir="figures/physics_tests"):

    # --- axon parameters
    L = 1000    #in µm
    d = 0.5       # diameter in µm
    Nx = 101
    axon = RA(L=L, d=d, Nx=Nx, celsius=37)

    #Inject current 1ms pulse 
    t_start = 1.0
    duration = 1.0
    amplitude = 2
    axon.insert_I_Clamp(position=L / 2, t_start=t_start, duration=duration, amplitude=amplitude)

    # --- solver setup
    solver = Euler()
    tsim = 10.0         # total simulation time [ms]
    dt = 0.001          # time step [ms]

    res = solver.solve(axon, tsim=tsim, dt=dt)



    # ---- Choose positions along the axon ----
    x_positions = [L/4, L/3, L/2, 2*L/3, 3*L/4]
    indices = [np.argmin(np.abs(axon.x - xp)) for xp in x_positions]

    fig, axs = plt.subplots(1,3, figsize=(12,5))
    for idx, xp in zip(indices, x_positions):
        axs[0].plot(res.t, res.Vm[:, idx], label=f'x = {xp:.1f} µm')

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

    ## -- Rasterplot
    res.rasterplot(axs[2])

    fig.tight_layout()
    fig_path = save_dir + "/test_rasterplot.png"
    fig.savefig(fig_path)




