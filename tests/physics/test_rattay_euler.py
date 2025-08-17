# tests/physic/test_rattay_euler_plot.py
import numpy as np
import matplotlib.pyplot as plt

from axonscope.axons import RattayAberham
from axonscope.solvers import Euler


def test_rattay_euler_response_plot(save_dir="figures/physics_tests"):
    """
    Physical test of the RattayAberham axon model with Euler solver.
    A brief current pulse is injected and the membrane potential is recorded.
    The resulting figure is saved to disk.
    """

    # --- axon parameters
    L = 1000    #in µm
    d = 1     # diameter in µm
    Nx = 101
    axon = RattayAberham(L=L, d=d, Nx=Nx, celsius=37)

    #Inject current 1ms pulse 
    axon.insert_I_Clamp(position=L / 2, t_start=1.0, duration=1.0, amplitude=1)

    # --- solver setup
    solver = Euler()
    tsim = 10.0    # total simulation time [ms]
    dt = 0.001     # time step [ms]

    V_all, t_vec = solver.solve(axon, tsim=tsim, dt=dt)

    # ---- Choose positions along the axon ----
    x_positions = [0, L/3, L/2, 2*L/3, L]
    indices = [np.argmin(np.abs(axon.x - xp)) for xp in x_positions]

    fig, ax_x = plt.subplots(figsize=(8,5))
    for idx, xp in zip(indices, x_positions):
        ax_x.plot(t_vec, V_all[:, idx], label=f'x = {xp:.1f} µm')
    ax_x.set_xlabel('Time [ms]')
    ax_x.set_ylabel('V_m [mV]')
    ax_x.legend()
    ax_x.grid(True)

    todo: comparison with NRV + better plot

    Todo: proper test with HH

    create/push to git

    

    plt.show()
    exit()

    # save the figure into pytest's tmp_path
    # --- Plot 2: 2D space–time voltage map
    plt.figure(figsize=(8, 5))
    extent = [0, tsim, 0, L * 1e-4]  # [time(ms), position(cm)]
    plt.imshow(
        V_all.T,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="viridis"
    )
    plt.colorbar(label="Membrane potential (mV)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Position along axon (cm)")
    plt.title("Space–time membrane potential")
    #fig2_path = tmp_path / "rattay_space_time.png"
    #plt.savefig(fig2_path)
    #plt.close()
    plt.show()
    #plt.savefig(fig_path)
    #plt.close()




