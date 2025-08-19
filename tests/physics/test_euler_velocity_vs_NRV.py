# tests/physic/test_rattay_euler_plot.py
import numpy as np
import matplotlib.pyplot as plt

from axonscope.axons import RattayAberham, HodgkinHuxley
from axonscope.solvers import Euler

import nrv

def test_rattay_velocity_vs_NRV(save_dir="figures/physics_tests"):
    """
    Physical test of the RattayAberham axon model with Euler solver.
    A current pulse is injected and the membrane potential is recorded.
    We compare propagation velocity with NRV
    The resulting figure is saved to disk.
    """

    # --- axon parameters
    L = 1000    #in µm
    d = 0.5       # diameter in µm
    Nx = 101

    #Inject current 1ms pulse 
    t_start = 1.0
    duration = .5
    amplitude = 1

    tsim = 10.0         # total simulation time [ms]
    dt = 0.001          # time step [ms]

    vel_l = []
    vel_l_NRV = []
    d_l = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for d in d_l:
        axon = RattayAberham(L=L, d=d, Nx=Nx, celsius=37)
        axon.insert_I_Clamp(position=L / 2, t_start=t_start, duration=duration, amplitude=amplitude)

        solver = Euler()


        res = solver.solve(axon, tsim=tsim, dt=dt)


        
        # ---- NRV Axon ----
        axon_nrv = nrv.unmyelinated(
            y=0,
            z=0,
            d=d,
            L = L,
            Nsec = Nx,
            dt = dt,
            V_init = -70
        )
        axon_nrv.insert_I_Clamp(0.5, t_start, duration, amplitude)
        results_NRV = axon_nrv.simulate(t_sim=tsim)
        del axon_nrv

        results_NRV.rasterize()
        
        vel_NRV = results_NRV.get_avg_AP_speed()
        vel_axon_scope = res.average_velocity()
        print(f"d = {d} - Vel = {np.round(vel_axon_scope,3)} (AS) - {np.round(vel_NRV,3)} (NRV)")

        vel_l.append(vel_axon_scope)
        vel_l_NRV.append(vel_NRV)



    fig, ax = plt.subplots()
    ax.plot(d_l, vel_l_NRV, label = 'NRV')
    ax.plot(d_l, vel_l, label = 'AxonScope')
    ax.legend()
    ax.set_xlabel("daxon (µm)")
    ax.set_ylabel("AP Prop. Speed (m/s)")
    fig.tight_layout()
    fig.savefig(save_dir + "/euler_rattay_speed_vs_NRV.png")
    assert np.allclose(np.array(vel_l), np.array(vel_l_NRV), rtol=0.10)  # within 10%. 



def test_hh_velocity_vs_NRV(save_dir="figures/physics_tests"):
    """
    Physical test of the HodgkinHuxley axon model with Euler solver.
    A current pulse is injected and the membrane potential is recorded.
    We compare propagation velocity with NRV
    The resulting figure is saved to disk.
    """

    # --- axon parameters
    L = 1000    #in µm
    d = 0.5       # diameter in µm
    Nx = 101

    #Inject current 1ms pulse 
    t_start = 1.0
    duration = .5
    amplitude = 1

    tsim = 10.0         # total simulation time [ms]
    dt = 0.001          # time step [ms]

    vel_l = []
    vel_l_NRV = []
    d_l = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for d in d_l:
        axon = HodgkinHuxley(L=L, d=d, Nx=Nx, celsius=6.3)
        axon.insert_I_Clamp(position=L / 2, t_start=t_start, duration=duration, amplitude=amplitude)

        solver = Euler()


        res = solver.solve(axon, tsim=tsim, dt=dt)

        # ---- NRV Axon ----
        axon_nrv = nrv.unmyelinated(
            y=0,
            z=0,
            d=d,
            L = L,
            Nsec = Nx,
            dt = dt,
            V_init = -70, 
            T = 6.3,
            model = "HH"
        )
        axon_nrv.insert_I_Clamp(0.5, t_start, duration, amplitude)
        results_NRV = axon_nrv.simulate(t_sim=tsim)
        del axon_nrv

        results_NRV.rasterize()
        
        vel_NRV = results_NRV.get_avg_AP_speed()
        vel_axon_scope = res.average_velocity()
        print(f"d = {d} - Vel = {np.round(vel_axon_scope,3)} (AS) - {np.round(vel_NRV,3)} (NRV)")

        vel_l.append(vel_axon_scope)
        vel_l_NRV.append(vel_NRV)



    fig, ax = plt.subplots()
    ax.plot(d_l, vel_l_NRV, label = 'NRV')
    ax.plot(d_l, vel_l, label = 'AxonScope')
    ax.legend()
    ax.set_xlabel("daxon (µm)")
    ax.set_ylabel("AP Prop. Speed (m/s)")
    fig.tight_layout()
    fig.savefig(save_dir + "/euler_HH_speed_vs_NRV.png")

    assert np.allclose(np.array(vel_l), np.array(vel_l_NRV), rtol=0.15)  # within 15%. 
