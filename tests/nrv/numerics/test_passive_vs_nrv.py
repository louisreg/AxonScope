# tests/physics/test_compare_nrv_physics.py
import numpy as np
import matplotlib.pyplot as plt
import pytest
from pathlib import Path
from axonscope.axons.generic import Passive
from axonscope.solvers.euler import Euler
from axonscope.stimulus import Stimulus

import sys
sys.path.append("./external/")
#from uNRV import Axon as NRV_axon #for comparison  --> importing uNRV AND nrv crashes but no passive in NRV :(

pytestmark = pytest.mark.nrv_numerics

def test_compare_nrv_physics(save_dir="figures/physics_tests"):
    # ---- Parameters ----
    L = 1_000        # µm
    d = 5            # µm
    Nx = 101
    Cm = 10.0        # µF/cm^2
    Gl = 1e-4        # S/cm^2
    EL = -70.0       # mV
    rho = 100.0      # ohm*cm
    I_inj_nA = 10    # nA
    t_start = 2      # ms
    t_on = 1         # ms
    Tsim = 10        # ms

    # ---- Create output directory ----
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ---- Python Axon ----
    
    axon_py = Passive(L=L, d=d, Nx=Nx, Cm=Cm, Rm=1/Gl, Ra=rho, EL=EL)
    axon_py.insert_I_Clamp(position=0.5*L, stimulus=Stimulus.pulse(start=t_start, duration=t_on, amplitude=I_inj_nA))
    
    solver = Euler()
    res = solver.solve(axon_py, tsim=Tsim, dt=1e-3)
    dt = (res.t[1]-res.t[0])

    
    """
    # ---- NRV Axon ----
    axon_nrv = NRV_axon(
        y=0,
        z=0,
        d=d,
        L = L,
        Nsec = Nx,
        passif_cable = True,
        Ra = rho,
        cm = Cm,
        e_pas = EL,
        g_pas = Gl,
        dt = dt,
        V_init = EL
    )
    axon_nrv.insert_I_Clamp(0.5, t_start, t_on, I_inj_nA)
    results_NRV = axon_nrv.simulate(t_sim=Tsim)
    
    """
    # ---- Choose positions along the axon ----
    x_positions = [0, L/3, L/2, 2*L/3, L]
    indices = [np.argmin(np.abs(axon_py.x - xp)) for xp in x_positions]
    
    fig, ax_x = plt.subplots(figsize=(8,5))
    for idx, xp in zip(indices, x_positions):
        ax_x.plot(res.t, res.Vm[:, idx], label=f'x = {xp:.1f} µm')
    
    ax_x.set_xlabel('Time [ms]')
    ax_x.set_ylabel('V_m [mV]')
    ax_x.legend()
    ax_x.grid(True)

    #t = results_NRV['t'].ravel()          # s'assure que t est 1D
    #x_rec = results_NRV['x_rec']          # positions [µm]
    #Vm = results_NRV['V_mem']             # shape (Nt, Nx)

    #L_p = x_rec[-1]
    #x_positions = [0, L_p/3, L_p/2, 2*L_p/3, L_p]
    #indices = [np.argmin(np.abs(x_rec - xp)) for xp in x_positions]

    
    #for idx, xp in zip(indices, x_positions):
    #    ax_x.plot(t, Vm[idx, :],'--', label=f"x = {xp:.1f} µm - NRV")
    ax_x.legend()
    filename = save_path / "axon_compare_passive_nrv.png"
    fig.savefig(filename)
    plt.close(fig)


    
##
