# tests/physics/test_compare_nrv_physics.py
import numpy as np
import matplotlib.pyplot as plt
import pytest
from pathlib import Path
from axonscope import AxonSimulation
from axonscope import membranes
from axonscope.axons import Axon, Layout, Section
from axonscope.solvers.crank_nicholson import CrankNicholson
from axonscope.stimulation import Stimulus
from axonscope.utils import units

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
    
    axon_py = Axon(
        layout=Layout.single_uniform(
            Section(
                "passive",
                membrane=membranes.Passive(Rm=1 / Gl, EL=EL),
                diameter=units.Q_(d, "micrometer"),
                Ra=units.Q_(rho, "ohm * centimeter"),
                Cm=units.Q_(Cm, "microfarad / centimeter ** 2"),
            ),
            length=units.Q_(L, "micrometer"),
            compartments=Nx,
        ),
        v_init=units.Q_(EL, "millivolt"),
    )
    sim_py = AxonSimulation(axon_py)
    sim_py.add_current_clamp(position_um=0.5*L, current=Stimulus.pulse(start=t_start, duration=t_on, amplitude=I_inj_nA))
    
    solver = CrankNicholson()
    res = solver.solve(sim_py, tsim=Tsim, dt=1e-3)
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
    x_um = axon_py.layout.position_values(unit="micrometer")
    indices = [np.argmin(np.abs(x_um - xp)) for xp in x_positions]
    
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
