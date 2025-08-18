import numpy as np
import pytest
from axonscope.axons import Passive
from axonscope.solvers import Euler

import sys
sys.path.append("./external/")
from uNRV import Axon as NRV_axon #for comparison

def test_init():
    axon = Passive(L=1_000.0, d=1)
    assert axon.L == 1_000.0
    assert axon.d == 1.0

def test_passive_geometry():
    axon = Passive(L=1_000.0, d=1.0, Nx=11)
    assert axon.L == 1_000
    assert axon.d == 1.0
    assert axon.Nx == 11
    # dx = L / (Nx - 1)
    assert np.isclose(axon.dx, 100)
    # Positions array length
    assert len(axon.x) == axon.Nx

def test_injection_index():
    axon = Passive(L=1000.0, d=1, Nx=11)
    pos_um = 450
    axon.insert_I_Clamp(position=pos_um, t_start=0.0, duration=1.0, amplitude=1.0)
    idx_expected = np.argmin(np.abs(axon.x - pos_um))
    assert hasattr(axon, "t_start_inj")
    assert idx_expected in range(axon.Nx)

def test_solver_runs():
    axon = Passive(L=1000.0, d=1, Nx=11)
    axon.insert_I_Clamp(position=500, t_start=0.0, duration=1.0, amplitude=1.0)
    solver = Euler()
    res = solver.solve(axon, tsim=1.0, dt=1e-3)
    assert res.Vm.shape[1] == axon.Nx
    assert len(res.t) == res.Vm.shape[0]
    assert np.all(np.isfinite(res.Vm))


def test_compare_nrv():
    # ---- Parameters ----
    L = 1_000        # µm
    d = 1            # µm
    Nx = 101
    Cm = 10          # µF/cm^2
    Gl = 1e-4        # S/cm^2
    EL = -70.0       # mV
    rho = 100.0      # ohm*cm
    I_inj_nA = 10    # nA
    t_start = 2      # ms
    t_on = 1         # ms
    Tsim = 10        # ms

    # ---- Python Axon ----
    axon_py = Passive(L=L, d=d, Nx=Nx, Cm=Cm, Rm=1/Gl, Ra=rho, EL=EL)
    axon_py.insert_I_Clamp(position=0.5*L, t_start=t_start, duration=t_on, amplitude=I_inj_nA)
    
    solver = Euler()
    res = solver.solve(axon_py, tsim=Tsim, dt=None)
    
    dt = res.t[1]-res.t[0]

    # ---- NRV Axon ----
    axon_nrv = NRV_axon(
        y=0,
        z=0,
        d=d,
        L=L,
        Nsec=Nx,
        passif_cable=True,
        Ra=rho,
        cm=Cm,
        e_pas=EL,
        g_pas=Gl,
        dt=dt,
        V_init=EL
    )
    axon_nrv.insert_I_Clamp(0.5, t_start, t_on, I_inj_nA)
    results_NRV = axon_nrv.simulate(t_sim=Tsim)
    Vm_nrv = results_NRV['V_mem'] 

    idx_ascope = np.argmin(np.abs(axon_py.x - L/2))
    V_ascope_center = res.Vm[:, idx_ascope]

    x_rec = results_NRV['x_rec']          # positions [µm]
    Vm_nrv = results_NRV['V_mem']     
    L_p = x_rec[-1]
    idx_nrv = np.argmin(np.abs(x_rec - L_p/2)) 
    V_nrv_center = Vm_nrv[idx_nrv, :]


    # check that the maximum difference is below tolerance
    diff_max = np.max(np.abs(V_ascope_center - V_nrv_center))
    tol = 30  # tolerance in mV
    assert diff_max < tol, f"Max difference {diff_max:.4e} exceeds tolerance {tol}"

