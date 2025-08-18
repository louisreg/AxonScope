import numpy as np
import pytest
from axonscope.axons import RattayAberham
from axonscope.solvers import Euler
from axonscope.simresult import SimResult

def test_compute_propagation_velocity():
    # --- axon parameters
    L = 1000    #in µm
    d = 0.5       # diameter in µm
    Nx = 101
    axon = RattayAberham(L=L, d=d, Nx=Nx, celsius=37)

    #Inject current 1ms pulse 
    t_start = 1.0
    duration = 1.0
    amplitude = 2
    axon.insert_I_Clamp(position=L / 2, t_start=t_start, duration=duration, amplitude=amplitude)

    # --- solver setup
    solver = Euler()
    tsim = 10.0         # total simulation time [ms]
    dt = 0.001          # time step [ms]

    simres = solver.solve(axon, tsim=tsim, dt=dt)

    velocity_true = 10

    # --- Rasterize & compute velocity ---
    raster = simres.rasterize()
    velocity_est = simres.average_velocity()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    simres.rasterplot(ax)
    plt.show()
    print(velocity_est)

    # --- Assertions ---
    assert velocity_est is not None
    assert np.isfinite(velocity_est)
    assert np.allclose(velocity_est, velocity_true, rtol=0.05)  # within 5%

    # Optional: check raster
    assert len(raster) == Nx
    for spikes in raster:
        assert isinstance(spikes, list)
