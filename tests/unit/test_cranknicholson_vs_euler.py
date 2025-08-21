import numpy as np
import pytest

from axonscope.solvers import CrankNicholson, Euler
from axonscope.axons import Passive, HodgkinHuxley, RattayAberham

@pytest.mark.parametrize("axon_class", [
    Passive,
    HodgkinHuxley,
    RattayAberham,
])
def test_cn_vs_euler(axon_class):
    """
    Compare Crank-Nicholson vs Euler solvers on different axon models.
    Compares all spatial points along the axon using interpolation.
    """

    # --- Axon parameters ---
    L = 1000    # length in µm
    d = 1       # diameter in µm
    Nx = 101
    axon = axon_class(L=L, d=d, Nx=Nx)

    # Inject 1 ms current pulse in the middle
    t_start = 1.0
    duration = 1.0
    amplitude = 5.0
    axon.insert_I_Clamp(position=L / 2, t_start=t_start, duration=duration, amplitude=amplitude)

    # --- Simulation parameters ---
    tsim = 25.0

    # Crank-Nicholson
    solver_CN = CrankNicholson()
    dt_CN = 0.001
    res_CN = solver_CN.solve(axon, tsim=tsim, dt=dt_CN)

    # Euler
    solver_Euler = Euler()
    dt_Euler = 0.001
    res_EULER = solver_Euler.solve(axon, tsim=tsim, dt=dt_Euler)

    # --- Sanity checks ---
    assert not np.allclose(res_CN.Vm, res_CN.Vm[0, 0])
    assert not np.allclose(res_EULER.Vm, res_EULER.Vm[0, 0])

    diff = np.abs(res_CN.Vm - res_EULER.Vm)
    #print(diff.max())

    assert diff.max() < 30 #yes, 30mV .....
