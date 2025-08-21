# test_cranknicholson.py
import numpy as np
import pytest
from axonscope.solvers import CrankNicholson
from axonscope.simresult import SimResult 

class MockAxon:
    """
    Minimal fake Axon object to test the solver.
    Provides required attributes and methods.
    """
    def __init__(self, Nx=5, Cm=1.0, D=0.1, dx_cm=1e-4, Vinit=-65.0):
        self.Nx = Nx
        self.Cm = Cm
        self.D = D
        self.dx_cm = dx_cm
        self.Vinit = Vinit
        self.calls = {
            "Iinj": 0,
            "Iion": 0,
            "half_step_gates": 0
        }

    def Iinj_uAcm2(self, t):
        """Constant injection current for testing"""
        self.calls["Iinj"] += 1
        return np.zeros(self.Nx)

    def Iion(self, V):
        """No ionic current: purely passive cable"""
        self.calls["Iion"] += 1
        return np.zeros_like(V)

    def half_step_gates(self, dt, V):
        """Just record calls (HH dynamics skipped)"""
        self.calls["half_step_gates"] += 1
        return None


def test_shape_and_types():
    axon = MockAxon(Nx=5)
    solver = CrankNicholson()
    tsim = 1.0
    dt = 0.1

    result = solver.solve(axon, tsim, dt)

    assert isinstance(result, SimResult)
    assert result.Vm.shape[0] == int(np.ceil(tsim/dt))
    assert result.Vm.shape[1] == axon.Nx
    assert np.allclose(result.Vm[:, 0], axon.Vinit)
    assert np.allclose(result.Vm[:, -1], axon.Vinit)


def test_no_current_constant_voltage():
    axon = MockAxon(Nx=5)
    solver = CrankNicholson()
    tsim = 1.0
    dt = 0.1

    result = solver.solve(axon, tsim, dt)

    # If Iinj = 0 and Iion = 0, voltage should remain constant
    assert np.allclose(result.Vm, axon.Vinit)


def test_function_calls():
    axon = MockAxon(Nx=5)
    solver = CrankNicholson()
    tsim = 1.0
    dt = 0.1

    _ = solver.solve(axon, tsim, dt)

    # Ensure gating and currents were evaluated each timestep
    Nt = int(np.ceil(tsim/dt))
    assert axon.calls["Iinj"] == Nt
    assert axon.calls["Iion"] == Nt
    assert axon.calls["half_step_gates"] == Nt


def test_stability_small_dt():
    axon = MockAxon(Nx=10)
    solver = CrankNicholson()
    tsim = 5.0
    dt = 0.01

    result = solver.solve(axon, tsim, dt)

    # Ensure solution is finite and bounded
    assert np.isfinite(result.Vm).all()
    assert np.all(result.Vm <= 1000.0)
    assert np.all(result.Vm >= -1000.0)


