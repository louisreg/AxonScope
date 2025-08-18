import numpy as np
import pytest
from axonscope.axons import RattayAberham 


@pytest.fixture
def axon():
    """Return a default RattayAberham axon instance."""
    return RattayAberham(L=1000.0, d=1.0, Nx=11, Vinit=-70.0)


def test_initial_gates_in_bounds(axon):
    """Gating variables should be initialized between 0 and 1."""
    assert np.all((0 <= axon.m) & (axon.m <= 1))
    assert np.all((0 <= axon.h) & (axon.h <= 1))
    assert np.all((0 <= axon.n) & (axon.n <= 1))


def test_step_gates_evolution(axon):
    """Gates should evolve but remain in [0,1] after a step."""
    V = np.ones(axon.Nx) * -60.0  # depolarized
    m0, h0, n0 = axon.m.copy(), axon.h.copy(), axon.n.copy()

    axon.step_gates(0.1, V)  # dt=0.1 ms

    # gates should change a bit
    assert not np.allclose(m0, axon.m)
    assert not np.allclose(h0, axon.h)
    assert not np.allclose(n0, axon.n)

    # still bounded
    assert np.all((0 <= axon.m) & (axon.m <= 1))
    assert np.all((0 <= axon.h) & (axon.h <= 1))
    assert np.all((0 <= axon.n) & (axon.n <= 1))


def test_Iion_shape_and_units(axon):
    """Iion should return µA/cm² values of correct shape."""
    V = np.ones(axon.Nx) * -65.0
    axon.step_gates(0.1, V)  # update gates first

    I = axon.Iion(V=V)

    # correct shape
    assert I.shape == (axon.Nx,)

    # not NaN or inf
    assert np.all(np.isfinite(I))

    # values should be in a physiological range (say -5000 to +5000 µA/cm²)
    assert np.all((-5000 < I) & (I < 5000))
