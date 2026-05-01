import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from axonscope.simresult import SimResult
from axonscope.axons.unmyelinated import RattayAberham
from axonscope.solvers.Euler import Euler


class _DummyAxon:
    def __init__(self, Nx):
        self.Nx = Nx
        self.x = np.linspace(0, 1000, Nx)


@pytest.fixture
def fake_result():
    t = np.linspace(0, 50, 5001)
    Vm = np.zeros((len(t), 3))
    Vm[:, 0] = np.exp(-0.5 * ((t - 10) / 0.5) ** 2) * 80 - 70
    Vm[:, 1] = np.exp(-0.5 * ((t - 30) / 0.5) ** 2) * 80 - 70
    return SimResult(axon=_DummyAxon(Nx=3), Vm=Vm, t=t)


# ── rasterize ─────────────────────────────────────────────────────────────────

def test_rasterize_detects_spikes(fake_result):
    tAP, xAP = fake_result.rasterize(threshold=0.0, min_distance=2.0)
    assert np.allclose(tAP, [10, 30], atol=0.5)
    assert np.allclose(xAP, fake_result.axon.x[:2])


def test_rasterplot_uses_axons_x(fake_result):
    _, ax = plt.subplots()
    fake_result.rasterplot(ax, threshold=0.0, min_distance=2.0)
    assert "Axon position" in ax.get_ylabel()
    plt.close("all")


# ── average_velocity ──────────────────────────────────────────────────────────

def test_compute_propagation_velocity():
    L, d, Nx = 1000, 0.5, 101
    axon = RattayAberham(L=L, d=d, Nx=Nx, celsius=37)
    axon.insert_I_Clamp(position=L / 2, t_start=1.0, duration=1.0, amplitude=2)

    simres = Euler().solve(axon, tsim=10.0, dt=0.001)
    velocity = simres.average_velocity()

    assert velocity is not None
    assert np.isfinite(velocity)
    assert 0.44 < velocity < 0.50
