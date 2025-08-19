import numpy as np
import pytest
from axonscope.simresult import SimResult

class DummyAxon:
    def __init__(self, Nx):
        self.Nx = Nx
        self.x = np.linspace(0, 1000, Nx)  # 0 → 1000 µm

@pytest.fixture
def fake_result():
    t = np.linspace(0, 50, 5001)
    Vm = np.zeros((len(t), 3))  # 3 compartments
    Vm[:, 0] = np.exp(-0.5*((t-10)/0.5)**2)*80 - 70
    Vm[:, 1] = np.exp(-0.5*((t-30)/0.5)**2)*80 - 70
    axon = DummyAxon(Nx=3)
    return SimResult(axon=axon, Vm=Vm, t=t)

def test_rasterize_detects_spikes(fake_result):
    tAP, xAP = fake_result.rasterize(threshold=0.0, min_distance=2.0)
    assert np.allclose(tAP, [10, 30], atol=0.5)
    assert np.allclose(xAP, fake_result.axon.x[:2])

def test_rasterplot_uses_axons_x(fake_result):
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    fake_result.rasterplot(ax, threshold=0.0, min_distance=2.0)

    # y-label should contain "Axon position"
    assert "Axon position" in ax.get_ylabel()
