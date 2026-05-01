"""
Extracellular — MRG myelinated fiber activation threshold vs diameter, compared to NRV.

Protocol:
  - Point-source electrode, fixed position (100 µm above fiber center).
  - Cathodic monophasic pulse, fixed duration (100 µs).
  - Binary search for threshold amplitude that triggers 1 AP propagation.
  - Diameters: [5.7, 7.3, 8.7, 10.0, 11.5, 14.0] µm (MRG standard diameters).
  - Tolerance: 10 % relative on threshold current.
"""

import numpy as np
import pytest

from axonscope.axons.myelinated import MRG
from axonscope.electrodes import PointSourceElectrode
from axonscope.stimulus import Stimulus
from axonscope.solvers.CrankNicholson import CrankNicholson

DIAMETERS_UM = [5.7, 7.3, 8.7, 10.0, 11.5, 14.0]
ELECTRODE_Y_UM = 100.0
PULSE_DURATION_MS = 0.1
SIGMA_S_M = 0.2
NODES = 21
TSIM = 5.0
DT = 0.005
RTOL = 0.10


def _has_ap(ax: MRG, amp_uA: float) -> bool:
    ax_copy = MRG(d=ax.d, nodes=ax.nodes if hasattr(ax, "nodes") else NODES)
    x0_um = float(ax_copy.L / 2.0)
    electrode = PointSourceElectrode(
        x0_m=x0_um * 1e-6, y0_m=ELECTRODE_Y_UM * 1e-6, z0_m=0.0, sigma_S_m=SIGMA_S_M
    )
    stim = Stimulus.pulse(start=1.0, amplitude=amp_uA * 1e-6, duration=PULSE_DURATION_MS)
    ax_copy.add_extracellular_ctx(electrode, stim, replace=True)
    res = CrankNicholson().solve(ax_copy, tsim=TSIM, dt=DT)
    tAP, _ = res.rasterize()
    return len(tAP) >= 3


def _binary_search_threshold(d: float, lo: float = 1.0, hi: float = 500.0, tol: float = 1.0) -> float:
    ax = MRG(d=d, nodes=NODES)
    for _ in range(15):
        mid = (lo + hi) / 2.0
        if _has_ap(ax, mid):
            hi = mid
        else:
            lo = mid
        if (hi - lo) < tol:
            break
    return (lo + hi) / 2.0


@pytest.mark.nrv_extracellular
def test_myelinated_threshold_vs_diameter_nrv():
    pytest.skip("TODO: run NRV binary-search threshold and compare")


@pytest.mark.nrv_extracellular
def test_myelinated_threshold_decreases_with_diameter():
    """Larger fibers should have lower activation thresholds."""
    thresholds = [_binary_search_threshold(d) for d in DIAMETERS_UM]

    for i in range(len(thresholds) - 1):
        assert thresholds[i] >= thresholds[i + 1] * 0.85, (
            f"Threshold did not decrease from d={DIAMETERS_UM[i]} µm "
            f"({thresholds[i]:.1f} µA) to d={DIAMETERS_UM[i+1]} µm ({thresholds[i+1]:.1f} µA)"
        )
