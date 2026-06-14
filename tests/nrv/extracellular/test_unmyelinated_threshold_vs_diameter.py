"""
Extracellular — Tigerholm C-fiber activation threshold monotonicity.

Protocol:
  - Point-source electrode, 100 µm above fiber center.
  - Cathodic monophasic pulse, 100 µs.
  - Binary search for threshold amplitude.
  - Diameters: [0.3, 0.5, 0.7, 0.8, 1.0, 1.2] µm.
"""

import numpy as np
import pytest

from axonscope import AxonInstance, S_per_m, degC, ms, um
from axonscope.analysis import rasterize
from axonscope.axons.unmyelinated import Tigerholm
from axonscope.stimulation import AnalyticalExtracellularContext, PointSourceElectrode
from axonscope.stimulation import Stimulus
from axonscope.solvers.crank_nicholson import CrankNicholson

DIAMETERS_UM = [0.3, 0.5, 0.7, 0.8, 1.0, 1.2]
ELECTRODE_Y_UM = 100.0
PULSE_DURATION_MS = 0.1
SIGMA_S_M = 0.2
L_UM = 5000.0
NX = 101
CELSIUS = 37.0 * degC
TSIM = 30.0
DT = 0.025


def _has_ap(d: float, amp_uA: float) -> bool:
    axon = Tigerholm(length=L_UM * um, diameter=d * um, compartments=NX, celsius=CELSIUS)
    x0_um = L_UM / 2.0
    electrode = PointSourceElectrode(x=x0_um * um, y=ELECTRODE_Y_UM * um, z=0.0 * um)
    stim = Stimulus.pulse(
        start=5.0 * ms,
        amplitude=amp_uA * 1e-6,
        duration=PULSE_DURATION_MS * ms,
    )
    sim = AxonInstance(axon)
    sim.add_extracellular_context(
        context=AnalyticalExtracellularContext(
            electrodes=[electrode.with_stimulus(stim)],
            sigma=SIGMA_S_M * S_per_m,
        ),
        replace=True,
    )
    res = CrankNicholson().solve(sim, tsim=TSIM, dt=DT)
    tAP, _ = rasterize(res)
    return len(tAP) >= 3


def _binary_search_threshold(d: float, lo: float = 1.0, hi: float = 1000.0, tol: float = 5.0) -> float:
    for _ in range(12):
        mid = (lo + hi) / 2.0
        if _has_ap(d, mid):
            hi = mid
        else:
            lo = mid
        if (hi - lo) < tol:
            break
    return (lo + hi) / 2.0


@pytest.mark.nrv_extracellular
def test_unmyelinated_threshold_decreases_with_diameter():
    """Larger C-fibers should have lower activation thresholds."""
    thresholds = [_binary_search_threshold(d) for d in DIAMETERS_UM]

    for i in range(len(thresholds) - 1):
        assert thresholds[i] >= thresholds[i + 1] * 0.85, (
            f"Threshold did not decrease from d={DIAMETERS_UM[i]} µm "
            f"({thresholds[i]:.1f} µA) to d={DIAMETERS_UM[i+1]} µm ({thresholds[i+1]:.1f} µA)"
        )
