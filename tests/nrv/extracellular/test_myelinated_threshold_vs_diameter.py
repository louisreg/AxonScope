"""
Extracellular — MRG myelinated fiber activation threshold monotonicity.

Protocol:
  - Point-source electrode, fixed position (100 µm above fiber center).
  - Cathodic monophasic pulse, fixed duration (100 µs).
  - Binary search for threshold amplitude that triggers 1 AP propagation.
  - Diameters: [5.7, 7.3, 8.7, 10.0, 11.5, 14.0] µm (MRG standard diameters).

The electrode is explicitly aligned with the central node of Ranvier for each
diameter. This keeps the protocol focused on diameter-dependent excitability
rather than on electrode/node phase differences.
"""

import numpy as np
import pytest

from axonscope import AxonInstance, S_per_m, ms, um
from axonscope.analysis import rasterize
from axonscope.axons.myelinated import MRG
from axonscope.analytical import PointSourceElectrode, point_source_stimulation
from axonscope.stimulation import Stimulus
from axonscope.solvers.crank_nicholson import CrankNicholson
from tests.nrv._helpers import axonscope_x_um

DIAMETERS_UM = [5.7, 7.3, 8.7, 10.0, 11.5, 14.0]
ELECTRODE_Y_UM = 100.0
PULSE_DURATION_MS = 0.1
SIGMA_S_M = 0.2
NODES = 21
TSIM = 5.0
DT = 0.005


def _central_node_x_um(ax: MRG) -> float:
    node_indices = np.asarray(ax.node_indices, dtype=int)
    center_node = int(node_indices[node_indices.shape[0] // 2])
    return float(axonscope_x_um(ax)[center_node])


def _has_ap(diameter_um: float, amp_uA: float) -> bool:
    ax_copy = MRG(diameter=diameter_um * um, nodes=NODES)
    x0_um = _central_node_x_um(ax_copy)
    electrode = PointSourceElectrode(x=x0_um * um, y=ELECTRODE_Y_UM * um, z=0.0 * um)
    stim = Stimulus.pulse(
        start=1.0 * ms,
        amplitude=amp_uA * 1e-6,
        duration=PULSE_DURATION_MS * ms,
    )
    sim = AxonInstance(ax_copy)
    sim.add_extracellular_stimulation(
        stimulation=point_source_stimulation(
            electrode,
            ax_copy.layout.position_values(unit=um) * um,
            stimulus=stim,
            sigma=SIGMA_S_M * S_per_m,
        ),
        replace=True,
    )
    res = CrankNicholson().solve(sim, tsim=TSIM, dt=DT)
    tAP, _ = rasterize(res)
    return len(tAP) >= 3


def _binary_search_threshold(d: float, lo: float = 1.0, hi: float = 500.0, tol: float = 1.0) -> float:
    for _ in range(15):
        mid = (lo + hi) / 2.0
        if _has_ap(d, mid):
            hi = mid
        else:
            lo = mid
        if (hi - lo) < tol:
            break
    return (lo + hi) / 2.0


@pytest.mark.nrv_extracellular
def test_myelinated_threshold_decreases_with_diameter_when_node_aligned():
    """Larger node-aligned MRG fibers should have lower activation thresholds."""
    thresholds = [_binary_search_threshold(d) for d in DIAMETERS_UM]

    assert np.isfinite(thresholds).all()
    assert min(thresholds) > 1.0
    assert max(thresholds) < 500.0

    for i in range(len(thresholds) - 1):
        assert thresholds[i] >= thresholds[i + 1] * 0.85, (
            f"Threshold did not decrease from d={DIAMETERS_UM[i]} um "
            f"({thresholds[i]:.1f} uA) to d={DIAMETERS_UM[i+1]} um ({thresholds[i+1]:.1f} uA)"
        )
