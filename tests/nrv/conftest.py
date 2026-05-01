"""
Shared fixtures and constants for the NRV validation suite.

The suite is organized around three model-validation families:
- intracellular
- velocity_vs_diameter
- extracellular

Additional NRV-adjacent solver diagnostics live under `tests/nrv/numerics`.
"""

import pytest

# ── Tolerance constants ────────────────────────────────────────────────────────
ATOL_CURRENT_uAcm2 = 2.0   # µA/cm²  — per-channel current comparison
RTOL_VELOCITY = 0.10        # 10 %    — conduction velocity
RTOL_VELOCITY_STRICT = 0.05 # 5 %     — for well-validated models
RTOL_THRESHOLD = 0.10       # 10 %    — activation threshold current
ATOL_RATE_ms = 1e-3         # ms⁻¹   — gate kinetics rates


def pytest_configure(config):
    config.addinivalue_line("markers", "nrv: requires NRV installed (slow)")
    config.addinivalue_line(
        "markers",
        "nrv_intracellular: intracellular stimulation comparisons vs NRV",
    )
    config.addinivalue_line(
        "markers",
        "nrv_velocity: conduction velocity vs diameter comparisons vs NRV",
    )
    config.addinivalue_line(
        "markers",
        "nrv_extracellular: extracellular stimulation comparisons vs NRV",
    )
    config.addinivalue_line(
        "markers",
        "nrv_numerics: NRV-adjacent numerical and solver diagnostics",
    )


def pytest_collection_modifyitems(items):
    marker_by_dir = {
        "intracellular": pytest.mark.nrv_intracellular,
        "velocity_vs_diameter": pytest.mark.nrv_velocity,
        "extracellular": pytest.mark.nrv_extracellular,
        "numerics": pytest.mark.nrv_numerics,
    }

    for item in items:
        item.add_marker(pytest.mark.nrv)
        for dirname, marker in marker_by_dir.items():
            if dirname in item.path.parts:
                item.add_marker(marker)
                break
