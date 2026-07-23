"""Shared fixtures and constants for the NRV validation suite.

The suite is organized around three model-validation families:
- intracellular
- velocity_vs_diameter
- extracellular

Additional NRV-adjacent solver diagnostics live under `tests/nrv/numerics`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

def _make_path_writable(path: str | None) -> str | None:
    if path is None:
        return path

    target = Path(path).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=True)
        return str(target)
    except OSError:
        fallback_dir = Path(tempfile.gettempdir()) / "axonfleet-nrv-tests"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return str(fallback_dir / target.name)


def _redirect_nrv_reporter_log() -> None:
    """Keep NRV imports independent from write permissions in site-packages."""

    try:
        import pyswarms.utils as pyswarms_utils
        from pyswarms.utils import Reporter
    except Exception:
        return

    class TestReporter(Reporter):
        def __init__(self, *args, **kwargs):
            kwargs["log_path"] = _make_path_writable(kwargs.get("log_path"))
            super().__init__(*args, **kwargs)

    pyswarms_utils.Reporter = TestReporter


def pytest_configure(config):
    _redirect_nrv_reporter_log()


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
