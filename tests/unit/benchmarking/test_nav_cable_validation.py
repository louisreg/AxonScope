from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import axonfleet as axs


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmark"
    / "curves"
    / "nav_cable_validation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("nav_cable_validation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_campaign_describes_both_canonical_cable_routes():
    campaign = _load_module()

    description = campaign.describe_workloads()

    assert description["composition"]["components"] == [
        "nav16",
        "borg_kdr",
        "passive",
    ]
    assert set(description["cases"]) == {"single", "double"}
    assert description["validation"] == [
        "waveform",
        "threshold",
        "velocity",
        "recruitment",
    ]


def test_campaign_builds_one_single_and_one_double_cable_path():
    campaign = _load_module()

    single = campaign.build_axon("single")
    double = campaign.build_axon("double")

    assert single.formulation is axs.axons.CableFormulation.SINGLE_CABLE
    assert double.formulation is axs.axons.CableFormulation.DOUBLE_CABLE
    assert single.layout.compartments == 201
    assert double.layout.compartments == 111
    assert double.nodes == 11


def test_campaign_uses_one_typed_extracellular_drive_update():
    campaign = _load_module()

    simulation = campaign.build_simulation("single")
    update = campaign.waveform_update()

    assert isinstance(update, axs.protocols.ExtracellularWaveformUpdate)
    assert simulation.extracellular_stimulation is not None
    assert len(simulation.extracellular_stimulation.drives) == 1
    assert campaign.main(["--dry-run"]) == 0


def test_campaign_acceptance_rejects_non_monotone_recruitment():
    campaign = _load_module()
    summary = {
        "waveform": {},
        "threshold": {},
        "recruitment": {},
    }
    for cable in ("single", "double"):
        summary["waveform"][cable] = {
            "control_max_abs_drift_mV": 1.0,
            "distal_activated": True,
            "distal_peak_mV": 30.0,
            "velocity_m_s": 1.0,
        }
        summary["threshold"][cable] = {
            "status": "threshold",
            "threshold_uA": 5.0,
        }
        summary["recruitment"][cable] = {"fraction": [0.0, 1.0, 0.5, 1.0]}

    acceptance = campaign.validate_summary(summary)

    assert acceptance["accepted"] is False
    assert acceptance["checks"]["single.monotone_recruitment"] is False
