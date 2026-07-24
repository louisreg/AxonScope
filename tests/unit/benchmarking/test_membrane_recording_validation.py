from __future__ import annotations

import numpy as np

from benchmark.analysis.membrane_recording_validation import compare_tensor


def test_compare_tensor_reports_strict_pointwise_match():
    values = np.asarray([0.0, 1.0, 2.0])

    comparison = compare_tensor(values, values.copy(), group="Vm")

    assert comparison["status"] == "pass"
    assert comparison["pointwise_status"] == "pass"
    assert comparison["trajectory_status"] == "pass"


def test_compare_tensor_accepts_small_normalized_solver_drift():
    cpu = np.linspace(-80.0, 60.0, 1000)
    gpu = cpu + 0.05 * np.sin(np.linspace(0.0, np.pi, cpu.size))

    comparison = compare_tensor(cpu, gpu, group="Vm")

    assert comparison["pointwise_status"] == "fail"
    assert comparison["trajectory_status"] == "pass"
    assert comparison["status"] == "pass"


def test_compare_tensor_rejects_material_trajectory_drift():
    cpu = np.linspace(-80.0, 60.0, 1000)
    gpu = cpu + 2.0

    comparison = compare_tensor(cpu, gpu, group="Vm")

    assert comparison["pointwise_status"] == "fail"
    assert comparison["trajectory_status"] == "fail"
    assert comparison["status"] == "fail"
