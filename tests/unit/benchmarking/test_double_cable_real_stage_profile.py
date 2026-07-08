from __future__ import annotations

from argparse import Namespace

import numpy as np

from benchmark.analysis.double_cable_real_stage_profile import (
    _comparison_passes,
    _comparison_stats,
    _one_step_validation_group,
    _validation_atol,
    _validation_residual_tolerance,
    _validation_rtol,
)


def test_real_stage_comparison_stats_include_vm_difference():
    reference = (
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([[0.25, 0.5]], dtype=np.float32),
        np.asarray([[0.1, 0.2]], dtype=np.float32),
    )
    actual = (
        np.asarray([[1.0, 2.1]], dtype=np.float32),
        np.asarray([[0.25, 0.45]], dtype=np.float32),
        np.asarray([[0.1, 0.2]], dtype=np.float32),
    )

    stats = _comparison_stats(actual, reference)

    assert stats["all_finite"] is True
    np.testing.assert_allclose(stats["max_abs_diff"], 0.1, rtol=1e-6)
    np.testing.assert_allclose(stats["max_abs_vm_diff"], 0.15, rtol=1e-6)
    assert _comparison_passes(stats, atol=0.2, rtol=0.0)
    assert not _comparison_passes(stats, atol=0.01, rtol=0.0)


def test_real_stage_validation_defaults_follow_precision():
    fp32 = Namespace(
        preset="quick",
        precision="fp32",
        validation_atol=None,
        validation_rtol=None,
        validation_residual_tolerance=None,
    )
    fp64 = Namespace(
        preset="quick",
        precision="fp64",
        validation_atol=None,
        validation_rtol=None,
        validation_residual_tolerance=None,
    )

    assert _validation_atol(fp32) == 1e-3
    assert _validation_rtol(fp32) == 2e-4
    assert _validation_residual_tolerance(fp32) == 1e-3
    assert _validation_atol(fp64) == 1e-8
    assert _validation_rtol(fp64) == 1e-8
    assert _validation_residual_tolerance(fp64) == 1e-8


def test_one_step_validation_group_parses_current_variants():
    assert _one_step_validation_group("thomas_batched_scan_real") == "real"
    assert (
        _one_step_validation_group("jax_triton_tiled_thomas_real_precomputed_static")
        == "real_precomputed_static"
    )
    assert _one_step_validation_group("custom") == "unknown"
