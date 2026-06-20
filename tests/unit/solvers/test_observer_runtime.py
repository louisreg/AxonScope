from __future__ import annotations

import numpy as np
import pytest

import axonscope as axs
from axonscope.positions import ALL, CENTER, DISTAL, Indices
from axonscope.solvers.observer_runtime import (
    VM_RASTER_OBSERVATION_KEY,
    build_vm_raster_plan,
    finalize_vm_raster_state,
    init_vm_raster_state,
    unpack_vm_raster_words,
    update_vm_raster_state_batch,
)


def test_vm_raster_plan_lowers_shared_probe_tables():
    plan = build_vm_raster_plan(
        (
            axs.analysis.Activation(
                threshold=0.0 * axs.mV,
                target=CENTER,
                name="activation",
            ),
            axs.analysis.Latency(
                threshold=0.0 * axs.mV,
                target=Indices([1, 3]),
                name="latency",
            ),
        ),
        positions_um=np.asarray([0.0, 50.0, 100.0, 150.0]),
    )

    assert plan is not None
    assert plan.row_aware is False
    assert plan.raster_count == 2
    assert plan.probe_count == 2
    np.testing.assert_array_equal(np.asarray(plan.probe_indices), [[1, 0], [1, 3]])
    np.testing.assert_array_equal(np.asarray(plan.probe_mask), [[True, False], [True, True]])
    np.testing.assert_array_equal(np.asarray(plan.original_indices), [[1, -1], [1, 3]])


def test_vm_raster_update_packs_row_aware_threshold_bits():
    plan = build_vm_raster_plan(
        (
            axs.analysis.Activation(
                threshold=0.0 * axs.mV,
                target=ALL,
                name="activation",
            ),
            axs.analysis.Latency(
                threshold=0.0 * axs.mV,
                target=DISTAL,
                name="latency",
            ),
        ),
        positions_um=np.asarray(
            [
                [0.0, 50.0, 100.0, 150.0],
                [0.0, 60.0, 120.0, 180.0],
            ]
        ),
        original_indices=np.asarray(
            [
                [0, 1, 2, 3],
                [0, 1, -1, -1],
            ],
            dtype=np.int32,
        ),
    )

    assert plan is not None
    assert plan.row_aware is True
    np.testing.assert_array_equal(
        np.asarray(plan.probe_mask[:, 0, :]),
        [
            [True, True, True, True],
            [True, True, False, False],
        ],
    )
    np.testing.assert_array_equal(
        np.asarray(plan.probe_indices[:, 1, :]),
        [[3, 0, 0, 0], [1, 0, 0, 0]],
    )

    state = init_vm_raster_state(plan, batch_size=2, nt=35)
    state = update_vm_raster_state_batch(
        state,
        vm_mV=np.asarray(
            [
                [-1.0, 5.0, -1.0, 7.0],
                [-1.0, 2.0, 99.0, 99.0],
            ],
            dtype=np.float32,
        ),
        step_index=0,
        plan=plan,
    )
    state = update_vm_raster_state_batch(
        state,
        vm_mV=np.asarray(
            [
                [3.0, 5.0, -1.0, 7.0],
                [4.0, 2.0, 99.0, 99.0],
            ],
            dtype=np.float32,
        ),
        step_index=1,
        plan=plan,
    )
    state = update_vm_raster_state_batch(
        state,
        vm_mV=np.asarray(
            [
                [3.0, -1.0, 8.0, -1.0],
                [-1.0, 2.0, 99.0, 99.0],
            ],
            dtype=np.float32,
        ),
        step_index=33,
        plan=plan,
    )

    words = np.asarray(state)
    assert words.shape == (2, 2, 4, 2)
    assert words[0, 0, 1, 0] == 0b11
    assert words[0, 0, 2, 1] == 0b10
    assert words[1, 0, 2, 0] == 0
    assert words[1, 0, 3, 1] == 0

    raster = unpack_vm_raster_words(words, nt=35)
    assert raster.shape == (2, 2, 4, 35)
    np.testing.assert_array_equal(raster[0, 0, :, 0], [False, True, False, True])
    np.testing.assert_array_equal(raster[0, 0, :, 1], [True, True, False, True])
    np.testing.assert_array_equal(raster[0, 0, :, 33], [True, False, True, False])
    np.testing.assert_array_equal(raster[1, 0, :, 33], [False, True, False, False])

    observations = finalize_vm_raster_state(plan, state, nt=35, dt_ms=0.1)
    result = observations[VM_RASTER_OBSERVATION_KEY]
    assert result.names == ("activation", "latency")
    assert result.nt == 35
    assert result.dt_ms == 0.1
    np.testing.assert_array_equal(result.unpack(), raster)


def test_vm_raster_plan_rejects_non_threshold_observers():
    with pytest.raises(NotImplementedError, match="threshold-style Vm"):
        build_vm_raster_plan(
            (axs.analysis.PeakVoltage(target=CENTER),),
            positions_um=np.asarray([0.0, 50.0, 100.0]),
        )
