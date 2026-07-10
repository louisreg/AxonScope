from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import axonscope as axs
from axonscope.runtime.jax import recording_lowering
from axonscope.positions import ALL, CENTER, DISTAL, Indices
from axonscope.results import VM_RASTER_OBSERVATION_KEY, unpack_vm_raster_words
from axonscope.runtime.jax.observer_runtime import (
    build_vm_raster_plan,
    combine_vm_raster_chunk_states,
    finalize_vm_raster_state,
    init_vm_raster_state,
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


def test_vm_raster_plan_cache_survives_stimulation_replacement():
    recording_lowering._VM_RASTER_PLAN_CACHE.clear()
    axons = (object(), object())
    solver_axons = (
        SimpleNamespace(n_compartments=3),
        SimpleNamespace(n_compartments=3),
    )
    cohort = SimpleNamespace(
        group_id=0,
        mode="single",
        size=2,
        nx=3,
        geometry_shared=True,
        has_padding=False,
        axons=axons,
        solver_axons=solver_axons,
        stimulations=((object(),), (object(),)),
        x_positions_m=np.asarray(
            [
                [0.0, 5.0e-5, 1.0e-4],
                [0.0, 5.0e-5, 1.0e-4],
            ],
            dtype=float,
        ),
        axon_y_um=np.asarray([0.0, 10.0], dtype=float),
        axon_z_um=np.asarray([50.0, 60.0], dtype=float),
    )
    observers = (
        axs.analysis.Activation(
            threshold=0.0 * axs.mV,
            target=DISTAL,
            name="activation",
        ),
    )

    first = recording_lowering.lower_observers_for_cohort(
        observers,
        cohort=cohort,
        dtype=np.float32,
        prefer_vm_raster=True,
    )
    refreshed = SimpleNamespace(
        **{
            **cohort.__dict__,
            "stimulations": ((object(),), (object(),)),
        }
    )
    second = recording_lowering.lower_observers_for_cohort(
        observers,
        cohort=refreshed,
        dtype=np.float32,
        prefer_vm_raster=True,
    )

    assert first is second


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


def test_vm_raster_combines_local_chunk_states_across_word_boundaries():
    chunk0 = np.zeros((1, 1, 1, 2), dtype=np.uint32)
    chunk1 = np.zeros((1, 1, 1, 2), dtype=np.uint32)
    chunk2 = np.zeros((1, 1, 1, 1), dtype=np.uint32)
    chunk0[0, 0, 0, 0] |= np.uint32(1 << 0)
    chunk0[0, 0, 0, 0] |= np.uint32(1 << 31)
    chunk0[0, 0, 0, 1] |= np.uint32(1 << 8)
    chunk1[0, 0, 0, 0] |= np.uint32(1 << 0)
    chunk1[0, 0, 0, 0] |= np.uint32(1 << 9)
    chunk1[0, 0, 0, 1] |= np.uint32(1 << 9)
    chunk2[0, 0, 0, 0] |= np.uint32(1 << 4)

    combined = combine_vm_raster_chunk_states(
        [chunk0, chunk1, chunk2],
        starts=[0, 40, 90],
        lengths=[40, 50, 10],
        nt=100,
    )

    raster = unpack_vm_raster_words(np.asarray(combined), nt=100)
    expected = np.zeros((1, 1, 1, 100), dtype=bool)
    expected[0, 0, 0, [0, 31, 40, 49, 81, 94]] = True
    np.testing.assert_array_equal(raster, expected)


def test_vm_raster_keeps_single_full_chunk_state():
    chunk = np.zeros((1, 1, 1, 2), dtype=np.uint32)
    chunk[0, 0, 0, 0] |= np.uint32(1 << 3)
    chunk[0, 0, 0, 1] |= np.uint32(1 << 1)

    combined = combine_vm_raster_chunk_states([chunk], starts=[0], lengths=[64], nt=64)

    np.testing.assert_array_equal(np.asarray(combined), chunk)


def test_vm_raster_plan_rejects_non_threshold_observers():
    with pytest.raises(NotImplementedError, match="threshold-style Vm"):
        build_vm_raster_plan(
            (axs.analysis.PeakVoltage(target=CENTER),),
            positions_um=np.asarray([0.0, 50.0, 100.0]),
        )
