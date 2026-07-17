from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import axonscope as axs
from axonscope.runtime.jax.recording import lowering as recording_lowering
from axonscope.positions import ALL, CENTER, DISTAL, Indices
from axonscope.results import VM_RASTER_OBSERVATION_KEY, unpack_vm_raster_words
from axonscope.runtime.jax.recording.observer import (
    build_threshold_observer_plan,
    combine_threshold_observer_chunk_states,
    finalize_threshold_observer_state,
    init_threshold_observer_state,
    trim_threshold_observer_state,
    update_threshold_observer_state_batch_from_tables,
)


def test_vm_raster_trim_clears_padded_tail_bits():
    state = np.full((1, 1, 1, 3), np.uint32(0xFFFFFFFF), dtype=np.uint32)

    trimmed = trim_threshold_observer_state(state, nt=35)

    assert trimmed.shape == (1, 1, 1, 2)
    assert int(np.asarray(trimmed)[0, 0, 0, 0]) == 0xFFFFFFFF
    assert int(np.asarray(trimmed)[0, 0, 0, 1]) == 0b111


def test_threshold_observer_plan_lowers_shared_probe_tables():
    plan = build_threshold_observer_plan(
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
    assert plan.definition_count == 2
    assert plan.probe_count == 2
    np.testing.assert_array_equal(np.asarray(plan.probe_indices), [[1, 0], [1, 3]])
    np.testing.assert_array_equal(np.asarray(plan.probe_mask), [[True, False], [True, True]])
    np.testing.assert_array_equal(np.asarray(plan.original_indices), [[1, -1], [1, 3]])
    np.testing.assert_array_equal(plan.probe_indices_host, [[1, 0], [1, 3]])
    np.testing.assert_array_equal(plan.probe_mask_host, [[True, False], [True, True]])
    assert plan.probe_indices_host.flags.writeable is False
    assert plan.probe_mask_host.flags.writeable is False


def test_activation_only_plan_retains_bool_and_respects_blanking_across_chunks():
    activation = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=0.15 * axs.ms,
        target=Indices([0, 2]),
        name="activation",
    )
    plan = build_threshold_observer_plan(
        (activation,),
        positions_um=np.asarray([0.0, 50.0, 100.0]),
    )

    assert plan is not None
    assert plan.retention == "activation"
    first = init_threshold_observer_state(plan, batch_size=1, nt=2)
    assert first.shape == (1, 1)
    first = update_threshold_observer_state_batch_from_tables(
        first,
        vm_mV=np.asarray([[5.0, -1.0, -1.0]], dtype=np.float32),
        step_index=0,
        probe_indices=np.asarray(plan.probe_indices)[None, ...],
        probe_mask=np.asarray(plan.probe_mask)[None, ...],
        thresholds_mV=plan.thresholds_mV,
        blanking_ms=plan.blanking_ms,
        dt_ms=0.1,
        retention=plan.retention,
    )
    np.testing.assert_array_equal(first, [[False]])

    second = init_threshold_observer_state(plan, batch_size=1, nt=2)
    second = update_threshold_observer_state_batch_from_tables(
        second,
        vm_mV=np.asarray([[-1.0, -1.0, 5.0]], dtype=np.float32),
        step_index=1,
        probe_indices=np.asarray(plan.probe_indices)[None, ...],
        probe_mask=np.asarray(plan.probe_mask)[None, ...],
        thresholds_mV=plan.thresholds_mV,
        blanking_ms=plan.blanking_ms,
        dt_ms=0.1,
        retention=plan.retention,
    )
    combined = combine_threshold_observer_chunk_states(
        [first, second],
        starts=[0, 1],
        lengths=[1, 1],
        nt=2,
        retention=plan.retention,
    )
    observations = finalize_threshold_observer_state(plan, combined, nt=2, dt_ms=0.1)

    assert tuple(observations) == ("activation",)
    np.testing.assert_array_equal(observations["activation"].values, [True])


def test_latency_only_plan_retains_first_crossing_across_chunks():
    latency = axs.analysis.Latency(
        threshold=0.0 * axs.mV,
        blanking=0.25 * axs.ms,
        target=Indices([0, 2]),
        name="latency",
    )
    plan = build_threshold_observer_plan(
        (latency,),
        positions_um=np.asarray([0.0, 50.0, 100.0]),
    )

    assert plan is not None
    assert plan.retention == "first_crossing"
    chunks = []
    for values, local_step, blanking_ms in (
        ([5.0, -1.0, -1.0], 1, 0.25),
        ([-1.0, -1.0, 5.0], 1, 0.05),
        ([5.0, -1.0, -1.0], 0, -0.15),
    ):
        state = init_threshold_observer_state(plan, batch_size=1, nt=2)
        state = update_threshold_observer_state_batch_from_tables(
            state,
            vm_mV=np.asarray([values], dtype=np.float32),
            step_index=local_step,
            probe_indices=np.asarray(plan.probe_indices)[None, ...],
            probe_mask=np.asarray(plan.probe_mask)[None, ...],
            thresholds_mV=plan.thresholds_mV,
            blanking_ms=np.asarray([blanking_ms]),
            dt_ms=0.1,
            retention=plan.retention,
        )
        chunks.append(state)

    combined = combine_threshold_observer_chunk_states(
        chunks,
        starts=[0, 2, 4],
        lengths=[2, 2, 1],
        nt=5,
        retention=plan.retention,
    )
    observations = finalize_threshold_observer_state(plan, combined, nt=5, dt_ms=0.1)

    result = observations["latency"]
    np.testing.assert_allclose(result.values, [0.4])
    assert result.statuses == (axs.analysis.AnalysisStatus.VALID,)
    assert result.unit == "millisecond"


def test_latency_first_crossing_reports_undetermined_and_ignores_padded_tail():
    latency = axs.analysis.Latency(
        threshold=0.0 * axs.mV,
        target=CENTER,
    )
    plan = build_threshold_observer_plan(
        (latency,),
        positions_um=np.asarray([0.0, 50.0, 100.0]),
    )
    assert plan is not None
    missed = init_threshold_observer_state(plan, batch_size=1, nt=2)
    padded = np.asarray([[1]], dtype=np.int32)

    combined = combine_threshold_observer_chunk_states(
        [missed, padded],
        starts=[0, 2],
        lengths=[2, 1],
        nt=3,
        retention=plan.retention,
    )
    result = finalize_threshold_observer_state(plan, combined, nt=3, dt_ms=0.1)[
        "latency"
    ]

    assert np.isnan(result.values[0])
    assert result.statuses == (axs.analysis.AnalysisStatus.UNDETERMINED,)
    assert result.messages == ("threshold was not crossed at the requested target.",)


def test_spike_summary_counts_rearmed_crossings_with_blanking_and_refractory():
    spike_count = axs.analysis.SpikeCount(
        threshold=0.0 * axs.mV,
        reset_threshold=-20.0 * axs.mV,
        blanking=0.15 * axs.ms,
        refractory=0.25 * axs.ms,
        target=CENTER,
    )
    plan = build_threshold_observer_plan(
        (spike_count,),
        positions_um=np.asarray([0.0, 50.0, 100.0]),
    )

    assert plan is not None
    assert plan.retention == "spike_summary"
    state = init_threshold_observer_state(plan, batch_size=1, nt=8)
    assert state.shape == (1, 1, 1, 4)
    center_values = (5.0, 5.0, -30.0, 5.0, -30.0, 5.0, -30.0, 5.0)
    for step, center_value in enumerate(center_values):
        state = update_threshold_observer_state_batch_from_tables(
            state,
            vm_mV=np.asarray([[-70.0, center_value, -70.0]], dtype=np.float32),
            step_index=step,
            probe_indices=np.asarray(plan.probe_indices)[None, ...],
            probe_mask=np.asarray(plan.probe_mask)[None, ...],
            thresholds_mV=plan.thresholds_mV,
            reset_thresholds_mV=plan.reset_thresholds_mV,
            blanking_ms=plan.blanking_ms,
            refractory_ms=plan.refractory_ms,
            dt_ms=0.1,
            retention=plan.retention,
        )

    result = finalize_threshold_observer_state(plan, state, nt=8, dt_ms=0.1)[
        "spike_count"
    ]
    np.testing.assert_array_equal(result.values, [2])
    assert result.events == (
        axs.analysis.SpikeCountEvent(
            count=2,
            first_time_ms=0.4,
            last_time_ms=0.8,
            probe_counts=(2,),
        ),
    )
    with pytest.raises(ValueError, match="continuous across chunks"):
        combine_threshold_observer_chunk_states(
            [state, state],
            starts=[0, 4],
            lengths=[4, 4],
            nt=8,
            retention=plan.retention,
        )


def test_bounded_spike_events_store_k_timestamps_and_report_overflow():
    spike_count = axs.analysis.SpikeCount(
        threshold=0.0 * axs.mV,
        reset_threshold=-20.0 * axs.mV,
        refractory=0.0 * axs.ms,
        target=CENTER,
        max_spikes=2,
    )
    plan = build_threshold_observer_plan(
        (spike_count,),
        positions_um=np.asarray([0.0, 50.0, 100.0]),
    )

    assert plan is not None
    assert plan.retention == "spike_events"
    state = init_threshold_observer_state(plan, batch_size=1, nt=6)
    assert state.shape == (1, 1, 1, 7)
    for step, center_value in enumerate((-30.0, 5.0, -30.0, 5.0, -30.0, 5.0)):
        state = update_threshold_observer_state_batch_from_tables(
            state,
            vm_mV=np.asarray([[-70.0, center_value, -70.0]], dtype=np.float32),
            step_index=step,
            probe_indices=np.asarray(plan.probe_indices)[None, ...],
            probe_mask=np.asarray(plan.probe_mask)[None, ...],
            thresholds_mV=plan.thresholds_mV,
            reset_thresholds_mV=plan.reset_thresholds_mV,
            blanking_ms=plan.blanking_ms,
            refractory_ms=plan.refractory_ms,
            dt_ms=0.1,
            retention=plan.retention,
        )

    result = finalize_threshold_observer_state(plan, state, nt=6, dt_ms=0.1)[
        "spike_count"
    ]
    np.testing.assert_array_equal(result.values, [3])
    event = result.events[0]
    assert event.count == 3
    assert event.first_time_ms == pytest.approx(0.2)
    assert event.last_time_ms == pytest.approx(0.6)
    np.testing.assert_allclose(event.spike_times_ms, ((0.2, 0.4),))
    assert event.overflow == (True,)


def test_bounded_spike_events_require_explicit_all_compartment_policy():
    with pytest.raises(ValueError, match="allow_all_compartments=True"):
        build_threshold_observer_plan(
            (axs.analysis.SpikeCount(max_spikes=2),),
            positions_um=np.asarray([0.0, 50.0, 100.0]),
        )


def test_downsampled_vm_raster_ors_hits_within_each_window():
    raster_definition = axs.analysis.VmRaster(
        threshold=0.0 * axs.mV,
        target=CENTER,
        every_n_steps=4,
    )
    plan = build_threshold_observer_plan(
        (raster_definition,),
        positions_um=np.asarray([0.0, 50.0, 100.0]),
    )

    assert plan is not None
    assert plan.retention == "vm_raster"
    assert plan.temporal_stride == 4
    state = init_threshold_observer_state(plan, batch_size=1, nt=10)
    for step in range(10):
        state = update_threshold_observer_state_batch_from_tables(
            state,
            vm_mV=np.asarray(
                [[-70.0, 5.0 if step in {1, 4, 9} else -70.0, -70.0]],
                dtype=np.float32,
            ),
            step_index=step,
            probe_indices=np.asarray(plan.probe_indices)[None, ...],
            probe_mask=np.asarray(plan.probe_mask)[None, ...],
            thresholds_mV=plan.thresholds_mV,
            temporal_stride=plan.temporal_stride,
            retention=plan.retention,
        )

    raster = finalize_threshold_observer_state(plan, state, nt=10, dt_ms=0.1)[
        VM_RASTER_OBSERVATION_KEY
    ]
    assert raster.nt == 3
    assert raster.dt_ms == pytest.approx(0.4)
    np.testing.assert_array_equal(raster.unpack()[0, 0, 0], [True, True, True])

def test_threshold_observer_plan_cache_survives_stimulation_replacement(monkeypatch):
    recording_lowering._THRESHOLD_OBSERVER_PLAN_CACHE.clear()
    recording_lowering._THRESHOLD_OBSERVER_PLAN_IDENTITY_CACHE.clear()
    axons = (object(), object())
    solver_axons = (
        SimpleNamespace(n_compartments=3),
        SimpleNamespace(n_compartments=3),
    )
    spatial_cache_token = object()
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
        spatial_cache_token=spatial_cache_token,
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
        prefer_threshold_observer=True,
    )
    refreshed = SimpleNamespace(
        **{
            **cohort.__dict__,
            "stimulations": ((object(),), (object(),)),
        }
    )
    refreshed.spatial_cache_token = spatial_cache_token

    def fail_digest(_values):
        raise AssertionError("spatially unchanged cohort should use the identity cache")

    monkeypatch.setattr(recording_lowering, "_array_shape_dtype_digest", fail_digest)
    second = recording_lowering.lower_observers_for_cohort(
        observers,
        cohort=refreshed,
        dtype=np.float32,
        prefer_threshold_observer=True,
    )

    assert first is second


def test_threshold_observer_plan_identity_cache_reuses_same_prepared_cohort(monkeypatch):
    recording_lowering._THRESHOLD_OBSERVER_PLAN_CACHE.clear()
    recording_lowering._THRESHOLD_OBSERVER_PLAN_IDENTITY_CACHE.clear()
    cohort = SimpleNamespace(
        group_id=0,
        mode="single",
        size=2,
        nx=3,
        geometry_shared=True,
        has_padding=False,
        axons=(object(), object()),
        solver_axons=(
            SimpleNamespace(n_compartments=3),
            SimpleNamespace(n_compartments=3),
        ),
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
        prefer_threshold_observer=True,
    )

    def fail_digest(_values):
        raise AssertionError("same cohort should use the identity VmRaster cache")

    monkeypatch.setattr(recording_lowering, "_array_shape_dtype_digest", fail_digest)

    second = recording_lowering.lower_observers_for_cohort(
        observers,
        cohort=cohort,
        dtype=np.float32,
        prefer_threshold_observer=True,
    )

    assert second is first


def test_vm_raster_update_packs_row_aware_threshold_bits():
    plan = build_threshold_observer_plan(
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

    state = init_threshold_observer_state(plan, batch_size=2, nt=35)
    state = update_threshold_observer_state_batch_from_tables(
        state,
        vm_mV=np.asarray(
            [
                [-1.0, 5.0, -1.0, 7.0],
                [-1.0, 2.0, 99.0, 99.0],
            ],
            dtype=np.float32,
        ),
        step_index=0,
        probe_indices=plan.probe_indices,
        probe_mask=plan.probe_mask,
        thresholds_mV=plan.thresholds_mV,
    )
    state = update_threshold_observer_state_batch_from_tables(
        state,
        vm_mV=np.asarray(
            [
                [3.0, 5.0, -1.0, 7.0],
                [4.0, 2.0, 99.0, 99.0],
            ],
            dtype=np.float32,
        ),
        step_index=1,
        probe_indices=plan.probe_indices,
        probe_mask=plan.probe_mask,
        thresholds_mV=plan.thresholds_mV,
    )
    state = update_threshold_observer_state_batch_from_tables(
        state,
        vm_mV=np.asarray(
            [
                [3.0, -1.0, 8.0, -1.0],
                [-1.0, 2.0, 99.0, 99.0],
            ],
            dtype=np.float32,
        ),
        step_index=33,
        probe_indices=plan.probe_indices,
        probe_mask=plan.probe_mask,
        thresholds_mV=plan.thresholds_mV,
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

    observations = finalize_threshold_observer_state(plan, state, nt=35, dt_ms=0.1)
    result = observations[VM_RASTER_OBSERVATION_KEY]
    assert result.names == ("activation", "latency")
    assert result.nt == 35
    assert result.dt_ms == 0.1
    assert result.probe_indices is plan.probe_indices_host
    assert result.probe_mask is plan.probe_mask_host
    assert result.original_indices is plan.original_indices_host
    assert result.positions_um is plan.positions_um_host
    assert result.thresholds_mV is plan.thresholds_mV_host
    np.testing.assert_array_equal(result.unpack(), raster)

    lazy_observations = finalize_threshold_observer_state(
        plan,
        state,
        nt=35,
        dt_ms=0.1,
        materialize_words=False,
    )
    lazy_result = lazy_observations[VM_RASTER_OBSERVATION_KEY]
    assert hasattr(lazy_result.words, "block_until_ready")
    np.testing.assert_array_equal(lazy_result.unpack(), raster)
    np.testing.assert_array_equal(
        lazy_result.any_active("activation"),
        np.any(raster[:, 0], axis=(1, 2)),
    )


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

    combined = combine_threshold_observer_chunk_states(
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

    combined = combine_threshold_observer_chunk_states([chunk], starts=[0], lengths=[64], nt=64)

    np.testing.assert_array_equal(np.asarray(combined), chunk)


def test_threshold_observer_plan_rejects_non_threshold_observers():
    with pytest.raises(NotImplementedError, match="threshold-style Vm"):
        build_threshold_observer_plan(
            (axs.analysis.PeakVoltage(target=CENTER),),
            positions_um=np.asarray([0.0, 50.0, 100.0]),
        )
