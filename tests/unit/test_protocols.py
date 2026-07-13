from typing import get_args

import jax.numpy as jnp
import numpy as np
import pytest

import axonscope as axs
from axonscope.protocols import observer_path as observer_protocols
from axonscope.protocols import sweep as sweep_protocols
from axonscope.protocols import threshold as threshold_protocols
from axonscope.results import VM_RASTER_OBSERVATION_KEY, VmRasterResult
from axonscope.results.pool import _ResultBlock


class _DummyLayout:
    def position_values(self, *, unit="micrometer"):
        return np.asarray([0.0, 100.0])


class _DummyAxon:
    n_compartments = 2
    layout = _DummyLayout()


def _result_for_current(electrode_current, *, threshold_nA=1.0):
    amp_nA = axs.units.to_nA(electrode_current)
    t = np.asarray([0.0, 1.0, 2.0])
    vm = np.full((3, 2), -70.0)
    if amp_nA >= threshold_nA:
        vm[2, 1] = 20.0
    return _public_pool_result((vm,), axons=(_DummyAxon(),))


def _public_pool_result(vms, *, axons=None):
    vm_tuple = tuple(np.asarray(vm, dtype=float) for vm in vms)
    row_count = len(vm_tuple)
    if axons is None:
        axons = tuple(_DummyAxon() for _ in range(row_count))
    t = np.asarray([0.0, 1.0, 2.0])
    cohort = _ResultBlock(
        input_indices=tuple(range(row_count)),
        axons=tuple(axons),
        simulations=tuple(None for _ in range(row_count)),
        Vm=np.stack(vm_tuple, axis=0) if vm_tuple else np.zeros((0, 3, 2)),
        t=t,
        diagnostics=tuple({} for _ in range(row_count)),
        record_indices=tuple(None for _ in range(row_count)),
    )
    return axs.AxonSimulationResult((cohort,), size=row_count)


def _observer_only_pool_result(activated):
    flags = tuple(bool(value) for value in activated)
    row_count = len(flags)
    words = np.asarray(
        [[[[0b100 if flag else 0]] for _ in range(1)] for flag in flags],
        dtype=np.uint32,
    )
    raster = VmRasterResult(
        words=words,
        nt=3,
        dt_ms=1.0,
        definitions=(),
        names=("activation",),
        probe_indices=np.asarray([[1]], dtype=np.int32),
        probe_mask=np.asarray([[True]], dtype=bool),
        original_indices=np.asarray([[1]], dtype=np.int32),
        positions_um=np.asarray([[100.0]], dtype=float),
        thresholds_mV=np.asarray([0.0], dtype=float),
    )
    cohort = _ResultBlock(
        input_indices=tuple(range(row_count)),
        axons=tuple(_DummyAxon() for _ in range(row_count)),
        simulations=tuple(None for _ in range(row_count)),
        Vm=None,
        t=np.asarray([0.0, 1.0, 2.0]),
        diagnostics=tuple({} for _ in range(row_count)),
        record_indices=tuple(None for _ in range(row_count)),
        observations={VM_RASTER_OBSERVATION_KEY: raster},
    )
    return axs.AxonSimulationResult((cohort,), size=row_count)


def _observer_only_cohort(activated, *, input_indices):
    flags = tuple(bool(value) for value in activated)
    words = np.asarray(
        [[[[0b100 if flag else 0]] for _ in range(1)] for flag in flags],
        dtype=np.uint32,
    )
    raster = VmRasterResult(
        words=words,
        nt=3,
        dt_ms=1.0,
        definitions=(),
        names=("activation",),
        probe_indices=np.asarray([[1]], dtype=np.int32),
        probe_mask=np.asarray([[True]], dtype=bool),
        original_indices=np.asarray([[1]], dtype=np.int32),
        positions_um=np.asarray([[100.0]], dtype=float),
        thresholds_mV=np.asarray([0.0], dtype=float),
    )
    return _ResultBlock(
        input_indices=tuple(int(index) for index in input_indices),
        axons=tuple(_DummyAxon() for _ in flags),
        simulations=tuple(None for _ in flags),
        Vm=None,
        t=np.asarray([0.0, 1.0, 2.0]),
        diagnostics=tuple({} for _ in flags),
        record_indices=tuple(None for _ in flags),
        observations={VM_RASTER_OBSERVATION_KEY: raster},
    )


def _patch_simulation_runner(monkeypatch, runner):
    class FakeAxonSimulation:
        def __init__(self, axons, **kwargs):
            self.axons = axons
            self.kwargs = kwargs

        def run(self):
            return runner(self.axons, **self.kwargs)

    for module in (observer_protocols, sweep_protocols, threshold_protocols):
        monkeypatch.setattr(module, "AxonSimulation", FakeAxonSimulation)


def test_vm_raster_shared_activation_decoder_respects_blanking_and_probe_mask():
    raster = VmRasterResult(
        words=np.asarray([[[[0b001], [0]], [[0b111], [0b111]]]], dtype=np.uint32),
        nt=3,
        dt_ms=1.0,
        definitions=(),
        names=("early", "activation"),
        probe_indices=np.asarray([[0, 0], [0, 1]], dtype=np.int32),
        probe_mask=np.asarray([[True, False], [False, True]], dtype=bool),
        original_indices=np.asarray([[0, -1], [0, 1]], dtype=np.int32),
        positions_um=np.asarray([[0.0, np.nan], [0.0, 100.0]], dtype=float),
        thresholds_mV=np.asarray([0.0, 0.0], dtype=float),
    )
    activation = axs.Activation(
        threshold=0.0 * axs.mV,
        blanking=1.5 * axs.ms,
        target=axs.positions.DISTAL,
        name="activation",
    )
    early = axs.Activation(
        threshold=0.0 * axs.mV,
        blanking=1.5 * axs.ms,
        target=axs.positions.DISTAL,
        name="early",
    )

    np.testing.assert_array_equal(
        axs.results.activation_values_from_vm_raster(raster, activation),
        [True],
    )
    np.testing.assert_array_equal(
        raster.any_active(early, blanking=early.blanking),
        [False],
    )


def test_vm_raster_shared_activation_decoder_reports_missing_definition():
    raster = VmRasterResult(
        words=np.asarray([[[[0b100]]]], dtype=np.uint32),
        nt=3,
        dt_ms=1.0,
        definitions=(),
        names=("other",),
        probe_indices=np.asarray([[1]], dtype=np.int32),
        probe_mask=np.asarray([[True]], dtype=bool),
        original_indices=np.asarray([[1]], dtype=np.int32),
        positions_um=np.asarray([[100.0]], dtype=float),
        thresholds_mV=np.asarray([0.0], dtype=float),
    )
    activation = axs.Activation(name="activation")

    with pytest.raises(RuntimeError, match="missing from VmRaster"):
        axs.results.activation_values_from_vm_raster(raster, activation)


def test_vm_raster_activation_decoder_ignores_bits_outside_nt():
    raster = VmRasterResult(
        words=np.asarray([[[[0b100000]]]], dtype=np.uint32),
        nt=3,
        dt_ms=1.0,
        definitions=(),
        names=("activation",),
        probe_indices=np.asarray([[0]], dtype=np.int32),
        probe_mask=np.asarray([[True]], dtype=bool),
        original_indices=np.asarray([[0]], dtype=np.int32),
        positions_um=np.asarray([[0.0]], dtype=float),
        thresholds_mV=np.asarray([0.0], dtype=float),
    )
    activation = axs.Activation(name="activation")

    np.testing.assert_array_equal(
        axs.results.activation_values_from_vm_raster(raster, activation),
        [False],
    )


def test_vm_raster_activation_decoder_ignores_extra_words_outside_nt():
    raster = VmRasterResult(
        words=np.asarray([[[[0, 0xFFFFFFFF]]]], dtype=np.uint32),
        nt=3,
        dt_ms=1.0,
        definitions=(),
        names=("activation",),
        probe_indices=np.asarray([[0]], dtype=np.int32),
        probe_mask=np.asarray([[True]], dtype=bool),
        original_indices=np.asarray([[0]], dtype=np.int32),
        positions_um=np.asarray([[0.0]], dtype=float),
        thresholds_mV=np.asarray([0.0], dtype=float),
    )
    activation = axs.Activation(name="activation")

    np.testing.assert_array_equal(
        axs.results.activation_values_from_vm_raster(raster, activation),
        [False],
    )


def test_vm_raster_activation_decoder_supports_device_words():
    raster = VmRasterResult(
        words=jnp.asarray([[[[0b100]]], [[[0]]]], dtype=jnp.uint32),
        nt=3,
        dt_ms=1.0,
        definitions=(),
        names=("activation",),
        probe_indices=np.asarray([[0]], dtype=np.int32),
        probe_mask=np.asarray([[True]], dtype=bool),
        original_indices=np.asarray([[0]], dtype=np.int32),
        positions_um=np.asarray([[0.0]], dtype=float),
        thresholds_mV=np.asarray([0.0], dtype=float),
    )
    activation = axs.Activation(name="activation")

    np.testing.assert_array_equal(
        axs.results.activation_values_from_vm_raster(raster, activation),
        [True, False],
    )


def test_find_activation_threshold_accepts_simulation_factory(monkeypatch):
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    calls = []

    def factory(tested_current):
        candidate = _DummyAxon()
        candidate.tested_current = tested_current
        return candidate

    def fake_simulate(candidate, **kwargs):
        calls.append(kwargs)
        return _result_for_current(candidate.tested_current, threshold_nA=1.0)

    _patch_simulation_runner(monkeypatch, fake_simulate)

    threshold = axs.protocols.find_activation_threshold(
        factory,
        bounds=(0.0 * axs.nA, 2.0 * axs.nA),
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        tolerance=0.25 * axs.nA,
    )

    assert threshold.status == "threshold"
    assert threshold.amplitude is not None
    assert 1.0 <= threshold.amplitude.to(axs.nA).magnitude <= 1.25
    assert threshold.n_iterations >= 3
    assert calls


def test_find_activation_threshold_requires_current_units():
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )

    with pytest.raises(TypeError, match="bounds\\[0\\] must include units compatible with current"):
        axs.protocols.find_activation_threshold(
            lambda tested_current: _result_for_current(tested_current),
            bounds=(0.0, 2.0 * axs.nA),
            duration=2.0 * axs.ms,
            dt=1.0 * axs.ms,
            criterion=criterion,
            tolerance=0.25 * axs.nA,
        )

    with pytest.raises(TypeError, match="tolerance must include units compatible with current"):
        axs.protocols.find_activation_threshold(
            lambda tested_current: _result_for_current(tested_current),
            bounds=(0.0 * axs.nA, 2.0 * axs.nA),
            duration=2.0 * axs.ms,
            dt=1.0 * axs.ms,
            criterion=criterion,
            tolerance=0.25,
        )


def test_recruitment_sweep_accepts_pool_update(monkeypatch):
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
        require_propagation=True,
    )
    tested_values_nA: list[float] = []
    pool = (_DummyAxon(), _DummyAxon())
    threshold_by_row = dict(zip(pool, (0.5, 1.5), strict=True))
    progress_values: list[bool | str] = []

    def update(row, tested_current):
        current_nA = float(tested_current.to(axs.nA).magnitude)
        tested_values_nA.append(current_nA)
        row.tested_current = tested_current

    def fake_simulation_runner(updated_pool, **kwargs):
        progress_values.append(kwargs.get("progress", False))
        return _public_pool_result(
            tuple(
                _result_for_current(
                    row.tested_current,
                    threshold_nA=threshold_by_row[row],
                ).single.Vm
                for row in updated_pool
            ),
            axons=tuple(updated_pool),
        )

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update,
        values=np.asarray([0.0, 1.0, 2.0]) * axs.nA,
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        solver_progress="plain",
    )

    np.testing.assert_array_equal(
        curve.activated,
        [[False, False], [True, False], [True, True]],
    )
    np.testing.assert_allclose(curve.count, [0, 1, 2])
    np.testing.assert_allclose(curve.fraction, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(curve.first_activation_uA * 1000.0, [1.0, 2.0])
    first_activation = curve.to_analysis_result()
    np.testing.assert_allclose(first_activation.values * 1000.0, [1.0, 2.0])
    assert first_activation.statuses == (
        axs.AnalysisStatus.VALID,
        axs.AnalysisStatus.VALID,
    )
    np.testing.assert_allclose(tested_values_nA, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    assert progress_values == ["plain", False, False]


def test_recruitment_sweep_uses_observer_only_recording(monkeypatch):
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    pool = (_DummyAxon(), _DummyAxon())
    thresholds_nA = (0.5, 1.5)
    calls = []
    progress_values: list[bool | str] = []

    def update(row, tested_current):
        row.tested_current_nA = float(tested_current.to(axs.nA).magnitude)

    def fake_simulation_runner(updated_pool, **kwargs):
        progress_values.append(kwargs.get("progress", False))
        calls.append(kwargs)
        return _observer_only_pool_result(
            row.tested_current_nA >= thresholds_nA[pool.index(row)]
            for row in updated_pool
        )

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update,
        values=np.asarray([0.0, 1.0, 2.0]) * axs.nA,
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        recording=axs.Recording.none(),
        solver_progress="plain",
    )

    assert len(calls) == 3
    assert progress_values == ["plain", False, False]
    for call in calls:
        assert isinstance(call["recording"], axs.Recording)
        assert not call["recording"].voltage
        assert call["observers"][0].name == "activation"
        assert call["observers"][0].target is axs.positions.DISTAL
    np.testing.assert_array_equal(
        curve.activated,
        [[False, False], [True, False], [True, True]],
    )


def test_recruitment_sweep_keeps_axoninstance_observer_values_sequential(monkeypatch):
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    pool = tuple(
        axs.AxonInstance(
            axs.axons.HodgkinHuxley(
                length=100.0 * axs.um,
                diameter=0.5 * axs.um,
                compartments=3,
            )
        )
        for _ in range(2)
    )
    thresholds_nA = (0.5, 1.5)
    calls = []

    def update(row, tested_current):
        row.tested_current_nA = float(tested_current.to(axs.nA).magnitude)

    def fake_simulation_runner(updated_pool, **kwargs):
        updated_pool = tuple(updated_pool)
        calls.append((updated_pool, kwargs))
        return _observer_only_pool_result(
            row.tested_current_nA >= thresholds_nA[index % len(pool)]
            for index, row in enumerate(updated_pool)
        )

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update,
        values=np.asarray([0.0, 1.0, 2.0]) * axs.nA,
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        recording=axs.Recording.none(),
        batch_options=axs.BatchOptions.none(time_chunk_steps=123),
        solver_progress="plain",
    )

    assert len(calls) == 3
    assert [len(updated_pool) for updated_pool, _call in calls] == [2, 2, 2]
    assert [call["progress"] for _updated_pool, call in calls] == ["plain", False, False]
    for _updated_pool, call in calls:
        assert isinstance(call["recording"], axs.Recording)
        assert not call["recording"].voltage
        assert call["batch_options"].time_chunk_steps == 123
        assert call["observers"][0].name == "activation"
    np.testing.assert_array_equal(
        curve.activated,
        [[False, False], [True, False], [True, True]],
    )


def test_recruitment_sweep_keeps_observer_sweeps_sequential_by_default(monkeypatch):
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    pool = tuple(
        axs.AxonInstance(
            axs.axons.HodgkinHuxley(
                length=100.0 * axs.um,
                diameter=0.5 * axs.um,
                compartments=3,
            )
        )
        for _ in range(2)
    )
    calls = []
    progress_values: list[bool | str] = []

    def update(row, tested_current):
        row.tested_current_nA = float(tested_current.to(axs.nA).magnitude)

    def fake_simulation_runner(updated_pool, **kwargs):
        updated_pool = tuple(updated_pool)
        calls.append(updated_pool)
        progress_values.append(kwargs.get("progress", False))
        return _observer_only_pool_result(
            row.tested_current_nA >= 1.0
            for row in updated_pool
        )

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update,
        values=np.asarray([0.0, 1.0, 2.0]) * axs.nA,
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        recording=axs.Recording.none(),
        solver_progress="plain",
    )

    assert [len(call) for call in calls] == [2, 2, 2]
    assert progress_values == ["plain", False, False]
    np.testing.assert_array_equal(
        curve.activated,
        [[False, False], [False, False], [True, True]],
    )


def test_recruitment_sweep_can_batch_observer_amplitudes_into_native_pool(monkeypatch):
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    pool = tuple(
        axs.AxonInstance(
            axs.axons.HodgkinHuxley(
                length=100.0 * axs.um,
                diameter=0.5 * axs.um,
                compartments=3,
            )
        )
        for _ in range(2)
    )
    calls = []
    progress_values: list[bool | str] = []

    def update(row, tested_current):
        row.tested_current_nA = float(tested_current.to(axs.nA).magnitude)

    def fake_simulation_runner(updated_pool, **kwargs):
        updated_pool = tuple(updated_pool)
        calls.append(updated_pool)
        progress_values.append(kwargs.get("progress", False))
        return _observer_only_pool_result(
            row.tested_current_nA >= (0.5 if index % 2 == 0 else 1.5)
            for index, row in enumerate(updated_pool)
        )

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update,
        values=np.asarray([0.0, 1.0, 2.0]) * axs.nA,
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        recording=axs.Recording.none(),
        batch_options=axs.BatchOptions.none(time_chunk_steps=123),
        solver_progress="plain",
        batch_amplitudes=True,
    )

    assert len(calls) == 1
    assert len(calls[0]) == 6
    assert progress_values == ["plain"]
    assert all(row is not original for row in calls[0] for original in pool)
    assert not any(hasattr(row, "tested_current_nA") for row in pool)
    np.testing.assert_allclose(
        [row.tested_current_nA for row in calls[0]],
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
    )
    np.testing.assert_array_equal(
        curve.activated,
        [[False, False], [True, False], [True, True]],
    )


def test_recruitment_sweep_can_chunk_batched_observer_amplitudes(monkeypatch):
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    pool = tuple(
        axs.AxonInstance(
            axs.axons.HodgkinHuxley(
                length=100.0 * axs.um,
                diameter=0.5 * axs.um,
                compartments=3,
            )
        )
        for _ in range(2)
    )
    calls = []
    progress_values: list[bool | str] = []

    def update(row, tested_current):
        row.tested_current_nA = float(tested_current.to(axs.nA).magnitude)

    def fake_simulation_runner(updated_pool, **kwargs):
        updated_pool = tuple(updated_pool)
        calls.append(updated_pool)
        progress_values.append(kwargs.get("progress", False))
        return _observer_only_pool_result(
            row.tested_current_nA >= (0.5 if index % 2 == 0 else 1.5)
            for index, row in enumerate(updated_pool)
        )

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update,
        values=np.asarray([0.0, 1.0, 2.0]) * axs.nA,
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        recording=axs.Recording.none(),
        solver_progress="plain",
        batch_amplitudes=True,
        amplitude_batch_size=2,
    )

    assert len(calls) == 2
    assert [len(call) for call in calls] == [4, 2]
    assert progress_values == ["plain", False]
    np.testing.assert_allclose(
        [row.tested_current_nA for call in calls for row in call],
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
    )
    np.testing.assert_array_equal(
        curve.activated,
        [[False, False], [True, False], [True, True]],
    )


def test_recruitment_sweep_can_batch_double_cable_observer_amplitudes(monkeypatch):
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    pool = (
        axs.AxonInstance(
            axs.axons.MRG(
                diameter=7.3 * axs.um,
                nodes=4,
                length=1500.0 * axs.um,
                compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
            )
        ),
    )
    calls = []

    def update(row, tested_current):
        row.tested_current_nA = float(tested_current.to(axs.nA).magnitude)

    def fake_simulation_runner(updated_pool, **kwargs):
        updated_pool = tuple(updated_pool)
        calls.append(updated_pool)
        return _observer_only_pool_result(
            row.tested_current_nA >= 0.5 for row in updated_pool
        )

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update,
        values=np.asarray([0.0, 1.0]) * axs.nA,
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        recording=axs.Recording.none(),
        batch_amplitudes=True,
    )

    assert len(calls) == 1
    assert len(calls[0]) == 2
    np.testing.assert_array_equal(curve.activated, [[False], [True]])


def test_activation_observer_pool_result_uses_cohort_vector_path():
    result = axs.AxonSimulationResult(
        (
            _observer_only_cohort((True, False), input_indices=(2, 0)),
            _observer_only_cohort((True,), input_indices=(1,)),
        ),
        size=3,
    )
    activation = axs.Activation(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
        name="activation",
    )

    values = observer_protocols._activation_observations_from_pool_result(
        result,
        activation,
    )

    np.testing.assert_array_equal(values, [False, True, True])


def test_recruitment_sweep_requires_current_units():
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )

    with pytest.raises(TypeError, match="values must include units compatible with current"):
        axs.protocols.recruitment_sweep(
            (_DummyAxon(),),
            update=lambda row, tested_current: _result_for_current(tested_current),
            values=np.asarray([0.0, 1.0, 2.0]),
            duration=2.0 * axs.ms,
            dt=1.0 * axs.ms,
            criterion=criterion,
        )


def test_pool_sweep_accepts_generic_observer(monkeypatch):
    tested_values_nA: list[float] = []
    pool = (_DummyAxon(), _DummyAxon())

    def update(row, tested_current):
        current_nA = float(tested_current.to(axs.nA).magnitude)
        tested_values_nA.append(current_nA)
        row.tested_current = tested_current

    def fake_simulation_runner(updated_pool, **kwargs):
        del kwargs
        return _public_pool_result(
            tuple(
                _result_for_current(row.tested_current, threshold_nA=1.0).single.Vm
                for row in updated_pool
            ),
            axons=tuple(updated_pool),
        )

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    sweep = axs.protocols.pool_sweep(
        pool,
        update=update,
        values=np.asarray([0.0, 1.0, 2.0]) * axs.nA,
        observe=lambda result: float(np.max(result.voltage_values(unit=axs.mV))),
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
    )

    assert sweep.n_values == 3
    assert sweep.n_rows == 2
    np.testing.assert_allclose(sweep.value_values(unit=axs.nA), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        sweep.observations,
        [[-70.0, -70.0], [20.0, 20.0], [20.0, 20.0]],
    )
    np.testing.assert_allclose(tested_values_nA, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0])


def test_pool_sweep_solver_progress_is_first_run_only(monkeypatch, capsys):
    progress_values: list[bool | str] = []
    pool = (_DummyAxon(), _DummyAxon())

    def update(row, tested_current):
        row.tested_current = tested_current

    def fake_simulation_runner(updated_pool, **kwargs):
        progress_values.append(kwargs.get("progress", False))
        return _public_pool_result(
            tuple(
                _result_for_current(row.tested_current, threshold_nA=1.0).single.Vm
                for row in updated_pool
            ),
            axons=tuple(updated_pool),
        )

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    axs.protocols.pool_sweep(
        pool,
        update=update,
        values=np.asarray([0.0, 1.0, 2.0]) * axs.nA,
        observe=lambda result: float(np.max(result.voltage_values(unit=axs.mV))),
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        progress="plain",
        solver_progress="plain",
    )

    captured = capsys.readouterr()
    assert progress_values == ["plain", False, False]
    assert "Protocol sweep completed:" in captured.out
    assert "cold_start=" in captured.out
    assert "per_iteration=" in captured.out


def test_find_threshold_accepts_mutating_or_replacing_update(monkeypatch):
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
        require_propagation=True,
    )
    thresholds_nA = np.asarray([0.5, 1.5], dtype=float)
    rows = np.asarray([0.5, 1.5]) * axs.um
    tested_currents_nA: list[float] = []
    pool = tuple(_DummyAxon() for _ in thresholds_nA)

    def update(row, tested_current):
        current_nA = float(tested_current.to(axs.nA).magnitude)
        tested_currents_nA.append(current_nA)
        row.tested_current = tested_current

    def fake_simulation_runner(updated_pool, **kwargs):
        del kwargs
        return _public_pool_result(
            tuple(
                _result_for_current(
                    row.tested_current,
                    threshold_nA=thresholds_nA[pool.index(row)],
                ).single.Vm
                for row in updated_pool
            ),
            axons=tuple(updated_pool),
        )

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    curve = axs.protocols.find_threshold(
        pool,
        rows=rows,
        update=update,
        bounds=(0.0 * axs.nA, 2.0 * axs.nA),
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        tolerance=0.25 * axs.nA,
        max_iterations=8,
    )

    assert curve.status == ("threshold", "threshold")
    np.testing.assert_allclose(curve.threshold_uA * 1000.0, [0.5, 1.5], atol=0.25)
    np.testing.assert_allclose(curve.row_values(unit=axs.um), [0.5, 1.5])
    assert curve.n_iterations >= 3
    assert len(tested_currents_nA) >= 2 * curve.n_iterations


def test_threshold_curve_solver_progress_is_first_run_only(monkeypatch):
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
        require_propagation=True,
    )
    progress_values: list[bool | str] = []
    thresholds_nA = np.asarray([0.5, 1.5], dtype=float)
    pool = tuple(_DummyAxon() for _ in thresholds_nA)

    def update(row, current):
        row.tested_current = current

    def fake_simulation_runner(updated_pool, **kwargs):
        progress_values.append(kwargs.get("progress", False))
        return _public_pool_result(
            tuple(
                _result_for_current(
                    row.tested_current,
                    threshold_nA=thresholds_nA[pool.index(row)],
                ).single.Vm
                for row in updated_pool
            ),
            axons=tuple(updated_pool),
        )

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    axs.protocols.find_threshold(
        pool,
        update=update,
        bounds=(0.0 * axs.nA, 2.0 * axs.nA),
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        tolerance=0.25 * axs.nA,
        max_iterations=8,
        solver_progress="plain",
    )

    assert progress_values[0] == "plain"
    assert progress_values[1:]
    assert all(value is False for value in progress_values[1:])


def test_find_threshold_uses_activation_observer_only_path(monkeypatch):
    criterion = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    thresholds_nA = np.asarray([0.5, 1.5], dtype=float)
    pool = tuple(_DummyAxon() for _ in thresholds_nA)
    calls: list[dict] = []

    def update(row, current):
        row.tested_current = current

    def fake_simulation_runner(updated_pool, **kwargs):
        calls.append(kwargs)
        flags = [
            axs.units.to_nA(row.tested_current) >= thresholds_nA[pool.index(row)]
            for row in updated_pool
        ]
        return _observer_only_pool_result(flags)

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    curve = axs.protocols.find_threshold(
        pool,
        update=update,
        bounds=(0.0 * axs.nA, 2.0 * axs.nA),
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        tolerance=0.25 * axs.nA,
        max_iterations=8,
        recording=axs.Recording.none(),
        batch_options=axs.BatchOptions.full(time_chunk_steps=5),
    )

    assert curve.status == ("threshold", "threshold")
    np.testing.assert_allclose(curve.threshold_uA * 1000.0, [0.5, 1.5], atol=0.25)
    assert calls
    assert all(not call["recording"].voltage for call in calls)
    assert all(not call["recording"].wants_observables for call in calls)
    assert all(call["observers"][0].name == "activation" for call in calls)
    assert all(call["batch_options"].time_chunk_steps == 5 for call in calls)


def test_find_threshold_supports_conduction_block_observer_only_path(monkeypatch):
    criterion = axs.analysis.ConductionBlock(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    block_thresholds_nA = np.asarray([0.5, 1.5], dtype=float)
    pool = tuple(_DummyAxon() for _ in block_thresholds_nA)
    calls: list[dict] = []

    def update(row, current):
        row.tested_current = current

    def fake_simulation_runner(updated_pool, **kwargs):
        calls.append(kwargs)
        activation_flags = [
            axs.units.to_nA(row.tested_current) < block_thresholds_nA[pool.index(row)]
            for row in updated_pool
        ]
        return _observer_only_pool_result(activation_flags)

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    curve = axs.protocols.find_threshold(
        pool,
        update=update,
        bounds=(0.0 * axs.nA, 2.0 * axs.nA),
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        tolerance=0.25 * axs.nA,
        max_iterations=8,
        recording=axs.Recording.none(),
    )

    assert curve.status == ("threshold", "threshold")
    np.testing.assert_allclose(curve.threshold_uA * 1000.0, [0.5, 1.5], atol=0.25)
    assert calls
    assert all(not call["recording"].voltage for call in calls)
    assert all(not call["recording"].wants_observables for call in calls)
    assert all(call["observers"][0].name == "activation" for call in calls)


def test_find_threshold_requires_current_units():
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )

    with pytest.raises(TypeError, match="bounds\\[0\\] must include units compatible with current"):
        axs.protocols.find_threshold(
            pool=(_DummyAxon(),),
            update=lambda row, current: _result_for_current(current),
            bounds=(0.0, 2.0 * axs.nA),
            duration=2.0 * axs.ms,
            dt=1.0 * axs.ms,
            criterion=criterion,
            tolerance=0.25 * axs.nA,
        )


def test_find_threshold_accepts_callable_bounds_and_relative_tolerance(monkeypatch):
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
        require_propagation=True,
    )
    rows = ("low", "high")
    pool = tuple(_DummyAxon() for _ in rows)
    label_by_row = dict(zip(pool, rows, strict=True))
    thresholds_nA = {"low": 1.0, "high": 4.0}

    def update(row, current):
        row.tested_current = current

    def fake_simulation_runner(updated_pool, **kwargs):
        del kwargs
        return _public_pool_result(
            tuple(
                _result_for_current(
                    row.tested_current,
                    threshold_nA=thresholds_nA[label_by_row[row]],
                ).single.Vm
                for row in updated_pool
            ),
            axons=tuple(updated_pool),
        )

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)

    curve = axs.protocols.find_threshold(
        pool=pool,
        rows=rows,
        update=update,
        bounds=lambda row: (
            0.0 * axs.nA,
            (2.0 if row == "low" else 8.0) * axs.nA,
        ),
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        tolerance=None,
        relative_tolerance=0.1,
        max_iterations=12,
    )

    assert curve.status == ("threshold", "threshold")
    np.testing.assert_allclose(curve.threshold_uA * 1000.0, [1.0, 4.0], rtol=0.1)


def test_protocol_threshold_search_result_views(capsys):
    history = (
        axs.protocols.ThresholdHistoryEntry(
            amplitude_uA=5.0,
            activated=False,
            event=axs.analysis.ActivationEvent(
                activated=False,
                peak_mV=-35.0,
                peak_time_ms=0.4,
                peak_index=12,
            ),
        ),
        axs.protocols.ThresholdHistoryEntry(
            amplitude_uA=10.0,
            activated=True,
            event=axs.analysis.ActivationEvent(
                activated=True,
                first_time_ms=0.6,
                first_position_um=100.0,
                first_index=1,
                peak_mV=15.0,
                peak_time_ms=0.7,
                peak_index=2,
            ),
        ),
    )
    result = axs.protocols.ThresholdSearchResult(
        amplitude_uA=7.5,
        lower_bound_uA=5.0,
        upper_bound_uA=10.0,
        status="threshold",
        history=history,
    )

    rows = axs.protocols.views.threshold_search_rows(result, unit=axs.uA)
    assert result.rows(unit=axs.uA) == rows
    assert rows[0]["amplitude"] == 5.0
    assert rows[1]["activated"] is True
    dataframe = result.to_dataframe(unit=axs.uA)
    assert list(dataframe.columns) == [
        "iteration",
        "amplitude",
        "activated",
        "first_time_ms",
        "first_position_um",
        "peak_mV",
    ]
    assert "amplitude=7.5" in result.format(unit=axs.uA)
    threshold_metric = result.to_analysis_result()
    assert threshold_metric.name == "threshold"
    assert threshold_metric.value == pytest.approx(7.5)
    assert threshold_metric.status is axs.AnalysisStatus.VALID

    result.print(unit=axs.uA)
    assert "AxonScope threshold search" in capsys.readouterr().out

    import matplotlib.pyplot as plt

    ax = result.plot(unit=axs.uA)
    assert ax.get_xlabel().startswith("Amplitude [")
    plt.close(ax.figure)


def test_protocol_threshold_status_vocabulary_is_not_analysis_status():
    threshold_statuses = set(get_args(axs.protocols.ThresholdStatus))
    analysis_statuses = {status.value for status in axs.AnalysisStatus}

    assert threshold_statuses == {"threshold", "below_range", "above_range"}
    assert threshold_statuses.isdisjoint(analysis_statuses)
    assert "find_threshold" in (axs.protocols.ThresholdSearchResult.__doc__ or "")
    assert "find_activation_threshold" not in (
        axs.protocols.ThresholdSearchResult.__doc__ or ""
    )


def test_protocol_recruitment_pool_and_threshold_curve_views(capsys):
    recruitment = axs.protocols.RecruitmentCurve(
        amplitudes_uA=np.asarray([0.0, 10.0, 20.0]),
        activated=np.asarray(
            [[False, False], [True, False], [True, True]],
            dtype=bool,
        ),
        row_labels=("a", "b"),
    )
    recruitment_table = recruitment.to_dataframe(unit=axs.uA)
    assert recruitment.rows(unit=axs.uA)[0]["amplitude"] == 0.0
    assert list(recruitment_table.columns) == ["amplitude", "count", "fraction"]
    np.testing.assert_allclose(recruitment_table["fraction"].to_numpy(), [0.0, 0.5, 1.0])
    assert "rows=2" in recruitment.format(unit=axs.uA)
    first_activation = recruitment.to_analysis_result()
    assert first_activation.row_labels == ("a", "b")
    np.testing.assert_allclose(first_activation.values, [10.0, 20.0])
    assert first_activation.statuses == (
        axs.AnalysisStatus.VALID,
        axs.AnalysisStatus.VALID,
    )

    sweep = axs.protocols.PoolSweepResult(
        values=tuple(np.asarray([0.0, 1.0, 2.0]) * axs.nA),
        observations=np.asarray([[-70.0, -69.0], [-10.0, -65.0], [20.0, -20.0]]),
        row_labels=("left", "right"),
    )
    sweep_table = sweep.to_dataframe(value_name="current_nA", value_unit=axs.nA)
    assert sweep.rows(value_name="current_nA", value_unit=axs.nA)[0]["current_nA"] == 0.0
    assert list(sweep_table.columns) == [
        "value_index",
        "row",
        "row_label",
        "current_nA",
        "observation",
    ]
    assert sweep_table["row_label"].tolist()[:2] == ["left", "right"]
    np.testing.assert_allclose(sweep.value_values(unit=axs.nA), [0.0, 1.0, 2.0])
    assert "values=3, rows=2" in sweep.format(
        value_name="current_nA",
        value_unit=axs.nA,
    )

    curve = axs.protocols.ThresholdCurve(
        row_labels=tuple(np.asarray([0.5, 1.0]) * axs.um),
        threshold_uA=np.asarray([12.0, 8.0]),
        lower_bound_uA=np.asarray([11.5, 7.5]),
        upper_bound_uA=np.asarray([12.5, 8.5]),
        status=("threshold", "threshold"),
        tested_uA=(np.asarray([10.0, 15.0]), np.asarray([7.0, 9.0])),
        satisfied=(np.asarray([False, True]), np.asarray([False, True])),
    )
    curve_table = curve.to_dataframe(
        row_name="diameter_um",
        row_unit=axs.um,
        threshold_unit=axs.uA,
    )
    assert curve.rows(
        row_name="diameter_um",
        row_unit=axs.um,
        threshold_unit=axs.uA,
    )[0]["diameter_um"] == 0.5
    assert list(curve_table.columns) == [
        "diameter_um",
        "threshold",
        "lower_bound",
        "upper_bound",
        "status",
    ]
    np.testing.assert_allclose(curve_table["threshold"].to_numpy(), [12.0, 8.0])
    threshold_metric = curve.to_analysis_result()
    assert threshold_metric.row_labels == curve.row_labels
    np.testing.assert_allclose(threshold_metric.values, [12.0, 8.0])
    assert threshold_metric.statuses == (
        axs.AnalysisStatus.VALID,
        axs.AnalysisStatus.VALID,
    )
    assert "diameter_um=0.5" in curve.format(
        row_name="diameter_um",
        row_unit=axs.um,
        threshold_unit=axs.uA,
    )

    curve.print(row_name="diameter_um", row_unit=axs.um, threshold_unit=axs.uA)
    assert "AxonScope threshold curve" in capsys.readouterr().out

    import matplotlib.pyplot as plt

    axes = [
        recruitment.plot(unit=axs.uA),
        recruitment.plot_groups(("low", "high"), unit=axs.uA),
        sweep.plot(value_unit=axs.nA),
        curve.plot(row_unit=axs.um, threshold_unit=axs.uA),
    ]
    assert axes[0].get_ylabel() == "Recruitment fraction"
    assert axes[1].get_ylabel() == "Recruitment fraction"
    assert axes[2].get_xlabel() == "value [nA]"
    assert axes[3].get_xlabel().startswith("row [")
    for ax in axes:
        plt.close(ax.figure)
