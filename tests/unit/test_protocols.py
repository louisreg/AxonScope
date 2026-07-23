from types import SimpleNamespace
from typing import get_args

import jax.numpy as jnp
import numpy as np
import pytest

import axonfleet as axs
import axonfleet.protocols.recruitment as recruitment_protocols
from axonfleet.protocols import observer_path as observer_protocols
from axonfleet.protocols import sweep as sweep_protocols
from axonfleet.protocols import threshold as threshold_protocols
from axonfleet.results import VM_RASTER_OBSERVATION_KEY, VmRasterResult
from axonfleet.results.pool import _ResultBlock
from axonfleet.results.vm_raster import activation_values_from_vm_raster


class _DummyLayout:
    def position_values(self, *, unit="micrometer"):
        return np.asarray([0.0, 100.0])


class _DummyAxon:
    n_compartments = 2
    layout = _DummyLayout()


def _result_for_current(electrode_current, *, threshold_nA=1.0):
    amp_nA = axs.units.to_scalar(electrode_current, "nanoampere")
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
    return axs.results.AxonSimulationResult((cohort,), size=row_count)


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
    return axs.results.AxonSimulationResult((cohort,), size=row_count)


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

        def _run_numeric_axis(self, axis_input):
            return runner(self.axons, axis_input=axis_input, **self.kwargs)

    for module in (observer_protocols, sweep_protocols, threshold_protocols):
        monkeypatch.setattr(module, "AxonSimulation", FakeAxonSimulation)


def _extracellular_update_pool(count=2):
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=3,
    )
    positions = axon.layout.position_values(unit=axs.um) * axs.um
    electrode = axs.analytical.PointSourceElectrode(
        x=50.0 * axs.um,
        y=10.0 * axs.um,
        z=0.0 * axs.um,
    )
    zero = axs.Stimulus.pulse(
        start=0.2 * axs.ms,
        duration=0.1 * axs.ms,
        amplitude=0.0 * axs.uA,
    )
    rows = []
    for offset in range(count):
        stimulation = axs.analytical.point_source_stimulation(
            electrode,
            positions,
            sigma=0.3 * axs.S_per_m,
            stimulus=zero,
            axon_y=float(offset) * axs.um,
        )
        row = axs.AxonInstance(axon)
        row.add_extracellular_stimulation(stimulation=stimulation)
        rows.append(row)
    return tuple(rows)


def test_extracellular_waveform_axis_preserves_independent_phase_amplitudes():
    pool = _extracellular_update_pool(count=1)
    source_stimulus = pool[0].extracellular_stimulation.drives[0].stimulus

    def waveform(positive_amplitude):
        positive_uA = float(positive_amplitude.to(axs.uA).magnitude)
        return axs.Stimulus.from_samples(
            t=np.asarray([0.0, 0.2, 0.3, 0.4]) * axs.ms,
            y=np.asarray([0.0, -1.0, positive_uA, 0.0]),
            unit=axs.uA,
        )

    update = axs.protocols.ExtracellularWaveformUpdate(waveform)
    prepared = update.prepare_numeric_axis(pool)
    axis_input = prepared.numeric_axis_input((2.0 * axs.uA, 3.0 * axs.uA))

    np.testing.assert_allclose(
        axis_input.waveforms[0].y,
        [0.0, -1e-6, 2e-6, 0.0],
    )
    np.testing.assert_allclose(
        axis_input.waveforms[1].y,
        [0.0, -1e-6, 3e-6, 0.0],
    )
    assert pool[0].extracellular_stimulation.drives[0].stimulus is source_stimulus


def test_extracellular_waveform_axis_prepares_selected_multi_drive_only():
    row = _extracellular_update_pool(count=1)[0]
    stimulation = row.extracellular_stimulation
    first = stimulation.drives[0]
    first_static = axs.Stimulus.constant(0.25 * axs.uA)
    second_static = axs.Stimulus.constant(-0.5 * axs.uA)
    row.add_extracellular_stimulation(
        stimulation=axs.ExtracellularStimulation(
            (
                axs.ExtracellularDrive(
                    id=first.id,
                    footprint=first.footprint,
                    stimulus=first_static,
                ),
                axs.ExtracellularDrive(
                    id=axs.DriveId("second"),
                    footprint=first.footprint,
                    stimulus=second_static,
                ),
            )
        ),
        replace=True,
    )
    source = row.extracellular_stimulation
    update = axs.protocols.ExtracellularWaveformUpdate(
        lambda value: axs.Stimulus.constant(value),
        drive_id=axs.DriveId("second"),
    )

    prepared = update.prepare_numeric_axis((row,))
    axis_input = prepared.numeric_axis_input((1.0 * axs.uA, 2.0 * axs.uA))

    assert axis_input.drive_count == 2
    assert axis_input.selected_drive_indices == (1,)
    assert axis_input.source_drive_waveforms[0][0] is source.drives[0].stimulus
    assert axis_input.source_drive_waveforms[0][1] is source.drives[1].stimulus
    np.testing.assert_allclose(axis_input.waveforms[0].y, [1e-6])
    np.testing.assert_allclose(axis_input.waveforms[1].y, [2e-6])
    assert row.extracellular_stimulation is source


def test_recruitment_reuses_one_observer_simulation_for_typed_waveform(monkeypatch):
    pool = _extracellular_update_pool()
    factory_values = []
    build_calls = []
    evaluate_calls = []

    def waveform(value):
        value_uA = float(value.to(axs.uA).magnitude)
        factory_values.append(value_uA)
        return axs.Stimulus.pulse(
            start=0.2 * axs.ms,
            duration=0.1 * axs.ms,
            amplitude=-value,
        )

    def build_simulation(updated_pool, **kwargs):
        build_calls.append((updated_pool, kwargs))
        return SimpleNamespace(progress=kwargs.get("progress", False)), "activation"

    def evaluate_numeric_axis(simulation, activation, axis_input):
        evaluate_calls.append((simulation, activation, axis_input))
        return np.asarray(
            [
                np.full(
                    (len(pool),),
                    abs(float(np.min(waveform.y))) * 1e6 >= 2.0,
                    dtype=bool,
                )
                for waveform in axis_input.waveforms
            ]
        )

    monkeypatch.setattr(
        observer_protocols,
        "_build_activation_observer_simulation",
        build_simulation,
    )
    monkeypatch.setattr(
        observer_protocols,
        "_evaluate_activation_observer_numeric_axis",
        evaluate_numeric_axis,
    )

    curve = axs.protocols.recruitment_sweep(
        pool,
        update=axs.protocols.ExtracellularWaveformUpdate(waveform),
        values=np.asarray([1.0, 2.0, 3.0]) * axs.uA,
        duration=1.0 * axs.ms,
        dt=0.1 * axs.ms,
        criterion=axs.analysis.Activation(),
        recording=axs.Recording.none(),
    )

    assert factory_values == [1.0, 2.0, 3.0]
    assert len(build_calls) == 1
    assert len(evaluate_calls) == 3
    assert all(call[0] is evaluate_calls[0][0] for call in evaluate_calls)
    assert [call[1] for call in evaluate_calls] == ["activation"] * 3
    assert [call[2].size for call in evaluate_calls] == [1, 1, 1]
    assert evaluate_calls[0][0].progress is False
    np.testing.assert_array_equal(
        curve.activated,
        [[False, False], [True, True], [True, True]],
    )


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
    activation = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=1.5 * axs.ms,
        target=axs.positions.DISTAL,
        name="activation",
    )
    early = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=1.5 * axs.ms,
        target=axs.positions.DISTAL,
        name="early",
    )

    np.testing.assert_array_equal(
        activation_values_from_vm_raster(raster, activation),
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
    activation = axs.analysis.Activation(name="activation")

    with pytest.raises(RuntimeError, match="missing from VmRaster"):
        activation_values_from_vm_raster(raster, activation)


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
    activation = axs.analysis.Activation(name="activation")

    np.testing.assert_array_equal(
        activation_values_from_vm_raster(raster, activation),
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
    activation = axs.analysis.Activation(name="activation")

    np.testing.assert_array_equal(
        activation_values_from_vm_raster(raster, activation),
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
    activation = axs.analysis.Activation(name="activation")

    np.testing.assert_array_equal(
        activation_values_from_vm_raster(raster, activation),
        [True, False],
    )


def test_recruitment_sweep_accepts_pool_update(monkeypatch):
    criterion = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
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
        recording=axs.Recording.voltage(),
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
        axs.analysis.AnalysisStatus.VALID,
        axs.analysis.AnalysisStatus.VALID,
    )
    np.testing.assert_allclose(tested_values_nA, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    assert progress_values == ["plain", False, False]


def test_recruitment_sweep_uses_observer_only_recording(monkeypatch):
    criterion = axs.analysis.Activation(
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
    criterion = axs.analysis.Activation(
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
    criterion = axs.analysis.Activation(
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


def test_recruitment_sweep_rejects_opaque_batched_update():
    criterion = axs.analysis.Activation(
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
    def update(row, tested_current):
        row.tested_current_nA = float(tested_current.to(axs.nA).magnitude)

    with pytest.raises(ValueError, match="NumericAxisUpdate"):
        axs.protocols.recruitment_sweep(
            pool,
            update=update,
            values=np.asarray([0.0, 1.0, 2.0]) * axs.nA,
            duration=2.0 * axs.ms,
            dt=1.0 * axs.ms,
            criterion=criterion,
            recording=axs.Recording.none(),
            batch_amplitudes=True,
        )
    assert not any(hasattr(row, "tested_current_nA") for row in pool)


def test_recruitment_sweep_can_chunk_batched_observer_amplitudes(monkeypatch):
    criterion = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    pool = _extracellular_update_pool()
    factory_values = []
    simulations = []

    def waveform(value):
        factory_values.append(float(value.to(axs.nA).magnitude))
        return axs.Stimulus.pulse(
            start=0.2 * axs.ms,
            duration=0.1 * axs.ms,
            amplitude=-value,
        )

    def build_simulation(updated_pool, **kwargs):
        simulation = SimpleNamespace(progress=kwargs.get("progress", False))
        simulations.append((simulation, tuple(id(row) for row in updated_pool)))
        return simulation, "activation"

    def evaluate_numeric_axis(_simulation, _activation, axis_input):
        return np.asarray(
            [
                [
                    abs(float(np.min(waveform.y))) * 1e9 >= 0.5,
                    abs(float(np.min(waveform.y))) * 1e9 >= 1.5,
                ]
                for waveform in axis_input.waveforms
            ]
        )

    monkeypatch.setattr(
        observer_protocols,
        "_build_activation_observer_simulation",
        build_simulation,
    )
    monkeypatch.setattr(
        observer_protocols,
        "_evaluate_activation_observer_numeric_axis",
        evaluate_numeric_axis,
    )

    curve = axs.protocols.recruitment_sweep(
        pool,
        update=axs.protocols.ExtracellularWaveformUpdate(waveform),
        values=np.asarray([0.0, 1.0, 2.0]) * axs.nA,
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        recording=axs.Recording.none(),
        solver_progress="plain",
        batch_amplitudes=True,
        amplitude_batch_size=2,
    )

    np.testing.assert_allclose(factory_values, [0.0, 1.0, 2.0])
    assert len(simulations) == 1
    assert simulations[0][1] == tuple(id(row) for row in pool)
    assert simulations[0][0].progress is False
    np.testing.assert_array_equal(
        curve.activated,
        [[False, False], [True, False], [True, True]],
    )


def test_recruitment_batch_planning_is_separate_from_execution():
    pool = _extracellular_update_pool()
    update = axs.protocols.ExtracellularWaveformUpdate(
        lambda value: axs.Stimulus.pulse(
            start=0.2 * axs.ms,
            duration=0.1 * axs.ms,
            amplitude=-value,
        )
    )
    plan = sweep_protocols._plan_numeric_pool_sweep(
        pool,
        update=update,
        values=tuple(np.asarray([0.0, 1.0, 2.0]) * axs.nA),
        value_batch_size=2,
    )

    assert plan.source_pool_size == 2
    assert plan.source_pool == pool
    assert plan.update is update
    assert [batch.start_index for batch in plan.batches] == [0, 2]
    assert [len(batch.values) for batch in plan.batches] == [2, 1]
    assert all(batch.values[0] is plan.values[batch.start_index] for batch in plan.batches)
    assert tuple(id(row) for row in plan.source_pool) == tuple(id(row) for row in pool)


def test_recruitment_numeric_waveform_axis_matches_single_value_chunks():
    pool = _extracellular_update_pool()
    values = np.asarray([0.0, 1.0, 2.0]) * axs.uA

    def make_update():
        return axs.protocols.ExtracellularWaveformUpdate(
            lambda value: axs.Stimulus.pulse(
                start=0.2 * axs.ms,
                duration=0.1 * axs.ms,
                amplitude=-value,
            )
        )

    kwargs = {
        "pool": pool,
        "values": values,
        "duration": 0.6 * axs.ms,
        "dt": 0.05 * axs.ms,
        "criterion": axs.analysis.Activation(
            threshold=0.0 * axs.mV,
            blanking=0.2 * axs.ms,
            target=axs.positions.ALL,
        ),
        "recording": axs.Recording.none(),
        "batch_amplitudes": True,
    }
    one = axs.protocols.recruitment_sweep(
        update=make_update(),
        amplitude_batch_size=1,
        **kwargs,
    )
    full = axs.protocols.recruitment_sweep(
        update=make_update(),
        amplitude_batch_size=None,
        **kwargs,
    )

    np.testing.assert_array_equal(full.activated, one.activated)


def test_recruitment_numeric_axis_matches_single_chunks_with_multiple_drives():
    row = _extracellular_update_pool(count=1)[0]
    source_stimulation = row.extracellular_stimulation
    source_drive = source_stimulation.drives[0]
    row.add_extracellular_stimulation(
        stimulation=axs.ExtracellularStimulation(
            (
                axs.ExtracellularDrive(
                    id=source_drive.id,
                    footprint=source_drive.footprint,
                    stimulus=axs.Stimulus.constant(0.0 * axs.uA),
                ),
                axs.ExtracellularDrive(
                    id=axs.DriveId("variable"),
                    footprint=source_drive.footprint,
                    stimulus=axs.Stimulus.constant(0.0 * axs.uA),
                ),
            )
        ),
        replace=True,
    )
    pool = (row,)
    source = row.extracellular_stimulation

    def make_update():
        return axs.protocols.ExtracellularWaveformUpdate(
            lambda value: axs.Stimulus.pulse(
                start=0.2 * axs.ms,
                duration=0.1 * axs.ms,
                amplitude=-value,
            ),
            drive_id=axs.DriveId("variable"),
        )

    kwargs = {
        "pool": pool,
        "values": np.asarray([0.0, 1.0, 2.0]) * axs.uA,
        "duration": 0.6 * axs.ms,
        "dt": 0.05 * axs.ms,
        "criterion": axs.analysis.Activation(
            threshold=0.0 * axs.mV,
            blanking=0.2 * axs.ms,
            target=axs.positions.ALL,
        ),
        "recording": axs.Recording.none(),
        "batch_amplitudes": True,
    }
    one = axs.protocols.recruitment_sweep(
        update=make_update(),
        amplitude_batch_size=1,
        **kwargs,
    )
    full = axs.protocols.recruitment_sweep(
        update=make_update(),
        amplitude_batch_size=None,
        **kwargs,
    )

    np.testing.assert_array_equal(full.activated, one.activated)
    assert row.extracellular_stimulation is source


def test_recruitment_double_cable_numeric_axis_supports_multiple_drives():
    axon = axs.axons.MRG(
        diameter=7.3 * axs.um,
        nodes=4,
        length=1500.0 * axs.um,
        compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
    )
    positions = axon.layout.position_values(unit=axs.um) * axs.um
    stimulation = axs.analytical.point_source_stimulation(
        axs.analytical.PointSourceElectrode(
            x=750.0 * axs.um,
            y=20.0 * axs.um,
            z=0.0 * axs.um,
        ),
        positions,
        sigma=0.3 * axs.S_per_m,
        stimulus=axs.Stimulus.constant(0.0 * axs.uA),
    )
    source_drive = stimulation.drives[0]
    row = axs.AxonInstance(axon)
    row.add_extracellular_stimulation(
        stimulation=axs.ExtracellularStimulation(
            (
                source_drive,
                axs.ExtracellularDrive(
                    id=axs.DriveId("variable"),
                    footprint=source_drive.footprint,
                    stimulus=axs.Stimulus.constant(0.0 * axs.uA),
                ),
            )
        )
    )

    def make_update():
        return axs.protocols.ExtracellularWaveformUpdate(
            lambda value: axs.Stimulus.pulse(
                start=0.1 * axs.ms,
                duration=0.05 * axs.ms,
                amplitude=-value,
            ),
            drive_id=axs.DriveId("variable"),
        )

    kwargs = {
        "pool": (row,),
        "values": np.asarray([0.0, 1.0]) * axs.uA,
        "duration": 0.3 * axs.ms,
        "dt": 0.05 * axs.ms,
        "criterion": axs.analysis.Activation(
            threshold=0.0 * axs.mV,
            blanking=0.1 * axs.ms,
            target=axs.positions.ALL,
        ),
        "recording": axs.Recording.none(),
        "batch_amplitudes": True,
    }
    one = axs.protocols.recruitment_sweep(
        update=make_update(),
        amplitude_batch_size=1,
        **kwargs,
    )
    full = axs.protocols.recruitment_sweep(
        update=make_update(),
        amplitude_batch_size=None,
        **kwargs,
    )

    np.testing.assert_array_equal(full.activated, one.activated)


def test_recruitment_sweep_can_batch_double_cable_observer_amplitudes(monkeypatch):
    criterion = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    row = axs.AxonInstance(
        axs.axons.MRG(
                diameter=7.3 * axs.um,
                nodes=4,
                length=1500.0 * axs.um,
                compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
        )
    )
    row.add_extracellular_stimulation(
        stimulation=_extracellular_update_pool()[0].extracellular_stimulation
    )
    pool = (row,)
    calls = []

    update = axs.protocols.ExtracellularWaveformUpdate(
        lambda value: axs.Stimulus.pulse(
            start=0.2 * axs.ms,
            duration=0.1 * axs.ms,
            amplitude=-value,
        )
    )

    def build_simulation(updated_pool, **_kwargs):
        calls.append(tuple(updated_pool))
        return SimpleNamespace(progress=False), "activation"

    def evaluate_numeric_axis(_simulation, _activation, axis_input):
        return np.asarray(
            [
                [abs(float(np.min(waveform.y))) * 1e9 >= 0.5]
                for waveform in axis_input.waveforms
            ]
        )

    monkeypatch.setattr(observer_protocols, "_build_activation_observer_simulation", build_simulation)
    monkeypatch.setattr(
        observer_protocols,
        "_evaluate_activation_observer_numeric_axis",
        evaluate_numeric_axis,
    )

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
    assert calls[0] == pool
    np.testing.assert_array_equal(curve.activated, [[False], [True]])


def test_activation_observer_pool_result_uses_cohort_vector_path():
    result = axs.results.AxonSimulationResult(
        (
            _observer_only_cohort((True, False), input_indices=(2, 0)),
            _observer_only_cohort((True,), input_indices=(1,)),
        ),
        size=3,
    )
    activation = axs.analysis.Activation(
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
    criterion = axs.analysis.Activation(
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


def test_pool_sweep_uses_generic_numeric_axis_for_typed_updates(monkeypatch):
    pool = _extracellular_update_pool()
    axis_inputs = []

    def fake_simulation_runner(source_pool, *, axis_input, **_kwargs):
        axis_inputs.append(axis_input)
        amplitude_uA = abs(float(np.min(axis_input.waveforms[0].y))) * 1e6
        voltage = 20.0 if amplitude_uA >= 1.0 else -70.0
        return _public_pool_result(
            tuple(np.full((3, 2), voltage, dtype=float) for _row in source_pool),
            axons=tuple(source_pool),
        )

    _patch_simulation_runner(monkeypatch, fake_simulation_runner)
    sweep = axs.protocols.pool_sweep(
        pool,
        update=axs.protocols.ExtracellularWaveformUpdate(
            lambda value: axs.Stimulus.pulse(
                start=0.2 * axs.ms,
                duration=0.1 * axs.ms,
                amplitude=-value,
            )
        ),
        values=np.asarray([0.0, 1.0, 2.0]) * axs.uA,
        observe=lambda result: float(np.max(result.voltage_values(unit=axs.mV))),
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
    )

    assert [axis_input.size for axis_input in axis_inputs] == [1, 1, 1]
    assert all(
        type(axis_input).__name__ == "ExtracellularWaveformAxisInput"
        for axis_input in axis_inputs
    )
    np.testing.assert_allclose(
        sweep.observations,
        [[-70.0, -70.0], [20.0, 20.0], [20.0, 20.0]],
    )


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
    criterion = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
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
        recording=axs.Recording.voltage(),
        tolerance=0.25 * axs.nA,
        max_iterations=8,
    )

    assert curve.status == ("threshold", "threshold")
    np.testing.assert_allclose(curve.threshold_uA * 1000.0, [0.5, 1.5], atol=0.25)
    np.testing.assert_allclose(curve.row_values(unit=axs.um), [0.5, 1.5])
    assert curve.n_iterations >= 3
    assert len(tested_currents_nA) >= 2 * curve.n_iterations


def test_threshold_curve_solver_progress_is_first_run_only(monkeypatch):
    criterion = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
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
        recording=axs.Recording.voltage(),
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
            axs.units.to_scalar(row.tested_current, "nanoampere")
            >= thresholds_nA[pool.index(row)]
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


def test_find_threshold_requires_current_units():
    criterion = axs.analysis.Activation(
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
    criterion = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
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
        recording=axs.Recording.voltage(),
        tolerance=None,
        relative_tolerance=0.1,
        max_iterations=12,
    )

    assert curve.status == ("threshold", "threshold")
    np.testing.assert_allclose(curve.threshold_uA * 1000.0, [1.0, 4.0], rtol=0.1)


def test_protocol_threshold_status_vocabulary_is_not_analysis_status():
    threshold_statuses = set(get_args(axs.protocols.ThresholdStatus))
    analysis_statuses = {status.value for status in axs.analysis.AnalysisStatus}

    assert threshold_statuses == {"threshold", "below_range", "above_range"}
    assert threshold_statuses.isdisjoint(analysis_statuses)


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
        axs.analysis.AnalysisStatus.VALID,
        axs.analysis.AnalysisStatus.VALID,
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
        axs.analysis.AnalysisStatus.VALID,
        axs.analysis.AnalysisStatus.VALID,
    )
    assert "diameter_um=0.5" in curve.format(
        row_name="diameter_um",
        row_unit=axs.um,
        threshold_unit=axs.uA,
    )

    curve.print(row_name="diameter_um", row_unit=axs.um, threshold_unit=axs.uA)
    assert "AxonFleet threshold curve" in capsys.readouterr().out

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
