import numpy as np
import pytest

import axonscope as axs
from axonscope.protocols import activation as activation_protocols
from axonscope.results import VM_RASTER_OBSERVATION_KEY, VmRasterResult


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
    cohort = axs.CohortResult(
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
    cohort = axs.CohortResult(
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

    monkeypatch.setattr(activation_protocols, "simulate", fake_simulate)

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

    def update(row, tested_current):
        current_nA = float(tested_current.to(axs.nA).magnitude)
        tested_values_nA.append(current_nA)
        row.tested_current = tested_current

    def fake_simulate_pool(updated_pool, **kwargs):
        del kwargs
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

    monkeypatch.setattr(activation_protocols, "simulate_pool", fake_simulate_pool)

    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update,
        amplitudes=np.asarray([0.0, 1.0, 2.0]) * axs.nA,
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
    )

    np.testing.assert_array_equal(
        curve.activated,
        [[False, False], [True, False], [True, True]],
    )
    np.testing.assert_allclose(curve.count, [0, 1, 2])
    np.testing.assert_allclose(curve.fraction, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(curve.threshold_like_uA * 1000.0, [1.0, 2.0])
    np.testing.assert_allclose(tested_values_nA, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0])


def test_recruitment_sweep_uses_observer_only_recording(monkeypatch):
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    pool = (_DummyAxon(), _DummyAxon())
    thresholds_nA = (0.5, 1.5)
    calls = []

    def update(row, tested_current):
        row.tested_current_nA = float(tested_current.to(axs.nA).magnitude)

    def fake_simulate_pool(updated_pool, **kwargs):
        calls.append(kwargs)
        return _observer_only_pool_result(
            row.tested_current_nA >= thresholds_nA[pool.index(row)]
            for row in updated_pool
        )

    monkeypatch.setattr(activation_protocols, "simulate_pool", fake_simulate_pool)

    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update,
        amplitudes=np.asarray([0.0, 1.0, 2.0]) * axs.nA,
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        recording=axs.Recording.none(),
    )

    assert len(calls) == 3
    for call in calls:
        assert isinstance(call["recording"], axs.Recording)
        assert not call["recording"].voltage
        assert call["observers"][0].name == "activation"
        assert call["observers"][0].target is axs.positions.DISTAL
    np.testing.assert_array_equal(
        curve.activated,
        [[False, False], [True, False], [True, True]],
    )


def test_recruitment_sweep_batches_observer_only_independent_values(monkeypatch):
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

    def fake_simulate_pool(updated_pool, **kwargs):
        updated_pool = tuple(updated_pool)
        calls.append((updated_pool, kwargs))
        return _observer_only_pool_result(
            row.tested_current_nA >= thresholds_nA[index % len(pool)]
            for index, row in enumerate(updated_pool)
        )

    monkeypatch.setattr(activation_protocols, "simulate_pool", fake_simulate_pool)

    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update,
        amplitudes=np.asarray([0.0, 1.0, 2.0]) * axs.nA,
        duration=2.0 * axs.ms,
        dt=1.0 * axs.ms,
        criterion=criterion,
        recording=axs.Recording.none(),
    )

    assert len(calls) == 1
    flat_pool, call = calls[0]
    assert len(flat_pool) == 6
    assert isinstance(call["recording"], axs.Recording)
    assert not call["recording"].voltage
    assert call["observers"][0].name == "activation"
    np.testing.assert_array_equal(
        curve.activated,
        [[False, False], [True, False], [True, True]],
    )


def test_recruitment_sweep_requires_current_units():
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )

    with pytest.raises(TypeError, match="amplitudes must include units compatible with current"):
        axs.protocols.recruitment_sweep(
            (_DummyAxon(),),
            update=lambda row, tested_current: _result_for_current(tested_current),
            amplitudes=np.asarray([0.0, 1.0, 2.0]),
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

    def fake_simulate_pool(updated_pool, **kwargs):
        del kwargs
        return _public_pool_result(
            tuple(
                _result_for_current(row.tested_current, threshold_nA=1.0).single.Vm
                for row in updated_pool
            ),
            axons=tuple(updated_pool),
        )

    monkeypatch.setattr(activation_protocols, "simulate_pool", fake_simulate_pool)

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


def test_find_activation_threshold_curve_accepts_mutating_or_replacing_update(monkeypatch):
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

    def fake_simulate_pool(updated_pool, **kwargs):
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

    monkeypatch.setattr(activation_protocols, "simulate_pool", fake_simulate_pool)

    curve = axs.protocols.find_activation_threshold_curve(
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


def test_find_activation_threshold_curve_requires_current_units():
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )

    with pytest.raises(TypeError, match="bounds\\[0\\] must include units compatible with current"):
        axs.protocols.find_activation_threshold_curve(
            pool=(_DummyAxon(),),
            update=lambda row, current: _result_for_current(current),
            bounds=(0.0, 2.0 * axs.nA),
            duration=2.0 * axs.ms,
            dt=1.0 * axs.ms,
            criterion=criterion,
            tolerance=0.25 * axs.nA,
        )


def test_find_activation_threshold_curve_accepts_callable_bounds_and_relative_tolerance(monkeypatch):
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

    def fake_simulate_pool(updated_pool, **kwargs):
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

    monkeypatch.setattr(activation_protocols, "simulate_pool", fake_simulate_pool)

    curve = axs.protocols.find_activation_threshold_curve(
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
