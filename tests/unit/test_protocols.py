import numpy as np
import pytest

import axonscope as axs


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
    return axs.SimResult(axon=_DummyAxon(), Vm=vm, t=t)


def test_find_activation_threshold_accepts_result_factory():
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )

    threshold = axs.protocols.find_activation_threshold(
        lambda tested_current: _result_for_current(
            tested_current,
            threshold_nA=1.0,
        ),
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


def test_recruitment_sweep_accepts_pool_update():
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    tested_values_nA: list[float] = []
    pool = (_DummyAxon(), _DummyAxon())
    threshold_by_row = dict(zip(pool, (0.5, 1.5), strict=True))

    def update(row, tested_current):
        tested_values_nA.append(float(tested_current.to(axs.nA).magnitude))
        return _result_for_current(tested_current, threshold_nA=threshold_by_row[row])

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


def test_pool_sweep_accepts_generic_observer():
    tested_values_nA: list[float] = []
    pool = (_DummyAxon(), _DummyAxon())

    def update(row, tested_current):
        del row
        tested_values_nA.append(float(tested_current.to(axs.nA).magnitude))
        return _result_for_current(tested_current, threshold_nA=1.0)

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


def test_find_activation_threshold_curve_accepts_mutating_or_replacing_update():
    criterion = axs.analysis.ActivationCriterion(
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
        index = pool.index(row)
        threshold = thresholds_nA[index]
        return _result_for_current(tested_current, threshold_nA=threshold)

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


def test_find_activation_threshold_curve_accepts_callable_bounds_and_relative_tolerance():
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        target=axs.positions.DISTAL,
    )
    rows = ("low", "high")
    pool = tuple(_DummyAxon() for _ in rows)
    label_by_row = dict(zip(pool, rows, strict=True))
    thresholds_nA = {"low": 1.0, "high": 4.0}

    curve = axs.protocols.find_activation_threshold_curve(
        pool=pool,
        rows=rows,
        update=lambda row, current: _result_for_current(
            current,
            threshold_nA=thresholds_nA[label_by_row[row]],
        ),
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
