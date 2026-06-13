import numpy as np

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
    criterion = axs.results.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        positions="distal",
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


def test_recruitment_sweep_accepts_result_pool_factory():
    criterion = axs.results.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.5 * axs.ms,
        positions="distal",
    )
    tested_values_nA: list[float] = []

    def pool_factory(tested_current):
        tested_values_nA.append(float(tested_current.to(axs.nA).magnitude))
        return (
            _result_for_current(tested_current, threshold_nA=0.5),
            _result_for_current(tested_current, threshold_nA=1.5),
        )

    curve = axs.protocols.recruitment_sweep(
        pool_factory,
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
    np.testing.assert_allclose(tested_values_nA, [0.0, 1.0, 2.0])
