import numpy as np
import pytest

import axonscope as axs
from axonscope.results import CohortResult, AxonSimulationResult
from axonscope.results.single import SimResult


class _DummyLayout:
    def __init__(self, positions_um):
        self._positions_um = np.asarray(positions_um, dtype=float)

    def position_values(self, *, unit="micrometer"):
        if unit in ("millimeter", axs.mm):
            return self._positions_um * 1e-3
        return self._positions_um


class _DummyAxon:
    def __init__(self, positions_um):
        self.n_compartments = len(positions_um)
        self.layout = _DummyLayout(positions_um)


def _fake_result(*, second_peak: bool = True, include_vm: bool = True) -> SimResult:
    t = np.linspace(0.0, 50.0, 5001)
    vm = np.full((len(t), 3), -70.0)
    vm[:, 0] += np.exp(-0.5 * ((t - 10.0) / 0.5) ** 2) * 80.0
    if second_peak:
        vm[:, 1] += np.exp(-0.5 * ((t - 30.0) / 0.5) ** 2) * 80.0
    recordings = {"Vm": vm} if include_vm else {"gates": np.zeros((len(t), 3, 1))}
    return SimResult(
        axon=_DummyAxon([0.0, 500.0, 1000.0]),
        t=t,
        recordings=recordings,
    )


def _fake_pool_result() -> AxonSimulationResult:
    active = _fake_result(second_peak=True)
    silent_distal = _fake_result(second_peak=False)
    cohort = CohortResult(
        input_indices=(0, 1),
        axons=(active.axon, silent_distal.axon),
        simulations=(None, None),
        Vm=np.stack([active.Vm, silent_distal.Vm], axis=0),
        t=active.t,
        diagnostics=({}, {}),
        record_indices=(None, None),
    )
    return AxonSimulationResult((cohort,), size=2)


def test_analysis_namespace_is_public_and_not_results_forwarding_alias():
    assert not hasattr(axs.results, "analysis")
    assert axs.Activation is axs.analysis.Activation
    assert axs.analysis.ActivationCriterion is not None
    assert axs.AnalysisStatus.VALID.value == "VALID"


def test_activation_definition_declares_requirements_and_statuses():
    result = _fake_result()
    definition = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        target=axs.positions.CENTER,
    )

    analyzed = definition.evaluate(result)

    assert analyzed.name == "activation"
    assert analyzed.status is axs.analysis.AnalysisStatus.VALID
    assert analyzed.value is True
    assert analyzed.events[0].first_index == 1
    assert definition.requirements.required_signals == (axs.signals.Vm,)
    assert definition.requirements.required_result_fields == ("Vm", "t", "positions")
    assert definition.requirements.required_capabilities == ("membrane_voltage_trace",)
    assert definition.requirements.required_positions == (axs.positions.CENTER,)
    assert definition.requirements.online_supported is True
    assert definition.requirements.algorithm_version == "activation_threshold_v1"
    assert definition.requirements.recording_hint is not None


def test_result_analyze_returns_report_with_population_denominators():
    result = _fake_pool_result()

    report = result.report(
        axs.analysis.Activation(threshold=0.0 * axs.mV, target=axs.positions.CENTER),
        axs.analysis.PeakVoltage(target=axs.positions.CENTER),
    )

    assert report.names == ("activation", "peak_voltage")
    assert report["activation"].population.n_total == 2
    assert report["activation"].population.n_applicable == 2
    assert report["activation"].population.n_valid == 2
    np.testing.assert_array_equal(report["activation"].values, [True, False])
    np.testing.assert_allclose(report["peak_voltage"].values, [10.0, -70.0], atol=1e-6)


def test_analysis_missing_input_is_reported_per_row():
    result = _fake_result(include_vm=False)

    analyzed = result.analyze(axs.analysis.PeakVoltage())

    assert analyzed.status is axs.analysis.AnalysisStatus.MISSING_INPUT
    assert analyzed.population.n_failed == 1
    assert "membrane-voltage" in analyzed.messages[0]
    assert len(analyzed.missing_input_requirements) == 1
    requirement = analyzed.missing_input_requirements[0]
    assert requirement.required_signals == (axs.signals.Vm,)
    assert requirement.required_result_fields == ("Vm",)
    assert requirement.recording_hint is not None


def test_non_membrane_voltage_analysis_is_not_applicable_until_supported():
    custom = axs.Signal(
        id=axs.SignalId("custom_signal_for_analysis_test"),
        result_key="custom",
    )
    result = _fake_result()

    analyzed = result.analyze(axs.analysis.Activation(signal=custom))

    assert analyzed.status is axs.analysis.AnalysisStatus.NOT_APPLICABLE


def test_spike_count_and_velocity_use_structured_undetermined_status():
    propagated = _fake_result(second_peak=True)
    local_only = _fake_result(second_peak=False)

    spike_count = propagated.analyze(
        axs.analysis.SpikeCount(threshold=0.0 * axs.mV, min_distance=2.0 * axs.ms)
    )
    velocity = propagated.analyze(axs.analysis.ConductionVelocity(threshold=0.0 * axs.mV))
    no_velocity = local_only.analyze(axs.analysis.ConductionVelocity(threshold=0.0 * axs.mV))

    assert spike_count.value == 2
    assert velocity.status is axs.analysis.AnalysisStatus.VALID
    assert velocity.value > 0.0
    assert no_velocity.status is axs.analysis.AnalysisStatus.UNDETERMINED


def test_activation_online_observer_cross_validates_posthoc_result():
    result = _fake_result()
    definition = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    observer = definition.online_observer(
        positions=result.position_values(unit=axs.um) * axs.um,
        original_indices=result.record_indices,
    )

    observer.update(result.t[:2000] * axs.ms, result.Vm[:2000] * axs.mV)
    observer.update(result.t[2000:] * axs.ms, result.Vm[2000:] * axs.mV)
    online = observer.finalize()
    posthoc = result.analyze(definition)

    assert isinstance(observer, axs.analysis.ActivationObserver)
    assert observer.requirements == definition.requirements
    assert online.value == posthoc.value
    assert online.status is posthoc.status
    assert online.events[0].first_index == posthoc.events[0].first_index
    assert online.events[0].first_time_ms == pytest.approx(posthoc.events[0].first_time_ms)


def test_peak_voltage_online_observer_cross_validates_posthoc_result():
    result = _fake_result()
    definition = axs.analysis.PeakVoltage(target=axs.positions.CENTER)
    observer = definition.online_observer(
        positions=result.position_values(unit=axs.um) * axs.um,
        original_indices=result.record_indices,
    )

    observer.update(result.t[:1500] * axs.ms, result.Vm[:1500] * axs.mV)
    observer.update(result.t[1500:] * axs.ms, result.Vm[1500:] * axs.mV)
    online = observer.finalize()
    posthoc = result.analyze(definition)

    assert isinstance(observer, axs.analysis.PeakVoltageObserver)
    assert observer.requirements == definition.requirements
    assert online.value == pytest.approx(posthoc.value)
    assert online.status is posthoc.status
