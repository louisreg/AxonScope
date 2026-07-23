import numpy as np
import pytest

import axonfleet as axs
from axonfleet.results import AxonSimulationResult
from axonfleet.results.pool import _ResultBlock
from tests.helpers import FakeSingleAxonResult


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


def _fake_result(
    *,
    second_peak: bool = True,
    include_vm: bool = True,
) -> FakeSingleAxonResult:
    t = np.linspace(0.0, 50.0, 5001)
    vm = np.full((len(t), 3), -70.0)
    vm[:, 0] += np.exp(-0.5 * ((t - 10.0) / 0.5) ** 2) * 80.0
    if second_peak:
        vm[:, 1] += np.exp(-0.5 * ((t - 30.0) / 0.5) ** 2) * 80.0
    recordings = {"Vm": vm} if include_vm else {"gates": np.zeros((len(t), 3, 1))}
    return FakeSingleAxonResult(
        axon=_DummyAxon([0.0, 500.0, 1000.0]),
        t=t,
        recordings=recordings,
    )


def _fake_pool_result() -> AxonSimulationResult:
    active = _fake_result(second_peak=True)
    silent_distal = _fake_result(second_peak=False)
    cohort = _ResultBlock(
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
    assert not hasattr(axs, "Activation")
    assert axs.analysis.Activation is not None
    assert axs.analysis.AnalysisStatus.VALID.value == "VALID"


def test_activation_definition_declares_requirements_and_statuses():
    result = _fake_result()
    definition = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        target=axs.positions.CENTER,
    )

    analyzed = definition.evaluate(result)

    assert analyzed.name == "activation"
    assert analyzed.row_labels == (0,)
    assert analyzed.status is axs.analysis.AnalysisStatus.VALID
    assert analyzed.value is True
    assert analyzed.events[0].first_index == 1
    assert definition.requirements.required_signals == (axs.signals.Vm,)
    assert definition.requirements.required_result_fields == ("Vm", "t", "positions")
    assert definition.requirements.required_positions == (axs.positions.CENTER,)
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


def test_population_activation_dense_fast_path_matches_events(monkeypatch):
    from axonfleet.analysis import definitions as analysis_definitions

    result = _fake_pool_result()
    definition = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=20.0 * axs.ms,
        target=axs.positions.ALL,
    )

    def fail_row_fallback(*args, **kwargs):
        raise AssertionError("population Activation should use the dense fast path")

    monkeypatch.setattr(analysis_definitions, "_evaluate_rows", fail_row_fallback)

    analyzed = result.analyze(definition)

    np.testing.assert_array_equal(analyzed.values, [True, False])
    assert analyzed.statuses == (
        axs.analysis.AnalysisStatus.VALID,
        axs.analysis.AnalysisStatus.VALID,
    )
    assert analyzed.events[0].activated is True
    assert analyzed.events[0].first_index == 1
    assert analyzed.events[0].first_time_ms == pytest.approx(29.75)
    assert analyzed.events[0].peak_time_ms == pytest.approx(30.0)
    assert analyzed.events[1].activated is False
    assert analyzed.events[1].first_time_ms is None
    assert analyzed.events[1].peak_time_ms == pytest.approx(20.0)


def test_analysis_report_views_format_dataframe_and_plot(capsys):
    result = _fake_pool_result()
    report = result.report(
        axs.analysis.Activation(threshold=0.0 * axs.mV, target=axs.positions.CENTER),
        axs.analysis.PeakVoltage(target=axs.positions.CENTER),
    )

    text = report.format()
    rows = axs.analysis.views.analysis_report_rows(report)
    dataframe = report.to_dataframe()

    assert "AxonFleet analysis report" in text
    assert report.rows() == rows
    assert report["activation"].rows() == axs.analysis.views.analysis_result_rows(
        report["activation"]
    )
    assert rows[0]["analysis"] == "activation"
    assert rows[0]["row_label"] == 0
    assert set(dataframe["analysis"]) == {"activation", "peak_voltage"}
    assert list(dataframe["row"]) == [0, 1, 0, 1]
    report["activation"].print()
    assert "activation" in capsys.readouterr().out

    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    returned = report.plot(ax=ax)
    assert returned is ax
    assert len(ax.lines) == 2
    _, result_ax = plt.subplots()
    returned_result = report["peak_voltage"].plot(
        ax=result_ax,
        x=np.asarray([0.0, 1.0]),
        x_label="dose",
    )
    assert returned_result is result_ax
    assert len(result_ax.lines) == 1
    returned_result = report["peak_voltage"].plot(
        x_unit=None,
        x_label="row label",
    )
    assert returned_result.get_xlabel() == "row label"
    plt.close("all")


def test_analysis_missing_input_is_reported_per_row():
    result = _fake_result(include_vm=False)

    analyzed = result.analyze(axs.analysis.PeakVoltage())

    assert analyzed.status is axs.analysis.AnalysisStatus.MISSING_INPUT
    assert analyzed.population.n_failed == 1
    assert "membrane-voltage" in analyzed.messages[0]
    requirement = analyzed.input_requirements[0]
    assert requirement is not None
    assert requirement.required_signals == (axs.signals.Vm,)
    assert requirement.required_result_fields == ("Vm",)
    assert requirement.recording_hint is not None


def test_non_membrane_voltage_analysis_is_not_applicable_until_supported():
    custom = axs.signals.Signal(
        id=axs.identifiers.SignalId("custom_signal_for_analysis_test"),
        result_key="custom",
    )
    result = _fake_result()

    analyzed = result.analyze(axs.analysis.Activation(signal=custom))

    assert analyzed.status is axs.analysis.AnalysisStatus.NOT_APPLICABLE


def test_spike_count_and_velocity_use_structured_undetermined_status():
    propagated = _fake_result(second_peak=True)
    local_only = _fake_result(second_peak=False)

    spike_count = propagated.analyze(
        axs.analysis.SpikeCount(
            threshold=0.0 * axs.mV,
            reset_threshold=-20.0 * axs.mV,
            refractory=2.0 * axs.ms,
        )
    )
    velocity = propagated.analyze(axs.analysis.ConductionVelocity(threshold=0.0 * axs.mV))
    no_velocity = local_only.analyze(axs.analysis.ConductionVelocity(threshold=0.0 * axs.mV))

    assert spike_count.value == 2
    assert velocity.status is axs.analysis.AnalysisStatus.VALID
    assert velocity.value > 0.0
    assert no_velocity.status is axs.analysis.AnalysisStatus.UNDETERMINED


def test_conduction_velocity_can_decode_vm_raster_observation():
    velocity = axs.analysis.ConductionVelocity(threshold=0.0 * axs.mV)
    raster = axs.VmRasterResult(
        words=np.asarray([[[[1 << 20], [1 << 28]]]], dtype=np.uint32),
        nt=64,
        dt_ms=0.5,
        definitions=(velocity,),
        names=(velocity.name,),
        probe_indices=np.asarray([[0, 1]], dtype=np.int32),
        probe_mask=np.asarray([[True, True]], dtype=bool),
        original_indices=np.asarray([[0, 1]], dtype=np.int32),
        positions_um=np.asarray([[0.0, 1000.0]], dtype=float),
        thresholds_mV=np.asarray([0.0], dtype=float),
    )
    cohort = _ResultBlock(
        input_indices=(0,),
        axons=(_DummyAxon([0.0, 1000.0]),),
        simulations=(None,),
        Vm=None,
        t=np.arange(64, dtype=float) * 0.5,
        diagnostics=({},),
        record_indices=(None,),
        observations={axs.VM_RASTER_OBSERVATION_KEY: raster},
    )
    result = AxonSimulationResult((cohort,), size=1)

    analyzed = result.analyze(velocity)

    assert analyzed.status is axs.analysis.AnalysisStatus.VALID
    np.testing.assert_allclose(analyzed.values, [0.25])


def test_conduction_velocity_is_vm_raster_compatible_observer():
    from axonfleet.runtime.outputs.contracts import (
        observers_are_vm_raster_compatible,
        vm_raster_definitions,
    )

    velocity = axs.analysis.ConductionVelocity()

    assert observers_are_vm_raster_compatible((velocity,))
    assert vm_raster_definitions((velocity,)) == (velocity,)
