import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import axonscope as axs
from axonscope.results import SimResult
from axonscope.results.activation import ActivationCriterion, detect_activation
from axonscope.results.analysis import conduction_velocity, rasterize, recorded_positions_um
from axonscope.axons.unmyelinated import RattayAberham
from axonscope.solvers.crank_nicholson import CrankNicholson
from axonscope.stimulation import Stimulus
from axonscope.results.visualization import plot_raster


class _DummyLayout:
    def __init__(self, Nx):
        self._x_um = np.linspace(0, 1000, Nx)

    def position_values(self, *, unit="micrometer"):
        if unit in ("millimeter", axs.mm):
            return self._x_um * 1e-3
        return self._x_um


class _DummyAxon:
    def __init__(self, n_compartments):
        self.n_compartments = n_compartments
        self.layout = _DummyLayout(n_compartments)


@pytest.fixture
def fake_result():
    t = np.linspace(0, 50, 5001)
    Vm = np.zeros((len(t), 3))
    Vm[:, 0] = np.exp(-0.5 * ((t - 10) / 0.5) ** 2) * 80 - 70
    Vm[:, 1] = np.exp(-0.5 * ((t - 30) / 0.5) ** 2) * 80 - 70
    return SimResult(axon=_DummyAxon(n_compartments=3), Vm=Vm, t=t)


# ── rasterize ─────────────────────────────────────────────────────────────────

def test_rasterize_detects_spikes(fake_result):
    tAP, xAP = rasterize(fake_result, threshold_mV=0.0, min_distance_ms=2.0)
    assert np.allclose(tAP, [10, 30], atol=0.5)
    assert np.allclose(xAP, fake_result.axon.layout.position_values()[:2])


def test_rasterplot_uses_axons_x(fake_result):
    _, ax = plt.subplots()
    plot_raster(fake_result, ax=ax, threshold_mV=0.0, min_distance_ms=2.0)
    assert "Axon position" in ax.get_ylabel()
    plt.close("all")


def test_analysis_uses_record_indices_for_filtered_results(fake_result):
    filtered = SimResult(
        axon=fake_result.axon,
        Vm=fake_result.Vm[:, [1]],
        t=fake_result.t,
        record_indices=(1,),
    )

    assert np.allclose(recorded_positions_um(filtered), [500.0])
    _, xAP = rasterize(filtered, threshold_mV=0.0, min_distance_ms=2.0)
    assert np.allclose(xAP, [500.0])


def test_analysis_rejects_filtered_results_without_position_mapping(fake_result):
    filtered = SimResult(
        axon=fake_result.axon,
        Vm=fake_result.Vm[:, [1]],
        t=fake_result.t,
    )

    with pytest.raises(ValueError, match="record_indices"):
        recorded_positions_um(filtered)


def test_result_value_helpers_convert_units(fake_result):
    assert fake_result.recordings is not None
    np.testing.assert_allclose(fake_result.recordings["Vm"], fake_result.Vm)
    np.testing.assert_allclose(fake_result.time_values(unit=axs.ms), fake_result.t)
    np.testing.assert_allclose(fake_result.position_values(unit=axs.mm), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(fake_result.voltage_values(unit=axs.mV), fake_result.Vm)
    np.testing.assert_allclose(fake_result.voltage_values(unit=axs.V), fake_result.Vm * 1e-3)
    np.testing.assert_allclose(fake_result.peak_voltage_values(unit=axs.mV), [10.0, 10.0, 0.0])

    t_ms, vm_mV = fake_result.trace_values(position=0.5 * axs.mm)
    np.testing.assert_allclose(t_ms, fake_result.t)
    np.testing.assert_allclose(vm_mV, fake_result.Vm[:, 1])


def test_activation_criterion_detects_first_crossing(fake_result):
    event = detect_activation(
        fake_result,
        threshold=1.0 * axs.mV,
        blanking=0.0 * axs.ms,
    )

    assert event.activated
    assert event.first_index == 0
    assert event.first_position_um == pytest.approx(0.0)
    assert 9.0 < event.first_time_ms < 10.0
    assert event.peak_mV == pytest.approx(10.0, abs=1e-3)


def test_activation_criterion_respects_blanking_and_indices(fake_result):
    event = ActivationCriterion(
        threshold=1.0 * axs.mV,
        blanking=20.0 * axs.ms,
        indices=[1],
    ).evaluate(fake_result)

    assert event.activated
    assert event.first_index == 1
    assert event.first_position_um == pytest.approx(500.0)
    assert 29.0 < event.first_time_ms < 30.0


def test_activation_criterion_handles_filtered_results(fake_result):
    filtered = SimResult(
        axon=fake_result.axon,
        Vm=fake_result.Vm[:, [1]],
        t=fake_result.t,
        record_indices=(1,),
    )

    event = ActivationCriterion(threshold=1.0 * axs.mV, indices=[1]).evaluate(filtered)

    assert event.activated
    assert event.first_index == 1
    assert event.first_position_um == pytest.approx(500.0)


def test_activation_criterion_reports_no_activation(fake_result):
    event = ActivationCriterion(threshold=100.0 * axs.mV).evaluate(fake_result)

    assert not event.activated
    assert event.first_time_ms is None
    assert event.first_index is None
    assert event.peak_mV == pytest.approx(10.0, abs=1e-3)


def test_result_plot_trace_uses_nearest_position(fake_result):
    _, ax = plt.subplots()
    returned = fake_result.plot_trace(ax=ax, position=0.5 * axs.mm)
    assert returned is ax
    assert len(ax.lines) == 1
    assert "x=500" in ax.get_title()
    plt.close("all")


def test_result_plot_map_uses_recorded_positions(fake_result):
    _, ax = plt.subplots()
    returned = fake_result.plot_map(ax=ax, position_unit=axs.mm, voltage_unit=axs.mV)
    assert returned is ax
    assert len(ax.images) == 1
    assert "mm" in ax.get_ylabel()
    plt.close("all")


def test_rasterize_rejects_negative_min_distance(fake_result):
    with pytest.raises(ValueError, match="min_distance_ms"):
        rasterize(fake_result, min_distance_ms=-1.0)


# ── average_velocity ──────────────────────────────────────────────────────────

def test_compute_propagation_velocity():
    L, d, Nx = 1000, 0.5, 101
    axon = axs.AxonSimulation(
        RattayAberham(length=L * axs.um, diameter=d * axs.um, compartments=Nx, celsius=37 * axs.degC)
    )
    axon.add_current_clamp(position_um=L / 2,
        current=Stimulus.pulse(start=1.0, duration=1.0, amplitude=2),
    )

    simres = CrankNicholson().solve(axon, tsim=10.0, dt=0.01)
    velocity = conduction_velocity(simres)

    assert velocity is not None
    assert np.isfinite(velocity)
    assert velocity > 0.0
