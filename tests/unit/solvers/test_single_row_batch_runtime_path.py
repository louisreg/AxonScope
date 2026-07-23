import numpy as np
import pytest

import axonfleet as axs
from axonfleet import AxonInstance
from axonfleet.analysis import ConductionVelocity, rasterize
from axonfleet.axons.unmyelinated import (
    HodgkinHuxley,
    Schild94,
    Schild97,
    Tigerholm,
)
from axonfleet.stimulation import Stimulus
from axonfleet.runtime.timebase import simulation_step_count


def _hh_axon(nx: int = 51) -> AxonInstance:
    axon = AxonInstance(
        HodgkinHuxley(
            length=1000.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=nx,
            celsius=6.3 * axs.degC,
        )
    )
    axon.add_current_clamp(
        position=500.0 * axs.um,
        current=Stimulus.pulse(
            start=1.0 * axs.ms,
            duration=1.0 * axs.ms,
            amplitude=2.0,
        ),
    )
    return axon


def _run_public(axon: AxonInstance, *, tsim: float, dt: float, recording=None):
    return axs.AxonSimulation(
        axon,
        duration=tsim,
        dt=dt,
        recording=recording,
    ).run().single


def test_single_row_public_run_uses_batch_route():
    result = _run_public(_hh_axon(31), tsim=2.0, dt=0.01)

    assert result.diagnostics["dispatch_method"] == "batch-single-cable"
    assert result.diagnostics["dispatch_group_size"] == 1
    assert result.diagnostics["dispatch_batch_kind"] != "scalar"


def test_single_row_batch_route_returns_finite_output():
    tsim = 5.0
    dt = 0.01
    nx = 51
    nt = simulation_step_count(tsim, dt)
    result = _run_public(_hh_axon(nx), tsim=tsim, dt=dt)

    assert result.Vm.shape == (nt, nx)
    assert result.t.shape == (nt,)
    assert not np.any(np.isnan(np.asarray(result.Vm)))
    np.testing.assert_allclose(np.asarray(result.t[0]), dt, atol=1e-8, rtol=0.0)


@pytest.mark.parametrize(
    "axon",
    (
        Tigerholm(length=100.0 * axs.um, diameter=1.0 * axs.um, compartments=5),
        Schild94(length=100.0 * axs.um, diameter=0.8 * axs.um, compartments=5),
        Schild97(length=100.0 * axs.um, diameter=0.8 * axs.um, compartments=5),
    ),
    ids=("tigerholm", "schild94", "schild97"),
)
def test_stateful_membranes_execute_through_single_row_batch_route(axon):
    result = _run_public(AxonInstance(axon), tsim=0.01, dt=0.005)

    assert result.diagnostics["dispatch_method"] == "batch-single-cable"
    assert np.all(np.isfinite(np.asarray(result.Vm)))


def test_stateful_membrane_executes_through_compact_observer_route():
    axon = AxonInstance(
        Tigerholm(length=100.0 * axs.um, diameter=1.0 * axs.um, compartments=5)
    )
    activation = axs.analysis.Activation(
        threshold=-100.0 * axs.mV,
        target=axs.positions.CENTER,
    )

    result = axs.AxonSimulation(
        axon,
        duration=0.01 * axs.ms,
        dt=0.005 * axs.ms,
        recording=axs.Recording.none(),
        observers=(activation,),
    ).run()

    assert result.observations is not None
    assert bool(np.asarray(result.observations["activation"].values)[0])


def test_single_row_batch_route_propagates_action_potential():
    result = _run_public(_hh_axon(), tsim=10.0, dt=0.01)
    t_ap, _ = rasterize(result)

    assert len(t_ap) > 5
    velocity = ConductionVelocity().detect(result)
    assert np.isfinite(velocity)
    assert velocity > 0.0


def test_dense_observable_recording_does_not_fall_back_to_scalar_route():
    nx = 21
    nt = simulation_step_count(2.0, 0.01)

    result = _run_public(
        _hh_axon(nx),
        tsim=2.0,
        dt=0.01,
        recording=axs.Recording(
            voltage=True,
            gates=True,
            currents=True,
            conductances=True,
        ),
    )

    assert result.diagnostics["dispatch_method"] == "batch-single-cable"
    assert result.diagnostics["dispatch_batch_kind"] != "scalar"
    assert result.Vm.shape == (nt, nx)
    assert result.recordings is not None
    assert set(result.recordings) == {"Vm", "gates", "currents", "conductances"}
    assert set(result.signal(axs.signals.GATES)) == {
        "hodgkin_huxley.m",
        "hodgkin_huxley.h",
        "hodgkin_huxley.n",
    }
    assert set(result.signal(axs.signals.CURRENTS)) == {
        "I_na",
        "I_k",
        "I_l",
    }
    assert set(result.signal(axs.signals.CONDUCTANCES)) == {
        "g_na",
        "g_k",
        "g_l",
    }
    for group in ("gates", "currents", "conductances"):
        for values in result.recordings[group].values():
            assert np.asarray(values).shape == (nt, nx)
            assert np.all(np.isfinite(np.asarray(values)))
