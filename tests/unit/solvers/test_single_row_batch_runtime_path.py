import numpy as np
import pytest

import axonscope as axs
from axonscope import AxonInstance
from axonscope.analysis import conduction_velocity, rasterize
from axonscope.axons.unmyelinated import HodgkinHuxley
from axonscope.stimulation import Stimulus
from axonscope.timebase import simulation_step_count


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


def test_single_row_batch_route_propagates_action_potential():
    result = _run_public(_hh_axon(), tsim=10.0, dt=0.01)
    t_ap, _ = rasterize(result)

    assert len(t_ap) > 5
    velocity = conduction_velocity(result)
    assert np.isfinite(velocity)
    assert velocity > 0.0


def test_dense_observable_recording_does_not_fall_back_to_scalar_route():
    with pytest.raises(NotImplementedError, match="Vm only"):
        _run_public(
            _hh_axon(21),
            tsim=2.0,
            dt=0.01,
            recording=axs.Recording(
                voltage=True,
                gates=True,
                currents=True,
                conductances=True,
            ),
        )
