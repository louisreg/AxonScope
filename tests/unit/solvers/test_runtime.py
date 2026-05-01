from __future__ import annotations

import numpy as np

from axonscope.axons import HodgkinHuxley
from axonscope.electrodes import PointSourceElectrode
from axonscope.solvers.runtime import (
    precompute_extracellular_potential_mV,
    prepare_solver_runtime,
)
from axonscope.stimulus import Stimulus


def test_prepare_solver_runtime_collects_membrane_cable_and_stimulus_arrays():
    axon = HodgkinHuxley(L=300.0, d=0.5, Nx=11, celsius=6.3)
    axon.insert_I_Clamp(
        position=150.0,
        stimulus=Stimulus.pulse(start=0.2, duration=0.1, amplitude=1.0),
    )

    runtime = prepare_solver_runtime(axon, tsim_ms=1.0, dt_ms=0.1)

    assert runtime.grid.Nt == 10
    assert runtime.grid.t_vec_ms.shape == (10,)
    assert runtime.membrane.Nx == 11
    assert runtime.membrane.Vm0_mV.shape == (11,)
    assert runtime.membrane.gates0.shape[0] == 11
    assert runtime.cable.lower.shape == (11,)
    assert runtime.cable.area_cm2.shape == (11,)

    inj_on = np.asarray(runtime.stimulation.intracellular_current_density(0.25))
    inj_off = np.asarray(runtime.stimulation.intracellular_current_density(0.5))
    assert inj_on.max() > 0.0
    assert np.allclose(inj_off, 0.0)


def test_precompute_extracellular_potential_matches_callable_wrapper():
    axon = HodgkinHuxley(L=300.0, d=0.5, Nx=11, celsius=6.3)
    electrode = PointSourceElectrode(
        x0_m=150e-6,
        y0_m=100e-6,
        sigma_S_m=0.2,
    )
    stim = Stimulus.pulse(start=0.2, duration=0.1, amplitude=10e-6)
    axon.add_extracellular_ctx(electrode, stim, replace=True)

    t_ms = np.asarray([0.1, 0.25, 0.5], dtype=float)
    sampled = np.asarray(precompute_extracellular_potential_mV(axon, t_ms))

    assert sampled.shape == (3, axon.Nx)
    assert np.allclose(sampled[0], np.asarray(axon.Vext_mV(0.1)))
    assert np.allclose(sampled[1], np.asarray(axon.Vext_mV(0.25)))
    assert np.allclose(sampled[2], np.asarray(axon.Vext_mV(0.5)))
