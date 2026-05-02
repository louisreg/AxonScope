from __future__ import annotations

import numpy as np

from axonscope.axons.base import AxonBase
from axonscope.axons import HodgkinHuxley
from axonscope.channel_models.passive import PassiveICM
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


def test_precompute_extracellular_potential_matches_axon_method():
    axon = HodgkinHuxley(L=300.0, d=0.5, Nx=11, celsius=6.3)
    electrode = PointSourceElectrode(
        x0_m=150e-6,
        y0_m=100e-6,
        sigma_S_m=0.2,
    )
    stim = Stimulus.pulse(start=0.2, duration=0.1, amplitude=10e-6)
    axon.add_extracellular_context(electrode, stim, replace=True)

    t_ms = np.asarray([0.1, 0.25, 0.5], dtype=float)
    sampled = np.asarray(precompute_extracellular_potential_mV(axon, t_ms))

    assert sampled.shape == (3, axon.Nx)
    assert np.allclose(sampled[0], np.asarray(axon.extracellular_potential_mV(0.1)))
    assert np.allclose(sampled[1], np.asarray(axon.extracellular_potential_mV(0.25)))
    assert np.allclose(sampled[2], np.asarray(axon.extracellular_potential_mV(0.5)))


def test_prepare_solver_runtime_precomputes_extracellular_step_potentials():
    axon = AxonBase(
        ion_channel=PassiveICM(Rm=1e4, EL=-70.0),
        L=300.0,
        d=1.0,
        Nx=11,
        Vinit=-70.0,
    )
    axon.set_extracellular_layer(
        xraxial_MOhm_per_cm=np.full((axon.Nx,), 1e8, dtype=float),
        xg_S_per_cm2=np.full((axon.Nx,), 1e-3, dtype=float),
        xc_uF_per_cm2=np.full((axon.Nx,), 0.01, dtype=float),
        use_extracellular=True,
    )
    electrode = PointSourceElectrode(
        x0_m=150e-6,
        y0_m=100e-6,
        sigma_S_m=0.2,
    )
    stim = Stimulus.pulse(start=0.2, duration=0.1, amplitude=10e-6)
    axon.add_extracellular_context(electrode, stim, replace=True)

    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=1.0,
        dt_ms=0.1,
        include_extracellular=True,
    )

    vext_mid = runtime.stimulation.extracellular_potential_mid_mV
    vext_initial_previous = runtime.stimulation.extracellular_potential_initial_previous_mV
    assert vext_mid is not None
    assert vext_initial_previous is not None
    assert vext_mid.shape == (runtime.grid.Nt, axon.Nx)
    assert vext_initial_previous.shape == (axon.Nx,)
    assert np.allclose(np.asarray(vext_mid[0]), np.asarray(axon.extracellular_potential_mV(0.05)))
    assert np.allclose(np.asarray(vext_initial_previous), np.asarray(axon.extracellular_potential_mV(-0.05)))
    assert np.allclose(np.asarray(vext_mid[2]), np.asarray(axon.extracellular_potential_mV(0.25)))
