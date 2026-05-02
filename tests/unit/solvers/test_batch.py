from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from axonscope.axons import HodgkinHuxley
from axonscope.electrodes import PointSourceElectrode
from axonscope.solvers import SingleCableVStimBatchKernel
from axonscope.solvers.experimental import CrankNicholsonVStimForcing
from axonscope.solvers.runtime import prepare_solver_runtime
from axonscope.stimulus import Stimulus


def _hh_extracellular_axon() -> HodgkinHuxley:
    axon = HodgkinHuxley(L=400.0, d=0.5, Nx=41, celsius=6.3)
    axon.insert_I_Clamp(
        position=200.0,
        stimulus=Stimulus.pulse(start=0.4, duration=0.05, amplitude=0.8),
    )
    electrode = PointSourceElectrode(
        x0_m=200e-6,
        y0_m=100e-6,
        z0_m=100e-6,
        sigma_S_m=0.3,
    )
    stim = Stimulus.pulse(start=0.3, amplitude=20e-6, duration=0.1, baseline=0.0)
    axon.add_extracellular_context(electrode, stim, replace=True)
    return axon


def test_single_cable_vstim_batch_matches_scalar_reference_row():
    axon = _hh_extracellular_axon()
    tsim = 1.2
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=True,
        precompute_extracellular=True,
    )
    assert runtime.stimulation.intracellular_current_density_mid is not None
    assert runtime.stimulation.extracellular_potential_mid_mV is not None

    vext_mid = runtime.stimulation.extracellular_potential_mid_mV
    vext_batch = jnp.stack([vext_mid, 0.5 * vext_mid])
    batch = SingleCableVStimBatchKernel(
        runtime=runtime,
        Cm_uF_cm2=jnp.asarray(axon.Cm, dtype=runtime.membrane.dtype),
    ).run(extracellular_potential_mid_mV=vext_batch)
    scalar = CrankNicholsonVStimForcing().solve(_hh_extracellular_axon(), tsim=tsim, dt=dt)

    assert batch.Vm.shape == (2, scalar.Vm.shape[0], scalar.Vm.shape[1])
    np.testing.assert_allclose(np.asarray(batch.t), np.asarray(scalar.t), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(batch.Vm[0]), np.asarray(scalar.Vm), atol=1e-3, rtol=0.0)
    assert np.isfinite(np.asarray(batch.Vm)).all()
    assert float(np.max(np.abs(np.asarray(batch.Vm[0]) - np.asarray(batch.Vm[1])))) > 1e-8


def test_single_cable_vstim_batch_validates_shapes():
    axon = _hh_extracellular_axon()
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=0.2,
        dt_ms=0.01,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=True,
        precompute_extracellular=True,
    )
    kernel = SingleCableVStimBatchKernel(
        runtime=runtime,
        Cm_uF_cm2=jnp.asarray(axon.Cm, dtype=runtime.membrane.dtype),
    )

    with pytest.raises(ValueError, match="extracellular_potential_mid_mV"):
        kernel.run(extracellular_potential_mid_mV=jnp.zeros((runtime.grid.Nt, axon.Nx + 1)))
