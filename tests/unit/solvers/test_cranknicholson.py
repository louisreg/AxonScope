import numpy as np
import jax.numpy as jnp
import pytest

import axonscope as axs
from axonscope.analysis import conduction_velocity, rasterize
from axonscope import AxonInstance
from axonscope.axons.unmyelinated import HodgkinHuxley
from axonscope.solvers.crank_nicholson import (
    CrankNicholson,
)
from axonscope.backends.jax.experimental import (
    CrankNicholson_unoptimized,
)
from axonscope.backends.jax.kernels import SingleCableKernel
from axonscope.backends.jax.runtime import prepare_solver_runtime
from axonscope.stimulation import Stimulus
from axonscope.timebase import simulation_step_count


def _hh_axon(Nx: int = 51) -> AxonInstance:
    axon = AxonInstance(
        HodgkinHuxley(
            length=1000.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=Nx,
            celsius=6.3 * axs.degC,
        )
    )
    axon.add_current_clamp(position=500.0 * axs.um,
        current=Stimulus.pulse(start=1.0 * axs.ms, duration=1.0 * axs.ms, amplitude=2.0),
    )
    return axon


CONCRETE_SOLVERS = [
    pytest.param(CrankNicholson_unoptimized(), 0.01, 5.0, id="CrankNicholson_unoptimized"),
    pytest.param(CrankNicholson(), 0.01, 5.0, id="CrankNicholson"),
]


CN_FAMILY = [
    pytest.param(CrankNicholson_unoptimized(), 0.01, id="dense"),
    pytest.param(CrankNicholson(), 0.01, id="tridiag"),
]


def test_cranknicholson_dense_and_tridiagonal_match():
    x_uniform = jnp.linspace(-1.0, 1.0, 31, dtype=jnp.float32)
    x_vec = (jnp.sinh(1.5 * x_uniform) / jnp.sinh(1.5) + 1.0) * 150.0
    axon = AxonInstance(HodgkinHuxley(x=x_vec * axs.um, diameter=1.0 * axs.um))
    axon.add_current_clamp(position=150.0 * axs.um,
        current=Stimulus.pulse(start=0.5 * axs.ms, duration=0.5 * axs.ms, amplitude=5.0),
    )

    dense = CrankNicholson_unoptimized().solve(axon, tsim=2.0, dt=0.01)
    optimized = CrankNicholson().solve(axon, tsim=2.0, dt=0.01)

    np.testing.assert_allclose(np.asarray(dense.t), np.asarray(optimized.t), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(dense.t[0]), 0.01, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(np.asarray(dense.Vm), np.asarray(optimized.Vm), atol=3e-4, rtol=0.0)


def test_single_cable_kernel_matches_public_solver_path():
    axon = _hh_axon(Nx=31)
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=2.0,
        dt_ms=0.01,
        include_extracellular=False,
        include_area=False,
    )

    direct = SingleCableKernel(
        runtime=runtime,
        Cm_uF_cm2=jnp.asarray(runtime.axon.Cm_uF_cm2, dtype=runtime.membrane.dtype),
    ).run()
    public = CrankNicholson().solve(axon, tsim=2.0, dt=0.01)

    np.testing.assert_allclose(np.asarray(direct.t), np.asarray(public.t), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(direct.Vm), np.asarray(public.Vm), atol=0.0, rtol=0.0)

    precomputed_runtime = prepare_solver_runtime(
        axon,
        tsim_ms=2.0,
        dt_ms=0.01,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=True,
    )
    assert precomputed_runtime.stimulation.intracellular_current_density_mid is not None


@pytest.mark.parametrize(("solver", "dt", "tsim"), CONCRETE_SOLVERS)
def test_all_concrete_solvers_return_finite_output(solver, dt, tsim):
    Nx = 51
    Nt = simulation_step_count(tsim, dt)
    res = solver.solve(_hh_axon(Nx), tsim=tsim, dt=dt)

    assert res.Vm.shape == (Nt, Nx)
    assert res.t.shape == (Nt,)
    assert not np.any(np.isnan(np.asarray(res.Vm)))
    np.testing.assert_allclose(np.asarray(res.t[0]), dt, atol=1e-8, rtol=0.0)


@pytest.mark.parametrize(("solver", "dt"), CN_FAMILY)
def test_cn_family_propagates_action_potential(solver, dt):
    res = solver.solve(_hh_axon(), tsim=10.0, dt=dt)
    tAP, _ = rasterize(res)

    assert len(tAP) > 5, "Expected AP detected at multiple compartments"
    velocity = conduction_velocity(res)
    assert np.isfinite(velocity)
    assert velocity > 0.0


def test_cranknicholson_can_record_generic_membrane_observables():
    axon = _hh_axon(Nx=21)
    res = CrankNicholson().solve(axon, tsim=2.0, dt=0.01, record_observables=True)

    assert res.recordings is not None
    assert set(res.recordings) == {"Vm", "gates", "currents", "conductances"}
    assert set(res.recordings["gates"]) == {"m", "h", "n"}
    assert set(res.recordings["currents"]) == {"I_na", "I_k", "I_l"}
    assert set(res.recordings["conductances"]) == {"g_na", "g_k", "g_l"}

    Nt, Nx = res.Vm.shape
    for group_name, group in res.recordings.items():
        if group_name == "Vm":
            continue
        for trace in group.values():
            arr = np.asarray(trace)
            assert arr.shape == (Nt, Nx)
            assert np.isfinite(arr).all()


def test_cranknicholson_aggregates_duplicate_current_and_conductance_names():
    axon = AxonInstance(
        HodgkinHuxley(
            length=1000.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=21,
            celsius=6.3 * axs.degC,
            include_passive_leak=True,
            g_pas=0.001,
            e_pas=-70.0,
        )
    )
    axon.add_current_clamp(position=500.0 * axs.um,
        current=Stimulus.pulse(start=1.0 * axs.ms, duration=0.5 * axs.ms, amplitude=2.0),
    )
    res = CrankNicholson().solve(axon, tsim=2.0, dt=0.01, record_observables=True)

    assert res.recordings is not None
    assert "I_l" in res.recordings["currents"]
    assert "I_l_2" not in res.recordings["currents"]
    assert "g_l" in res.recordings["conductances"]
    assert "g_l_2" not in res.recordings["conductances"]
