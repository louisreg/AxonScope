import numpy as np
import jax.numpy as jnp
import pytest

from axonscope.axons.unmyelinated import HodgkinHuxley
from axonscope.solvers.CrankNicholson import (
    CrankNicholson,
    CrankNicholson_unoptimized,
    CrankNicholsonSemiImplicit,
    CrankNicholsonImplicit,
    CrankNicholsonImplicitFast,
    CrankNicholsonImplicitFastMultiStep,
    CrankNicholsonQuasiNewtonFast,
)

from axonscope.solvers.Euler import Euler


def _hh_axon(Nx: int = 51) -> HodgkinHuxley:
    axon = HodgkinHuxley(L=1000.0, d=0.5, Nx=Nx, celsius=6.3)
    axon.insert_I_Clamp(position=500.0, t_start=1.0, duration=1.0, amplitude=2.0)
    return axon


CONCRETE_SOLVERS = [
    pytest.param(Euler(), 0.001, 5.0, id="Euler"),
    pytest.param(CrankNicholson_unoptimized(), 0.01, 5.0, id="CrankNicholson_unoptimized"),
    pytest.param(CrankNicholson(), 0.01, 5.0, id="CrankNicholson"),
    pytest.param(CrankNicholsonSemiImplicit(), 0.01, 5.0, id="CrankNicholsonSemiImplicit"),
    pytest.param(CrankNicholsonImplicit(n_newton=3), 0.01, 5.0, id="CrankNicholsonImplicit"),
    pytest.param(CrankNicholsonImplicitFast(), 0.01, 5.0, id="CrankNicholsonImplicitFast"),
    pytest.param(
        CrankNicholsonImplicitFastMultiStep(),
        0.01,
        5.0,
        id="CrankNicholsonImplicitFastMultiStep",
    ),
    pytest.param(CrankNicholsonQuasiNewtonFast(), 0.01, 5.0, id="CrankNicholsonQuasiNewtonFast"),
]


CN_FAMILY = [
    pytest.param(CrankNicholson_unoptimized(), 0.01, id="dense"),
    pytest.param(CrankNicholson(), 0.01, id="tridiag"),
    pytest.param(CrankNicholsonSemiImplicit(), 0.01, id="semi-implicit"),
    pytest.param(CrankNicholsonImplicit(n_newton=3), 0.01, id="implicit"),
    pytest.param(CrankNicholsonImplicitFast(), 0.01, id="implicit-fast"),
    pytest.param(CrankNicholsonImplicitFastMultiStep(), 0.01, id="implicit-fast-multistep"),
    pytest.param(CrankNicholsonQuasiNewtonFast(), 0.01, id="quasi-newton-fast"),
]


def test_cranknicholson_dense_and_tridiagonal_match():
    x_uniform = jnp.linspace(-1.0, 1.0, 31, dtype=jnp.float32)
    x_vec = (jnp.sinh(1.5 * x_uniform) / jnp.sinh(1.5) + 1.0) * 150.0
    axon = HodgkinHuxley(x_vec=x_vec, d=1.0, Nx=None)
    axon.insert_I_Clamp(position=150.0, t_start=0.5, duration=0.5, amplitude=5.0)

    dense = CrankNicholson_unoptimized().solve(axon, tsim=2.0, dt=0.01)
    optimized = CrankNicholson().solve(axon, tsim=2.0, dt=0.01)

    np.testing.assert_allclose(np.asarray(dense.t), np.asarray(optimized.t), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(dense.t[0]), 0.01, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(np.asarray(dense.Vm), np.asarray(optimized.Vm), atol=1.5e-4, rtol=0.0)


@pytest.mark.parametrize(("solver", "dt", "tsim"), CONCRETE_SOLVERS)
def test_all_concrete_solvers_return_finite_output(solver, dt, tsim):
    Nx = 51
    Nt = int(np.ceil(tsim / dt))
    res = solver.solve(_hh_axon(Nx), tsim=tsim, dt=dt)

    assert res.Vm.shape == (Nt, Nx)
    assert res.t.shape == (Nt,)
    assert not np.any(np.isnan(np.asarray(res.Vm)))
    np.testing.assert_allclose(np.asarray(res.t[0]), dt, atol=1e-8, rtol=0.0)


@pytest.mark.parametrize(("solver", "dt"), CN_FAMILY)
def test_cn_family_propagates_action_potential(solver, dt):
    res = solver.solve(_hh_axon(), tsim=10.0, dt=dt)
    tAP, _ = res.rasterize()

    assert len(tAP) > 5, "Expected AP detected at multiple compartments"
    velocity = res.average_velocity()
    assert np.isfinite(velocity)
    assert velocity > 0.0


def test_fast_variants_close_to_implicit_reference():
    ref = CrankNicholsonImplicit(n_newton=3).solve(_hh_axon(), tsim=10.0, dt=0.01)
    v_ref = ref.average_velocity()

    for solver in (
        CrankNicholsonImplicitFast(),
        CrankNicholsonImplicitFastMultiStep(),
        CrankNicholsonQuasiNewtonFast(),
    ):
        res = solver.solve(_hh_axon(), tsim=10.0, dt=0.01)
        v = res.average_velocity()
        assert np.isfinite(v)
        assert abs(v - v_ref) < 0.02


def test_cranknicholson_can_record_generic_membrane_observables():
    axon = _hh_axon(Nx=21)
    res = CrankNicholson().solve(axon, tsim=2.0, dt=0.01, record_observables=True)

    assert res.recordings is not None
    assert set(res.recordings) == {"gates", "currents", "conductances"}
    assert set(res.recordings["gates"]) == {"m", "h", "n"}
    assert set(res.recordings["currents"]) == {"I_na", "I_k", "I_l"}
    assert set(res.recordings["conductances"]) == {"g_na", "g_k", "g_l"}

    Nt, Nx = res.Vm.shape
    for group in res.recordings.values():
        for trace in group.values():
            arr = np.asarray(trace)
            assert arr.shape == (Nt, Nx)
            assert np.isfinite(arr).all()


def test_cranknicholson_aggregates_duplicate_current_and_conductance_names():
    axon = HodgkinHuxley(
        L=1000.0,
        d=0.5,
        Nx=21,
        celsius=6.3,
        include_passive_leak=True,
        g_pas=0.001,
        e_pas=-70.0,
    )
    axon.insert_I_Clamp(position=500.0, t_start=1.0, duration=0.5, amplitude=2.0)
    res = CrankNicholson().solve(axon, tsim=2.0, dt=0.01, record_observables=True)

    assert res.recordings is not None
    assert "I_l" in res.recordings["currents"]
    assert "I_l_2" not in res.recordings["currents"]
    assert "g_l" in res.recordings["conductances"]
    assert "g_l_2" not in res.recordings["conductances"]
