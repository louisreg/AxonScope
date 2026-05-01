import jax.numpy as jnp
import numpy as np
import pytest
from axonscope.solvers.Euler import Euler
from axonscope.axons.unmyelinated import AxonBase
from axonscope.channel_models.passive import PassiveICM

def test_eulerjax_passive_simulation():
    # Création d'un axon minimal avec un modèle passif
    class DummyAxon(AxonBase):
        def __init__(self):
            super().__init__(
                ion_channel=PassiveICM(Rm=1e4, EL=-70.0),
                L=400.0,
                d=1.0,
                Nx=5,
                Vinit=-65.0,
            )

    axon = DummyAxon()
    solver = Euler()
    tsim = 1.0  # ms
    dt = 0.1    # ms

    result = solver.solve(axon, tsim, dt)

    # Vérifications
    Nt = int(jnp.ceil(tsim / dt))
    assert result.Vm.shape == (Nt, axon.Nx)
    assert result.t.shape == (Nt,)
    assert result.t[0] == pytest.approx(dt)
    assert not jnp.any(jnp.isnan(result.Vm))
    assert jnp.allclose(result.Vm[:, 0], result.Vm[:, 1])    # sealed-end BC
    assert jnp.allclose(result.Vm[:, -1], result.Vm[:, -2])  # sealed-end BC


def test_euler_can_record_generic_membrane_observables():
    class DummyPassiveAxon(AxonBase):
        def __init__(self):
            super().__init__(
                ion_channel=PassiveICM(Rm=1e4, EL=-70.0),
                L=100.0,
                d=1.0,
                Nx=5,
                Vinit=-65.0,
            )

    res = Euler().solve(DummyPassiveAxon(), tsim=0.5, dt=0.1, record_observables=True)

    assert res.recordings is not None
    assert set(res.recordings) == {"currents", "conductances"}
    assert set(res.recordings["currents"]) == {"I_l"}
    assert set(res.recordings["conductances"]) == {"g_l"}

    Nt, Nx = res.Vm.shape
    for group in res.recordings.values():
        for trace in group.values():
            arr = np.asarray(trace)
            assert arr.shape == (Nt, Nx)
            assert np.isfinite(arr).all()
