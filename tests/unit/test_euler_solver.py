import jax.numpy as jnp
import pytest
from axonscope.solvers import Euler
from axonscope.axons import AxonBase
from axonscope.channel_models.passive import PassiveICM

def test_eulerjax_passive_simulation():
    # Création d'un axon minimal avec un modèle passif
    class DummyAxon(AxonBase):
        Nx = 5
        Vinit = -65.0
        dx_cm = 0.01
        D = 0.1

        def __init__(self):
            self.ion_channel = PassiveICM(Rm=1e4, EL=-70.0)
            self.Cm = 1.0

        def Iinj_uAcm2(self, t):
            return jnp.zeros(self.Nx)  # pas de courant injecté

    axon = DummyAxon()
    solver = Euler()
    tsim = 1.0  # ms
    dt = 0.1    # ms

    result = solver.solve(axon, tsim, dt)

    # Vérifications
    Nt = int(jnp.ceil(tsim / dt))
    assert result.Vm.shape == (Nt, axon.Nx)
    assert result.t.shape == (Nt,)
    assert not jnp.any(jnp.isnan(result.Vm))
    assert jnp.all(result.Vm[:, 0] == axon.Vinit)  # conditions aux limites
    assert jnp.all(result.Vm[:, -1] == axon.Vinit) # conditions aux limites
