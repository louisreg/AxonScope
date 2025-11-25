import jax
import jax.numpy as jnp
import pytest

from axonscope.channel_models.passive import PassiveICM
from axonscope.channel_models.hodgkin_huxley import HodgkinHuxleyICM 
from axonscope.icm_compute import Gating
from axonscope.settings import dtype

# ----------------- Tests Passive -----------------
def test_passive_model():
    model = PassiveICM(Rm=1e4, EL=-70.0)
    
    # Test init gates
    V0 = jnp.array([-65.0, -60.0])
    gates = model.init_gates(V0)
    assert gates.shape == (2, 0)
    
    # Test g_bar and E_rev
    assert model.g_bar.shape == (1,)
    assert model.E_rev.shape == (1,)
    
    # Test g_func
    g_vals = model.g_funcs(gates, model.g_bar)
    assert g_vals.shape == (2, 1)
    assert jnp.allclose(g_vals, model.g_bar[0])
    
    # Test compute currents
    I = Gating.compute_currents(V0, gates, model.g_bar, model.g_funcs, model.E_rev)
    assert I.shape == (2,)
    # Passive current formula: I = g_leak*(V - EL)
    assert jnp.allclose(I, model.g_bar[0] * (V0 - model.E_rev[0]))

# ----------------- Tests Hodgkin-Huxley -----------------
def test_hh_model():
    model = HodgkinHuxleyICM()
    V0 = jnp.array([-65.0, -60.0])
    
    # Test init gates
    gates = model.init_gates(V0)
    assert gates.shape == (2, 3)
    assert jnp.all((gates >= 0) & (gates <= 1))
    
    # Test g_bar and E_rev
    g_bar = jnp.array([model.gnabar, model.gkbar, model.gl], dtype=dtype)
    E_rev = jnp.array([model.ena, model.ek, model.el], dtype=dtype)
    assert g_bar.shape == (3,)
    assert E_rev.shape == (3,)
    
    # Test g_funcs
    g_vals = model.g_funcs(gates, g_bar)
    assert g_vals.shape == (2, 3)
    
    # Test compute currents
    I = Gating.compute_currents(V0, gates, g_bar, model.g_funcs, E_rev)
    assert I.shape == (2,)
    
    # Test rates
    g_inf, tau = Gating.rates(V0, model.q10, model.alpha_funcs, model.beta_funcs)
    assert g_inf.shape == (2, 3)
    assert tau.shape == (2, 3)
    
    # Test update gates
    dt = 0.1
    new_gates = Gating.update_gates(V=V0, dt=dt, gates = gates, q10=model.q10, alpha_fun=model.alpha_funcs, beta_fun=model.beta_funcs)
    assert new_gates.shape == (2, 3)
    assert jnp.all((new_gates >= 0) & (new_gates <= 1))

# ----------------- Tests JIT -----------------
def test_jit_compatibility():
    # Passive
    passive = PassiveICM()
    V = jnp.array([-65.0, -60.0])
    gates = passive.init_gates(V)
    g_bar = passive.g_bar

    # JIT g_funcs
    g_func_jit = jax.jit(passive.g_funcs)
    g_vals = g_func_jit(gates, g_bar)
    assert g_vals.shape == (2, 1)

    # JIT compute_currents
    compute_currents_jit = jax.jit(lambda V, gates, g_bar: Gating.compute_currents(V, gates, g_bar, passive.g_funcs, passive.E_rev))
    I = compute_currents_jit(V, gates, g_bar)
    assert I.shape == (2,)

    # Hodgkin-Huxley
    hh = HodgkinHuxleyICM()
    V = jnp.array([-65.0, -60.0])
    gates = hh.init_gates(V)
    g_bar = hh.g_bar

    g_func_jit = jax.jit(hh.g_funcs)
    g_vals = g_func_jit(gates, g_bar)
    assert g_vals.shape == (2, 3)

    compute_currents_jit = jax.jit(lambda V, gates, g_bar: Gating.compute_currents(V, gates, g_bar, hh.g_funcs, hh.E_rev))
    I = compute_currents_jit(V, gates, g_bar)
    assert I.shape == (2,)

    dt = 0.1
    gates_updated = jax.jit(Gating.update_gates, static_argnames=("alpha_fun", "beta_fun"))(
        gates, V, dt, hh.q10, hh.alpha_funcs, hh.beta_funcs
    )
    assert gates_updated.shape == (2, 3)
    assert jnp.all((gates_updated >= 0.0) & (gates_updated <= 1.0))

    # Test half_step_gates
    half_step_jit = jax.jit(
        Gating.half_step_gates,
        static_argnames=("alpha_fun", "beta_fun")
    )
    gates_updated = half_step_jit(
        g_prev=gates,
        alpha_fun=hh.alpha_funcs,
        beta_fun=hh.beta_funcs,
        V=V,
        dt=dt,
        q10=hh.q10
    )
    assert gates_updated.shape == (2, 3)
    assert jnp.all((gates_updated >= 0.0) & (gates_updated <= 1.0))
