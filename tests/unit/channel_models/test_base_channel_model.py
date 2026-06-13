import jax
import jax.numpy as jnp
import pytest

from axonscope.channel_models import RateTableConfig, enable_rate_tables
from axonscope.channel_models.passive import PassiveICM
from axonscope.channel_models.base_channel_model import CompositeICM, IonChannelModelBase, MembraneStateSpec
from axonscope.channel_models.hodgkin_huxley import HodgkinHuxleyICM 
from axonscope.icm import Gating
from axonscope.utils.settings import dtype


def test_base_membrane_state_api_is_generic():
    model = PassiveICM()
    assert model.membrane_state_specs() == ()
    assert model.membrane_state_names() == ()
    assert model.membrane_state_dict(()) == {}

    assert MembraneStateSpec("nai").name == "nai"
    for method_name in (
        "compute_I_Na_dyn",
        "compute_I_K_dyn",
        "compute_I_Ca_budget",
        "dynamics_correction_ca",
        "has_nai_dynamics",
        "has_ko_dynamics",
    ):
        assert not hasattr(IonChannelModelBase, method_name)


def test_channel_model_static_identity_is_structural():
    passive_a = PassiveICM(Rm=1e4, EL=-70.0)
    passive_b = PassiveICM(Rm=1e4, EL=-70.0)
    passive_c = PassiveICM(Rm=1e4, EL=-65.0)
    assert passive_a == passive_b
    assert hash(passive_a) == hash(passive_b)
    assert passive_a != passive_c

    hh_a = HodgkinHuxleyICM(celsius=6.3)
    hh_b = HodgkinHuxleyICM(celsius=6.3)
    hh_hot = HodgkinHuxleyICM(celsius=20.0)
    assert hh_a == hh_b
    assert hash(hh_a) == hash(hh_b)
    assert hh_a != hh_hot

    comp_a = CompositeICM([HodgkinHuxleyICM(), PassiveICM(Rm=1e3, EL=-70.0)])
    comp_b = CompositeICM([HodgkinHuxleyICM(), PassiveICM(Rm=1e3, EL=-70.0)])
    assert comp_a == comp_b
    assert hash(comp_a) == hash(comp_b)

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

    g_vals_generic = model.conductances(gates)
    assert jnp.allclose(g_vals_generic, g_vals)
    
    # Test compute currents
    I = Gating.compute_currents(V0, gates, model.g_bar, model.g_funcs, model.E_rev)
    assert I.shape == (2,)
    # Passive current formula: I = g_leak*(V - EL)
    assert jnp.allclose(I, model.g_bar[0] * (V0 - model.E_rev[0]))
    assert jnp.allclose(model.currents(V0, gates), I)
    assert jnp.allclose(model.total_conductance(gates), jnp.full((2,), model.g_bar[0], dtype=dtype))

    Gm, GE = model.membrane_conductance_terms(gates)
    assert Gm.shape == (2,)
    assert GE.shape == (2,)
    assert jnp.allclose(Gm, model.g_bar[0])
    assert jnp.allclose(GE, model.g_bar[0] * model.E_rev[0])

# ----------------- Tests Hodgkin-Huxley -----------------
def test_hh_model():
    model = HodgkinHuxleyICM()
    V0 = jnp.array([-65.0, -60.0])
    
    # Test init gates
    gates = model.init_gates(V0)
    assert gates.shape == (2, 3)
    assert jnp.all((gates >= 0) & (gates <= 1))
    
    # Test g_bar and E_rev
    g_bar = model.g_bar
    E_rev = model.E_rev
    assert g_bar.shape == (3,)
    assert E_rev.shape == (3,)
    
    # Test g_funcs
    g_vals = model.g_funcs(gates, g_bar)
    assert g_vals.shape == (2, 3)
    
    # Test compute currents
    I = Gating.compute_currents(V0, gates, g_bar, model.g_funcs, E_rev)
    assert I.shape == (2,)
    assert jnp.allclose(model.currents(V0, gates), I)
    
    # Test rates
    g_inf, tau = Gating.rates(V0, model.q10, model.alpha_funcs, model.beta_funcs)
    assert g_inf.shape == (2, 3)
    assert tau.shape == (2, 3)
    
    # Test update gates
    dt = 0.1
    new_gates = Gating.update_gates(V=V0, dt=dt, gates = gates, q10=model.q10, alpha_fun=model.alpha_funcs, beta_fun=model.beta_funcs)
    assert new_gates.shape == (2, 3)
    assert jnp.all((new_gates >= 0) & (new_gates <= 1))

    cn_gates = model.cn_gate_update(g_prev=gates, V_mV=V0, dt=dt)
    assert cn_gates.shape == (2, 3)
    assert jnp.all(jnp.isfinite(cn_gates))

    Gm, GE = model.membrane_conductance_terms(gates)
    assert Gm.shape == (2,)
    assert GE.shape == (2,)
    assert jnp.all(Gm >= 0.0)


def test_rate_constants_api_matches_legacy_alpha_beta():
    model = HodgkinHuxleyICM()
    V0 = jnp.array([-65.0, -62.5, -60.0], dtype=dtype)

    alpha, beta = model.rate_constants(V0)
    assert jnp.allclose(alpha, model.alpha_funcs(V0))
    assert jnp.allclose(beta, model.beta_funcs(V0))

    g_inf, tau = model.gating_inf_tau(V0)
    legacy_inf, legacy_tau = Gating.rates(V0, model.q10, model.alpha_funcs, model.beta_funcs)
    assert jnp.allclose(g_inf, legacy_inf)
    assert jnp.allclose(tau, legacy_tau)


def test_rate_table_interpolates_rates_and_gate_update():
    exact = HodgkinHuxleyICM()
    tabulated = HodgkinHuxleyICM().enable_rate_table(
        v_min_mV=-80.0,
        v_max_mV=-40.0,
        step_mV=0.05,
    )
    V0 = jnp.array([-65.03, -62.51, -59.97], dtype=dtype)
    gates = exact.init_gates(V0)

    alpha_exact, beta_exact = exact.rate_constants(V0)
    alpha_lut, beta_lut = tabulated.rate_constants(V0)
    assert jnp.allclose(alpha_lut, alpha_exact, rtol=2e-4, atol=2e-5)
    assert jnp.allclose(beta_lut, beta_exact, rtol=2e-4, atol=2e-5)

    exact_update = exact.cn_gate_update(g_prev=gates, V_mV=V0, dt=0.01)
    lut_update = tabulated.cn_gate_update(g_prev=gates, V_mV=V0, dt=0.01)
    assert jnp.allclose(lut_update, exact_update, rtol=2e-5, atol=2e-6)
    assert tabulated.has_rate_table
    assert tabulated.rate_table_config == RateTableConfig(
        v_min_mV=-80.0,
        v_max_mV=-40.0,
        step_mV=0.05,
        clamp=True,
    )

    tabulated.disable_rate_table()
    assert not tabulated.has_rate_table


def test_composite_rate_constants_pack_submodels_once():
    model = CompositeICM([HodgkinHuxleyICM(), PassiveICM(Rm=1e4, EL=-70.0)])
    V0 = jnp.array([-65.0, -60.0], dtype=dtype)
    alpha, beta = model.rate_constants(V0)
    assert alpha.shape == (2, 3)
    assert beta.shape == (2, 3)

    tabulated = CompositeICM([HodgkinHuxleyICM(), PassiveICM(Rm=1e4, EL=-70.0)])
    tabulated.enable_rate_table(v_min_mV=-80.0, v_max_mV=-40.0, step_mV=0.05)
    alpha_lut, beta_lut = tabulated.rate_constants(V0)
    assert jnp.allclose(alpha_lut, alpha)
    assert jnp.allclose(beta_lut, beta)


def test_enable_rate_tables_helper_uses_aggregate_table_for_composites():
    model = CompositeICM([HodgkinHuxleyICM(), PassiveICM(Rm=1e4, EL=-70.0)])
    count = enable_rate_tables(model, v_min_mV=-80.0, v_max_mV=-40.0, step_mV=0.05)
    assert count == 1
    assert model.has_rate_table

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

    channel_currents_jit = jax.jit(passive.currents)
    assert channel_currents_jit(V, gates).shape == (2,)

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

    channel_currents_jit = jax.jit(hh.currents)
    assert channel_currents_jit(V, gates).shape == (2,)

    dt = 0.1
    gates_updated = jax.jit(Gating.update_gates, static_argnames=("alpha_fun", "beta_fun"))(
        gates, V, dt, hh.q10, hh.alpha_funcs, hh.beta_funcs
    )
    assert gates_updated.shape == (2, 3)
    assert jnp.all((gates_updated >= 0.0) & (gates_updated <= 1.0))

    # Test cn_gate_update
    half_step_jit = jax.jit(
        Gating.cn_gate_update,
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

    channel_cn_jit = jax.jit(hh.cn_gate_update)
    gates_updated = channel_cn_jit(g_prev=gates, V_mV=V, dt=dt)
    assert gates_updated.shape == (2, 3)
