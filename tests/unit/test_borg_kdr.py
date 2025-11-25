import pytest
import jax.numpy as jnp
from axonscope.channel_models.borg_kdr import BorgKDRICM

def test_BorgKDRMM_basic():
    # --- Create model
    model = BorgKDRICM(gkdrbar=0.3, ek=-77.0, celsius=36.0)
    
    # --- Test properties
    g_bar = model.g_bar
    E_rev = model.E_rev
    assert g_bar.shape == (1,)
    assert E_rev.shape == (1,)
    assert g_bar[0] == 0.3
    assert E_rev[0] == -77.0

    # --- Test alpha/beta shapes
    V_test = jnp.array([-65.0, 0.0, 50.0])
    alpha = model.alpha_funcs(V_test)
    beta = model.beta_funcs(V_test)
    assert alpha.shape == (3, 2)
    assert beta.shape == (3, 2)
    assert jnp.all(alpha >= 0)
    assert jnp.all(beta >= 0)

    # --- Test gate initialization
    gates0 = model.init_gates(V_test)
    assert gates0.shape == (3, 2)
    assert jnp.all((gates0 >= 0) & (gates0 <= 1))

    # --- Test conductance calculation
    g_kdr = model.g_funcs(gates0, g_bar)
    assert g_kdr.shape == (3, 1)
    assert jnp.all(g_kdr >= 0)
