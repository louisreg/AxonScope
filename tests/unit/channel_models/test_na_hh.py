import pytest
import jax.numpy as jnp
from axonscope.channel_models.na_hh import NaHHICM

def test_NaHHMM_basic():
    # --- Create model
    model = NaHHICM(gnabar=0.3, ena=50.0, celsius=36.0)
    
    # --- Test properties
    g_bar = model.g_bar
    E_rev = model.E_rev
    assert g_bar.shape == (1,)
    assert E_rev.shape == (1,)
    assert g_bar[0] == 0.3*1e3
    assert E_rev[0] == 50.0

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
    g_na = model.g_funcs(gates0, g_bar)
    assert g_na.shape == (3, 1)
    assert jnp.all(g_na >= 0)
