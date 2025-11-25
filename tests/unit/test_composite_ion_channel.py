import pytest
import jax.numpy as jnp
import numpy as np

from axonscope.channel_models.hodgkin_huxley import HodgkinHuxleyICM, HHLeakICM, HHKICM, HHNaICM
from axonscope.channel_models.base_channel_model import CompositeICM, IonChannelModelBase

# ===========================================================
# Utility to compare one mono-model vs CompositeICM([mono])
# ===========================================================
def _assert_same_icm(mono: IonChannelModelBase, comp: CompositeICM):
    V = jnp.linspace(-80.0, 40.0, 7)

    # Alpha
    a_mono = mono.alpha_funcs(V)
    a_comp = comp.alpha_funcs(V)
    assert a_mono.shape == a_comp.shape
    assert np.allclose(a_mono, a_comp)

    # Beta
    b_mono = mono.beta_funcs(V)
    b_comp = comp.beta_funcs(V)
    assert b_mono.shape == b_comp.shape
    assert np.allclose(b_mono, b_comp)

    # init gates
    g0_mono = mono.init_gates(V)
    g0_comp = comp.init_gates(V)
    assert g0_mono.shape == g0_comp.shape
    assert np.allclose(g0_mono, g0_comp)

    # g_bar
    assert mono.g_bar.shape == comp.g_bar.shape
    assert np.allclose(mono.g_bar, comp.g_bar)

    # g_funcs
    gvals_mono = mono.g_funcs(g0_mono, mono.g_bar)
    gvals_comp = comp.g_funcs(g0_comp, comp.g_bar)
    assert gvals_mono.shape == gvals_comp.shape
    assert np.allclose(gvals_mono, gvals_comp)


# ===========================================================
# Test 1 — Na-only
# ===========================================================
def test_composite_single_na():
    mono = HHNaICM()
    comp = CompositeICM([HHNaICM()])
    _assert_same_icm(mono, comp)


# ===========================================================
# Test 2 — K-only
# ===========================================================
def test_composite_single_k():
    mono = HHKICM()
    comp = CompositeICM([HHKICM()])
    _assert_same_icm(mono, comp)


# ===========================================================
# Test 3 — Leak-only
# ===========================================================
def test_composite_single_leak():
    mono = HHLeakICM()
    comp = CompositeICM([HHLeakICM()])
    _assert_same_icm(mono, comp)


# ===========================================================
# Test 4 — Full composite = Na + K + Leak
# Must match HodgkinHuxleyICM
# ===========================================================
def test_composite_vs_full_hodgkin_huxley():
    """
    A CompositeICM consisting of (Na, K, Leak) channels must be equivalent
    to the canonical HodgkinHuxleyICM implementation.

    We compare alpha, beta, initial gates, g_bar, and g_funcs.
    Because HH returns gates in the canonical m,h,n order, we must reorder.
    """

    # Mono channels
    na = HHNaICM()
    k  = HHKICM()
    l  = HHLeakICM()

    comp = CompositeICM([na, k, l])
    hh   = HodgkinHuxleyICM()   # reference model

    V = jnp.linspace(-80.0, 40.0, 7)

    # -------------------------------------------------------
    # 1) Alpha — reorder composite to match HH gate order m,h,n
    # -------------------------------------------------------
    a_comp = comp.alpha_funcs(V)
    a_hh   = hh.alpha_funcs(V)

    # comp order = [m, h] + [n] + []
    a_comp_reordered = jnp.concatenate([
        a_comp[:, 0:1],  # m
        a_comp[:, 1:2],  # h
        a_comp[:, 2:3],  # n
    ], axis=-1)

    assert np.allclose(a_comp_reordered, a_hh), "alpha mismatch"

    # -------------------------------------------------------
    # 2) Beta — same reorder
    # -------------------------------------------------------
    b_comp = comp.beta_funcs(V)
    b_hh   = hh.beta_funcs(V)

    b_comp_reordered = jnp.concatenate([
        b_comp[:, 0:1],
        b_comp[:, 1:2],
        b_comp[:, 2:3],
    ], axis=-1)

    assert np.allclose(b_comp_reordered, b_hh), "beta mismatch"

    # -------------------------------------------------------
    # 3) Initial gates — same reorder
    # -------------------------------------------------------
    g0_comp = comp.init_gates(V)
    g0_hh   = hh.init_gates(V)

    g0_comp_reordered = jnp.concatenate([
        g0_comp[:, 0:1],
        g0_comp[:, 1:2],
        g0_comp[:, 2:3],
    ], axis=-1)

    assert np.allclose(g0_comp_reordered, g0_hh), "init_gates mismatch"

    # -------------------------------------------------------
    # 4) g_bar — reorder channels: (Na,K,Leak)
    # -------------------------------------------------------
    gbar_comp = comp.g_bar
    gbar_hh   = hh.g_bar

    gbar_comp_reordered = jnp.array([
        gbar_comp[0],  # Na
        gbar_comp[1],  # K
        gbar_comp[2],  # Leak
    ])

    assert np.allclose(gbar_comp_reordered, gbar_hh), "g_bar mismatch"

    # -------------------------------------------------------
    # 5) Effective conductances g_funcs
    # Composite output ordering = [g_Na, g_K, g_Leak]
    # HH ordering is identical
    # -------------------------------------------------------
    gvals_comp = comp.g_funcs(g0_comp, comp.g_bar)
    gvals_hh   = hh.g_funcs(g0_hh, hh.g_bar)

    assert gvals_comp.shape == gvals_hh.shape
    assert np.allclose(gvals_comp, gvals_hh), "g_funcs mismatch"
