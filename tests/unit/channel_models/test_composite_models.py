import pytest
import jax.numpy as jnp
import numpy as np

from axonscope import AxonInstance
from axonscope import membranes
from axonscope.channel_models.hodgkin_huxley import HodgkinHuxleyICM
from tests.unit.channel_models.fixtures import HHLeakICM, HHKICM, HHNaICM
from axonscope.channel_models.base_channel_model import CompositeICM, IonChannelModelBase
from axonscope.solvers.crank_nicholson import CrankNicholson
from axonscope.solvers.runtime import compile_membrane_model
from axonscope.axons import Axon, Layout, Section
from axonscope.stimulation import Stimulus
from axonscope.utils import units


def _assert_same_icm(mono: IonChannelModelBase, comp: CompositeICM):
    V = jnp.linspace(-80.0, 40.0, 7)

    a_mono = mono.alpha_funcs(V)
    a_comp = comp.alpha_funcs(V)
    assert a_mono.shape == a_comp.shape
    assert np.allclose(a_mono, a_comp)

    b_mono = mono.beta_funcs(V)
    b_comp = comp.beta_funcs(V)
    assert b_mono.shape == b_comp.shape
    assert np.allclose(b_mono, b_comp)

    g0_mono = mono.init_gates(V)
    g0_comp = comp.init_gates(V)
    assert g0_mono.shape == g0_comp.shape
    assert np.allclose(g0_mono, g0_comp)

    assert mono.g_bar.shape == comp.g_bar.shape
    assert np.allclose(mono.g_bar, comp.g_bar)

    gvals_mono = mono.g_funcs(g0_mono, mono.g_bar)
    gvals_comp = comp.g_funcs(g0_comp, comp.g_bar)
    assert gvals_mono.shape == gvals_comp.shape
    assert np.allclose(gvals_mono, gvals_comp)


def test_composite_single_na():
    mono = HHNaICM()
    comp = CompositeICM([HHNaICM()])
    _assert_same_icm(mono, comp)


def test_composite_single_k():
    mono = HHKICM()
    comp = CompositeICM([HHKICM()])
    _assert_same_icm(mono, comp)


def test_composite_single_leak():
    mono = HHLeakICM()
    comp = CompositeICM([HHLeakICM()])
    _assert_same_icm(mono, comp)


def test_composite_vs_full_hodgkin_huxley():
    na = HHNaICM()
    k  = HHKICM()
    l  = HHLeakICM()

    comp = CompositeICM([na, k, l])
    hh   = HodgkinHuxleyICM()

    V = jnp.linspace(-80.0, 40.0, 7)

    a_comp = comp.alpha_funcs(V)
    a_hh   = hh.alpha_funcs(V)
    a_comp_reordered = jnp.concatenate([a_comp[:, 0:1], a_comp[:, 1:2], a_comp[:, 2:3]], axis=-1)
    assert np.allclose(a_comp_reordered, a_hh), "alpha mismatch"

    b_comp = comp.beta_funcs(V)
    b_hh   = hh.beta_funcs(V)
    b_comp_reordered = jnp.concatenate([b_comp[:, 0:1], b_comp[:, 1:2], b_comp[:, 2:3]], axis=-1)
    assert np.allclose(b_comp_reordered, b_hh), "beta mismatch"

    g0_comp = comp.init_gates(V)
    g0_hh   = hh.init_gates(V)
    g0_comp_reordered = jnp.concatenate([g0_comp[:, 0:1], g0_comp[:, 1:2], g0_comp[:, 2:3]], axis=-1)
    assert np.allclose(g0_comp_reordered, g0_hh), "init_gates mismatch"

    gbar_comp = comp.g_bar
    gbar_hh   = hh.g_bar
    gbar_comp_reordered = jnp.array([gbar_comp[0], gbar_comp[1], gbar_comp[2]])
    assert np.allclose(gbar_comp_reordered, gbar_hh), "g_bar mismatch"

    gvals_comp = comp.g_funcs(g0_comp, comp.g_bar)
    gvals_hh   = hh.g_funcs(g0_hh, hh.g_bar)
    assert gvals_comp.shape == gvals_hh.shape
    assert np.allclose(gvals_comp, gvals_hh), "g_funcs mismatch"


def test_composite_keeps_common_q10():
    na = HHNaICM(celsius=6.3)
    k = HHKICM(celsius=6.3)
    l = HHLeakICM()

    comp = CompositeICM([na, k, l])

    assert np.isclose(float(comp.q10), float(na.q10))
    assert np.isclose(float(comp.q10), float(k.q10))


def test_composite_rejects_stateful_membrane_components():
    membrane = membranes.Composite(
        [
            membranes.Schild97(diameter=0.8 * units.ureg.um),
            membranes.Passive(Rm=1e4, EL=-70.0),
        ]
    )

    with pytest.raises(NotImplementedError, match="stateful membrane components"):
        compile_membrane_model(membrane)


def test_axon_composite_vs_mono_hodgkin_huxley():
    """End-to-end: CompositeICM(Na,K,Leak) vs HodgkinHuxleyICM — Vm must match within 0.1 mV."""
    mono_icm = HodgkinHuxleyICM()
    comp_icm = CompositeICM([HHNaICM(), HHKICM(), HHLeakICM()])

    L, d, Nx = 1_000, 0.5, 11
    ax_mono = Axon(
        layout=Layout.single_uniform(
            Section(
                "axon",
                membrane=mono_icm,
                diameter=units.Q_(d, "micrometer"),
            ),
            length=units.Q_(L, "micrometer"),
            compartments=Nx,
        ),
        v_init=-70.0 * units.ureg.mV,
        temperature=6.3 * units.ureg.degC,
    )
    ax_comp = Axon(
        layout=Layout.single_uniform(
            Section(
                "axon",
                membrane=comp_icm,
                diameter=units.Q_(d, "micrometer"),
            ),
            length=units.Q_(L, "micrometer"),
            compartments=Nx,
        ),
        v_init=-70.0 * units.ureg.mV,
        temperature=6.3 * units.ureg.degC,
    )

    solver = CrankNicholson()
    stim = Stimulus.pulse(
        start=units.Q_(1.0, "millisecond"),
        duration=units.Q_(1.0, "millisecond"),
        amplitude=5,
    )
    sim_mono = AxonInstance(ax_mono)
    sim_comp = AxonInstance(ax_comp)
    sim_mono.add_current_clamp(position=units.Q_(L / 2, "micrometer"), current=stim)
    sim_comp.add_current_clamp(position=units.Q_(L / 2, "micrometer"), current=stim)

    res_mono = solver.solve(sim_mono, 10, 0.001)
    res_comp = solver.solve(sim_comp, 10, 0.001)

    Vm_mono = np.array(res_mono.Vm)
    Vm_comp = np.array(res_comp.Vm)
    assert Vm_mono.shape == Vm_comp.shape

    max_err = np.abs(Vm_mono - Vm_comp).max()
    assert max_err <= 0.001, f"Vm differ by {max_err:.4f} mV"
