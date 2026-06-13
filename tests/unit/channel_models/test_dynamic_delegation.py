from __future__ import annotations

import numpy as np

import axonscope as axs
from axonscope import AxonSimulation
from axonscope.axons.unmyelinated import Tigerholm, Schild97
from axonscope.channel_models.composite_models import (
    TigerholmCompositeICM,
    Schild97CompositeICM,
)
from axonscope.solvers.crank_nicholson import CrankNicholson
from axonscope.solvers.runtime import compile_membrane_model
from axonscope.stimulation import Stimulus


def test_tigerholm_membrane_dynamics_live_on_channel_and_axon_delegates():
    ax = Tigerholm(length=300.0 * axs.um, diameter=1.0 * axs.um, compartments=9)
    membrane = compile_membrane_model(ax.layout.sections[0].membrane)
    assert isinstance(membrane, TigerholmCompositeICM)
    assert membrane.membrane_state_names() == ("nai", "nao", "ki", "ko")
    assert not hasattr(ax, "compute_I_Na_dyn")
    assert not hasattr(ax, "compute_I_K_dyn")
    assert not hasattr(ax, "dynamics_correction")

    V = np.full((ax.n_compartments,), ax.v_init, dtype=np.float32)
    gates = np.asarray(membrane.init_gates(V))
    nai = np.full((ax.n_compartments,), membrane.nai0, dtype=np.float32)
    nao = np.full((ax.n_compartments,), membrane.nao0, dtype=np.float32)
    ko = np.full((ax.n_compartments,), membrane.ko0, dtype=np.float32)
    ki = np.full((ax.n_compartments,), membrane.ki0, dtype=np.float32)

    I_na = np.asarray(membrane.compute_I_Na_dyn(V, gates, nai, nao))
    I_k = np.asarray(membrane.compute_I_K_dyn(V, gates, nai, ko, ki))
    I_corr = np.asarray(membrane.dynamics_correction(V, gates, nai, ko, nao, ki))
    assert I_na.shape == (ax.n_compartments,)
    assert I_k.shape == (ax.n_compartments,)
    assert I_corr.shape == (ax.n_compartments,)


def test_schild_membrane_dynamics_live_on_channel_and_axon_delegates():
    ax = Schild97(length=300.0 * axs.um, diameter=0.8 * axs.um, compartments=7)
    membrane = compile_membrane_model(ax.layout.sections[0].membrane)
    assert isinstance(membrane, Schild97CompositeICM)
    assert membrane.membrane_state_names() == ("cai", "Oc", "cao", "c_kca")
    assert not hasattr(ax, "init_c_kca")
    assert not hasattr(ax, "compute_I_Ca_budget")
    assert not hasattr(ax, "dynamics_correction_ca")

    V = np.full((ax.n_compartments,), ax.v_init, dtype=np.float32)
    gates = np.asarray(membrane.init_gates(V))
    cai = np.full((ax.n_compartments,), membrane.cai0, dtype=np.float32)
    cao = np.full((ax.n_compartments,), membrane.cao0, dtype=np.float32)
    c_kca = np.asarray(membrane.init_c_kca(V, cai))

    I_ca = np.asarray(membrane.compute_I_Ca_budget(V, gates, cai, cao))
    I_kca = np.asarray(membrane.compute_I_kca(V, c_kca))
    I_corr = np.asarray(membrane.dynamics_correction_ca(V, gates, cai, cao))
    assert I_ca.shape == (ax.n_compartments,)
    assert I_kca.shape == (ax.n_compartments,)
    assert I_corr.shape == (ax.n_compartments,)


def test_schild_diagnostics_are_provided_by_membrane_not_solver():
    ax = AxonSimulation(Schild97(length=200.0 * axs.um, diameter=0.8 * axs.um, compartments=7))
    ax.add_current_clamp(position_um=100.0,
        current=Stimulus.pulse(start=0.2, duration=0.2, amplitude=0.3),
    )

    res = CrankNicholson().solve(ax, tsim=0.8, dt=0.01, record_diagnostics=True)

    assert res.diagnostics is not None
    assert set(res.diagnostics) == {
        "I_na_total_uAcm2",
        "I_k_total_uAcm2",
        "I_ca_total_uAcm2",
        "I_total_rhs_uAcm2",
    }
    for values in res.diagnostics.values():
        arr = np.asarray(values)
        assert arr.shape == (res.Vm.shape[0], ax.n_compartments)
        assert np.isfinite(arr).all()
