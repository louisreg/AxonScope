from __future__ import annotations

import numpy as np

from axonscope.axons.unmyelinated import Tigerholm, Schild97
from axonscope.channel_models.composite_models import (
    TigerholmCompositeICM,
    Schild97CompositeICM,
)
from axonscope.solvers.CrankNicholson import CrankNicholson


def test_tigerholm_membrane_dynamics_live_on_channel_and_axon_delegates():
    ax = Tigerholm(L=300.0, d=1.0, Nx=9)
    assert isinstance(ax.ion_channel, TigerholmCompositeICM)
    assert ax.ion_channel.membrane_state_names() == ("nai", "nao", "ki", "ko")
    assert not hasattr(ax, "compute_I_Na_dyn")
    assert not hasattr(ax, "compute_I_K_dyn")
    assert not hasattr(ax, "dynamics_correction")

    V = np.full((ax.Nx,), ax.Vinit, dtype=np.float32)
    gates = np.asarray(ax.ion_channel.init_gates(V))
    nai = np.full((ax.Nx,), ax.ion_channel.nai0, dtype=np.float32)
    nao = np.full((ax.Nx,), ax.ion_channel.nao0, dtype=np.float32)
    ko = np.full((ax.Nx,), ax.ion_channel.ko0, dtype=np.float32)
    ki = np.full((ax.Nx,), ax.ion_channel.ki0, dtype=np.float32)

    I_na = np.asarray(ax.ion_channel.compute_I_Na_dyn(V, gates, nai, nao))
    I_k = np.asarray(ax.ion_channel.compute_I_K_dyn(V, gates, nai, ko, ki))
    I_corr = np.asarray(ax.ion_channel.dynamics_correction(V, gates, nai, ko, nao, ki))
    assert I_na.shape == (ax.Nx,)
    assert I_k.shape == (ax.Nx,)
    assert I_corr.shape == (ax.Nx,)


def test_schild_membrane_dynamics_live_on_channel_and_axon_delegates():
    ax = Schild97(L=300.0, d=0.8, Nx=7)
    assert isinstance(ax.ion_channel, Schild97CompositeICM)
    assert ax.ion_channel.membrane_state_names() == ("cai", "Oc", "cao", "c_kca")
    assert not hasattr(ax, "init_c_kca")
    assert not hasattr(ax, "compute_I_Ca_budget")
    assert not hasattr(ax, "dynamics_correction_ca")

    V = np.full((ax.Nx,), ax.Vinit, dtype=np.float32)
    gates = np.asarray(ax.ion_channel.init_gates(V))
    cai = np.full((ax.Nx,), ax.ion_channel.cai0, dtype=np.float32)
    cao = np.full((ax.Nx,), ax.ion_channel.cao0, dtype=np.float32)
    c_kca = np.asarray(ax.ion_channel.init_c_kca(V, cai))

    I_ca = np.asarray(ax.ion_channel.compute_I_Ca_budget(V, gates, cai, cao))
    I_kca = np.asarray(ax.ion_channel.compute_I_kca(V, c_kca))
    I_corr = np.asarray(ax.ion_channel.dynamics_correction_ca(V, gates, cai, cao))
    assert I_ca.shape == (ax.Nx,)
    assert I_kca.shape == (ax.Nx,)
    assert I_corr.shape == (ax.Nx,)


def test_schild_diagnostics_are_provided_by_membrane_not_solver():
    ax = Schild97(L=200.0, d=0.8, Nx=7)
    ax.insert_I_Clamp(position=100.0, t_start=0.2, duration=0.2, amplitude=0.3)

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
        assert arr.shape == (res.Vm.shape[0], ax.Nx)
        assert np.isfinite(arr).all()
