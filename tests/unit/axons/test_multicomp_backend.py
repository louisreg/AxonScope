from __future__ import annotations

import numpy as np

from axonscope.axons.myelinated import MRG
from axonscope.axons.multicomp import GenericMultiCompAxon
from axonscope.axons.unmyelinated import HodgkinHuxley
from axonscope.channel_models.passive import PassiveICM
from axonscope.channel_models.hodgkin_huxley import HodgkinHuxleyICM
from axonscope.icm import HeterogeneousICMBackend, UniformICMBackend


def test_uniform_backend_api_from_axon():
    ax = HodgkinHuxley(L=400.0, d=0.5, Nx=41)
    backend = ax.build_icm_backend()
    V = np.full((ax.Nx,), -70.0, dtype=np.float32)

    gates = np.asarray(backend.init_gates(V))
    I = np.asarray(backend.currents(V, gates))
    I_model = np.asarray(ax.ion_channel.currents(V, gates))
    Gm_backend, GE_backend = backend.membrane_conductance_terms(gates)
    Gm_model, GE_model = ax.ion_channel.membrane_conductance_terms(gates)

    assert isinstance(backend, UniformICMBackend)
    assert backend.Nx == ax.Nx
    assert gates.shape[0] == ax.Nx
    assert I.shape == (ax.Nx,)
    assert np.isfinite(I).all()
    assert np.allclose(I, I_model)
    assert np.allclose(np.asarray(Gm_backend), np.asarray(Gm_model))
    assert np.allclose(np.asarray(GE_backend), np.asarray(GE_model))


def test_multicomp_backend_shapes_and_currents():
    Nx = 8
    icm_vec = [
        HodgkinHuxleyICM() if i % 2 == 0 else PassiveICM(Rm=1e4, EL=-70.0)
        for i in range(Nx)
    ]
    ax = GenericMultiCompAxon(L=400.0, Nx=Nx, icm_vec=icm_vec, Vinit=-70.0)

    backend = ax.build_icm_backend()
    V = np.full((Nx,), -70.0, dtype=np.float32)

    gates = np.asarray(backend.init_gates(V))
    alpha = np.asarray(backend.alpha(V))
    beta = np.asarray(backend.beta(V))
    g = np.asarray(backend.conductances(gates))
    I = np.asarray(backend.currents(V, gates))
    Gm = np.asarray(backend.total_conductance(gates))
    Gm_terms, GE_terms = backend.membrane_conductance_terms(gates)

    assert gates.shape == (Nx, backend.n_gates_max)
    assert alpha.shape == gates.shape
    assert beta.shape == gates.shape
    assert g.shape == (Nx, backend.n_channels_max)
    assert I.shape == (Nx,)
    assert Gm.shape == (Nx,)
    assert np.asarray(Gm_terms).shape == (Nx,)
    assert np.asarray(GE_terms).shape == (Nx,)
    assert np.isfinite(gates).all()
    assert np.isfinite(alpha).all()
    assert np.isfinite(beta).all()
    assert np.isfinite(g).all()
    assert np.isfinite(I).all()
    assert np.isfinite(Gm).all()


def test_double_cable_backend_api_from_axon():
    ax = MRG(d=10.0, nodes=7)
    backend = ax.build_icm_backend()
    V = np.full((ax.Nx,), -80.0, dtype=np.float32)

    gates = np.asarray(backend.init_gates(V))
    g = np.asarray(backend.conductances(gates))

    assert isinstance(backend, HeterogeneousICMBackend)
    assert backend.Nx == ax.Nx
    assert g.shape == (ax.Nx, backend.n_channels_max)
    assert np.isfinite(g).all()


def test_multicomp_stimulus_api():
    ax = GenericMultiCompAxon(L=300.0, Nx=11)
    ax.insert_I_Clamp(position=150.0, t_start=0.2, duration=0.4, amplitude=1.0)

    I_on = np.asarray(ax.Iinj_uAcm2(0.3))
    I_off = np.asarray(ax.Iinj_uAcm2(1.0))

    assert I_on.shape == (ax.Nx,)
    assert I_off.shape == (ax.Nx,)
    assert float(np.max(I_on)) > 0.0
    assert np.allclose(I_off, 0.0)


def test_multicomp_backend_type_from_axon():
    ax = GenericMultiCompAxon(L=300.0, Nx=11)
    backend = ax.build_icm_backend()

    assert isinstance(backend, HeterogeneousICMBackend)
    assert backend.Nx == ax.Nx
