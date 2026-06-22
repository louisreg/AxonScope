from __future__ import annotations

import numpy as np

import axonscope as axs
from axonscope import AxonInstance
from axonscope.axons.myelinated import MRG
from axonscope.axons import Axon, Layout, LayoutElement, Section
from axonscope.axons.unmyelinated import HodgkinHuxley
from axonscope.channel_models.passive import PassiveICM
from axonscope.channel_models.hodgkin_huxley import HodgkinHuxleyICM
from axonscope.icm import HeterogeneousICMBackend, UniformICMBackend
from axonscope.backends.jax.runtime import build_icm_backend_from_axon, compile_membrane_model
from axonscope.stimulation.runtime import build_intracellular_current_density_fn
from axonscope.stimulation import Stimulus
from axonscope.utils import units


def _heterogeneous_single_cable_axon(*, L: float, Nx: int, membranes=None, v_init=None):
    if v_init is None:
        v_init = -70.0 * axs.mV
    if membranes is None:
        membranes = [PassiveICM(Rm=1e4, EL=-70.0) for _ in range(Nx)]
    dx = L / Nx
    return Axon(
        layout=Layout(
            [
                LayoutElement(
                    Section(
                        f"s{i}",
                        membrane=membranes[i],
                        diameter=units.Q_(1.0, "micrometer"),
                    ),
                    length=units.Q_(dx, "micrometer"),
                )
                for i in range(Nx)
            ]
        ),
        formulation=axs.axons.CableFormulation.SINGLE_CABLE,
        v_init=v_init,
    )


def test_uniform_backend_api_from_axon():
    ax = HodgkinHuxley(length=400.0 * axs.um, diameter=0.5 * axs.um, compartments=41)
    backend = build_icm_backend_from_axon(ax)
    V = np.full((ax.n_compartments,), -70.0, dtype=np.float32)

    gates = np.asarray(backend.init_gates(V))
    I = np.asarray(backend.currents(V, gates))
    membrane = compile_membrane_model(ax.layout.sections[0].membrane)
    I_model = np.asarray(membrane.currents(V, gates))
    Gm_backend, GE_backend = backend.membrane_conductance_terms(gates)
    Gm_model, GE_model = membrane.membrane_conductance_terms(gates)

    assert isinstance(backend, UniformICMBackend)
    assert backend.Nx == ax.n_compartments
    assert gates.shape[0] == ax.n_compartments
    assert I.shape == (ax.n_compartments,)
    assert np.isfinite(I).all()
    assert np.allclose(I, I_model)
    assert np.allclose(np.asarray(Gm_backend), np.asarray(Gm_model))
    assert np.allclose(np.asarray(GE_backend), np.asarray(GE_model))


def test_heterogeneous_single_cable_backend_shapes_and_currents():
    Nx = 8
    membranes = [
        HodgkinHuxleyICM() if i % 2 == 0 else PassiveICM(Rm=1e4, EL=-70.0)
        for i in range(Nx)
    ]
    ax = _heterogeneous_single_cable_axon(
        L=400.0,
        Nx=Nx,
        membranes=membranes,
        v_init=-70.0 * axs.mV,
    )

    backend = build_icm_backend_from_axon(ax)
    V = np.full((Nx,), -70.0, dtype=np.float32)

    gates = np.asarray(backend.init_gates(V))
    alpha = np.asarray(backend.alpha(V))
    beta = np.asarray(backend.beta(V))
    g = np.asarray(backend.conductances(gates))
    I = np.asarray(backend.currents(V, gates))
    Gm = np.asarray(backend.total_conductance(gates))
    Gm_terms, GE_terms = backend.membrane_conductance_terms(gates)

    assert isinstance(ax, Axon)
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
    ax = MRG(diameter=10.0 * axs.um, nodes=7)
    backend = build_icm_backend_from_axon(ax)
    V = np.full((ax.n_compartments,), -80.0, dtype=np.float32)

    gates = np.asarray(backend.init_gates(V))
    g = np.asarray(backend.conductances(gates))

    assert isinstance(backend, HeterogeneousICMBackend)
    assert backend.Nx == ax.n_compartments
    assert g.shape == (ax.n_compartments, backend.n_channels_max)
    assert np.isfinite(g).all()


def test_double_cable_backend_static_identity_is_structural():
    ax_a = MRG(diameter=10.0 * axs.um, nodes=5)
    ax_b = MRG(diameter=10.0 * axs.um, nodes=5)
    backend_a = build_icm_backend_from_axon(ax_a)
    backend_b = build_icm_backend_from_axon(ax_b)

    membranes_a = axs.axons.flatten_layout(ax_a.layout).membrane_models
    membranes_b = axs.axons.flatten_layout(ax_b.layout).membrane_models
    assert membranes_a == membranes_b
    assert hash(membranes_a[0]) == hash(membranes_b[0])
    assert backend_a == backend_b
    assert hash(backend_a) == hash(backend_b)
    assert len(backend_a.groups) < backend_a.Nx
    assert sum(len(group.indices) for group in backend_a.groups) == backend_a.Nx


def test_heterogeneous_single_cable_stimulus_api():
    ax = AxonInstance(_heterogeneous_single_cable_axon(L=300.0, Nx=11))
    ax.add_current_clamp(position=150.0 * axs.um,
        current=Stimulus.pulse(start=0.2 * axs.ms, duration=0.4 * axs.ms, amplitude=1.0),
    )

    current_density = build_intracellular_current_density_fn(ax)
    I_on = np.asarray(current_density(0.3))
    I_off = np.asarray(current_density(1.0))

    assert I_on.shape == (ax.n_compartments,)
    assert I_off.shape == (ax.n_compartments,)
    assert float(np.max(I_on)) > 0.0
    assert np.allclose(I_off, 0.0)


def test_heterogeneous_single_cable_backend_type_from_axon():
    membranes = [
        HodgkinHuxleyICM() if i % 2 == 0 else PassiveICM(Rm=1e4, EL=-70.0)
        for i in range(11)
    ]
    ax = _heterogeneous_single_cable_axon(L=300.0, Nx=11, membranes=membranes)
    backend = build_icm_backend_from_axon(ax)

    assert isinstance(backend, HeterogeneousICMBackend)
    assert backend.Nx == ax.n_compartments
