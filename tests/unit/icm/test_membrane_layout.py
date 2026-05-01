import numpy as np

from axonscope.channel_models.axnode import AxnodeICM
from axonscope.channel_models.passive import PassiveICM
from axonscope.icm import CompartmentMembraneLayout, HeterogeneousICMBackend


def test_compartment_membrane_layout_builds_heterogeneous_backend():
    layout = CompartmentMembraneLayout(
        [
            AxnodeICM(),
            PassiveICM(Rm=1e4, EL=-80.0),
            AxnodeICM(),
        ]
    )
    membrane = layout.as_membrane_model()
    backend = membrane.build_backend()

    assert isinstance(backend, HeterogeneousICMBackend)
    assert backend.Nx == 3
    assert membrane.gate_names() == ("mp", "m", "h", "s")
    assert set(membrane.conductance_names()) == {"g_nap", "g_na", "g_k", "g_l"}
    assert set(membrane.current_names()) == {"I_nap", "I_na", "I_k", "I_l"}

    V = np.full((3,), -80.0, dtype=np.float32)
    gates = backend.init_gates(V)
    currents = backend.currents(V, gates)
    gate_trace = membrane.gate_trace_matrix(gates)
    conductance_trace = membrane.conductance_trace_matrix(gates)
    current_trace = membrane.ionic_current_trace_matrix(V, gates)

    assert gates.shape[0] == 3
    assert currents.shape == (3,)
    assert gate_trace.shape == (3, 4)
    assert conductance_trace.shape == (3, 4)
    assert current_trace.shape == (3, 4)
    assert np.all(np.isfinite(np.asarray(currents)))
    assert np.allclose(np.asarray(gate_trace)[1], 0.0)
    assert float(np.asarray(conductance_trace)[1, membrane.conductance_names().index("g_l")]) > 0.0
