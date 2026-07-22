from __future__ import annotations

import axonscope as axs
import jax.numpy as jnp
import neuron
import nrv
import numpy as np
import pytest

from axonscope.runtime.jax.membranes.compile import compile_membrane_model


@pytest.mark.parametrize(
    ("nrv_model", "axon_class", "node_class", "internode_class", "suffix"),
    (
        (
            "Gaines_motor",
            axs.axons.GainesMotor,
            axs.membranes.GainesMotorNode,
            axs.membranes.GainesMotorInternode,
            "motor",
        ),
        (
            "Gaines_sensory",
            axs.axons.GainesSensory,
            axs.membranes.GainesSensoryNode,
            axs.membranes.GainesSensoryInternode,
            "sensory",
        ),
    ),
)
def test_gaines_rates_currents_and_section_defaults_match_nrv(
    nrv_model,
    axon_class,
    node_class,
    internode_class,
    suffix,
):
    axon_nrv = nrv.myelinated(
        0,
        0,
        10.0,
        3000.0,
        model=nrv_model,
        dt=0.005,
        node_shift=0,
        Nseg_per_sec=1,
        rec="all",
        T=37.0,
    )
    axon_as = axon_class(diameter=10.0 * axs.um, nodes=3)
    neuron.h.celsius = 37.0

    for section_name, voltage, gate_values in (
        (
            "node",
            -33.0,
            {"mp": 0.21, "m": 0.37, "h": 0.62, "s": 0.43, "n": 0.31},
        ),
        ("MYSA", -72.0, {"s": 0.27, "q": 0.42, "n": 0.31}),
        ("FLUT", -72.0, {"s": 0.27, "q": 0.42, "n": 0.31}),
        ("STIN", -72.0, {"s": 0.27, "q": 0.42, "n": 0.31}),
    ):
        section_key = section_name.lower()
        source_model = node_class if section_key == "node" else internode_class
        membrane_as = next(
            element.section.membrane
            for element in axon_as.layout.elements
            if element.section.name.lower() == section_key
        )
        assert membrane_as.kind == source_model.kind_name()
        compiled = compile_membrane_model(membrane_as)

        mechanism = f"{section_key}_{suffix}"
        segment = getattr(axon_nrv, section_name)[0](0.5)
        neuron.h.finitialize(voltage)
        gate_names = tuple(name.rsplit(".", 1)[-1] for name in compiled.gate_names())
        alpha_nrv = []
        beta_nrv = []
        for gate in gate_names:
            inf = float(getattr(segment, f"{gate}_inf_{mechanism}"))
            tau = float(getattr(segment, f"tau_{gate}_{mechanism}"))
            alpha_nrv.append(inf / tau)
            beta_nrv.append((1.0 - inf) / tau)
            setattr(segment, f"{gate}_{mechanism}", gate_values[gate])
        neuron.h.fcurrent()

        alpha_as, beta_as = compiled.exact_rate_constants(
            jnp.asarray([voltage], dtype=jnp.float32)
        )
        np.testing.assert_allclose(np.asarray(alpha_as)[0], alpha_nrv, rtol=1e-5, atol=2e-6)
        np.testing.assert_allclose(np.asarray(beta_as)[0], beta_nrv, rtol=1e-5, atol=2e-6)

        current_names = (
            ("inap", "ina", "ik", "il", "ikf")
            if section_key == "node"
            else ("ik", "il", "iq", "ikf")
        )
        current_nrv = np.asarray(
            [float(getattr(segment, f"{name}_{mechanism}")) * 1e3 for name in current_names]
        )
        gates_as = jnp.asarray([[gate_values[name] for name in gate_names]])
        current_as = np.asarray(
            compiled.ionic_current_trace_matrix(jnp.asarray([voltage]), gates_as)
        )[0]
        np.testing.assert_allclose(current_as, current_nrv, rtol=2e-6, atol=3e-6)

        for parameter in ("gkbar", "gl", "gkfbar", "ek", "el", "ekf"):
            nrv_value = float(getattr(segment, f"{parameter}_{mechanism}"))
            scale = 1e3 if parameter.startswith("g") else 1.0
            np.testing.assert_allclose(
                compiled.parameter_values[parameter],
                nrv_value * scale,
                rtol=1e-7,
            )
        if section_key != "node":
            for parameter in ("gqbar", "eq"):
                nrv_value = float(getattr(segment, f"{parameter}_{mechanism}"))
                scale = 1e3 if parameter.startswith("g") else 1.0
                np.testing.assert_allclose(
                    compiled.parameter_values[parameter],
                    nrv_value * scale,
                    rtol=1e-7,
                )
