from __future__ import annotations

import axonscope as axs
import jax.numpy as jnp
import numpy as np

from axonscope.runtime.jax.membranes.compile import compile_membrane_model


def _vtrap(x: float, y: float) -> float:
    z = x / y
    if abs(z) < 1e-6:
        return y * (1.0 - z / 2.0)
    return x / np.expm1(z)


def _section_membrane(axon, section_name: str):
    return next(
        element.section.membrane
        for element in axon.layout.elements
        if element.section.name.lower() == section_name.lower()
    )


def test_gaines_families_delegate_to_two_shared_source_topologies():
    assert axs.membranes.GainesMotorNode.source_model is axs.membranes.GainesSensoryNode.source_model
    assert (
        axs.membranes.GainesMotorInternode.source_model
        is axs.membranes.GainesSensoryInternode.source_model
    )
    assert (
        axs.membranes.GainesMotorNode.source_model
        is not axs.membranes.GainesMotorInternode.source_model
    )


def test_gaines_node_defaults_match_nrv_mechanisms():
    motor = compile_membrane_model(axs.membranes.GainesMotorNode())
    sensory = compile_membrane_model(axs.membranes.GainesSensoryNode())

    assert tuple(name.rsplit(".", 1)[-1] for name in motor.gate_names()) == (
        "mp",
        "m",
        "h",
        "s",
        "n",
    )
    assert motor.current_names() == ("I_nap", "I_na", "I_k", "I_l", "I_kf")
    assert sensory.current_names() == motor.current_names()
    for name, expected in {
        "gnapbar": 10.0,
        "gnabar": 3000.0,
        "gkbar": 80.0,
        "gl": 7.0,
        "gkfbar": 25.68,
        "ena": 50.0,
        "ek": -90.0,
        "el": -90.0,
        "ekf": -90.0,
    }.items():
        assert motor.parameter_values[name] == expected
    for name, expected in {
        "gkbar": 41.06,
        "gl": 6.005,
        "gkfbar": 27.37,
        "amp_a": 0.00957,
        "amp_b": 26.852,
        "bmp_a": 0.0002401,
        "bmp_b": 33.8333,
        "am_a": 1.77753,
        "am_b": 20.1795,
        "bm_a": 0.0823,
        "bm_b": 25.4746,
        "ah_a": 0.075286,
        "ah_b": 112.7124,
        "ah_c": 8.391,
        "bh_a": 2.8083,
        "bh_b": 30.5435,
        "bh_c": 10.2263,
    }.items():
        np.testing.assert_allclose(sensory.parameter_values[name], expected, rtol=1e-7)


def test_gaines_node_rates_match_nrv_equations():
    voltage = -33.0
    for model in (axs.membranes.GainesMotorNode(), axs.membranes.GainesSensoryNode()):
        membrane = compile_membrane_model(model)
        p = membrane.parameter_values
        alpha, beta = membrane.exact_rate_constants(
            jnp.asarray([voltage], dtype=jnp.float32)
        )
        q_na = 2.2 ** ((p["celsius"] - 20.0) / 10.0)
        q_h = 2.9 ** ((p["celsius"] - 20.0) / 10.0)
        q_k = 3.0 ** ((p["celsius"] - 36.0) / 10.0)
        v_traub = voltage + 80.0
        expected_alpha = np.asarray(
            [
                q_na * p["amp_a"] * _vtrap(-(voltage + p["amp_b"]), p["amp_c"]),
                q_na * p["am_a"] * _vtrap(-(voltage + p["am_b"]), p["am_c"]),
                q_h * p["ah_a"] * _vtrap(voltage + p["ah_b"], p["ah_c"]),
                q_k * 0.3 / (np.exp((v_traub - 27.0) / -5.0) + 1.0),
                q_k * 0.0462 * _vtrap(-(voltage + 83.2), 1.1),
            ]
        )
        expected_beta = np.asarray(
            [
                q_na * p["bmp_a"] * _vtrap(voltage + p["bmp_b"], p["bmp_c"]),
                q_na * p["bm_a"] * _vtrap(voltage + p["bm_b"], p["bm_c"]),
                q_h * p["bh_a"] / (1.0 + np.exp(-(voltage + p["bh_b"]) / p["bh_c"])),
                q_k * 0.03 / (np.exp((v_traub + 10.0) / -1.0) + 1.0),
                q_k * 0.0824 * _vtrap(voltage + 66.0, 10.5),
            ]
        )
        np.testing.assert_allclose(np.asarray(alpha)[0], expected_alpha, rtol=2e-6)
        np.testing.assert_allclose(np.asarray(beta)[0], expected_beta, rtol=2e-6)


def test_gaines_internode_defaults_rates_and_currents_match_nrv_equations():
    voltage = -72.0
    gates = np.asarray([[0.27, 0.42, 0.31]], dtype=np.float32)
    for model, expected in (
        (
            axs.membranes.GainesMotorInternode(),
            {"gkbar": 2.581, "gl": 0.2, "gqbar": 2.232, "gkfbar": 25.68, "hcn_midpoint": -107.3},
        ),
        (
            axs.membranes.GainesSensoryInternode(),
            {"gkbar": 1.324, "gl": 0.1716, "gqbar": 3.102, "gkfbar": 27.37, "hcn_midpoint": -94.2},
        ),
    ):
        membrane = compile_membrane_model(model)
        p = membrane.parameter_values
        for name, value in expected.items():
            np.testing.assert_allclose(p[name], value, rtol=1e-7)
        alpha, beta = membrane.exact_rate_constants(jnp.asarray([voltage]))
        q_k = 3.0 ** ((p["celsius"] - 36.0) / 10.0)
        v_traub = voltage + 80.0
        hcn_x = (voltage - p["hcn_midpoint"]) / -12.2
        expected_alpha = np.asarray(
            [
                q_k * 0.3 / (np.exp((v_traub - 27.0) / -5.0) + 1.0),
                q_k * 0.00522 * np.exp(hcn_x),
                q_k * 0.0462 * _vtrap(-(voltage + 83.2), 1.1),
            ]
        )
        expected_beta = np.asarray(
            [
                q_k * 0.03 / (np.exp((v_traub + 10.0) / -1.0) + 1.0),
                q_k * 0.00522 / np.exp(hcn_x),
                q_k * 0.0824 * _vtrap(voltage + 66.0, 10.5),
            ]
        )
        np.testing.assert_allclose(np.asarray(alpha)[0], expected_alpha, rtol=2e-6)
        np.testing.assert_allclose(np.asarray(beta)[0], expected_beta, rtol=2e-6)

        actual_current = np.asarray(
            membrane.ionic_current_trace_matrix(
                jnp.asarray([voltage], dtype=jnp.float32),
                jnp.asarray(gates),
            )
        )[0]
        s, q, n = gates[0]
        expected_current = np.asarray(
            [
                p["gkbar"] * s * (voltage - p["ek"]),
                p["gl"] * (voltage - p["el"]),
                p["gqbar"] * q * (voltage - p["eq"]),
                p["gkfbar"] * n**4 * (voltage - p["ekf"]),
            ]
        )
        np.testing.assert_allclose(actual_current, expected_current, rtol=2e-6)


def test_gaines_axons_reuse_mrg_geometry_with_family_section_membranes():
    motor = axs.axons.GainesMotor(diameter=10.0 * axs.um, nodes=3)
    sensory = axs.axons.GainesSensory(diameter=10.0 * axs.um, nodes=3)

    assert motor.v_init == -85.9411
    assert sensory.v_init == -79.3565
    assert motor.formulation is axs.axons.CableFormulation.DOUBLE_CABLE
    assert sensory.formulation is axs.axons.CableFormulation.DOUBLE_CABLE
    np.testing.assert_allclose(
        motor.layout.position_values(unit="micrometer"),
        axs.axons.MRG(diameter=10.0 * axs.um, nodes=3).layout.position_values(
            unit="micrometer"
        ),
    )

    expected = {
        "GainesMotor": {
            "node": ("gaines_motor_node", 7.0, 25.68),
            "mysa": ("gaines_motor_internode", 2.0, 150.74),
            "flut": ("gaines_motor_internode", 0.2, 25.68),
            "stin": ("gaines_motor_internode", 0.2, 25.68),
        },
        "GainesSensory": {
            "node": ("gaines_sensory_node", 6.005, 27.37),
            "mysa": ("gaines_sensory_internode", 1.716, 164.2),
            "flut": ("gaines_sensory_internode", 0.1716, 27.37),
            "stin": ("gaines_sensory_internode", 0.1716, 27.37),
        },
    }
    for axon in (motor, sensory):
        for section_name, (kind, gl, gkfbar) in expected[type(axon).__name__].items():
            membrane = _section_membrane(axon, section_name)
            compiled = compile_membrane_model(membrane)
            assert membrane.kind == kind
            np.testing.assert_allclose(compiled.parameter_values["gl"], gl)
            np.testing.assert_allclose(compiled.parameter_values["gkfbar"], gkfbar)
