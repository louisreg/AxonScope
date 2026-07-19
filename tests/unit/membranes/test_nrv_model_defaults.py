from __future__ import annotations

import axonscope as axs
import jax.numpy as jnp
import numpy as np
from axonscope.runtime.jax.membranes.compile import compile_membrane_model


def _compiled_parameters(axon) -> dict[str, float]:
    membrane = axon.layout.sections[0].membrane
    return dict(compile_membrane_model(membrane).parameter_values)


def test_hh_axon_defaults_match_nrv_template():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=5,
    )

    params = _compiled_parameters(axon)

    assert axon.temperature == 32.0
    assert params["celsius"] == 32.0
    assert params["Rm"] == 1000.0
    assert params["EL"] == -70.0


def test_rattay_axon_defaults_match_nrv_template():
    axon = axs.axons.RattayAberham(
        length=100.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=5,
    )

    params = _compiled_parameters(axon)

    assert params["ena"] == 45.0
    assert params["ek"] == -82.0
    assert params["Rm"] == 1000.0
    assert params["EL"] == -70.0


def test_sundt_axon_defaults_match_nrv_template():
    axon = axs.axons.Sundt(
        length=100.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=5,
    )

    params = _compiled_parameters(axon)

    assert params["ena"] == 50.0
    assert params["ek"] == -90.0
    assert params["Rm"] == 10000.0
    assert params["El"] == -60.0


def test_axnode_currents_match_nrv_axnode_mod_equations():
    membrane = compile_membrane_model(axs.membranes.AxNode())
    params = membrane.parameter_values
    gate_values = {"mp": 0.21, "m": 0.37, "h": 0.62, "s": 0.43}
    gates = jnp.asarray(
        [[gate_values[name.rsplit(".", 1)[-1]] for name in membrane.gate_names()]],
        dtype=jnp.float32,
    )
    voltage = -33.0

    actual = np.asarray(
        membrane.ionic_current_trace_matrix(
            jnp.asarray([voltage], dtype=jnp.float32),
            gates,
        )
    )[0]
    expected_by_name = {
        "I_nap": params["gnapbar"] * gate_values["mp"] ** 3 * (voltage - params["ena"]),
        "I_na": (
            params["gnabar"]
            * gate_values["m"] ** 3
            * gate_values["h"]
            * (voltage - params["ena"])
        ),
        "I_k": params["gkbar"] * gate_values["s"] * (voltage - params["ek"]),
        "I_l": params["gl"] * (voltage - params["el"]),
    }
    expected = np.asarray(
        [expected_by_name[name.rsplit(".", 1)[-1]] for name in membrane.current_names()],
        dtype=np.float32,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
