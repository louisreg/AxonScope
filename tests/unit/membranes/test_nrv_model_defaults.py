from __future__ import annotations

import axonscope as axs
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
