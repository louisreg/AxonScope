from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import axonscope as axs
from axonscope.runtime.jax.membranes.compile import compile_membrane_model
from benchmark.kinetic_transition_tables import (
    TableSpec,
    apply_transition_table,
    build_transition_table,
)


@pytest.mark.parametrize("operator", ("implicit", "exponential"))
def test_generated_tables_are_stochastic_and_fully_keyed(operator: str) -> None:
    membrane = compile_membrane_model(axs.membranes.Nav16())
    spec = TableSpec(-120.0, 80.0, 1.0, 0.005, "float32", operator)  # type: ignore[arg-type]

    table, stationary, manifest = build_transition_table(membrane, spec)

    np.testing.assert_allclose(table.sum(axis=1), 1.0, atol=2e-6)
    np.testing.assert_allclose(stationary.sum(axis=1), 1.0, atol=2e-6)
    assert np.min(table) >= -2e-7
    assert len(manifest["cache_key"]) == 64
    assert manifest["parameterized_hash"]
    assert manifest["table"] == {
        "v_min_mV": -120.0,
        "v_max_mV": 80.0,
        "dv_mV": 1.0,
        "dt_ms": 0.005,
        "dtype": "float32",
        "operator": operator,
    }


def test_linear_lookup_is_exact_at_grid_points_for_implicit_operator() -> None:
    membrane = compile_membrane_model(axs.membranes.Nav16())
    spec = TableSpec(-120.0, 80.0, 1.0, 0.005, "float32", "implicit")
    table, _, _ = build_transition_table(membrane, spec)
    voltage = jnp.asarray([-120.0, -70.0, 0.0, 80.0], dtype=jnp.float32)
    previous = jnp.asarray(
        np.random.default_rng(18).dirichlet(np.ones(6), size=len(voltage)),
        dtype=jnp.float32,
    )

    actual = apply_transition_table(
        previous,
        voltage,
        jnp.asarray(table),
        v_min_mV=spec.v_min_mV,
        dv_mV=spec.dv_mV,
        interpolation="linear",
    )
    expected = membrane.cn_gate_update(previous, voltage, spec.dt_ms)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
