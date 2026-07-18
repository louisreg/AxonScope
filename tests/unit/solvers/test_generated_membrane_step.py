from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

from axonscope.runtime.jax.kernels import double_cable_step
from axonscope.runtime.jax.membranes.backend import (
    advance_stateless_membrane_terms,
)


def test_stateless_membrane_step_prefers_generated_terms():
    expected = (
        jnp.full((3, 2), 1.0, dtype=jnp.float32),
        jnp.full((3,), 2.0, dtype=jnp.float32),
        jnp.full((3,), 3.0, dtype=jnp.float32),
    )

    class Backend:
        def generated_triton_advance_membrane_terms(self, **kwargs):
            assert kwargs["linearize_previous"] is True
            assert kwargs["static_gates"] is None
            return expected

        def cn_gate_update(self, **kwargs):
            raise AssertionError("The JAX fallback must not run.")

    actual = advance_stateless_membrane_terms(
        Backend(),
        gates=jnp.zeros((3, 2), dtype=jnp.float32),
        static_gates=None,
        V_mV=jnp.zeros((3,), dtype=jnp.float32),
        dt_ms=jnp.asarray(0.001, dtype=jnp.float32),
        linearize_previous=True,
    )

    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)


def test_double_cable_gpu_solve_reuses_precomputed_membrane_terms(monkeypatch):
    batch_size = 2
    nx = 3
    Gm = jnp.arange(batch_size * nx, dtype=jnp.float32).reshape((batch_size, nx))
    GE = Gm + 10.0
    captured = {}

    def fake_solve(**kwargs):
        captured.update(kwargs)
        return kwargs["Vi"], kwargs["Ve"]

    monkeypatch.setattr(
        double_cable_step,
        "solve_double_cable_physical_system_jax_triton_loop_xb",
        fake_solve,
    )
    monkeypatch.setattr(
        double_cable_step,
        "batch_membrane_conductance_terms",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Membrane terms must not be recomputed.")
        ),
    )
    zeros = jnp.zeros((batch_size, nx), dtype=jnp.float32)

    double_cable_step.solve_double_cable_batch_step(
        Vi=zeros,
        Ve=zeros,
        gates_new=jnp.zeros((batch_size, nx, 1), dtype=jnp.float32),
        Iinj_abs=zeros,
        I_outward_abs=zeros,
        I_corr_abs=zeros,
        extracellular_drive_abs=zeros,
        backend=object(),
        row_indices=jnp.arange(batch_size),
        linear_static_xb=object(),
        batch_size=batch_size,
        nx=nx,
        double_cable_block_solver="jax_triton_loop_xb",
        tiled_thomas_block_b=128,
        membrane_terms=(Gm, GE),
    )

    np.testing.assert_array_equal(captured["Gm_density"], Gm.T)
    np.testing.assert_array_equal(captured["GE_density"], GE.T)


def test_double_cable_gpu_fuses_generated_membrane_plan_into_thomas(monkeypatch):
    batch_size = 2
    nx = 3
    gate_count = 4
    captured = {}

    def fake_solve(**kwargs):
        captured.update(kwargs)
        return kwargs["gates"], kwargs["Vi"], kwargs["Ve"]

    monkeypatch.setattr(
        double_cable_step,
        "solve_double_cable_physical_system_jax_triton_loop_xb",
        fake_solve,
    )
    monkeypatch.setattr(
        double_cable_step,
        "batch_membrane_conductance_terms",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Fused execution must not materialize Gm/GE.")
        ),
    )
    zeros = jnp.zeros((batch_size, nx), dtype=jnp.float32)
    gates = jnp.zeros((batch_size, nx, gate_count), dtype=jnp.float32)
    static_gates = jnp.zeros((batch_size, nx, 3), dtype=jnp.float32)
    plan = SimpleNamespace(gate_count=gate_count)

    gates_new, vi_new, ve_new = double_cable_step.solve_double_cable_batch_step(
        Vi=zeros,
        Ve=zeros,
        gates_new=gates,
        Iinj_abs=zeros,
        I_outward_abs=zeros,
        I_corr_abs=zeros,
        extracellular_drive_abs=zeros,
        backend=object(),
        row_indices=jnp.arange(batch_size),
        linear_static_xb=object(),
        batch_size=batch_size,
        nx=nx,
        double_cable_block_solver="jax_triton_loop_xb",
        tiled_thomas_block_b=128,
        static_gates=static_gates,
        membrane_plan=plan,
        dt_ms=jnp.asarray(0.001, dtype=jnp.float32),
        linearize_previous=True,
    )

    assert captured["membrane_plan"] is plan
    assert captured["linearize_previous"] is True
    np.testing.assert_array_equal(captured["gates"], jnp.swapaxes(gates, 0, 1))
    np.testing.assert_array_equal(
        captured["static_gates"], jnp.swapaxes(static_gates, 0, 1)
    )
    np.testing.assert_array_equal(gates_new, captured["gates"])
    np.testing.assert_array_equal(vi_new, captured["Vi"])
    np.testing.assert_array_equal(ve_new, captured["Ve"])
