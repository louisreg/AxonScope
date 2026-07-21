from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from axonscope.runtime.jax.kernels import double_cable_step
from axonscope.runtime.jax.membranes.backend import (
    GatedLeakStackMembraneBackend,
    advance_stateless_membrane_terms,
)
from axonscope.runtime.jax.membranes import triton_generated


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


def test_stateless_membrane_step_batches_canonical_spatial_backend():
    class Backend:
        def cn_gate_update(self, *, g_prev, V_mV, dt):
            assert g_prev.shape == (3, 2)
            assert V_mV.shape == (3,)
            return g_prev + V_mV[:, None] * dt

        def membrane_conductance_terms(self, gates, state=()):
            assert gates.shape == (3, 2)
            assert state == ()
            return gates.sum(axis=-1), gates[..., 0]

    gates = jnp.arange(12, dtype=jnp.float32).reshape((2, 3, 2))
    voltage = jnp.arange(6, dtype=jnp.float32).reshape((2, 3))
    actual_gates, actual_gm, actual_ge = advance_stateless_membrane_terms(
        Backend(),
        gates=gates,
        static_gates=None,
        V_mV=voltage,
        dt_ms=jnp.asarray(0.5, dtype=jnp.float32),
        linearize_previous=False,
    )

    expected_gates = gates + voltage[..., None] * 0.5
    np.testing.assert_array_equal(actual_gates, expected_gates)
    np.testing.assert_array_equal(actual_gm, expected_gates.sum(axis=-1))
    np.testing.assert_array_equal(actual_ge, expected_gates[..., 0])


def test_generated_triton_step_declines_contract_without_optional_kernel(monkeypatch):
    contract = SimpleNamespace(has_function=lambda name: False)
    monkeypatch.setattr(
        triton_generated,
        "load_generated_membrane_contract",
        lambda module: contract,
    )

    actual = triton_generated.advance_generated_membrane_terms(
        SimpleNamespace(),
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.zeros((2, 1), dtype=jnp.float32),
        jnp.asarray(0.005, dtype=jnp.float32),
        parameter_values={},
        linearize_previous=False,
    )

    assert actual is None


def test_gated_stack_propagates_missing_optional_triton_kernel(monkeypatch):
    monkeypatch.setattr(
        triton_generated,
        "load_generated_triton_module",
        lambda model: SimpleNamespace(),
    )
    monkeypatch.setattr(
        triton_generated,
        "advance_generated_membrane_terms",
        lambda *args, **kwargs: None,
    )
    backend = GatedLeakStackMembraneBackend(
        gated_model=SimpleNamespace(parameter_values={}),
        target_nx=2,
        dtype=jnp.float32,
        gated_gate_count=1,
        gated_channel_count=1,
    )

    actual = backend.generated_triton_advance_membrane_terms(
        g_prev=jnp.zeros((2, 4), dtype=jnp.float32),
        V_mV=jnp.zeros((2,), dtype=jnp.float32),
        dt=jnp.asarray(0.005, dtype=jnp.float32),
        linearize_previous=False,
    )

    assert actual is None


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
