from __future__ import annotations

import json
from functools import partial

import jax
import jax.numpy as jnp

import axonscope.runtime.jax.kernels.double_cable as double_cable
from benchmark.analysis.jax_phase_capture import (
    install_production_double_cable_capture,
)


def test_production_double_cable_capture_splits_jax_cold_phases(tmp_path, monkeypatch):
    @partial(
        jax.jit,
        static_argnames=(
            "backend",
            "membrane",
            "has_driven_extracellular",
            "stateless_vm_only",
            "double_cable_block_solver",
            "tiled_thomas_block_b",
        ),
    )
    def fake_kernel(
        *,
        backend,
        membrane,
        has_driven_extracellular,
        stateless_vm_only,
        double_cable_block_solver,
        tiled_thomas_block_b,
        Vi0_mV,
    ):
        _ = (
            backend,
            membrane,
            has_driven_extracellular,
            stateless_vm_only,
            double_cable_block_solver,
            tiled_thomas_block_b,
        )
        return Vi0_mV + 1.0

    monkeypatch.setattr(
        double_cable,
        "_run_double_cable_batch_observer_integrated_scan",
        fake_kernel,
    )
    output = tmp_path / "phases.json"
    install_production_double_cable_capture(output)

    result = double_cable._run_double_cable_batch_observer_integrated_scan(
        backend="backend",
        membrane="membrane",
        has_driven_extracellular=True,
        stateless_vm_only=True,
        double_cable_block_solver="jax_triton_loop_xb",
        tiled_thomas_block_b=64,
        Vi0_mV=jnp.zeros((2, 3), dtype=jnp.float32),
    )

    assert jnp.array_equal(result, jnp.ones((2, 3), dtype=jnp.float32))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["trace_s"] >= 0.0
    assert payload["lower_s"] >= 0.0
    assert payload["compile_s"] >= 0.0
    assert payload["first_execution_s"] >= 0.0
    assert payload["stablehlo_bytes"] > 0
    assert payload["stablehlo_custom_calls"] >= 0
    assert payload["triton_kernel_cache"] is None
    assert payload["dynamic"]["Vi0_mV"][0]["shape"] == [2, 3]
