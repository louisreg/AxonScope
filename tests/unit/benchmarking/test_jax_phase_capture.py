from __future__ import annotations

import json
from functools import partial

import jax
import jax.numpy as jnp

import axonscope.runtime.jax.kernels.double_cable as double_cable
import axonscope.runtime.jax.kernels.single_cable as single_cable
from benchmark.analysis.jax_phase_capture import (
    install_production_jax_captures,
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
    install_production_jax_captures(
        tmp_path,
        cables=("double",),
        platform="gpu",
    )

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
    payload = json.loads(
        (tmp_path / "double.jit_phases.json").read_text(encoding="utf-8")
    )
    assert payload["trace_s"] >= 0.0
    assert payload["lower_s"] >= 0.0
    assert payload["compile_s"] >= 0.0
    assert payload["first_execution_s"] >= 0.0
    assert payload["stablehlo"]["bytes"] > 0
    assert payload["stablehlo"]["custom_calls"] >= 0
    assert len(payload["stablehlo"]["sha256"]) == 64
    assert payload["optimized_hlo"]["bytes"] > 0
    assert len(payload["optimized_hlo"]["sha256"]) == 64
    assert payload["triton_kernel_cache"] is None
    assert payload["dynamic"]["Vi0_mV"][0]["shape"] == [2, 3]
    assert (tmp_path / "double.stablehlo.txt").is_file()
    assert (tmp_path / "double.compiled.optimized_hlo.txt").is_file()


def test_production_single_cable_capture_supports_recording_route(
    tmp_path,
    monkeypatch,
):
    @partial(
        jax.jit,
        static_argnames=(
            "backend",
            "membrane",
            "has_driven_extracellular",
            "stateless_vm_only",
            "record_full",
            "record_gates",
            "record_currents",
            "record_conductances",
            "record_states",
        ),
    )
    def fake_kernel(
        *,
        backend,
        membrane,
        has_driven_extracellular,
        stateless_vm_only,
        record_full,
        record_gates,
        record_currents,
        record_conductances,
        record_states,
        Vm0_mV,
    ):
        _ = (
            backend,
            membrane,
            has_driven_extracellular,
            stateless_vm_only,
            record_full,
            record_gates,
            record_currents,
            record_conductances,
            record_states,
        )
        return Vm0_mV + 1.0

    monkeypatch.setattr(
        single_cable,
        "_run_single_cable_vstim_batch_stateful_scan",
        fake_kernel,
    )
    install_production_jax_captures(
        tmp_path,
        cables=("single",),
        platform="cpu",
        route="recording",
    )

    result = single_cable._run_single_cable_vstim_batch_stateful_scan(
        backend="backend",
        membrane="membrane",
        has_driven_extracellular=False,
        stateless_vm_only=False,
        record_full=False,
        record_gates=False,
        record_currents=False,
        record_conductances=False,
        record_states=False,
        Vm0_mV=jnp.zeros((2, 3), dtype=jnp.float32),
    )

    assert jnp.array_equal(result, jnp.ones((2, 3), dtype=jnp.float32))
    payload = json.loads((tmp_path / "single.jit_phases.json").read_text())
    assert payload["route"] == "recording"
    assert payload["static"]["record_full"] is False
