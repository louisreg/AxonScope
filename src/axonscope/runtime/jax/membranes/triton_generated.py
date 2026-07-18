"""JAX custom-call adapter for generated stateless Triton membrane kernels."""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
import sys
from typing import Any

import jax
import jax.numpy as jnp

from axonscope.benchmarking import record_benchmark_metadata
from axonscope.runtime.jax.kernels.triton_call_cache import cached_triton_call

from .generated_contract import load_generated_membrane_contract


_TRITON_MODULES: dict[str, Any] = {}


def load_generated_triton_module(program: Any) -> Any | None:
    """Load one generated Triton target lazily from a membrane program cache."""

    target_path = getattr(program, "generated_target_path", None)
    if not callable(target_path):
        return None
    path = target_path("triton")
    if path is None:
        return None
    cache_key = str(program.codegen_cache.get("key", ""))
    cached = _TRITON_MODULES.get(cache_key)
    if cached is not None:
        return cached
    module_name = f"axonscope_model_codegen_{cache_key}_triton"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load generated Triton module {path}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    if getattr(module, "CACHE_KEY", None) != cache_key:
        raise ImportError("Generated Triton module has an unexpected cache key.")
    if getattr(module, "TARGET", None) != "triton":
        raise ImportError("Generated Triton module has an unexpected target.")
    _TRITON_MODULES[cache_key] = module
    return module


def advance_generated_membrane_terms(
    module: Any,
    Vm_mV: Any,
    gates: Any,
    dt_ms: Any,
    *,
    parameter_values: Mapping[str, Any],
    linearize_previous: bool,
    block_size: int = 256,
) -> tuple[Any, Any, Any]:
    """Run one generated gate-update plus Gm/GE kernel over flat compartments."""

    Vm = jnp.asarray(Vm_mV)
    gate_values = jnp.asarray(gates)
    if Vm.dtype != jnp.float32 or gate_values.dtype != jnp.float32:
        raise TypeError("Generated Triton membrane kernels require float32 inputs.")
    if gate_values.shape[:-1] != Vm.shape:
        raise ValueError(
            "Generated Triton membrane gates must have shape "
            f"{Vm.shape} + (Ngates,), got {gate_values.shape}."
        )
    if int(block_size) < 1:
        raise ValueError("block_size must be >= 1.")

    contract = load_generated_membrane_contract(module)
    kernel_spec = contract.function("advance_gates_and_membrane_terms_kernel")
    if kernel_spec.args[:3] != ("Vm", "gates", "dt"):
        raise ValueError("Generated Triton membrane kernel has an invalid signature.")
    parameter_names = kernel_spec.args[3:]
    missing = tuple(name for name in parameter_names if name not in parameter_values)
    if missing:
        raise ValueError(f"Generated Triton membrane parameters are missing: {missing!r}.")
    record_benchmark_metadata(
        generated_triton_membrane=True,
        generated_triton_membrane_cache_key=str(module.CACHE_KEY),
        generated_triton_membrane_gate_count=int(gate_values.shape[-1]),
        generated_triton_membrane_linearize_previous=bool(linearize_previous),
    )

    dt = jnp.asarray(dt_ms, dtype=Vm.dtype)
    parameters = tuple(
        jnp.asarray(parameter_values[name], dtype=Vm.dtype)
        for name in parameter_names
    )
    vm_shape = jax.ShapeDtypeStruct(Vm.shape, Vm.dtype)
    gate_shape = jax.ShapeDtypeStruct(gate_values.shape, gate_values.dtype)
    total = int(Vm.size)
    grid = ((total + int(block_size) - 1) // int(block_size),)
    gates_out, gm, ge = cached_triton_call(
        Vm,
        gate_values,
        dt,
        *parameters,
        kernel=module.advance_gates_and_membrane_terms_kernel,
        source_hash=f"{module.CACHE_KEY}:{module.SOURCE_HASH}",
        out_shape=(gate_shape, vm_shape, vm_shape),
        grid=grid,
        name=f"axonscope_generated_membrane_{module.CACHE_KEY[:12]}",
        TOTAL=total,
        LINEARIZE_PREVIOUS=bool(linearize_previous),
        BLOCK_SIZE=int(block_size),
        num_warps=4,
        num_stages=1,
        input_output_aliases={1: 0},
        vmap_flatten_elements=True,
    )
    return gates_out, gm, ge
