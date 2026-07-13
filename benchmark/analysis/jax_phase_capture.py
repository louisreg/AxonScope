"""Benchmark-only phase capture for production JAX callables."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax


_DOUBLE_CABLE_STATIC_ARGS = frozenset(
    {
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "double_cable_block_solver",
        "tiled_thomas_block_b",
    }
)


def install_production_double_cable_capture(output_path: Path) -> None:
    """Capture the first production integrated double-cable JIT invocation."""

    import axonscope.runtime.jax.kernels.double_cable as double_cable

    original = double_cable._run_double_cable_batch_observer_integrated_scan
    captured = False

    def capture_first_call(**kwargs: Any) -> Any:
        nonlocal captured
        if captured:
            return original(**kwargs)
        captured = True

        trace_start = time.perf_counter()
        traced = original.trace(**kwargs)
        trace_s = time.perf_counter() - trace_start

        lower_start = time.perf_counter()
        lowered = traced.lower()
        lower_s = time.perf_counter() - lower_start
        stablehlo = lowered.as_text()

        compile_start = time.perf_counter()
        executable = lowered.compile()
        compile_s = time.perf_counter() - compile_start

        dynamic_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name not in _DOUBLE_CABLE_STATIC_ARGS
        }
        first_start = time.perf_counter()
        result = executable(**dynamic_kwargs)
        jax.block_until_ready(result)
        first_execution_s = time.perf_counter() - first_start

        payload = {
            "callable": (
                "axonscope.runtime.jax.kernels.double_cable_gpu."
                "_run_double_cable_batch_observer_integrated_scan"
            ),
            "trace_s": trace_s,
            "lower_s": lower_s,
            "compile_s": compile_s,
            "first_execution_s": first_execution_s,
            "total_cold_s": trace_s + lower_s + compile_s + first_execution_s,
            "stablehlo_bytes": len(stablehlo.encode("utf-8")),
            "stablehlo_lines": stablehlo.count("\n") + 1,
            "stablehlo_custom_calls": stablehlo.count("stablehlo.custom_call"),
            "static": {
                name: _json_scalar(kwargs[name])
                for name in sorted(_DOUBLE_CABLE_STATIC_ARGS)
                if name in kwargs
            },
            "dynamic": {
                name: _shape_tree(value)
                for name, value in dynamic_kwargs.items()
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    double_cable._run_double_cable_batch_observer_integrated_scan = capture_first_call


def _shape_tree(value: Any) -> Any:
    if value is None:
        return None
    leaves = jax.tree_util.tree_leaves(value)
    return [
        {
            "dtype": str(getattr(leaf, "dtype", type(leaf).__name__)),
            "shape": list(getattr(leaf, "shape", ())),
        }
        for leaf in leaves
    ]


def _json_scalar(value: Any) -> Any:
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    static_signature = getattr(value, "static_signature", None)
    if callable(static_signature):
        return repr(static_signature())
    return f"{type(value).__module__}.{type(value).__qualname__}"
