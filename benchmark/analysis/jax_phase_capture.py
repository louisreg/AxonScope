"""Benchmark-only phase and compiler-IR capture for production JAX callables."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import jax


_COMMON_STATIC_ARGS = frozenset(
    {
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "observer_retention",
        "raster_temporal_stride",
    }
)
_SINGLE_CABLE_STATIC_ARGS = _COMMON_STATIC_ARGS
_DOUBLE_CABLE_STATIC_ARGS = _COMMON_STATIC_ARGS | frozenset(
    {
        "double_cable_block_solver",
        "tiled_thomas_block_b",
    }
)


def install_production_jax_captures(
    output_dir: Path,
    *,
    cables: tuple[str, ...],
) -> None:
    """Capture the first compact factorized JIT invocation for each cable."""

    requested = frozenset(cables)
    unknown = requested - {"single", "double"}
    if unknown:
        raise ValueError(f"unsupported JAX phase-capture cables: {sorted(unknown)}")

    if "single" in requested:
        import axonscope.runtime.jax.kernels.single_cable as single_cable

        _install_capture(
            single_cable,
            attribute="_run_single_cable_factorized_vstim_batch_sparse_observer_scan",
            static_args=_SINGLE_CABLE_STATIC_ARGS,
            label="single",
            output_dir=output_dir,
        )

    if "double" in requested:
        import axonscope.runtime.jax.kernels.double_cable as double_cable

        _install_capture(
            double_cable,
            attribute="_run_double_cable_batch_observer_integrated_scan",
            static_args=_DOUBLE_CABLE_STATIC_ARGS,
            label="double",
            output_dir=output_dir,
        )


def _install_capture(
    module: ModuleType,
    *,
    attribute: str,
    static_args: frozenset[str],
    label: str,
    output_dir: Path,
) -> None:
    original = getattr(module, attribute)
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

        triton_kernel_cache = None
        if label == "double":
            from axonscope.runtime.jax.kernels.triton_call_cache import (
                last_triton_kernel_cache_event,
            )

            triton_kernel_cache = last_triton_kernel_cache_event()

        compile_start = time.perf_counter()
        executable = lowered.compile()
        compile_s = time.perf_counter() - compile_start
        optimized_hlo = executable.as_text()

        dynamic_kwargs = {
            name: value for name, value in kwargs.items() if name not in static_args
        }
        first_start = time.perf_counter()
        result = executable(**dynamic_kwargs)
        jax.block_until_ready(result)
        first_execution_s = time.perf_counter() - first_start

        output_dir.mkdir(parents=True, exist_ok=True)
        stablehlo_path = output_dir / f"{label}.stablehlo.txt"
        optimized_hlo_path = output_dir / f"{label}.compiled.optimized_hlo.txt"
        stablehlo_path.write_text(stablehlo, encoding="utf-8")
        optimized_hlo_path.write_text(optimized_hlo, encoding="utf-8")

        payload = {
            "cable": label,
            "callable": f"{module.__name__}.{attribute}",
            "trace_s": trace_s,
            "lower_s": lower_s,
            "compile_s": compile_s,
            "first_execution_s": first_execution_s,
            "total_cold_s": trace_s + lower_s + compile_s + first_execution_s,
            "stablehlo": _text_metadata(stablehlo, stablehlo_path),
            "optimized_hlo": _text_metadata(optimized_hlo, optimized_hlo_path),
            "triton_kernel_cache": triton_kernel_cache,
            "static": {
                name: _json_scalar(kwargs[name])
                for name in sorted(static_args)
                if name in kwargs
            },
            "dynamic": {
                name: _shape_tree(value)
                for name, value in dynamic_kwargs.items()
            },
        }
        (output_dir / f"{label}.jit_phases.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    setattr(module, attribute, capture_first_call)


def _text_metadata(text: str, path: Path) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "lines": text.count("\n") + 1,
        "custom_calls": text.count("custom_call"),
    }


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


__all__ = ["install_production_jax_captures"]
