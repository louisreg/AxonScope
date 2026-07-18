"""Measure Model IR and autonomous generated membrane cache-hit paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from axonscope.model_ir.source import (
    compile_model_source_file,
    load_generated_source_runtime,
)
from axonscope import membranes
from axonscope.runtime.jax.membranes.compile import compile_membrane_model
from axonscope.runtime.jax.membranes.program import JaxMembraneProgram


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "src" / "axonscope" / "membranes" / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="hodgkin_huxley,schild97",
        help="Comma-separated built-in source model names.",
    )
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "repeats": int(args.repeats),
        "models": {
            name: measure_model(name, repeats=int(args.repeats))
            for name in args.models.split(",")
            if name
        },
        "composite": measure_composite(repeats=int(args.repeats)),
        "notes": [
            "Fresh temporary codegen cache per model.",
            "Generated modules remain imported between repeats.",
            "Times isolate cache loading and membrane-program construction.",
            "Composite uses labelled Rattay-Aberham plus Passive components.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def measure_composite(*, repeats: int) -> dict[str, float | bool]:
    model = membranes.Composite(
        {
            "excitable": membranes.RattayAberham(),
            "extra_leak": membranes.Passive(Rm=12_000.0, EL=-68.0),
        }
    )
    previous_cache = os.environ.get("AXONSCOPE_MODEL_CODEGEN_CACHE")
    with TemporaryDirectory() as cache_root:
        os.environ["AXONSCOPE_MODEL_CODEGEN_CACHE"] = cache_root
        start = perf_counter()
        cold = compile_membrane_model(model)
        cold_ms = 1e3 * (perf_counter() - start)
        warm: list[float] = []
        for _ in range(repeats):
            start = perf_counter()
            compiled = compile_membrane_model(model)
            warm.append(perf_counter() - start)
        key_reused = cold.codegen_cache["key"] == compiled.codegen_cache["key"]
        generated_only = cold.model_ir is None and compiled.model_ir is None
    if previous_cache is None:
        os.environ.pop("AXONSCOPE_MODEL_CODEGEN_CACHE", None)
    else:
        os.environ["AXONSCOPE_MODEL_CODEGEN_CACHE"] = previous_cache
    return {
        "first_compile_ms": cold_ms,
        "cache_hit_compile_median_ms": 1e3 * median(warm),
        "generated_runtime_only": generated_only,
        "cache_key_reused": key_reused,
    }


def measure_model(name: str, *, repeats: int) -> dict[str, float]:
    source = MODEL_ROOT / f"{name}.py"
    if not source.is_file():
        raise ValueError(f"Unknown built-in membrane source {name!r}.")
    with TemporaryDirectory() as cache_root:
        compile_model_source_file(
            source,
            cache_root=cache_root,
            generated_targets=("jax", "numpy"),
            load_generated_modules=("jax", "numpy"),
        )
        graph_load: list[float] = []
        generated_load: list[float] = []
        model_ir_build: list[float] = []
        generated_build: list[float] = []
        for _ in range(repeats):
            start = perf_counter()
            compiled = compile_model_source_file(
                source,
                cache_root=cache_root,
                generated_targets=("jax", "numpy"),
                load_generated_modules=("jax", "numpy"),
            )
            graph_load.append(perf_counter() - start)

            start = perf_counter()
            cached = load_generated_source_runtime(
                source,
                cache_root=cache_root,
                targets=("jax", "numpy"),
            )
            generated_load.append(perf_counter() - start)
            assert cached is not None

            start = perf_counter()
            JaxMembraneProgram.from_model_ir(
                compiled.model,
                generated_module=compiled.cache.loaded_modules["jax"],
                host_module=compiled.cache.loaded_modules["numpy"],
            )
            model_ir_build.append(perf_counter() - start)

            start = perf_counter()
            JaxMembraneProgram.from_generated_module(
                cached.cache.loaded_modules["jax"],
                host_module=cached.cache.loaded_modules["numpy"],
                parameter_overrides={},
            )
            generated_build.append(perf_counter() - start)

    graph_load_ms = 1e3 * median(graph_load)
    generated_load_ms = 1e3 * median(generated_load)
    model_ir_build_ms = 1e3 * median(model_ir_build)
    generated_build_ms = 1e3 * median(generated_build)
    return {
        "cached_graph_load_median_ms": graph_load_ms,
        "generated_runtime_load_median_ms": generated_load_ms,
        "load_speedup": graph_load_ms / generated_load_ms,
        "model_ir_program_build_median_ms": model_ir_build_ms,
        "generated_program_build_median_ms": generated_build_ms,
        "program_build_speedup": model_ir_build_ms / generated_build_ms,
    }


if __name__ == "__main__":
    main()
