from __future__ import annotations

from typing import Any, cast

from axonscope.axons.axon import Axon
from axonscope.benchmarking import record_benchmark_metadata
from axonscope.membranes.compiler import lower_membrane_model_with_sources
from axonscope.membranes.model import ensure_membrane_model
from axonscope.model_ir.source import SourceModelCompileResult
from axonscope.runtime.jax.membranes.backend import (
    MembraneBackend,
    UniformMembraneBackend,
    membrane_static_signature,
)
from axonscope.runtime.jax.membranes.layout import CompartmentMembraneLayout
from axonscope.runtime.jax.membranes.program import JaxMembraneProgram
from axonscope.runtime.solver_axon import SolverAxon, build_solver_axon
from axonscope.solvers.options import SolverOptions


_MEMBRANE_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}
_BACKEND_CACHE: dict[tuple[Any, ...], MembraneBackend] = {}


def _resolve_solver_options(options: SolverOptions | None) -> SolverOptions:
    return SolverOptions() if options is None else options


def _solver_options_cache_key(options: SolverOptions) -> tuple[Any, ...]:
    _ = options
    return ("solver_options",)


def compile_membrane_model(
    model: Any,
    *,
    solver_options: SolverOptions | None = None,
) -> JaxMembraneProgram:
    """Compile a public membrane description to the JAX membrane program."""

    _resolve_solver_options(solver_options)
    model = ensure_membrane_model(model)

    try:
        lowered = lower_membrane_model_with_sources(
            model,
            load_generated_modules=("jax",),
            generated_targets=("jax",),
        )
    except ValueError as exc:
        raise ValueError(f"Unknown membrane model kind: {model.kind!r}") from exc
    _record_membrane_source_compile_metadata(model.kind, lowered.source_results)
    return cast(
        JaxMembraneProgram,
        JaxMembraneProgram.from_model_ir(
            lowered.model,
            dtype_local=model.dtype,
            generated_module=_single_generated_module(
                lowered.source_results,
                target="jax",
            ),
        ),
    )


def compile_axon_membrane(
    axon: Axon,
    *,
    solver_axon: SolverAxon | None = None,
    solver_options: SolverOptions | None = None,
    membrane_signatures: tuple[Any, ...] | None = None,
) -> Any:
    """Compile the membrane description carried by an axon."""

    options = _resolve_solver_options(solver_options)
    solver_data = build_solver_axon(axon) if solver_axon is None else solver_axon
    membrane_models = solver_data.membrane_models
    if len(membrane_models) == 0:
        raise ValueError("Axon membrane_models cannot be empty.")
    if membrane_signatures is None:
        membrane_signatures = tuple(model._static_signature() for model in membrane_models)
    cache_key = (
        "axon_membrane",
        membrane_signatures,
        _solver_options_cache_key(options),
    )
    cached = _MEMBRANE_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    first_signature = membrane_signatures[0]
    compiled: Any
    if all(signature == first_signature for signature in membrane_signatures):
        compiled = compile_membrane_model(
            membrane_models[0],
            solver_options=options,
        )
    else:
        compiled_by_signature: dict[tuple[Any, ...], Any] = {}
        for model, signature in zip(membrane_models, membrane_signatures, strict=True):
            compiled_component = compiled_by_signature.get(signature)
            if compiled_component is None:
                compiled_component = compile_membrane_model(
                    model,
                    solver_options=options,
                )
                compiled_by_signature[signature] = compiled_component
        compiled_components = tuple(
            compiled_by_signature[signature]
            for signature in membrane_signatures
        )
        compiled = CompartmentMembraneLayout(compiled_components).as_membrane_model()
    _MEMBRANE_MODEL_CACHE[cache_key] = compiled
    return compiled


def backend_from_membrane(membrane: Any, nx: int) -> MembraneBackend:
    cache_key = ("backend", membrane_static_signature(membrane), int(nx))
    cached = _BACKEND_CACHE.get(cache_key)
    if cached is not None:
        return cached
    build_backend = getattr(membrane, "build_backend", None)
    if callable(build_backend):
        backend = build_backend()
    else:
        backend = UniformMembraneBackend.from_model(membrane, int(nx))
    _BACKEND_CACHE[cache_key] = backend
    return backend


def _record_membrane_source_compile_metadata(
    kind: str,
    source_results: tuple[SourceModelCompileResult, ...],
) -> None:
    if not source_results:
        return
    statuses = tuple(
        "hit" if source.cache.cache_hit else "miss"
        for source in source_results
    )
    reasons = tuple(source.cache.cache_reason for source in source_results)
    keys = tuple(source.cache.key for source in source_results)
    hashes = tuple(source.source_hash for source in source_results)
    paths = tuple(str(source.source_path) for source in source_results)
    loaded_targets = tuple(
        tuple(sorted(source.cache.loaded_modules))
        for source in source_results
    )
    single = len(source_results) == 1
    generated_module_policy = _generated_module_policy(
        source_results,
        target="jax",
    )
    record_benchmark_metadata(
        membrane_source_cache=statuses[0] if single else statuses,
        membrane_source_cache_all_hit=all(status == "hit" for status in statuses),
        membrane_source_cache_reasons=reasons[0] if single else reasons,
        membrane_source_generated_module_policy=generated_module_policy,
        membrane_source_loaded_targets=loaded_targets[0] if single else loaded_targets,
        membrane_source_cache_keys=keys[0] if single else keys,
        membrane_source_hashes=hashes[0] if single else hashes,
        membrane_source_kind=kind,
        membrane_source_paths=paths[0] if single else paths,
        membrane_source_count=len(source_results),
    )


def _single_generated_module(
    source_results: tuple[SourceModelCompileResult, ...],
    *,
    target: str,
) -> Any | None:
    if len(source_results) != 1:
        return None
    return source_results[0].cache.loaded_modules.get(target)


def _generated_module_policy(
    source_results: tuple[SourceModelCompileResult, ...],
    *,
    target: str,
) -> str:
    if len(source_results) == 1:
        if target in source_results[0].cache.loaded_modules:
            return "single_source_loaded"
        return "single_source_missing"
    return "multi_source_fallback"
