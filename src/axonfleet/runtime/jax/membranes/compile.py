from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

from axonfleet.axons.axon import Axon
from axonfleet.benchmarking import benchmark_span, record_benchmark_metadata
from axonfleet.membranes.compiler import (
    lower_membrane_model_with_sources,
    membrane_source_path,
)
from axonfleet.membranes.model import ensure_membrane_model
from axonfleet.model_ir.source import (
    GeneratedCodeCache,
    GeneratedSourceRuntimeResult,
    SOURCE_COMPILER_VERSION,
    SOURCE_CONTRACT_VERSION,
    SourceModelCompileResult,
    ensure_generated_model_ir_runtime,
    load_generated_model_ir_runtime,
    load_generated_source_runtime,
)
from axonfleet.runtime.jax.membranes.backend import (
    MembraneBackend,
    UniformMembraneBackend,
    membrane_static_signature,
)
from axonfleet.runtime.jax.membranes.layout import CompartmentMembraneLayout
from axonfleet.runtime.jax.membranes.generated_contract import (
    load_generated_membrane_contract,
)
from axonfleet.runtime.jax.membranes.program import JaxMembraneProgram
from axonfleet.runtime.solver_axon import SolverAxon, build_solver_axon


_MEMBRANE_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}
_BACKEND_CACHE: dict[tuple[Any, ...], MembraneBackend] = {}


@dataclass(frozen=True, slots=True)
class _CachedCompositeRuntime:
    cache: GeneratedCodeCache
    source_results: tuple[GeneratedSourceRuntimeResult, ...]
    parameter_overrides: dict[str, Any]


def compile_membrane_model(model: Any) -> JaxMembraneProgram:
    """Compile a public membrane description to the JAX membrane program."""

    model = ensure_membrane_model(model)
    cached_runtime: GeneratedSourceRuntimeResult | None = None
    cached_composite: _CachedCompositeRuntime | None = None
    lowered = None
    generated_targets = _runtime_codegen_targets()

    with benchmark_span("runtime.prepare.membrane_compile.source_lowering"):
        try:
            if model.kind == "composite":
                cached_composite = _load_cached_composite_runtime(model)
            else:
                cached_runtime = _load_cached_generated_runtime(model)
            if cached_runtime is None and cached_composite is None:
                lowered = lower_membrane_model_with_sources(
                    model,
                    load_generated_modules=("jax", "numpy"),
                    generated_targets=generated_targets,
                )
        except ValueError as exc:
            raise ValueError(f"Unknown membrane model kind: {model.kind!r}") from exc
    source_results = (
        cached_composite.source_results
        if cached_composite is not None
        else (cached_runtime,)
        if cached_runtime is not None
        else lowered.source_results
    )
    composite_cache = (
        None if cached_composite is None else cached_composite.cache
    )
    if model.kind == "composite" and composite_cache is None:
        assert lowered is not None
        composite_cache = ensure_generated_model_ir_runtime(
            lowered.model,
            cache_identity=_composite_cache_identity(model, lowered.source_results),
            load_generated_modules=("jax", "numpy"),
            generated_targets=generated_targets,
        )
    _record_membrane_source_compile_metadata(
        model.kind,
        source_results,
        generated_cache=composite_cache,
    )
    with benchmark_span("runtime.prepare.membrane_compile.program_build"):
        if cached_composite is not None:
            return JaxMembraneProgram.from_generated_module(
                cached_composite.cache.loaded_modules["jax"],
                host_module=cached_composite.cache.loaded_modules["numpy"],
                parameter_overrides=cached_composite.parameter_overrides,
                dtype_local=model.dtype,
                codegen_cache=_codegen_cache_metadata(cached_composite.cache),
                public_model_name=model.kind,
            )
        if cached_runtime is not None:
            return JaxMembraneProgram.from_generated_module(
                cached_runtime.cache.loaded_modules["jax"],
                host_module=cached_runtime.cache.loaded_modules["numpy"],
                parameter_overrides={
                    str(name): value for name, value in model.params.items()
                },
                dtype_local=model.dtype,
                codegen_cache=_codegen_cache_metadata(cached_runtime.cache),
                public_model_name=model.kind,
            )
        assert lowered is not None
        if composite_cache is not None:
            return JaxMembraneProgram.from_generated_module(
                composite_cache.loaded_modules["jax"],
                host_module=composite_cache.loaded_modules["numpy"],
                parameter_overrides={
                    parameter.name: parameter.default
                    for parameter in lowered.model.parameters
                    if parameter.default is not None
                },
                dtype_local=model.dtype,
                codegen_cache=_codegen_cache_metadata(composite_cache),
                public_model_name=model.kind,
            )
        if len(lowered.source_results) != 1:
            raise RuntimeError(
                "Membrane compilation did not produce one generated runtime artifact."
            )
        source_cache = lowered.source_results[0].cache
        missing_targets = {"jax", "numpy"}.difference(source_cache.loaded_modules)
        if missing_targets:
            names = ", ".join(sorted(missing_targets))
            raise RuntimeError(
                f"Membrane compilation is missing generated targets: {names}."
            )
        return JaxMembraneProgram.from_generated_module(
            source_cache.loaded_modules["jax"],
            host_module=source_cache.loaded_modules["numpy"],
            parameter_overrides={
                parameter.name: parameter.default
                for parameter in lowered.model.parameters
                if parameter.default is not None
            },
            dtype_local=model.dtype,
            codegen_cache=_codegen_cache_metadata(source_cache),
            public_model_name=model.kind,
        )


def _load_cached_generated_runtime(model: Any) -> GeneratedSourceRuntimeResult | None:
    if model.kind == "composite":
        return None
    source_path = (
        Path(model.source_path).resolve()
        if model.source_path is not None
        else membrane_source_path(model.kind)
    )
    return load_generated_source_runtime(
        source_path,
        model_class_name=model.source_class,
        targets=_runtime_codegen_targets(),
        load_targets=("jax", "numpy"),
    )


def _load_cached_composite_runtime(model: Any) -> _CachedCompositeRuntime | None:
    source_results: list[GeneratedSourceRuntimeResult] = []
    contracts = []
    for component in model.components:
        if component.kind == "composite":
            return None
        runtime = _load_cached_generated_runtime(component)
        if runtime is None:
            return None
        source_results.append(runtime)
        contracts.append(
            load_generated_membrane_contract(runtime.cache.loaded_modules["jax"])
        )
    identity = _composite_cache_identity(model, tuple(source_results))
    cache = load_generated_model_ir_runtime(
        cache_identity=identity,
        targets=_runtime_codegen_targets(),
        load_targets=("jax", "numpy"),
    )
    if cache is None:
        return None
    symbol_counts = Counter(
        value.name
        for contract in contracts
        for value in (*contract.parameters, *contract.states)
    )
    parameter_overrides: dict[str, Any] = {}
    for index, (component, contract) in enumerate(
        zip(model.components, contracts, strict=True)
    ):
        for name, value in contract.parameter_values(component.params).items():
            runtime_name = f"c{index}__{name}" if symbol_counts[name] > 1 else name
            parameter_overrides[runtime_name] = value
    return _CachedCompositeRuntime(
        cache=cache,
        source_results=tuple(source_results),
        parameter_overrides=parameter_overrides,
    )


def _composite_cache_identity(
    model: Any,
    source_results: tuple[
        SourceModelCompileResult | GeneratedSourceRuntimeResult,
        ...,
    ],
) -> dict[str, Any]:
    return {
        "component_cache_keys": tuple(source.cache.key for source in source_results),
        "component_labels": model.component_labels,
        "kind": "composite",
    }


def _codegen_cache_metadata(cache: GeneratedCodeCache) -> dict[str, Any]:
    return {
        "compiler": SOURCE_COMPILER_VERSION,
        "contract": SOURCE_CONTRACT_VERSION,
        "directory": str(cache.directory),
        "files": tuple(path.name for path in cache.generated_files),
        "key": cache.key,
        "manifest": cache.manifest_path.name,
        "targets": tuple(sorted(cache.loaded_modules)),
    }


def _runtime_codegen_targets() -> tuple[str, ...]:
    targets = ["jax", "numpy"]
    if (
        importlib.util.find_spec("triton") is not None
        and importlib.util.find_spec("jax_triton") is not None
    ):
        targets.append("triton")
    return tuple(targets)


def compile_axon_membrane(
    axon: Axon,
    *,
    solver_axon: SolverAxon | None = None,
    membrane_signatures: tuple[Any, ...] | None = None,
) -> Any:
    """Compile the membrane description carried by an axon."""

    solver_data = build_solver_axon(axon) if solver_axon is None else solver_axon
    membrane_models = solver_data.membrane_models
    if len(membrane_models) == 0:
        raise ValueError("Axon membrane_models cannot be empty.")
    if membrane_signatures is None:
        membrane_signatures = tuple(model._static_signature() for model in membrane_models)
    cache_key = (
        "axon_membrane",
        membrane_signatures,
    )
    cached = _MEMBRANE_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    first_signature = membrane_signatures[0]
    compiled: Any
    if all(signature == first_signature for signature in membrane_signatures):
        compiled = compile_membrane_model(membrane_models[0])
    else:
        compiled_by_signature: dict[tuple[Any, ...], Any] = {}
        for model, signature in zip(membrane_models, membrane_signatures, strict=True):
            compiled_component = compiled_by_signature.get(signature)
            if compiled_component is None:
                compiled_component = compile_membrane_model(model)
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
    source_results: tuple[
        SourceModelCompileResult | GeneratedSourceRuntimeResult,
        ...,
    ],
    *,
    generated_cache: GeneratedCodeCache | None = None,
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
    generated_module_policy = (
        "composite_generated"
        if generated_cache is not None
        else "single_source_loaded"
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
        membrane_composite_cache=(
            None
            if generated_cache is None
            else "hit" if generated_cache.cache_hit else "miss"
        ),
        membrane_composite_cache_key=(
            None if generated_cache is None else generated_cache.key
        ),
    )
