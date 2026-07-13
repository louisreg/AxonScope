"""Persistent lowering cache for AxonScope-owned jax-triton kernels."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
import zlib
from functools import partial
from pathlib import Path
from typing import Any

import jax
from jax import extend as jex
from jax._src import core
from jax.interpreters import ad, batching, mlir, xla


_CACHE_CONTRACT = 1
# The lowering below mirrors private jax-triton internals; review upstream before
# admitting another version instead of silently replaying an incompatible proto.
_SUPPORTED_JAX_TRITON_VERSIONS = frozenset({"0.3.1"})
_LAST_CACHE_EVENT: dict[str, Any] | None = None


_cached_triton_kernel_call_p = jex.core.Primitive(
    "axonscope_cached_triton_kernel_call"
)
_cached_triton_kernel_call_p.multiple_results = True
_cached_triton_kernel_call_p.def_impl(
    partial(xla.apply_primitive, _cached_triton_kernel_call_p)
)


@_cached_triton_kernel_call_p.def_abstract_eval
def _cached_triton_kernel_call_abstract_eval(
    *_,
    out_shapes: tuple[Any, ...],
    **__,
) -> list[core.ShapedArray]:
    return [
        core.ShapedArray(out_shape.shape, out_shape.dtype)
        for out_shape in out_shapes
    ]


def cached_triton_call(
    *args: Any,
    kernel: Any,
    source_hash: str,
    out_shape: Any,
    grid: int | tuple[int, ...],
    name: str,
    num_warps: int,
    num_stages: int,
    num_ctas: int = 1,
    compute_capability: int | None = None,
    enable_fp_fusion: bool = True,
    input_output_aliases: dict[int, int] | None = None,
    serialized_metadata: bytes = b"",
    **metaparams: Any,
) -> Any:
    """Call a fixed Triton kernel through a persistent compiled-call cache."""

    import jax_triton as jt

    jax_triton_version = _package_version("jax-triton")
    if jax_triton_version not in _SUPPORTED_JAX_TRITON_VERSIONS:
        _record_cache_event(
            status="unsupported",
            reason=f"jax-triton {jax_triton_version!r} is not cache-enabled",
        )
        return jt.triton_call(
            *args,
            kernel=kernel,
            out_shape=out_shape,
            grid=grid,
            name=name,
            num_warps=num_warps,
            num_stages=num_stages,
            num_ctas=num_ctas,
            compute_capability=compute_capability,
            enable_fp_fusion=enable_fp_fusion,
            input_output_aliases=input_output_aliases,
            serialized_metadata=serialized_metadata,
            **metaparams,
        )

    single_output = hasattr(out_shape, "shape") and hasattr(out_shape, "dtype")
    out_shapes = (out_shape,) if single_output else tuple(out_shape)
    normalized_out_shapes = tuple(
        jax.ShapeDtypeStruct(value.shape, value.dtype) for value in out_shapes
    )
    grid_tuple = (int(grid),) if isinstance(grid, int) else tuple(map(int, grid))
    outputs = _cached_triton_kernel_call_p.bind(
        *args,
        fn=kernel,
        source_hash=str(source_hash),
        out_shapes=normalized_out_shapes,
        grid=grid_tuple,
        name=str(name),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
        num_ctas=int(num_ctas),
        compute_capability=compute_capability,
        enable_fp_fusion=bool(enable_fp_fusion),
        input_output_aliases=tuple(sorted((input_output_aliases or {}).items())),
        serialized_metadata=bytes(serialized_metadata),
        metaparams=tuple(sorted(metaparams.items())),
    )
    if single_output:
        return outputs[0]
    return tuple(outputs)


def last_triton_kernel_cache_event() -> dict[str, Any] | None:
    """Return metadata for the most recent cached Triton lowering."""

    return None if _LAST_CACHE_EVENT is None else dict(_LAST_CACHE_EVENT)


def _cached_triton_kernel_call_lowering(
    ctx: Any,
    *array_args: Any,
    fn: Any,
    source_hash: str,
    out_shapes: tuple[Any, ...],
    grid: tuple[int, ...],
    name: str,
    num_warps: int,
    num_stages: int,
    num_ctas: int,
    compute_capability: int | None,
    enable_fp_fusion: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    serialized_metadata: bytes,
    metaparams: tuple[tuple[str, Any], ...],
) -> Any:
    del out_shapes

    import jax_triton.triton_lib as jtlib
    from jax._src.lib import gpu_triton as triton_kernel_call_lib

    platform = ctx.module_context.platforms[0]
    if platform != "cuda":
        raise ValueError(
            f"AxonScope's persistent Triton cache supports CUDA, got {platform!r}."
        )
    resolved_compute_capability = (
        triton_kernel_call_lib.get_compute_capability(0)
        if compute_capability is None
        else int(compute_capability)
    )
    metaparam_dict = dict(metaparams)
    normalized_grid = jtlib.normalize_grid(grid, metaparam_dict)
    abstract_args = [*ctx.avals_in, *ctx.avals_out]
    arg_dtypes = [jtlib.get_triton_type(value) for value in abstract_args]
    payload = _kernel_cache_payload(
        name=name,
        source_hash=source_hash,
        platform=platform,
        compute_capability=resolved_compute_capability,
        abstract_args=abstract_args,
        arg_dtypes=arg_dtypes,
        grid=normalized_grid,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
        enable_fp_fusion=enable_fp_fusion,
        input_output_aliases=input_output_aliases,
        serialized_metadata=serialized_metadata,
        metaparams=metaparams,
    )
    key = _hash_payload(payload)
    cache_root = _triton_cache_root()
    backend_config, reason = _read_backend_config(cache_root, key, payload)

    if backend_config is None:
        compile_start = time.perf_counter()
        compiled_kernel, specialization_attr = jtlib.get_or_create_triton_kernel(
            jtlib.get_cuda_backend,
            platform,
            fn,
            arg_dtypes,
            (),
            num_warps=num_warps,
            num_stages=num_stages,
            num_ctas=num_ctas,
            compute_capability=resolved_compute_capability,
            enable_fp_fusion=enable_fp_fusion,
            metaparams=metaparam_dict,
            dump=False,
        )
        kernel_params = []
        for index, abstract_arg in enumerate(abstract_args):
            arg_attrs = specialization_attr[(index,)]
            kernel_params.append(
                triton_kernel_call_lib.create_array_parameter(
                    0,
                    16 if (["tt.divisibility", 16] in arg_attrs) else 0,
                )
            )
        kernel_call = triton_kernel_call_lib.TritonKernelCall(
            compiled_kernel,
            normalized_grid[0],
            normalized_grid[1],
            normalized_grid[2],
            kernel_params,
        )
        backend_config = zlib.compress(
            kernel_call.to_proto(name, serialized_metadata)
        )
        compile_s = time.perf_counter() - compile_start
        write_reason = _write_backend_config(
            cache_root,
            key,
            payload,
            backend_config,
        )
        _record_cache_event(
            status="miss",
            reason=f"{reason}; {write_reason}",
            key=key,
            directory=str(cache_root / key),
            compile_s=compile_s,
        )
    else:
        _record_cache_event(
            status="hit",
            reason=reason,
            key=key,
            directory=str(cache_root / key),
            compile_s=0.0,
        )

    rule = jax.ffi.ffi_lowering(
        "triton_kernel_call",
        api_version=2,
        backend_config=backend_config,
        operand_output_aliases=dict(input_output_aliases),
    )
    return rule(ctx, *array_args)


mlir.register_lowering(
    _cached_triton_kernel_call_p,
    _cached_triton_kernel_call_lowering,
    platform="cuda",
)


def _raise_on_jvp(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise NotImplementedError("Cached AxonScope Triton calls do not support JVP.")


def _raise_on_vmap(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise NotImplementedError("Cached AxonScope Triton calls do not support vmap.")


ad.primitive_jvps[_cached_triton_kernel_call_p] = _raise_on_jvp
batching.primitive_batchers[_cached_triton_kernel_call_p] = _raise_on_vmap


def _kernel_cache_payload(
    *,
    name: str,
    source_hash: str,
    platform: str,
    compute_capability: int,
    abstract_args: list[Any],
    arg_dtypes: list[str],
    grid: tuple[int, int, int],
    num_warps: int,
    num_stages: int,
    num_ctas: int,
    enable_fp_fusion: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    serialized_metadata: bytes,
    metaparams: tuple[tuple[str, Any], ...],
) -> dict[str, Any]:
    return {
        "cache_contract": _CACHE_CONTRACT,
        "name": name,
        "source_hash": source_hash,
        "platform": platform,
        "compute_capability": compute_capability,
        "arguments": [
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "triton_type": triton_type,
            }
            for value, triton_type in zip(abstract_args, arg_dtypes)
        ],
        "grid": list(grid),
        "num_warps": num_warps,
        "num_stages": num_stages,
        "num_ctas": num_ctas,
        "enable_fp_fusion": enable_fp_fusion,
        "input_output_aliases": [list(value) for value in input_output_aliases],
        "serialized_metadata_sha256": hashlib.sha256(serialized_metadata).hexdigest(),
        "metaparams": [[name, value] for name, value in metaparams],
        "packages": {
            name: _package_version(name)
            for name in ("jax", "jaxlib", "jax-triton", "triton")
        },
    }


def _triton_cache_root() -> Path:
    configured = os.environ.get("AXONSCOPE_TRITON_KERNEL_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path.cwd() / ".axonscope_cache" / "runtime" / "jax" / "triton"
    ).resolve()


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _read_backend_config(
    root: Path,
    key: str,
    payload: dict[str, Any],
) -> tuple[bytes | None, str]:
    directory = root / key
    manifest_path = directory / "manifest.json"
    artifact_path = directory / "kernel_call.zlib"
    if not manifest_path.is_file() or not artifact_path.is_file():
        return None, "cache artifact absent"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = artifact_path.read_bytes()
    except (OSError, json.JSONDecodeError):
        return None, "cache artifact unreadable"
    if (
        manifest.get("cache_contract") != _CACHE_CONTRACT
        or manifest.get("cache_key") != key
        or manifest.get("payload") != payload
    ):
        return None, "cache manifest mismatch"
    if manifest.get("artifact_sha256") != hashlib.sha256(artifact).hexdigest():
        return None, "cache artifact checksum mismatch"
    return artifact, "compiled kernel call reused"


def _write_backend_config(
    root: Path,
    key: str,
    payload: dict[str, Any],
    artifact: bytes,
) -> str:
    directory = root / key
    try:
        directory.mkdir(parents=True, exist_ok=True)
        artifact_path = directory / "kernel_call.zlib"
        manifest_path = directory / "manifest.json"
        suffix = f".{os.getpid()}.tmp"
        artifact_tmp = artifact_path.with_name(artifact_path.name + suffix)
        manifest_tmp = manifest_path.with_name(manifest_path.name + suffix)
        artifact_tmp.write_bytes(artifact)
        manifest_tmp.write_text(
            json.dumps(
                {
                    "artifact": artifact_path.name,
                    "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
                    "cache_contract": _CACHE_CONTRACT,
                    "cache_key": key,
                    "payload": payload,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(artifact_tmp, artifact_path)
        os.replace(manifest_tmp, manifest_path)
    except OSError as exc:
        return f"cache write skipped ({type(exc).__name__})"
    return "compiled kernel call persisted"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _record_cache_event(*, status: str, reason: str, **details: Any) -> None:
    global _LAST_CACHE_EVENT
    _LAST_CACHE_EVENT = {"status": status, "reason": reason, **details}


__all__ = ["cached_triton_call", "last_triton_kernel_cache_event"]
