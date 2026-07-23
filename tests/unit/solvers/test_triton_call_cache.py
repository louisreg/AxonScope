from __future__ import annotations

import json

import jax
import numpy as np

from axonfleet.runtime.jax.kernels import triton_call_cache


def _payload(*, name: str = "test_kernel", source_hash: str = "source"):
    shape = jax.ShapeDtypeStruct((17, 4), np.dtype("float32"))
    return triton_call_cache._kernel_cache_payload(
        name=name,
        source_hash=source_hash,
        platform="cuda",
        compute_capability=70,
        abstract_args=[shape, shape],
        arg_dtypes=["*fp32", "*fp32"],
        grid=(1, 1, 1),
        num_warps=4,
        num_stages=1,
        num_ctas=1,
        enable_fp_fusion=True,
        input_output_aliases=(),
        serialized_metadata=b"",
        metaparams=(("BLOCK", 128),),
    )


def test_triton_cache_root_defaults_under_axonfleet_cache(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AXONFLEET_CACHE", raising=False)

    assert triton_call_cache._triton_cache_root() == (
        tmp_path / ".axonfleet_cache" / "runtime" / "jax" / "triton"
    )


def test_triton_cache_root_honors_environment_override(
    monkeypatch,
    tmp_path,
) -> None:
    configured = tmp_path / "compiled"
    monkeypatch.setenv("AXONFLEET_CACHE", str(configured))

    assert triton_call_cache._triton_cache_root() == (
        configured / "runtime" / "jax" / "triton"
    )


def test_triton_cache_artifact_round_trip(tmp_path) -> None:
    payload = _payload()
    key = triton_call_cache._hash_payload(payload)
    artifact = b"compressed kernel call"

    reason = triton_call_cache._write_backend_config(
        tmp_path,
        key,
        payload,
        artifact,
    )
    loaded, read_reason = triton_call_cache._read_backend_config(
        tmp_path,
        key,
        payload,
    )

    assert reason == "compiled kernel call persisted"
    assert loaded == artifact
    assert read_reason == "compiled kernel call reused"
    manifest = json.loads((tmp_path / key / "manifest.json").read_text())
    assert manifest["payload"] == payload


def test_triton_cache_rejects_corrupt_artifact(tmp_path) -> None:
    payload = _payload()
    key = triton_call_cache._hash_payload(payload)
    triton_call_cache._write_backend_config(tmp_path, key, payload, b"valid")
    (tmp_path / key / "kernel_call.zlib").write_bytes(b"corrupt")

    artifact, reason = triton_call_cache._read_backend_config(
        tmp_path,
        key,
        payload,
    )

    assert artifact is None
    assert reason == "cache artifact checksum mismatch"


def test_triton_cache_key_covers_call_identity() -> None:
    baseline = triton_call_cache._hash_payload(_payload())

    assert triton_call_cache._hash_payload(_payload(name="other")) != baseline
    assert (
        triton_call_cache._hash_payload(_payload(source_hash="changed"))
        != baseline
    )
