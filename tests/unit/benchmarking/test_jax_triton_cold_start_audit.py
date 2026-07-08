from __future__ import annotations

from argparse import Namespace

import numpy as np

from benchmark.analysis.jax_triton_cold_start_audit import (
    _cache_paths,
    _output_nbytes,
    _path_snapshot,
)


def test_cold_start_audit_path_snapshot_counts_files(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "a.bin").write_bytes(b"abc")
    nested = cache_dir / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"12345")

    snapshot = _path_snapshot(cache_dir)

    assert snapshot["exists"] is True
    assert snapshot["files"] == 2
    assert snapshot["dirs"] == 1
    assert snapshot["bytes"] == 8


def test_cold_start_audit_cache_paths_deduplicate_explicit_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    args = Namespace(
        jax_cache_dir=tmp_path / "jax",
        triton_cache_dir=tmp_path / "triton",
    )

    paths = _cache_paths(args)

    assert paths.count(tmp_path / "triton") == 1
    assert tmp_path / "jax" in paths


def test_cold_start_audit_output_nbytes_sums_nested_outputs():
    value = (
        np.zeros((2, 3), dtype=np.float32),
        [np.zeros((4,), dtype=np.float32)],
    )

    assert _output_nbytes(value) == 40
