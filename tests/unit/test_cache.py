from __future__ import annotations

from pathlib import Path

import axonfleet as axs


def test_cache_inspect_and_clean_cover_known_sections(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("AXONFLEET_CACHE", str(root))
    files = {
        root / "model_codegen" / "model" / "jax_model.py": b"jax",
        root / "runtime" / "jax" / "xla" / "entry": b"xla-cache",
        root / "runtime" / "jax" / "triton" / "kernel": b"triton",
    }
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    snapshot = axs.cache.inspect()

    assert snapshot.enabled is True
    assert snapshot.directory == root
    assert snapshot.file_count == 3
    assert snapshot.bytes == sum(len(payload) for payload in files.values())
    assert {section.name for section in snapshot.sections} == {
        "model_codegen",
        "jax_xla",
        "jax_triton",
    }

    cleaned = axs.cache.clean()

    assert cleaned.file_count == 0
    assert cleaned.bytes == 0
    assert not root.exists()


def test_cache_can_be_disabled_without_disabling_required_temporary_codegen(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AXONFLEET_CACHE", "off")

    snapshot = axs.cache.inspect()
    temporary = axs.cache.writable_directory("model_codegen")

    assert snapshot.enabled is False
    assert snapshot.directory is None
    assert temporary.is_absolute()
    assert "axonfleet-cache-" in str(temporary)
    assert axs.cache.directory() is None


def test_cache_clean_preserves_unknown_files(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("AXONFLEET_CACHE", str(root))
    unknown = root / "user-note.txt"
    unknown.parent.mkdir(parents=True)
    unknown.write_text("keep", encoding="utf-8")

    axs.cache.clean()

    assert unknown.read_text(encoding="utf-8") == "keep"
