"""Persistent artifact cache policy and user-facing maintenance helpers."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile


_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disabled"})
_CACHE_ENV = "AXONFLEET_CACHE"
_CACHE_MARKER = ".axonfleet-cache"
_KNOWN_SECTIONS = {
    "model_codegen": ("model_codegen",),
    "jax_xla": ("runtime", "jax", "xla"),
    "jax_triton": ("runtime", "jax", "triton"),
}
_EPHEMERAL_ROOT: Path | None = None


@dataclass(frozen=True, slots=True)
class CacheSection:
    """Size and location of one persistent cache section."""

    name: str
    directory: Path | None
    file_count: int
    bytes: int


@dataclass(frozen=True, slots=True)
class CacheSnapshot:
    """Current persistent AxonFleet cache state."""

    enabled: bool
    directory: Path | None
    sections: tuple[CacheSection, ...]

    @property
    def file_count(self) -> int:
        return sum(section.file_count for section in self.sections)

    @property
    def bytes(self) -> int:
        return sum(section.bytes for section in self.sections)


def directory() -> Path | None:
    """Return the configured persistent cache root, or ``None`` when disabled."""

    configured = os.environ.get(_CACHE_ENV, "").strip()
    if configured.lower() in _FALSE_VALUES:
        return None
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / ".axonfleet_cache").resolve()


def inspect() -> CacheSnapshot:
    """Inspect persistent generated-code and runtime artifacts without creating them."""

    root = directory()
    sections = tuple(
        _section_snapshot(name, None if root is None else root.joinpath(*parts))
        for name, parts in _KNOWN_SECTIONS.items()
    )
    return CacheSnapshot(enabled=root is not None, directory=root, sections=sections)


def clean() -> CacheSnapshot:
    """Remove every known persistent AxonFleet cache section and return its new state."""

    root = directory()
    if root is None or not root.exists():
        return inspect()
    for parts in _KNOWN_SECTIONS.values():
        target = root.joinpath(*parts)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    _prune_empty_cache_directories(root)
    return inspect()


def persistent_directory(*parts: str) -> Path | None:
    """Return one path below the persistent root without creating it."""

    root = directory()
    return None if root is None else root.joinpath(*parts)


def writable_directory(*parts: str) -> Path:
    """Return persistent storage, or process-temporary storage when disabled."""

    root = directory()
    if root is None:
        root = _ephemeral_root()
    else:
        root.mkdir(parents=True, exist_ok=True)
        (root / _CACHE_MARKER).touch(exist_ok=True)
    return root.joinpath(*parts)


def _ephemeral_root() -> Path:
    global _EPHEMERAL_ROOT
    if _EPHEMERAL_ROOT is None:
        _EPHEMERAL_ROOT = Path(tempfile.mkdtemp(prefix="axonfleet-cache-"))
        atexit.register(shutil.rmtree, _EPHEMERAL_ROOT, True)
    return _EPHEMERAL_ROOT


def _section_snapshot(name: str, path: Path | None) -> CacheSection:
    if path is None or not path.exists():
        return CacheSection(
            name=name,
            directory=path,
            file_count=0,
            bytes=0,
        )
    files = tuple(item for item in path.rglob("*") if item.is_file())
    return CacheSection(
        name=name,
        directory=path,
        file_count=len(files),
        bytes=sum(item.stat().st_size for item in files),
    )


def _prune_empty_cache_directories(root: Path) -> None:
    marker = root / _CACHE_MARKER
    marker.unlink(missing_ok=True)
    for candidate in (root / "runtime" / "jax", root / "runtime", root):
        try:
            candidate.rmdir()
        except OSError:
            pass


__all__ = [
    "CacheSection",
    "CacheSnapshot",
    "clean",
    "directory",
    "inspect",
]
