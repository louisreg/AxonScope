"""Shared filesystem evidence helpers for fresh-process cache replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def cache_tree_snapshot(root: Path) -> dict[str, int]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }


def cache_tree_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, Any]:
    new_files = sorted(set(after) - set(before))
    changed_files = sorted(
        path for path in set(after) & set(before) if after[path] != before[path]
    )
    return {
        "file_count": len(after),
        "bytes": sum(after.values()),
        "new_file_count": len(new_files),
        "new_bytes": sum(after[path] for path in new_files),
        "changed_file_count": len(changed_files),
        "new_files": new_files,
        "changed_files": changed_files,
    }


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else float("inf")


__all__ = ["cache_tree_delta", "cache_tree_snapshot", "ratio"]
