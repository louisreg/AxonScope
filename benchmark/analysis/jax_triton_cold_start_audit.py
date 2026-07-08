from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform as host_platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SUMMARY_FIELDS = (
    "phase",
    "variant",
    "platform",
    "device",
    "nx",
    "batch_size",
    "block_b",
    "elapsed_ms",
    "rss_delta_mib",
    "output_bytes",
    "status",
    "notes",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Isolate jax-triton cold-start cost for the benchmark-only "
            "double-cable Thomas solver."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/p11c_jax_triton_cold_start_audit"),
    )
    parser.add_argument("--platform", choices=("gpu",), default="gpu")
    parser.add_argument(
        "--variant",
        choices=("tiled_thomas", "untiled_thomas"),
        default="tiled_thomas",
    )
    parser.add_argument("--nx", type=int, default=89)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--block-b", type=int, default=128)
    parser.add_argument("--compiled-calls", type=int, default=2)
    parser.add_argument("--jax-cache-dir", type=Path)
    parser.add_argument("--triton-cache-dir", type=Path)
    parser.add_argument(
        "--clear-cache-dirs",
        action="store_true",
        help="Clear only the explicit --jax-cache-dir/--triton-cache-dir paths.",
    )
    parser.add_argument("--no-cache-snapshot", action="store_true")
    args = parser.parse_args(argv)

    if args.nx < 2:
        parser.error("--nx must be >= 2.")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1.")
    if args.block_b < 1:
        parser.error("--block-b must be >= 1.")
    if args.compiled_calls < 1:
        parser.error("--compiled-calls must be >= 1.")

    _configure_cache(args)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    cache_snapshots: dict[str, Any] = {}
    metadata = _metadata(args)

    def measure(
        phase: str,
        fn: Callable[[], Any],
        *,
        block_output: bool = False,
        output_bytes: bool = False,
    ) -> Any:
        if not args.no_cache_snapshot:
            cache_snapshots[f"{phase}.before"] = _cache_snapshot(args)
        gc.collect()
        rss_start = _rss_mib()
        start = perf_counter()
        status = "ok"
        notes = ""
        result = None
        try:
            result = fn()
            if block_output:
                result = _block_until_ready(result)
        except Exception as exc:  # pragma: no cover - exercised on benchmark workers.
            status = "error"
            notes = repr(exc)
            raise
        finally:
            elapsed_ms = (perf_counter() - start) * 1000.0
            rss_end = _rss_mib()
            rows.append(
                _row(
                    args=args,
                    phase=phase,
                    device=str(_current_device_name()),
                    elapsed_ms=elapsed_ms,
                    rss_delta_mib=_delta(rss_start, rss_end),
                    output_bytes=_output_nbytes(result) if output_bytes else 0,
                    status=status,
                    notes=notes,
                )
            )
            if not args.no_cache_snapshot:
                cache_snapshots[f"{phase}.after"] = _cache_snapshot(args)
        return result

    try:
        device = measure("select_device", lambda: _select_device(args.platform))
        metadata["device"] = str(device)
        with jax.default_device(device):
            measure("dependency_probe", _check_jax_triton_ready)
            inputs = measure(
                "build_inputs",
                lambda: _make_inputs(batch_size=args.batch_size, nx=args.nx),
                block_output=True,
                output_bytes=True,
            )
            solver = _solver_fn(args.variant, block_b=args.block_b)
            lowered = measure("lower", lambda: solver.lower(*inputs))
            compiled = measure("compile", lowered.compile)
            for index in range(args.compiled_calls):
                phase = "compiled_first_call" if index == 0 else f"compiled_call_{index + 1}"
                measure(
                    phase,
                    lambda compiled=compiled, inputs=inputs: compiled(*inputs),
                    block_output=True,
                    output_bytes=True,
                )
    except Exception as exc:
        metadata["error"] = repr(exc)
        _write_outputs(args.output, rows, metadata, cache_snapshots)
        print(f"jax-triton cold-start audit failed: {exc}", file=sys.stderr)
        return 2

    _write_outputs(args.output, rows, metadata, cache_snapshots)
    print(f"wrote: {args.output / 'cold_start_summary.csv'}")
    print(f"wrote: {args.output / 'cold_start_report.md'}")
    return 0


def _configure_cache(args: argparse.Namespace) -> None:
    for path in (args.jax_cache_dir, args.triton_cache_dir):
        if path is None:
            continue
        if args.clear_cache_dirs:
            _clear_explicit_cache_dir(path)
        path.mkdir(parents=True, exist_ok=True)
    if args.triton_cache_dir is not None:
        os.environ["TRITON_CACHE_DIR"] = str(args.triton_cache_dir)
    if args.jax_cache_dir is not None:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(args.jax_cache_dir)
        try:
            jax.config.update("jax_compilation_cache_dir", str(args.jax_cache_dir))
        except Exception:
            pass
        for name, value in (
            ("jax_persistent_cache_min_compile_time_secs", 0),
            ("jax_persistent_cache_min_entry_size_bytes", -1),
        ):
            try:
                jax.config.update(name, value)
            except Exception:
                pass


def _clear_explicit_cache_dir(path: Path) -> None:
    resolved = path.resolve()
    if resolved in {Path("/").resolve(), Path.home().resolve()}:
        raise ValueError(f"refusing to clear unsafe cache directory: {path}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _check_jax_triton_ready() -> None:
    from axonscope.backends.jax.jax_triton_double_cable import (
        jax_triton_thomas_dependency_skip_reason,
    )

    skip_reason = jax_triton_thomas_dependency_skip_reason()
    if skip_reason is not None:
        raise RuntimeError(skip_reason)


def _solver_fn(variant: str, *, block_b: int) -> Any:
    if variant == "tiled_thomas":
        return jax.jit(partial(_solve_tiled_thomas, block_b=int(block_b)))
    if variant == "untiled_thomas":
        return jax.jit(_solve_untiled_thomas)
    raise ValueError(f"unsupported variant: {variant!r}")


def _solve_tiled_thomas(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
    *,
    block_b: int,
) -> tuple[Any, Any]:
    from axonscope.backends.jax.jax_triton_double_cable import (
        solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_batched,
    )

    return solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_batched(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        block_b=block_b,
    )


def _solve_untiled_thomas(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    from axonscope.backends.jax.jax_triton_double_cable import (
        solve_block_tridiagonal_2x2_jax_triton_thomas_batched,
    )

    return solve_block_tridiagonal_2x2_jax_triton_thomas_batched(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    )


def _make_inputs(*, batch_size: int, nx: int) -> tuple[Any, ...]:
    x = jnp.linspace(0.0, 1.0, nx, dtype=jnp.float32)
    rows = jnp.arange(batch_size, dtype=jnp.float32)[:, None]
    a00 = jnp.broadcast_to(2.5 + 0.01 * x, (batch_size, nx))
    a01 = jnp.broadcast_to(-0.35 - 0.01 * x, (batch_size, nx))
    a10 = jnp.broadcast_to(-0.30 - 0.01 * x, (batch_size, nx))
    a11 = jnp.broadcast_to(2.7 + 0.02 * x, (batch_size, nx))
    off_x = jnp.linspace(-0.08, -0.04, nx - 1, dtype=jnp.float32)
    off0 = jnp.broadcast_to(off_x, (batch_size, nx - 1))
    off1 = jnp.broadcast_to(off_x * 0.75, (batch_size, nx - 1))
    rhs0 = 0.1 + 0.001 * rows + jnp.broadcast_to(x[None, :], (batch_size, nx))
    rhs1 = -0.2 + 0.0005 * rows + jnp.broadcast_to((1.0 - x)[None, :], (batch_size, nx))
    return a00, a01, a10, a11, off0, off1, rhs0, rhs1


def _select_device(platform_name: str) -> Any:
    devices = jax.devices(platform_name)
    if not devices:
        raise RuntimeError(f"No JAX {platform_name} device is available.")
    return devices[0]


def _current_device_name() -> str:
    try:
        return str(jax.devices()[0])
    except Exception:
        return "unknown"


def _row(
    *,
    args: argparse.Namespace,
    phase: str,
    device: str,
    elapsed_ms: float,
    rss_delta_mib: float | None,
    output_bytes: int,
    status: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "variant": args.variant,
        "platform": args.platform,
        "device": device,
        "nx": int(args.nx),
        "batch_size": int(args.batch_size),
        "block_b": int(args.block_b),
        "elapsed_ms": elapsed_ms,
        "rss_delta_mib": rss_delta_mib,
        "output_bytes": output_bytes,
        "status": status,
        "notes": notes,
    }


def _write_outputs(
    output: Path,
    rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    cache_snapshots: dict[str, Any],
) -> None:
    _write_csv(output / "cold_start_summary.csv", SUMMARY_FIELDS, rows)
    _write_json(output / "metadata.json", metadata)
    _write_json(output / "cache_snapshots.json", cache_snapshots)
    _write_report(output / "cold_start_report.md", rows, metadata)


def _write_report(path: Path, rows: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> None:
    lines = [
        "# jax-triton Cold-Start Audit",
        "",
        "Benchmark-only audit for the experimental double-cable jax-triton solver.",
        "",
        "## Context",
        "",
        f"- Variant: `{metadata['options']['variant']}`",
        f"- Device: `{metadata.get('device', 'unknown')}`",
        f"- JAX: `{metadata['jax_version']}`",
        f"- Git commit: `{metadata['git'].get('commit')}`",
        f"- Git dirty: `{metadata['git'].get('dirty')}`",
        f"- Nx: `{metadata['options']['nx']}`",
        f"- Batch size: `{metadata['options']['batch_size']}`",
        f"- Block B: `{metadata['options']['block_b']}`",
        "",
        "## Phase Timings",
        "",
        "| phase | elapsed ms | rss delta MiB | output KiB | status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        rss = "" if row["rss_delta_mib"] is None else f"{float(row['rss_delta_mib']):.3f}"
        lines.append(
            f"| {row['phase']} | {float(row['elapsed_ms']):.3f} | "
            f"{rss} | {float(row['output_bytes']) / 1024.0:.1f} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `lower` is JAX lowering of the jitted Python wrapper.",
            "- `compile` is explicit `lowered.compile()` before any compiled call.",
            "- `compiled_first_call` is the first execution of the compiled object.",
            "- `compiled_call_2` and later rows show same-process cache reuse.",
        ]
    )
    if "error" in metadata:
        lines.extend(["", f"Error: `{metadata['error']}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script": "benchmark/analysis/jax_triton_cold_start_audit.py",
        "purpose": "Isolate jax-triton lower/compile/first-call cost.",
        "jax_version": jax.__version__,
        "python": host_platform.python_version(),
        "host": {
            "system": host_platform.system(),
            "release": host_platform.release(),
            "machine": host_platform.machine(),
            "processor": host_platform.processor(),
        },
        "git": _git_metadata(),
        "options": {
            "platform": args.platform,
            "variant": args.variant,
            "nx": int(args.nx),
            "batch_size": int(args.batch_size),
            "block_b": int(args.block_b),
            "compiled_calls": int(args.compiled_calls),
            "jax_cache_dir": None if args.jax_cache_dir is None else str(args.jax_cache_dir),
            "triton_cache_dir": (
                None if args.triton_cache_dir is None else str(args.triton_cache_dir)
            ),
            "clear_cache_dirs": bool(args.clear_cache_dirs),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "JAX_COMPILATION_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "XDG_CACHE_HOME",
                "CUDA_VERSION",
            )
        },
    }


def _git_metadata() -> dict[str, Any]:
    def run_git(*cmd: str) -> str | None:
        try:
            result = subprocess.run(
                ("git", *cmd),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status,
    }


def _cache_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    for path in _cache_paths(args):
        snapshots[str(path)] = _path_snapshot(path)
    return snapshots


def _cache_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in (args.jax_cache_dir, args.triton_cache_dir):
        if path is not None:
            paths.append(path)
    for env_name in ("JAX_COMPILATION_CACHE_DIR", "TRITON_CACHE_DIR"):
        value = os.environ.get(env_name)
        if value:
            paths.append(Path(value))
    home = Path.home()
    paths.extend((home / ".triton" / "cache", home / ".cache" / "triton", home / ".cache" / "jax"))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.expanduser())
        if key not in seen:
            seen.add(key)
            unique.append(path.expanduser())
    return tuple(unique)


def _path_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "files": 0, "dirs": 0, "bytes": 0}
    files = 0
    dirs = 0
    total_bytes = 0
    for item in path.rglob("*"):
        try:
            if item.is_dir():
                dirs += 1
            elif item.is_file():
                files += 1
                total_bytes += int(item.stat().st_size)
        except OSError:
            continue
    return {"exists": True, "files": files, "dirs": dirs, "bytes": total_bytes}


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")


def _block_until_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_block_until_ready(item) for item in value)
    block = getattr(value, "block_until_ready", None)
    if callable(block):
        block()
    return value


def _output_nbytes(value: Any) -> int:
    if isinstance(value, (tuple, list)):
        return sum(_output_nbytes(item) for item in value)
    nbytes = getattr(value, "nbytes", None)
    return int(nbytes or 0)


def _rss_mib() -> float | None:
    try:
        import psutil
    except Exception:
        return None
    return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)


def _delta(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return end - start


if __name__ == "__main__":
    raise SystemExit(main())
