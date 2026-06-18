"""Benchmark JAX FFI CUDA exact double-cable block solver candidates."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


DEFAULT_OUT_DIR = Path("benchmark/results/cuda_ffi_solver")

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

from benchmark.cuda_ffi_solver.cuda_ffi_thomas import (  # noqa: E402
    cuda_ffi_thomas_dependency_skip_reason,
    ensure_cuda_ffi_thomas_registered,
    solve_block_tridiagonal_2x2_cuda_ffi_thomas,
)
from benchmark.triton_solver.bench_double_cable_triton import (  # noqa: E402
    dense_reference_error_jax,
    format_row,
    generate_system_jax,
    percentile,
    residual_stats_jax,
)


@dataclass(frozen=True)
class CudaFfiCase:
    solver: str
    batch_size: int
    nx: int
    dtype: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--nx", type=int, nargs="+", default=[51, 96, 128])
    parser.add_argument("--dtypes", nargs="+", choices=("float32",), default=["float32"])
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=("cuda_ffi_block_thomas",),
        default=["cuda_ffi_block_thomas"],
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)

    skip_reason = cuda_ffi_thomas_dependency_skip_reason()
    run_root = make_run_root(args.out_dir, prefix=args.prefix)
    if skip_reason is not None:
        write_summary_csv(run_root / "summary.csv", [])
        (run_root / "summary.json").write_text("[]\n", encoding="utf-8")
        write_manifest(
            run_root,
            rows=[],
            parameters=vars(args),
            status="skipped",
            reason=skip_reason,
        )
        print(f"CUDA FFI benchmark skipped: {skip_reason}")
        return 2 if args.strict else 0

    build_start = time.perf_counter()
    library_path = ensure_cuda_ffi_thomas_registered(force_rebuild=args.force_rebuild)
    build_ms = (time.perf_counter() - build_start) * 1e3
    print(f"CUDA FFI library: {library_path}")
    print(f"CUDA FFI build/load/register: {build_ms:.3f} ms")

    rows: list[dict[str, Any]] = []
    for solver in args.solvers:
        for dtype in args.dtypes:
            for batch_size in args.batch_sizes:
                if int(batch_size) < 1:
                    raise ValueError("all batch sizes must be >= 1.")
                for nx in args.nx:
                    if int(nx) < 2:
                        raise ValueError("all Nx values must be >= 2 for this CUDA FFI spike.")
                    row = run_case(
                        CudaFfiCase(
                            solver=str(solver),
                            batch_size=int(batch_size),
                            nx=int(nx),
                            dtype=dtype,
                        ),
                        warmups=int(args.warmups),
                        repeats=int(args.repeats),
                        ffi_build_ms=build_ms,
                    )
                    rows.append(row)
                    print(format_row(row))

    write_outputs(run_root, rows=rows, parameters=vars(args))
    print(f"results: {run_root}")
    return 0


def run_case(
    case: CudaFfiCase,
    *,
    warmups: int,
    repeats: int,
    ffi_build_ms: float,
) -> dict[str, Any]:
    import jax

    tensors = generate_system_jax(
        batch_size=case.batch_size,
        nx=case.nx,
        dtype=case.dtype,
    )
    solve = jax.jit(solver_function(case.solver))

    compile_start = time.perf_counter()
    out0, out1 = solve(*tensors)
    out0.block_until_ready()
    out1.block_until_ready()
    compile_ms = (time.perf_counter() - compile_start) * 1e3

    for _ in range(warmups):
        out0, out1 = solve(*tensors)
        out0.block_until_ready()
        out1.block_until_ready()

    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        out0, out1 = solve(*tensors)
        out0.block_until_ready()
        out1.block_until_ready()
        times_ms.append((time.perf_counter() - start) * 1e3)

    max_residual, median_residual = residual_stats_jax(*tensors, out0, out1)
    reference_error = dense_reference_error_jax(*tensors, out0, out1)
    median_ms = float(statistics.median(times_ms))
    node_solves = int(case.batch_size) * int(case.nx)
    return {
        "solver": case.solver,
        "batch_size": case.batch_size,
        "nx": case.nx,
        "dtype": case.dtype,
        "ffi_build_ms": ffi_build_ms,
        "compile_first_ms": compile_ms,
        "steady_min_ms": min(times_ms),
        "steady_median_ms": median_ms,
        "steady_p95_ms": percentile(times_ms, 95.0),
        "node_solves_per_s": node_solves / (median_ms * 1e-3),
        "max_abs_error_vs_dense64_smoke": reference_error,
        "max_block_residual_norm": max_residual,
        "median_block_residual_norm": median_residual,
    }


def solver_function(solver: str):
    if solver == "cuda_ffi_block_thomas":
        return solve_block_tridiagonal_2x2_cuda_ffi_thomas
    raise ValueError(f"unsupported CUDA FFI solver: {solver!r}.")


def make_run_root(out_dir: Path, *, prefix: str | None) -> Path:
    stem = prefix or f"double_cable_cuda_ffi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = out_dir / stem
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_outputs(run_root: Path, *, rows: Sequence[dict[str, Any]], parameters: dict[str, Any]):
    write_summary_csv(run_root / "summary.csv", rows)
    (run_root / "summary.json").write_text(
        json.dumps(list(rows), indent=2, sort_keys=True), encoding="utf-8"
    )
    write_manifest(run_root, rows=rows, parameters=parameters, status="complete", reason=None)


def write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = (
        "solver",
        "batch_size",
        "nx",
        "dtype",
        "ffi_build_ms",
        "compile_first_ms",
        "steady_min_ms",
        "steady_median_ms",
        "steady_p95_ms",
        "node_solves_per_s",
        "max_abs_error_vs_dense64_smoke",
        "max_block_residual_norm",
        "median_block_residual_norm",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_manifest(
    run_root: Path,
    *,
    rows: Sequence[dict[str, Any]],
    parameters: dict[str, Any],
    status: str,
    reason: str | None,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "reason": reason,
        "parameters": _jsonable_parameters(parameters),
        "row_count": len(rows),
        "summary_csv": str(run_root / "summary.csv"),
        "summary_json": str(run_root / "summary.json"),
    }
    (run_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _jsonable_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in parameters.items()}


if __name__ == "__main__":
    raise SystemExit(main())

