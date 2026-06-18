"""Benchmark standalone Triton exact double-cable block solver candidates."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


DEFAULT_OUT_DIR = Path("benchmark/results/triton_solver")

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))


@dataclass(frozen=True)
class TritonCase:
    solver: str
    batch_size: int
    nx: int
    dtype: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--nx", type=int, nargs="+", default=[51, 96])
    parser.add_argument("--dtypes", nargs="+", choices=("float32",), default=["float32"])
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=(
            "triton_block_thomas",
            "triton_block_thomas_jax_bridge",
            "triton_pcr_soa",
        ),
        default=["triton_block_thomas"],
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    skip_reason = dependency_skip_reason()
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
        print(f"Triton benchmark skipped: {skip_reason}")
        return 2 if args.strict else 0

    rows: list[dict[str, Any]] = []
    for solver in args.solvers:
        for dtype in args.dtypes:
            for batch_size in args.batch_sizes:
                if int(batch_size) < 1:
                    raise ValueError("all batch sizes must be >= 1.")
                for nx in args.nx:
                    if int(nx) < 2:
                        raise ValueError("all Nx values must be >= 2 for this Triton spike.")
                    row = run_case(
                        TritonCase(
                            solver=str(solver),
                            batch_size=int(batch_size),
                            nx=int(nx),
                            dtype=dtype,
                        ),
                        warmups=int(args.warmups),
                        repeats=int(args.repeats),
                    )
                    rows.append(row)
                    print(format_row(row))

    write_outputs(run_root, rows=rows, parameters=vars(args))
    print(f"results: {run_root}")
    return 0


def dependency_skip_reason() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return "Python package 'torch' is not installed."
    if importlib.util.find_spec("triton") is None:
        return "Python package 'triton' is not installed."

    import torch

    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is false."
    return None


def run_case(case: TritonCase, *, warmups: int, repeats: int) -> dict[str, Any]:
    if case.solver == "triton_block_thomas_jax_bridge":
        return run_case_jax_bridge(case, warmups=warmups, repeats=repeats)

    import torch

    tensors = generate_system_torch(
        batch_size=case.batch_size,
        nx=case.nx,
        dtype=case.dtype,
        device=torch.device("cuda"),
    )
    solve = solver_function(case.solver)

    compile_start = time.perf_counter()
    out0, out1 = solve(*tensors)
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - compile_start) * 1e3

    for _ in range(warmups):
        out0, out1 = solve(*tensors)
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out0, out1 = solve(*tensors)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    max_residual, median_residual = residual_stats(*tensors, out0, out1)
    reference_error = dense_reference_error(*tensors, out0, out1)
    median_ms = float(statistics.median(times_ms))
    node_solves = int(case.batch_size) * int(case.nx)
    return {
        "solver": case.solver,
        "batch_size": case.batch_size,
        "nx": case.nx,
        "dtype": case.dtype,
        "compile_first_ms": compile_ms,
        "steady_min_ms": min(times_ms),
        "steady_median_ms": median_ms,
        "steady_p95_ms": percentile(times_ms, 95.0),
        "node_solves_per_s": node_solves / (median_ms * 1e-3),
        "max_abs_error_vs_dense64_smoke": reference_error,
        "max_block_residual_norm": max_residual,
        "median_block_residual_norm": median_residual,
    }


def run_case_jax_bridge(case: TritonCase, *, warmups: int, repeats: int) -> dict[str, Any]:
    import jax

    tensors = generate_system_jax(
        batch_size=case.batch_size,
        nx=case.nx,
        dtype=case.dtype,
    )

    compile_start = time.perf_counter()
    out0, out1 = solve_triton_block_thomas_jax_bridge(*tensors)
    out0.block_until_ready()
    out1.block_until_ready()
    compile_ms = (time.perf_counter() - compile_start) * 1e3

    for _ in range(warmups):
        out0, out1 = solve_triton_block_thomas_jax_bridge(*tensors)
        out0.block_until_ready()
        out1.block_until_ready()

    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        out0, out1 = solve_triton_block_thomas_jax_bridge(*tensors)
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
        "compile_first_ms": compile_ms,
        "steady_min_ms": min(times_ms),
        "steady_median_ms": median_ms,
        "steady_p95_ms": percentile(times_ms, 95.0),
        "node_solves_per_s": node_solves / (median_ms * 1e-3),
        "max_abs_error_vs_dense64_smoke": reference_error,
        "max_block_residual_norm": max_residual,
        "median_block_residual_norm": median_residual,
    }


def generate_system_torch(*, batch_size: int, nx: int, dtype: str, device):
    import torch

    torch_dtype = torch.float32 if dtype == "float32" else None
    if torch_dtype is None:
        raise ValueError(f"unsupported dtype: {dtype!r}.")

    batch = torch.arange(batch_size, device=device, dtype=torch.float64)[:, None]
    x = torch.arange(nx, device=device, dtype=torch.float64)[None, :]
    edge = torch.arange(nx - 1, device=device, dtype=torch.float64)[None, :]

    phase = 0.17 * x + 0.013 * batch
    a00 = 4.0 + 0.04 * (torch.remainder(x, 5.0)) + 0.0007 * torch.remainder(batch, 17.0)
    a11 = 5.0 + 0.03 * (torch.remainder(x, 7.0)) + 0.0005 * torch.remainder(batch, 19.0)
    a01 = -0.42 - 0.025 * torch.sin(phase)
    a10 = -0.36 + 0.020 * torch.cos(1.3 * phase)

    edge_phase = 0.11 * edge + 0.019 * batch
    off0 = -0.055 - 0.008 * torch.sin(edge_phase)
    off1 = -0.040 - 0.006 * torch.cos(1.7 * edge_phase)

    rhs0 = torch.sin(0.07 * x + 0.031 * batch)
    rhs1 = torch.cos(0.05 * x - 0.023 * batch)

    return tuple(
        tensor.to(dtype=torch_dtype).contiguous()
        for tensor in (a00, a01, a10, a11, off0, off1, rhs0, rhs1)
    )


def generate_system_jax(*, batch_size: int, nx: int, dtype: str):
    import jax.numpy as jnp

    jax_dtype = jnp.float32 if dtype == "float32" else None
    if jax_dtype is None:
        raise ValueError(f"unsupported dtype: {dtype!r}.")
    c = lambda value: jnp.asarray(value, dtype=jax_dtype)

    batch = jnp.arange(batch_size, dtype=jax_dtype)[:, None]
    x = jnp.arange(nx, dtype=jax_dtype)[None, :]
    edge = jnp.arange(nx - 1, dtype=jax_dtype)[None, :]

    phase = c(0.17) * x + c(0.013) * batch
    a00 = (
        c(4.0)
        + c(0.04) * jnp.remainder(x, c(5.0))
        + c(0.0007) * jnp.remainder(batch, c(17.0))
    )
    a11 = (
        c(5.0)
        + c(0.03) * jnp.remainder(x, c(7.0))
        + c(0.0005) * jnp.remainder(batch, c(19.0))
    )
    a01 = -c(0.42) - c(0.025) * jnp.sin(phase)
    a10 = -c(0.36) + c(0.020) * jnp.cos(c(1.3) * phase)

    edge_phase = c(0.11) * edge + c(0.019) * batch
    off0 = -c(0.055) - c(0.008) * jnp.sin(edge_phase)
    off1 = -c(0.040) - c(0.006) * jnp.cos(c(1.7) * edge_phase)

    rhs0 = jnp.sin(c(0.07) * x + c(0.031) * batch)
    rhs1 = jnp.cos(c(0.05) * x - c(0.023) * batch)

    return tuple(
        jnp.asarray(tensor, dtype=jax_dtype)
        for tensor in (a00, a01, a10, a11, off0, off1, rhs0, rhs1)
    )


def solver_function(solver: str):
    if solver == "triton_block_thomas":
        return solve_triton_block_thomas
    if solver == "triton_block_thomas_jax_bridge":
        return solve_triton_block_thomas_jax_bridge
    if solver == "triton_pcr_soa":
        return solve_triton_pcr_soa
    raise ValueError(f"unsupported Triton solver: {solver!r}.")


def solve_triton_block_thomas_jax_bridge(a00, a01, a10, a11, off0, off1, rhs0, rhs1):
    from axonscope.solvers.triton_thomas import solve_block_tridiagonal_2x2_triton_thomas_jax

    return solve_block_tridiagonal_2x2_triton_thomas_jax(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    )


def solve_triton_block_thomas(a00, a01, a10, a11, off0, off1, rhs0, rhs1):
    import torch

    from benchmark.triton_solver.triton_double_cable_kernels import (
        block_thomas_backward_kernel,
        block_thomas_forward_kernel,
    )

    batch_size, nx = rhs0.shape
    c00 = torch.empty_like(rhs0)
    c01 = torch.empty_like(rhs0)
    c10 = torch.empty_like(rhs0)
    c11 = torch.empty_like(rhs0)
    d0 = torch.empty_like(rhs0)
    d1 = torch.empty_like(rhs0)
    out0 = torch.empty_like(rhs0)
    out1 = torch.empty_like(rhs1)
    grid = (batch_size,)
    block_thomas_forward_kernel[grid](
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        c00,
        c01,
        c10,
        c11,
        d0,
        d1,
        N=nx,
        num_warps=1,
    )
    block_thomas_backward_kernel[grid](
        c00,
        c01,
        c10,
        c11,
        d0,
        d1,
        out0,
        out1,
        N=nx,
        num_warps=1,
    )
    return out0, out1


def solve_triton_pcr_soa(a00, a01, a10, a11, off0, off1, rhs0, rhs1):
    import torch
    import triton

    from benchmark.triton_solver.triton_double_cable_kernels import (
        pcr_soa_final_kernel,
        pcr_soa_init_kernel,
        pcr_soa_stage_kernel,
    )

    batch_size, nx = rhs0.shape
    total = int(batch_size) * int(nx)
    block_size = 128
    grid = (triton.cdiv(total, block_size),)

    lower00 = torch.empty_like(rhs0)
    lower01 = torch.empty_like(rhs0)
    lower10 = torch.empty_like(rhs0)
    lower11 = torch.empty_like(rhs0)
    upper00 = torch.empty_like(rhs0)
    upper01 = torch.empty_like(rhs0)
    upper10 = torch.empty_like(rhs0)
    upper11 = torch.empty_like(rhs0)
    diag00 = torch.empty_like(rhs0)
    diag01 = torch.empty_like(rhs0)
    diag10 = torch.empty_like(rhs0)
    diag11 = torch.empty_like(rhs0)
    r0 = torch.empty_like(rhs0)
    r1 = torch.empty_like(rhs0)

    next_lower00 = torch.empty_like(rhs0)
    next_lower01 = torch.empty_like(rhs0)
    next_lower10 = torch.empty_like(rhs0)
    next_lower11 = torch.empty_like(rhs0)
    next_upper00 = torch.empty_like(rhs0)
    next_upper01 = torch.empty_like(rhs0)
    next_upper10 = torch.empty_like(rhs0)
    next_upper11 = torch.empty_like(rhs0)
    next_diag00 = torch.empty_like(rhs0)
    next_diag01 = torch.empty_like(rhs0)
    next_diag10 = torch.empty_like(rhs0)
    next_diag11 = torch.empty_like(rhs0)
    next_r0 = torch.empty_like(rhs0)
    next_r1 = torch.empty_like(rhs0)
    out0 = torch.empty_like(rhs0)
    out1 = torch.empty_like(rhs1)

    pcr_soa_init_kernel[grid](
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        lower00,
        lower01,
        lower10,
        lower11,
        upper00,
        upper01,
        upper10,
        upper11,
        diag00,
        diag01,
        diag10,
        diag11,
        r0,
        r1,
        N=nx,
        TOTAL=total,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    stride = 1
    while stride < nx:
        pcr_soa_stage_kernel[grid](
            lower00,
            lower01,
            lower10,
            lower11,
            upper00,
            upper01,
            upper10,
            upper11,
            diag00,
            diag01,
            diag10,
            diag11,
            r0,
            r1,
            next_lower00,
            next_lower01,
            next_lower10,
            next_lower11,
            next_upper00,
            next_upper01,
            next_upper10,
            next_upper11,
            next_diag00,
            next_diag01,
            next_diag10,
            next_diag11,
            next_r0,
            next_r1,
            N=nx,
            TOTAL=total,
            STRIDE=stride,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        lower00, next_lower00 = next_lower00, lower00
        lower01, next_lower01 = next_lower01, lower01
        lower10, next_lower10 = next_lower10, lower10
        lower11, next_lower11 = next_lower11, lower11
        upper00, next_upper00 = next_upper00, upper00
        upper01, next_upper01 = next_upper01, upper01
        upper10, next_upper10 = next_upper10, upper10
        upper11, next_upper11 = next_upper11, upper11
        diag00, next_diag00 = next_diag00, diag00
        diag01, next_diag01 = next_diag01, diag01
        diag10, next_diag10 = next_diag10, diag10
        diag11, next_diag11 = next_diag11, diag11
        r0, next_r0 = next_r0, r0
        r1, next_r1 = next_r1, r1
        stride *= 2

    pcr_soa_final_kernel[grid](
        diag00,
        diag01,
        diag10,
        diag11,
        r0,
        r1,
        out0,
        out1,
        TOTAL=total,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out0, out1


def residual_stats(a00, a01, a10, a11, off0, off1, rhs0, rhs1, x0, x1):
    import torch

    res0 = a00 * x0 + a01 * x1 - rhs0
    res1 = a10 * x0 + a11 * x1 - rhs1
    res0[:, 1:] = res0[:, 1:] + off0 * x0[:, :-1]
    res1[:, 1:] = res1[:, 1:] + off1 * x1[:, :-1]
    res0[:, :-1] = res0[:, :-1] + off0 * x0[:, 1:]
    res1[:, :-1] = res1[:, :-1] + off1 * x1[:, 1:]
    norm = torch.maximum(torch.abs(res0), torch.abs(res1))
    return float(torch.max(norm).item()), float(torch.median(norm).item())


def residual_stats_jax(a00, a01, a10, a11, off0, off1, rhs0, rhs1, x0, x1):
    import jax
    import jax.numpy as jnp

    res0 = a00 * x0 + a01 * x1 - rhs0
    res1 = a10 * x0 + a11 * x1 - rhs1
    res0 = res0.at[:, 1:].add(off0 * x0[:, :-1])
    res1 = res1.at[:, 1:].add(off1 * x1[:, :-1])
    res0 = res0.at[:, :-1].add(off0 * x0[:, 1:])
    res1 = res1.at[:, :-1].add(off1 * x1[:, 1:])
    norm = jnp.maximum(jnp.abs(res0), jnp.abs(res1))
    return (
        float(jax.device_get(jnp.max(norm))),
        float(jax.device_get(jnp.median(norm))),
    )


def dense_reference_error(a00, a01, a10, a11, off0, off1, rhs0, rhs1, x0, x1) -> float:
    import numpy as np

    batch_limit = min(2, int(rhs0.shape[0]))
    nx = int(rhs0.shape[1])
    max_error = 0.0
    host = [tensor[:batch_limit].detach().cpu().numpy().astype(np.float64) for tensor in (
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        x0,
        x1,
    )]
    ha00, ha01, ha10, ha11, hoff0, hoff1, hrhs0, hrhs1, hx0, hx1 = host
    for batch in range(batch_limit):
        matrix = np.zeros((2 * nx, 2 * nx), dtype=np.float64)
        rhs = np.empty((2 * nx,), dtype=np.float64)
        for i in range(nx):
            row0 = 2 * i
            row1 = row0 + 1
            matrix[row0, row0] = ha00[batch, i]
            matrix[row0, row1] = ha01[batch, i]
            matrix[row1, row0] = ha10[batch, i]
            matrix[row1, row1] = ha11[batch, i]
            rhs[row0] = hrhs0[batch, i]
            rhs[row1] = hrhs1[batch, i]
            if i > 0:
                matrix[row0, row0 - 2] = hoff0[batch, i - 1]
                matrix[row1, row1 - 2] = hoff1[batch, i - 1]
            if i < nx - 1:
                matrix[row0, row0 + 2] = hoff0[batch, i]
                matrix[row1, row1 + 2] = hoff1[batch, i]
        ref = np.linalg.solve(matrix, rhs)
        got = np.empty_like(ref)
        got[0::2] = hx0[batch]
        got[1::2] = hx1[batch]
        max_error = max(max_error, float(np.max(np.abs(got - ref))))
    return max_error


def dense_reference_error_jax(a00, a01, a10, a11, off0, off1, rhs0, rhs1, x0, x1) -> float:
    import jax
    import numpy as np

    batch_limit = min(2, int(rhs0.shape[0]))
    nx = int(rhs0.shape[1])
    max_error = 0.0
    host = [
        np.asarray(jax.device_get(tensor[:batch_limit])).astype(np.float64)
        for tensor in (a00, a01, a10, a11, off0, off1, rhs0, rhs1, x0, x1)
    ]
    ha00, ha01, ha10, ha11, hoff0, hoff1, hrhs0, hrhs1, hx0, hx1 = host
    for batch in range(batch_limit):
        matrix = np.zeros((2 * nx, 2 * nx), dtype=np.float64)
        rhs = np.empty((2 * nx,), dtype=np.float64)
        for i in range(nx):
            row0 = 2 * i
            row1 = row0 + 1
            matrix[row0, row0] = ha00[batch, i]
            matrix[row0, row1] = ha01[batch, i]
            matrix[row1, row0] = ha10[batch, i]
            matrix[row1, row1] = ha11[batch, i]
            rhs[row0] = hrhs0[batch, i]
            rhs[row1] = hrhs1[batch, i]
            if i > 0:
                matrix[row0, row0 - 2] = hoff0[batch, i - 1]
                matrix[row1, row1 - 2] = hoff1[batch, i - 1]
            if i < nx - 1:
                matrix[row0, row0 + 2] = hoff0[batch, i]
                matrix[row1, row1 + 2] = hoff1[batch, i]
        ref = np.linalg.solve(matrix, rhs)
        got = np.empty_like(ref)
        got[0::2] = hx0[batch]
        got[1::2] = hx1[batch]
        max_error = max(max_error, float(np.max(np.abs(got - ref))))
    return max_error


def make_run_root(out_dir: Path, *, prefix: str | None) -> Path:
    stem = prefix or f"double_cable_triton_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values must be non-empty.")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q / 100.0
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = index - lo
    return float(ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction)


def format_row(row: dict[str, Any]) -> str:
    return (
        f"{row['solver']} B={row['batch_size']} Nx={row['nx']} {row['dtype']}: "
        f"median={row['steady_median_ms']:.3f} ms, "
        f"max_abs_err={row['max_abs_error_vs_dense64_smoke']:.3e}, "
        f"max_residual={row['max_block_residual_norm']:.3e}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
