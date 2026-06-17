"""Benchmark exact double-cable 2x2 block-tridiagonal linear solvers.

This runner isolates the linear solve used inside each implicit double-cable
time step. It deliberately avoids building axon models, stimulation contexts,
dispatch plans, or public results so solver throughput can be compared without
packaging noise.

Examples:
    python benchmark/solvers/bench_double_cable_linear_solvers.py --dry-run

    python benchmark/solvers/bench_double_cable_linear_solvers.py \
      --batch-sizes 128 512 1024 \
      --nx 32 51 64 \
      --solvers thomas pcr pcr_soa split_jacobi_4 split_gs_4 pcr_adaptive \
      --dtypes float32 \
      --warmups 1 \
      --repeats 5
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

from axonscope.solvers import resolve_double_cable_block_solver
from axonscope.solvers.common import (
    double_cable_block_residual_norm,
    solve_double_cable_split_gauss_seidel_batched,
    solve_double_cable_split_jacobi_batched,
    solve_double_cable_split_jacobi_then_gauss_seidel_batched,
    solve_double_cable_split_richardson_batched,
    solve_block_tridiagonal_2x2_assoc_backward_batched,
    solve_block_tridiagonal_2x2_assoc_transfer_dense_batched,
    solve_block_tridiagonal_2x2_pcr,
    solve_block_tridiagonal_2x2_pcr_soa,
    solve_block_tridiagonal_2x2_pcr_soa_batched,
    solve_block_tridiagonal_2x2_pcr_soa_hybrid_batched,
    solve_block_tridiagonal_2x2_pcr_soa_batched_padded,
    solve_block_tridiagonal_2x2_pcr_soa_batched_transposed,
    solve_block_tridiagonal_2x2_scalar_batched,
    solve_block_tridiagonal_2x2_scalar,
)
from axonscope.solvers.pallas_kernels import solve_block_tridiagonal_2x2_pallas_thomas_batched


DEFAULT_OUT_DIR = Path("benchmark/results/solvers")
SOLVER_CHOICES = (
    "auto",
    "thomas",
    "thomas_batched",
    "pcr",
    "pcr_soa",
    "pcr_soa_hybrid_4",
    "pcr_soa_hybrid_8",
    "pcr_soa_hybrid_16",
    "pcr_soa_transposed",
    "pcr_soa_padded",
    "pcr_adaptive",
    "assoc_backward",
    "assoc_transfer_dense",
    "pallas_thomas_128",
    "split_jacobi_4",
    "split_jacobi_8",
    "split_jacobi4_gs1",
    "split_gs_2",
    "split_gs_3",
    "split_gs_4",
    "split_gs_8",
    "split_richardson_4",
)
KERNEL_SOLVERS = (
    "thomas",
    "thomas_batched",
    "pcr",
    "pcr_soa",
    "pcr_soa_hybrid_4",
    "pcr_soa_hybrid_8",
    "pcr_soa_hybrid_16",
    "pcr_soa_transposed",
    "pcr_soa_padded",
    "assoc_backward",
    "assoc_transfer_dense",
    "pallas_thomas_128",
    "split_jacobi_4",
    "split_jacobi_8",
    "split_jacobi4_gs1",
    "split_gs_2",
    "split_gs_3",
    "split_gs_4",
    "split_gs_8",
    "split_richardson_4",
)
BENCHMARK_ONLY_SOLVER_RESOLUTIONS = {
    "thomas_batched": "thomas",
    "assoc_backward": "thomas",
    "assoc_transfer_dense": "thomas",
    "pallas_thomas_128": "thomas",
    "pcr_soa_hybrid_4": "pcr_soa",
    "pcr_soa_hybrid_8": "pcr_soa",
    "pcr_soa_hybrid_16": "pcr_soa",
    "pcr_soa_transposed": "pcr_soa",
    "pcr_soa_padded": "pcr_soa",
    "split_jacobi_4": "split_iterative",
    "split_jacobi_8": "split_iterative",
    "split_jacobi4_gs1": "split_iterative",
    "split_gs_2": "split_iterative",
    "split_gs_3": "split_iterative",
    "split_gs_4": "split_iterative",
    "split_gs_8": "split_iterative",
    "split_richardson_4": "split_iterative",
}
PCR_SOA_MAX_BATCH = 4096


@dataclass(frozen=True)
class LinearSolverCase:
    """One solver benchmark case."""

    batch_size: int
    nx: int
    dtype: str
    requested_solver: str
    resolved_solver: str
    kernel_solver: str

    @property
    def label(self) -> str:
        return (
            f"{self.requested_solver}_as_{self.kernel_solver}"
            f"_B{self.batch_size}_Nx{self.nx}_{self.dtype}"
        )


def generate_system(
    *,
    batch_size: int,
    nx: int,
    dtype: str,
) -> tuple[jax.Array, ...]:
    """Return a deterministic well-conditioned 2x2 block-tridiagonal system."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if nx < 1:
        raise ValueError("nx must be >= 1.")
    np_dtype = _numpy_dtype(dtype)

    batch = np.arange(batch_size, dtype=np.float64)[:, None]
    x = np.arange(nx, dtype=np.float64)[None, :]
    edge = np.arange(max(nx - 1, 0), dtype=np.float64)[None, :]

    phase = 0.17 * x + 0.013 * batch
    a00 = 4.0 + 0.04 * (x % 5.0) + 0.0007 * (batch % 17.0)
    a11 = 5.0 + 0.03 * (x % 7.0) + 0.0005 * (batch % 19.0)
    a01 = -0.42 - 0.025 * np.sin(phase)
    a10 = -0.36 + 0.020 * np.cos(1.3 * phase)

    edge_phase = 0.11 * edge + 0.019 * batch
    off0 = -0.055 - 0.008 * np.sin(edge_phase)
    off1 = -0.040 - 0.006 * np.cos(1.7 * edge_phase)

    rhs0 = np.sin(0.07 * x + 0.031 * batch)
    rhs1 = np.cos(0.05 * x - 0.023 * batch)

    arrays = (a00, a01, a10, a11, off0, off1, rhs0, rhs1)
    return tuple(jnp.asarray(array.astype(np_dtype)) for array in arrays)


def planned_cases(
    *,
    batch_sizes: Sequence[int],
    nx_values: Sequence[int],
    dtypes: Sequence[str],
    solvers: Sequence[str],
    platform: str,
) -> tuple[LinearSolverCase, ...]:
    """Expand CLI dimensions into concrete cases."""

    cases: list[LinearSolverCase] = []
    for dtype in dtypes:
        _numpy_dtype(dtype)
        for batch_size in batch_sizes:
            if int(batch_size) < 1:
                raise ValueError("all batch sizes must be >= 1.")
            for nx in nx_values:
                if int(nx) < 1:
                    raise ValueError("all Nx values must be >= 1.")
                for solver in solvers:
                    if solver not in SOLVER_CHOICES:
                        raise ValueError(f"unknown solver choice: {solver!r}.")
                    if solver in BENCHMARK_ONLY_SOLVER_RESOLUTIONS:
                        resolved = BENCHMARK_ONLY_SOLVER_RESOLUTIONS[solver]
                        kernel_solver = solver
                    else:
                        resolved = resolve_double_cable_block_solver(
                            solver,
                            platform=platform,
                        )
                        kernel_solver = resolve_kernel_solver(
                            resolved,
                            batch_size=int(batch_size),
                        )
                    cases.append(
                        LinearSolverCase(
                            batch_size=int(batch_size),
                            nx=int(nx),
                            dtype=dtype,
                            requested_solver=solver,
                            resolved_solver=resolved,
                            kernel_solver=kernel_solver,
                        )
                    )
    return tuple(cases)


def resolve_kernel_solver(solver: str, *, batch_size: int) -> str:
    """Resolve adaptive benchmark choices to concrete low-level functions."""

    if solver == "pcr_adaptive":
        return "pcr_soa" if int(batch_size) <= PCR_SOA_MAX_BATCH else "pcr"
    if solver in KERNEL_SOLVERS:
        return solver
    raise ValueError(f"unsupported resolved solver: {solver!r}.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 128, 512, 1024, 2048, 4096],
        help="Leading batch sizes B.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        nargs="+",
        default=[16, 32, 51, 64, 96, 100, 128],
        help="Compartment counts Nx.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("float32", "float64"),
        default=["float32"],
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=SOLVER_CHOICES,
        default=["thomas", "pcr", "pcr_soa", "pcr_adaptive"],
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip Thomas float64 reference solves and leave error columns empty.",
    )
    parser.add_argument(
        "--jax-trace",
        action="store_true",
        help="Trace the first measured run for each case with jax.profiler.",
    )
    parser.add_argument(
        "--jax-trace-dir",
        type=Path,
        default=None,
        help="Trace root. Defaults to <run-root>/jax_traces when tracing.",
    )
    parser.add_argument(
        "--jax-trace-create-perfetto",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(argv)

    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0.")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    if not args.skip_reference or "float64" in args.dtypes:
        jax.config.update("jax_enable_x64", True)

    platform = jax.default_backend()
    cases = planned_cases(
        batch_sizes=args.batch_sizes,
        nx_values=args.nx,
        dtypes=args.dtypes,
        solvers=args.solvers,
        platform=platform,
    )

    if args.dry_run:
        for case in cases:
            print(
                f"{case.requested_solver} -> {case.resolved_solver}"
                f" -> {case.kernel_solver}"
                f" B={case.batch_size} Nx={case.nx} dtype={case.dtype}"
            )
        return

    run_root = _make_run_root(args.out_dir, prefix=args.prefix)
    trace_root = None
    if args.jax_trace or args.jax_trace_dir is not None:
        trace_root = args.jax_trace_dir or run_root / "jax_traces"
        trace_root.mkdir(parents=True, exist_ok=True)

    reference_cache: dict[tuple[int, int], jax.Array] = {}
    rows: list[dict[str, Any]] = []
    for case in cases:
        reference = None
        if not args.skip_reference:
            reference = reference_cache.get((case.batch_size, case.nx))
            if reference is None:
                reference = _compute_reference(case.batch_size, case.nx)
                reference_cache[(case.batch_size, case.nx)] = reference

        row = run_case(
            case,
            warmups=args.warmups,
            repeats=args.repeats,
            reference=reference,
            trace_root=trace_root,
            create_perfetto_trace=bool(args.jax_trace_create_perfetto),
        )
        rows.append(row)
        print(_format_row(row))

    _write_outputs(
        run_root,
        rows=rows,
        parameters={
            "batch_sizes": [int(value) for value in args.batch_sizes],
            "nx": [int(value) for value in args.nx],
            "dtypes": list(args.dtypes),
            "solvers": list(args.solvers),
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
            "skip_reference": bool(args.skip_reference),
            "jax_trace": trace_root is not None,
            "jax_trace_dir": None if trace_root is None else str(trace_root),
            "jax_trace_create_perfetto": bool(args.jax_trace_create_perfetto),
        },
        platform=platform,
    )
    print(f"results: {run_root}")


def run_case(
    case: LinearSolverCase,
    *,
    warmups: int,
    repeats: int,
    reference: jax.Array | None,
    trace_root: Path | None,
    create_perfetto_trace: bool,
) -> dict[str, Any]:
    """Compile, run, time, and summarize one linear-solver case."""

    args = generate_system(
        batch_size=case.batch_size,
        nx=case.nx,
        dtype=case.dtype,
    )
    solve = _make_batched_solver(case.kernel_solver)

    compile_start = time.perf_counter()
    compiled = solve.lower(*args).compile()
    compile_seconds = time.perf_counter() - compile_start

    first_start = time.perf_counter()
    first_output = _block_until_ready(compiled(*args))
    first_seconds = time.perf_counter() - first_start

    for _ in range(int(warmups)):
        _block_until_ready(compiled(*args))

    trace_dir = None if trace_root is None else trace_root / case.label
    run_times: list[float] = []
    last_output = first_output
    for repeat_index in range(int(repeats)):
        with _maybe_trace(
            trace_dir if repeat_index == 0 else None,
            create_perfetto_trace=create_perfetto_trace,
        ):
            start = time.perf_counter()
            last_output = _block_until_ready(compiled(*args))
            run_times.append(time.perf_counter() - start)

    max_abs_error = None
    max_rel_error = None
    if reference is not None:
        candidate = np.asarray(last_output, dtype=np.float64)
        ref = np.asarray(reference, dtype=np.float64)
        delta = np.abs(candidate - ref)
        max_abs_error = float(np.max(delta))
        denominator = np.maximum(np.abs(ref), 1e-12)
        max_rel_error = float(np.max(delta / denominator))

    residual = _block_until_ready(
        double_cable_block_residual_norm(
            *args,
            last_output[..., 0],
            last_output[..., 1],
        )
    )
    residual_np = np.asarray(residual, dtype=np.float64)

    median_seconds = float(statistics.median(run_times))
    p95_seconds = float(np.percentile(np.asarray(run_times), 95.0))
    node_solves = int(case.batch_size) * int(case.nx)
    return {
        "requested_solver": case.requested_solver,
        "resolved_solver": case.resolved_solver,
        "kernel_solver": case.kernel_solver,
        "batch_size": int(case.batch_size),
        "nx": int(case.nx),
        "dtype": case.dtype,
        "compile_ms": compile_seconds * 1e3,
        "first_run_ms": first_seconds * 1e3,
        "steady_min_ms": min(run_times) * 1e3,
        "steady_median_ms": median_seconds * 1e3,
        "steady_p95_ms": p95_seconds * 1e3,
        "node_solves_per_s": node_solves / median_seconds,
        "max_abs_error_vs_thomas64": max_abs_error,
        "max_rel_error_vs_thomas64": max_rel_error,
        "max_block_residual_norm": float(np.max(residual_np)),
        "median_block_residual_norm": float(np.median(residual_np)),
        "trace_dir": None if trace_dir is None else str(trace_dir),
    }


def _compute_reference(batch_size: int, nx: int) -> jax.Array:
    reference_args = generate_system(
        batch_size=batch_size,
        nx=nx,
        dtype="float64",
    )
    solve = _make_batched_solver("thomas")
    return _block_until_ready(solve.lower(*reference_args).compile()(*reference_args))


def _make_batched_solver(kernel_solver: str):
    if kernel_solver in {
        "split_jacobi_4",
        "split_jacobi_8",
        "split_jacobi4_gs1",
        "split_gs_2",
        "split_gs_3",
        "split_gs_4",
        "split_gs_8",
        "split_richardson_4",
    }:
        split_solve, split_kwargs = {
            "split_jacobi_4": (
                solve_double_cable_split_jacobi_batched,
                {"iterations": 4, "init": "rhs_guess"},
            ),
            "split_jacobi_8": (
                solve_double_cable_split_jacobi_batched,
                {"iterations": 8, "init": "rhs_guess"},
            ),
            "split_jacobi4_gs1": (
                solve_double_cable_split_jacobi_then_gauss_seidel_batched,
                {
                    "jacobi_iterations": 4,
                    "gauss_seidel_iterations": 1,
                    "init": "rhs_guess",
                },
            ),
            "split_gs_2": (
                solve_double_cable_split_gauss_seidel_batched,
                {"iterations": 2, "init": "rhs_guess"},
            ),
            "split_gs_3": (
                solve_double_cable_split_gauss_seidel_batched,
                {"iterations": 3, "init": "rhs_guess"},
            ),
            "split_gs_4": (
                solve_double_cable_split_gauss_seidel_batched,
                {"iterations": 4, "init": "rhs_guess"},
            ),
            "split_gs_8": (
                solve_double_cable_split_gauss_seidel_batched,
                {"iterations": 8, "init": "rhs_guess"},
            ),
            "split_richardson_4": (
                solve_double_cable_split_richardson_batched,
                {"iterations": 4, "relaxation": 0.75, "init": "rhs_guess"},
            ),
        }[kernel_solver]

        @jax.jit
        def solve_split(
            a00,
            a01,
            a10,
            a11,
            off0,
            off1,
            rhs0,
            rhs1,
        ):
            x0, x1 = split_solve(
                a00,
                a01,
                a10,
                a11,
                off0,
                off1,
                rhs0,
                rhs1,
                **split_kwargs,
            )
            return jnp.stack((x0, x1), axis=-1)

        return solve_split

    if kernel_solver == "thomas_batched":

        @jax.jit
        def solve_batch_native_thomas(
            a00,
            a01,
            a10,
            a11,
            off0,
            off1,
            rhs0,
            rhs1,
        ):
            x0, x1 = solve_block_tridiagonal_2x2_scalar_batched(
                a00,
                a01,
                a10,
                a11,
                off0,
                off1,
                rhs0,
                rhs1,
            )
            return jnp.stack((x0, x1), axis=-1)

        return solve_batch_native_thomas

    if kernel_solver == "assoc_backward":

        @jax.jit
        def solve_assoc_backward(
            a00,
            a01,
            a10,
            a11,
            off0,
            off1,
            rhs0,
            rhs1,
        ):
            x0, x1 = solve_block_tridiagonal_2x2_assoc_backward_batched(
                a00,
                a01,
                a10,
                a11,
                off0,
                off1,
                rhs0,
                rhs1,
            )
            return jnp.stack((x0, x1), axis=-1)

        return solve_assoc_backward

    if kernel_solver == "assoc_transfer_dense":

        @jax.jit
        def solve_assoc_transfer_dense(
            a00,
            a01,
            a10,
            a11,
            off0,
            off1,
            rhs0,
            rhs1,
        ):
            x0, x1 = solve_block_tridiagonal_2x2_assoc_transfer_dense_batched(
                a00,
                a01,
                a10,
                a11,
                off0,
                off1,
                rhs0,
                rhs1,
            )
            return jnp.stack((x0, x1), axis=-1)

        return solve_assoc_transfer_dense

    if kernel_solver == "pallas_thomas_128":

        @jax.jit
        def solve_pallas_thomas_128(
            a00,
            a01,
            a10,
            a11,
            off0,
            off1,
            rhs0,
            rhs1,
        ):
            x0, x1 = solve_block_tridiagonal_2x2_pallas_thomas_batched(
                a00,
                a01,
                a10,
                a11,
                off0,
                off1,
                rhs0,
                rhs1,
                block_b=128,
            )
            return jnp.stack((x0, x1), axis=-1)

        return solve_pallas_thomas_128

    if kernel_solver in {
        "pcr_soa",
        "pcr_soa_hybrid_4",
        "pcr_soa_hybrid_8",
        "pcr_soa_hybrid_16",
        "pcr_soa_transposed",
        "pcr_soa_padded",
    }:
        solve_pcr_soa_batch = {
            "pcr_soa": solve_block_tridiagonal_2x2_pcr_soa_batched,
            "pcr_soa_hybrid_4": lambda *args: solve_block_tridiagonal_2x2_pcr_soa_hybrid_batched(
                *args,
                chain_stride=4,
            ),
            "pcr_soa_hybrid_8": lambda *args: solve_block_tridiagonal_2x2_pcr_soa_hybrid_batched(
                *args,
                chain_stride=8,
            ),
            "pcr_soa_hybrid_16": lambda *args: solve_block_tridiagonal_2x2_pcr_soa_hybrid_batched(
                *args,
                chain_stride=16,
            ),
            "pcr_soa_transposed": solve_block_tridiagonal_2x2_pcr_soa_batched_transposed,
            "pcr_soa_padded": solve_block_tridiagonal_2x2_pcr_soa_batched_padded,
        }[kernel_solver]

        @jax.jit
        def solve_batch_native(
            a00,
            a01,
            a10,
            a11,
            off0,
            off1,
            rhs0,
            rhs1,
        ):
            x0, x1 = solve_pcr_soa_batch(
                a00,
                a01,
                a10,
                a11,
                off0,
                off1,
                rhs0,
                rhs1,
            )
            return jnp.stack((x0, x1), axis=-1)

        return solve_batch_native

    solve_one = _solve_fn(kernel_solver)

    @jax.jit
    def solve_batch(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    ):
        x0, x1 = jax.vmap(solve_one)(a00, a01, a10, a11, off0, off1, rhs0, rhs1)
        return jnp.stack((x0, x1), axis=-1)

    return solve_batch


def _solve_fn(kernel_solver: str):
    if kernel_solver == "thomas":
        return solve_block_tridiagonal_2x2_scalar
    if kernel_solver == "pcr":
        return solve_block_tridiagonal_2x2_pcr
    if kernel_solver == "pcr_soa":
        return solve_block_tridiagonal_2x2_pcr_soa
    raise ValueError(f"unknown kernel solver: {kernel_solver!r}.")


def _block_until_ready(value: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
        value,
    )


@contextmanager
def _maybe_trace(
    trace_dir: Path | None,
    *,
    create_perfetto_trace: bool,
) -> Iterator[None]:
    if trace_dir is None:
        with nullcontext():
            yield
        return

    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(
        str(trace_dir),
        create_perfetto_trace=bool(create_perfetto_trace),
    ):
        with jax.profiler.StepTraceAnnotation("double_cable_linear_solve"):
            yield


def _make_run_root(out_dir: Path, *, prefix: str | None) -> Path:
    stem = prefix or f"double_cable_linear_solvers_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = out_dir / stem
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_outputs(
    run_root: Path,
    *,
    rows: Sequence[dict[str, Any]],
    parameters: dict[str, Any],
    platform: str,
) -> None:
    csv_path = run_root / "summary.csv"
    json_path = run_root / "summary.json"
    manifest_path = run_root / "manifest.json"

    fieldnames = (
        "requested_solver",
        "resolved_solver",
        "kernel_solver",
        "batch_size",
        "nx",
        "dtype",
        "compile_ms",
        "first_run_ms",
        "steady_min_ms",
        "steady_median_ms",
        "steady_p95_ms",
        "node_solves_per_s",
        "max_abs_error_vs_thomas64",
        "max_rel_error_vs_thomas64",
        "max_block_residual_norm",
        "median_block_residual_norm",
        "trace_dir",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})

    json_path.write_text(json.dumps(list(rows), indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "platform": platform,
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "parameters": parameters,
                "summary_csv": str(csv_path),
                "summary_json": str(json_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _format_row(row: dict[str, Any]) -> str:
    err = row["max_abs_error_vs_thomas64"]
    err_text = "n/a" if err is None else f"{err:.3e}"
    residual_text = f"{row['max_block_residual_norm']:.3e}"
    return (
        f"{row['requested_solver']}({row['kernel_solver']}) "
        f"B={row['batch_size']} Nx={row['nx']} {row['dtype']}: "
        f"median={row['steady_median_ms']:.3f} ms, "
        f"nodes/s={row['node_solves_per_s']:.3e}, "
        f"max_abs_err={err_text}, "
        f"max_residual={residual_text}"
    )


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def _numpy_dtype(dtype: str) -> Any:
    if dtype == "float32":
        return np.float32
    if dtype == "float64":
        return np.float64
    raise ValueError(f"unsupported dtype: {dtype!r}.")


if __name__ == "__main__":
    main()
