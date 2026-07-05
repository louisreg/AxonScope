"""Capture a JAX profiler trace for one exact double-cable linear solver case."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

from benchmark.solvers.bench_double_cable_linear_solvers import main as run_benchmark


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="pcr_adaptive")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark/results/solvers"))
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument(
        "--create-perfetto",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)

    run_id = (
        f"profile_double_cable_linear_{args.solver}"
        f"_B{args.batch_size}_Nx{args.nx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    trace_dir = args.trace_dir or args.out_dir / run_id / "jax_traces"
    benchmark_args = [
        "--batch-sizes",
        str(args.batch_size),
        "--nx",
        str(args.nx),
        "--dtypes",
        args.dtype,
        "--solvers",
        args.solver,
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--out-dir",
        str(args.out_dir),
        "--prefix",
        run_id,
        "--jax-trace",
        "--jax-trace-dir",
        str(trace_dir),
    ]
    if args.create_perfetto:
        benchmark_args.append("--jax-trace-create-perfetto")
    else:
        benchmark_args.append("--no-jax-trace-create-perfetto")
    run_benchmark(benchmark_args)


if __name__ == "__main__":
    main()

