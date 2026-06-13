from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

from benchmark.runtime.suites import RUNTIME_SUITES, RuntimeSuite


DEFAULT_OUT_DIR = Path("benchmark/results/runtime")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run named AxonScope runtime benchmark suites.")
    parser.add_argument(
        "--suite",
        default="smoke",
        choices=tuple(RUNTIME_SUITES),
        help="Named runtime suite to run.",
    )
    parser.add_argument("--list", action="store_true", help="List available suites and exit.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for JSON/CSV outputs.")
    parser.add_argument("--prefix", default=None, help="Optional output filename prefix.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to the suite runner. Use '--' before forwarded options.",
    )
    args = parser.parse_args(argv)

    if args.list:
        print_suites()
        return

    suite = RUNTIME_SUITES[args.suite]
    runner_argv = suite_argv(
        suite,
        out_dir=args.out_dir,
        prefix=args.prefix,
        extra_args=args.extra_args,
    )

    if suite.runner == "benchmark_solver":
        from benchmark.runtime import benchmark_solver

        benchmark_solver.main(runner_argv)
        return
    if suite.runner == "benchmark_vstim_batch":
        from benchmark.runtime import benchmark_vstim_batch

        benchmark_vstim_batch.main(runner_argv)
        return
    if suite.runner == "benchmark_double_cable_batch":
        from benchmark.runtime import benchmark_double_cable_batch

        benchmark_double_cable_batch.main(runner_argv)
        return
    if suite.runner == "pool_memory":
        from benchmark.runtime import pool_memory

        pool_memory.main(runner_argv)
        return

    raise ValueError(f"Unsupported runtime runner: {suite.runner}")


def print_suites() -> None:
    print("Runtime suites:")
    for name, suite in RUNTIME_SUITES.items():
        print(f"  {name:12s} {suite.description}")


def suite_argv(
    suite: RuntimeSuite,
    *,
    out_dir: Path | None = None,
    prefix: str | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    argv = list(suite.args)
    if out_dir is not None:
        argv.extend(["--out-dir", str(out_dir)])
    if prefix is not None:
        argv.extend(["--prefix", prefix])
    forwarded = list(extra_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    argv.extend(forwarded)
    return argv


if __name__ == "__main__":
    main()
