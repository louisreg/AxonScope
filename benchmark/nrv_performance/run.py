from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

from benchmark.nrv_performance.suites import (
    NRV_PERFORMANCE_SUITES,
    NrvPerformanceSuite,
    suite_choices,
)


DEFAULT_OUT_DIR = Path("benchmark/results/nrv_performance/nrv_axonscope_grid")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run named AxonScope-vs-NRV performance benchmark suites.")
    parser.add_argument(
        "--suite",
        default="smoke",
        choices=suite_choices(),
        help="Named NRV performance suite to run.",
    )
    parser.add_argument("--list", action="store_true", help="List available suites and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded cases without running simulations.")
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

    suite = NRV_PERFORMANCE_SUITES[args.suite]
    runner_argv = suite_argv(
        suite,
        out_dir=args.out_dir,
        prefix=args.prefix,
        dry_run=bool(args.dry_run),
        extra_args=args.extra_args,
    )

    if suite.runner == "nrv_axonscope_grid":
        from benchmark.nrv_performance import nrv_axonscope_grid

        nrv_axonscope_grid.main(runner_argv)
        return

    raise ValueError(f"Unsupported NRV performance runner: {suite.runner}")


def print_suites() -> None:
    print("NRV performance suites:")
    for name, suite in NRV_PERFORMANCE_SUITES.items():
        print(f"  {name:28s} {suite.description}")


def suite_argv(
    suite: NrvPerformanceSuite,
    *,
    out_dir: Path | None = None,
    prefix: str | None = None,
    dry_run: bool = False,
    extra_args: Sequence[str] = (),
) -> list[str]:
    argv = list(suite.args)
    if out_dir is not None:
        argv.extend(["--out-dir", str(out_dir)])
    if prefix is not None:
        argv.extend(["--prefix", prefix])
    if dry_run:
        argv.append("--dry-run")
    forwarded = list(extra_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    argv.extend(forwarded)
    return argv


if __name__ == "__main__":
    main()
