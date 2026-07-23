from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.workloads.curve_options import build_parser, dry_run
from benchmark.workloads.curve_runtime import run_threshold_curves


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(
        "threshold_curves",
        description="Activation-threshold curve benchmark.",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        return dry_run("threshold_curves", args)
    return run_threshold_curves(args)


if __name__ == "__main__":
    raise SystemExit(main())
