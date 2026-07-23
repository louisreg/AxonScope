from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.workloads.curve_options import build_parser, dry_run
from benchmark.workloads.curve_runtime import run_recruitment_curves


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(
        "recruitment_curves",
        description="Recruitment-curve benchmark.",
    )
    parser.add_argument("--amplitude-count", type=int, default=None)
    args = parser.parse_args(argv)
    if args.dry_run:
        return dry_run("recruitment_curves", args)
    return run_recruitment_curves(args)


if __name__ == "__main__":
    raise SystemExit(main())
