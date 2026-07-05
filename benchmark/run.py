from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.curves import recruitment_curves, threshold_curves
from benchmark.workloads.curve_options import PRESETS


SCRIPTS = {
    "threshold_curves": threshold_curves,
    "recruitment_curves": recruitment_curves,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AxonScope benchmark scripts.")
    parser.add_argument("--list", action="store_true", help="List scripts and presets.")
    parser.add_argument("--script", choices=tuple(SCRIPTS), help="Benchmark script to run.")
    parser.add_argument("--preset", default="quick", choices=tuple(PRESETS))
    parser.add_argument("--platform", choices=("cpu", "gpu", "nrv"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--memory-trace", choices=("off", "rss", "tracemalloc", "device", "all"))
    parser.add_argument("--memory-top-n", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-backend", choices=("auto", "jax", "none"))
    parser.add_argument("--profile-output")
    parser.add_argument("--profile-create-perfetto", action="store_true")
    parser.add_argument("--jax-device-memory-profile", action="store_true")
    parser.add_argument("--case-filter")
    args, extra = parser.parse_known_args(argv)

    if args.list:
        print("scripts:")
        for name in sorted(SCRIPTS):
            print(f"  {name}")
        print("presets:")
        for name in PRESETS:
            print(f"  {name}")
        return 0

    if args.script is None:
        parser.error("--script is required unless --list is used")

    script_args = ["--preset", args.preset]
    for flag in (
        "platform",
        "output",
        "memory_trace",
        "memory_top_n",
        "profile_backend",
        "profile_output",
        "case_filter",
    ):
        value = getattr(args, flag)
        if value is not None:
            script_args.extend((f"--{flag.replace('_', '-')}", str(value)))
    if args.dry_run:
        script_args.append("--dry-run")
    if args.resume:
        script_args.append("--resume")
    if args.profile:
        script_args.append("--profile")
    if args.profile_create_perfetto:
        script_args.append("--profile-create-perfetto")
    if args.jax_device_memory_profile:
        script_args.append("--jax-device-memory-profile")
    script_args.extend(extra)
    return SCRIPTS[args.script].main(script_args)


if __name__ == "__main__":
    raise SystemExit(main())
