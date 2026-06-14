"""Run Phase 2.5 hotpath diagnostic workloads.

Examples:
    python benchmark/hotpaths/run.py --list
    python benchmark/hotpaths/run.py --workload all --preset smoke
    python benchmark/hotpaths/run.py --workload all --preset scale
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

import axonscope as axs
from benchmark.hotpaths.catalog import HOTPATH_PRESETS, HOTPATH_WORKLOADS


DEFAULT_OUT_DIR = Path("benchmark/results/hotpaths")


@dataclass(frozen=True)
class PlannedRun:
    """One workload/size combination."""

    workload: str
    size: int


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        default="all",
        choices=("all", *HOTPATH_WORKLOADS),
        help="Workload to run.",
    )
    parser.add_argument(
        "--preset",
        default="smoke",
        choices=tuple(HOTPATH_PRESETS),
        help="Named size preset.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Explicit population sizes. Overrides --preset.",
    )
    parser.add_argument("--compartments", type=int, default=11)
    parser.add_argument("--length-um", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=0.30, help="Duration in ms.")
    parser.add_argument("--dt", type=float, default=0.05, help="Step size in ms.")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=None, help="Optional run directory prefix.")
    parser.add_argument("--list", action="store_true", help="List workloads and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs and exit.")
    parser.add_argument(
        "--print-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print each AxonScope hotpath report.",
    )
    parser.add_argument(
        "--sync-device",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize JAX arrays at kernel.wait.",
    )
    args = parser.parse_args(argv)

    if args.list:
        print_workloads()
        return
    if args.compartments < 3:
        raise ValueError("--compartments must be >= 3.")
    if args.duration <= 0.0:
        raise ValueError("--duration must be > 0.")
    if args.dt <= 0.0:
        raise ValueError("--dt must be > 0.")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0.")

    runs = planned_runs(args.workload, resolve_sizes(args.preset, args.sizes))
    if args.dry_run:
        for run in runs:
            print(f"{run.workload} size={run.size}")
        return

    run_root = make_run_root(args.out_dir, prefix=args.prefix)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": args.preset,
        "sizes": list(dict.fromkeys(run.size for run in runs)),
        "workloads": list(dict.fromkeys(run.workload for run in runs)),
        "runs": [],
    }

    for run in runs:
        simulation = build_simulation(
            run.workload,
            size=run.size,
            compartments=args.compartments,
            length_um=args.length_um,
            duration_ms=args.duration,
            dt_ms=args.dt,
        )
        for _ in range(args.warmups):
            simulation.run()

        output_dir = run_root / f"{run.workload}_n{run.size}"
        axs.enable_benchmark(
            output_dir,
            print_summary=False,
            sync_device=bool(args.sync_device),
        )
        try:
            results = simulation.run()
        finally:
            report = axs.disable_benchmark(print_summary=bool(args.print_summary))

        run_record = {
            "workload": run.workload,
            "size": run.size,
            "output_dir": str(output_dir),
            "result_count": len(results),
            "vm_shapes": [list(np.asarray(result.Vm).shape) for result in results[:3]],
            "event_count": 0 if report is None else len(report.events),
            "summary": [] if report is None else [row.to_dict() for row in report.summary],
        }
        manifest["runs"].append(run_record)
        print(
            f"{run.workload} size={run.size}: "
            f"{run_record['event_count']} events -> {output_dir}"
        )

    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"manifest: {manifest_path}")


def print_workloads() -> None:
    print("Hotpath workloads:")
    for name, workload in HOTPATH_WORKLOADS.items():
        print(f"  {name:28s} {workload.description}")
    print("Presets:")
    for name, sizes in HOTPATH_PRESETS.items():
        joined = ", ".join(str(size) for size in sizes)
        print(f"  {name:28s} sizes={joined}")


def resolve_sizes(preset: str, sizes: Sequence[int] | None) -> tuple[int, ...]:
    if sizes is not None:
        resolved = tuple(int(size) for size in sizes)
    else:
        resolved = HOTPATH_PRESETS[preset]
    if not resolved:
        raise ValueError("at least one size is required.")
    if any(size < 1 for size in resolved):
        raise ValueError("all sizes must be >= 1.")
    return resolved


def planned_runs(workload: str, sizes: Sequence[int]) -> tuple[PlannedRun, ...]:
    workload_names = tuple(HOTPATH_WORKLOADS) if workload == "all" else (workload,)
    return tuple(
        PlannedRun(workload=name, size=int(size))
        for name in workload_names
        for size in sizes
    )


def make_run_root(out_dir: Path, *, prefix: str | None) -> Path:
    stem = prefix or f"hotpaths_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = out_dir / stem
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_simulation(
    workload: str,
    *,
    size: int,
    compartments: int,
    length_um: float,
    duration_ms: float,
    dt_ms: float,
) -> axs.AxonSimulation:
    if workload == "intracellular_only":
        instances = build_intracellular_pool(
            size=size,
            compartments=compartments,
            length_um=length_um,
        )
    elif workload == "point_source_extracellular":
        instances = build_point_source_pool(
            size=size,
            compartments=compartments,
            length_um=length_um,
        )
    else:
        raise ValueError(f"Unknown hotpath workload: {workload!r}.")

    return axs.AxonSimulation(
        axs.AxonPopulation(instances),
        duration=duration_ms * axs.ms,
        dt=dt_ms * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )


def build_intracellular_pool(
    *,
    size: int,
    compartments: int,
    length_um: float,
) -> list[axs.AxonInstance]:
    axon = axs.axons.HodgkinHuxley(
        length=length_um * axs.um,
        diameter=0.9 * axs.um,
        compartments=compartments,
        celsius=6.3 * axs.degC,
    )
    instances = []
    for index in range(size):
        instance = axs.AxonInstance(axon)
        instance.add_current_clamp(
            position=0.5 * length_um * axs.um,
            current=axs.Stimulus.pulse(
                start=0.10 * axs.ms,
                duration=0.10 * axs.ms,
                amplitude=(0.5 + 0.01 * index) * axs.nA,
            ),
        )
        instances.append(instance)
    return instances


def build_point_source_pool(
    *,
    size: int,
    compartments: int,
    length_um: float,
) -> list[axs.AxonInstance]:
    axon = axs.axons.HodgkinHuxley(
        length=length_um * axs.um,
        diameter=0.9 * axs.um,
        compartments=compartments,
        celsius=6.3 * axs.degC,
    )
    stimulus = axs.Stimulus.pulse(
        start=0.10 * axs.ms,
        duration=0.10 * axs.ms,
        amplitude=25.0 * axs.uA,
    )
    electrode = axs.PointSourceElectrode(
        x=0.5 * length_um * axs.um,
        z=120.0 * axs.um,
        stimulus=stimulus,
    )
    context = axs.AnalyticalExtracellularContext(
        electrodes=[electrode],
        sigma=0.3 * axs.S_per_m,
    )

    offsets = np.linspace(-40.0, 40.0, size) if size > 1 else np.asarray([0.0])
    instances = []
    for offset_um in offsets:
        instance = axs.AxonInstance(axon, y=float(offset_um) * axs.um)
        instance.add_extracellular_context(context=context)
        instances.append(instance)
    return instances


if __name__ == "__main__":
    main()
