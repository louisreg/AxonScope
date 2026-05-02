from __future__ import annotations

import argparse
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

from benchmark.runtime.batch_utils import write_rows
from examples.basic.population_batch_demo import (
    build_population_inputs,
    choose_record_indices,
    run_population_mode,
)


SCENARIOS = (
    "full",
    "center",
    "probes",
    "center_chunked",
    "probes_chunked",
)


@dataclass(frozen=True)
class PopulationMemoryRow:
    scenario: str
    mode: str
    fibers: int
    nx: int
    nt: int
    tsim_ms: float
    dt_ms: float
    recording: str
    recorded_width: int
    time_chunk_steps: int | None
    vstim_builder: str
    vstim_build_s: float | None
    batch_mean_s: float
    vm_peak_min_mV: float
    vm_peak_max_mV: float
    dtype_bytes: int
    estimated_vstim_peak_mib: float
    estimated_recorded_vm_mib: float


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark population batch runtime and dominant tensor sizes."
    )
    parser.add_argument("--mode", choices=("single", "double", "both"), default="double")
    parser.add_argument("--fibers", type=int, default=128)
    parser.add_argument("--nx", type=int, default=201)
    parser.add_argument("--length-um", type=float, default=800.0)
    parser.add_argument("--tsim", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--radial-min-um", type=float, default=80.0)
    parser.add_argument("--radial-max-um", type=float, default=240.0)
    parser.add_argument("--x-spread-um", type=float, default=200.0)
    parser.add_argument("--probe-count", type=int, default=8)
    parser.add_argument("--time-chunk-steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
        help="Memory/runtime scenarios to run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark/results/runtime"),
        help="Directory for JSON and CSV outputs.",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix.")
    args = parser.parse_args(argv)

    if args.fibers < 1:
        raise ValueError("--fibers must be >= 1.")
    if args.nx < 1:
        raise ValueError("--nx must be >= 1.")
    if args.probe_count < 1:
        raise ValueError("--probe-count must be >= 1.")
    if args.time_chunk_steps < 1:
        raise ValueError("--time-chunk-steps must be >= 1.")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0.")

    population = build_population_inputs(
        fibers=args.fibers,
        nx=args.nx,
        length_um=args.length_um,
        radial_min_um=args.radial_min_um,
        radial_max_um=args.radial_max_um,
        x_spread_um=args.x_spread_um,
    )
    modes = ("single", "double") if args.mode == "both" else (args.mode,)
    rows = [
        run_scenario(
            population,
            scenario=scenario,
            mode=mode,
            tsim_ms=args.tsim,
            dt_ms=args.dt,
            probe_count=args.probe_count,
            chunk_steps=args.time_chunk_steps,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        for mode in modes
        for scenario in args.scenarios
    ]

    prefix = args.prefix or datetime.now().strftime("population_memory_%Y%m%d_%H%M%S")
    json_path, csv_path = write_rows(
        rows,
        args.out_dir,
        prefix=prefix,
        metadata={
            "benchmark": "population_memory",
            "fibers": int(args.fibers),
            "nx": int(args.nx),
            "tsim_ms": float(args.tsim),
            "dt_ms": float(args.dt),
            "probe_count": int(args.probe_count),
            "time_chunk_steps": int(args.time_chunk_steps),
            "repeats": int(args.repeats),
            "warmups": int(args.warmups),
        },
    )

    print("=== Population memory/runtime benchmark ===")
    for row in rows:
        vstim = "streamed" if row.vstim_build_s is None else f"{row.vstim_build_s:.4f}s"
        chunk = "n/a" if row.time_chunk_steps is None else str(row.time_chunk_steps)
        print(
            f"{row.mode:6s} {row.scenario:15s} "
            f"record={row.recording:6s} width={row.recorded_width:4d} chunk={chunk:>4s} "
            f"Vstim_peak={row.estimated_vstim_peak_mib:7.2f} MiB "
            f"Vm_out={row.estimated_recorded_vm_mib:7.2f} MiB "
            f"Vstim_build={vstim:>9s} batch={row.batch_mean_s:.4f}s "
            f"peak={row.vm_peak_min_mV:.2f}/{row.vm_peak_max_mV:.2f} mV"
        )
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")


def run_scenario(
    population,
    *,
    scenario: str,
    mode: str,
    tsim_ms: float,
    dt_ms: float,
    probe_count: int,
    chunk_steps: int,
    repeats: int,
    warmups: int,
) -> PopulationMemoryRow:
    recording, time_chunk_steps = scenario_config(scenario, chunk_steps=chunk_steps)
    record_indices = choose_record_indices(
        recording,
        nx=population.axon.Nx,
        probe_count=probe_count,
    )
    timing = run_population_mode(
        population,
        mode=mode,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        repeats=repeats,
        warmups=warmups,
        batch_only=True,
        use_generic_vstim=False,
        record_indices=record_indices,
        recording=recording,
        time_chunk_steps=time_chunk_steps,
    )

    ion_channel = getattr(population.axon, "ion_channel", None)
    dtype_bytes = np.dtype(getattr(ion_channel, "dtype", np.float32)).itemsize
    recorded_width = population.axon.Nx if record_indices is None else int(len(record_indices))
    effective_chunk = timing.nt if time_chunk_steps is None else min(time_chunk_steps, timing.nt)
    vstim_peak_elements = timing.fibers * effective_chunk * timing.nx
    recorded_vm_elements = timing.fibers * timing.nt * recorded_width

    return PopulationMemoryRow(
        scenario=scenario,
        mode=mode,
        fibers=timing.fibers,
        nx=timing.nx,
        nt=timing.nt,
        tsim_ms=float(tsim_ms),
        dt_ms=float(dt_ms),
        recording=recording,
        recorded_width=recorded_width,
        time_chunk_steps=time_chunk_steps,
        vstim_builder=timing.vstim_builder,
        vstim_build_s=timing.vstim_build_s,
        batch_mean_s=timing.batch_warm_s,
        vm_peak_min_mV=timing.vm_peak_min_mV,
        vm_peak_max_mV=timing.vm_peak_max_mV,
        dtype_bytes=dtype_bytes,
        estimated_vstim_peak_mib=bytes_to_mib(vstim_peak_elements * dtype_bytes),
        estimated_recorded_vm_mib=bytes_to_mib(recorded_vm_elements * dtype_bytes),
    )


def scenario_config(scenario: str, *, chunk_steps: int) -> tuple[str, int | None]:
    if scenario == "full":
        return "full", None
    if scenario == "center":
        return "center", None
    if scenario == "probes":
        return "probes", None
    if scenario == "center_chunked":
        return "center", chunk_steps
    if scenario == "probes_chunked":
        return "probes", chunk_steps
    raise ValueError(f"unknown scenario: {scenario}")


def bytes_to_mib(value: int) -> float:
    return float(value) / float(1024**2)


if __name__ == "__main__":
    main()
