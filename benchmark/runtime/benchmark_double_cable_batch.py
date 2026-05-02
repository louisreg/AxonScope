from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

import jax.numpy as jnp
import numpy as np

from axonscope.axons import HodgkinHuxley
from axonscope.electrodes import PointSourceElectrode
from axonscope.solvers import (
    DoubleCableBatchKernel,
    DoubleCableKernel,
    build_vstim_initial_previous_batch,
    build_vstim_midpoint_batch,
)
from axonscope.solvers.runtime import SolverRuntime, prepare_solver_runtime
from axonscope.stimulus import Stimulus
from benchmark.runtime.batch_utils import (
    TimingStats,
    scaled_context_batch,
    time_call,
    write_rows,
)


@dataclass(frozen=True)
class DoubleCableBatchBenchmarkRow:
    model: str
    nx: int
    nt: int
    tsim_ms: float
    dt_ms: float
    batch_size: int
    scalar_first_s: float
    batch_first_s: float
    scalar_warm: TimingStats
    batch_warm: TimingStats
    warm_speedup: float
    max_abs_diff_mV: float
    rmse_mV: float
    vm_min_mV: float
    vm_max_mV: float


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark scalar-loop vs batched full double-cable kernels."
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--nx", type=int, default=41, help="HH compartment count.")
    parser.add_argument("--tsim", type=float, default=1.2, help="Simulation duration in ms.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step in ms.")
    parser.add_argument("--repeats", type=int, default=5, help="Measured warm repetitions.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup repetitions before timing.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark/results/runtime"),
        help="Directory for JSON and CSV outputs.",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix.")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0.")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise ValueError("--batch-sizes values must be >= 1.")

    axon = _build_hh_double_cable(args.nx)
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=args.tsim,
        dt_ms=args.dt,
        include_extracellular=True,
        include_area=True,
        precompute_intracellular=True,
        precompute_extracellular=False,
    )

    rows = [
        benchmark_batch_size(
            runtime,
            axon=axon,
            batch_size=batch_size,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        for batch_size in args.batch_sizes
    ]

    prefix = args.prefix or datetime.now().strftime("double_cable_batch_%Y%m%d_%H%M%S")
    json_path, csv_path = write_rows(
        rows,
        args.out_dir,
        prefix=prefix,
        metadata={
            "benchmark": "double_cable_batch",
            "model": "HodgkinHuxley",
            "nx": int(args.nx),
            "tsim_ms": float(args.tsim),
            "dt_ms": float(args.dt),
            "repeats": int(args.repeats),
            "warmups": int(args.warmups),
        },
    )

    print("=== Double-cable batch benchmark ===")
    for row in rows:
        print(
            f"B={row.batch_size:3d} nx={row.nx:4d} nt={row.nt:4d} "
            f"scalar={row.scalar_warm.mean_s:.5f}s "
            f"batch={row.batch_warm.mean_s:.5f}s "
            f"speedup={row.warm_speedup:.3f} "
            f"diff={row.max_abs_diff_mV:.4g} mV"
        )
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")


def benchmark_batch_size(
    runtime: SolverRuntime,
    *,
    axon: HodgkinHuxley,
    batch_size: int,
    repeats: int,
    warmups: int,
) -> DoubleCableBatchBenchmarkRow:
    vext_mid, vext_previous = _make_scaled_vstim_context_batch(
        axon,
        tsim_ms=runtime.grid.tsim_ms,
        dt_ms=runtime.grid.dt_ms,
        batch_size=batch_size,
    )
    batch_kernel = DoubleCableBatchKernel(
        runtime=runtime,
        Veinit_mV=float(axon.Veinit),
    )

    scalar_first_s, scalar_first = time_call(
        lambda: _run_scalar_loop(runtime, axon, vext_mid, vext_previous)
    )
    batch_first_s, batch_first = time_call(
        lambda: batch_kernel.run(
            extracellular_potential_mid_mV=vext_mid,
            extracellular_potential_initial_previous_mV=vext_previous,
        ).Vm
    )

    for _ in range(warmups):
        time_call(lambda: _run_scalar_loop(runtime, axon, vext_mid, vext_previous))
        time_call(
            lambda: batch_kernel.run(
                extracellular_potential_mid_mV=vext_mid,
                extracellular_potential_initial_previous_mV=vext_previous,
            ).Vm
        )

    scalar_samples = [
        time_call(lambda: _run_scalar_loop(runtime, axon, vext_mid, vext_previous))[0]
        for _ in range(repeats)
    ]
    batch_samples = [
        time_call(
            lambda: batch_kernel.run(
                extracellular_potential_mid_mV=vext_mid,
                extracellular_potential_initial_previous_mV=vext_previous,
            ).Vm
        )[0]
        for _ in range(repeats)
    ]

    scalar_np = np.asarray(scalar_first)
    batch_np = np.asarray(batch_first)
    diff = batch_np - scalar_np
    scalar_stats = TimingStats.from_samples(scalar_samples)
    batch_stats = TimingStats.from_samples(batch_samples)

    return DoubleCableBatchBenchmarkRow(
        model="HodgkinHuxley",
        nx=int(runtime.membrane.Nx),
        nt=int(runtime.grid.Nt),
        tsim_ms=float(runtime.grid.tsim_ms),
        dt_ms=float(runtime.grid.dt_ms),
        batch_size=int(batch_size),
        scalar_first_s=float(scalar_first_s),
        batch_first_s=float(batch_first_s),
        scalar_warm=scalar_stats,
        batch_warm=batch_stats,
        warm_speedup=float(scalar_stats.mean_s / batch_stats.mean_s),
        max_abs_diff_mV=float(np.max(np.abs(diff))),
        rmse_mV=float(np.sqrt(np.mean(diff * diff))),
        vm_min_mV=float(np.min(batch_np)),
        vm_max_mV=float(np.max(batch_np)),
    )


def _run_scalar_loop(
    runtime: SolverRuntime,
    axon: HodgkinHuxley,
    vext_mid_batch,
    vext_previous_batch,
):
    rows = []
    for batch_index in range(int(vext_mid_batch.shape[0])):
        row_stimulation = replace(
            runtime.stimulation,
            extracellular_potential_mid_mV=vext_mid_batch[batch_index],
            extracellular_potential_initial_previous_mV=vext_previous_batch[batch_index],
        )
        row_runtime = replace(runtime, stimulation=row_stimulation)
        rows.append(
            DoubleCableKernel(
                runtime=row_runtime,
                Veinit_mV=float(axon.Veinit),
            ).run().Vm
        )
    return jnp.stack(rows)


def _make_scaled_vstim_context_batch(
    axon: HodgkinHuxley,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_size: int,
):
    base_contexts = tuple(axon.extracellular_contexts)
    context_batch = scaled_context_batch(base_contexts, batch_size=batch_size)
    vext_mid = build_vstim_midpoint_batch(
        axon,
        context_batch,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
    )
    vext_previous = build_vstim_initial_previous_batch(
        axon,
        context_batch,
        dt_ms=dt_ms,
    )
    return vext_mid, vext_previous


def _build_hh_double_cable(nx: int) -> HodgkinHuxley:
    length_um = 400.0
    axon = HodgkinHuxley(L=length_um, d=0.5, Nx=nx, celsius=6.3)
    axon.set_extracellular_layer(
        xraxial_MOhm_per_cm=np.full((axon.Nx,), 1e8, dtype=float),
        xg_S_per_cm2=np.full((axon.Nx,), 1e-3, dtype=float),
        xc_uF_per_cm2=np.full((axon.Nx,), 0.01, dtype=float),
        use_extracellular=True,
        Veinit=0.0,
    )
    axon.insert_I_Clamp(
        position=length_um / 2.0,
        stimulus=Stimulus.pulse(start=0.4, duration=0.05, amplitude=0.8),
    )
    electrode = PointSourceElectrode(
        x0_m=(length_um / 2.0) * 1e-6,
        y0_m=100e-6,
        z0_m=100e-6,
        sigma_S_m=0.3,
    )
    stim = Stimulus.pulse(start=0.3, amplitude=20e-6, duration=0.1, baseline=0.0)
    axon.add_extracellular_context(electrode, stim, replace=True)
    return axon


if __name__ == "__main__":
    main()
