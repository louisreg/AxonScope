"""Batch a small extracellular fiber population.

Run:
    python examples/basic/population_batch_demo.py --mode both --fibers 16

With a JAX profiler trace:
    python examples/basic/population_batch_demo.py \
        --mode double \
        --fibers 64 \
        --nx 201 \
        --jax-profile-dir benchmark/results/jax_profiles
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal, Sequence

import jax.numpy as jnp
import numpy as np

from axonscope.axons import HodgkinHuxley
from axonscope.benchmarking import jax_profile_trace, trace_annotation
from axonscope.electrodes import PointSourceElectrode
from axonscope.solvers import (
    DoubleCableBatchKernel,
    DoubleCableKernel,
    SingleCableKernel,
    SingleCableVStimBatchKernel,
    build_footprint_vstim_initial_previous_batch,
    build_footprint_vstim_midpoint_batch,
    build_vstim_initial_previous_batch,
    build_vstim_midpoint_batch,
    prepare_solver_runtime,
)
from axonscope.stimulation import ExtracellularContext
from axonscope.stimulus import Stimulus


Mode = Literal["single", "double"]


@dataclass(frozen=True)
class PopulationInputs:
    axon: HodgkinHuxley
    context_batch: list[tuple[ExtracellularContext, ...]]
    stimulus: Stimulus
    footprint_V_per_A: np.ndarray
    x_positions_m: np.ndarray
    radial_um: np.ndarray
    longitudinal_offsets_um: np.ndarray


@dataclass(frozen=True)
class PopulationTiming:
    mode: Mode
    fibers: int
    nx: int
    nt: int
    vstim_builder: str
    recording: str
    time_chunk_steps: int | None
    vstim_build_s: float | None
    scalar_warm_s: float | None
    batch_warm_s: float
    speedup: float | None
    max_abs_diff_mV: float | None
    vm_peak_min_mV: float
    vm_peak_max_mV: float


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "double", "both"), default="both")
    parser.add_argument("--fibers", type=int, default=16)
    parser.add_argument("--nx", type=int, default=101)
    parser.add_argument("--length-um", type=float, default=800.0)
    parser.add_argument("--tsim", type=float, default=1.2)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--radial-min-um", type=float, default=80.0)
    parser.add_argument("--radial-max-um", type=float, default=240.0)
    parser.add_argument("--x-spread-um", type=float, default=200.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--batch-only",
        action="store_true",
        help="Skip the scalar loop comparison. Useful for low-memory profiler runs.",
    )
    parser.add_argument(
        "--generic-vstim",
        action="store_true",
        help="Use the generic context-based Vstim builder instead of the footprint fast path.",
    )
    parser.add_argument(
        "--record",
        choices=("full", "center", "probes"),
        default="full",
        help="Vm output to keep. Probes reduce memory for large population runs.",
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        default=8,
        help="Number of evenly spaced probes when --record=probes.",
    )
    parser.add_argument(
        "--time-chunk-steps",
        type=int,
        default=None,
        help="Run the batch solver in time chunks. Best used with --batch-only and --record.",
    )
    parser.add_argument("--plot", action="store_true", help="Show a small peak-Vm population plot.")
    parser.add_argument(
        "--jax-profile-dir",
        type=Path,
        default=None,
        help="Optional directory where a JAX profiler trace should be written.",
    )
    parser.add_argument(
        "--jax-profile-name",
        default="population_batch_demo",
        help="Profile run subdirectory name when --jax-profile-dir is provided.",
    )
    args = parser.parse_args(argv)
    if args.fibers < 1:
        raise ValueError("--fibers must be >= 1.")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0.")
    if args.probe_count < 1:
        raise ValueError("--probe-count must be >= 1.")
    if args.time_chunk_steps is not None and args.time_chunk_steps < 1:
        raise ValueError("--time-chunk-steps must be >= 1.")

    profile_context = (
        jax_profile_trace(args.jax_profile_dir / args.jax_profile_name)
        if args.jax_profile_dir is not None
        else nullcontext()
    )

    modes: tuple[Mode, ...] = ("single", "double") if args.mode == "both" else (args.mode,)
    with profile_context:
        population = build_population_inputs(
            fibers=args.fibers,
            nx=args.nx,
            length_um=args.length_um,
            radial_min_um=args.radial_min_um,
            radial_max_um=args.radial_max_um,
            x_spread_um=args.x_spread_um,
        )
        record_indices = choose_record_indices(
            args.record,
            nx=population.axon.Nx,
            probe_count=args.probe_count,
        )
        timings = [
            run_population_mode(
                population,
                mode=mode,
                tsim_ms=args.tsim,
                dt_ms=args.dt,
                repeats=args.repeats,
                warmups=args.warmups,
                batch_only=bool(args.batch_only),
                use_generic_vstim=bool(args.generic_vstim),
                record_indices=record_indices,
                recording=args.record,
                time_chunk_steps=args.time_chunk_steps,
            )
            for mode in modes
        ]

    print_summary(population, timings, tsim_ms=args.tsim, dt_ms=args.dt)
    if args.jax_profile_dir is not None:
        print(f"jax profile: {args.jax_profile_dir / args.jax_profile_name}")
    if args.plot:
        plot_population(population, timings)


def build_population_inputs(
    *,
    fibers: int,
    nx: int,
    length_um: float,
    radial_min_um: float,
    radial_max_um: float,
    x_spread_um: float,
) -> PopulationInputs:
    axon = HodgkinHuxley(L=length_um, d=0.5, Nx=nx, celsius=6.3)
    axon.insert_I_Clamp(
        position=length_um / 2.0,
        stimulus=Stimulus.pulse(start=0.4, duration=0.05, amplitude=0.8),
    )
    axon.set_extracellular_layer(
        xraxial_MOhm_per_cm=np.full((axon.Nx,), 1e8, dtype=float),
        xg_S_per_cm2=np.full((axon.Nx,), 1e-3, dtype=float),
        xc_uF_per_cm2=np.full((axon.Nx,), 0.01, dtype=float),
        use_extracellular=True,
        Veinit=0.0,
    )

    stimulus = Stimulus.pulse(start=0.3, amplitude=20e-6, duration=0.1, baseline=0.0)
    radial_um = np.linspace(radial_min_um, radial_max_um, fibers)
    longitudinal_offsets_um = np.linspace(-0.5 * x_spread_um, 0.5 * x_spread_um, fibers)
    base_x_m = np.asarray(axon.x, dtype=float) * 1e-6
    x_positions_m = base_x_m[None, :] + longitudinal_offsets_um[:, None] * 1e-6

    context_batch = []
    footprint_rows = []
    for fiber_index, radial in enumerate(radial_um):
        electrode = PointSourceElectrode(
            x0_m=(length_um / 2.0) * 1e-6,
            y0_m=float(radial) * 1e-6,
            z0_m=0.0,
            sigma_S_m=0.3,
        )
        context_batch.append((electrode.attach_stimulus(stimulus),))
        footprint_rows.append(electrode.footprint(x_positions_m[fiber_index]))

    return PopulationInputs(
        axon=axon,
        context_batch=context_batch,
        stimulus=stimulus,
        footprint_V_per_A=np.asarray(footprint_rows, dtype=float),
        x_positions_m=x_positions_m,
        radial_um=radial_um,
        longitudinal_offsets_um=longitudinal_offsets_um,
    )


def run_population_mode(
    population: PopulationInputs,
    *,
    mode: Mode,
    tsim_ms: float,
    dt_ms: float,
    repeats: int,
    warmups: int,
    batch_only: bool = False,
    use_generic_vstim: bool = False,
    record_indices: np.ndarray | None = None,
    recording: str = "full",
    time_chunk_steps: int | None = None,
) -> PopulationTiming:
    axon = population.axon
    include_extracellular = mode == "double"
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        include_extracellular=include_extracellular,
        include_area=include_extracellular,
        precompute_intracellular=True,
        precompute_extracellular=False,
    )

    stream_footprint = (
        not use_generic_vstim
        and batch_only
        and (record_indices is not None or time_chunk_steps is not None)
    )
    vstim_mid = None
    vstim_previous = None
    vstim_build_s = None

    with trace_annotation(f"population/{mode}/build_vstim"):
        if use_generic_vstim:
            vstim_builder = "generic-context"
            vstim_build_s, vstim_mid = time_call(
                lambda: build_vstim_midpoint_batch(
                    axon,
                    population.context_batch,
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                    x_positions_m=population.x_positions_m,
                )
            )
        elif stream_footprint:
            vstim_builder = "footprint-stream"
        else:
            vstim_builder = "footprint"
            vstim_build_s, vstim_mid = time_call(
                lambda: build_footprint_vstim_midpoint_batch(
                    stimulus=population.stimulus,
                    footprint_V_per_A=population.footprint_V_per_A,
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                )
            )
        if mode == "double" and not stream_footprint:
            if use_generic_vstim:
                previous_s, vstim_previous = time_call(
                    lambda: build_vstim_initial_previous_batch(
                        axon,
                        population.context_batch,
                        dt_ms=dt_ms,
                        x_positions_m=population.x_positions_m,
                    )
                )
            else:
                previous_s, vstim_previous = time_call(
                    lambda: build_footprint_vstim_initial_previous_batch(
                        stimulus=population.stimulus,
                        footprint_V_per_A=population.footprint_V_per_A,
                        dt_ms=dt_ms,
                    )
                )
            vstim_build_s += previous_s

    scalar_fn = _single_scalar_loop if mode == "single" else _double_scalar_loop
    batch_fn = _single_batch_run if mode == "single" else _double_batch_run
    footprint_batch_fn = (
        _single_batch_footprint_run if mode == "single" else _double_batch_footprint_run
    )

    scalar_first = None
    if not batch_only:
        if vstim_mid is None:
            raise ValueError("scalar comparison requires materialized Vstim.")
        _, scalar_first = _timed_annotated(
            f"population/{mode}/scalar_first",
            lambda: scalar_fn(runtime, axon, vstim_mid, vstim_previous),
        )
    _, batch_first = _timed_annotated(
        f"population/{mode}/batch_first",
        lambda: (
            footprint_batch_fn(
                runtime,
                axon,
                population,
                record_indices=record_indices,
                time_chunk_steps=time_chunk_steps,
            )
            if stream_footprint
            else batch_fn(
                runtime,
                axon,
                vstim_mid,
                vstim_previous,
                record_indices=record_indices,
                time_chunk_steps=time_chunk_steps,
            )
        ),
    )

    for _ in range(warmups):
        if not batch_only:
            _timed_annotated(
                f"population/{mode}/scalar_warmup",
                lambda: scalar_fn(runtime, axon, vstim_mid, vstim_previous),
            )
        _timed_annotated(
            f"population/{mode}/batch_warmup",
            lambda: (
                footprint_batch_fn(
                    runtime,
                    axon,
                    population,
                    record_indices=record_indices,
                    time_chunk_steps=time_chunk_steps,
                )
                if stream_footprint
                else batch_fn(
                    runtime,
                    axon,
                    vstim_mid,
                    vstim_previous,
                    record_indices=record_indices,
                    time_chunk_steps=time_chunk_steps,
                )
            ),
        )

    scalar_samples = []
    if not batch_only:
        scalar_samples = [
            _timed_annotated(
                f"population/{mode}/scalar_measured",
                lambda: scalar_fn(runtime, axon, vstim_mid, vstim_previous),
            )[0]
            for _ in range(repeats)
        ]
    batch_samples = [
        _timed_annotated(
            f"population/{mode}/batch_measured",
            lambda: (
                footprint_batch_fn(
                    runtime,
                    axon,
                    population,
                    record_indices=record_indices,
                    time_chunk_steps=time_chunk_steps,
                )
                if stream_footprint
                else batch_fn(
                    runtime,
                    axon,
                    vstim_mid,
                    vstim_previous,
                    record_indices=record_indices,
                    time_chunk_steps=time_chunk_steps,
                )
            ),
        )[0]
        for _ in range(repeats)
    ]

    batch_np = np.asarray(batch_first)
    fiber_peaks = np.max(batch_np, axis=(1, 2))
    batch_warm = float(np.mean(batch_samples))
    scalar_warm = float(np.mean(scalar_samples)) if scalar_samples else None
    max_abs_diff = None
    speedup = None
    if scalar_first is not None and scalar_warm is not None:
        scalar_np = np.asarray(scalar_first)
        if record_indices is not None:
            scalar_np = scalar_np[:, :, record_indices]
        diff = batch_np - scalar_np
        max_abs_diff = float(np.max(np.abs(diff)))
        speedup = float(scalar_warm / batch_warm)

    return PopulationTiming(
        mode=mode,
        fibers=int(population.footprint_V_per_A.shape[0]),
        nx=int(runtime.membrane.Nx),
        nt=int(runtime.grid.Nt),
        vstim_builder=vstim_builder,
        recording=recording,
        time_chunk_steps=time_chunk_steps,
        vstim_build_s=None if vstim_build_s is None else float(vstim_build_s),
        scalar_warm_s=scalar_warm,
        batch_warm_s=batch_warm,
        speedup=speedup,
        max_abs_diff_mV=max_abs_diff,
        vm_peak_min_mV=float(np.min(fiber_peaks)),
        vm_peak_max_mV=float(np.max(fiber_peaks)),
    )


def _single_scalar_loop(runtime, axon, vstim_mid, vstim_previous):
    del vstim_previous
    rows = []
    for batch_index in range(int(vstim_mid.shape[0])):
        stimulation = replace(
            runtime.stimulation,
            extracellular_potential_mid_mV=vstim_mid[batch_index],
        )
        row_runtime = replace(runtime, stimulation=stimulation)
        rows.append(
            SingleCableKernel(
                runtime=row_runtime,
                Cm_uF_cm2=jnp.asarray(axon.Cm, dtype=runtime.membrane.dtype),
            ).run().Vm
        )
    return jnp.stack(rows)


def _single_batch_run(
    runtime,
    axon,
    vstim_mid,
    vstim_previous,
    *,
    record_indices=None,
    time_chunk_steps=None,
):
    del vstim_previous
    if vstim_mid is None:
        raise ValueError("vstim_mid is required for materialized batch runs.")
    return SingleCableVStimBatchKernel(
        runtime=runtime,
        Cm_uF_cm2=jnp.asarray(axon.Cm, dtype=runtime.membrane.dtype),
    ).run(
        extracellular_potential_mid_mV=vstim_mid,
        record_indices=record_indices,
        time_chunk_steps=time_chunk_steps,
    ).Vm


def _single_batch_footprint_run(
    runtime,
    axon,
    population: PopulationInputs,
    *,
    record_indices=None,
    time_chunk_steps=None,
):
    return SingleCableVStimBatchKernel(
        runtime=runtime,
        Cm_uF_cm2=jnp.asarray(axon.Cm, dtype=runtime.membrane.dtype),
    ).run_footprint(
        stimulus=population.stimulus,
        footprint_V_per_A=population.footprint_V_per_A,
        record_indices=record_indices,
        time_chunk_steps=time_chunk_steps,
    ).Vm


def _double_scalar_loop(runtime, axon, vstim_mid, vstim_previous):
    if vstim_previous is None:
        raise ValueError("vstim_previous is required for double-cable scalar loop.")
    rows = []
    for batch_index in range(int(vstim_mid.shape[0])):
        stimulation = replace(
            runtime.stimulation,
            extracellular_potential_mid_mV=vstim_mid[batch_index],
            extracellular_potential_initial_previous_mV=vstim_previous[batch_index],
        )
        row_runtime = replace(runtime, stimulation=stimulation)
        rows.append(
            DoubleCableKernel(
                runtime=row_runtime,
                Veinit_mV=float(axon.Veinit),
            ).run().Vm
        )
    return jnp.stack(rows)


def _double_batch_run(
    runtime,
    axon,
    vstim_mid,
    vstim_previous,
    *,
    record_indices=None,
    time_chunk_steps=None,
):
    if vstim_mid is None:
        raise ValueError("vstim_mid is required for materialized batch runs.")
    if vstim_previous is None:
        raise ValueError("vstim_previous is required for double-cable batch run.")
    return DoubleCableBatchKernel(
        runtime=runtime,
        Veinit_mV=float(axon.Veinit),
    ).run(
        extracellular_potential_mid_mV=vstim_mid,
        extracellular_potential_initial_previous_mV=vstim_previous,
        record_indices=record_indices,
        time_chunk_steps=time_chunk_steps,
    ).Vm


def _double_batch_footprint_run(
    runtime,
    axon,
    population: PopulationInputs,
    *,
    record_indices=None,
    time_chunk_steps=None,
):
    return DoubleCableBatchKernel(
        runtime=runtime,
        Veinit_mV=float(axon.Veinit),
    ).run_footprint(
        stimulus=population.stimulus,
        footprint_V_per_A=population.footprint_V_per_A,
        record_indices=record_indices,
        time_chunk_steps=time_chunk_steps,
    ).Vm


def choose_record_indices(recording: str, *, nx: int, probe_count: int) -> np.ndarray | None:
    if recording == "full":
        return None
    if recording == "center":
        return np.asarray([nx // 2], dtype=np.int32)
    if recording == "probes":
        count = min(int(probe_count), nx)
        return np.unique(np.linspace(0, nx - 1, count, dtype=np.int32))
    raise ValueError(f"unknown recording mode: {recording}")


def _timed_annotated(label: str, fn: Callable[[], object]) -> tuple[float, object]:
    with trace_annotation(label):
        return time_call(fn)


def time_call(fn: Callable[[], object]) -> tuple[float, object]:
    import time

    start = time.perf_counter()
    value = fn()
    block_until_ready(value)
    return time.perf_counter() - start, value


def block_until_ready(value: object) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            block_until_ready(item)


def print_summary(
    population: PopulationInputs,
    timings: Sequence[PopulationTiming],
    *,
    tsim_ms: float,
    dt_ms: float,
) -> None:
    print("=== Population batch demo ===")
    print(
        f"fibers={len(population.context_batch)} "
        f"nx={population.axon.Nx} "
        f"tsim={tsim_ms:g} ms dt={dt_ms:g} ms "
        f"radial={population.radial_um[0]:.1f}-{population.radial_um[-1]:.1f} um "
        f"x_offset={population.longitudinal_offsets_um[0]:.1f}.."
        f"{population.longitudinal_offsets_um[-1]:.1f} um"
    )
    for timing in timings:
        vstim = "streamed" if timing.vstim_build_s is None else f"{timing.vstim_build_s:.4f}s"
        scalar = "n/a" if timing.scalar_warm_s is None else f"{timing.scalar_warm_s:.4f}s"
        speedup = "n/a" if timing.speedup is None else f"{timing.speedup:.3f}"
        diff = "n/a" if timing.max_abs_diff_mV is None else f"{timing.max_abs_diff_mV:.4g} mV"
        chunk = "n/a" if timing.time_chunk_steps is None else str(timing.time_chunk_steps)
        print(
            f"{timing.mode:6s} "
            f"builder={timing.vstim_builder} "
            f"record={timing.recording} "
            f"chunk={chunk} "
            f"Vstim_build={vstim} "
            f"scalar={scalar} "
            f"batch={timing.batch_warm_s:.4f}s "
            f"speedup={speedup} "
            f"diff={diff} "
            f"peak={timing.vm_peak_min_mV:.2f}/{timing.vm_peak_max_mV:.2f} mV"
        )


def plot_population(
    population: PopulationInputs,
    timings: Sequence[PopulationTiming],
) -> None:
    import matplotlib.pyplot as plt

    labels = [timing.mode for timing in timings]
    speedups = [0.0 if timing.speedup is None else timing.speedup for timing in timings]
    plt.figure(figsize=(6, 3))
    plt.bar(labels, speedups)
    plt.ylabel("Warm speedup vs scalar loop")
    plt.title(f"Batch population speedup ({len(population.context_batch)} fibers)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
