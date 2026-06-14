"""Batch a small extracellular fiber pool.

Run:
    python benchmark/runtime/pool_batch_demo.py --mode both --fibers 16

With a JAX profiler trace:
    python benchmark/runtime/pool_batch_demo.py \
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

from axonscope import degC, um
from axonscope.axons import HodgkinHuxley
from axonscope.axon_instance import AxonInstance
from axonscope.backends.jax.input_batches import (
    build_footprint_vstim_initial_previous_batch,
    build_footprint_vstim_midpoint_batch,
    build_vstim_initial_previous_batch,
    build_vstim_midpoint_batch,
)
from axonscope.benchmarking import jax_profile_trace, trace_annotation
from axonscope.channel_models import enable_rate_tables
from axonscope.stimulation import AnalyticalExtracellularContext, PointSourceElectrode
from axonscope.solvers import (
    BatchOptions,
    BatchRecording,
    DoubleCableBatchKernel,
    DoubleCableKernel,
    SingleCableKernel,
    SingleCableVStimBatchKernel,
    prepare_solver_runtime,
)
from axonscope.stimulation import ExtracellularContext, Stimulus


Mode = Literal["single", "double"]


@dataclass(frozen=True)
class BatchInputs:
    axon: AxonInstance
    context_batch: list[tuple[ExtracellularContext, ...]]
    stimulus: Stimulus
    footprint_V_per_A: np.ndarray
    x_positions_m: np.ndarray
    radial_um: np.ndarray
    longitudinal_offsets_um: np.ndarray


@dataclass(frozen=True)
class PoolTiming:
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
        help="Vm output to keep. Probes reduce memory for large pool runs.",
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
    parser.add_argument(
        "--rate-table",
        action="store_true",
        help="Use tabulated alpha/beta rates for the shared membrane model.",
    )
    parser.add_argument(
        "--rate-table-step-mv",
        type=float,
        default=0.05,
        help="Voltage step for --rate-table.",
    )
    parser.add_argument("--plot", action="store_true", help="Show a small peak-Vm pool plot.")
    parser.add_argument(
        "--jax-profile-dir",
        type=Path,
        default=None,
        help="Optional directory where a JAX profiler trace should be written.",
    )
    parser.add_argument(
        "--jax-profile-name",
        default="pool_batch_demo",
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
    if args.rate_table and args.rate_table_step_mv <= 0:
        raise ValueError("--rate-table-step-mv must be > 0.")

    profile_context = (
        jax_profile_trace(args.jax_profile_dir / args.jax_profile_name)
        if args.jax_profile_dir is not None
        else nullcontext()
    )

    modes: tuple[Mode, ...] = ("single", "double") if args.mode == "both" else (args.mode,)
    with profile_context:
        pool_inputs = build_pool_inputs(
            fibers=args.fibers,
            nx=args.nx,
            length_um=args.length_um,
            radial_min_um=args.radial_min_um,
            radial_max_um=args.radial_max_um,
            x_spread_um=args.x_spread_um,
        )
        if args.rate_table:
            enable_rate_tables(pool_inputs.axon.layout.sections[0].membrane, step_mV=args.rate_table_step_mv)
        options = BatchOptions(
            recording=BatchRecording.from_mode(
                args.record,
                probe_count=args.probe_count,
            ),
            time_chunk_steps=args.time_chunk_steps,
        )
        timings = [
            run_pool_mode(
                pool_inputs,
                mode=mode,
                tsim_ms=args.tsim,
                dt_ms=args.dt,
                repeats=args.repeats,
                warmups=args.warmups,
                batch_only=bool(args.batch_only),
                use_generic_vstim=bool(args.generic_vstim),
                options=options,
            )
            for mode in modes
        ]

    print_summary(
        pool_inputs,
        timings,
        tsim_ms=args.tsim,
        dt_ms=args.dt,
        rate_table=bool(args.rate_table),
        rate_table_step_mV=float(args.rate_table_step_mv),
    )
    if args.jax_profile_dir is not None:
        print(f"jax profile: {args.jax_profile_dir / args.jax_profile_name}")
    if args.plot:
        plot_pool(pool_inputs, timings)


def build_pool_inputs(
    *,
    fibers: int,
    nx: int,
    length_um: float,
    radial_min_um: float,
    radial_max_um: float,
    x_spread_um: float,
) -> BatchInputs:
    axon_model = HodgkinHuxley(
        length=length_um * um,
        diameter=0.5 * um,
        compartments=nx,
        celsius=6.3 * degC,
    )
    axon = AxonInstance(axon_model)
    axon.add_current_clamp(position_um=length_um / 2.0,
        current=Stimulus.pulse(start=0.4, duration=0.05, amplitude=0.8),
    )
    axon.set_extracellular_layer(
        xraxial_MOhm_per_cm=np.full((axon.n_compartments,), 1e8, dtype=float),
        xg_S_per_cm2=np.full((axon.n_compartments,), 1e-3, dtype=float),
        xc_uF_per_cm2=np.full((axon.n_compartments,), 0.01, dtype=float),
        use_extracellular=True,
        Veinit=0.0,
    )

    stimulus = Stimulus.pulse(start=0.3, amplitude=20e-6, duration=0.1, baseline=0.0)
    radial_um = np.linspace(radial_min_um, radial_max_um, fibers)
    longitudinal_offsets_um = np.linspace(-0.5 * x_spread_um, 0.5 * x_spread_um, fibers)
    base_x_m = np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
    x_positions_m = base_x_m[None, :] + longitudinal_offsets_um[:, None] * 1e-6

    context_batch = []
    footprint_rows = []
    for axon_index, radial in enumerate(radial_um):
        electrode = PointSourceElectrode(
            x_um=length_um / 2.0,
            y_um=float(radial),
            z_um=0.0,
        )
        context = AnalyticalExtracellularContext(
            electrodes=[electrode.with_stimulus(stimulus)],
            sigma=0.3,
        )
        context_batch.append((context,))
        footprint_rows.append(
            context.footprint_for_electrode(context.electrodes[0], x_positions_m[axon_index])
        )

    return BatchInputs(
        axon=axon,
        context_batch=context_batch,
        stimulus=stimulus,
        footprint_V_per_A=np.asarray(footprint_rows, dtype=float),
        x_positions_m=x_positions_m,
        radial_um=radial_um,
        longitudinal_offsets_um=longitudinal_offsets_um,
    )


def run_pool_mode(
    pool_inputs: BatchInputs,
    *,
    mode: Mode,
    tsim_ms: float,
    dt_ms: float,
    repeats: int,
    warmups: int,
    batch_only: bool = False,
    use_generic_vstim: bool = False,
    options: BatchOptions | None = None,
) -> PoolTiming:
    axon = pool_inputs.axon
    options = BatchOptions.full() if options is None else options
    record_indices = options.recording.indices_for(axon.n_compartments)
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

    vstim_mid = None
    vstim_previous = None
    vstim_build_s = None

    with trace_annotation(f"pool/{mode}/build_vstim"):
        if use_generic_vstim:
            vstim_builder = "generic-context"
            vstim_build_s, vstim_mid = time_call(
                lambda: build_vstim_midpoint_batch(
                    axon,
                    pool_inputs.context_batch,
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                    x_positions_m=pool_inputs.x_positions_m,
                )
            )
        else:
            vstim_builder = "footprint"
            vstim_build_s, vstim_mid = time_call(
                lambda: build_footprint_vstim_midpoint_batch(
                    stimulus=pool_inputs.stimulus,
                    footprint_V_per_A=pool_inputs.footprint_V_per_A,
                    tsim_ms=tsim_ms,
                    dt_ms=dt_ms,
                )
            )
        if mode == "double":
            if use_generic_vstim:
                previous_s, vstim_previous = time_call(
                    lambda: build_vstim_initial_previous_batch(
                        axon,
                        pool_inputs.context_batch,
                        dt_ms=dt_ms,
                        x_positions_m=pool_inputs.x_positions_m,
                    )
                )
            else:
                previous_s, vstim_previous = time_call(
                    lambda: build_footprint_vstim_initial_previous_batch(
                        stimulus=pool_inputs.stimulus,
                        footprint_V_per_A=pool_inputs.footprint_V_per_A,
                        dt_ms=dt_ms,
                    )
                )
            vstim_build_s += previous_s

    scalar_fn = _single_scalar_loop if mode == "single" else _double_scalar_loop
    batch_fn = _single_batch_run if mode == "single" else _double_batch_run

    scalar_first = None
    if not batch_only:
        if vstim_mid is None:
            raise ValueError("scalar comparison requires materialized Vstim.")
        _, scalar_first = _timed_annotated(
            f"pool/{mode}/scalar_first",
            lambda: scalar_fn(runtime, axon, vstim_mid, vstim_previous),
        )
    _, batch_first = _timed_annotated(
        f"pool/{mode}/batch_first",
        lambda: batch_fn(
            runtime,
            axon,
            vstim_mid,
            vstim_previous,
            options=options,
        ),
    )

    for _ in range(warmups):
        if not batch_only:
            _timed_annotated(
                f"pool/{mode}/scalar_warmup",
                lambda: scalar_fn(runtime, axon, vstim_mid, vstim_previous),
            )
        _timed_annotated(
            f"pool/{mode}/batch_warmup",
            lambda: batch_fn(
                runtime,
                axon,
                vstim_mid,
                vstim_previous,
                options=options,
            ),
        )

    scalar_samples = []
    if not batch_only:
        scalar_samples = [
            _timed_annotated(
                f"pool/{mode}/scalar_measured",
                lambda: scalar_fn(runtime, axon, vstim_mid, vstim_previous),
            )[0]
            for _ in range(repeats)
        ]
    batch_samples = [
        _timed_annotated(
            f"pool/{mode}/batch_measured",
            lambda: batch_fn(
                runtime,
                axon,
                vstim_mid,
                vstim_previous,
                options=options,
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

    return PoolTiming(
        mode=mode,
        fibers=int(pool_inputs.footprint_V_per_A.shape[0]),
        nx=int(runtime.membrane.Nx),
        nt=int(runtime.grid.Nt),
        vstim_builder=vstim_builder,
        recording=options.recording.label,
        time_chunk_steps=options.time_chunk_steps,
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
                Cm_uF_cm2=jnp.asarray(runtime.axon.Cm_uF_cm2, dtype=runtime.membrane.dtype),
            ).run().Vm
        )
    return jnp.stack(rows)


def _single_batch_run(
    runtime,
    axon,
    vstim_mid,
    vstim_previous,
    *,
    options: BatchOptions,
):
    del vstim_previous
    if vstim_mid is None:
        raise ValueError("vstim_mid is required for materialized batch runs.")
    return SingleCableVStimBatchKernel(
        runtime=runtime,
        Cm_uF_cm2=jnp.asarray(runtime.axon.Cm_uF_cm2, dtype=runtime.membrane.dtype),
    ).run(
        extracellular_potential_mid_mV=vstim_mid,
        options=options,
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
    options: BatchOptions,
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
        options=options,
    ).Vm


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
    pool_inputs: BatchInputs,
    timings: Sequence[PoolTiming],
    *,
    tsim_ms: float,
    dt_ms: float,
    rate_table: bool = False,
    rate_table_step_mV: float | None = None,
) -> None:
    print("=== Pool batch demo ===")
    print(
        f"fibers={len(pool_inputs.context_batch)} "
        f"nx={pool_inputs.axon.n_compartments} "
        f"tsim={tsim_ms:g} ms dt={dt_ms:g} ms "
        f"radial={pool_inputs.radial_um[0]:.1f}-{pool_inputs.radial_um[-1]:.1f} um "
        f"x_offset={pool_inputs.longitudinal_offsets_um[0]:.1f}.."
        f"{pool_inputs.longitudinal_offsets_um[-1]:.1f} um "
        f"rate_table={'on' if rate_table else 'off'}"
        + ("" if not rate_table else f" step={rate_table_step_mV:g} mV")
    )
    for timing in timings:
        vstim = "n/a" if timing.vstim_build_s is None else f"{timing.vstim_build_s:.4f}s"
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


def plot_pool(
    pool_inputs: BatchInputs,
    timings: Sequence[PoolTiming],
) -> None:
    import matplotlib.pyplot as plt

    labels = [timing.mode for timing in timings]
    speedups = [0.0 if timing.speedup is None else timing.speedup for timing in timings]
    plt.figure(figsize=(6, 3))
    plt.bar(labels, speedups)
    plt.ylabel("Warm speedup vs scalar loop")
    plt.title(f"Batch pool speedup ({len(pool_inputs.context_batch)} fibers)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
