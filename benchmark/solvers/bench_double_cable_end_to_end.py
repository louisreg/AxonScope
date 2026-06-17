"""Benchmark end-to-end exact double-cable batch-kernel cases.

This runner complements ``bench_double_cable_linear_solvers.py``. It builds a
homogeneous MRG-like double-cable pool, prepares runtime arrays, materializes
extracellular inputs, optionally materializes dense intracellular input, and
runs ``DoubleCableBatchKernel`` with selected recording and solver policies.

It is intended for Phase 7.6.3 evidence gathering, especially in Colab GPU
runtimes where local GPU execution is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

import axonscope as axs
from axonscope.backends.jax.input_batches import (
    build_vstim_initial_previous_batch,
    build_vstim_midpoint_batch,
)
from axonscope.solvers import (
    BatchOptions,
    BatchRecording,
    DoubleCableBatchKernel,
    prepare_solver_runtime,
    resolve_double_cable_block_solver,
)
from axonscope.solvers.observer_runtime import build_solver_observer_plan
from benchmark.hotpaths.run import build_double_cable_extracellular_pool


DEFAULT_OUT_DIR = Path("benchmark/results/solvers")
PUBLIC_SOLVER_CHOICES = ("auto", "thomas", "pcr", "pcr_soa", "pcr_adaptive")
BENCHMARK_ONLY_SOLVER_CHOICES = ("split_gs_3", "split_gs_4")
SOLVER_CHOICES = PUBLIC_SOLVER_CHOICES + BENCHMARK_ONLY_SOLVER_CHOICES
RECORDING_CHOICES = ("none", "center", "full")
IINJ_CHOICES = ("none", "dense_zero", "nonzero")


@dataclass(frozen=True)
class EndToEndCase:
    """One end-to-end double-cable benchmark case."""

    batch_size: int
    target_nx: int
    nt: int
    dt_ms: float
    recording: str
    iinj_mode: str
    requested_solver: str

    @property
    def duration_ms(self) -> float:
        return int(self.nt) * float(self.dt_ms)

    @property
    def label(self) -> str:
        return (
            f"{self.requested_solver}_B{self.batch_size}_targetNx{self.target_nx}"
            f"_Nt{self.nt}_{self.recording}_{self.iinj_mode}"
        )


def planned_cases(
    *,
    batch_sizes: Sequence[int],
    nx_values: Sequence[int],
    nt_values: Sequence[int],
    dt_ms: float,
    recordings: Sequence[str],
    iinj_modes: Sequence[str],
    solvers: Sequence[str],
) -> tuple[EndToEndCase, ...]:
    """Expand benchmark dimensions into cases."""

    cases: list[EndToEndCase] = []
    for batch_size in batch_sizes:
        if int(batch_size) < 1:
            raise ValueError("all batch sizes must be >= 1.")
        for nx in nx_values:
            if int(nx) < 1:
                raise ValueError("all Nx values must be >= 1.")
            for nt in nt_values:
                if int(nt) < 1:
                    raise ValueError("all Nt values must be >= 1.")
                for recording in recordings:
                    if recording not in RECORDING_CHOICES:
                        raise ValueError(f"unknown recording mode: {recording!r}.")
                    for iinj_mode in iinj_modes:
                        if iinj_mode not in IINJ_CHOICES:
                            raise ValueError(f"unknown Iinj mode: {iinj_mode!r}.")
                        for solver in solvers:
                            if solver not in SOLVER_CHOICES:
                                raise ValueError(f"unknown solver choice: {solver!r}.")
                            if (
                                solver in BENCHMARK_ONLY_SOLVER_CHOICES
                                and recording == "none"
                            ):
                                raise ValueError(
                                    "split benchmark-only E2E solvers require "
                                    "recording='center' or 'full' so the "
                                    "batch-native array kernel is exercised."
                                )
                            cases.append(
                                EndToEndCase(
                                    batch_size=int(batch_size),
                                    target_nx=int(nx),
                                    nt=int(nt),
                                    dt_ms=float(dt_ms),
                                    recording=recording,
                                    iinj_mode=iinj_mode,
                                    requested_solver=solver,
                                )
                            )
    return tuple(cases)


def resolve_e2e_solver(solver: str, *, platform: str) -> str:
    if solver in BENCHMARK_ONLY_SOLVER_CHOICES:
        return "split_iterative"
    return resolve_double_cable_block_solver(solver, platform=platform)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--nx", type=int, nargs="+", default=[32, 51, 64, 96])
    parser.add_argument("--nt", type=int, nargs="+", default=[500, 1000])
    parser.add_argument("--dt", type=float, default=0.01, help="Step size in ms.")
    parser.add_argument("--recordings", nargs="+", choices=RECORDING_CHOICES, default=["none", "center"])
    parser.add_argument("--iinj-modes", nargs="+", choices=IINJ_CHOICES, default=["none"])
    parser.add_argument("--solvers", nargs="+", choices=SOLVER_CHOICES, default=["auto", "thomas", "pcr_adaptive"])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--time-chunk-steps", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jax-trace", action="store_true")
    parser.add_argument("--jax-trace-dir", type=Path, default=None)
    parser.add_argument("--jax-trace-create-perfetto", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)

    if args.dt <= 0.0:
        raise ValueError("--dt must be > 0.")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0.")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    if args.time_chunk_steps is not None and args.time_chunk_steps < 1:
        raise ValueError("--time-chunk-steps must be >= 1.")

    cases = planned_cases(
        batch_sizes=args.batch_sizes,
        nx_values=args.nx,
        nt_values=args.nt,
        dt_ms=args.dt,
        recordings=args.recordings,
        iinj_modes=args.iinj_modes,
        solvers=args.solvers,
    )
    platform = jax.default_backend()

    if args.dry_run:
        for case in cases:
            resolved = resolve_e2e_solver(case.requested_solver, platform=platform)
            print(
                f"{case.requested_solver}->{resolved}"
                f" B={case.batch_size} targetNx={case.target_nx}"
                f" Nt={case.nt} dt={case.dt_ms}"
                f" recording={case.recording} iinj={case.iinj_mode}"
            )
        return

    run_root = _make_run_root(args.out_dir, prefix=args.prefix)
    trace_root = None
    if args.jax_trace or args.jax_trace_dir is not None:
        trace_root = args.jax_trace_dir or run_root / "jax_traces"
        trace_root.mkdir(parents=True, exist_ok=True)

    parameters = {
        "batch_sizes": [int(value) for value in args.batch_sizes],
        "nx": [int(value) for value in args.nx],
        "nt": [int(value) for value in args.nt],
        "dt_ms": float(args.dt),
        "recordings": list(args.recordings),
        "iinj_modes": list(args.iinj_modes),
        "solvers": list(args.solvers),
        "warmups": int(args.warmups),
        "repeats": int(args.repeats),
        "time_chunk_steps": args.time_chunk_steps,
        "jax_trace": trace_root is not None,
        "jax_trace_dir": None if trace_root is None else str(trace_root),
        "jax_trace_create_perfetto": bool(args.jax_trace_create_perfetto),
    }

    rows = []
    for case_index, case in enumerate(cases, start=1):
        print(
            "case "
            f"{case_index}/{len(cases)}: "
            f"{case.requested_solver} B={case.batch_size} targetNx={case.target_nx} "
            f"Nt={case.nt} rec={case.recording} iinj={case.iinj_mode}",
            flush=True,
        )
        row = run_case(
            case,
            platform=platform,
            warmups=args.warmups,
            repeats=args.repeats,
            time_chunk_steps=args.time_chunk_steps,
            trace_root=trace_root,
            create_perfetto_trace=bool(args.jax_trace_create_perfetto),
        )
        rows.append(row)
        _write_outputs(run_root, rows=rows, parameters=parameters, platform=platform)
        print(_format_row(row), flush=True)
    _write_outputs(
        run_root,
        rows=rows,
        parameters=parameters,
        platform=platform,
    )
    print(f"results: {run_root}")


def run_case(
    case: EndToEndCase,
    *,
    platform: str,
    warmups: int,
    repeats: int,
    time_chunk_steps: int | None,
    trace_root: Path | None,
    create_perfetto_trace: bool,
) -> dict[str, Any]:
    """Run one end-to-end double-cable case."""

    total_start = time.perf_counter()
    setup_start = time.perf_counter()
    instances = build_double_cable_extracellular_pool(
        size=case.batch_size,
        compartments=case.target_nx,
    )
    setup_ms = (time.perf_counter() - setup_start) * 1e3

    representative = instances[0]
    runtime_start = time.perf_counter()
    runtime = prepare_solver_runtime(
        representative,
        tsim_ms=case.duration_ms,
        dt_ms=case.dt_ms,
        include_extracellular=True,
        include_area=True,
        precompute_intracellular=False,
        precompute_extracellular=False,
        compile_stimulation=False,
    )
    runtime_ms = (time.perf_counter() - runtime_start) * 1e3
    actual_nx = int(runtime.membrane.Nx)

    contexts = [instance.extracellular_context for instance in instances]
    vext_start = time.perf_counter()
    vext_mid = build_vstim_midpoint_batch(
        representative,
        contexts,
        tsim_ms=case.duration_ms,
        dt_ms=case.dt_ms,
        x_positions_m=_x_positions_m_for_instances(instances),
        axon_y_um=_axon_y_um_for_instances(instances),
        axon_z_um=_axon_z_um_for_instances(instances),
        dtype_local=runtime.membrane.dtype,
    )
    vext_previous = build_vstim_initial_previous_batch(
        representative,
        contexts,
        dt_ms=case.dt_ms,
        x_positions_m=_x_positions_m_for_instances(instances),
        axon_y_um=_axon_y_um_for_instances(instances),
        axon_z_um=_axon_z_um_for_instances(instances),
        dtype_local=runtime.membrane.dtype,
    )
    vext_ms = (time.perf_counter() - vext_start) * 1e3

    iinj_start = time.perf_counter()
    iinj = _build_iinj(
        case.iinj_mode,
        batch_size=case.batch_size,
        nt=case.nt,
        nx=actual_nx,
        dtype=runtime.membrane.dtype,
    )
    iinj_ms = (time.perf_counter() - iinj_start) * 1e3

    observers = _observer_plan(representative, runtime) if case.recording == "none" else None
    options = BatchOptions(
        recording=_batch_recording(case.recording),
        time_chunk_steps=time_chunk_steps,
        double_cable_block_solver=(
            "pcr_soa"
            if case.requested_solver in BENCHMARK_ONLY_SOLVER_CHOICES
            else case.requested_solver
        ),
    )
    resolved_solver = resolve_e2e_solver(case.requested_solver, platform=platform)
    benchmark_solver_override = (
        case.requested_solver
        if case.requested_solver in BENCHMARK_ONLY_SOLVER_CHOICES
        else None
    )
    kernel = DoubleCableBatchKernel(
        runtime=runtime,
        Veinit_mV=float(getattr(representative.axon, "Veinit", 0.0)),
    )

    for _ in range(int(warmups)):
        warm = kernel.run(
            extracellular_potential_mid_mV=vext_mid,
            extracellular_potential_initial_previous_mV=vext_previous,
            intracellular_current_density_mid=iinj,
            options=options,
            observers=observers,
            benchmark_double_cable_block_solver=benchmark_solver_override,
        )
        _block_until_ready(warm)

    trace_dir = None if trace_root is None else trace_root / _safe_label(case.label)
    enqueue_times: list[float] = []
    wait_times: list[float] = []
    output = None
    for repeat_index in range(int(repeats)):
        with _maybe_trace(
            trace_dir if repeat_index == 0 else None,
            create_perfetto_trace=create_perfetto_trace,
        ):
            enqueue_start = time.perf_counter()
            output = kernel.run(
                extracellular_potential_mid_mV=vext_mid,
                extracellular_potential_initial_previous_mV=vext_previous,
                intracellular_current_density_mid=iinj,
                options=options,
                observers=observers,
                benchmark_double_cable_block_solver=benchmark_solver_override,
            )
            enqueue_times.append((time.perf_counter() - enqueue_start) * 1e3)
            wait_start = time.perf_counter()
            _block_until_ready(output)
            wait_times.append((time.perf_counter() - wait_start) * 1e3)

    kernel_enqueue_median_ms = float(statistics.median(enqueue_times))
    kernel_wait_median_ms = float(statistics.median(wait_times))
    total_setup_ms = setup_ms + runtime_ms + vext_ms + iinj_ms
    total_with_inputs_ms = total_setup_ms + kernel_enqueue_median_ms + kernel_wait_median_ms
    vm = None if output is None else output.Vm
    observations = None if output is None else output.observations
    return {
        "requested_solver": case.requested_solver,
        "resolved_solver": resolved_solver,
        "batch_size": int(case.batch_size),
        "target_nx": int(case.target_nx),
        "actual_nx": actual_nx,
        "nt": int(case.nt),
        "dt_ms": float(case.dt_ms),
        "duration_ms": float(case.duration_ms),
        "recording": case.recording,
        "iinj_mode": case.iinj_mode,
        "setup_ms": setup_ms,
        "runtime_ms": runtime_ms,
        "vext_ms": vext_ms,
        "iinj_ms": iinj_ms,
        "kernel_enqueue_median_ms": kernel_enqueue_median_ms,
        "kernel_wait_median_ms": kernel_wait_median_ms,
        "kernel_enqueue_min_ms": min(enqueue_times),
        "kernel_wait_min_ms": min(wait_times),
        "total_setup_ms": total_setup_ms,
        "total_with_inputs_ms": total_with_inputs_ms,
        "total_case_wall_ms": (time.perf_counter() - total_start) * 1e3,
        "vext_mid_bytes": _nbytes(vext_mid),
        "vext_previous_bytes": _nbytes(vext_previous),
        "iinj_bytes": _nbytes(iinj),
        "vm_output_bytes": _nbytes(vm),
        "vm_shape": "" if vm is None else list(getattr(vm, "shape", ())),
        "observation_names": "" if observations is None else sorted(observations),
        "trace_dir": "" if trace_dir is None else str(trace_dir),
    }


def _build_iinj(
    mode: str,
    *,
    batch_size: int,
    nt: int,
    nx: int,
    dtype: Any,
) -> Any | None:
    if mode == "none":
        return None
    if mode == "dense_zero":
        return jnp.zeros((batch_size, nt, nx), dtype=dtype)
    if mode == "nonzero":
        values = jnp.zeros((batch_size, nt, nx), dtype=dtype)
        center = int(nx) // 2
        start = max(0, int(0.10 * nt))
        stop = max(start + 1, int(0.20 * nt))
        amplitudes = jnp.linspace(
            jnp.asarray(0.02, dtype=dtype),
            jnp.asarray(0.04, dtype=dtype),
            int(batch_size),
            dtype=dtype,
        )
        return values.at[:, start:stop, center].set(amplitudes[:, None])
    raise ValueError(f"unknown Iinj mode: {mode!r}.")


def _batch_recording(mode: str) -> BatchRecording:
    if mode == "none":
        return BatchRecording.none()
    if mode == "center":
        return BatchRecording.center()
    if mode == "full":
        return BatchRecording.full()
    raise ValueError(f"unknown recording mode: {mode!r}.")


def _observer_plan(instance: axs.AxonInstance, runtime: Any):
    positions_um = np.asarray(
        instance.axon.layout.position_values(unit="micrometer"),
        dtype=float,
    )
    return build_solver_observer_plan(
        (
            axs.analysis.PeakVoltage(target=axs.positions.CENTER),
            axs.analysis.Activation(threshold=-80.0 * axs.mV, target=axs.positions.CENTER),
        ),
        positions_um=positions_um,
        dtype=runtime.membrane.dtype,
    )


def _x_positions_m_for_instances(instances: Sequence[axs.AxonInstance]) -> np.ndarray:
    rows = []
    for instance in instances:
        x_um = np.asarray(instance.axon.layout.position_values(unit="micrometer"), dtype=float)
        rows.append((x_um + float(getattr(instance, "x_offset_um", 0.0))) * 1e-6)
    return np.stack(rows, axis=0)


def _axon_y_um_for_instances(instances: Sequence[axs.AxonInstance]) -> np.ndarray:
    return np.asarray([float(getattr(instance, "y_um", 0.0)) for instance in instances])


def _axon_z_um_for_instances(instances: Sequence[axs.AxonInstance]) -> np.ndarray:
    return np.asarray([float(getattr(instance, "z_um", 0.0)) for instance in instances])


def _block_until_ready(value: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
        value,
    )


@contextmanager
def _maybe_trace(
    trace_dir: Path | None,
    *,
    create_perfetto_trace: bool,
) -> Iterator[None]:
    if trace_dir is None:
        with nullcontext():
            yield
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(
        str(trace_dir),
        create_perfetto_trace=bool(create_perfetto_trace),
    ):
        with jax.profiler.StepTraceAnnotation("double_cable_end_to_end"):
            yield


def _make_run_root(out_dir: Path, *, prefix: str | None) -> Path:
    stem = prefix or f"double_cable_end_to_end_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = out_dir / stem
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_outputs(
    run_root: Path,
    *,
    rows: Sequence[dict[str, Any]],
    parameters: dict[str, Any],
    platform: str,
) -> None:
    csv_path = run_root / "summary.csv"
    json_path = run_root / "summary.json"
    manifest_path = run_root / "manifest.json"
    fieldnames = tuple(rows[0]) if rows else ()
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
    json_path.write_text(json.dumps(list(rows), indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "platform": platform,
                "parameters": parameters,
                "summary_csv": str(csv_path),
                "summary_json": str(json_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _format_row(row: dict[str, Any]) -> str:
    return (
        f"{row['requested_solver']}->{row['resolved_solver']} "
        f"B={row['batch_size']} targetNx={row['target_nx']} actualNx={row['actual_nx']} "
        f"Nt={row['nt']} rec={row['recording']} iinj={row['iinj_mode']}: "
        f"kernel={row['kernel_enqueue_median_ms'] + row['kernel_wait_median_ms']:.3f} ms, "
        f"total+inputs={row['total_with_inputs_ms']:.3f} ms"
    )


def _nbytes(value: Any) -> int:
    if value is None:
        return 0
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return 0
    return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    return value


def _safe_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("_")


if __name__ == "__main__":
    main()
