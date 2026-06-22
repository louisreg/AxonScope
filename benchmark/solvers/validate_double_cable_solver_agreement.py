"""Validate E2E double-cable solver-output agreement.

This runner compares recorded ``Vm`` traces from candidate solver routes against
public exact double-cable solvers on held-out batch-kernel workloads. It is a
correctness/physiology harness, not a timing benchmark.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

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
from axonscope.solvers import BatchOptions
from axonscope.backends.jax.batch_kernels import DoubleCableBatchKernel
from axonscope.backends.jax.runtime import prepare_solver_runtime
from benchmark.hotpaths.run import build_double_cable_extracellular_pool
from benchmark.solvers.bench_double_cable_end_to_end import (
    IINJ_CHOICES,
    PUBLIC_SOLVER_CHOICES,
    SOLVER_CHOICES,
    _batch_recording,
    _block_until_ready,
    _build_iinj,
    _make_run_root,
    _write_outputs,
    _axon_y_um_for_instances,
    _axon_z_um_for_instances,
    _x_positions_m_for_instances,
    resolve_e2e_solver,
)


DEFAULT_OUT_DIR = Path("benchmark/results/solvers")
VALIDATION_RECORDING_CHOICES = ("center", "full")
DEFAULT_MAX_ABS_MV_TOL = 1.0e-2
DEFAULT_RMS_MV_TOL = 5.0e-3
DEFAULT_PEAK_MV_TOL = 1.0e-2
DEFAULT_FIRST_CROSSING_TOL_MS = 0.05


@dataclass(frozen=True)
class AgreementCase:
    """One reference-vs-candidate E2E validation case."""

    batch_size: int
    target_nx: int
    nt: int
    dt_ms: float
    recording: str
    iinj_mode: str
    reference_solver: str
    candidate_solver: str

    @property
    def duration_ms(self) -> float:
        return int(self.nt) * float(self.dt_ms)

    @property
    def label(self) -> str:
        return (
            f"{self.candidate_solver}_vs_{self.reference_solver}"
            f"_B{self.batch_size}_targetNx{self.target_nx}"
            f"_Nt{self.nt}_{self.recording}_{self.iinj_mode}"
        )


@dataclass(frozen=True)
class PreparedValidationInputs:
    """Prepared model/input tensors reused across solver comparisons."""

    representative: axs.AxonInstance
    runtime: Any
    kernel: DoubleCableBatchKernel
    vext_mid: Any
    vext_previous: Any
    iinj: Any | None
    actual_nx: int
    setup_ms: float
    runtime_ms: float
    vext_ms: float
    iinj_ms: float


def planned_agreement_cases(
    *,
    batch_sizes: Sequence[int],
    nx_values: Sequence[int],
    nt_values: Sequence[int],
    dt_ms: float,
    recordings: Sequence[str],
    iinj_modes: Sequence[str],
    reference_solvers: Sequence[str],
    candidate_solvers: Sequence[str],
) -> tuple[AgreementCase, ...]:
    """Expand validation dimensions into reference/candidate comparisons."""

    cases: list[AgreementCase] = []
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
                    if recording not in VALIDATION_RECORDING_CHOICES:
                        raise ValueError(
                            "validation requires recorded Vm: use 'center' or 'full'."
                        )
                    for iinj_mode in iinj_modes:
                        if iinj_mode not in IINJ_CHOICES:
                            raise ValueError(f"unknown Iinj mode: {iinj_mode!r}.")
                        for reference_solver in reference_solvers:
                            if reference_solver not in PUBLIC_SOLVER_CHOICES:
                                raise ValueError(
                                    f"reference solver must be public, got {reference_solver!r}."
                                )
                            for candidate_solver in candidate_solvers:
                                if candidate_solver not in SOLVER_CHOICES:
                                    raise ValueError(
                                        f"unknown candidate solver: {candidate_solver!r}."
                                    )
                                if candidate_solver == reference_solver:
                                    raise ValueError(
                                        "candidate solver must differ from reference solver."
                                    )
                                cases.append(
                                    AgreementCase(
                                        batch_size=int(batch_size),
                                        target_nx=int(nx),
                                        nt=int(nt),
                                        dt_ms=float(dt_ms),
                                        recording=recording,
                                        iinj_mode=iinj_mode,
                                        reference_solver=reference_solver,
                                        candidate_solver=candidate_solver,
                                    )
                                )
    return tuple(cases)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--nx", type=int, nargs="+", default=[51, 96])
    parser.add_argument("--nt", type=int, nargs="+", default=[300])
    parser.add_argument("--dt", type=float, default=0.01, help="Step size in ms.")
    parser.add_argument(
        "--recordings",
        nargs="+",
        choices=VALIDATION_RECORDING_CHOICES,
        default=["center"],
    )
    parser.add_argument("--iinj-modes", nargs="+", choices=IINJ_CHOICES, default=["none"])
    parser.add_argument(
        "--reference-solvers",
        nargs="+",
        choices=PUBLIC_SOLVER_CHOICES,
        default=["thomas"],
    )
    parser.add_argument(
        "--candidate-solvers",
        nargs="+",
        choices=SOLVER_CHOICES,
        default=["pcr_adaptive"],
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--time-chunk-steps", type=int, default=None)
    parser.add_argument("--activation-threshold-mv", type=float, default=-20.0)
    parser.add_argument("--activation-blanking-ms", type=float, default=0.0)
    parser.add_argument("--max-abs-mv-tol", type=float, default=DEFAULT_MAX_ABS_MV_TOL)
    parser.add_argument("--rms-mv-tol", type=float, default=DEFAULT_RMS_MV_TOL)
    parser.add_argument("--peak-mv-tol", type=float, default=DEFAULT_PEAK_MV_TOL)
    parser.add_argument(
        "--first-crossing-time-ms-tol",
        type=float,
        default=DEFAULT_FIRST_CROSSING_TOL_MS,
    )
    parser.add_argument(
        "--fail-on-thresholds",
        action="store_true",
        help="Exit non-zero if any comparison exceeds validation tolerances.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dt <= 0.0:
        raise ValueError("--dt must be > 0.")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0.")
    if args.time_chunk_steps is not None and args.time_chunk_steps < 1:
        raise ValueError("--time-chunk-steps must be >= 1.")

    cases = planned_agreement_cases(
        batch_sizes=args.batch_sizes,
        nx_values=args.nx,
        nt_values=args.nt,
        dt_ms=args.dt,
        recordings=args.recordings,
        iinj_modes=args.iinj_modes,
        reference_solvers=args.reference_solvers,
        candidate_solvers=args.candidate_solvers,
    )
    platform = jax.default_backend()

    if args.dry_run:
        for case in cases:
            print(
                f"{case.candidate_solver} vs {case.reference_solver}"
                f" B={case.batch_size} targetNx={case.target_nx}"
                f" Nt={case.nt} dt={case.dt_ms}"
                f" recording={case.recording} iinj={case.iinj_mode}"
            )
        return

    run_root = _make_run_root(args.out_dir, prefix=args.prefix)
    parameters = {
        "batch_sizes": [int(value) for value in args.batch_sizes],
        "nx": [int(value) for value in args.nx],
        "nt": [int(value) for value in args.nt],
        "dt_ms": float(args.dt),
        "recordings": list(args.recordings),
        "iinj_modes": list(args.iinj_modes),
        "reference_solvers": list(args.reference_solvers),
        "candidate_solvers": list(args.candidate_solvers),
        "warmups": int(args.warmups),
        "time_chunk_steps": args.time_chunk_steps,
        "activation_threshold_mV": float(args.activation_threshold_mv),
        "activation_blanking_ms": float(args.activation_blanking_ms),
        "max_abs_mV_tol": float(args.max_abs_mv_tol),
        "rms_mV_tol": float(args.rms_mv_tol),
        "peak_mV_tol": float(args.peak_mv_tol),
        "first_crossing_time_ms_tol": float(args.first_crossing_time_ms_tol),
        "fail_on_thresholds": bool(args.fail_on_thresholds),
    }

    rows: list[dict[str, Any]] = []
    failed = False
    grouped = _group_cases_by_inputs(cases)
    total = len(cases)
    case_index = 0
    for _input_key, input_cases in grouped:
        first = input_cases[0]
        prepared = prepare_validation_inputs(first)
        reference_outputs: dict[str, Any] = {}
        for reference_solver in sorted({case.reference_solver for case in input_cases}):
            reference_outputs[reference_solver] = run_solver_output(
                prepared,
                solver=reference_solver,
                recording=first.recording,
                time_chunk_steps=args.time_chunk_steps,
                warmups=args.warmups,
            )
        for case in input_cases:
            case_index += 1
            print(
                "case "
                f"{case_index}/{total}: {case.candidate_solver} vs {case.reference_solver} "
                f"B={case.batch_size} targetNx={case.target_nx} Nt={case.nt} "
                f"rec={case.recording} iinj={case.iinj_mode}",
                flush=True,
            )
            candidate_output = run_solver_output(
                prepared,
                solver=case.candidate_solver,
                recording=case.recording,
                time_chunk_steps=args.time_chunk_steps,
                warmups=args.warmups,
            )
            row = make_validation_row(
                case,
                prepared=prepared,
                reference_output=reference_outputs[case.reference_solver],
                candidate_output=candidate_output,
                platform=platform,
                activation_threshold_mV=args.activation_threshold_mv,
                activation_blanking_ms=args.activation_blanking_ms,
                max_abs_mV_tol=args.max_abs_mv_tol,
                rms_mV_tol=args.rms_mv_tol,
                peak_mV_tol=args.peak_mv_tol,
                first_crossing_time_ms_tol=args.first_crossing_time_ms_tol,
            )
            failed = failed or not bool(row["passed_thresholds"])
            rows.append(row)
            _write_outputs(run_root, rows=rows, parameters=parameters, platform=platform)
            print(_format_row(row), flush=True)
        _block_until_ready(reference_outputs)

    _write_outputs(run_root, rows=rows, parameters=parameters, platform=platform)
    print(f"results: {run_root}")
    if failed and args.fail_on_thresholds:
        raise SystemExit(1)


def prepare_validation_inputs(case: AgreementCase) -> PreparedValidationInputs:
    """Build model/runtime/input tensors for one validation input workload."""

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

    return PreparedValidationInputs(
        representative=representative,
        runtime=runtime,
        kernel=DoubleCableBatchKernel(
            runtime=runtime,
            Veinit_mV=float(getattr(representative.axon, "Veinit", 0.0)),
        ),
        vext_mid=vext_mid,
        vext_previous=vext_previous,
        iinj=iinj,
        actual_nx=actual_nx,
        setup_ms=setup_ms,
        runtime_ms=runtime_ms,
        vext_ms=vext_ms,
        iinj_ms=iinj_ms,
    )


def run_solver_output(
    prepared: PreparedValidationInputs,
    *,
    solver: str,
    recording: str,
    time_chunk_steps: int | None,
    warmups: int,
) -> Any:
    """Run one solver and return its recorded Vm output."""

    options = BatchOptions(
        recording=_batch_recording(recording),
        time_chunk_steps=time_chunk_steps,
        double_cable_block_solver=solver,
    )
    output = None
    for _ in range(int(warmups)):
        output = prepared.kernel.run(
            extracellular_potential_mid_mV=prepared.vext_mid,
            extracellular_potential_initial_previous_mV=prepared.vext_previous,
            intracellular_current_density_mid=prepared.iinj,
            options=options,
        )
        _block_vm_output(output)
    output = prepared.kernel.run(
        extracellular_potential_mid_mV=prepared.vext_mid,
        extracellular_potential_initial_previous_mV=prepared.vext_previous,
        intracellular_current_density_mid=prepared.iinj,
        options=options,
    )
    _block_vm_output(output)
    if output.Vm is None:
        raise ValueError("solver agreement validation requires recorded Vm output.")
    return output


def make_validation_row(
    case: AgreementCase,
    *,
    prepared: PreparedValidationInputs,
    reference_output: Any,
    candidate_output: Any,
    platform: str,
    activation_threshold_mV: float,
    activation_blanking_ms: float,
    max_abs_mV_tol: float,
    rms_mV_tol: float,
    peak_mV_tol: float,
    first_crossing_time_ms_tol: float,
) -> dict[str, Any]:
    metrics = compute_agreement_metrics(
        reference_output.Vm,
        candidate_output.Vm,
        dt_ms=case.dt_ms,
        activation_threshold_mV=activation_threshold_mV,
        activation_blanking_ms=activation_blanking_ms,
    )
    passed = (
        metrics["max_abs_mV"] <= max_abs_mV_tol
        and metrics["rms_mV"] <= rms_mV_tol
        and metrics["peak_abs_error_mV_max"] <= peak_mV_tol
        and metrics["activation_agreement"] == 1.0
        and metrics["missed_activation_count"] == 0
        and metrics["extra_activation_count"] == 0
        and _none_or_leq(
            metrics["first_crossing_time_abs_error_max_ms"],
            first_crossing_time_ms_tol,
        )
    )
    return {
        "reference_solver": case.reference_solver,
        "candidate_solver": case.candidate_solver,
        "resolved_reference_solver": resolve_e2e_solver(case.reference_solver, platform=platform),
        "resolved_candidate_solver": resolve_e2e_solver(case.candidate_solver, platform=platform),
        "batch_size": int(case.batch_size),
        "target_nx": int(case.target_nx),
        "actual_nx": int(prepared.actual_nx),
        "nt": int(case.nt),
        "dt_ms": float(case.dt_ms),
        "duration_ms": float(case.duration_ms),
        "recording": case.recording,
        "iinj_mode": case.iinj_mode,
        "setup_ms": prepared.setup_ms,
        "runtime_ms": prepared.runtime_ms,
        "vext_ms": prepared.vext_ms,
        "iinj_ms": prepared.iinj_ms,
        "vm_shape": list(getattr(candidate_output.Vm, "shape", ())),
        "activation_threshold_mV": float(activation_threshold_mV),
        "activation_blanking_ms": float(activation_blanking_ms),
        "max_abs_mV_tol": float(max_abs_mV_tol),
        "rms_mV_tol": float(rms_mV_tol),
        "peak_mV_tol": float(peak_mV_tol),
        "first_crossing_time_ms_tol": float(first_crossing_time_ms_tol),
        "passed_thresholds": bool(passed),
        **metrics,
    }


def compute_agreement_metrics(
    reference_vm: Any,
    candidate_vm: Any,
    *,
    dt_ms: float,
    activation_threshold_mV: float,
    activation_blanking_ms: float,
) -> dict[str, Any]:
    """Return scalar agreement metrics for two ``[B, Nt, R]`` Vm tensors."""

    reference = jnp.asarray(reference_vm)
    candidate = jnp.asarray(candidate_vm)
    if reference.shape != candidate.shape:
        raise ValueError(
            "reference and candidate Vm shapes must match, got "
            f"{reference.shape} and {candidate.shape}."
        )
    if reference.ndim != 3:
        raise ValueError(f"Vm outputs must have shape [B, Nt, R], got {reference.shape}.")

    batch_size, nt, record_count = (
        int(reference.shape[0]),
        int(reference.shape[1]),
        int(reference.shape[2]),
    )
    diff = candidate - reference
    abs_diff = jnp.abs(diff)
    flat_reference = jnp.reshape(reference, (batch_size, nt * record_count))
    flat_candidate = jnp.reshape(candidate, (batch_size, nt * record_count))
    reference_peak = jnp.max(flat_reference, axis=1)
    candidate_peak = jnp.max(flat_candidate, axis=1)
    peak_abs_error = jnp.abs(candidate_peak - reference_peak)

    reference_peak_index = jnp.argmax(flat_reference, axis=1)
    candidate_peak_index = jnp.argmax(flat_candidate, axis=1)
    reference_peak_time = reference_peak_index // record_count
    candidate_peak_time = candidate_peak_index // record_count
    reference_peak_position = reference_peak_index % record_count
    candidate_peak_position = candidate_peak_index % record_count

    reference_crossing = _crossing_by_time(
        reference,
        threshold_mV=activation_threshold_mV,
        blanking_ms=activation_blanking_ms,
        dt_ms=dt_ms,
    )
    candidate_crossing = _crossing_by_time(
        candidate,
        threshold_mV=activation_threshold_mV,
        blanking_ms=activation_blanking_ms,
        dt_ms=dt_ms,
    )
    reference_activated = jnp.any(reference_crossing, axis=1)
    candidate_activated = jnp.any(candidate_crossing, axis=1)
    reference_first = jnp.argmax(reference_crossing, axis=1)
    candidate_first = jnp.argmax(candidate_crossing, axis=1)
    both_activated = reference_activated & candidate_activated
    both_count = jnp.sum(both_activated)
    first_time_error = (
        jnp.abs(candidate_first - reference_first).astype(reference.dtype)
        * jnp.asarray(dt_ms, dtype=reference.dtype)
    )
    first_time_error_masked = jnp.where(both_activated, first_time_error, 0.0)
    first_time_error_max = jnp.where(
        both_count > 0,
        jnp.max(first_time_error_masked),
        jnp.asarray(jnp.nan, dtype=reference.dtype),
    )
    first_time_error_mean = jnp.where(
        both_count > 0,
        jnp.sum(first_time_error_masked) / both_count,
        jnp.asarray(jnp.nan, dtype=reference.dtype),
    )
    activation_agreement = jnp.mean(
        (reference_activated == candidate_activated).astype(reference.dtype)
    )

    values = {
        "max_abs_mV": jnp.max(abs_diff),
        "rms_mV": jnp.sqrt(jnp.mean(diff * diff)),
        "mean_abs_mV": jnp.mean(abs_diff),
        "peak_abs_error_mV_max": jnp.max(peak_abs_error),
        "peak_abs_error_mV_mean": jnp.mean(peak_abs_error),
        "peak_time_abs_error_max_ms": (
            jnp.max(jnp.abs(candidate_peak_time - reference_peak_time)).astype(reference.dtype)
            * jnp.asarray(dt_ms, dtype=reference.dtype)
        ),
        "peak_position_index_abs_error_max": jnp.max(
            jnp.abs(candidate_peak_position - reference_peak_position)
        ),
        "activation_agreement": activation_agreement,
        "reference_activated_count": jnp.sum(reference_activated),
        "candidate_activated_count": jnp.sum(candidate_activated),
        "both_activated_count": both_count,
        "missed_activation_count": jnp.sum(reference_activated & ~candidate_activated),
        "extra_activation_count": jnp.sum(~reference_activated & candidate_activated),
        "first_crossing_time_abs_error_max_ms": first_time_error_max,
        "first_crossing_time_abs_error_mean_ms": first_time_error_mean,
    }
    ready = _block_until_ready(values)
    return {
        "max_abs_mV": _float_scalar(ready["max_abs_mV"]),
        "rms_mV": _float_scalar(ready["rms_mV"]),
        "mean_abs_mV": _float_scalar(ready["mean_abs_mV"]),
        "peak_abs_error_mV_max": _float_scalar(ready["peak_abs_error_mV_max"]),
        "peak_abs_error_mV_mean": _float_scalar(ready["peak_abs_error_mV_mean"]),
        "peak_time_abs_error_max_ms": _float_scalar(ready["peak_time_abs_error_max_ms"]),
        "peak_position_index_abs_error_max": _int_scalar(
            ready["peak_position_index_abs_error_max"]
        ),
        "activation_agreement": _float_scalar(ready["activation_agreement"]),
        "reference_activated_count": _int_scalar(ready["reference_activated_count"]),
        "candidate_activated_count": _int_scalar(ready["candidate_activated_count"]),
        "both_activated_count": _int_scalar(ready["both_activated_count"]),
        "missed_activation_count": _int_scalar(ready["missed_activation_count"]),
        "extra_activation_count": _int_scalar(ready["extra_activation_count"]),
        "first_crossing_time_abs_error_max_ms": _optional_float_scalar(
            ready["first_crossing_time_abs_error_max_ms"]
        ),
        "first_crossing_time_abs_error_mean_ms": _optional_float_scalar(
            ready["first_crossing_time_abs_error_mean_ms"]
        ),
    }


def _crossing_by_time(
    vm: Any,
    *,
    threshold_mV: float,
    blanking_ms: float,
    dt_ms: float,
) -> Any:
    nt = int(vm.shape[1])
    eligible = (
        jnp.arange(nt, dtype=vm.dtype) * jnp.asarray(dt_ms, dtype=vm.dtype)
        >= jnp.asarray(blanking_ms, dtype=vm.dtype)
    )
    crossing = vm >= jnp.asarray(threshold_mV, dtype=vm.dtype)
    return jnp.any(crossing & eligible[None, :, None], axis=2)


def _group_cases_by_inputs(
    cases: Sequence[AgreementCase],
) -> tuple[tuple[tuple[Any, ...], list[AgreementCase]], ...]:
    grouped: dict[tuple[Any, ...], list[AgreementCase]] = {}
    for case in cases:
        key = (
            case.batch_size,
            case.target_nx,
            case.nt,
            case.dt_ms,
            case.recording,
            case.iinj_mode,
        )
        grouped.setdefault(key, []).append(case)
    return tuple(grouped.items())


def _block_vm_output(output: Any) -> None:
    if output.Vm is not None and hasattr(output.Vm, "block_until_ready"):
        output.Vm.block_until_ready()
        return
    _block_until_ready(output)


def _format_row(row: dict[str, Any]) -> str:
    first_crossing = row["first_crossing_time_abs_error_max_ms"]
    first_crossing_text = "n/a" if first_crossing is None else f"{first_crossing:.4g} ms"
    status = "PASS" if row["passed_thresholds"] else "CHECK"
    return (
        f"{status} {row['candidate_solver']} vs {row['reference_solver']} "
        f"B={row['batch_size']} targetNx={row['target_nx']} actualNx={row['actual_nx']} "
        f"Nt={row['nt']} rec={row['recording']} iinj={row['iinj_mode']}: "
        f"max_abs={row['max_abs_mV']:.4g} mV, rms={row['rms_mV']:.4g} mV, "
        f"peak={row['peak_abs_error_mV_max']:.4g} mV, "
        f"activation={row['activation_agreement']:.3f}, "
        f"first_crossing={first_crossing_text}"
    )


def _float_scalar(value: Any) -> float:
    return float(np.asarray(value))


def _optional_float_scalar(value: Any) -> float | None:
    scalar = float(np.asarray(value))
    return None if math.isnan(scalar) else scalar


def _int_scalar(value: Any) -> int:
    return int(np.asarray(value))


def _none_or_leq(value: float | None, threshold: float) -> bool:
    return value is None or float(value) <= float(threshold)


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\nInterrupted at {_timestamp()}.")
        raise
