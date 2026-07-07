from __future__ import annotations

import argparse
import csv
import json
import os
import platform as host_platform
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp

from axonscope.backends.jax.common import (
    solve_block_tridiagonal_2x2_pcr,
    solve_block_tridiagonal_2x2_pcr_soa,
    solve_block_tridiagonal_2x2_pcr_soa_batched,
    solve_block_tridiagonal_2x2_scalar,
    solve_block_tridiagonal_2x2_scalar_batched,
)
from axonscope.backends.jax.observer_runtime import (
    update_vm_raster_state_batch_from_tables,
)


REPEAT_FIELDS = (
    "stage",
    "variant",
    "phase",
    "repeat",
    "platform",
    "device",
    "dtype",
    "nx",
    "batch_size",
    "coefficient_mode",
    "elapsed_ms",
    "rss_delta_mib",
    "output_bytes",
)

SUMMARY_FIELDS = (
    "stage",
    "variant",
    "platform",
    "device",
    "dtype",
    "nx",
    "batch_size",
    "coefficient_mode",
    "repeats",
    "mean_ms",
    "min_ms",
    "max_ms",
    "first_run_ms",
    "rss_delta_mib_max",
    "output_bytes",
)


@dataclass(frozen=True)
class StageCase:
    stage: str
    variant: str
    fn: Callable[..., Any]
    args: tuple[Any, ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile synthetic low-level double-cable JAX stages without adding "
            "policy branches to the AxonScope runtime."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/p11b_double_cable_solver_stage_profile"),
    )
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--nx", type=int, nargs="+", default=[21, 51, 101])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[4, 32, 128])
    parser.add_argument("--dtype", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument(
        "--coefficient-mode",
        choices=("shared", "batched"),
        default="batched",
        help="Use shared cable coefficients or one coefficient row per axon.",
    )
    parser.add_argument(
        "--solver",
        action="append",
        choices=(
            "thomas_vmap",
            "thomas_batched_scan",
            "pcr_matrix_vmap",
            "pcr_soa_vmap",
            "pcr_soa_batched",
        ),
        help="Solver variant to include. Repeat to select several. Defaults to all.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be >= 1.")
    if args.warmups < 0:
        parser.error("--warmups must be >= 0.")
    if any(value < 2 for value in args.nx):
        parser.error("--nx values must be >= 2.")
    if any(value < 1 for value in args.batch_size):
        parser.error("--batch-size values must be >= 1.")

    device = _select_device(args.platform)
    if args.dtype == "fp64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float32 if args.dtype == "fp32" else jnp.float64
    requested_solvers = tuple(args.solver or (
        "thomas_vmap",
        "thomas_batched_scan",
        "pcr_matrix_vmap",
        "pcr_soa_vmap",
        "pcr_soa_batched",
    ))

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(args=args, device=device, solvers=requested_solvers)
    _write_json(args.output / "metadata.json", metadata)

    repeat_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    with jax.default_device(device):
        for nx in args.nx:
            for batch_size in args.batch_size:
                inputs = _make_inputs(
                    batch_size=batch_size,
                    nx=nx,
                    dtype=dtype,
                    coefficient_mode=args.coefficient_mode,
                )
                assembled = _block_until_ready(_assemble_system(*inputs["assemble"]))
                solver_inputs = _solver_inputs(
                    assembled,
                    coefficient_mode=args.coefficient_mode,
                )
                stage_cases = _stage_cases(
                    inputs=inputs,
                    assembled=assembled,
                    solver_inputs=solver_inputs,
                    solvers=requested_solvers,
                )
                for case in stage_cases:
                    rows, summary = _measure_case(
                        case,
                        repeats=args.repeats,
                        warmups=args.warmups,
                        platform_name=args.platform,
                        device_name=str(device),
                        dtype_name=args.dtype,
                        nx=nx,
                        batch_size=batch_size,
                        coefficient_mode=args.coefficient_mode,
                    )
                    repeat_rows.extend(rows)
                    summary_rows.append(summary)

    repeat_csv = args.output / "solver_stage_repeats.csv"
    summary_csv = args.output / "solver_stage_summary.csv"
    _write_csv(repeat_csv, REPEAT_FIELDS, repeat_rows)
    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    if not args.no_plots:
        _write_plots(args.output / "plots", summary_rows)
    _write_report(args.output / "solver_stage_report.md", summary_rows, metadata)

    print(f"wrote: {summary_csv}")
    print(f"wrote: {args.output / 'solver_stage_report.md'}")
    return 0


@jax.jit
def _assemble_system(
    Vi: Any,
    Ve: Any,
    gates: Any,
    cm_over_dt: Any,
    cx_over_dt: Any,
    gmem: Any,
    gext: Any,
    gx_abs: Any,
    left_i: Any,
    right_i: Any,
    left_e: Any,
    right_e: Any,
    off_i: Any,
    off_e: Any,
    iinj_abs: Any,
    iout_abs: Any,
    icorr_abs: Any,
    extracellular_drive_abs: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    del gates
    Vm = Vi - Ve
    a00 = cm_over_dt + gmem + left_i + right_i
    a01 = -(cm_over_dt + gmem)
    a10 = a01
    a11 = cm_over_dt + gmem + cx_over_dt + gx_abs + left_e + right_e
    rhs0 = cm_over_dt * Vm + gext + iinj_abs - iout_abs - icorr_abs
    rhs1 = (
        -cm_over_dt * Vm
        - gext
        + cx_over_dt * Ve
        + extracellular_drive_abs
        + iout_abs
        + icorr_abs
    )
    return a00, a01, a10, a11, off_i, off_e, rhs0, rhs1


@jax.jit
def _vm_gate_update(Vi: Any, Ve: Any, gates: Any) -> tuple[Any, Any]:
    Vm = Vi - Ve
    target = 1.0 / (1.0 + jnp.exp(-(Vm + 40.0) / 8.0))
    gates_new = gates + 0.05 * (target - gates)
    return Vm, gates_new


@jax.jit
def _observer_write(
    state: Any,
    vm: Any,
    probe_indices: Any,
    probe_mask: Any,
    thresholds_mV: Any,
) -> Any:
    return update_vm_raster_state_batch_from_tables(
        state,
        vm_mV=vm,
        step_index=jnp.asarray(0, dtype=jnp.int32),
        probe_indices=probe_indices,
        probe_mask=probe_mask,
        thresholds_mV=thresholds_mV,
    )


@jax.jit
def _solve_thomas_vmap_shared(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return jax.vmap(
        lambda r0, r1: solve_block_tridiagonal_2x2_scalar(
            a00, a01, a10, a11, off0, off1, r0, r1
        )
    )(rhs0, rhs1)


@jax.jit
def _solve_thomas_vmap_batched(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return jax.vmap(solve_block_tridiagonal_2x2_scalar)(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_thomas_batched_scan(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_scalar_batched(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_pcr_matrix_vmap_shared(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return jax.vmap(
        lambda r0, r1: solve_block_tridiagonal_2x2_pcr(
            a00, a01, a10, a11, off0, off1, r0, r1
        )
    )(rhs0, rhs1)


@jax.jit
def _solve_pcr_matrix_vmap_batched(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return jax.vmap(solve_block_tridiagonal_2x2_pcr)(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_pcr_soa_vmap_shared(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return jax.vmap(
        lambda r0, r1: solve_block_tridiagonal_2x2_pcr_soa(
            a00, a01, a10, a11, off0, off1, r0, r1
        )
    )(rhs0, rhs1)


@jax.jit
def _solve_pcr_soa_vmap_batched(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return jax.vmap(solve_block_tridiagonal_2x2_pcr_soa)(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_pcr_soa_batched(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_pcr_soa_batched(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


def _make_inputs(
    *,
    batch_size: int,
    nx: int,
    dtype: Any,
    coefficient_mode: str,
) -> dict[str, Any]:
    x = jnp.linspace(0.0, 1.0, nx, dtype=dtype)
    edge_x = jnp.linspace(0.0, 1.0, nx - 1, dtype=dtype)
    row = jnp.arange(batch_size, dtype=dtype)[:, None]
    row_scale = 1.0 + 0.001 * row

    Vi = -70.0 + 0.25 * x[None, :] + 0.005 * row
    Ve = 0.02 * jnp.sin(6.283185307179586 * x)[None, :] * row_scale
    gates = jnp.full((batch_size, nx), 0.35, dtype=dtype) + 0.01 * x[None, :]

    cm_over_dt = row_scale * (0.20 + 0.02 * x[None, :])
    cx_over_dt = row_scale * (0.04 + 0.01 * x[None, :])
    gmem = row_scale * (0.012 + 0.002 * gates)
    gext = row_scale * (0.001 + 0.0001 * x[None, :])
    gx_abs = row_scale * (0.008 + 0.001 * x[None, :])
    left_i = row_scale * (0.018 + 0.002 * x[None, :])
    right_i = row_scale * (0.018 + 0.002 * (1.0 - x)[None, :])
    left_e = row_scale * (0.012 + 0.001 * x[None, :])
    right_e = row_scale * (0.012 + 0.001 * (1.0 - x)[None, :])
    off_i = -(0.015 + 0.001 * edge_x[None, :]) * row_scale
    off_e = -(0.010 + 0.001 * edge_x[None, :]) * row_scale
    iinj_abs = row_scale * (0.0002 * jnp.cos(6.283185307179586 * x)[None, :])
    iout_abs = row_scale * (0.0003 + 0.0001 * x[None, :])
    icorr_abs = row_scale * (0.00001 * jnp.sin(3.141592653589793 * x)[None, :])
    extracellular_drive_abs = row_scale * (0.0005 * jnp.sin(6.283185307179586 * x)[None, :])

    if coefficient_mode == "shared":
        cm_over_dt = cm_over_dt.at[:].set(cm_over_dt[0])
        cx_over_dt = cx_over_dt.at[:].set(cx_over_dt[0])
        gmem = gmem.at[:].set(gmem[0])
        gext = gext.at[:].set(gext[0])
        gx_abs = gx_abs.at[:].set(gx_abs[0])
        left_i = left_i.at[:].set(left_i[0])
        right_i = right_i.at[:].set(right_i[0])
        left_e = left_e.at[:].set(left_e[0])
        right_e = right_e.at[:].set(right_e[0])
        off_i = off_i.at[:].set(off_i[0])
        off_e = off_e.at[:].set(off_e[0])

    probe_count = min(8, nx)
    probe_indices_1d = jnp.linspace(0, nx - 1, probe_count, dtype=jnp.int32)
    probe_indices = jnp.broadcast_to(probe_indices_1d[None, None, :], (batch_size, 1, probe_count))
    probe_mask = jnp.ones((batch_size, 1, probe_count), dtype=jnp.bool_)
    thresholds = jnp.asarray([-20.0], dtype=dtype)
    observer_state = jnp.zeros((batch_size, 1, probe_count, 1), dtype=jnp.uint32)

    return {
        "assemble": (
            Vi,
            Ve,
            gates,
            cm_over_dt,
            cx_over_dt,
            gmem,
            gext,
            gx_abs,
            left_i,
            right_i,
            left_e,
            right_e,
            off_i,
            off_e,
            iinj_abs,
            iout_abs,
            icorr_abs,
            extracellular_drive_abs,
        ),
        "vm_gate": (Vi, Ve, gates),
        "observer": (observer_state, Vi - Ve, probe_indices, probe_mask, thresholds),
    }


def _solver_inputs(
    assembled: tuple[Any, Any, Any, Any, Any, Any, Any, Any],
    *,
    coefficient_mode: str,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    a00, a01, a10, a11, off0, off1, rhs0, rhs1 = assembled
    if coefficient_mode == "shared":
        return a00[0], a01[0], a10[0], a11[0], off0[0], off1[0], rhs0, rhs1
    return assembled


def _stage_cases(
    *,
    inputs: dict[str, Any],
    assembled: tuple[Any, Any, Any, Any, Any, Any, Any, Any],
    solver_inputs: tuple[Any, Any, Any, Any, Any, Any, Any, Any],
    solvers: Sequence[str],
) -> list[StageCase]:
    cases = [
        StageCase("assemble_system", "synthetic_double_cable", _assemble_system, inputs["assemble"]),
        StageCase("vm_gate_update", "synthetic_gate", _vm_gate_update, inputs["vm_gate"]),
        StageCase("observer_write", "vm_raster_batch", _observer_write, inputs["observer"]),
    ]
    coefficient_mode = "shared" if jnp.asarray(solver_inputs[0]).ndim == 1 else "batched"
    solver_map: dict[str, Callable[..., Any]]
    if coefficient_mode == "shared":
        solver_map = {
            "thomas_vmap": _solve_thomas_vmap_shared,
            "thomas_batched_scan": _solve_thomas_batched_scan,
            "pcr_matrix_vmap": _solve_pcr_matrix_vmap_shared,
            "pcr_soa_vmap": _solve_pcr_soa_vmap_shared,
            "pcr_soa_batched": _solve_pcr_soa_batched,
        }
    else:
        solver_map = {
            "thomas_vmap": _solve_thomas_vmap_batched,
            "thomas_batched_scan": _solve_thomas_batched_scan,
            "pcr_matrix_vmap": _solve_pcr_matrix_vmap_batched,
            "pcr_soa_vmap": _solve_pcr_soa_vmap_batched,
            "pcr_soa_batched": _solve_pcr_soa_batched,
        }
    for solver in solvers:
        cases.append(StageCase("block_solve", solver, solver_map[solver], solver_inputs))
    cases.append(StageCase("full_numeric_step", "pcr_soa_batched_update", _full_numeric_step, assembled))
    return cases


@jax.jit
def _full_numeric_step(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    vi, ve = solve_block_tridiagonal_2x2_pcr_soa_batched(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )
    return vi - ve, vi + ve


def _measure_case(
    case: StageCase,
    *,
    repeats: int,
    warmups: int,
    platform_name: str,
    device_name: str,
    dtype_name: str,
    nx: int,
    batch_size: int,
    coefficient_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    rss_start = _rss_mib()
    start = perf_counter()
    first_out = _block_until_ready(case.fn(*case.args))
    first_run_ms = (perf_counter() - start) * 1000.0
    rss_end = _rss_mib()
    output_bytes = _output_nbytes(first_out)
    rows.append(
        _repeat_row(
            case=case,
            phase="first_run",
            repeat=0,
            platform_name=platform_name,
            device_name=device_name,
            dtype_name=dtype_name,
            nx=nx,
            batch_size=batch_size,
            coefficient_mode=coefficient_mode,
            elapsed_ms=first_run_ms,
            rss_delta_mib=_delta(rss_start, rss_end),
            output_bytes=output_bytes,
        )
    )

    for index in range(warmups):
        start = perf_counter()
        _block_until_ready(case.fn(*case.args))
        rows.append(
            _repeat_row(
                case=case,
                phase="warmup",
                repeat=index,
                platform_name=platform_name,
                device_name=device_name,
                dtype_name=dtype_name,
                nx=nx,
                batch_size=batch_size,
                coefficient_mode=coefficient_mode,
                elapsed_ms=(perf_counter() - start) * 1000.0,
                rss_delta_mib=None,
                output_bytes=output_bytes,
            )
        )

    measured: list[float] = []
    rss_deltas: list[float] = []
    for index in range(repeats):
        rss_start = _rss_mib()
        start = perf_counter()
        out = _block_until_ready(case.fn(*case.args))
        elapsed_ms = (perf_counter() - start) * 1000.0
        rss_end = _rss_mib()
        output_bytes = _output_nbytes(out)
        rss_delta = _delta(rss_start, rss_end)
        if rss_delta is not None:
            rss_deltas.append(rss_delta)
        measured.append(elapsed_ms)
        rows.append(
            _repeat_row(
                case=case,
                phase="measured",
                repeat=index,
                platform_name=platform_name,
                device_name=device_name,
                dtype_name=dtype_name,
                nx=nx,
                batch_size=batch_size,
                coefficient_mode=coefficient_mode,
                elapsed_ms=elapsed_ms,
                rss_delta_mib=rss_delta,
                output_bytes=output_bytes,
            )
        )

    summary = {
        "stage": case.stage,
        "variant": case.variant,
        "platform": platform_name,
        "device": device_name,
        "dtype": dtype_name,
        "nx": nx,
        "batch_size": batch_size,
        "coefficient_mode": coefficient_mode,
        "repeats": repeats,
        "mean_ms": sum(measured) / len(measured),
        "min_ms": min(measured),
        "max_ms": max(measured),
        "first_run_ms": first_run_ms,
        "rss_delta_mib_max": max(rss_deltas) if rss_deltas else None,
        "output_bytes": output_bytes,
    }
    return rows, summary


def _repeat_row(
    *,
    case: StageCase,
    phase: str,
    repeat: int,
    platform_name: str,
    device_name: str,
    dtype_name: str,
    nx: int,
    batch_size: int,
    coefficient_mode: str,
    elapsed_ms: float,
    rss_delta_mib: float | None,
    output_bytes: int,
) -> dict[str, Any]:
    return {
        "stage": case.stage,
        "variant": case.variant,
        "phase": phase,
        "repeat": repeat,
        "platform": platform_name,
        "device": device_name,
        "dtype": dtype_name,
        "nx": nx,
        "batch_size": batch_size,
        "coefficient_mode": coefficient_mode,
        "elapsed_ms": elapsed_ms,
        "rss_delta_mib": rss_delta_mib,
        "output_bytes": output_bytes,
    }


def _block_until_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_block_until_ready(item) for item in value)
    if isinstance(value, list):
        return [_block_until_ready(item) for item in value]
    block = getattr(value, "block_until_ready", None)
    if callable(block):
        block()
    return value


def _output_nbytes(value: Any) -> int:
    if isinstance(value, (tuple, list)):
        return sum(_output_nbytes(item) for item in value)
    nbytes = getattr(value, "nbytes", None)
    return int(nbytes or 0)


def _select_device(platform_name: str) -> Any:
    devices = jax.devices(platform_name)
    if not devices:
        raise RuntimeError(f"No JAX {platform_name} device is available.")
    return devices[0]


def _rss_mib() -> float | None:
    try:
        import psutil
    except Exception:
        return None
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


def _delta(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return end - start


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _metadata(*, args: argparse.Namespace, device: Any, solvers: Sequence[str]) -> dict[str, Any]:
    return {
        "script": "benchmark/analysis/double_cable_solver_stage_profile.py",
        "purpose": "Synthetic low-level double-cable solver stage cartography.",
        "platform": args.platform,
        "device": str(device),
        "jax_version": jax.__version__,
        "python": platform_python(),
        "host": {
            "system": host_platform.system(),
            "release": host_platform.release(),
            "machine": host_platform.machine(),
            "processor": host_platform.processor(),
        },
        "git": _git_metadata(),
        "options": {
            "nx": args.nx,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "coefficient_mode": args.coefficient_mode,
            "solvers": list(solvers),
            "repeats": args.repeats,
            "warmups": args.warmups,
        },
        "limitations": [
            "Synthetic coefficients isolate numerical stage costs; they are not a full AxonSimulation run.",
            "Membrane model compilation and public result assembly are measured by the curve benchmarks.",
            "Use this report to pick low-level solver targets, not runtime policy defaults.",
        ],
    }


def platform_python() -> str:
    return host_platform.python_version()


def _git_metadata() -> dict[str, Any]:
    def run_git(*cmd: str) -> str | None:
        try:
            result = subprocess.run(
                ("git", *cmd),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status,
    }


def _write_report(path: Path, rows: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> None:
    fastest_by_case: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        if row["stage"] != "block_solve":
            continue
        key = (str(row["dtype"]), int(row["nx"]), int(row["batch_size"]))
        current = fastest_by_case.get(key)
        if current is None or float(row["mean_ms"]) < float(current["mean_ms"]):
            fastest_by_case[key] = row

    lines = [
        "# Double-Cable Solver Stage Profile",
        "",
        "Synthetic low-level cartography for the JAX double-cable numerical path.",
        "This does not choose runtime policy; it exposes stage costs to guide low-level optimization.",
        "",
        "## Context",
        "",
        f"- Platform: `{metadata['platform']}`",
        f"- Device: `{metadata['device']}`",
        f"- JAX: `{metadata['jax_version']}`",
        f"- Git commit: `{metadata['git'].get('commit')}`",
        f"- Git dirty: `{metadata['git'].get('dirty')}`",
        "",
        "## Fastest Block Solver By Shape",
        "",
        "| dtype | Nx | batch | variant | mean ms | max ms |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for key in sorted(fastest_by_case):
        row = fastest_by_case[key]
        lines.append(
            "| {dtype} | {nx} | {batch_size} | {variant} | {mean_ms:.3f} | {max_ms:.3f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Stage Means",
            "",
            "| stage | variant | Nx | batch | mean ms | first run ms | output KiB |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["stage"], int(item["nx"]), int(item["batch_size"]), item["variant"])):
        lines.append(
            "| {stage} | {variant} | {nx} | {batch_size} | {mean_ms:.3f} | {first_run_ms:.3f} | {output_kib:.1f} |".format(
                output_kib=float(row["output_bytes"]) / 1024.0,
                **row,
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `assemble_system` covers coefficient/RHS arithmetic only.",
            "- `block_solve` compares Thomas, matrix PCR, SoA PCR, and batch-native alternatives where selected.",
            "- `observer_write` calls the current VmRaster batch write helper with synthetic probes.",
            "- `full_numeric_step` is a compact solve+Vm arithmetic proxy, not the public runtime loop.",
            "- `first_run_ms` may include compilation unless the same jitted function was already compiled during setup.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "PLOTS_SKIPPED.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    block_rows = [row for row in rows if row["stage"] == "block_solve"]
    for nx in sorted({int(row["nx"]) for row in block_rows}):
        for batch_size in sorted({int(row["batch_size"]) for row in block_rows}):
            subset = [
                row
                for row in block_rows
                if int(row["nx"]) == nx and int(row["batch_size"]) == batch_size
            ]
            if not subset:
                continue
            subset.sort(key=lambda row: float(row["mean_ms"]))
            labels = [str(row["variant"]) for row in subset]
            values = [float(row["mean_ms"]) for row in subset]
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.bar(labels, values, color="#4b6f8f")
            ax.set_ylabel("mean measured time (ms)")
            ax.set_title(f"Double-cable block solve, Nx={nx}, batch={batch_size}")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(output_dir / f"block_solve_nx{nx}_batch{batch_size}.png", dpi=160)
            plt.close(fig)

    stage_rows = [row for row in rows if int(row["nx"]) == max(int(item["nx"]) for item in rows)]
    stage_rows = [
        row
        for row in stage_rows
        if int(row["batch_size"]) == max(int(item["batch_size"]) for item in rows)
    ]
    labels = [f"{row['stage']}:{row['variant']}" for row in stage_rows]
    values = [float(row["mean_ms"]) for row in stage_rows]
    if labels:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.bar(labels, values, color="#6f7f4b")
        ax.set_ylabel("mean measured time (ms)")
        ax.set_title("Largest requested shape: stage overview")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output_dir / "stage_overview_largest_shape.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
