from __future__ import annotations

import argparse
import csv
import json
import os
import platform as host_platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from axonscope.backends.jax.common import (
    double_cable_block_residual_norm,
    solve_block_tridiagonal_2x2_pcr_soa_batched,
    solve_block_tridiagonal_2x2_scalar_batched,
)
from axonscope.backends.jax.jax_triton_double_cable import (
    solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_batched,
    solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_xb,
    solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_batched,
)
from axonscope.backends.jax.large_population_solver import (
    LargePopulationLayoutName,
    block_b_candidates_for_nx_bucket,
    make_large_population_layout_plan,
    select_large_population_nx_bucket,
    solve_large_population_exact_double_cable_jax,
)


SUMMARY_FIELDS = (
    "variant",
    "layout",
    "platform",
    "device",
    "dtype",
    "batch_size",
    "nx_true",
    "nx_pad",
    "block_b",
    "n_tiles",
    "coefficient_mode",
    "repeats",
    "mean_ms",
    "min_ms",
    "max_ms",
    "first_run_ms",
    "rss_delta_mib_max",
    "output_bytes",
    "node_solves_per_s",
    "axon_steps_per_s",
    "max_residual_norm",
    "median_residual_norm",
)

REPEAT_FIELDS = (
    "variant",
    "layout",
    "phase",
    "repeat",
    "platform",
    "device",
    "dtype",
    "batch_size",
    "nx_true",
    "nx_pad",
    "block_b",
    "n_tiles",
    "coefficient_mode",
    "elapsed_ms",
    "rss_delta_mib",
    "output_bytes",
)


@dataclass(frozen=True)
class SolverCase:
    variant: str
    layout: str
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    batch_size: int
    nx_true: int
    nx_pad: int
    block_b: int | None
    n_tiles: int | None
    coefficient_mode: str
    validation_args: tuple[Any, ...] | None = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "P11C benchmark-private large-population exact double-cable solver "
            "profile. This does not add or change public solver routing."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/p11c_large_population_solver_profile"),
    )
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--nx", type=int, nargs="+", default=[47, 89, 129])
    parser.add_argument("--dtype", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument(
        "--coefficient-mode",
        choices=("shared", "batched", "both"),
        default="batched",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=(
            "current_pcr_soa",
            "thomas_batched_scan",
            "jax_triton_tiled_thomas",
            "jax_triton_tiled_thomas_loop",
            "large_population_exact_double_cable_jax",
        ),
        help=(
            "Variant to include. Repeat to select several. Defaults to current "
            "PCR-SoA plus the tiled large-population candidate."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("BX", "XB", "TILED"),
        default="TILED",
        help="Layout for the large-population candidate.",
    )
    parser.add_argument(
        "--block-b",
        type=int,
        nargs="+",
        help="Tile widths for the large-population candidate. Defaults depend on Nx bucket.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be >= 1.")
    if args.warmups < 0:
        parser.error("--warmups must be >= 0.")
    if any(value < 1 for value in args.batch_size):
        parser.error("--batch-size values must be >= 1.")
    if any(value < 2 for value in args.nx):
        parser.error("--nx values must be >= 2.")
    if args.block_b and any(value < 1 for value in args.block_b):
        parser.error("--block-b values must be >= 1.")

    if args.dtype == "fp64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float32 if args.dtype == "fp32" else jnp.float64
    device = _select_device(args.platform)
    variants = tuple(args.variant or ("current_pcr_soa", "large_population_exact_double_cable_jax"))
    coefficient_modes = (
        ("shared", "batched")
        if args.coefficient_mode == "both"
        else (str(args.coefficient_mode),)
    )

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(args=args, device=device, variants=variants)
    _write_json(args.output / "metadata.json", metadata)

    repeat_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    with jax.default_device(device):
        for batch_size in args.batch_size:
            for nx_true in args.nx:
                nx_pad = select_large_population_nx_bucket(nx_true)
                block_bs = tuple(args.block_b or block_b_candidates_for_nx_bucket(nx_pad))
                for coefficient_mode in coefficient_modes:
                    inputs = _make_inputs(
                        batch_size=batch_size,
                        nx=nx_true,
                        dtype=dtype,
                        coefficient_mode=coefficient_mode,
                    )
                    cases = _cases(
                        variants=variants,
                        inputs=inputs,
                        batch_size=batch_size,
                        nx_true=nx_true,
                        nx_pad=nx_pad,
                        block_bs=block_bs,
                        layout=args.layout,
                        coefficient_mode=coefficient_mode,
                    )
                    for case in cases:
                        rows, summary = _measure_case(
                            case,
                            repeats=args.repeats,
                            warmups=args.warmups,
                            platform_name=args.platform,
                            device_name=str(device),
                            dtype_name=args.dtype,
                        )
                        repeat_rows.extend(rows)
                        summary_rows.append(summary)

    _write_csv(args.output / "large_population_solver_repeats.csv", REPEAT_FIELDS, repeat_rows)
    _write_csv(args.output / "large_population_solver_summary.csv", SUMMARY_FIELDS, summary_rows)
    if not args.no_plots:
        _write_plots(args.output / "plots", summary_rows)
    _write_report(args.output / "large_population_solver_report.md", summary_rows, metadata)
    print(f"wrote: {args.output / 'large_population_solver_summary.csv'}")
    print(f"wrote: {args.output / 'large_population_solver_report.md'}")
    return 0


@jax.jit
def _solve_current(
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
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
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
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    )


@jax.jit
def _solve_jax_triton_tiled_thomas(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_batched(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        block_b=128,
    )


@jax.jit
def _solve_jax_triton_tiled_thomas_loop(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_batched(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        block_b=64,
    )


def _solve_jax_triton_tiled_thomas_with_block_b(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
    *,
    block_b: int,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_batched(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        block_b=int(block_b),
    )


def _solve_jax_triton_tiled_thomas_loop_with_block_b(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
    *,
    block_b: int,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_batched(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        block_b=int(block_b),
    )


def _solve_jax_triton_tiled_thomas_loop_xb_to_bx(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
    *,
    block_b: int,
) -> tuple[Any, Any]:
    out0_xb, out1_xb = solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_xb(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        block_b=int(block_b),
    )
    return jnp.swapaxes(out0_xb, 0, 1), jnp.swapaxes(out1_xb, 0, 1)


def _make_block_b_solver(fn: Callable[..., Any], *, block_b: int) -> Callable[..., Any]:
    return jax.jit(partial(fn, block_b=int(block_b)))


def _space_to_xb(values: Any) -> Any:
    arr = jnp.asarray(values)
    return jnp.swapaxes(arr, 0, 1) if arr.ndim == 2 else arr


def _edge_to_xb(values: Any) -> Any:
    arr = jnp.asarray(values)
    return jnp.swapaxes(arr, 0, 1) if arr.ndim == 2 else arr


def _inputs_to_xb(inputs: tuple[Any, Any, Any, Any, Any, Any, Any, Any]) -> tuple[Any, ...]:
    a00, a01, a10, a11, off0, off1, rhs0, rhs1 = inputs
    return (
        _space_to_xb(a00),
        _space_to_xb(a01),
        _space_to_xb(a10),
        _space_to_xb(a11),
        _edge_to_xb(off0),
        _edge_to_xb(off1),
        _space_to_xb(rhs0),
        _space_to_xb(rhs1),
    )


def _cases(
    *,
    variants: Sequence[str],
    inputs: tuple[Any, ...],
    batch_size: int,
    nx_true: int,
    nx_pad: int,
    block_bs: Sequence[int],
    layout: LargePopulationLayoutName,
    coefficient_mode: str,
) -> list[SolverCase]:
    cases: list[SolverCase] = []
    for variant in variants:
        if variant == "current_pcr_soa":
            cases.append(
                SolverCase(
                    variant=variant,
                    layout="BX",
                    fn=_solve_current,
                    args=inputs,
                    batch_size=batch_size,
                    nx_true=nx_true,
                    nx_pad=nx_true,
                    block_b=None,
                    n_tiles=None,
                    coefficient_mode=coefficient_mode,
                )
            )
            continue
        if variant == "thomas_batched_scan":
            cases.append(
                SolverCase(
                    variant=variant,
                    layout="XB_SCAN",
                    fn=_solve_thomas_batched_scan,
                    args=inputs,
                    batch_size=batch_size,
                    nx_true=nx_true,
                    nx_pad=nx_true,
                    block_b=None,
                    n_tiles=None,
                    coefficient_mode=coefficient_mode,
                )
            )
            continue
        if variant == "jax_triton_tiled_thomas":
            for block_b in tuple(block_bs or (128,)):
                cases.append(
                    SolverCase(
                        variant=variant,
                        layout="BX_WRAPPER_TRITON_TILED",
                        fn=_make_block_b_solver(
                            _solve_jax_triton_tiled_thomas_with_block_b,
                            block_b=int(block_b),
                        ),
                        args=inputs,
                        batch_size=batch_size,
                        nx_true=nx_true,
                        nx_pad=nx_true,
                        block_b=int(block_b),
                        n_tiles=(batch_size + int(block_b) - 1) // int(block_b),
                        coefficient_mode=coefficient_mode,
                    )
                )
            continue
        if variant == "jax_triton_tiled_thomas_loop":
            inputs_xb = _inputs_to_xb(inputs)
            for block_b in tuple(block_bs or (64,)):
                cases.append(
                    SolverCase(
                        variant=variant,
                        layout="BX_WRAPPER_TRITON_TILED_LOOP",
                        fn=_make_block_b_solver(
                            _solve_jax_triton_tiled_thomas_loop_with_block_b,
                            block_b=int(block_b),
                        ),
                        args=inputs,
                        batch_size=batch_size,
                        nx_true=nx_true,
                        nx_pad=nx_true,
                        block_b=int(block_b),
                        n_tiles=(batch_size + int(block_b) - 1) // int(block_b),
                        coefficient_mode=coefficient_mode,
                    )
                )
                cases.append(
                    SolverCase(
                        variant=f"{variant}_xb",
                        layout="XB_DIRECT_TRITON_TILED_LOOP",
                        fn=_make_block_b_solver(
                            _solve_jax_triton_tiled_thomas_loop_xb_to_bx,
                            block_b=int(block_b),
                        ),
                        args=inputs_xb,
                        batch_size=batch_size,
                        nx_true=nx_true,
                        nx_pad=nx_true,
                        block_b=int(block_b),
                        n_tiles=(batch_size + int(block_b) - 1) // int(block_b),
                        coefficient_mode=coefficient_mode,
                        validation_args=inputs,
                    )
                )
            continue
        for block_b in block_bs:
            plan = make_large_population_layout_plan(
                batch_size=batch_size,
                nx_true=nx_true,
                nx_pad=nx_pad,
                block_b=int(block_b),
                layout=layout,
            )
            solve = jax.jit(
                partial(
                    solve_large_population_exact_double_cable_jax,
                    nx_pad=plan.nx_pad,
                    block_b=plan.block_b,
                    layout=plan.layout,
                )
            )
            cases.append(
                SolverCase(
                    variant=variant,
                    layout=layout,
                    fn=solve,
                    args=inputs,
                    batch_size=batch_size,
                    nx_true=nx_true,
                    nx_pad=plan.nx_pad,
                    block_b=plan.block_b,
                    n_tiles=plan.n_tiles,
                    coefficient_mode=coefficient_mode,
                )
            )
    return cases


def _make_inputs(
    *,
    batch_size: int,
    nx: int,
    dtype: Any,
    coefficient_mode: str,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    x = jnp.linspace(0.0, 1.0, nx, dtype=dtype)
    edge_x = jnp.linspace(0.0, 1.0, nx - 1, dtype=dtype)
    row = jnp.arange(batch_size, dtype=dtype)[:, None]
    row_scale = 1.0 + 0.0001 * row

    a00 = row_scale * (2.0 + 0.10 * x[None, :])
    a01 = row_scale * (-0.20 - 0.02 * x[None, :])
    a10 = row_scale * (-0.18 - 0.01 * x[None, :])
    a11 = row_scale * (2.4 + 0.12 * x[None, :])
    off0 = row_scale * (-(0.03 + 0.005 * edge_x[None, :]))
    off1 = row_scale * (-(0.02 + 0.004 * edge_x[None, :]))
    rhs0 = row_scale * (0.5 + 0.05 * jnp.sin(6.283185307179586 * x)[None, :])
    rhs1 = row_scale * (-0.25 + 0.04 * jnp.cos(6.283185307179586 * x)[None, :])

    if coefficient_mode == "shared":
        return a00[0], a01[0], a10[0], a11[0], off0[0], off1[0], rhs0, rhs1
    return a00, a01, a10, a11, off0, off1, rhs0, rhs1


def _measure_case(
    case: SolverCase,
    *,
    repeats: int,
    warmups: int,
    platform_name: str,
    device_name: str,
    dtype_name: str,
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
                elapsed_ms=(perf_counter() - start) * 1000.0,
                rss_delta_mib=None,
                output_bytes=output_bytes,
            )
        )

    measured: list[float] = []
    rss_deltas: list[float] = []
    out = first_out
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
                elapsed_ms=elapsed_ms,
                rss_delta_mib=rss_delta,
                output_bytes=output_bytes,
            )
        )

    residual_args = case.validation_args if case.validation_args is not None else case.args
    max_residual, median_residual = _residual_stats(residual_args, out)
    mean_ms = sum(measured) / len(measured)
    node_solves = float(case.batch_size * case.nx_true)
    return rows, {
        "variant": case.variant,
        "layout": case.layout,
        "platform": platform_name,
        "device": device_name,
        "dtype": dtype_name,
        "batch_size": case.batch_size,
        "nx_true": case.nx_true,
        "nx_pad": case.nx_pad,
        "block_b": case.block_b,
        "n_tiles": case.n_tiles,
        "coefficient_mode": case.coefficient_mode,
        "repeats": repeats,
        "mean_ms": mean_ms,
        "min_ms": min(measured),
        "max_ms": max(measured),
        "first_run_ms": first_run_ms,
        "rss_delta_mib_max": max(rss_deltas) if rss_deltas else None,
        "output_bytes": output_bytes,
        "node_solves_per_s": node_solves / (mean_ms * 1e-3),
        "axon_steps_per_s": float(case.batch_size) / (mean_ms * 1e-3),
        "max_residual_norm": max_residual,
        "median_residual_norm": median_residual,
    }


def _repeat_row(
    *,
    case: SolverCase,
    phase: str,
    repeat: int,
    platform_name: str,
    device_name: str,
    dtype_name: str,
    elapsed_ms: float,
    rss_delta_mib: float | None,
    output_bytes: int,
) -> dict[str, Any]:
    return {
        "variant": case.variant,
        "layout": case.layout,
        "phase": phase,
        "repeat": repeat,
        "platform": platform_name,
        "device": device_name,
        "dtype": dtype_name,
        "batch_size": case.batch_size,
        "nx_true": case.nx_true,
        "nx_pad": case.nx_pad,
        "block_b": case.block_b,
        "n_tiles": case.n_tiles,
        "coefficient_mode": case.coefficient_mode,
        "elapsed_ms": elapsed_ms,
        "rss_delta_mib": rss_delta_mib,
        "output_bytes": output_bytes,
    }


def _residual_stats(
    inputs: tuple[Any, Any, Any, Any, Any, Any, Any, Any],
    output: tuple[Any, Any],
) -> tuple[float, float]:
    residuals = _block_until_ready(double_cable_block_residual_norm(*inputs, *output))
    return float(jnp.max(residuals)), float(jnp.median(residuals))


def _block_until_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_block_until_ready(item) for item in value)
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
    return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)


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


def _metadata(
    *,
    args: argparse.Namespace,
    device: Any,
    variants: Sequence[str],
) -> dict[str, Any]:
    return {
        "script": "benchmark/analysis/large_population_double_cable_solver_profile.py",
        "purpose": "P11C benchmark-private large-population exact double-cable solver gate.",
        "platform": args.platform,
        "device": str(device),
        "jax_version": jax.__version__,
        "python": host_platform.python_version(),
        "host": {
            "system": host_platform.system(),
            "release": host_platform.release(),
            "machine": host_platform.machine(),
            "processor": host_platform.processor(),
        },
        "git": _git_metadata(),
        "options": {
            "batch_size": args.batch_size,
            "nx": args.nx,
            "dtype": args.dtype,
            "coefficient_mode": args.coefficient_mode,
            "variants": list(variants),
            "layout": args.layout,
            "block_b": args.block_b,
            "repeats": args.repeats,
            "warmups": args.warmups,
        },
    }


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
    lines = [
        "# P11C Large-Population Solver Profile",
        "",
        "Benchmark-private cartography for a large-population exact double-cable JAX candidate.",
        "This report does not change public solver routing.",
        "",
        "## Context",
        "",
        f"- Platform: `{metadata['platform']}`",
        f"- Device: `{metadata['device']}`",
        f"- JAX: `{metadata['jax_version']}`",
        f"- Git commit: `{metadata['git'].get('commit')}`",
        f"- Git dirty: `{metadata['git'].get('dirty')}`",
        "",
        "## Fastest Variant By Shape",
        "",
        "| B | Nx | coeffs | variant | layout | Nx pad | block B | mean ms | node-solves/s | max residual |",
        "| ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _fastest_rows(rows):
        report_row = _report_row(row)
        lines.append(
            "| {batch_size} | {nx_true} | {coefficient_mode} | {variant} | {layout} | {nx_pad} | {block_b} | {mean_ms:.3f} | {node_solves_per_s:.3e} | {max_residual_norm:.3e} |".format(
                **report_row,
            )
        )

    lines.extend(
        [
            "",
            "## Stage Means",
            "",
            "| variant | layout | coeffs | B | Nx | Nx pad | block B | mean ms | first run ms | output KiB | max residual |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        rows,
        key=lambda item: (
            int(item["batch_size"]),
            int(item["nx_true"]),
            str(item["coefficient_mode"]),
            str(item["variant"]),
            int(item["block_b"] or 0),
        ),
    ):
        report_row = _report_row(row)
        lines.append(
            "| {variant} | {layout} | {coefficient_mode} | {batch_size} | {nx_true} | {nx_pad} | {block_b} | {mean_ms:.3f} | {first_run_ms:.3f} | {output_kib:.1f} | {max_residual_norm:.3e} |".format(
                **report_row,
            )
        )

    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- Promote nothing from this benchmark alone.",
            "- The candidate must win real-stage large-population gates before runtime integration.",
            "- If the candidate wins only through HLO/solver-only counters, reject or keep it benchmark-only.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_row(row: dict[str, Any]) -> dict[str, Any]:
    formatted = dict(row)
    formatted["block_b"] = "" if row.get("block_b") is None else row.get("block_b")
    formatted["output_kib"] = float(row["output_bytes"]) / 1024.0
    return formatted


def _fastest_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["batch_size"]), int(row["nx_true"]), str(row["coefficient_mode"]))
        current = best.get(key)
        if current is None or float(row["mean_ms"]) < float(current["mean_ms"]):
            best[key] = dict(row)
    return [best[key] for key in sorted(best)]


def _write_plots(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "PLOTS_SKIPPED.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for nx in sorted({int(row["nx_true"]) for row in rows}):
        subset = [row for row in rows if int(row["nx_true"]) == nx]
        subset.sort(key=lambda row: (int(row["batch_size"]), str(row["variant"]), int(row["block_b"] or 0)))
        labels = [
            f"B{row['batch_size']}\n{row['variant']}\n{row['block_b'] or ''}"
            for row in subset
        ]
        values = [float(row["mean_ms"]) for row in subset]
        fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.7), 4.8))
        ax.bar(labels, values, color="#4f6f7f")
        ax.set_ylabel("mean measured time (ms)")
        ax.set_title(f"P11C large-pop exact double-cable, Nx={nx}")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(output_dir / f"p11c_large_population_nx{nx}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
