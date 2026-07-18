"""Gate a benchmark-only scalar Triton Thomas solve against JAX/cuSPARSE."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from axonscope.runtime.jax.kernels.triton_single_cable import (
    single_cable_triton_dependency_skip_reason,
    solve_single_cable_tridiagonal_xb,
)


DEFAULT_OUTPUT = Path("benchmark/results/single_cable_triton_gate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="quick")
    parser.add_argument("--platform", choices=("gpu",), default="gpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nx", default="200")
    parser.add_argument("--batch-sizes", default="5120,20480")
    parser.add_argument("--block-b", default="64,128,256")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    nx_values = _positive_ints(args.nx, name="nx")
    batch_sizes = _positive_ints(args.batch_sizes, name="batch-sizes")
    block_sizes = _positive_ints(args.block_b, name="block-b")
    if min(nx_values) < 2:
        raise ValueError("--nx values must be >= 2.")
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("--warmups must be >= 0 and --repeats must be >= 1.")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "script": "single_cable_triton_gate",
        "preset": args.preset,
        "platform": args.platform,
        "nx": nx_values,
        "batch_sizes": batch_sizes,
        "block_b": block_sizes,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": args.seed,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0

    skip_reason = single_cable_triton_dependency_skip_reason()
    if skip_reason is not None:
        raise RuntimeError(skip_reason)

    import jax
    import jax.numpy as jnp

    rows: list[dict[str, Any]] = []
    for nx in nx_values:
        for batch_size in batch_sizes:
            host_system = make_system_xb(
                nx,
                batch_size,
                seed=args.seed + nx * 100_003 + batch_size,
            )
            device_system = tuple(jnp.asarray(value) for value in host_system)
            reference = jax.jit(_jax_solve_xb)
            jax_times, jax_result = _time_callable(
                lambda: reference(*device_system),
                warmups=args.warmups,
                repeats=args.repeats,
            )
            rows.append(
                _timing_row(
                    solver="jax_tridiagonal_solve",
                    nx=nx,
                    batch_size=batch_size,
                    block_b=None,
                    times=jax_times,
                    max_abs_error=0.0,
                    max_rel_error=0.0,
                )
            )

            dense_reference = dense_reference_subset(
                host_system, count=min(8, batch_size)
            )
            np.testing.assert_allclose(
                np.asarray(jax_result)[:, : dense_reference.shape[1]],
                dense_reference,
                rtol=2e-5,
                atol=2e-5,
            )
            for block_b in block_sizes:
                candidate = jax.jit(
                    lambda dl, d, du, rhs, block_b=block_b: (
                        solve_single_cable_tridiagonal_xb(
                            dl, d, du, rhs, block_b=block_b
                        )
                    )
                )
                candidate_times, candidate_result = _time_callable(
                    lambda: candidate(*device_system),
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
                candidate_host = np.asarray(candidate_result)
                jax_host = np.asarray(jax_result)
                abs_error = np.abs(candidate_host - jax_host)
                rel_error = abs_error / np.maximum(
                    np.abs(jax_host), np.float32(1e-7)
                )
                np.testing.assert_allclose(
                    candidate_host, jax_host, rtol=2e-5, atol=2e-5
                )
                rows.append(
                    _timing_row(
                        solver="triton_tiled_thomas",
                        nx=nx,
                        batch_size=batch_size,
                        block_b=block_b,
                        times=candidate_times,
                        max_abs_error=float(abs_error.max(initial=0.0)),
                        max_rel_error=float(rel_error.max(initial=0.0)),
                    )
                )

    _write_rows(output / "runs.csv", rows)
    summary = _build_summary(config, rows, jax.devices())
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _print_summary(rows)
    print(f"results: {output}")
    return 0


def make_system_xb(
    nx: int,
    batch_size: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build heterogeneous, strictly diagonally dominant float32 systems."""

    rng = np.random.default_rng(seed)
    lower = -rng.uniform(0.02, 0.28, size=(nx, batch_size)).astype(np.float32)
    upper = -rng.uniform(0.02, 0.28, size=(nx, batch_size)).astype(np.float32)
    lower[0] = 0.0
    upper[-1] = 0.0
    diagonal = (
        1.0
        + np.abs(lower)
        + np.abs(upper)
        + rng.uniform(0.05, 0.5, size=(nx, batch_size)).astype(np.float32)
    )
    rhs = rng.normal(0.0, 20.0, size=(nx, batch_size)).astype(np.float32)
    return lower, diagonal, upper, rhs


def dense_reference_subset(
    system: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    count: int,
) -> np.ndarray:
    """Resolve a small subset with NumPy dense linear algebra."""

    lower, diagonal, upper, rhs = system
    nx = diagonal.shape[0]
    output = np.empty((nx, count), dtype=np.float32)
    for batch_index in range(count):
        matrix = np.diag(diagonal[:, batch_index].astype(np.float64))
        matrix += np.diag(lower[1:, batch_index].astype(np.float64), k=-1)
        matrix += np.diag(upper[:-1, batch_index].astype(np.float64), k=1)
        output[:, batch_index] = np.linalg.solve(
            matrix, rhs[:, batch_index].astype(np.float64)
        ).astype(np.float32)
    return output


def _jax_solve_xb(dl: Any, d: Any, du: Any, rhs: Any) -> Any:
    import jax

    return jax.vmap(
        lambda lower, diagonal, upper, values: jax.lax.linalg.tridiagonal_solve(
            lower, diagonal, upper, values[:, None]
        )[:, 0]
    )(dl.T, d.T, du.T, rhs.T).T


def _time_callable(
    function: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
) -> tuple[list[float], Any]:
    times: list[float] = []
    result = None
    for index in range(1 + warmups + repeats):
        start = time.perf_counter()
        result = function()
        result.block_until_ready()
        elapsed = time.perf_counter() - start
        if index == 0 or index > warmups:
            times.append(elapsed)
    assert result is not None
    return times, result


def _timing_row(
    *,
    solver: str,
    nx: int,
    batch_size: int,
    block_b: int | None,
    times: list[float],
    max_abs_error: float,
    max_rel_error: float,
) -> dict[str, Any]:
    warm = np.asarray(times[1:], dtype=np.float64)
    return {
        "solver": solver,
        "nx": nx,
        "batch_size": batch_size,
        "block_b": "" if block_b is None else block_b,
        "cold_s": times[0],
        "warm_median_s": float(np.median(warm)),
        "warm_min_s": float(warm.min()),
        "warm_max_s": float(warm.max()),
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(
    config: dict[str, Any], rows: list[dict[str, Any]], devices: Sequence[Any]
) -> dict[str, Any]:
    reference = {
        (int(row["nx"]), int(row["batch_size"])): float(row["warm_median_s"])
        for row in rows
        if row["solver"] == "jax_tridiagonal_solve"
    }
    enriched = []
    for row in rows:
        item = dict(row)
        item["warm_speedup_vs_jax"] = (
            reference[(int(row["nx"]), int(row["batch_size"]))]
            / float(row["warm_median_s"])
        )
        enriched.append(item)
    return {
        "config": config,
        "devices": [str(device) for device in devices],
        "rows": enriched,
    }


def _print_summary(rows: list[dict[str, Any]]) -> None:
    reference = {
        (int(row["nx"]), int(row["batch_size"])): float(row["warm_median_s"])
        for row in rows
        if row["solver"] == "jax_tridiagonal_solve"
    }
    print("solver,nx,batch,block,cold_ms,warm_ms,speedup,max_abs_error")
    for row in rows:
        speedup = (
            reference[(int(row["nx"]), int(row["batch_size"]))]
            / float(row["warm_median_s"])
        )
        print(
            f"{row['solver']},{row['nx']},{row['batch_size']},{row['block_b']},"
            f"{1000.0 * float(row['cold_s']):.3f},"
            f"{1000.0 * float(row['warm_median_s']):.3f},{speedup:.3f},"
            f"{float(row['max_abs_error']):.3e}"
        )


def _positive_ints(value: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 1 for item in values):
        raise ValueError(f"--{name} must contain positive comma-separated integers.")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
