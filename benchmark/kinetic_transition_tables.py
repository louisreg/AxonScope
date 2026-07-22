"""Benchmark-only voltage tables for generated finite-state transitions.

This candidate deliberately stays outside the runtime. It compares a table of
the canonical implicit operator with the exponential operator proposed in the
P18 design note, so interpolation error is not confused with integration error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

import axonscope as axs
from axonscope.runtime.jax.membranes.compile import compile_membrane_model


@dataclass(frozen=True)
class TableSpec:
    v_min_mV: float
    v_max_mV: float
    dv_mV: float
    dt_ms: float
    dtype: str
    operator: Literal["implicit", "exponential"]


def build_transition_table(
    membrane: Any,
    spec: TableSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate one benchmark table from the compiled kinetic contract."""
    lowering = membrane.lowering
    contract = membrane.generated_contract
    if len(contract.kinetic_blocks) != 1:
        raise ValueError("The benchmark candidate currently requires one kinetic block.")
    block = contract.kinetic_blocks[0]
    voltages = np.arange(
        spec.v_min_mV,
        spec.v_max_mV + 0.5 * spec.dv_mV,
        spec.dv_mV,
        dtype=np.float64,
    )
    rates = lowering._kinetic_rates(jnp.asarray(voltages), parameters=None)
    generators = np.asarray(
        lowering._kinetic_matrix(block, rates, len(voltages)),
        dtype=np.float64,
    )
    identity = np.eye(len(block.states), dtype=np.float64)
    if spec.operator == "implicit":
        operators = np.linalg.inv(identity[None, :, :] - spec.dt_ms * generators)
    else:
        operators = np.stack([expm(spec.dt_ms * generator) for generator in generators])
    target_dtype = np.dtype(spec.dtype)
    operators = operators.astype(target_dtype)
    stationary = np.asarray(membrane.init_gates(jnp.asarray(voltages))).astype(target_dtype)
    key_payload = {
        "runtime_contract_version": contract.version,
        "structural_hash": contract.structural_hash,
        "parameterized_hash": contract.parameterized_hash,
        "model": contract.model_name,
        "temperature": membrane.parameter_values.get("temperature", None),
        "table": asdict(spec),
        "compiler_source": contract.source_provenance,
    }
    encoded = json.dumps(key_payload, sort_keys=True, default=str).encode("utf-8")
    manifest = {
        **key_payload,
        "cache_key": hashlib.sha256(encoded).hexdigest(),
        "shape": list(operators.shape),
        "operator_bytes": int(operators.nbytes),
        "stationary_bytes": int(stationary.nbytes),
    }
    return operators, stationary, manifest


def apply_transition_table(
    previous: jnp.ndarray,
    voltage_mV: jnp.ndarray,
    table: jnp.ndarray,
    *,
    v_min_mV: float,
    dv_mV: float,
    interpolation: Literal["nearest", "linear"],
) -> jnp.ndarray:
    """Apply a nearest or linearly interpolated local transition operator."""
    coordinate = (voltage_mV - v_min_mV) / dv_mV
    if interpolation == "nearest":
        index = jnp.clip(jnp.rint(coordinate).astype(jnp.int32), 0, table.shape[0] - 1)
        operator = table[index]
    elif interpolation == "linear":
        lower = jnp.clip(jnp.floor(coordinate).astype(jnp.int32), 0, table.shape[0] - 2)
        weight = jnp.clip(coordinate - lower, 0.0, 1.0)
        operator = (1.0 - weight[:, None, None]) * table[lower]
        operator = operator + weight[:, None, None] * table[lower + 1]
    else:
        raise ValueError(f"unsupported interpolation: {interpolation}")
    return jnp.einsum("nij,nj->ni", operator, previous)


def _timed_pair(
    exact: Any,
    candidate: Any,
    *args: Any,
    repeats: int,
) -> tuple[dict[str, float], dict[str, float]]:
    first: dict[str, float] = {}
    for name, function in (("exact", exact), ("candidate", candidate)):
        start = perf_counter()
        result = function(*args)
        jax.block_until_ready(result)
        first[name] = 1e3 * (perf_counter() - start)

    samples: dict[str, list[float]] = {"exact": [], "candidate": []}
    for repeat in range(repeats):
        functions = (
            (("exact", exact), ("candidate", candidate))
            if repeat % 2 == 0
            else (("candidate", candidate), ("exact", exact))
        )
        for name, function in functions:
            start = perf_counter()
            result = function(*args)
            jax.block_until_ready(result)
            samples[name].append(1e3 * (perf_counter() - start))
    return (
        {"first_ms": first["exact"], "warm_median_ms": median(samples["exact"])},
        {
            "first_ms": first["candidate"],
            "warm_median_ms": median(samples["candidate"]),
        },
    )


def evaluate_candidate(
    *,
    spec: TableSpec,
    interpolation: Literal["nearest", "linear"],
    points: int,
    trajectory_steps: int,
    repeats: int,
    seed: int = 1810,
) -> dict[str, Any]:
    membrane = compile_membrane_model(axs.membranes.Nav16())
    table, stationary, manifest = build_transition_table(membrane, spec)
    del stationary
    rng = np.random.default_rng(seed)
    voltage = rng.uniform(spec.v_min_mV, spec.v_max_mV, points).astype(spec.dtype)
    previous = rng.dirichlet(np.ones(table.shape[-1]), size=points).astype(spec.dtype)
    voltage_jax = jnp.asarray(voltage)
    previous_jax = jnp.asarray(previous)
    table_jax = jnp.asarray(table)
    exact = jax.jit(lambda state, vm: membrane.cn_gate_update(state, vm, spec.dt_ms))
    lookup = jax.jit(
        lambda state, vm: apply_transition_table(
            state,
            vm,
            table_jax,
            v_min_mV=spec.v_min_mV,
            dv_mV=spec.dv_mV,
            interpolation=interpolation,
        )
    )
    exact_values = np.asarray(exact(previous_jax, voltage_jax))
    table_values = np.asarray(lookup(previous_jax, voltage_jax))

    phase = np.linspace(0.0, 2.0 * np.pi, trajectory_steps, endpoint=False)
    trajectory = (
        -55.0 + 45.0 * np.sin(phase) + 10.0 * np.sin(3.0 * phase)
    ).astype(spec.dtype)
    trajectory_jax = jnp.asarray(trajectory)
    initial = jnp.asarray(previous[: min(points, 4096)])
    exact_scan = jax.jit(
        lambda state: jax.lax.scan(
            lambda carry, vm: (
                exact(carry, jnp.full((carry.shape[0],), vm, dtype=carry.dtype)),
                None,
            ),
            state,
            trajectory_jax,
        )[0]
    )
    table_scan = jax.jit(
        lambda state: jax.lax.scan(
            lambda carry, vm: (
                lookup(carry, jnp.full((carry.shape[0],), vm, dtype=carry.dtype)),
                None,
            ),
            state,
            trajectory_jax,
        )[0]
    )
    exact_final = np.asarray(exact_scan(initial))
    table_final = np.asarray(table_scan(initial))
    exact_timing, table_timing = _timed_pair(
        exact,
        lookup,
        previous_jax,
        voltage_jax,
        repeats=repeats,
    )
    return {
        "spec": asdict(spec),
        "interpolation": interpolation,
        "points": points,
        "trajectory_steps": trajectory_steps,
        "manifest": manifest,
        "one_step_max_abs": float(np.max(np.abs(table_values - exact_values))),
        "trajectory_max_abs": float(np.max(np.abs(table_final - exact_final))),
        "probability_sum_max_abs": float(np.max(np.abs(table_values.sum(axis=1) - 1.0))),
        "minimum_state": float(np.min(table_values)),
        "exact_timing": exact_timing,
        "table_timing": table_timing,
        "warm_speedup": exact_timing["warm_median_ms"] / table_timing["warm_median_ms"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("quick", "local_smoke", "gpu_smoke", "gpu_realistic"),
        default="quick",
    )
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacings-mv", default="0.25,0.5,1.0")
    parser.add_argument("--operators", default="implicit,exponential")
    parser.add_argument("--interpolations", default="nearest,linear")
    parser.add_argument("--points", type=int)
    parser.add_argument("--trajectory-steps", type=int)
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args(argv)
    defaults = {
        "quick": (4096, 100, 3),
        "local_smoke": (204800, 300, 20),
        "gpu_smoke": (204800, 300, 20),
        "gpu_realistic": (823296, 1000, 30),
    }[args.preset]
    points = args.points or defaults[0]
    trajectory_steps = args.trajectory_steps or defaults[1]
    repeats = args.repeats or defaults[2]
    if args.platform == "gpu" and jax.default_backend() != "gpu":
        raise RuntimeError("GPU transition-table benchmark did not initialize a GPU backend.")
    rows = []
    for operator in args.operators.split(","):
        for spacing in (float(value) for value in args.spacings_mv.split(",")):
            spec = TableSpec(-120.0, 80.0, spacing, 0.005, "float32", operator)  # type: ignore[arg-type]
            for interpolation in args.interpolations.split(","):
                rows.append(
                    evaluate_candidate(
                        spec=spec,
                        interpolation=interpolation,  # type: ignore[arg-type]
                        points=points,
                        trajectory_steps=trajectory_steps,
                        repeats=repeats,
                    )
                )
    payload = {
        "platform": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "rows": rows,
    }
    output_file = args.output if args.output.suffix == ".json" else args.output / "summary.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
