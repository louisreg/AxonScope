"""Prove the block-elimination structure of one frozen AxonFleet time step."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from axonfleet.runtime.jax.kernels.block_tridiagonal import (
    solve_block_tridiagonal_2x2_scalar,
)
from axonfleet.runtime.jax.kernels.single_cable_scans import (
    _solve_single_cable_tridiagonal_jax_row,
)
from axonfleet.runtime.jax.membranes.kinetics import (
    dense_kinetic_matrix,
    solve_kinetic_transitions,
)


@dataclass(frozen=True)
class ProofResult:
    cable: str
    axons: int
    compartments: int
    state_width: int
    staged_vs_dense_max_abs: float
    kinetic_vs_dense_max_abs: float
    cable_vs_dense_max_abs: float
    direct_sum_vs_independent_max_abs: float
    cross_axon_nonzeros: int
    accepted: bool


def _block_diag(blocks: Sequence[np.ndarray]) -> np.ndarray:
    rows = sum(block.shape[0] for block in blocks)
    columns = sum(block.shape[1] for block in blocks)
    result = np.zeros((rows, columns), dtype=np.result_type(*blocks))
    row = 0
    column = 0
    for block in blocks:
        next_row = row + block.shape[0]
        next_column = column + block.shape[1]
        result[row:next_row, column:next_column] = block
        row = next_row
        column = next_column
    return result


def _random_transitions(
    rng: np.random.Generator,
    *,
    width: int,
    compartments: int,
) -> list[tuple[int, int, jnp.ndarray]]:
    transitions: list[tuple[int, int, jnp.ndarray]] = []
    for source in range(width):
        target = (source + 1) % width
        transitions.append(
            (source, target, jnp.asarray(rng.uniform(0.1, 8.0, compartments)))
        )
        transitions.append(
            (target, source, jnp.asarray(rng.uniform(0.1, 8.0, compartments)))
        )
    return transitions


def _kinetic_system(
    transitions: Sequence[tuple[int, int, jnp.ndarray]],
    *,
    width: int,
    compartments: int,
    dt: float,
) -> np.ndarray:
    generators = np.asarray(
        dense_kinetic_matrix(
            width=width,
            transitions=transitions,
            node_count=compartments,
            dtype=jnp.float64,
        )
    )
    return _block_diag(
        [np.eye(width, dtype=np.float64) - dt * generator for generator in generators]
    )


def _single_cable_system(
    rng: np.random.Generator,
    compartments: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    edge = -rng.uniform(0.05, 0.2, compartments - 1)
    diagonal = np.full(compartments, 2.0)
    diagonal[:-1] += np.abs(edge)
    diagonal[1:] += np.abs(edge)
    lower = np.concatenate(([0.0], edge))
    upper = np.concatenate((edge, [0.0]))
    matrix = np.diag(diagonal) + np.diag(edge, -1) + np.diag(edge, 1)
    return matrix, (lower, diagonal, upper)


def _double_cable_system(
    rng: np.random.Generator,
    compartments: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    off0 = -rng.uniform(0.03, 0.12, compartments - 1)
    off1 = -rng.uniform(0.03, 0.12, compartments - 1)
    coupling = rng.uniform(0.15, 0.35, compartments)
    a00 = 2.5 + coupling
    a11 = 3.0 + coupling
    a00[:-1] += np.abs(off0)
    a00[1:] += np.abs(off0)
    a11[:-1] += np.abs(off1)
    a11[1:] += np.abs(off1)
    a01 = -coupling
    a10 = -coupling

    matrix = np.zeros((2 * compartments, 2 * compartments), dtype=np.float64)
    for node in range(compartments):
        index = 2 * node
        matrix[index : index + 2, index : index + 2] = (
            (a00[node], a01[node]),
            (a10[node], a11[node]),
        )
        if node + 1 < compartments:
            next_index = index + 2
            block = np.diag((off0[node], off1[node]))
            matrix[index : index + 2, next_index : next_index + 2] = block
            matrix[next_index : next_index + 2, index : index + 2] = block
    return matrix, (a00, a01, a10, a11, off0, off1)


def _local_matrix(
    rng: np.random.Generator,
    *,
    output_width: int,
    input_width: int,
    compartments: int,
    scale: float,
) -> np.ndarray:
    matrix = np.zeros(
        (compartments * output_width, compartments * input_width),
        dtype=np.float64,
    )
    for node in range(compartments):
        row = slice(node * output_width, (node + 1) * output_width)
        column = slice(node * input_width, (node + 1) * input_width)
        matrix[row, column] = rng.normal(
            scale=scale,
            size=(output_width, input_width),
        )
    return matrix


def _one_axon_proof(
    rng: np.random.Generator,
    *,
    cable: str,
    compartments: int,
    state_width: int,
    dt: float,
) -> dict[str, Any]:
    transitions = _random_transitions(
        rng,
        width=state_width,
        compartments=compartments,
    )
    previous = rng.dirichlet(np.ones(state_width), size=compartments)
    kinetic = _kinetic_system(
        transitions,
        width=state_width,
        compartments=compartments,
        dt=dt,
    )
    predicted = np.asarray(
        solve_kinetic_transitions(
            width=state_width,
            transitions=transitions,
            previous=jnp.asarray(previous),
            dt=jnp.asarray(dt, dtype=jnp.float64),
            node_count=compartments,
            dtype=jnp.float64,
            conserve_probability=True,
        )
    ).reshape(-1)
    predicted_dense = np.linalg.solve(kinetic, previous.reshape(-1))

    cable_width = 1 if cable == "single" else 2
    if cable == "single":
        cable_matrix, operands = _single_cable_system(rng, compartments)
    elif cable == "double":
        cable_matrix, operands = _double_cable_system(rng, compartments)
    else:
        raise ValueError(f"unsupported cable: {cable}")

    coupling = _local_matrix(
        rng,
        output_width=cable_width,
        input_width=state_width,
        compartments=compartments,
        scale=0.02,
    )
    cable_rhs = rng.normal(size=compartments * cable_width)
    reduced_rhs = cable_rhs - coupling @ predicted
    if cable == "single":
        lower, diagonal, upper = operands
        cable_solution = np.asarray(
            _solve_single_cable_tridiagonal_jax_row(
                jnp.asarray(lower),
                jnp.asarray(diagonal),
                jnp.asarray(upper),
                jnp.asarray(reduced_rhs),
            )
        )
    else:
        a00, a01, a10, a11, off0, off1 = operands
        rhs_rows = reduced_rhs.reshape(compartments, 2)
        solution0, solution1 = solve_block_tridiagonal_2x2_scalar(
            *(jnp.asarray(value) for value in (a00, a01, a10, a11, off0, off1)),
            jnp.asarray(rhs_rows[:, 0]),
            jnp.asarray(rhs_rows[:, 1]),
        )
        cable_solution = np.stack((np.asarray(solution0), np.asarray(solution1)), axis=1).reshape(-1)
    cable_dense = np.linalg.solve(cable_matrix, reduced_rhs)

    finalize_from_state = _local_matrix(
        rng,
        output_width=state_width,
        input_width=state_width,
        compartments=compartments,
        scale=0.01,
    )
    finalize_from_cable = _local_matrix(
        rng,
        output_width=state_width,
        input_width=cable_width,
        compartments=compartments,
        scale=0.01,
    )
    final_rhs = rng.normal(size=compartments * state_width)
    finalized = (
        final_rhs
        - finalize_from_state @ predicted
        - finalize_from_cable @ cable_solution
    )

    state_size = compartments * state_width
    cable_size = compartments * cable_width
    staged_matrix = np.block(
        [
            [kinetic, np.zeros((state_size, cable_size)), np.zeros((state_size, state_size))],
            [coupling, cable_matrix, np.zeros((cable_size, state_size))],
            [finalize_from_state, finalize_from_cable, np.eye(state_size)],
        ]
    )
    staged_rhs = np.concatenate((previous.reshape(-1), cable_rhs, final_rhs))
    staged_solution = np.concatenate((predicted, cable_solution, finalized))
    dense_solution = np.linalg.solve(staged_matrix, staged_rhs)
    return {
        "matrix": staged_matrix,
        "rhs": staged_rhs,
        "solution": staged_solution,
        "staged_vs_dense_max_abs": float(np.max(np.abs(staged_solution - dense_solution))),
        "kinetic_vs_dense_max_abs": float(np.max(np.abs(predicted - predicted_dense))),
        "cable_vs_dense_max_abs": float(np.max(np.abs(cable_solution - cable_dense))),
    }


def prove_block_elimination(
    *,
    cable: str,
    axons: int = 3,
    compartments: int = 5,
    state_width: int = 3,
    seed: int = 1800,
    tolerance: float = 2e-11,
) -> ProofResult:
    """Compare staged production solves with one assembled direct-sum solve."""
    rng = np.random.default_rng(seed + (0 if cable == "single" else 100))
    with jax.experimental.enable_x64(True):
        proofs = [
            _one_axon_proof(
                rng,
                cable=cable,
                compartments=compartments,
                state_width=state_width,
                dt=0.005,
            )
            for _ in range(axons)
        ]
    direct_sum_matrix = _block_diag([proof["matrix"] for proof in proofs])
    direct_sum_rhs = np.concatenate([proof["rhs"] for proof in proofs])
    independent = np.concatenate([proof["solution"] for proof in proofs])
    direct_sum = np.linalg.solve(direct_sum_matrix, direct_sum_rhs)

    block_size = proofs[0]["matrix"].shape[0]
    cross_axon_nonzeros = 0
    for row_axon in range(axons):
        for column_axon in range(axons):
            if row_axon == column_axon:
                continue
            block = direct_sum_matrix[
                row_axon * block_size : (row_axon + 1) * block_size,
                column_axon * block_size : (column_axon + 1) * block_size,
            ]
            cross_axon_nonzeros += int(np.count_nonzero(block))

    errors = {
        "staged": max(proof["staged_vs_dense_max_abs"] for proof in proofs),
        "kinetic": max(proof["kinetic_vs_dense_max_abs"] for proof in proofs),
        "cable": max(proof["cable_vs_dense_max_abs"] for proof in proofs),
        "direct_sum": float(np.max(np.abs(independent - direct_sum))),
    }
    accepted = max(errors.values()) <= tolerance and cross_axon_nonzeros == 0
    return ProofResult(
        cable=cable,
        axons=axons,
        compartments=compartments,
        state_width=state_width,
        staged_vs_dense_max_abs=errors["staged"],
        kinetic_vs_dense_max_abs=errors["kinetic"],
        cable_vs_dense_max_abs=errors["cable"],
        direct_sum_vs_independent_max_abs=errors["direct_sum"],
        cross_axon_nonzeros=cross_axon_nonzeros,
        accepted=accepted,
    )


def operator_storage_estimates(
    *,
    populations: Sequence[int] = (1, 1024, 4096),
    compartments: int = 201,
    state_width: int = 6,
    transition_edges: int = 12,
) -> list[dict[str, Any]]:
    """Estimate explicit CSR storage against production-oriented arrays."""
    rows: list[dict[str, Any]] = []
    value_bytes = 4
    index_bytes = 4
    for cable, cable_width in (("single", 1), ("double", 2)):
        cable_nonzeros = (
            3 * compartments - 2
            if cable == "single"
            else 4 * compartments + 4 * (compartments - 1)
        )
        per_axon_rows = compartments * (state_width + cable_width)
        per_axon_nonzeros = (
            compartments * state_width * state_width
            + cable_nonzeros
            + compartments * cable_width * state_width
        )
        thomas_arrays = 4 * compartments if cable == "single" else 8 * compartments - 2
        matrix_free_values = compartments * (state_width + transition_edges) + thomas_arrays
        for population in populations:
            total_rows = population * per_axon_rows
            total_nonzeros = population * per_axon_nonzeros
            csr_bytes = (
                total_nonzeros * (value_bytes + index_bytes)
                + (total_rows + 1) * index_bytes
            )
            matrix_free_bytes = population * matrix_free_values * value_bytes
            rows.append(
                {
                    "cable": cable,
                    "axons": population,
                    "compartments": compartments,
                    "state_width": state_width,
                    "explicit_csr_bytes": csr_bytes,
                    "matrix_free_core_bytes": matrix_free_bytes,
                    "storage_ratio": csr_bytes / matrix_free_bytes,
                }
            )
    return rows


def _write_report(output: Path, payload: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "full_step_operator.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Frozen full-step operator proof",
        "",
        "| Cable | Staged vs dense | Kinetic vs dense | Thomas vs dense | Direct sum | Accepted |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for proof in payload["proofs"]:
        lines.append(
            f"| {proof['cable']} | {proof['staged_vs_dense_max_abs']:.3e} | "
            f"{proof['kinetic_vs_dense_max_abs']:.3e} | "
            f"{proof['cable_vs_dense_max_abs']:.3e} | "
            f"{proof['direct_sum_vs_independent_max_abs']:.3e} | "
            f"{'yes' if proof['accepted'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The assembled reference is block lower triangular. Its off-diagonal "
            "axon blocks contain zero nonzeros, so `Naxon` is a direct-sum batch axis.",
            "",
            "| Cable | Axons | Explicit CSR | Matrix-free core | Ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["storage_estimates"]:
        lines.append(
            f"| {row['cable']} | {row['axons']} | "
            f"{row['explicit_csr_bytes'] / 2**20:.2f} MiB | "
            f"{row['matrix_free_core_bytes'] / 2**20:.2f} MiB | "
            f"{row['storage_ratio']:.2f}x |"
        )
    (output / "full_step_operator.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof-axons", type=int, default=3)
    parser.add_argument("--proof-compartments", type=int, default=5)
    parser.add_argument("--state-width", type=int, default=3)
    args = parser.parse_args(argv)
    proofs = [
        prove_block_elimination(
            cable=cable,
            axons=args.proof_axons,
            compartments=args.proof_compartments,
            state_width=args.state_width,
        )
        for cable in ("single", "double")
    ]
    payload = {
        "accepted": all(proof.accepted for proof in proofs),
        "proofs": [asdict(proof) for proof in proofs],
        "storage_estimates": operator_storage_estimates(),
    }
    _write_report(args.output, payload)
    print(f"wrote: {args.output / 'full_step_operator.json'}")
    print(f"accepted: {payload['accepted']}")
    return 0 if payload["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
