"""Reproduce the ModelDB 230137 Nav1.x voltage-clamp surfaces.

This is a validation runner, not a public voltage-clamp API. It exercises the
canonical generated JAX membrane program while keeping protocol-specific data
and plotting under ``benchmark/``.

Run:
    MPLBACKEND=Agg python benchmark/curves/nav_isoform_voltage_clamp.py \
        --output benchmark/results/p18_nav_voltage_clamp_local
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs
from axonfleet.runtime.jax.membranes.compile import compile_membrane_model


DT_MS = 0.0125
NAV_MODELS = (
    axs.membranes.Nav11,
    axs.membranes.Nav12,
    axs.membranes.Nav13,
    axs.membranes.Nav14,
    axs.membranes.Nav15,
    axs.membranes.Nav16,
    axs.membranes.Nav17,
    axs.membranes.Nav18,
    axs.membranes.Nav19,
)


@dataclass(frozen=True)
class ClampProtocol:
    holding_mV: float
    step_ms: float
    start_mV: int
    stop_mV: int
    availability_start_mV: int
    availability_stop_mV: int
    test_mV: float
    conditioning_ms: float
    recovery_condition_mV: float
    recovery_condition_ms: float
    recovery_min_ms: float
    recovery_max_ms: float


PROTOCOLS = {
    "nav11": ClampProtocol(
        -120, 15, -80, 60, -140, 0, -10, 100, -10, 100, 1, 10_000
    ),
    "nav12": ClampProtocol(
        -120, 10, -80, 60, -140, -10, -10, 100, -10, 100, 1, 5_000
    ),
    "nav13": ClampProtocol(
        -90, 20, -100, 60, -100, 15, -10, 1_000, -10, 100, 1, 5_000
    ),
    "nav14": ClampProtocol(
        -120, 12, -80, 60, -140, -20, -10, 100, -10, 100, 1, 1_000
    ),
    "nav15": ClampProtocol(
        -120, 20, -90, 60, -120, 0, -10, 500, -20, 1_000, 0.1, 5_000
    ),
    "nav16": ClampProtocol(
        -90, 7.5, -80, 80, -120, 0, 0, 1_000, 0, 100, 0.1, 200
    ),
    "nav17": ClampProtocol(
        -140, 25, -80, 60, -150, -10, -20, 500, -20, 50, 0.1, 2_000
    ),
    "nav18": ClampProtocol(
        -70, 50, -80, 60, -80, 20, 0, 500, 0, 100, 1, 1_000
    ),
    "nav19": ClampProtocol(
        -120, 150, -100, 40, -140, 10, -40, 300, -40, 300, 1, 1_000
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modeldb-reference", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    results = {
        model_class.kind_name(): run_isoform(model_class, PROTOCOLS[model_class.kind_name()])
        for model_class in NAV_MODELS
    }
    summary: dict[str, Any] = {
        "source": "ModelDB 230137 voltage-clamp protocols",
        "dt_ms": DT_MS,
        "isoforms": results,
    }
    if args.modeldb_reference is not None:
        reference = json.loads(args.modeldb_reference.read_text(encoding="utf-8"))
        summary["modeldb_comparison"] = compare_modeldb_surfaces(results, reference)

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_surfaces(results, args.output)
    print(json.dumps(summary.get("modeldb_comparison", {}), indent=2))
    print(f"Nav voltage-clamp results: {args.output}")


def run_isoform(model_class, protocol: ClampProtocol) -> dict[str, Any]:
    program = compile_membrane_model(model_class())
    voltages = np.arange(protocol.start_mV, protocol.stop_mV + 1, 5, dtype=float)
    initial = program.init_gates_host(
        np.full(voltages.size, protocol.holding_mV),
        dtype_local=np.dtype(np.float32),
    )
    iv_current, iv_conductance, _ = peak_response(
        program, initial, voltages, protocol.step_ms
    )

    availability_voltages = np.arange(
        protocol.availability_start_mV,
        protocol.availability_stop_mV + 1,
        5,
        dtype=float,
    )
    availability_initial = program.init_gates_host(
        np.full(availability_voltages.size, protocol.holding_mV),
        dtype_local=np.dtype(np.float32),
    )
    conditioned = advance_constant(
        program,
        availability_initial,
        availability_voltages,
        protocol.conditioning_ms,
    )
    availability_current, _, _ = peak_response(
        program,
        conditioned,
        np.full(availability_voltages.size, protocol.test_mV),
        _test_duration_ms(model_class.kind_name()),
    )
    availability = np.abs(availability_current)
    availability /= max(float(np.max(availability)), np.finfo(float).tiny)

    recovery_ms = modeldb_recovery_intervals(
        protocol.recovery_min_ms, protocol.recovery_max_ms
    )
    resting = program.init_gates_host(
        np.asarray([protocol.holding_mV]), dtype_local=np.dtype(np.float32)
    )
    first_current, _, _ = peak_response(
        program,
        resting,
        np.asarray([protocol.recovery_condition_mV]),
        min(10.0, protocol.recovery_condition_ms),
    )
    inactivated = advance_constant(
        program,
        resting,
        np.asarray([protocol.recovery_condition_mV]),
        protocol.recovery_condition_ms,
    )
    recovered = np.concatenate(
        [
            advance_constant(
                program,
                inactivated,
                np.asarray([protocol.holding_mV]),
                interval,
            )
            for interval in recovery_ms
        ],
        axis=0,
    )
    second_current, _, _ = peak_response(
        program,
        recovered,
        np.full(recovery_ms.size, protocol.recovery_condition_mV),
        min(10.0, _recovery_test_duration_ms(model_class.kind_name())),
    )

    return {
        "protocol": asdict(protocol),
        "voltage_mV": voltages.tolist(),
        "peak_current_mA_cm2": (iv_current / 1_000.0).tolist(),
        "peak_conductance_S_cm2": (iv_conductance / 1_000.0).tolist(),
        "normalized_conductance": (iv_conductance / np.max(iv_conductance)).tolist(),
        "availability_voltage_mV": availability_voltages.tolist(),
        "availability": availability.tolist(),
        "recovery_ms": recovery_ms.tolist(),
        "recovery": (np.abs(second_current) / abs(float(first_current[0]))).tolist(),
    }


def peak_response(program, states, voltages_mV, duration_ms):
    states = np.asarray(states, dtype=np.float32)
    voltages = np.asarray(voltages_mV, dtype=np.float32)
    operators = transition_operators(program, voltages)
    peak_current = np.zeros(voltages.shape, dtype=np.float32)
    peak_conductance = np.zeros(voltages.shape, dtype=np.float32)
    gbar = float(program.parameter_values["gbar"])
    ena = float(program.parameter_values["ena"])
    for _ in range(_steps(duration_ms)):
        states = np.einsum("bi,bij->bj", states, operators, optimize=True)
        conductance = gbar * (states[:, 2] + states[:, 3])
        current = conductance * (voltages - ena)
        replace = np.abs(current) > np.abs(peak_current)
        peak_current = np.where(replace, current, peak_current)
        peak_conductance = np.maximum(peak_conductance, conductance)
    return peak_current, peak_conductance, states


def advance_constant(program, states, voltages_mV, duration_ms):
    states = np.asarray(states, dtype=np.float64)
    operators = transition_operators(program, voltages_mV).astype(np.float64)
    powers = np.stack(
        [np.linalg.matrix_power(operator, _steps(duration_ms)) for operator in operators]
    )
    return np.einsum("bi,bij->bj", states, powers, optimize=True).astype(np.float32)


def transition_operators(program, voltages_mV):
    voltages = np.asarray(voltages_mV, dtype=np.float32)
    count = voltages.size
    basis = np.tile(np.eye(6, dtype=np.float32), (count, 1))
    expanded_voltage = np.repeat(voltages, 6)
    updated = program.cn_gate_update(
        jnp.asarray(basis), jnp.asarray(expanded_voltage), DT_MS
    )
    return np.asarray(updated).reshape(count, 6, 6)


def modeldb_recovery_intervals(start_ms: float, stop_ms: float) -> np.ndarray:
    values = []
    value = float(start_ms)
    while value <= stop_ms * (1.0 + 1e-12):
        values.append(value)
        if value < 1.0:
            value += 0.1
        elif value < 10.0:
            value += 1.0
        elif value < 100.0:
            value += 10.0
        elif value < 1_000.0:
            value += 100.0
        elif value < 10_000.0:
            value += 1_000.0
        else:
            value += 10_000.0
    return np.asarray(values, dtype=float)


def compare_modeldb_surfaces(results, reference):
    comparisons = {}
    for name, result in results.items():
        expected_result = reference[name]
        expected = np.column_stack(
            (
                expected_result["voltage_mV"],
                expected_result["peak_current_mA_cm2"],
                expected_result["peak_conductance_S_cm2"],
            )
        )
        current = np.asarray(result["peak_current_mA_cm2"], dtype=float)
        conductance = np.asarray(result["peak_conductance_S_cm2"], dtype=float)
        expected_peak_current = max(float(np.max(np.abs(expected[:, 1]))), 1e-30)
        expected_peak_conductance = max(float(np.max(np.abs(expected[:, 2]))), 1e-30)
        current_error = current - expected[:, 1]
        conductance_error = conductance - expected[:, 2]
        comparisons[name] = {
            "current_nrmse_percent": 100.0
            * float(np.sqrt(np.mean(current_error**2)))
            / expected_peak_current,
            "conductance_nrmse_percent": 100.0
            * float(np.sqrt(np.mean(conductance_error**2)))
            / expected_peak_conductance,
            "current_max_abs_over_peak_percent": 100.0
            * float(np.max(np.abs(current_error)))
            / expected_peak_current,
            "conductance_max_abs_over_peak_percent": 100.0
            * float(np.max(np.abs(conductance_error)))
            / expected_peak_conductance,
            "availability_nrmse_percent": normalized_rmse_percent(
                result["availability"], expected_result["availability"]
            ),
            "recovery_nrmse_percent": normalized_rmse_percent(
                result["recovery"], expected_result["recovery"]
            ),
        }
    return comparisons


def normalized_rmse_percent(values, reference) -> float:
    values_array = np.asarray(values, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    scale = max(float(np.max(np.abs(reference_array))), 1e-30)
    return 100.0 * float(
        np.sqrt(np.mean((values_array - reference_array) ** 2))
    ) / scale


def plot_surfaces(results, output_dir: Path) -> None:
    surfaces = (
        ("peak_current_mA_cm2", "Peak current (mA/cm2)", "iv"),
        ("normalized_conductance", "Normalized conductance", "conductance_voltage"),
        ("availability", "Availability", "availability"),
        ("recovery", "Recovery", "recovery"),
    )
    for field, ylabel, file_stem in surfaces:
        figure, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
        for axis, (name, result) in zip(axes.flat, results.items(), strict=True):
            if field == "recovery":
                x = result["recovery_ms"]
                axis.set_xscale("log")
                xlabel = "Recovery interval (ms)"
            elif field == "availability":
                x = result["availability_voltage_mV"]
                xlabel = "Conditioning voltage (mV)"
            else:
                x = result["voltage_mV"]
                xlabel = "Voltage (mV)"
            axis.plot(x, result[field], color="#176B87", linewidth=1.8)
            axis.set_title(name.replace("nav1", "Nav1."))
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
        figure.savefig(output_dir / f"{file_stem}.png", dpi=160)
        plt.close(figure)


def _steps(duration_ms: float) -> int:
    return int(round(float(duration_ms) / DT_MS))


def _test_duration_ms(name: str) -> float:
    return {"nav14": 50.0, "nav18": 40.0, "nav19": 50.0}.get(name, 20.0)


def _recovery_test_duration_ms(name: str) -> float:
    return {"nav18": 10.0, "nav19": 50.0}.get(name, 20.0)


if __name__ == "__main__":
    main()
