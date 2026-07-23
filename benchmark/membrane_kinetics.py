"""Compare generated membrane-update costs at identical point counts.

Run:
    MPLBACKEND=Agg python benchmark/membrane_kinetics.py \
        --models nav16,hodgkin_huxley,axnode \
        --axons 1,128,1024 --nodes 200 --repeats 20 \
        --output benchmark/results/p18_membrane_kinetics_local/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import axonfleet as axs
from axonfleet.membranes.model import Model, currents, markov, state
from axonfleet.membranes.types import (
    ConductanceDensity,
    CurrentDensity,
    Occupancy,
    Rate,
    Voltage,
)
from axonfleet.model_ir.source import compile_model_source_file
from axonfleet.runtime.jax.membranes.compile import compile_membrane_model
from axonfleet.runtime.jax.membranes.program import JaxMembraneProgram
from axonfleet.runtime.jax.membranes.kinetics import (
    solve_conserved_kinetic_step,
    solve_kinetic_step,
)
from axonfleet.utils.units import cm2, mS, mV, ms


class SixStateKinetics(Model):
    """Model-independent Balbi-shaped workload with constant transition rates."""

    model_kind = "benchmark_six_state_kinetics"

    gbar: ConductanceDensity = 100.0 * mS / cm2
    ena: Voltage = 50.0 * mV
    r01: Rate = 1.0 / ms
    r10: Rate = 0.7 / ms
    r12: Rate = 1.1 / ms
    r21: Rate = 0.8 / ms
    r13: Rate = 0.2 / ms
    r31: Rate = 0.1 / ms
    r24: Rate = 0.6 / ms
    r42: Rate = 0.05 / ms
    r40: Rate = 0.08 / ms
    r04: Rate = 0.12 / ms
    r45: Rate = 0.02 / ms
    r54: Rate = 0.01 / ms

    C1: Occupancy = state(1.0)
    C2: Occupancy = state(0.0)
    O1: Occupancy = state(0.0)
    O2: Occupancy = state(0.0)
    I1: Occupancy = state(0.0)
    I2: Occupancy = state(0.0)

    @markov(
        "channel",
        states=("C1", "C2", "O1", "O2", "I1", "I2"),
        transitions=(
            ("C1", "C2", "C1C2", "C2C1"),
            ("C2", "O1", "C2O1", "O1C2"),
            ("C2", "O2", "C2O2", "O2C2"),
            ("O1", "I1", "O1I1", "I1O1"),
            ("I1", "C1", "I1C1", "C1I1"),
            ("I1", "I2", "I1I2", "I2I1"),
        ),
    )
    def channel(self, Vm: Voltage):
        C1C2: Rate = self.r01
        C2C1: Rate = self.r10
        C2O1: Rate = self.r12
        O1C2: Rate = self.r21
        C2O2: Rate = self.r13
        O2C2: Rate = self.r31
        O1I1: Rate = self.r24
        I1O1: Rate = self.r42
        I1C1: Rate = self.r40
        C1I1: Rate = self.r04
        I1I2: Rate = self.r45
        I2I1: Rate = self.r54
        self.keep(C1C2, C2C1, C2O1, O1C2, C2O2, O2C2)
        self.keep(O1I1, I1O1, I1C1, C1I1, I1I2, I2I1)

    @currents(outputs=("I_na",), observables=("g_na",))
    def currents(
        self,
        Vm: Voltage,
        C1: Occupancy,
        C2: Occupancy,
        O1: Occupancy,
        O2: Occupancy,
        I1: Occupancy,
        I2: Occupancy,
    ):
        g_na: ConductanceDensity = self.gbar * (O1 + O2)
        I_na: CurrentDensity = g_na * (Vm - self.ena)
        return I_na, g_na


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="six_state_constant,nav16,hodgkin_huxley,axnode",
    )
    parser.add_argument("--axons", default="1,128,1024")
    parser.add_argument("--nodes", type=int, default=200)
    parser.add_argument("--dt-ms", type=float, default=0.005)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    stage_rows = []
    axon_counts = [int(value) for value in args.axons.split(",") if value]
    for model_name in args.models.split(","):
        membrane = membrane_program(model_name.strip())
        for axon_count in axon_counts:
            rows.append(
                measure(
                    membrane,
                    model_name=model_name.strip(),
                    axon_count=axon_count,
                    node_count=int(args.nodes),
                    dt_ms=float(args.dt_ms),
                    repeats=int(args.repeats),
                )
            )
        if args.profile_stages and membrane.generated_contract.kinetic_blocks:
            stage_rows.append(
                measure_kinetic_stages(
                    membrane,
                    model_name=model_name.strip(),
                    point_count=max(axon_counts) * int(args.nodes),
                    dt_ms=float(args.dt_ms),
                    repeats=int(args.repeats),
                )
            )
    result = {
        "platform": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "nodes": int(args.nodes),
        "dt_ms": float(args.dt_ms),
        "repeats": int(args.repeats),
        "models": [value for value in args.models.split(",") if value],
        "rows": rows,
        "kinetic_stages": stage_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def measure(
    membrane: JaxMembraneProgram,
    *,
    model_name: str,
    axon_count: int,
    node_count: int,
    dt_ms: float,
    repeats: int,
) -> dict[str, float | int]:
    voltage = jnp.full((axon_count, node_count), -70.0, dtype=jnp.float32)
    initial_row = membrane.init_gates(voltage[0])
    gates = jnp.broadcast_to(initial_row, (axon_count, *initial_row.shape))
    update = jax.jit(
        jax.vmap(
            lambda state, vm: membrane.cn_gate_update(state, vm, dt_ms)
        )
    )
    start = perf_counter()
    result = update(gates, voltage)
    result.block_until_ready()
    first_ms = 1e3 * (perf_counter() - start)
    warm_ms: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        result = update(gates, voltage)
        result.block_until_ready()
        warm_ms.append(1e3 * (perf_counter() - start))
    state_count = int(gates.shape[-1])
    return {
        "model": model_name,
        "axons": axon_count,
        "kinetic_nodes": axon_count * node_count,
        "evolving_states_per_node": state_count,
        "evolving_state_bytes": int(gates.size * gates.dtype.itemsize),
        "first_update_ms": first_ms,
        "warm_update_median_ms": median(warm_ms),
        "warm_ns_per_kinetic_node": 1e6 * median(warm_ms) / (axon_count * node_count),
    }


def membrane_program(name: str) -> JaxMembraneProgram:
    factories = {
        "nav11": axs.membranes.Nav11,
        "nav16": axs.membranes.Nav16,
        "hodgkin_huxley": axs.membranes.HodgkinHuxley,
        "axnode": axs.membranes.AxNode,
    }
    if name == "six_state_constant":
        compiled = compile_model_source_file(
            Path(__file__),
            model_class_name="SixStateKinetics",
            load_generated_modules=("jax", "numpy"),
        )
        return JaxMembraneProgram.from_generated_module(
            compiled.cache.loaded_modules["jax"],
            parameter_overrides={},
            host_module=compiled.cache.loaded_modules["numpy"],
        )
    try:
        factory = factories[name]
    except KeyError as exc:
        choices = ", ".join(("six_state_constant", *factories))
        raise ValueError(f"unknown model {name!r}; expected one of: {choices}") from exc
    return compile_membrane_model(factory())


def measure_kinetic_stages(
    membrane: JaxMembraneProgram,
    *,
    model_name: str,
    point_count: int,
    dt_ms: float,
    repeats: int,
) -> dict[str, object]:
    lowering = membrane.lowering
    block = membrane.generated_contract.kinetic_blocks[0]
    voltage = jnp.full((point_count,), -70.0, dtype=jnp.float32)
    previous = membrane.init_gates(voltage)

    rates_fn = jax.jit(
        lambda vm: tuple(
            lowering._kinetic_rates(vm, parameters=None).values()
        )
    )
    matrix_fn = jax.jit(
        lambda vm: lowering._kinetic_matrix(
            block,
            lowering._kinetic_rates(vm, parameters=None),
            vm.shape[0],
        )
    )
    matrix = matrix_fn(voltage)
    matrix.block_until_ready()
    solve_fn = jax.jit(
        lambda value, state: jnp.linalg.solve(
            jnp.eye(len(block.states), dtype=jnp.float32) - dt_ms * value,
            state[..., None],
        )[..., 0]
    )
    unrolled_solve_fn = jax.jit(
        lambda value, state: solve_kinetic_step(value, state, dt_ms)
    )
    conserved_solve_fn = jax.jit(
        lambda value, state: solve_conserved_kinetic_step(value, state, dt_ms)
    )
    dense_full_fn = jax.jit(
        lambda state, vm: solve_conserved_kinetic_step(
            lowering._kinetic_matrix(
                block,
                lowering._kinetic_rates(vm, parameters=None),
                vm.shape[0],
            ),
            state,
            dt_ms,
        )
    )
    full_fn = jax.jit(
        lambda state, vm: membrane.cn_gate_update(state, vm, dt_ms)
    )
    stages = {
        "rates_only": measure_callable(rates_fn, voltage, repeats=repeats),
        "rates_and_matrix": measure_callable(
            matrix_fn, voltage, repeats=repeats
        ),
        "generic_solve_only": measure_callable(
            solve_fn, matrix, previous, repeats=repeats
        ),
        "unrolled_solve_only": measure_callable(
            unrolled_solve_fn, matrix, previous, repeats=repeats
        ),
        "conserved_reduced_solve_only": measure_callable(
            conserved_solve_fn, matrix, previous, repeats=repeats
        ),
        "dense_conserved_full_update": measure_callable(
            dense_full_fn, previous, voltage, repeats=repeats
        ),
        "full_update": measure_callable(
            full_fn, previous, voltage, repeats=repeats
        ),
    }
    return {
        "model": model_name,
        "points": point_count,
        "matrix_intermediate_bytes": point_count
        * len(block.states)
        * len(block.states)
        * np.dtype(np.float32).itemsize,
        "stages": stages,
    }


def measure_callable(function, *args, repeats: int) -> dict[str, float]:
    start = perf_counter()
    result = function(*args)
    jax.block_until_ready(result)
    first_ms = 1e3 * (perf_counter() - start)
    warm_ms = []
    for _ in range(repeats):
        start = perf_counter()
        result = function(*args)
        jax.block_until_ready(result)
        warm_ms.append(1e3 * (perf_counter() - start))
    return {
        "first_ms": first_ms,
        "warm_median_ms": median(warm_ms),
    }


if __name__ == "__main__":
    main()
