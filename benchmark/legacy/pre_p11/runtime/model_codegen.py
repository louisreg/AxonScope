from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

import jax
import jax.numpy as jnp
import numpy as np

import axonscope as axs
from axonscope.backends.jax.runtime import compile_membrane_model
from axonscope.membranes.compiler import lower_membrane_model_with_sources
from axonscope.membranes.model import MembraneModel
from axonscope.model_ir.interpreter import NumpyModelInterpreter
from axonscope.membranes.types import (
    ConductanceDensity,
    CurrentDensity,
    Gate,
    Rate,
    ResistanceArea,
    Voltage,
)


@dataclass(frozen=True)
class ModelCase:
    """One membrane model benchmark case."""

    name: str
    factory: Callable[[], axs.membranes.Model]
    group: str


@dataclass(frozen=True)
class SimulationCase:
    """One tiny public AxonSimulation benchmark case."""

    name: str
    model: str
    factory: Callable[[], Any]
    duration_ms: float = 0.05
    dt_ms: float = 0.05
    current_nA: float = 0.1
    threshold_mV: float = -80.0


@dataclass(frozen=True)
class CodegenRow:
    """One measured codegen/cache row."""

    model: str
    group: str
    phase: str
    repeat: int
    seconds: float
    cache_status: str
    cache_reason: str
    cache_key: str
    source_hash: str
    source_path: str
    generated_bytes: int
    generated_files: tuple[str, ...]


@dataclass(frozen=True)
class ModelStepRow:
    """One measured generated/interpreted model-step row."""

    model: str
    group: str
    target: str
    repeat: int
    seconds: float
    nodes: int
    outputs: tuple[str, ...]
    status: str
    note: str


@dataclass(frozen=True)
class SimulationRow:
    """One measured public AxonSimulation row."""

    case: str
    model: str
    phase: str
    repeat: int
    seconds: float
    status: str
    vm_shape: tuple[int, ...]
    vm_peak_mV: float | None
    activated: bool | None
    first_time_ms: float | None
    note: str


@dataclass(frozen=True)
class CorrectnessRow:
    """One pre-timing correctness check."""

    scope: str
    case: str
    target: str
    status: str
    max_abs: float | None
    max_rel: float | None
    note: str


class BenchmarkLeak(axs.membranes.Model):
    """Small custom membrane model used by the codegen benchmark."""

    model_kind = "benchmark_leak"

    Rm: ResistanceArea = 10_000.0 * axs.ohm_cm2
    EL: Voltage = -70.0 * axs.mV

    @axs.membranes.currents
    def currents(self, Vm: Voltage):
        drive: Voltage = Vm - self.EL
        g_l: ConductanceDensity = 1.0 / self.Rm
        I_l: CurrentDensity = g_l * drive
        return I_l, g_l, drive


class BenchmarkSodiumLeak(axs.membranes.Model):
    """Small custom gated model used by the codegen benchmark."""

    model_kind = "benchmark_sodium_leak"

    gna: ConductanceDensity = 20.0 * axs.mS_per_cm2
    gl: ConductanceDensity = 0.1 * axs.mS_per_cm2
    ena: Voltage = 45.0 * axs.mV
    el: Voltage = -70.0 * axs.mV

    @axs.membranes.rates
    def rates(self, Vm: Voltage):
        alpha_m: Rate = 0.1 / (axs.ms * axs.mV) * (Vm + 35.0 * axs.mV)
        beta_m: Rate = 4.0 / axs.ms
        self.keep(alpha_m, beta_m)

    @axs.membranes.currents(outputs=("I_na", "I_l"), observables=("g_na", "g_l"))
    def currents(self, Vm: Voltage, m: Gate):
        g_na: ConductanceDensity = self.gna * m
        g_l: ConductanceDensity = self.gl
        I_na: CurrentDensity = g_na * (Vm - self.ena)
        I_l: CurrentDensity = g_l * (Vm - self.el)
        return I_na, I_l, g_na, g_l


BUILTIN_CASES: tuple[ModelCase, ...] = (
    ModelCase("passive", axs.membranes.Passive, "builtin"),
    ModelCase("hodgkin_huxley", axs.membranes.HodgkinHuxley, "builtin"),
    ModelCase("rattay_aberham", axs.membranes.RattayAberham, "builtin"),
    ModelCase("sundt", axs.membranes.Sundt, "builtin"),
    ModelCase("axnode", axs.membranes.AxNode, "builtin"),
    ModelCase("tigerholm", axs.membranes.Tigerholm, "builtin"),
    ModelCase("schild94", axs.membranes.Schild94, "builtin"),
    ModelCase("schild97", axs.membranes.Schild97, "builtin"),
)

CUSTOM_CASES: tuple[ModelCase, ...] = (
    ModelCase("benchmark_leak", BenchmarkLeak, "custom"),
    ModelCase("benchmark_sodium_leak", BenchmarkSodiumLeak, "custom"),
)

MODEL_CASES: dict[str, ModelCase] = {
    case.name: case for case in (*BUILTIN_CASES, *CUSTOM_CASES)
}

SIMULATION_CASES: tuple[SimulationCase, ...] = (
    SimulationCase(
        name="hh_template",
        model="hodgkin_huxley",
        factory=lambda: axs.axons.HodgkinHuxley(
            length=80.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=5,
            celsius=6.3 * axs.degC,
        ),
    ),
    SimulationCase(
        name="rattay_aberham_template",
        model="rattay_aberham",
        factory=lambda: axs.axons.RattayAberham(
            length=80.0 * axs.um,
            diameter=0.8 * axs.um,
            compartments=5,
            celsius=37.0 * axs.degC,
        ),
    ),
    SimulationCase(
        name="sundt_template",
        model="sundt",
        factory=lambda: axs.axons.Sundt(
            length=80.0 * axs.um,
            diameter=0.8 * axs.um,
            compartments=5,
            celsius=37.0 * axs.degC,
        ),
    ),
    SimulationCase(
        name="tigerholm_template",
        model="tigerholm",
        factory=lambda: axs.axons.Tigerholm(
            length=80.0 * axs.um,
            diameter=1.0 * axs.um,
            compartments=5,
            celsius=37.0 * axs.degC,
        ),
    ),
    SimulationCase(
        name="schild94_template",
        model="schild94",
        factory=lambda: axs.axons.Schild94(
            length=80.0 * axs.um,
            diameter=0.8 * axs.um,
            compartments=5,
        ),
    ),
    SimulationCase(
        name="schild97_template",
        model="schild97",
        factory=lambda: axs.axons.Schild97(
            length=80.0 * axs.um,
            diameter=0.8 * axs.um,
            compartments=5,
        ),
    ),
    SimulationCase(
        name="mrg_axnode_template",
        model="mrg_axnode",
        factory=lambda: axs.axons.MRG(
            diameter=5.7 * axs.um,
            nodes=3,
        ),
    ),
)

SIMULATION_CASE_MAP: dict[str, SimulationCase] = {
    case.name: case for case in SIMULATION_CASES
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark class-based membrane source/codegen cache behavior."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["builtins"],
        choices=("all", "builtins", "custom", *MODEL_CASES),
        help="Model cases to benchmark.",
    )
    parser.add_argument(
        "--warm-repeats",
        type=int,
        default=3,
        help="Warm cache-hit inspection repetitions per model.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional generated-code cache root. Defaults under --out-dir.",
    )
    parser.add_argument(
        "--model-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure deterministic generated/interpreted model-step paths.",
    )
    parser.add_argument(
        "--step-repeats",
        type=int,
        default=3,
        help="Measured repeats per model-step target.",
    )
    parser.add_argument(
        "--step-nodes",
        type=int,
        default=17,
        help="Number of voltage samples in model-step microbenchmarks.",
    )
    parser.add_argument(
        "--simulations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure tiny public AxonSimulation first/warm runs.",
    )
    parser.add_argument(
        "--simulation-cases",
        nargs="+",
        default=["none"],
        choices=("none", "smoke", "representative", "all", *SIMULATION_CASE_MAP),
        help="Simulation cases to benchmark.",
    )
    parser.add_argument(
        "--simulation-warm-repeats",
        type=int,
        default=1,
        help="Warm public AxonSimulation repeats per case.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark/results/runtime"),
        help="Directory for JSON and CSV benchmark outputs.",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix.")
    parser.add_argument("--list", action="store_true", help="List model cases and exit.")
    args = parser.parse_args(argv)

    if args.list:
        print_cases()
        return

    if args.warm_repeats < 0:
        raise ValueError("--warm-repeats must be >= 0.")
    if args.step_repeats < 1:
        raise ValueError("--step-repeats must be >= 1.")
    if args.step_nodes < 1:
        raise ValueError("--step-nodes must be >= 1.")
    if args.simulation_warm_repeats < 0:
        raise ValueError("--simulation-warm-repeats must be >= 0.")

    selected = select_cases(args.models)
    selected_simulations = (
        ()
        if not args.simulations
        else select_simulation_cases(args.simulation_cases)
    )
    prefix = args.prefix or datetime.now().strftime("model_codegen_%Y%m%d_%H%M%S")
    cache_root = (args.cache_root or args.out_dir / f"{prefix}_codegen_cache").resolve()
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    with model_codegen_cache(cache_root):
        codegen_rows = run_codegen_benchmark(
            selected,
            warm_repeats=int(args.warm_repeats),
        )
        if args.model_steps:
            model_step_rows, step_correctness = run_model_step_benchmark(
                selected,
                repeats=int(args.step_repeats),
                node_count=int(args.step_nodes),
            )
        else:
            model_step_rows, step_correctness = (), ()
        if selected_simulations:
            simulation_rows, simulation_correctness = run_simulation_benchmark(
                selected_simulations,
                warm_repeats=int(args.simulation_warm_repeats),
            )
        else:
            simulation_rows, simulation_correctness = (), ()

    correctness_rows = (*step_correctness, *simulation_correctness)
    paths = write_outputs(
        codegen_rows=codegen_rows,
        model_step_rows=model_step_rows,
        simulation_rows=simulation_rows,
        correctness_rows=correctness_rows,
        out_dir=args.out_dir,
        prefix=prefix,
        metadata=run_metadata(
            selected,
            simulation_cases=selected_simulations,
            cache_root=cache_root,
            warm_repeats=args.warm_repeats,
            step_repeats=args.step_repeats,
            step_nodes=args.step_nodes,
            simulation_warm_repeats=args.simulation_warm_repeats,
            model_steps=bool(args.model_steps),
            simulations=bool(args.simulations),
        ),
    )

    print("=== Membrane model codegen benchmark ===")
    for row in codegen_rows:
        print(
            f"{row.model:22s} {row.phase:5s} r{row.repeat:<2d} "
            f"{row.seconds:.4f}s cache={row.cache_status}/{row.cache_reason} "
            f"bytes={row.generated_bytes}"
        )
    if model_step_rows:
        print("=== Model-step microbenchmarks ===")
        for row in model_step_rows:
            print(
                f"{row.model:22s} {row.target:20s} r{row.repeat:<2d} "
                f"{row.seconds:.6f}s status={row.status}"
            )
    if simulation_rows:
        print("=== Public AxonSimulation benchmark ===")
        for row in simulation_rows:
            print(
                f"{row.case:26s} {row.phase:5s} r{row.repeat:<2d} "
                f"{row.seconds:.4f}s status={row.status} activated={row.activated}"
            )
    failed_correctness = [row for row in correctness_rows if row.status == "failed"]
    skipped_correctness = [row for row in correctness_rows if row.status == "skipped"]
    print(
        "correctness: "
        f"{len(correctness_rows) - len(failed_correctness) - len(skipped_correctness)}/"
        f"{len(correctness_rows)} ok, {len(skipped_correctness)} skipped"
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(f"cache: {cache_root}")


def print_cases() -> None:
    print("Model codegen cases:")
    for name, case in MODEL_CASES.items():
        print(f"  {name:22s} {case.group}")
    print("Simulation cases:")
    for case in SIMULATION_CASES:
        print(f"  {case.name:26s} {case.model}")


def select_cases(names: Sequence[str]) -> tuple[ModelCase, ...]:
    requested = tuple(names)
    if "all" in requested:
        return tuple(MODEL_CASES.values())
    cases: list[ModelCase] = []
    if "builtins" in requested:
        cases.extend(BUILTIN_CASES)
    if "custom" in requested:
        cases.extend(CUSTOM_CASES)
    for name in requested:
        if name in {"builtins", "custom"}:
            continue
        cases.append(MODEL_CASES[name])
    deduped: dict[str, ModelCase] = {}
    for case in cases:
        deduped.setdefault(case.name, case)
    return tuple(deduped.values())


def select_simulation_cases(names: Sequence[str]) -> tuple[SimulationCase, ...]:
    requested = tuple(names)
    if "none" in requested:
        return ()
    if "smoke" in requested:
        return tuple(
            SIMULATION_CASE_MAP[name]
            for name in ("hh_template", "mrg_axnode_template")
        )
    if "all" in requested or "representative" in requested:
        return SIMULATION_CASES
    deduped: dict[str, SimulationCase] = {}
    for name in requested:
        deduped.setdefault(name, SIMULATION_CASE_MAP[name])
    return tuple(deduped.values())


@contextmanager
def model_codegen_cache(cache_root: Path) -> Iterator[None]:
    previous_cache = os.environ.get("AXONSCOPE_MODEL_CODEGEN_CACHE")
    os.environ["AXONSCOPE_MODEL_CODEGEN_CACHE"] = str(cache_root)
    try:
        yield
    finally:
        if previous_cache is None:
            os.environ.pop("AXONSCOPE_MODEL_CODEGEN_CACHE", None)
        else:
            os.environ["AXONSCOPE_MODEL_CODEGEN_CACHE"] = previous_cache


def run_codegen_benchmark(
    cases: Sequence[ModelCase],
    *,
    warm_repeats: int,
) -> tuple[CodegenRow, ...]:
    rows: list[CodegenRow] = []
    for case in cases:
        rows.append(measure_inspection(case, phase="cold", repeat=0))
        for repeat in range(warm_repeats):
            rows.append(measure_inspection(case, phase="warm", repeat=repeat))
    return tuple(rows)


def measure_inspection(case: ModelCase, *, phase: str, repeat: int) -> CodegenRow:
    start = time.perf_counter()
    report = axs.membranes.inspect_generated_code(unparameterized_descriptor(case))
    seconds = time.perf_counter() - start
    source = report.sources[0]
    generated_files = tuple(generated.name for generated in source.files)
    generated_bytes = sum(int(generated.size_bytes) for generated in source.files)
    return CodegenRow(
        model=case.name,
        group=case.group,
        phase=phase,
        repeat=int(repeat),
        seconds=float(seconds),
        cache_status=source.cache_status,
        cache_reason=source.cache_reason,
        cache_key=source.cache_key,
        source_hash=source.source_hash,
        source_path=source.source_path,
        generated_bytes=int(generated_bytes),
        generated_files=generated_files,
    )


def unparameterized_descriptor(case: ModelCase) -> MembraneModel:
    model = case.factory()
    return MembraneModel(
        model.kind,
        source_path=model.__class__.source_path(),
        source_class=model.__class__.source_class(),
        dtype=model.dtype,
    )


def run_model_step_benchmark(
    cases: Sequence[ModelCase],
    *,
    repeats: int,
    node_count: int,
) -> tuple[tuple[ModelStepRow, ...], tuple[CorrectnessRow, ...]]:
    rows: list[ModelStepRow] = []
    correctness: list[CorrectnessRow] = []
    for case in cases:
        try:
            context = build_model_step_context(case, node_count=node_count)
        except Exception as exc:  # pragma: no cover - exercised by smoke commands
            note = f"{type(exc).__name__}: {exc}"
            correctness.append(CorrectnessRow("model_step", case.name, "setup", "failed", None, None, note))
            rows.append(
                ModelStepRow(
                    model=case.name,
                    group=case.group,
                    target="setup",
                    repeat=0,
                    seconds=0.0,
                    nodes=int(node_count),
                    outputs=(),
                    status="failed",
                    note=note,
                )
            )
            continue

        correctness.extend(check_model_step_correctness(context))
        for target, runner in model_step_runners(context).items():
            for repeat in range(repeats):
                seconds, status, note = measure_callable(runner)
                rows.append(
                    ModelStepRow(
                        model=case.name,
                        group=case.group,
                        target=target,
                        repeat=int(repeat),
                        seconds=float(seconds),
                        nodes=int(node_count),
                        outputs=tuple(context["output_names"]),
                        status=status,
                        note=note,
                    )
                )
    return tuple(rows), tuple(correctness)


def build_model_step_context(case: ModelCase, *, node_count: int) -> dict[str, Any]:
    public_model = case.factory()
    descriptor = public_model.to_membrane_model()
    lowered = lower_membrane_model_with_sources(
        descriptor,
        load_generated_modules=("numpy", "jax"),
    )
    source = lowered.source_results[0]
    model = lowered.model
    interpreter = NumpyModelInterpreter(model, dtype=descriptor.dtype)
    V = np.linspace(-85.0, 25.0, int(node_count), dtype=descriptor.dtype)
    gates = interpreter.init_gates(V)
    state = interpreter.init_membrane_state(V)
    numpy_module = source.cache.loaded_modules["numpy"]
    jax_module = source.cache.loaded_modules["jax"]
    membrane = compile_membrane_model(descriptor)
    output_names = tuple(str(name) for name in getattr(numpy_module, "OUTPUT_NAMES", ()))
    return {
        "case": case,
        "model": model,
        "interpreter": interpreter,
        "V_np": V,
        "gates_np": gates,
        "state_np": state,
        "numpy_module": numpy_module,
        "jax_module": jax_module,
        "membrane": membrane,
        "output_names": output_names,
    }


def model_step_runners(context: dict[str, Any]) -> dict[str, Callable[[], Any]]:
    interpreter: NumpyModelInterpreter = context["interpreter"]
    V_np = context["V_np"]
    gates_np = context["gates_np"]
    state_np = context["state_np"]
    numpy_module = context["numpy_module"]
    jax_module = context["jax_module"]
    membrane = context["membrane"]
    args_np = generated_model_step_args(numpy_module, context, target="numpy")
    args_jax = generated_model_step_args(jax_module, context, target="jax")
    V_jax = jnp.asarray(V_np, dtype=membrane.dtype)
    gates_jax = jnp.asarray(gates_np, dtype=membrane.dtype)
    state_jax = tuple(jnp.asarray(value, dtype=membrane.dtype) for value in state_np)

    def numpy_interpreter() -> Any:
        return (
            interpreter.current_matrix(V_np, gates_np, state=state_np),
            interpreter.conductances(gates_np, state=state_np),
            interpreter.gate_update(gates_np, V_np, 0.01),
        )

    def generated_numpy() -> Any:
        return numpy_module.model_step(*args_np)

    def generated_jax() -> Any:
        return sync_jax(jax_module.model_step(*args_jax))

    def jax_runtime_lowering() -> Any:
        return sync_jax(
            (
                membrane.currents(V_jax, gates_jax, state_jax),
                membrane.conductances(gates_jax, state_jax),
                membrane.cn_gate_update(gates_jax, V_jax, 0.01),
            )
        )

    return {
        "numpy_interpreter": numpy_interpreter,
        "generated_numpy": generated_numpy,
        "generated_jax": generated_jax,
        "jax_runtime_lowering": jax_runtime_lowering,
    }


def generated_model_step_args(module: Any, context: dict[str, Any], *, target: str) -> tuple[Any, ...]:
    interpreter: NumpyModelInterpreter = context["interpreter"]
    V = context["V_np"]
    gates = context["gates_np"]
    state = context["state_np"]
    env: dict[str, Any] = {"Vm": V}
    for index, name in enumerate(interpreter.program.gate_state_names):
        env[name] = gates[:, index]
    for index, name in enumerate(interpreter.program.membrane_state_names):
        env[name] = state[index]
    env.update(interpreter.with_parameters())
    env["Vm_prev"] = V
    env["Vm_new"] = V + np.asarray(0.1, dtype=interpreter.dtype)
    env["I_ion"] = interpreter.currents(V, gates, state=state)
    env["I_background"] = np.zeros_like(V)

    args = tuple(env[name] for name in module.ARG_NAMES)
    if target == "jax":
        return tuple(jnp.asarray(value) for value in args)
    return args


def check_model_step_correctness(context: dict[str, Any]) -> tuple[CorrectnessRow, ...]:
    rows: list[CorrectnessRow] = []
    try:
        numpy_module = context["numpy_module"]
        jax_module = context["jax_module"]
        numpy_actual = numpy_module.model_step(
            *generated_model_step_args(numpy_module, context, target="numpy")
        )
        jax_actual = sync_jax(
            jax_module.model_step(*generated_model_step_args(jax_module, context, target="jax"))
        )
        max_abs, max_rel = compare_output_tuple(jax_actual, tuple(np.asarray(v) for v in as_tuple(numpy_actual)))
        rows.append(
            CorrectnessRow(
                scope="model_step",
                case=context["case"].name,
                target="generated_jax_vs_numpy",
                status="ok" if max_abs <= 5e-4 or max_rel <= 5e-4 else "failed",
                max_abs=float(max_abs),
                max_rel=float(max_rel),
                note="",
            )
        )
        actual_subset, expected_subset, mapped_names = mappable_interpreter_outputs(
            context,
            numpy_actual,
        )
        if expected_subset:
            max_abs, max_rel = compare_output_tuple(actual_subset, expected_subset)
            rows.append(
                CorrectnessRow(
                    scope="model_step",
                    case=context["case"].name,
                    target="generated_numpy_vs_interpreter",
                    status="ok" if max_abs <= 5e-4 or max_rel <= 5e-4 else "failed",
                    max_abs=float(max_abs),
                    max_rel=float(max_rel),
                    note="mapped_outputs=" + ",".join(mapped_names),
                )
            )
        else:
            rows.append(
                CorrectnessRow(
                    scope="model_step",
                    case=context["case"].name,
                    target="generated_numpy_vs_interpreter",
                    status="skipped",
                    max_abs=None,
                    max_rel=None,
                    note="no generated outputs map cleanly to interpreter groups",
                )
            )
    except Exception as exc:  # pragma: no cover - smoke command guardrail
        rows.append(
            CorrectnessRow(
                scope="model_step",
                case=context["case"].name,
                target="setup",
                status="failed",
                max_abs=None,
                max_rel=None,
                note=f"{type(exc).__name__}: {exc}",
            )
        )
    return tuple(rows)


def mappable_interpreter_outputs(
    context: dict[str, Any],
    actual: Any,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[str, ...]]:
    interpreter: NumpyModelInterpreter = context["interpreter"]
    V = context["V_np"]
    gates = context["gates_np"]
    state = context["state_np"]
    currents = interpreter.current_matrix(V, gates, state=state)
    conductances = interpreter.conductances(gates, state=state)
    observables = interpreter.observables(gates, state=state)
    current_names = tuple(interpreter.program.raw_current_names)
    conductance_names = tuple(interpreter.program.raw_conductance_names)
    actual_values = as_tuple(actual)
    actual_subset: list[np.ndarray] = []
    expected_subset: list[np.ndarray] = []
    mapped_names: list[str] = []
    for index, name in enumerate(context["output_names"]):
        expected: np.ndarray | None = None
        if name in observables:
            expected = observables[name]
        elif current_names.count(name) == 1:
            expected = currents[:, current_names.index(name)]
        elif conductance_names.count(name) == 1:
            expected = conductances[:, conductance_names.index(name)]
        if expected is None:
            continue
        actual_subset.append(np.asarray(actual_values[index]))
        expected_subset.append(expected)
        mapped_names.append(name)
    return tuple(actual_subset), tuple(expected_subset), tuple(mapped_names)


def as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def compare_output_tuple(actual: Any, expected: tuple[np.ndarray, ...]) -> tuple[float, float]:
    actual_tuple = actual if isinstance(actual, tuple) else (actual,)
    if len(actual_tuple) != len(expected):
        raise ValueError(f"expected {len(expected)} outputs, got {len(actual_tuple)}.")
    max_abs = 0.0
    max_rel = 0.0
    for actual_value, expected_value in zip(actual_tuple, expected, strict=True):
        actual_arr = np.asarray(actual_value)
        expected_arr = np.asarray(expected_value)
        diff = np.abs(actual_arr - expected_arr)
        denom = np.maximum(np.abs(expected_arr), 1e-12)
        max_abs = max(max_abs, float(np.max(diff)) if diff.size else 0.0)
        max_rel = max(max_rel, float(np.max(diff / denom)) if diff.size else 0.0)
    return max_abs, max_rel


def sync_jax(value: Any) -> Any:
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    return value


def measure_callable(callback: Callable[[], Any]) -> tuple[float, str, str]:
    start = time.perf_counter()
    try:
        callback()
    except Exception as exc:  # pragma: no cover - smoke command guardrail
        return 0.0, "failed", f"{type(exc).__name__}: {exc}"
    return time.perf_counter() - start, "ok", ""


def run_simulation_benchmark(
    cases: Sequence[SimulationCase],
    *,
    warm_repeats: int,
) -> tuple[tuple[SimulationRow, ...], tuple[CorrectnessRow, ...]]:
    rows: list[SimulationRow] = []
    correctness: list[CorrectnessRow] = []
    for case in cases:
        try:
            simulation, activation = build_simulation(case)
            row, result = measure_simulation_run(
                case,
                simulation,
                activation,
                phase="first",
                repeat=0,
            )
            rows.append(row)
            correctness.append(check_simulation_result(case, result, activation))
            for repeat in range(warm_repeats):
                row, _ = measure_simulation_run(
                    case,
                    simulation,
                    activation,
                    phase="warm",
                    repeat=repeat,
                )
                rows.append(row)
        except Exception as exc:  # pragma: no cover - smoke command guardrail
            note = f"{type(exc).__name__}: {exc}"
            rows.append(
                SimulationRow(
                    case=case.name,
                    model=case.model,
                    phase="setup",
                    repeat=0,
                    seconds=0.0,
                    status="failed",
                    vm_shape=(),
                    vm_peak_mV=None,
                    activated=None,
                    first_time_ms=None,
                    note=note,
                )
            )
            correctness.append(
                CorrectnessRow(
                    scope="simulation",
                    case=case.name,
                    target="setup",
                    status="failed",
                    max_abs=None,
                    max_rel=None,
                    note=note,
                )
            )
    return tuple(rows), tuple(correctness)


def build_simulation(case: SimulationCase) -> tuple[axs.AxonSimulation, axs.analysis.Activation]:
    axon = case.factory()
    instance = axs.AxonInstance(axon)
    positions_um = np.asarray(axon.layout.position_values(unit=axs.um), dtype=float)
    position = float(positions_um[len(positions_um) // 2]) * axs.um
    instance.add_current_clamp(
        position=position,
        current=axs.Stimulus.pulse(
            start=(case.dt_ms * 0.4) * axs.ms,
            duration=max(case.dt_ms * 0.4, 0.001) * axs.ms,
            amplitude=case.current_nA * axs.nA,
        ),
    )
    activation = axs.analysis.Activation(
        threshold=case.threshold_mV * axs.mV,
        blanking=0.0 * axs.ms,
        target=axs.positions.CENTER,
    )
    simulation = axs.AxonSimulation(
        instance,
        duration=case.duration_ms * axs.ms,
        dt=case.dt_ms * axs.ms,
        recording=axs.Recording.center(),
        observers=[activation],
    )
    return simulation, activation


def measure_simulation_run(
    case: SimulationCase,
    simulation: axs.AxonSimulation,
    activation: axs.analysis.Activation,
    *,
    phase: str,
    repeat: int,
) -> tuple[SimulationRow, Any]:
    start = time.perf_counter()
    run = simulation.run()
    seconds = time.perf_counter() - start
    result = run.single
    analysis = result.analyze(activation)
    vm = np.asarray(result.Vm)
    activated = bool(np.asarray(analysis.values)[0])
    event = analysis.events[0] if analysis.events else None
    return (
        SimulationRow(
            case=case.name,
            model=case.model,
            phase=phase,
            repeat=int(repeat),
            seconds=float(seconds),
            status="ok",
            vm_shape=tuple(int(value) for value in vm.shape),
            vm_peak_mV=float(np.max(vm)) if vm.size else None,
            activated=activated,
            first_time_ms=None if event is None else event.first_time_ms,
            note="",
        ),
        result,
    )


def check_simulation_result(
    case: SimulationCase,
    result: Any,
    activation: axs.analysis.Activation,
) -> CorrectnessRow:
    try:
        vm = np.asarray(result.Vm)
        if vm.ndim != 2 or vm.shape[0] < 1 or vm.shape[1] != 1:
            raise ValueError(f"unexpected center Vm shape {vm.shape}.")
        if not np.isfinite(vm).all():
            raise ValueError("Vm contains non-finite values.")
        analysis = result.analyze(activation)
        posthoc = bool(np.asarray(analysis.values)[0])
        observations = result.observations or {}
        raster = observations.get(axs.VM_RASTER_OBSERVATION_KEY)
        if raster is None:
            raise ValueError("simulation did not return vm_raster observations.")
        raster_activated = bool(np.any(raster.unpack()))
        if posthoc != raster_activated:
            raise ValueError(
                f"post-hoc activation={posthoc} does not match vm_raster={raster_activated}."
            )
        return CorrectnessRow(
            scope="simulation",
            case=case.name,
            target="vm_activation",
            status="ok",
            max_abs=0.0,
            max_rel=0.0,
            note="",
        )
    except Exception as exc:
        return CorrectnessRow(
            scope="simulation",
            case=case.name,
            target="vm_activation",
            status="failed",
            max_abs=None,
            max_rel=None,
            note=f"{type(exc).__name__}: {exc}",
        )


def run_metadata(
    cases: Sequence[ModelCase],
    *,
    simulation_cases: Sequence[SimulationCase],
    cache_root: Path,
    warm_repeats: int,
    step_repeats: int,
    step_nodes: int,
    simulation_warm_repeats: int,
    model_steps: bool,
    simulations: bool,
) -> dict[str, Any]:
    return {
        "benchmark": "model_codegen",
        "models": [case.name for case in cases],
        "groups": sorted({case.group for case in cases}),
        "simulation_cases": [case.name for case in simulation_cases],
        "warm_repeats": int(warm_repeats),
        "step_repeats": int(step_repeats),
        "step_nodes": int(step_nodes),
        "simulation_warm_repeats": int(simulation_warm_repeats),
        "model_steps": bool(model_steps),
        "simulations": bool(simulations),
        "cache_root": str(cache_root),
        "python": sys.version,
        "platform": platform.platform(),
        "axonscope_version": axs.__version__,
        "jax_version": getattr(jax, "__version__", None),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "environment": {
            "AXONSCOPE_MODEL_CODEGEN_CACHE": os.environ.get("AXONSCOPE_MODEL_CODEGEN_CACHE"),
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        },
    }


def write_outputs(
    *,
    codegen_rows: Sequence[CodegenRow],
    model_step_rows: Sequence[ModelStepRow],
    simulation_rows: Sequence[SimulationRow],
    correctness_rows: Sequence[CorrectnessRow],
    out_dir: Path,
    prefix: str,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / f"{prefix}.json",
        "codegen_csv": out_dir / f"{prefix}_codegen.csv",
        "model_step_csv": out_dir / f"{prefix}_model_steps.csv",
        "simulation_csv": out_dir / f"{prefix}_simulations.csv",
        "correctness_csv": out_dir / f"{prefix}_correctness.csv",
    }
    payload = {
        "metadata": metadata,
        "codegen_rows": [asdict(row) for row in codegen_rows],
        "model_step_rows": [asdict(row) for row in model_step_rows],
        "simulation_rows": [asdict(row) for row in simulation_rows],
        "correctness_rows": [asdict(row) for row in correctness_rows],
    }
    paths["json"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(
        paths["codegen_csv"],
        codegen_rows,
        fieldnames=(
            "model",
            "group",
            "phase",
            "repeat",
            "seconds",
            "cache_status",
            "cache_reason",
            "cache_key",
            "source_hash",
            "source_path",
            "generated_bytes",
            "generated_files",
        ),
    )
    write_csv(
        paths["model_step_csv"],
        model_step_rows,
        fieldnames=(
            "model",
            "group",
            "target",
            "repeat",
            "seconds",
            "nodes",
            "outputs",
            "status",
            "note",
        ),
    )
    write_csv(
        paths["simulation_csv"],
        simulation_rows,
        fieldnames=(
            "case",
            "model",
            "phase",
            "repeat",
            "seconds",
            "status",
            "vm_shape",
            "vm_peak_mV",
            "activated",
            "first_time_ms",
            "note",
        ),
    )
    write_csv(
        paths["correctness_csv"],
        correctness_rows,
        fieldnames=(
            "scope",
            "case",
            "target",
            "status",
            "max_abs",
            "max_rel",
            "note",
        ),
    )
    return paths


def write_csv(
    path: Path,
    rows: Sequence[Any],
    *,
    fieldnames: tuple[str, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            for key, value in tuple(data.items()):
                if isinstance(value, tuple):
                    data[key] = ";".join(str(item) for item in value)
            writer.writerow(data)


if __name__ == "__main__":
    main()
