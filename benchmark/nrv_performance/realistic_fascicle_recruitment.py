from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import resource
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Sequence

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "axonscope-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "axonscope-cache"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
from rich.console import Console
from rich.table import Table

import axonscope as axs
from axonscope.integrations import nrv as axs_nrv
from axonscope.protocols import activation as activation_protocols
from axonscope.timebase import simulation_step_count
from axonscope.utils.progress_reporting import current_rss_mib


DEFAULT_OUT_DIR = Path("benchmark/results/nrv_performance/realistic_fascicle_recruitment")
DEFAULT_REPORT_DIR = Path("benchmark/reports/nrv_performance/realistic_fascicle_recruitment")
EXAMPLE_PATH = Path("examples/with_nrv/01_realistic_fascicle_geometry_comparison.py")


@dataclass(frozen=True)
class StepRecord:
    name: str
    elapsed_s: float
    rss_before_mib: float | None
    rss_after_mib: float | None
    rss_delta_mib: float | None
    rss_peak_mib: float | None
    maxrss_after_mib: float | None


class PeakMemorySampler:
    def __init__(self, *, interval_s: float) -> None:
        self.interval_s = max(float(interval_s), 0.01)
        self.samples_mib: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakMemorySampler":
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()

    @property
    def peak_mib(self) -> float | None:
        if not self.samples_mib:
            return None
        return float(max(self.samples_mib))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        rss = current_rss_mib()
        if rss is not None:
            self.samples_mib.append(float(rss))


class StepRecorder:
    def __init__(self, *, sample_interval_s: float) -> None:
        self.sample_interval_s = float(sample_interval_s)
        self.records: list[StepRecord] = []

    def measure(self, name: str, func: Callable[[], Any]) -> Any:
        before = current_rss_mib()
        start = time.perf_counter()
        with PeakMemorySampler(interval_s=self.sample_interval_s) as sampler:
            value = func()
        elapsed = time.perf_counter() - start
        after = current_rss_mib()
        delta = None if before is None or after is None else float(after - before)
        self.records.append(
            StepRecord(
                name=name,
                elapsed_s=float(elapsed),
                rss_before_mib=before,
                rss_after_mib=after,
                rss_delta_mib=delta,
                rss_peak_mib=sampler.peak_mib,
                maxrss_after_mib=maxrss_mib(),
            )
        )
        return value


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prefix = args.prefix or datetime.now().strftime("realistic_fascicle_%Y%m%d_%H%M%S")
    console = Console(width=120)
    if args.dry_run:
        print_dry_run(console, args)
        return

    example = load_example_module()
    config = build_example_config(example, args)
    amplitudes_uA = np.linspace(
        float(args.amplitude_min_uA),
        float(args.amplitude_max_uA),
        int(args.amplitudes_count),
    )
    recorder = StepRecorder(sample_interval_s=float(args.memory_sample_interval_s))

    import nrv

    geometry_step_name = (
        "build_synthetic_nrv_nerve"
        if config.geometry_mode == "synthetic_4_fascicles"
        else "load_and_build_histology_nrv_nerve"
    )
    nerve_contour, fascicle_contours, nerve = recorder.measure(
        geometry_step_name,
        lambda: example.build_nrv_nerve_from_config(nrv, config),
    )
    life_setup = recorder.measure(
        "attach_life_fem_electrode",
        lambda: example.attach_life_electrode(nrv, nerve, config),
    )
    rows = recorder.measure(
        "extract_fiber_rows",
        lambda: axs_nrv.select_rows(
            axs_nrv.extract_fiber_rows(
                nerve,
                include_unmyelinated=bool(config.include_unmyelinated),
            ),
            limit=int(config.max_fibers),
        ),
    )
    simulated_rows = select_simulated_rows(
        rows,
        per_fascicle=int(args.simulated_fibers_per_fascicle),
    )
    if not simulated_rows:
        raise RuntimeError("No fibers selected for the benchmark.")

    first_context = recorder.measure(
        "nrv_fem_solve_first_footprint",
        lambda: example.build_axonscope_context(
            simulated_rows[0],
            config=config,
            life_setup=life_setup,
        ),
    )
    remaining_contexts = []
    if len(simulated_rows) > 1:
        remaining_contexts = recorder.measure(
            "nrv_cached_footprint_sampling",
            lambda: [
                example.build_axonscope_context(row, config=config, life_setup=life_setup)
                for row in simulated_rows[1:]
            ],
        )
    contexts = [first_context, *remaining_contexts]
    pool = recorder.measure(
        "build_axonscope_pool",
        lambda: [
            example.build_axonscope_simulation_from_context(
                context,
                config=config,
                current_uA=0.0,
            )
            for context in contexts
        ],
    )
    if bool(args.clear_jax_caches):
        clear_jax_caches()

    profile_dir = (
        args.report_dir / prefix / "axonscope_profile" if bool(args.profile_axonscope) else None
    )
    curve, profile_report = run_axonscope_recruitment(
        config,
        pool,
        amplitudes_uA=amplitudes_uA,
        profile_dir=profile_dir,
        protocol_progress=bool(args.protocol_progress),
        solver_progress=solver_progress_option(args),
        recorder=recorder,
    )
    activation_comparisons = None
    if bool(args.validate_nrv):
        activation_comparisons = recorder.measure(
            "nrv_validation_single_amplitude",
            lambda: validate_nrv_at_amplitude(
                nerve,
                config,
                simulated_rows,
                curve=curve,
                validation_current_uA=float(config.nrv_validation_current_uA),
            ),
        )

    summary = build_summary(
        args,
        config,
        rows=rows,
        simulated_rows=simulated_rows,
        contexts=contexts,
        pool=pool,
        amplitudes_uA=amplitudes_uA,
        curve=curve,
        profile_report=profile_report,
        activation_comparisons=activation_comparisons,
    )
    summary.update(step_summary_metrics(recorder.records))
    summary.update(nrv_full_sweep_estimate_metrics(summary))
    write_outputs(
        args,
        prefix=prefix,
        summary=summary,
        steps=recorder.records,
    )
    print_summary(
        console,
        prefix=prefix,
        summary=summary,
        steps=recorder.records,
        out_dir=args.out_dir,
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Instrument the realistic NRV LIFE/FEM fascicle recruitment pipeline "
            "used by the with_nrv example."
        )
    )
    parser.add_argument("--nerve-diameter-um", type=float, default=1_000.0)
    parser.add_argument("--nerve-length-um", type=float, default=10_000.0)
    parser.add_argument(
        "--geometry-mode",
        choices=("histology", "synthetic_4_fascicles"),
        default="histology",
    )
    parser.add_argument("--axons-per-fascicle", type=int, default=25)
    parser.add_argument("--percent-unmyelinated", type=float, default=0.7)
    parser.add_argument("--delta-trace-um", type=float, default=10.0)
    parser.add_argument("--synthetic-fascicle-diameter-um", type=float, default=250.0)
    parser.add_argument("--synthetic-fascicle-offset-um", type=float, default=250.0)
    parser.add_argument("--fascicle-contour-epsilon-fraction", type=float, default=0.002)
    parser.add_argument("--max-fibers", type=int, default=0)
    parser.add_argument(
        "--simulated-fibers-per-fascicle",
        type=int,
        default=10,
        help="0 keeps every extracted fiber. Defaults to a small balanced subset.",
    )
    parser.add_argument("--duration-ms", type=float, default=3.0)
    parser.add_argument("--dt-ms", type=float, default=0.001)
    parser.add_argument("--stimulus-start-ms", type=float, default=0.1)
    parser.add_argument("--pulse-duration-ms", type=float, default=0.1)
    parser.add_argument("--amplitudes-count", type=int, default=2)
    parser.add_argument("--amplitude-min-uA", type=float, default=0.0)
    parser.add_argument("--amplitude-max-uA", type=float, default=300.0)
    parser.add_argument("--nrv-validation-current-uA", type=float, default=60.0)
    parser.add_argument("--activation-threshold-mV", type=float, default=0.0)
    parser.add_argument("--time-chunk-steps", type=int, default=250)
    parser.add_argument(
        "--solver-progress",
        choices=("off", "plain", "rich"),
        default="off",
        help="Show AxonScope solver progress for the recruitment sweep.",
    )
    parser.add_argument("--protocol-progress", action="store_true")
    parser.add_argument("--profile-axonscope", action="store_true")
    parser.add_argument("--clear-jax-caches", action="store_true")
    parser.add_argument("--validate-nrv", action="store_true")
    parser.add_argument("--fem-n-proc", type=int, default=0)
    parser.add_argument(
        "--gmsh-n-core",
        type=int,
        default=1,
        help=(
            "Gmsh mesh thread count for NRV FEM. 1 keeps the robust Delaunay path; "
            "0 uses NRV's configured default."
        ),
    )
    parser.add_argument("--memory-sample-interval-s", type=float, default=0.1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if int(args.axons_per_fascicle) < 1:
        raise ValueError("--axons-per-fascicle must be >= 1.")
    if int(args.max_fibers) < 0:
        raise ValueError("--max-fibers must be >= 0.")
    if int(args.simulated_fibers_per_fascicle) < 0:
        raise ValueError("--simulated-fibers-per-fascicle must be >= 0.")
    if float(args.synthetic_fascicle_diameter_um) <= 0.0:
        raise ValueError("--synthetic-fascicle-diameter-um must be > 0.")
    if float(args.synthetic_fascicle_offset_um) < 0.0:
        raise ValueError("--synthetic-fascicle-offset-um must be >= 0.")
    synthetic_radius = float(args.synthetic_fascicle_diameter_um) / 2.0
    synthetic_outer_radius = float(args.synthetic_fascicle_offset_um) + synthetic_radius
    if synthetic_outer_radius >= float(args.nerve_diameter_um) / 2.0:
        raise ValueError("Synthetic fascicles must fit inside the nerve diameter.")
    if float(args.fascicle_contour_epsilon_fraction) < 0.0:
        raise ValueError("--fascicle-contour-epsilon-fraction must be >= 0.")
    if int(args.amplitudes_count) < 1:
        raise ValueError("--amplitudes-count must be >= 1.")
    if float(args.duration_ms) <= 0.0 or float(args.dt_ms) <= 0.0:
        raise ValueError("--duration-ms and --dt-ms must be > 0.")
    simulation_step_count(float(args.duration_ms), float(args.dt_ms))
    if int(args.time_chunk_steps) < -1:
        raise ValueError("--time-chunk-steps must be >= -1.")
    if int(args.fem_n_proc) < 0:
        raise ValueError("--fem-n-proc must be >= 0.")
    if int(args.gmsh_n_core) < 0:
        raise ValueError("--gmsh-n-core must be >= 0.")


def load_example_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    example_path = repo_root / EXAMPLE_PATH
    spec = importlib.util.spec_from_file_location(
        "axonscope_with_nrv_realistic_fascicle_example",
        example_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {example_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_example_config(example: ModuleType, args: argparse.Namespace) -> Any:
    time_chunk_steps = None if int(args.time_chunk_steps) < 0 else int(args.time_chunk_steps)
    return example.ExampleConfig(
        nerve_diameter_um=float(args.nerve_diameter_um),
        nerve_length_um=float(args.nerve_length_um),
        geometry_mode=str(args.geometry_mode),
        axons_per_fascicle=int(args.axons_per_fascicle),
        percent_unmyelinated=float(args.percent_unmyelinated),
        delta_trace_um=float(args.delta_trace_um),
        synthetic_fascicle_diameter_um=float(args.synthetic_fascicle_diameter_um),
        synthetic_fascicle_offset_um=float(args.synthetic_fascicle_offset_um),
        fascicle_contour_epsilon_fraction=float(args.fascicle_contour_epsilon_fraction),
        include_unmyelinated=True,
        max_fibers=int(args.max_fibers),
        simulate_fibers=0,
        run_simulation=True,
        duration_ms=float(args.duration_ms),
        dt_ms=float(args.dt_ms),
        stimulus_start_ms=float(args.stimulus_start_ms),
        pulse_duration_ms=float(args.pulse_duration_ms),
        observer_time_chunk_steps=time_chunk_steps,
        solver_progress=solver_progress_option(args),
        recruitment_amplitudes_uA=tuple(
            float(value)
            for value in np.linspace(
                float(args.amplitude_min_uA),
                float(args.amplitude_max_uA),
                int(args.amplitudes_count),
            )
        ),
        nrv_validation_current_uA=float(args.nrv_validation_current_uA),
        activation_threshold_mV=float(args.activation_threshold_mV),
        fem_n_proc=None if int(args.fem_n_proc) == 0 else int(args.fem_n_proc),
        gmsh_n_core=None if int(args.gmsh_n_core) == 0 else int(args.gmsh_n_core),
    )


def solver_progress_option(args: argparse.Namespace) -> bool | str:
    if args.solver_progress == "off":
        return False
    return str(args.solver_progress)


def select_simulated_rows(rows: Sequence[Any], *, per_fascicle: int) -> list[Any]:
    if int(per_fascicle) <= 0:
        return list(rows)
    counts: dict[str, int] = {}
    selected = []
    for row in rows:
        fascicle_id = str(row.fascicle_id)
        count = counts.get(fascicle_id, 0)
        if count >= int(per_fascicle):
            continue
        selected.append(row)
        counts[fascicle_id] = count + 1
    return selected


def run_axonscope_recruitment(
    config: Any,
    pool: Sequence[Any],
    *,
    amplitudes_uA: np.ndarray,
    profile_dir: Path | None,
    protocol_progress: bool,
    solver_progress: bool | str,
    recorder: StepRecorder,
) -> tuple[Any, Any | None]:
    criterion = axs.analysis.ActivationCriterion(
        threshold=float(config.activation_threshold_mV) * axs.mV,
        blanking=float(config.stimulus_start_ms) * axs.ms,
        target=axs.positions.ALL,
    )

    def update_life_current(simulation: axs.AxonInstance, current_magnitude: Any) -> None:
        stimulation = simulation.extracellular_stimulation
        if stimulation is None:
            raise ValueError("simulation has no extracellular stimulation to update.")
        drive = stimulation.drives[0]
        updated = stimulation.replace_drive(
            drive.id,
            stimulus=axs.Stimulus.pulse(
                start=float(config.stimulus_start_ms) * axs.ms,
                duration=float(config.pulse_duration_ms) * axs.ms,
                amplitude=-current_magnitude,
            ),
        )
        simulation.add_extracellular_stimulation(stimulation=updated, replace=True)

    activation = axs.analysis.Activation(
        threshold=criterion.threshold,
        blanking=criterion.blanking,
        target=criterion.target,
    )
    batch_options = axs.BatchOptions.none(
        time_chunk_steps=config.observer_time_chunk_steps
    )
    base_pool = tuple(pool)
    observation_rows: list[np.ndarray] = []
    report = None

    def evaluate_step(index: int, amplitude_uA: float) -> np.ndarray:
        current = float(amplitude_uA) * axs.uA
        for simulation in base_pool:
            update_life_current(simulation, current)
        pool_result = axs.simulate_pool(
            base_pool,
            duration=float(config.duration_ms) * axs.ms,
            dt=float(config.dt_ms) * axs.ms,
            recording=axs.Recording.none(),
            batch_options=batch_options,
            observers=(activation,),
            progress=solver_progress if index == 0 else False,
        )
        return activation_protocols._activation_observations_from_pool_result(
            pool_result,
            activation,
        )

    if profile_dir is not None:
        axs.enable_benchmark(
            profile_dir,
            print_summary=False,
            save=True,
            sync_device=True,
            record_shapes=True,
            record_memory=True,
        )
    try:
        for index, amplitude_uA in enumerate(np.asarray(amplitudes_uA, dtype=float)):
            phase = "cold" if index == 0 else "warm" if index == 1 else f"warm_{index}"
            observations = recorder.measure(
                f"axonscope_step_{index}_{phase}",
                lambda index=index, amplitude_uA=float(amplitude_uA): evaluate_step(
                    index,
                    amplitude_uA,
                ),
            )
            observation_rows.append(np.asarray(observations, dtype=bool))
            if protocol_progress:
                activated = int(np.count_nonzero(observations))
                print(
                    f"amplitude step {index}: {float(amplitude_uA):.3f} uA, "
                    f"activated={activated}/{len(base_pool)}"
                )
    finally:
        if profile_dir is not None:
            report = axs.disable_benchmark(print_summary=False, save=True)
    curve = axs.protocols.RecruitmentCurve(
        amplitudes_uA=np.asarray(amplitudes_uA, dtype=float),
        activated=np.asarray(observation_rows, dtype=bool),
    )
    return curve, report


def validate_nrv_at_amplitude(
    nerve: Any,
    config: Any,
    rows: Sequence[Any],
    *,
    curve: Any,
    validation_current_uA: float,
) -> list[Any]:
    nrv_result = nerve.simulate(
        t_sim=float(config.duration_ms),
        postproc_script="is_recruited",
        dt=float(config.dt_ms),
        unmyelinated_nseg=max(3, int(float(config.nerve_length_um) // 25)),
        myelinated_nseg_per_sec=3,
    )
    nrv_activated = axs_nrv.nrv_activation_by_row(
        nrv_result,
        nerve,
        rows,
        t_start_ms=float(config.stimulus_start_ms),
    )
    validation_index = int(
        np.argmin(np.abs(np.asarray(curve.amplitudes_uA, dtype=float) - validation_current_uA))
    )
    return axs_nrv.activation_comparisons(
        rows,
        nrv_activated=nrv_activated,
        axonscope_activated=np.asarray(curve.activated[validation_index], dtype=bool),
    )


def build_summary(
    args: argparse.Namespace,
    config: Any,
    *,
    rows: Sequence[Any],
    simulated_rows: Sequence[Any],
    contexts: Sequence[Any],
    pool: Sequence[Any],
    amplitudes_uA: np.ndarray,
    curve: Any,
    profile_report: Any | None,
    activation_comparisons: Sequence[Any] | None,
) -> dict[str, Any]:
    nt = simulation_step_count(float(config.duration_ms), float(config.dt_ms))
    chunk_steps = config.observer_time_chunk_steps
    chunk_count = 1 if chunk_steps is None or int(chunk_steps) >= nt else int(np.ceil(nt / int(chunk_steps)))
    mrg_count = sum(row.kind == "mrg" for row in simulated_rows)
    single_count = len(simulated_rows) - mrg_count
    amplitudes_count = int(len(amplitudes_uA))
    profile_metrics = profile_report_metrics(profile_report)
    footprint_bytes = int(footprint_storage_bytes(contexts))
    dense_vm_fp32_per_step = int(
        dense_vm_bytes(pool, nt=nt, amplitudes_count=1, bytes_per_sample=4)
    )
    dense_vm_fp64_per_step = int(
        dense_vm_bytes(pool, nt=nt, amplitudes_count=1, bytes_per_sample=8)
    )
    summary = {
        "axons_per_fascicle": int(config.axons_per_fascicle),
        "geometry_mode": str(config.geometry_mode),
        "synthetic_fascicle_diameter_um": float(config.synthetic_fascicle_diameter_um),
        "synthetic_fascicle_offset_um": float(config.synthetic_fascicle_offset_um),
        "fascicle_contour_epsilon_fraction": float(
            config.fascicle_contour_epsilon_fraction
        ),
        "total_extracted_fibers": int(len(rows)),
        "simulated_fibers": int(len(simulated_rows)),
        "simulated_mrg_fibers": int(mrg_count),
        "simulated_single_cable_fibers": int(single_count),
        "amplitudes_count": amplitudes_count,
        "amplitude_min_uA": float(np.min(amplitudes_uA)),
        "amplitude_max_uA": float(np.max(amplitudes_uA)),
        "expanded_solver_rows": int(len(simulated_rows) * amplitudes_count),
        "expanded_mrg_rows": int(mrg_count * amplitudes_count),
        "expanded_single_cable_rows": int(single_count * amplitudes_count),
        "solver_rows_per_step": int(len(simulated_rows)),
        "mrg_rows_per_step": int(mrg_count),
        "single_cable_rows_per_step": int(single_count),
        "duration_ms": float(config.duration_ms),
        "dt_ms": float(config.dt_ms),
        "nt": int(nt),
        "fem_n_proc": None if config.fem_n_proc is None else int(config.fem_n_proc),
        "gmsh_n_core": None if config.gmsh_n_core is None else int(config.gmsh_n_core),
        "observer_time_chunk_steps": None if chunk_steps is None else int(chunk_steps),
        "chunks_per_group": int(chunk_count),
        "footprint_storage_bytes": footprint_bytes,
        "nrv_footprint_count": int(len(contexts)),
        "nrv_cached_footprint_count": max(int(len(contexts)) - 1, 0),
        "estimated_factorized_footprint_bytes_per_step": footprint_bytes,
        "estimated_factorized_footprint_bytes_if_batched": int(
            footprint_bytes * amplitudes_count
        ),
        "estimated_dense_vm_bytes_fp32_per_step": dense_vm_fp32_per_step,
        "estimated_dense_vm_bytes_fp32_if_batched": int(
            dense_vm_fp32_per_step * amplitudes_count
        ),
        "estimated_dense_vm_bytes_fp64_per_step": dense_vm_fp64_per_step,
        "estimated_dense_vm_bytes_fp64_if_batched": int(
            dense_vm_fp64_per_step * amplitudes_count
        ),
        "final_recruited_count": int(np.count_nonzero(np.asarray(curve.activated[-1], dtype=bool))),
        "jax_backend": current_jax_backend(),
        "current_rss_mib": current_rss_mib(),
        "maxrss_mib": maxrss_mib(),
    }
    if activation_comparisons is not None:
        matched = sum(item.matched for item in activation_comparisons)
        summary.update(
            {
                "nrv_validation_fibers": int(len(activation_comparisons)),
                "nrv_validation_matched": int(matched),
                "nrv_validation_match_fraction": (
                    None
                    if not activation_comparisons
                    else float(matched / len(activation_comparisons))
                ),
            }
        )
    summary.update(profile_metrics)
    return summary


def step_summary_metrics(steps: Sequence[StepRecord]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    axonscope_steps = [
        step for step in steps if step.name.startswith("axonscope_step_")
    ]
    if axonscope_steps:
        metrics["axonscope_steps_total_s"] = float(
            sum(step.elapsed_s for step in axonscope_steps)
        )
        metrics["axonscope_steps_peak_mib"] = max_optional(
            step.rss_peak_mib for step in axonscope_steps
        )
    for step in axonscope_steps:
        if "_cold" in step.name:
            prefix = "axonscope_cold_step"
        elif "_warm" in step.name:
            prefix = "axonscope_warm_step"
        else:
            prefix = step.name
        metrics[f"{prefix}_s"] = float(step.elapsed_s)
        metrics[f"{prefix}_rss_delta_mib"] = step.rss_delta_mib
        metrics[f"{prefix}_rss_peak_mib"] = step.rss_peak_mib
    fem_step = next(
        (step for step in steps if step.name == "nrv_fem_solve_first_footprint"),
        None,
    )
    cached_step = next(
        (step for step in steps if step.name == "nrv_cached_footprint_sampling"),
        None,
    )
    if fem_step is not None:
        metrics["nrv_fem_first_footprint_s"] = float(fem_step.elapsed_s)
        metrics["nrv_fem_first_footprint_rss_delta_mib"] = fem_step.rss_delta_mib
        metrics["nrv_fem_first_footprint_peak_mib"] = fem_step.rss_peak_mib
    if cached_step is not None:
        metrics["nrv_cached_footprints_s"] = float(cached_step.elapsed_s)
        metrics["nrv_cached_footprints_rss_delta_mib"] = cached_step.rss_delta_mib
        metrics["nrv_cached_footprints_peak_mib"] = cached_step.rss_peak_mib
    nrv_validation_step = next(
        (step for step in steps if step.name == "nrv_validation_single_amplitude"),
        None,
    )
    if nrv_validation_step is not None:
        metrics["nrv_validation_single_amplitude_s"] = float(nrv_validation_step.elapsed_s)
        metrics["nrv_validation_single_amplitude_rss_delta_mib"] = (
            nrv_validation_step.rss_delta_mib
        )
        metrics["nrv_validation_single_amplitude_peak_mib"] = (
            nrv_validation_step.rss_peak_mib
        )
    return metrics


def nrv_full_sweep_estimate_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    nrv_single_s = numeric_float(summary.get("nrv_validation_single_amplitude_s"))
    if nrv_single_s is None:
        return {}
    amplitude_count = max(int(summary.get("amplitudes_count", 1)), 1)
    estimate_s = float(nrv_single_s) * amplitude_count
    metrics: dict[str, Any] = {
        "nrv_full_sweep_estimated_s": estimate_s,
    }
    axonscope_s = numeric_float(summary.get("axonscope_steps_total_s"))
    if axonscope_s is not None and axonscope_s > 0.0:
        metrics["nrv_estimate_over_axonscope_steps"] = estimate_s / axonscope_s
    footprint_s = sum(
        value
        for value in (
            numeric_float(summary.get("nrv_fem_first_footprint_s")),
            numeric_float(summary.get("nrv_cached_footprints_s")),
        )
        if value is not None
    )
    if axonscope_s is not None and axonscope_s + footprint_s > 0.0:
        metrics["nrv_estimate_over_axonscope_with_footprints"] = (
            estimate_s / (axonscope_s + footprint_s)
        )
    return metrics


def numeric_float(value: Any) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def max_optional(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return max(numeric)


def footprint_storage_bytes(contexts: Sequence[Any]) -> int:
    total = 0
    for context in contexts:
        footprint = context.footprint
        total += int(np.asarray(footprint.values_V_per_A).nbytes)
        total += int(np.asarray(footprint.positions_um).nbytes)
    return total


def dense_vm_bytes(
    pool: Sequence[Any],
    *,
    nt: int,
    amplitudes_count: int,
    bytes_per_sample: int,
) -> int:
    total_nx = 0
    for simulation in pool:
        total_nx += int(simulation.axon.n_compartments)
    return int(total_nx * int(nt) * int(amplitudes_count) * int(bytes_per_sample))


def profile_report_metrics(report: Any | None) -> dict[str, Any]:
    if report is None:
        return {}
    metrics: dict[str, Any] = {
        "profile_event_count": int(len(getattr(report, "events", ()))),
    }
    summary_by_name = {
        str(row.name): float(row.total_ms)
        for row in getattr(report, "summary", ())
    }
    for event_name in (
        "simulation.pool.total",
        "dispatch.build_plan",
        "dispatch.group.total",
        "runtime.prepare",
        "inputs.extracellular",
        "observer.plan",
        "kernel.enqueue",
        "kernel.dispatch_jax",
        "kernel.wait",
        "kernel.finalize_observer",
        "results.to_public",
    ):
        value = summary_by_name.get(event_name)
        if value is not None:
            metrics[f"profile_{event_name.replace('.', '_')}_ms"] = value

    for event in getattr(report, "events", ()):
        metadata = getattr(event, "metadata", {})
        if not isinstance(metadata, dict):
            continue
        for key, value in metadata.items():
            if key.endswith("_nbytes"):
                record_profile_nbytes_metric(metrics, key, value)
            elif key in {
                "vstim_input_format",
                "vstim_footprint_cache",
                "shared_current",
            }:
                metrics[f"profile_{key}"] = normalize_json_value(value)
    report_metadata = getattr(report, "metadata", {})
    if isinstance(report_metadata, dict) and report_metadata.get("output_dir") is not None:
        metrics["profile_dir"] = str(report_metadata["output_dir"])
    return metrics


def record_profile_nbytes_metric(
    metrics: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    if isinstance(value, dict):
        total = 0
        has_numeric_component = False
        for component_key, component_value in value.items():
            numeric = numeric_nbytes_value(component_value)
            if numeric is None:
                continue
            has_numeric_component = True
            total += numeric
            component_metric = f"profile_{key}_{metric_suffix(component_key)}"
            set_max_int_metric(metrics, component_metric, numeric)
        if has_numeric_component:
            set_max_int_metric(metrics, f"profile_{key}", total)
        return

    numeric = numeric_nbytes_value(value)
    if numeric is None:
        return
    set_max_int_metric(metrics, f"profile_{key}", numeric)


def numeric_nbytes_value(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return int(value)
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return None
        if not np.isfinite(numeric):
            return None
        return int(numeric)
    return None


def set_max_int_metric(metrics: dict[str, Any], key: str, value: int) -> None:
    previous = numeric_nbytes_value(metrics.get(key))
    metrics[key] = int(value) if previous is None else max(previous, int(value))


def metric_suffix(value: Any) -> str:
    text = str(value).strip().lower()
    chars: list[str] = []
    previous_was_separator = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            previous_was_separator = False
        elif not previous_was_separator:
            chars.append("_")
            previous_was_separator = True
    suffix = "".join(chars).strip("_")
    return suffix or "value"


def write_outputs(
    args: argparse.Namespace,
    *,
    prefix: str,
    summary: dict[str, Any],
    steps: Sequence[StepRecord],
) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{prefix}.json"
    summary_csv = args.out_dir / f"{prefix}_summary.csv"
    steps_csv = args.out_dir / f"{prefix}_steps.csv"
    payload = {
        "summary": summary,
        "steps": [asdict(step) for step in steps],
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=normalize_json_value)
    write_csv(summary_csv, [summary])
    write_csv(steps_csv, [asdict(step) for step in steps])


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    console: Console,
    *,
    prefix: str,
    summary: dict[str, Any],
    steps: Sequence[StepRecord],
    out_dir: Path,
) -> None:
    console.print(f"\n[bold]Realistic fascicle recruitment benchmark[/bold] {prefix}")
    workload = Table(title="Workload")
    workload.add_column("item")
    workload.add_column("value", justify="right")
    for key in (
        "simulated_fibers",
        "amplitudes_count",
        "solver_rows_per_step",
        "expanded_solver_rows",
        "expanded_mrg_rows",
        "expanded_single_cable_rows",
        "nt",
        "observer_time_chunk_steps",
        "chunks_per_group",
        "nrv_footprint_count",
        "nrv_cached_footprint_count",
        "footprint_storage_bytes",
        "nrv_fem_first_footprint_s",
        "nrv_cached_footprints_s",
        "nrv_validation_single_amplitude_s",
        "nrv_full_sweep_estimated_s",
        "nrv_estimate_over_axonscope_steps",
        "nrv_estimate_over_axonscope_with_footprints",
        "estimated_factorized_footprint_bytes_per_step",
        "estimated_factorized_footprint_bytes_if_batched",
        "estimated_dense_vm_bytes_fp32_per_step",
        "estimated_dense_vm_bytes_fp32_if_batched",
        "axonscope_cold_step_s",
        "axonscope_warm_step_s",
        "axonscope_steps_total_s",
        "maxrss_mib",
    ):
        workload.add_row(key, format_summary_value(summary.get(key)))
    console.print(workload)

    step_table = Table(title="Step timings and memory")
    for column in ("step", "time (s)", "rss before", "rss after", "delta", "peak"):
        step_table.add_column(column, justify="right" if column != "step" else "left")
    for step in steps:
        step_table.add_row(
            step.name,
            f"{step.elapsed_s:.3f}",
            format_mib(step.rss_before_mib),
            format_mib(step.rss_after_mib),
            format_signed_mib(step.rss_delta_mib),
            format_mib(step.rss_peak_mib),
        )
    console.print(step_table)
    console.print(f"[dim]json/csv written under {out_dir}.[/dim]")


def print_dry_run(console: Console, args: argparse.Namespace) -> None:
    nt = simulation_step_count(float(args.duration_ms), float(args.dt_ms))
    chunk_steps = None if int(args.time_chunk_steps) < 0 else int(args.time_chunk_steps)
    chunk_count = 1 if chunk_steps is None or chunk_steps >= nt else int(np.ceil(nt / chunk_steps))
    console.print("[bold]Realistic fascicle benchmark dry run[/bold]")
    console.print(
        f"axons_per_fascicle={args.axons_per_fascicle}, "
        f"geometry_mode={args.geometry_mode}, "
        f"simulated_fibers_per_fascicle={args.simulated_fibers_per_fascicle}, "
        f"sequential_amplitude_steps={args.amplitudes_count}, "
        f"gmsh_n_core={args.gmsh_n_core}, "
        f"nt={nt}, chunks_per_group={chunk_count}"
    )


def current_jax_backend() -> str | None:
    try:
        import jax

        return str(jax.default_backend())
    except Exception:
        return None


def clear_jax_caches() -> None:
    try:
        import jax
    except Exception:
        return
    clear = getattr(jax, "clear_caches", None)
    if callable(clear):
        clear()


def maxrss_mib() -> float | None:
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def format_mib(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f} MiB"


def format_signed_mib(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.1f} MiB"


def format_summary_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
