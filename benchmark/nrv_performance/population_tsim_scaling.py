from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Any, Literal, Sequence

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "axonscope-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "axonscope-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import axonscope as axs
from axonscope.integrations import nrv as axs_nrv
from axonscope.solvers import BatchOptions


PopulationKind = Literal["mrg", "rattay", "mixed"]
AxonScopeMode = Literal["observer", "full-vm"]
RunnerKind = Literal["both", "axonscope", "nrv"]
GeometrySource = Literal["nrv", "synthetic"]
DeviceRequest = Literal["auto", "cpu", "gpu"]

DEFAULT_OUT_DIR = Path("benchmark/results/nrv_performance/population_tsim_scaling")
DEFAULT_REPORT_DIR = Path("benchmark/reports/nrv_performance")


@dataclass(frozen=True)
class TimedValue:
    elapsed_s: float
    value: Any


COLD_PATH_EVENT_COLUMNS = {
    "simulation.pool.total": "simulation_pool_ms",
    "dispatch.build_plan": "dispatch_build_plan_ms",
    "dispatch.group.total": "dispatch_group_total_ms",
    "runtime.prepare": "runtime_prepare_ms",
    "runtime.prepare.base_runtime": "runtime_prepare_base_runtime_ms",
    "runtime.prepare.membrane_compile": "runtime_prepare_membrane_compile_ms",
    "runtime.prepare.membrane_backend": "runtime_prepare_membrane_backend_ms",
    "runtime.prepare.membrane_init": "runtime_prepare_membrane_init_ms",
    "runtime.prepare.stack_cable": "runtime_prepare_stack_cable_ms",
    "runtime.prepare.stack_extracellular": "runtime_prepare_stack_extracellular_ms",
    "runtime.prepare.stack_membrane": "runtime_prepare_stack_membrane_ms",
    "inputs.positions": "inputs_positions_ms",
    "observer.plan": "observer_plan_ms",
    "inputs.intracellular": "inputs_intracellular_ms",
    "inputs.extracellular": "inputs_extracellular_ms",
    "kernel.enqueue": "kernel_enqueue_ms",
    "kernel.dispatch_jax": "kernel_dispatch_jax_ms",
    "kernel.wait": "kernel_wait_ms",
    "kernel.finalize_observer": "kernel_finalize_observer_ms",
    "results.split_batch": "results_split_batch_ms",
    "results.to_public": "results_to_public_ms",
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare AxonScope and NRV population runtime versus simulated time "
            "on a small point-source fascicle."
        )
    )
    parser.add_argument("--population", choices=("mrg", "rattay", "mixed"), default="mixed")
    parser.add_argument("--fiber-counts", nargs="+", type=int, default=(25, 50, 100))
    parser.add_argument("--tsim", nargs="+", type=float, default=(0.5, 1.0, 2.0))
    parser.add_argument("--dt-ms", type=float, default=0.01)
    parser.add_argument("--length-um", type=float, default=10_000.0)
    parser.add_argument("--fascicle-diameter-um", type=float, default=800.0)
    parser.add_argument("--percent-unmyelinated", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--mode", choices=("observer", "full-vm"), default="observer")
    parser.add_argument("--runner", choices=("both", "axonscope", "nrv"), default="both")
    parser.add_argument(
        "--geometry-source",
        choices=("nrv", "synthetic"),
        default="nrv",
        help=(
            "Fiber placement source. 'nrv' preserves the AxonScope-vs-NRV comparison; "
            "'synthetic' builds the same AxonScope population shape without requiring NRV."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="AxonScope execution device request used through ExecutionPolicy.",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument(
        "--time-chunk-steps",
        type=int,
        default=0,
        help=(
            "Observer mode: 0 uses the default observer chunk policy, -1 disables "
            "chunking, >0 forces that chunk size. Full-Vm mode: <=0 disables chunking."
        ),
    )
    parser.add_argument("--nrv-processes", type=int, default=0)
    parser.add_argument(
        "--profile-cold-path",
        action="store_true",
        help="Save AxonScope hotpath events for the first run of each case.",
    )
    parser.add_argument(
        "--profile-warm-path",
        action="store_true",
        help="Save AxonScope hotpath events for the first measured warm repeat.",
    )
    parser.add_argument(
        "--clear-jax-caches",
        action="store_true",
        help="Clear JAX in-process caches before each measured AxonScope cold run.",
    )
    parser.add_argument("--stimulus-start-ms", type=float, default=0.1)
    parser.add_argument("--pulse-duration-ms", type=float, default=0.1)
    parser.add_argument("--stimulus-current-uA", type=float, default=60.0)
    parser.add_argument("--electrode-y-um", type=float, default=100.0)
    parser.add_argument("--electrode-z-um", type=float, default=0.0)
    parser.add_argument("--sigma-S-m", type=float, default=0.2)
    parser.add_argument("--activation-threshold-mV", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)

    validate_args(args)
    cases = [
        (int(fiber_count), float(tsim_ms))
        for fiber_count in args.fiber_counts
        for tsim_ms in args.tsim
    ]
    if args.dry_run:
        print_cases(args, cases)
        return

    prefix = args.prefix or datetime.now().strftime("population_tsim_%Y%m%d_%H%M%S")
    setattr(args, "run_prefix", prefix)
    rows = [run_case(args, fiber_count=fiber_count, tsim_ms=tsim_ms) for fiber_count, tsim_ms in cases]
    json_path, csv_path = write_results(rows, args.out_dir, prefix)
    plot_path = None
    if not args.no_plot:
        plot_path = write_plot(rows, args.report_dir, prefix)
    print_summary(rows)
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")
    if plot_path is not None:
        print(f"plot: {plot_path}")


def validate_args(args: argparse.Namespace) -> None:
    max_fiber_count = 2_000 if args.geometry_source == "synthetic" else 100
    if any(count < 1 or count > max_fiber_count for count in args.fiber_counts):
        raise ValueError(f"--fiber-counts values must be in [1, {max_fiber_count}].")
    if any(value <= 0.0 for value in args.tsim):
        raise ValueError("--tsim values must be > 0.")
    if float(args.dt_ms) <= 0.0:
        raise ValueError("--dt-ms must be > 0.")
    if int(args.repeats) < 1:
        raise ValueError("--repeats must be >= 1.")
    if int(args.warmups) < 0:
        raise ValueError("--warmups must be >= 0.")
    if bool(args.profile_warm_path) and int(args.repeats) < 2:
        raise ValueError("--profile-warm-path requires --repeats >= 2.")
    if int(args.time_chunk_steps) < -1:
        raise ValueError("--time-chunk-steps must be >= -1.")
    if int(args.nrv_processes) < 0:
        raise ValueError("--nrv-processes must be >= 0.")
    if args.runner in {"both", "nrv"} and args.geometry_source != "nrv":
        raise ValueError("--runner both/nrv requires --geometry-source nrv.")
    if args.device == "gpu" and int(args.gpu_index) < 0:
        raise ValueError("--gpu-index must be >= 0.")


def print_cases(args: argparse.Namespace, cases: Sequence[tuple[int, float]]) -> None:
    print("Population tsim scaling cases:")
    print(
        f"  population={args.population} mode={args.mode} runner={args.runner} "
        f"geometry={args.geometry_source} device={args.device} "
        f"dt={args.dt_ms:g} ms repeats={args.repeats} warmups={args.warmups} "
        f"profile_cold={bool(args.profile_cold_path)} "
        f"profile_warm={bool(args.profile_warm_path)} "
        f"clear_jax_caches={bool(args.clear_jax_caches)}"
    )
    for fiber_count, tsim_ms in cases:
        print(f"  fibers={fiber_count:3d} tsim={tsim_ms:g} ms")


def run_case(args: argparse.Namespace, *, fiber_count: int, tsim_ms: float) -> dict[str, Any]:
    print(f"\n=== population={args.population} fibers={fiber_count} tsim={tsim_ms:g} ms ===")
    nrv_build: TimedValue | None = None
    fascicle = None
    if args.geometry_source == "nrv":
        nrv_build = timed(lambda: build_nrv_fascicle(args, fiber_count=fiber_count))
        fascicle = nrv_build.value
        rows = extract_rows(fascicle, population=args.population)
    else:
        rows = build_synthetic_rows(args, fiber_count=fiber_count)
    if not rows:
        raise RuntimeError(f"{args.geometry_source} generated no simulated fibers.")
    counts = fiber_counts(rows)
    nt = int(round(float(tsim_ms) / float(args.dt_ms)))
    activation_name = "activation"

    as_build: TimedValue | None = None
    as_first: TimedValue | None = None
    as_materialize: TimedValue | None = None
    as_warm_samples: list[float] = []
    as_total_samples: list[float] = []
    as_activated: int | None = None
    as_route_summary: dict[str, Any] = {}
    as_observer_bytes = None
    as_dense_bytes = None
    as_cold_report_metrics: dict[str, Any] = {}
    as_warm_report_metrics: dict[str, Any] = {}

    if args.runner in {"both", "axonscope"}:
        as_build = timed(lambda: build_axonscope_pool(args, rows))
        pool = as_build.value
        activation = axs.Activation(
            threshold=float(args.activation_threshold_mV) * axs.mV,
            blanking=float(args.stimulus_start_ms) * axs.ms,
            target=axs.positions.ALL,
            name=activation_name,
        )
        if bool(args.clear_jax_caches):
            clear_jax_caches()
        as_first, as_first_report = timed_axonscope_run(
            args,
            pool,
            activation,
            tsim_ms=tsim_ms,
            profile_dir=axonscope_profile_dir(args, fiber_count, tsim_ms, "cold")
            if bool(args.profile_cold_path)
            else None,
        )
        as_cold_report_metrics = benchmark_report_metrics(as_first_report, prefix="as_cold")
        as_materialize = timed(
            lambda: materialize_axonscope_summary(as_first.value, activation, mode=args.mode)
        )
        as_summary = as_materialize.value
        as_activated = int(as_summary["activated"])
        as_observer_bytes = as_summary.get("observer_bytes")
        as_dense_bytes = as_summary.get("dense_vm_bytes")
        as_route_summary = summarize_axonscope_routes(as_first.value)

        for _ in range(int(args.warmups)):
            warm = run_and_block_axonscope(args, pool, activation, tsim_ms=tsim_ms)
            materialize_axonscope_summary(warm, activation, mode=args.mode)
        for repeat_index in range(max(0, int(args.repeats) - 1)):
            sample_profile_dir = (
                axonscope_profile_dir(args, fiber_count, tsim_ms, "warm")
                if bool(args.profile_warm_path) and repeat_index == 0
                else None
            )
            sample, sample_report = timed_axonscope_run(
                args,
                pool,
                activation,
                tsim_ms=tsim_ms,
                profile_dir=sample_profile_dir,
            )
            if sample_report is not None:
                as_warm_report_metrics = benchmark_report_metrics(sample_report, prefix="as_warm")
            mat = timed(lambda: materialize_axonscope_summary(sample.value, activation, mode=args.mode))
            as_warm_samples.append(float(sample.elapsed_s))
            as_total_samples.append(float(sample.elapsed_s + mat.elapsed_s))

    nrv_first: TimedValue | None = None
    nrv_materialize: TimedValue | None = None
    nrv_warm_samples: list[float] = []
    nrv_total_samples: list[float] = []
    nrv_activated: int | None = None

    if args.runner in {"both", "nrv"}:
        if fascicle is None:
            raise RuntimeError("NRV runner requires an NRV geometry source.")
        nrv_first = timed(lambda: run_nrv(fascicle, args, tsim_ms=tsim_ms))
        nrv_materialize = timed(lambda: materialize_nrv_summary(nrv_first.value))
        nrv_activated = int(nrv_materialize.value["activated"])

        for _ in range(int(args.warmups)):
            warm_fascicle = build_nrv_fascicle(args, fiber_count=fiber_count)
            warm = run_nrv(warm_fascicle, args, tsim_ms=tsim_ms)
            materialize_nrv_summary(warm)
        for _ in range(max(0, int(args.repeats) - 1)):
            repeat_fascicle = build_nrv_fascicle(args, fiber_count=fiber_count)
            sample = timed(lambda fasc=repeat_fascicle: run_nrv(fasc, args, tsim_ms=tsim_ms))
            mat = timed(lambda result=sample.value: materialize_nrv_summary(result))
            nrv_warm_samples.append(float(sample.elapsed_s))
            nrv_total_samples.append(float(sample.elapsed_s + mat.elapsed_s))

    row = {
        "population": args.population,
        "fiber_count_requested": int(fiber_count),
        "fiber_count_simulated": int(len(rows)),
        "mrg_count": counts["mrg"],
        "rattay_count": counts["rattay"],
        "tsim_ms": float(tsim_ms),
        "dt_ms": float(args.dt_ms),
        "nt": int(nt),
        "mode": args.mode,
        "geometry_source": args.geometry_source,
        "as_device": args.device,
        "as_gpu_index": int(args.gpu_index) if args.device == "gpu" else None,
        "jax_backend": current_jax_backend(),
        "nrv_processes": int(args.nrv_processes),
        "time_chunk_steps": effective_axonscope_time_chunk_steps(args),
        "nrv_build_s": _elapsed(nrv_build),
        "nrv_first_s": _elapsed(nrv_first),
        "nrv_materialize_first_s": _elapsed(nrv_materialize),
        "nrv_total_first_s": _sum_elapsed(nrv_first, nrv_materialize),
        "nrv_warm_median_s": median_or_none(nrv_warm_samples),
        "nrv_total_warm_median_s": median_or_none(nrv_total_samples),
        "nrv_activated": nrv_activated,
        "as_build_s": _elapsed(as_build),
        "as_first_s": _elapsed(as_first),
        "as_materialize_first_s": _elapsed(as_materialize),
        "as_total_first_s": _sum_elapsed(as_first, as_materialize),
        "as_warm_median_s": median_or_none(as_warm_samples),
        "as_total_warm_median_s": median_or_none(as_total_samples),
        "as_activated": as_activated,
        "as_observer_bytes": as_observer_bytes,
        "as_dense_vm_bytes": as_dense_bytes,
        "as_dense_vm_estimate_bytes": estimate_dense_vm_bytes(rows, nt=nt),
        "as_group_count": as_route_summary.get("group_count"),
        "as_methods": as_route_summary.get("methods"),
        "as_batch_kinds": as_route_summary.get("batch_kinds"),
        "as_padded_groups": as_route_summary.get("padded_groups"),
        "rss_max_mb": maxrss_mb(),
    }
    row["as_cold_build_plus_total_first_s"] = _sum_float(
        row.get("as_build_s"),
        row.get("as_total_first_s"),
    )
    row["as_first_minus_warm_median_s"] = _diff_float(
        row.get("as_first_s"),
        row.get("as_warm_median_s"),
    )
    row["as_total_first_minus_total_warm_median_s"] = _diff_float(
        row.get("as_total_first_s"),
        row.get("as_total_warm_median_s"),
    )
    row["as_first_over_warm_median"] = ratio(row.get("as_first_s"), row.get("as_warm_median_s"))
    row.update(as_cold_report_metrics)
    row.update(as_warm_report_metrics)
    row["speedup_nrv_over_as_first"] = ratio(row["nrv_first_s"], row["as_first_s"])
    row["speedup_nrv_over_as_total_first"] = ratio(row["nrv_total_first_s"], row["as_total_first_s"])
    row["speedup_nrv_total_first_over_as_total_warm"] = ratio(
        row["nrv_total_first_s"],
        row["as_total_warm_median_s"],
    )
    row["speedup_nrv_over_as_warm"] = ratio(row["nrv_warm_median_s"], row["as_warm_median_s"])
    row["speedup_nrv_over_as_total_warm"] = ratio(
        row["nrv_total_warm_median_s"],
        row["as_total_warm_median_s"],
    )
    print_case_row(row)
    return row


def build_nrv_fascicle(args: argparse.Namespace, *, fiber_count: int):
    import nrv

    np.random.seed(int(args.seed) + 10_000 * int(fiber_count))
    fascicle = nrv.fascicle(diameter=float(args.fascicle_diameter_um), ID=0)
    fascicle.define_length(nrv_numeric(args.length_um))
    percent_unmyelinated = {
        "mrg": 0.0,
        "rattay": 1.0,
        "mixed": float(args.percent_unmyelinated),
    }[args.population]
    fascicle.fill(
        n_ax=int(fiber_count),
        percent_unmyel=percent_unmyelinated,
        delta_trace=10.0,
        with_node_shift=True,
        overwrite=True,
    )
    if int(args.nrv_processes) > 0:
        fascicle.n_proc = int(args.nrv_processes)

    electrode = nrv.point_source_electrode(
        float(args.length_um) / 2.0,
        float(args.electrode_y_um),
        float(args.electrode_z_um),
    )
    stimulus = nrv.stimulus()
    stimulus.pulse(
        float(args.stimulus_start_ms),
        -float(args.stimulus_current_uA),
        float(args.pulse_duration_ms),
    )
    extra = nrv.stimulation("endoneurium_bhadra")
    extra.add_electrode(electrode, stimulus)
    fascicle.attach_extracellular_stimulation(extra)
    return fascicle


def run_nrv(fascicle: Any, args: argparse.Namespace, *, tsim_ms: float) -> Any:
    return fascicle.simulate(
        pbar_off=not bool(args.progress),
        t_sim=float(tsim_ms),
        dt=float(args.dt_ms),
        postproc_script="is_recruited",
        save_V_mem=False,
        save_results=False,
        return_parameters_only=False,
        myelinated_nseg_per_sec=1,
        unmyelinated_nseg=max(3, int(float(args.length_um) // 25)),
    )


def extract_rows(fascicle: Any, *, population: PopulationKind) -> list[axs_nrv.NRVFiberRow]:
    rows: list[axs_nrv.NRVFiberRow] = []
    table = fascicle.axons.axon_pop
    for fiber_index, row in table.iterrows():
        nrv_type = int(float(row.get("types", 0)))
        kind = axs_nrv.fiber_kind_from_nrv(nrv_type, include_mrg=True)
        if population == "mrg" and kind != "mrg":
            continue
        if population == "rattay" and kind != "rattay":
            continue
        diameter_um = float(row.get("diameters", 1.0))
        node_shift = float(row.get("node_shift", 0.0))
        rows.append(
            axs_nrv.NRVFiberRow(
                fascicle_id="0",
                fiber_index=int(fiber_index),
                kind=kind,
                diameter_um=diameter_um,
                y_um=float(row.get("y", 0.0)),
                z_um=float(row.get("z", 0.0)),
                node_shift=node_shift,
                x_shift_um=axs_nrv.nrv_node_shift_to_x_shift_um(
                    node_shift,
                    diameter_um,
                    kind=kind,
                ),
            )
        )
    rows.sort(key=lambda item: item.fiber_index)
    return rows


def build_synthetic_rows(
    args: argparse.Namespace,
    *,
    fiber_count: int,
) -> list[axs_nrv.NRVFiberRow]:
    """Build deterministic NRV-shaped rows without importing NRV."""

    rng = np.random.default_rng(int(args.seed) + 10_000 * int(fiber_count))
    kinds = synthetic_kinds(
        population=args.population,
        fiber_count=int(fiber_count),
        percent_unmyelinated=float(args.percent_unmyelinated),
    )
    rng.shuffle(kinds)
    radius_um = float(args.fascicle_diameter_um) / 2.0
    rows: list[axs_nrv.NRVFiberRow] = []
    for fiber_index, kind in enumerate(kinds):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = radius_um * np.sqrt(rng.uniform(0.0, 1.0))
        diameter_um = (
            float(rng.uniform(2.5, 14.0))
            if kind == "mrg"
            else float(rng.uniform(0.2, 1.2))
        )
        node_shift = float(rng.uniform(0.0, 1.0)) if kind == "mrg" else 0.0
        rows.append(
            axs_nrv.NRVFiberRow(
                fascicle_id="0",
                fiber_index=int(fiber_index),
                kind=kind,
                diameter_um=diameter_um,
                y_um=float(radius * np.cos(angle)),
                z_um=float(radius * np.sin(angle)),
                node_shift=node_shift,
                x_shift_um=axs_nrv.nrv_node_shift_to_x_shift_um(
                    node_shift,
                    diameter_um,
                    kind=kind,
                ),
            )
        )
    rows.sort(key=lambda item: item.fiber_index)
    return rows


def synthetic_kinds(
    *,
    population: PopulationKind,
    fiber_count: int,
    percent_unmyelinated: float,
) -> list[axs_nrv.FiberKind]:
    if population == "mrg":
        return ["mrg"] * int(fiber_count)
    if population == "rattay":
        return ["rattay"] * int(fiber_count)
    rattay_count = int(round(int(fiber_count) * float(percent_unmyelinated)))
    rattay_count = min(max(rattay_count, 0), int(fiber_count))
    mrg_count = int(fiber_count) - rattay_count
    return ["mrg"] * mrg_count + ["rattay"] * rattay_count


def build_axonscope_pool(
    args: argparse.Namespace,
    rows: Sequence[axs_nrv.NRVFiberRow],
) -> list[axs.AxonInstance]:
    stimulus = axs.Stimulus.pulse(
        start=float(args.stimulus_start_ms) * axs.ms,
        duration=float(args.pulse_duration_ms) * axs.ms,
        amplitude=-float(args.stimulus_current_uA) * axs.uA,
    )
    electrode = axs.analytical.PointSourceElectrode(
        x=(float(args.length_um) / 2.0) * axs.um,
        y=float(args.electrode_y_um) * axs.um,
        z=float(args.electrode_z_um) * axs.um,
    )
    pool = []
    for row in rows:
        axon = build_axonscope_axon(row, length_um=float(args.length_um))
        positions_um = axon.layout.position_values(unit=axs.um)
        simulation = axs.AxonInstance(axon)
        simulation.add_extracellular_stimulation(
            stimulation=axs.analytical.point_source_stimulation(
                electrode,
                positions_um * axs.um,
                stimulus=stimulus,
                sigma=float(args.sigma_S_m) * axs.S_per_m,
                axon_y=float(row.y_um) * axs.um,
                axon_z=float(row.z_um) * axs.um,
            ),
            replace=True,
        )
        pool.append(simulation)
    return pool


def build_axonscope_axon(row: axs_nrv.NRVFiberRow, *, length_um: float):
    diameter = max(float(row.diameter_um), 0.2) * axs.um
    if row.kind == "mrg":
        nodes = max(
            2,
            axs.axons.mrg_like_nodes_from_length(
                diameter,
                length_um * axs.um,
                x_shift=float(row.x_shift_um) * axs.um,
            ),
        )
        return axs.axons.MRG(
            diameter=diameter,
            nodes=nodes,
            length=length_um * axs.um,
            x_shift=float(row.x_shift_um) * axs.um,
        )
    return axs.axons.RattayAberham(
        length=length_um * axs.um,
        diameter=diameter,
        compartments=max(3, int(length_um // 25)),
        celsius=37.0 * axs.degC,
    )


def run_axonscope(
    args: argparse.Namespace,
    pool: Sequence[axs.AxonInstance],
    activation: Any,
    *,
    tsim_ms: float,
) -> axs.AxonSimulationResult:
    if args.mode == "observer":
        recording = axs.Recording.none()
        if int(args.time_chunk_steps) == 0:
            batch_options = BatchOptions.none()
        else:
            time_chunk_steps = (
                None
                if int(args.time_chunk_steps) < 0
                else int(args.time_chunk_steps)
            )
            batch_options = BatchOptions.none(time_chunk_steps=time_chunk_steps)
    else:
        recording = axs.Recording(signals=axs.signals.Vm)
        time_chunk_steps = None if int(args.time_chunk_steps) <= 0 else int(args.time_chunk_steps)
        batch_options = BatchOptions.full(time_chunk_steps=time_chunk_steps)
    return axs.simulate_pool(
        pool,
        duration=float(tsim_ms) * axs.ms,
        dt=float(args.dt_ms) * axs.ms,
        recording=recording,
        observers=[activation],
        batch_options=batch_options,
        execution_policy=axonscope_execution_policy(args),
        progress=bool(args.progress),
    )


def axonscope_execution_policy(args: argparse.Namespace) -> axs.ExecutionPolicy | None:
    if args.device == "auto":
        return None
    device = axs.Device.cpu() if args.device == "cpu" else axs.Device.gpu(int(args.gpu_index))
    return axs.ExecutionPolicy(
        runtime=axs.Runtime.JAX,
        device=device,
        precision=axs.PrecisionPolicy.float32(),
    )


def current_jax_backend() -> str | None:
    try:
        import jax
    except Exception:
        return None
    try:
        return str(jax.default_backend())
    except Exception:
        return None


def effective_axonscope_time_chunk_steps(args: argparse.Namespace) -> int | None:
    if args.mode == "observer" and int(args.time_chunk_steps) == 0:
        return int(axs.DEFAULT_OBSERVER_TIME_CHUNK_STEPS)
    if int(args.time_chunk_steps) <= 0:
        return None
    return int(args.time_chunk_steps)


def run_and_block_axonscope(
    args: argparse.Namespace,
    pool: Sequence[axs.AxonInstance],
    activation: Any,
    *,
    tsim_ms: float,
) -> axs.AxonSimulationResult:
    result = run_axonscope(args, pool, activation, tsim_ms=tsim_ms)
    block_axonscope_result(result)
    return result


def timed_axonscope_run(
    args: argparse.Namespace,
    pool: Sequence[axs.AxonInstance],
    activation: Any,
    *,
    tsim_ms: float,
    profile_dir: Path | None = None,
) -> tuple[TimedValue, Any | None]:
    report = None
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
        sample = timed(lambda: run_and_block_axonscope(args, pool, activation, tsim_ms=tsim_ms))
    finally:
        if profile_dir is not None:
            report = axs.disable_benchmark(print_summary=False, save=True)
    return sample, report


def materialize_axonscope_summary(
    result: axs.AxonSimulationResult,
    activation: Any,
    *,
    mode: AxonScopeMode,
) -> dict[str, Any]:
    if mode == "observer":
        observations = result.observations
        if observations is None:
            raise RuntimeError("AxonScope observer run produced no observations.")
        if activation.name in observations:
            values = np.asarray(observations[activation.name].values, dtype=bool)
            return {"activated": int(np.count_nonzero(values)), "observer_bytes": int(values.nbytes)}
        raster = observations[axs.VM_RASTER_OBSERVATION_KEY]
        bits = np.asarray(raster.unpack(), dtype=bool)
        names = tuple(getattr(raster, "names", ()))
        raster_index = names.index(activation.name)
        mask = np.asarray(getattr(raster, "probe_mask", True), dtype=bool)
        if mask.ndim == 2:
            mask = np.broadcast_to(mask[None, :, :], bits.shape[:3])
        row_bits = bits[:, raster_index]
        row_mask = mask[:, raster_index]
        activated = np.any(row_bits & row_mask[:, :, None], axis=(1, 2))
        return {
            "activated": int(np.count_nonzero(activated)),
            "observer_bytes": int(np.asarray(raster.words).nbytes),
        }

    activated_values = np.asarray(result.analyze(activation).values, dtype=bool)
    dense_bytes = 0
    for view in result:
        dense_bytes += int(np.asarray(view.Vm).nbytes)
    return {
        "activated": int(np.count_nonzero(activated_values)),
        "dense_vm_bytes": int(dense_bytes),
    }


def materialize_nrv_summary(result: Any) -> dict[str, Any]:
    recruited = []
    keys = result.keys() if hasattr(result, "keys") else ()
    for key in keys:
        if not str(key).startswith("axon"):
            continue
        if not str(key)[4:].isdigit():
            continue
        axon_result = result[key]
        try:
            recruited.append(bool(axon_result["recruited"]))
        except Exception:
            continue
    if recruited:
        return {"activated": int(np.count_nonzero(recruited))}
    try:
        return {"activated": int(result.get_recruited_axons())}
    except Exception:
        return {"activated": None}


def block_axonscope_result(result: axs.AxonSimulationResult) -> None:
    observations = result.observations
    if observations:
        for value in observations.values():
            words = getattr(value, "words", None)
            if hasattr(words, "block_until_ready"):
                words.block_until_ready()
    for view in result:
        for attr in ("Vm", "t"):
            try:
                value = getattr(view, attr)
            except Exception:
                continue
            if hasattr(value, "block_until_ready"):
                value.block_until_ready()


def axonscope_profile_dir(
    args: argparse.Namespace,
    fiber_count: int,
    tsim_ms: float,
    phase: str,
) -> Path:
    prefix = str(getattr(args, "run_prefix", args.prefix or "population_tsim"))
    tsim_label = f"{float(tsim_ms):g}".replace(".", "p")
    return (
        args.report_dir
        / "population_tsim_cold_path"
        / prefix
        / f"n{int(fiber_count):03d}_tsim{tsim_label}ms_{phase}"
    )


def benchmark_report_metrics(report: Any | None, *, prefix: str) -> dict[str, Any]:
    if report is None:
        return {}
    metrics: dict[str, Any] = {
        f"{prefix}_profile_event_count": int(len(getattr(report, "events", ()))),
    }
    summary_by_name = {
        str(row.name): float(row.total_ms)
        for row in getattr(report, "summary", ())
    }
    for event_name, column_suffix in COLD_PATH_EVENT_COLUMNS.items():
        value = summary_by_name.get(event_name)
        if value is not None:
            metrics[f"{prefix}_{column_suffix}"] = value

    cache_hits = 0
    cache_misses = 0
    for event in getattr(report, "events", ()):
        metadata = getattr(event, "metadata", {})
        for key, value in dict(metadata).items():
            if not str(key).endswith("_cache"):
                continue
            if value == "hit":
                cache_hits += 1
            elif value == "miss":
                cache_misses += 1
    metrics[f"{prefix}_cache_hits"] = int(cache_hits)
    metrics[f"{prefix}_cache_misses"] = int(cache_misses)
    metadata = getattr(report, "metadata", {})
    output_dir = metadata.get("output_dir") if isinstance(metadata, dict) else None
    if output_dir is not None:
        metrics[f"{prefix}_profile_dir"] = str(output_dir)
    return metrics


def clear_jax_caches() -> None:
    try:
        import jax
    except Exception:
        return
    clear = getattr(jax, "clear_caches", None)
    if callable(clear):
        clear()


def summarize_axonscope_routes(result: axs.AxonSimulationResult) -> dict[str, Any]:
    diagnostics = result.diagnostics
    if not diagnostics:
        return {}
    groups = {}
    for item in diagnostics:
        group_id = item.get("dispatch_group_id")
        groups[group_id] = item
    return {
        "group_count": len(groups),
        "methods": ",".join(sorted({str(item.get("dispatch_method")) for item in groups.values()})),
        "batch_kinds": ",".join(sorted({str(item.get("dispatch_batch_kind")) for item in groups.values()})),
        "padded_groups": int(sum(bool(item.get("dispatch_has_padding")) for item in groups.values())),
    }


def fiber_counts(rows: Sequence[axs_nrv.NRVFiberRow]) -> dict[str, int]:
    return {
        "mrg": int(sum(row.kind == "mrg" for row in rows)),
        "rattay": int(sum(row.kind == "rattay" for row in rows)),
    }


def estimate_dense_vm_bytes(rows: Sequence[axs_nrv.NRVFiberRow], *, nt: int) -> int:
    bytes_per_sample = 4
    total_nx = 0
    for row in rows:
        if row.kind == "mrg":
            total_nx += max(2, int(round(row.diameter_um * 50.0)))
        else:
            total_nx += 400
    return int(total_nx * int(nt) * bytes_per_sample)


def timed(func) -> TimedValue:
    start = time.perf_counter()
    value = func()
    return TimedValue(elapsed_s=time.perf_counter() - start, value=value)


def write_results(rows: Sequence[dict[str, Any]], out_dir: Path, prefix: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, indent=2, sort_keys=True)
    columns = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def write_plot(rows: Sequence[dict[str, Any]], report_dir: Path, prefix: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{prefix}_timing_vs_tsim.png"
    fig, (time_ax, speed_ax) = plt.subplots(
        2,
        1,
        figsize=(9.5, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    cmap = plt.get_cmap("tab10")
    fiber_counts_sorted = sorted({int(row["fiber_count_simulated"]) for row in rows})
    for index, fiber_count in enumerate(fiber_counts_sorted):
        color = cmap(index % 10)
        subset = sorted(
            [row for row in rows if int(row["fiber_count_simulated"]) == fiber_count],
            key=lambda item: float(item["tsim_ms"]),
        )
        x = np.asarray([float(row["tsim_ms"]) for row in subset], dtype=float)
        as_first_y = np.asarray([nan_if_none(row.get("as_total_first_s")) for row in subset], dtype=float)
        as_warm_y = np.asarray(
            [nan_if_none(row.get("as_total_warm_median_s")) for row in subset],
            dtype=float,
        )
        nrv_y = np.asarray([nan_if_none(row.get("nrv_total_first_s")) for row in subset], dtype=float)
        speed_first = np.asarray(
            [nan_if_none(row.get("speedup_nrv_over_as_total_first")) for row in subset],
            dtype=float,
        )
        speed_warm = np.asarray(
            [nan_if_none(row.get("speedup_nrv_total_first_over_as_total_warm")) for row in subset],
            dtype=float,
        )
        time_ax.plot(
            x,
            as_first_y,
            marker="o",
            color=color,
            linewidth=2.0,
            label=f"AxonScope first n={fiber_count}",
        )
        if has_finite(as_warm_y):
            time_ax.plot(
                x,
                as_warm_y,
                marker="o",
                color=color,
                linestyle=":",
                linewidth=2.0,
                label=f"AxonScope warm n={fiber_count}",
            )
        time_ax.plot(
            x,
            nrv_y,
            marker="s",
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"NRV n={fiber_count}",
        )
        speed_ax.plot(
            x,
            speed_first,
            marker="o",
            color=color,
            linewidth=2.0,
            label=f"NRV/AS first n={fiber_count}",
        )
        if has_finite(speed_warm):
            speed_ax.plot(
                x,
                speed_warm,
                marker="^",
                color=color,
                linestyle=":",
                linewidth=2.0,
                label=f"NRV/AS warm n={fiber_count}",
            )
    time_ax.set_yscale("log")
    time_ax.set_ylabel("total runtime [s]")
    time_ax.set_title("AxonScope cold/warm vs NRV population point-source scaling")
    time_ax.grid(axis="y", alpha=0.25)
    time_ax.legend(fontsize=8)
    speed_ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.55)
    speed_ax.set_xlabel("tsim [ms]")
    speed_ax.set_ylabel("NRV / AxonScope")
    speed_ax.grid(axis="y", alpha=0.25)
    speed_ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def print_summary(rows: Sequence[dict[str, Any]]) -> None:
    print("\n=== Population tsim scaling ===")
    for row in rows:
        as_total = fmt_optional(row.get("as_total_first_s"))
        as_warm = fmt_optional(row.get("as_total_warm_median_s"))
        as_overhead = fmt_optional(row.get("as_total_first_minus_total_warm_median_s"))
        nrv_total = fmt_optional(row.get("nrv_total_first_s"))
        speed_first = fmt_optional(row.get("speedup_nrv_over_as_total_first"))
        speed_warm = fmt_optional(row.get("speedup_nrv_total_first_over_as_total_warm"))
        print(
            f"n={row['fiber_count_simulated']:3d} "
            f"mrg={row['mrg_count']:3d} rattay={row['rattay_count']:3d} "
            f"tsim={row['tsim_ms']:g} ms nt={row['nt']:5d} "
            f"AS_first={as_total}s AS_warm={as_warm}s "
            f"AS_cold-warm={as_overhead}s NRV_total={nrv_total}s "
            f"NRV/AS_first={speed_first} NRV/AS_warm={speed_warm} "
            f"groups={row.get('as_group_count')} padded={row.get('as_padded_groups')}"
        )


def print_case_row(row: dict[str, Any]) -> None:
    print(
        f"AS_first={fmt_optional(row.get('as_total_first_s'))}s "
        f"AS_warm={fmt_optional(row.get('as_total_warm_median_s'))}s "
        f"cold-warm={fmt_optional(row.get('as_total_first_minus_total_warm_median_s'))}s "
        f"NRV={fmt_optional(row.get('nrv_total_first_s'))}s "
        f"NRV/AS_first={fmt_optional(row.get('speedup_nrv_over_as_total_first'))} "
        f"NRV/AS_warm={fmt_optional(row.get('speedup_nrv_total_first_over_as_total_warm'))} "
        f"activated AS/NRV={row.get('as_activated')}/{row.get('nrv_activated')} "
        f"routes={row.get('as_methods')}"
    )


def nrv_numeric(value: float) -> int | float:
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator = float(denominator)
    if denominator == 0.0:
        return None
    return float(numerator) / denominator


def median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _elapsed(value: TimedValue | None) -> float | None:
    return None if value is None else float(value.elapsed_s)


def _sum_elapsed(left: TimedValue | None, right: TimedValue | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left.elapsed_s + right.elapsed_s)


def _sum_float(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) + float(right)


def _diff_float(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def maxrss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def fmt_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        if not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def nan_if_none(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def has_finite(values: np.ndarray) -> bool:
    return bool(np.any(np.isfinite(values)))


if __name__ == "__main__":
    main()
