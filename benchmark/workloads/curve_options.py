from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CurvePreset:
    """Shared scale defaults for the two canonical curve benchmarks."""

    tsim: float
    dt: float
    nx: int
    n_axons: int
    precision: str
    recording: str
    platform: str
    repeats: int
    warmups: int
    memory_trace: str
    memory_top_n: int
    profile: bool
    profile_backend: str
    profile_create_perfetto: bool
    jax_device_memory_profile: bool
    max_iterations: int
    amplitude_count: int


PRESETS: dict[str, CurvePreset] = {
    "quick": CurvePreset(
        tsim=2.0,
        dt=0.02,
        nx=21,
        n_axons=1,
        precision="fp32",
        recording="observer_only",
        platform="cpu",
        repeats=1,
        warmups=0,
        memory_trace="rss",
        memory_top_n=0,
        profile=False,
        profile_backend="auto",
        profile_create_perfetto=False,
        jax_device_memory_profile=False,
        max_iterations=2,
        amplitude_count=3,
    ),
    "local_smoke": CurvePreset(
        tsim=5.0,
        dt=0.01,
        nx=51,
        n_axons=8,
        precision="fp32",
        recording="probe_vm",
        platform="cpu",
        repeats=2,
        warmups=1,
        memory_trace="rss",
        memory_top_n=0,
        profile=False,
        profile_backend="auto",
        profile_create_perfetto=False,
        jax_device_memory_profile=False,
        max_iterations=6,
        amplitude_count=6,
    ),
    "local_realistic": CurvePreset(
        tsim=20.0,
        dt=0.005,
        nx=101,
        n_axons=64,
        precision="fp32",
        recording="probe_vm",
        platform="cpu",
        repeats=3,
        warmups=1,
        memory_trace="rss",
        memory_top_n=0,
        profile=False,
        profile_backend="auto",
        profile_create_perfetto=False,
        jax_device_memory_profile=False,
        max_iterations=10,
        amplitude_count=12,
    ),
    "cpu_publication": CurvePreset(
        tsim=50.0,
        dt=0.005,
        nx=201,
        n_axons=256,
        precision="fp64",
        recording="probe_vm",
        platform="cpu",
        repeats=5,
        warmups=2,
        memory_trace="rss",
        memory_top_n=0,
        profile=False,
        profile_backend="auto",
        profile_create_perfetto=False,
        jax_device_memory_profile=False,
        max_iterations=16,
        amplitude_count=24,
    ),
    "gpu_smoke": CurvePreset(
        tsim=2.0,
        dt=0.02,
        nx=31,
        n_axons=16,
        precision="fp32",
        recording="observer_only",
        platform="gpu",
        repeats=1,
        warmups=0,
        memory_trace="rss",
        memory_top_n=0,
        profile=False,
        profile_backend="auto",
        profile_create_perfetto=False,
        jax_device_memory_profile=False,
        max_iterations=2,
        amplitude_count=3,
    ),
    "gpu_trace_smoke": CurvePreset(
        tsim=1.0,
        dt=0.05,
        nx=21,
        n_axons=4,
        precision="fp32",
        recording="observer_only",
        platform="gpu",
        repeats=1,
        warmups=0,
        memory_trace="all",
        memory_top_n=5,
        profile=True,
        profile_backend="jax",
        profile_create_perfetto=True,
        jax_device_memory_profile=True,
        max_iterations=1,
        amplitude_count=3,
    ),
    "gpu_realistic": CurvePreset(
        tsim=50.0,
        dt=0.005,
        nx=201,
        n_axons=1000,
        precision="fp32",
        recording="observer_only",
        platform="gpu",
        repeats=5,
        warmups=2,
        memory_trace="rss",
        memory_top_n=0,
        profile=False,
        profile_backend="auto",
        profile_create_perfetto=False,
        jax_device_memory_profile=False,
        max_iterations=16,
        amplitude_count=24,
    ),
    "nrv_smoke": CurvePreset(
        tsim=2.0,
        dt=0.02,
        nx=21,
        n_axons=1,
        precision="fp32",
        recording="probe_vm",
        platform="nrv",
        repeats=1,
        warmups=0,
        memory_trace="rss",
        memory_top_n=0,
        profile=False,
        profile_backend="auto",
        profile_create_perfetto=False,
        jax_device_memory_profile=False,
        max_iterations=2,
        amplitude_count=3,
    ),
    "nrv_full": CurvePreset(
        tsim=20.0,
        dt=0.005,
        nx=101,
        n_axons=32,
        precision="fp64",
        recording="probe_vm",
        platform="nrv",
        repeats=3,
        warmups=1,
        memory_trace="rss",
        memory_top_n=5,
        profile=False,
        profile_backend="auto",
        profile_create_perfetto=False,
        jax_device_memory_profile=False,
        max_iterations=10,
        amplitude_count=12,
    ),
}


def build_parser(script_name: str, *, description: str) -> argparse.ArgumentParser:
    preset = PRESETS["quick"]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="quick")
    parser.add_argument("--source", choices=("point_source_axonscope", "nrv_nerve"), default="point_source_axonscope")
    parser.add_argument("--tsim", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--n-axons", type=int)
    parser.add_argument("--precision", choices=("fp32", "fp64"))
    parser.add_argument("--recording", choices=("full_vm", "probe_vm", "observer_only"))
    parser.add_argument("--cable", choices=("single_cable", "double_cable"), default="single_cable")
    parser.add_argument(
        "--double-cable-block-solver",
        choices=("auto", "thomas", "pcr", "pcr_soa", "pcr_adaptive"),
        default="auto",
        help="Benchmark-only override for the double-cable block solver.",
    )
    parser.add_argument(
        "--benchmark-double-cable-block-solver",
        choices=("jax_triton_loop_xb",),
        default=None,
        help=(
            "Private benchmark-only override that can activate experimental "
            "backend solver routes without adding public BatchOptions values."
        ),
    )
    parser.add_argument("--population", choices=("single_model", "mixed_models"), default="single_model")
    parser.add_argument(
        "--diameters",
        choices=("same_diameter", "different_diameters"),
        default="same_diameter",
    )
    parser.add_argument("--platform", choices=("cpu", "gpu", "nrv"))
    parser.add_argument("--execution-policy", default="default")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--memory-trace", choices=("off", "rss", "tracemalloc", "device", "all"))
    parser.add_argument("--memory-top-n", type=int)
    parser.add_argument("--profile", action="store_true", dest="profile", default=None)
    parser.add_argument("--no-profile", action="store_false", dest="profile")
    parser.add_argument("--profile-backend", choices=("auto", "jax", "none"))
    parser.add_argument("--profile-output")
    parser.add_argument(
        "--profile-create-perfetto",
        action="store_true",
        dest="profile_create_perfetto",
        default=None,
    )
    parser.add_argument(
        "--no-profile-create-perfetto",
        action="store_false",
        dest="profile_create_perfetto",
    )
    parser.add_argument(
        "--jax-device-memory-profile",
        action="store_true",
        dest="jax_device_memory_profile",
        default=None,
    )
    parser.add_argument(
        "--no-jax-device-memory-profile",
        action="store_false",
        dest="jax_device_memory_profile",
    )
    parser.add_argument(
        "--jax-device-memory-profile-stage",
        action="append",
        dest="jax_device_memory_profile_stages",
        help=(
            "Stage name for JAX device-memory profile capture. Repeat to select "
            "multiple stages. Defaults to kernel.wait; use 'all' to capture every span."
        ),
    )
    parser.add_argument("--output", default=f"benchmark/results/{script_name}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case-filter")
    parser.add_argument("--spatial-recording", choices=("center", "probes", "indices"), default="probes")
    parser.add_argument("--observer-criterion", default="vm_raster")
    parser.add_argument("--amplitude-min", type=float, default=0.0)
    parser.add_argument("--amplitude-max", type=float, default=1.0)
    parser.add_argument("--amplitude-tolerance", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--stimulation", choices=("monophasic", "biphasic", "custom"), default="biphasic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-mode", choices=("cold", "warm", "clear_codegen_cache"), default="warm")
    parser.add_argument(
        "--time-chunk-steps",
        type=_parse_time_chunk_steps_arg,
        metavar="{default,unchunked,N}",
        help=(
            "Observer/kernel time chunk policy. Omit or use 'default' to let "
            "AxonScope choose its recording-specific default, use 'unchunked' "
            "or 'none' to force one full scan, or pass an integer chunk size."
        ),
    )
    parser.add_argument("--amplitude-batch-size", type=int)
    parser.add_argument("--retention", choices=("summary_only", "raw_traces", "debug_artifacts"), default="summary_only")
    parser.set_defaults(
        _preset_defaults={
            "tsim": preset.tsim,
            "dt": preset.dt,
            "nx": preset.nx,
            "n_axons": preset.n_axons,
            "precision": preset.precision,
            "recording": preset.recording,
            "platform": preset.platform,
            "repeats": preset.repeats,
            "warmups": preset.warmups,
            "memory_trace": preset.memory_trace,
            "memory_top_n": preset.memory_top_n,
            "profile": preset.profile,
            "profile_backend": preset.profile_backend,
            "profile_create_perfetto": preset.profile_create_perfetto,
            "jax_device_memory_profile": preset.jax_device_memory_profile,
            "max_iterations": preset.max_iterations,
            "amplitude_count": preset.amplitude_count,
        }
    )
    return parser


def resolved_options(args: argparse.Namespace) -> dict[str, Any]:
    preset = PRESETS[args.preset]
    options = {
        "preset": args.preset,
        "source": args.source,
        "tsim": args.tsim if args.tsim is not None else preset.tsim,
        "dt": args.dt if args.dt is not None else preset.dt,
        "nx": args.nx if args.nx is not None else preset.nx,
        "n_axons": args.n_axons if args.n_axons is not None else preset.n_axons,
        "precision": args.precision or preset.precision,
        "recording": args.recording or preset.recording,
        "cable": args.cable,
        "double_cable_block_solver": args.double_cable_block_solver,
        "benchmark_double_cable_block_solver": args.benchmark_double_cable_block_solver,
        "population": args.population,
        "diameters": args.diameters,
        "platform": args.platform or preset.platform,
        "execution_policy": args.execution_policy,
        "repeats": args.repeats if args.repeats is not None else preset.repeats,
        "warmups": args.warmups if args.warmups is not None else preset.warmups,
        "memory_trace": args.memory_trace or preset.memory_trace,
        "memory_top_n": args.memory_top_n if args.memory_top_n is not None else preset.memory_top_n,
        "profile": preset.profile if args.profile is None else bool(args.profile),
        "profile_backend": args.profile_backend or preset.profile_backend,
        "profile_output": args.profile_output,
        "profile_create_perfetto": (
            preset.profile_create_perfetto
            if args.profile_create_perfetto is None
            else bool(args.profile_create_perfetto)
        ),
        "jax_device_memory_profile": (
            preset.jax_device_memory_profile
            if args.jax_device_memory_profile is None
            else bool(args.jax_device_memory_profile)
        ),
        "jax_device_memory_profile_stages": _profile_stages(args.jax_device_memory_profile_stages),
        "output": args.output,
        "resume": bool(args.resume),
        "case_filter": args.case_filter,
        "spatial_recording": args.spatial_recording,
        "observer_criterion": args.observer_criterion,
        "amplitude_min": args.amplitude_min,
        "amplitude_max": args.amplitude_max,
        "amplitude_tolerance": args.amplitude_tolerance,
        "max_iterations": (
            args.max_iterations
            if args.max_iterations is not None
            else preset.max_iterations
        ),
        "stimulation": args.stimulation,
        "seed": args.seed,
        "cache_mode": args.cache_mode,
        "time_chunk_policy": _time_chunk_policy(args.time_chunk_steps),
        "time_chunk_steps": _time_chunk_steps(args.time_chunk_steps),
        "amplitude_batch_size": args.amplitude_batch_size,
        "retention": args.retention,
    }
    if hasattr(args, "threshold_kind"):
        options["threshold_kind"] = args.threshold_kind
    if hasattr(args, "amplitude_count"):
        options["amplitude_count"] = (
            args.amplitude_count
            if args.amplitude_count is not None
            else preset.amplitude_count
        )
    return options


def _profile_stages(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ("kernel.wait",)
    stages = tuple(str(value) for value in values)
    if any(stage.lower() == "all" for stage in stages):
        return ()
    return stages


def _parse_time_chunk_steps_arg(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"", "default", "none", "off", "unchunked", "full"}:
        return text
    return str(_positive_int(text, option="--time-chunk-steps"))


def _time_chunk_policy(value: str | None) -> str:
    if value is None:
        return "default"
    text = str(value).strip().lower()
    if text in {"", "default"}:
        return "default"
    if text in {"none", "off", "unchunked", "full"}:
        return "unchunked"
    _positive_int(text, option="--time-chunk-steps")
    return "explicit"


def _time_chunk_steps(value: str | None) -> int | None:
    if _time_chunk_policy(value) != "explicit":
        return None
    return _positive_int(str(value).strip(), option="--time-chunk-steps")


def _positive_int(value: str, *, option: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option} must be 'default', 'unchunked', or a positive integer."
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"{option} must be >= 1.")
    return parsed


def dry_run(script_name: str, args: argparse.Namespace) -> int:
    options = resolved_options(args)
    selected_case_name = case_name(script_name, options)
    if options["case_filter"] and options["case_filter"] not in selected_case_name:
        print("No cases selected by --case-filter.")
        return 0
    output = Path(options["output"])
    output.mkdir(parents=True, exist_ok=True)
    write_cases_csv(output, script_name, options)
    print(f"dry-run: {selected_case_name}")
    print(f"wrote: {output / 'cases.csv'}")
    return 0


def case_row(script_name: str, options: dict[str, Any]) -> dict[str, Any]:
    return {"script": script_name, "case_name": case_name(script_name, options), **options}


def write_cases_csv(output: Path, script_name: str, options: dict[str, Any]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    row = case_row(script_name, options)
    path = output / "cases.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)
    return path


def unsupported_real_run(script_name: str) -> int:
    raise SystemExit(
        f"{script_name} real runs are not implemented yet; use --dry-run while P11A "
        "case lists are being validated."
    )


def case_name(script_name: str, options: dict[str, Any]) -> str:
    return (
        f"{script_name}__{options['source']}__{options['platform']}__"
        f"{options['precision']}__{options['recording']}__{options['cable']}__"
        f"{options['population']}__{options['diameters']}__"
        f"n{options['n_axons']}__nx{options['nx']}"
    )
