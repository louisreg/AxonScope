"""Benchmark runnable NRV integration examples as public workflow gates."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import axonscope as axs
from axonscope.benchmarking import benchmark_span


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "benchmark" / "results" / "with_nrv_examples"

INTERESTING_STAGES = (
    "example.run",
    "nrv_bridge.population_from_nrv",
    "nrv_bridge.footprints_from_nrv",
    "nrv_bridge.stimulated_population",
    "protocol.recruitment_sweep",
    "protocol.sweep.value",
    "simulation.setup",
    "simulation.run_pool",
    "dispatch.build_plan",
    "runtime.prepare",
    "kernel.enqueue",
    "kernel.dispatch_jax",
    "kernel.wait",
    "kernel.finalize_observer",
    "results.to_public",
)


@dataclass(frozen=True, slots=True)
class ExampleSpec:
    key: str
    path: Path
    label: str


EXAMPLES = {
    "01": ExampleSpec(
        key="01",
        path=REPO_ROOT / "examples" / "with_nrv" / "01_synthetic_fascicle_geometry.py",
        label="synthetic_fascicle_geometry",
    ),
    "02": ExampleSpec(
        key="02",
        path=REPO_ROOT / "examples" / "with_nrv" / "02_realistic_fascicle_geometry.py",
        label="realistic_fascicle_geometry",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="quick")
    parser.add_argument("--platform", choices=("cpu", "gpu", "nrv"), default="cpu")
    parser.add_argument("--examples", default="01,02")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--axons-per-fascicle", type=int, default=6)
    parser.add_argument("--amplitudes-uA", default="0,150,300")
    parser.add_argument("--duration-ms", type=float, default=0.5)
    parser.add_argument("--dt-ms", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory-trace", choices=("off", "rss", "tracemalloc", "device", "all"))
    parser.add_argument("--memory-top-n", type=int)
    parser.add_argument("--profile", action="store_true", dest="profile", default=None)
    parser.add_argument("--no-profile", action="store_false", dest="profile")
    parser.add_argument("--profile-runtime", choices=("auto", "jax", "none"))
    parser.add_argument("--profile-output")
    parser.add_argument("--profile-create-perfetto", action="store_true", default=None)
    parser.add_argument("--no-profile-create-perfetto", action="store_false", dest="profile_create_perfetto")
    parser.add_argument("--jax-device-memory-profile", action="store_true", default=None)
    parser.add_argument("--no-jax-device-memory-profile", action="store_false", dest="jax_device_memory_profile")
    parser.add_argument("--jax-device-memory-profile-stage", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = _parse_examples(args.examples)
    amplitudes = _parse_amplitudes(args.amplitudes_uA)
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1.")
    if args.warmups < 0:
        raise SystemExit("--warmups must be >= 0.")
    if args.axons_per_fascicle < 1:
        raise SystemExit("--axons-per-fascicle must be >= 1.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    _write_manifest(output, args, specs, amplitudes)

    if args.dry_run:
        _write_cases(output, args, specs, amplitudes)
        print(f"dry-run: with_nrv_examples -> {output}")
        return 0

    rows: list[dict[str, Any]] = []
    for spec in specs:
        module = _load_example(spec)
        phase_plan = [("cold", 0)]
        phase_plan.extend(("warmup", index) for index in range(args.warmups))
        phase_plan.extend(("warm", index) for index in range(args.repeats))
        for phase, repeat in phase_plan:
            row = _run_one(
                module,
                spec,
                phase=phase,
                repeat=repeat,
                args=args,
                amplitudes=amplitudes,
                output=output,
            )
            rows.append(row)
            _write_runs(output, rows)
            print(_format_progress(row))

    _write_report(output, rows)
    return 0


def _parse_examples(value: str) -> list[ExampleSpec]:
    selected: list[ExampleSpec] = []
    for raw in value.split(","):
        key = raw.strip()
        if not key:
            continue
        if key not in EXAMPLES:
            allowed = ", ".join(sorted(EXAMPLES))
            raise SystemExit(f"Unknown with_nrv example {key!r}; expected one of: {allowed}.")
        selected.append(EXAMPLES[key])
    if not selected:
        raise SystemExit("--examples selected no examples.")
    return selected


def _parse_amplitudes(value: str) -> tuple[float, ...]:
    amplitudes = tuple(float(raw.strip()) for raw in value.split(",") if raw.strip())
    if not amplitudes:
        raise SystemExit("--amplitudes-uA selected no amplitudes.")
    return amplitudes


def _load_example(spec: ExampleSpec):
    module_name = f"_axonscope_with_nrv_example_{spec.key}_{spec.label}"
    spec_obj = importlib.util.spec_from_file_location(module_name, spec.path)
    if spec_obj is None or spec_obj.loader is None:
        raise RuntimeError(f"Could not load {spec.path}.")
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    spec_obj.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise RuntimeError(f"{spec.path} does not define main().")
    return module


def _run_one(
    module: Any,
    spec: ExampleSpec,
    *,
    phase: str,
    repeat: int,
    args: argparse.Namespace,
    amplitudes: tuple[float, ...],
    output: Path,
) -> dict[str, Any]:
    run_dir = output / spec.key / f"{phase}_{repeat:02d}"
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    start = time.perf_counter_ns()
    failed = False
    error = ""
    try:
        with _plot_safe_context():
            with axs.benchmark(
                run_dir,
                print_summary=False,
                save=True,
                sync_device=True,
                record_shapes=True,
                memory_trace=args.memory_trace or "rss",
                memory_top_n=0 if args.memory_top_n is None else args.memory_top_n,
                profile=bool(args.profile) if args.profile is not None else False,
                profile_runtime=args.profile_runtime or "auto",
                profile_output=args.profile_output,
                profile_create_perfetto=bool(args.profile_create_perfetto)
                if args.profile_create_perfetto is not None
                else False,
                jax_device_memory_profile=bool(args.jax_device_memory_profile)
                if args.jax_device_memory_profile is not None
                else False,
                jax_device_memory_profile_stages=tuple(args.jax_device_memory_profile_stage),
            ):
                with benchmark_span("example.run", example=spec.key, phase=phase, repeat=repeat):
                    _ensure_nrv_imported()
                    with _maybe_quiet(args.quiet, stdout_buffer, stderr_buffer):
                        module.main(_example_config(module, args, amplitudes))
    except BaseException as exc:
        failed = True
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        end = time.perf_counter_ns()
        row = _row_from_run_dir(run_dir)
        row.update(
            {
                "example": spec.key,
                "label": spec.label,
                "phase": phase,
                "repeat": repeat,
                "wall_ms": (end - start) / 1_000_000.0,
                "failed": failed,
                "error": error,
                "stdout_chars": len(stdout_buffer.getvalue()),
                "stderr_chars": len(stderr_buffer.getvalue()),
            }
        )
        _write_text(run_dir / "stdout.txt", stdout_buffer.getvalue())
        _write_text(run_dir / "stderr.txt", stderr_buffer.getvalue())
    return row


def _example_config(
    module: Any,
    args: argparse.Namespace,
    amplitudes: tuple[float, ...],
) -> Any:
    return module.ExampleConfig(
        axons_per_fascicle=args.axons_per_fascicle,
        duration_ms=args.duration_ms,
        dt_ms=args.dt_ms,
        recruitment_amplitudes_uA=amplitudes,
        solver_progress=False,
        fem_n_proc=1,
        gmsh_n_core=1,
        execution_policy=_execution_policy_for_platform(args.platform),
    )


def _execution_policy_for_platform(platform: str) -> axs.ExecutionPolicy | None:
    if platform == "gpu":
        return axs.ExecutionPolicy(runtime=axs.runtime.jax, device=axs.Device.gpu())
    if platform == "cpu":
        return axs.ExecutionPolicy(runtime=axs.runtime.jax, device=axs.Device.cpu())
    return None


def _ensure_nrv_imported() -> None:
    # NRV enables faulthandler during import and expects stderr to expose a real
    # file descriptor, so import it before redirecting stdout/stderr to buffers.
    import nrv  # noqa: F401


def _row_from_run_dir(run_dir: Path) -> dict[str, Any]:
    totals = {f"{stage}_ms": "" for stage in INTERESTING_STAGES}
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return totals
    with summary_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("name", "")
            if name in INTERESTING_STAGES:
                totals[f"{name}_ms"] = row.get("total_ms", "")
    return totals


def _write_runs(output: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "example",
        "label",
        "phase",
        "repeat",
        "wall_ms",
        *[f"{stage}_ms" for stage in INTERESTING_STAGES],
        "failed",
        "error",
        "stdout_chars",
        "stderr_chars",
    ]
    with (output / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_cases(
    output: Path,
    args: argparse.Namespace,
    specs: list[ExampleSpec],
    amplitudes: tuple[float, ...],
) -> None:
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "script",
                "preset",
                "platform",
                "example",
                "label",
                "axons_per_fascicle",
                "amplitudes_uA",
                "duration_ms",
                "dt_ms",
            ),
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "script": "with_nrv_examples",
                    "preset": args.preset,
                    "platform": args.platform,
                    "example": spec.key,
                    "label": spec.label,
                    "axons_per_fascicle": args.axons_per_fascicle,
                    "amplitudes_uA": ",".join(str(value) for value in amplitudes),
                    "duration_ms": args.duration_ms,
                    "dt_ms": args.dt_ms,
                }
            )


def _write_manifest(
    output: Path,
    args: argparse.Namespace,
    specs: list[ExampleSpec],
    amplitudes: tuple[float, ...],
) -> None:
    payload = {
        "script": "with_nrv_examples",
        "preset": args.preset,
        "platform": args.platform,
        "examples": [spec.key for spec in specs],
        "repeats": args.repeats,
        "warmups": args.warmups,
        "axons_per_fascicle": args.axons_per_fascicle,
        "amplitudes_uA": amplitudes,
        "duration_ms": args.duration_ms,
        "dt_ms": args.dt_ms,
        "memory_trace": args.memory_trace or "rss",
        "output": str(output),
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_report(output: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# With NRV Example Benchmark",
        "",
        "| example | phase | repeat | wall ms | kernel.wait ms | results.to_public ms |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {example} {label} | {phase} | {repeat} | {wall_ms:.3f} | {wait} | {public} |".format(
                example=row["example"],
                label=row["label"],
                phase=row["phase"],
                repeat=row["repeat"],
                wall_ms=float(row["wall_ms"]),
                wait=_fmt_cell(row.get("kernel.wait_ms", "")),
                public=_fmt_cell(row.get("results.to_public_ms", "")),
            )
        )
    _write_text(output / "report.md", "\n".join(lines) + "\n")


def _fmt_cell(value: Any) -> str:
    if value in {"", None}:
        return ""
    return f"{float(value):.3f}"


def _format_progress(row: dict[str, Any]) -> str:
    return (
        f"{row['example']} {row['phase']}#{row['repeat']}: "
        f"wall={float(row['wall_ms']):.1f} ms "
        f"kernel.wait={_fmt_cell(row.get('kernel.wait_ms', '')) or 'n/a'} ms"
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@contextlib.contextmanager
def _maybe_quiet(quiet: bool, stdout: io.StringIO, stderr: io.StringIO):
    if not quiet:
        yield
        return
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        yield


@contextlib.contextmanager
def _plot_safe_context():
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    old_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.show = old_show
        plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())
