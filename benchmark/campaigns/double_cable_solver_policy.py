"""Run a double-cable solver policy benchmark matrix.

This campaign compares typed public double-cable solver choices through the
same curve scripts used for publication/workflow benchmarks. It is deliberately
separate from low-level solver-stage profilers: those explain a result, while
this campaign is the policy gate for keeping, promoting, or removing solver
routes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.run import SCRIPTS
from benchmark.workloads.curve_options import PRESETS


REPO_ROOT = Path(__file__).resolve().parents[2]
CURVE_SCRIPTS = ("threshold_curves", "recruitment_curves")
RECORDINGS = ("observer_only", "probe_vm", "full_vm")
DIAMETER_MODES = ("same_diameter", "different_diameters")
CPU_SOLVERS = ("auto", "thomas")
GPU_SOLVERS = ("auto", "thomas", "pcr", "pcr_soa", "tiled_thomas")
OBSERVER_STATE_SCOPES = ("default", "chunk", "full")

SUMMARY_FIELDS = (
    "label",
    "run_dir",
    "status",
    "returncode",
    "script",
    "platform",
    "solver",
    "tiled_thomas_block_b",
    "recording",
    "observer_state_scope",
    "time_chunk_policy",
    "time_chunk_steps",
    "n_axons",
    "nx",
    "precision",
    "diameters",
    "case_name",
    "effective_variants",
    "curve_simulate_total_ms",
    "curve_simulate_cold_ms",
    "curve_simulate_warm_mean_ms",
    "runtime_prepare_ms",
    "inputs_extracellular_ms",
    "kernel_prepare_inputs_ms",
    "kernel_prepare_arrays_ms",
    "kernel_prepare_double_coefficients_ms",
    "kernel_dispatch_jax_ms",
    "kernel_wait_ms",
    "kernel_finalize_observer_ms",
    "kernel_finalize_observer_to_host_ms",
    "results_assemble_rows_ms",
    "rss_end_mib_max",
)


@dataclass(frozen=True, slots=True)
class RunSpec:
    label: str
    run_dir: Path
    command: tuple[str, ...]
    script: str
    platform: str
    solver: str
    tiled_thomas_block_b: int | None
    recording: str
    observer_state_scope: str
    time_chunk_steps: str | None
    n_axons: int
    nx: int
    precision: str
    diameters: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="quick")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--script", dest="script", action="append", help="Comma-separated curve scripts.")
    parser.add_argument(
        "--curve-script",
        dest="script",
        action="append",
        help="Comma-separated curve scripts. Use this alias through benchmark/kaggle/run_kernel.py.",
    )
    parser.add_argument("--solver", action="append", help="Comma-separated solver policies.")
    parser.add_argument("--recording", action="append", help="Comma-separated recording modes.")
    parser.add_argument("--n-axons", action="append", help="Comma-separated population sizes.")
    parser.add_argument("--nx", action="append", help="Comma-separated target Nx values.")
    parser.add_argument("--precision", action="append", help="Comma-separated fp32/fp64 values.")
    parser.add_argument("--diameters", action="append", help="Comma-separated diameter cohort modes.")
    parser.add_argument("--tiled-thomas-block-b", action="append", help="Comma-separated tile widths.")
    parser.add_argument("--tsim", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--amplitude-count", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument(
        "--time-chunk-steps",
        action="append",
        help=(
            "Comma-separated time chunk policies: default, unchunked, none, "
            "or positive integer chunk sizes."
        ),
    )
    parser.add_argument(
        "--benchmark-observer-state-scope",
        action="append",
        help="Comma-separated observer state scopes: default, chunk, full.",
    )
    parser.add_argument("--memory-trace", choices=("off", "rss", "tracemalloc", "device", "all"))
    parser.add_argument("--memory-top-n", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/p11c_double_cable_solver_policy"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)

    scripts = _parse_scripts(args.script)
    solvers = _parse_solvers(args.solver, platform=args.platform)
    recordings = _parse_choices(
        args.recording,
        allowed=RECORDINGS,
        default=(PRESETS[args.preset].recording,),
        label="recording",
    )
    n_axons_values = _parse_int_values(
        args.n_axons,
        default=(PRESETS[args.preset].n_axons,),
        label="n_axons",
    )
    nx_values = _parse_int_values(
        args.nx,
        default=(PRESETS[args.preset].nx,),
        label="nx",
    )
    precisions = _parse_choices(
        args.precision,
        allowed=("fp32", "fp64"),
        default=(PRESETS[args.preset].precision,),
        label="precision",
    )
    diameter_modes = _parse_choices(
        args.diameters,
        allowed=DIAMETER_MODES,
        default=("different_diameters",),
        label="diameters",
    )
    block_bs = _parse_int_values(
        args.tiled_thomas_block_b,
        default=(32,),
        label="tiled_thomas_block_b",
    )
    observer_state_scopes = _parse_choices(
        args.benchmark_observer_state_scope,
        allowed=OBSERVER_STATE_SCOPES,
        default=("default",),
        label="benchmark_observer_state_scope",
    )
    time_chunk_steps_values = _parse_time_chunk_values(
        args.time_chunk_steps,
        default=(None,),
    )

    if args.repeats is not None and args.repeats < 1:
        parser.error("--repeats must be >= 1.")
    if args.warmups is not None and args.warmups < 0:
        parser.error("--warmups must be >= 0.")

    args.output.mkdir(parents=True, exist_ok=True)
    runs = _build_runs(
        args,
        scripts=scripts,
        solvers=solvers,
        recordings=recordings,
        n_axons_values=n_axons_values,
        nx_values=nx_values,
        precisions=precisions,
        diameter_modes=diameter_modes,
        block_bs=block_bs,
        observer_state_scopes=observer_state_scopes,
        time_chunk_steps_values=time_chunk_steps_values,
    )
    _write_manifest(
        args.output / "double_cable_solver_policy_manifest.json",
        args=args,
        runs=runs,
    )
    summary_path = args.output / "double_cable_solver_policy_summary.csv"
    report_path = args.output / "double_cable_solver_policy_report.md"
    print(f"planned: {len(runs)} double-cable solver policy runs", flush=True)

    if args.dry_run:
        for run in runs:
            print("$", shlex.join(run.command), flush=True)
        print(f"wrote: {args.output / 'double_cable_solver_policy_manifest.json'}", flush=True)
        return 0

    rows: list[dict[str, Any]] = []
    failed = False
    for index, run in enumerate(runs, start=1):
        print(f"running {index}/{len(runs)}: {run.label}", flush=True)
        run.run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run.run_dir / "campaign_command.json", _run_spec_json(run))
        result = subprocess.run(
            list(run.command),
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        (run.run_dir / "campaign_command.log").write_text(
            result.stdout + result.stderr,
            encoding="utf-8",
        )
        status = "passed" if result.returncode == 0 else "failed"
        print(
            f"{status}: {index}/{len(runs)} {run.label} "
            f"(returncode={result.returncode})",
            flush=True,
        )
        if result.returncode != 0:
            failed = True
            _print_failure_tail(run.run_dir / "campaign_command.log")
        rows.append(_summarize_curve_run(run, status=status, returncode=result.returncode))
        _write_csv(summary_path, rows)
        _write_report(report_path, rows)
        print(f"progress: wrote {len(rows)}/{len(runs)} rows to {summary_path}", flush=True)
        if failed and not args.keep_going:
            break

    _write_csv(summary_path, rows)
    _write_report(report_path, rows)
    print(f"wrote: {summary_path}", flush=True)
    print(f"wrote: {report_path}", flush=True)
    return 1 if failed else 0


def _build_runs(
    args: argparse.Namespace,
    *,
    scripts: Sequence[str],
    solvers: Sequence[str],
    recordings: Sequence[str],
    n_axons_values: Sequence[int],
    nx_values: Sequence[int],
    precisions: Sequence[str],
    diameter_modes: Sequence[str],
    block_bs: Sequence[int],
    observer_state_scopes: Sequence[str],
    time_chunk_steps_values: Sequence[str | None],
) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for script in scripts:
        for recording in recordings:
            for n_axons in n_axons_values:
                for nx in nx_values:
                    for precision in precisions:
                        for diameters in diameter_modes:
                            for time_chunk_steps in time_chunk_steps_values:
                                for observer_state_scope in observer_state_scopes:
                                    for solver in solvers:
                                        solver_block_bs: Sequence[int | None]
                                        solver_block_bs = block_bs if solver == "tiled_thomas" else (None,)
                                        for block_b in solver_block_bs:
                                            label = _label(
                                                script=script,
                                                platform=args.platform,
                                                solver=solver,
                                                block_b=block_b,
                                                recording=recording,
                                                observer_state_scope=observer_state_scope,
                                                time_chunk_steps=time_chunk_steps,
                                                n_axons=n_axons,
                                                nx=nx,
                                                precision=precision,
                                                diameters=diameters,
                                            )
                                            command = _curve_command(
                                                args,
                                                script=script,
                                                output=args.output / label,
                                                solver=solver,
                                                block_b=block_b,
                                                recording=recording,
                                                observer_state_scope=observer_state_scope,
                                                time_chunk_steps=time_chunk_steps,
                                                n_axons=n_axons,
                                                nx=nx,
                                                precision=precision,
                                                diameters=diameters,
                                            )
                                            runs.append(
                                                RunSpec(
                                                    label=label,
                                                    run_dir=args.output / label,
                                                    command=tuple(command),
                                                    script=script,
                                                    platform=str(args.platform),
                                                    solver=solver,
                                                    tiled_thomas_block_b=block_b,
                                                    recording=recording,
                                                    observer_state_scope=observer_state_scope,
                                                    time_chunk_steps=time_chunk_steps,
                                                    n_axons=n_axons,
                                                    nx=nx,
                                                    precision=precision,
                                                    diameters=diameters,
                                                )
                                            )
    return runs


def _curve_command(
    args: argparse.Namespace,
    *,
    script: str,
    output: Path,
    solver: str,
    block_b: int | None,
    recording: str,
    n_axons: int,
    nx: int,
    precision: str,
    diameters: str,
    observer_state_scope: str,
    time_chunk_steps: str | None,
) -> list[str]:
    command = [
        args.python,
        "benchmark/run.py",
        "--script",
        script,
        "--preset",
        args.preset,
        "--platform",
        args.platform,
        "--output",
        str(output),
        "--cable",
        "double_cable",
        "--population",
        "single_model",
        "--diameters",
        diameters,
        "--recording",
        recording,
        "--n-axons",
        str(n_axons),
        "--nx",
        str(nx),
        "--precision",
        precision,
        "--double-cable-block-solver",
        solver,
    ]
    if block_b is not None:
        command.extend(("--tiled-thomas-block-b", str(block_b)))
    if args.tsim is not None:
        command.extend(("--tsim", str(args.tsim)))
    if args.dt is not None:
        command.extend(("--dt", str(args.dt)))
    if args.repeats is not None:
        command.extend(("--repeats", str(args.repeats)))
    if args.warmups is not None:
        command.extend(("--warmups", str(args.warmups)))
    if args.memory_trace is not None:
        command.extend(("--memory-trace", args.memory_trace))
    if args.memory_top_n is not None:
        command.extend(("--memory-top-n", str(args.memory_top_n)))
    if time_chunk_steps is not None:
        command.extend(("--time-chunk-steps", time_chunk_steps))
    if observer_state_scope != "default":
        command.extend(
            ("--benchmark-observer-state-scope", observer_state_scope)
        )
    if script == "recruitment_curves" and args.amplitude_count is not None:
        command.extend(("--amplitude-count", str(args.amplitude_count)))
    if script == "threshold_curves":
        max_iterations = args.max_iterations
        if max_iterations is None and args.amplitude_count is not None:
            max_iterations = args.amplitude_count
        if max_iterations is not None:
            command.extend(("--max-iterations", str(max_iterations)))
    if args.resume:
        command.append("--resume")
    return command


def _summarize_curve_run(
    run: RunSpec,
    *,
    status: str,
    returncode: int,
) -> dict[str, Any]:
    manifest = _read_json(run.run_dir / "manifest.json")
    options = _mapping(manifest.get("options"))
    events = _read_events(run.run_dir / "events.jsonl")
    simulate = [
        event for event in events if str(event.get("name") or "") == "curve.simulate"
    ]
    simulate.sort(key=lambda event: float(event.get("start_ns") or 0.0))
    warm = simulate[1:] if len(simulate) > 1 else ()
    return {
        "label": run.label,
        "run_dir": str(run.run_dir),
        "status": status,
        "returncode": returncode,
        "script": manifest.get("script", run.script),
        "platform": options.get("platform", run.platform),
        "solver": run.solver,
        "tiled_thomas_block_b": "" if run.tiled_thomas_block_b is None else run.tiled_thomas_block_b,
        "recording": options.get("recording", run.recording),
        "observer_state_scope": options.get(
            "benchmark_observer_state_scope",
            run.observer_state_scope,
        ),
        "time_chunk_policy": options.get(
            "time_chunk_policy",
            _time_chunk_policy_from_token(run.time_chunk_steps),
        ),
        "time_chunk_steps": _summary_time_chunk_steps(
            options,
            run.time_chunk_steps,
        ),
        "n_axons": options.get("n_axons", run.n_axons),
        "nx": options.get("nx", run.nx),
        "precision": options.get("precision", run.precision),
        "diameters": options.get("diameters", run.diameters),
        "case_name": manifest.get("case_name", ""),
        "effective_variants": "/".join(_variants(events)),
        "curve_simulate_total_ms": _sum_duration(events, "curve.simulate"),
        "curve_simulate_cold_ms": _duration(simulate[0]) if simulate else "",
        "curve_simulate_warm_mean_ms": _mean_duration(warm),
        "runtime_prepare_ms": _sum_duration(events, "runtime.prepare"),
        "inputs_extracellular_ms": _sum_duration(events, "inputs.extracellular"),
        "kernel_prepare_inputs_ms": _sum_duration(events, "kernel.prepare_inputs"),
        "kernel_prepare_arrays_ms": _sum_duration(events, "kernel.prepare_arrays"),
        "kernel_prepare_double_coefficients_ms": _sum_duration(
            events,
            "kernel.prepare_double_coefficients",
        ),
        "kernel_dispatch_jax_ms": _sum_duration(events, "kernel.dispatch_jax"),
        "kernel_wait_ms": _sum_duration(events, "kernel.wait"),
        "kernel_finalize_observer_ms": _sum_duration(events, "kernel.finalize_observer"),
        "kernel_finalize_observer_to_host_ms": _sum_duration(
            events,
            "kernel.finalize_observer.to_host",
        ),
        "results_assemble_rows_ms": _sum_duration(events, "results.assemble_rows"),
        "rss_end_mib_max": _peak_memory(events, "rss_end_mib"),
    }


def _print_failure_tail(path: Path, *, max_lines: int = 80) -> None:
    if not path.exists():
        print(f"failure log missing: {path}", flush=True)
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-max_lines:]
    print(f"--- failure log tail: {path} ---", flush=True)
    for line in tail:
        print(line, flush=True)
    print("--- end failure log tail ---", flush=True)


def _parse_scripts(values: Sequence[str] | None) -> tuple[str, ...]:
    return _parse_choices(
        values,
        allowed=CURVE_SCRIPTS,
        default=CURVE_SCRIPTS,
        label="script",
    )


def _parse_solvers(values: Sequence[str] | None, *, platform: str) -> tuple[str, ...]:
    allowed = GPU_SOLVERS if platform == "gpu" else CPU_SOLVERS
    default = GPU_SOLVERS if platform == "gpu" else CPU_SOLVERS
    return _parse_choices(values, allowed=allowed, default=default, label="solver")


def _parse_choices(
    values: Sequence[str] | None,
    *,
    allowed: Sequence[str],
    default: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    if not values:
        return tuple(default)
    result: list[str] = []
    allowed_set = set(allowed)
    for value in values:
        for item in str(value).split(","):
            token = item.strip()
            if not token:
                continue
            if token not in allowed_set:
                allowed_text = ", ".join(allowed)
                raise SystemExit(f"{label} must be one of: {allowed_text}.")
            if token not in result:
                result.append(token)
    if not result:
        raise SystemExit(f"at least one {label} is required.")
    return tuple(result)


def _parse_int_values(
    values: Sequence[str] | None,
    *,
    default: Sequence[int],
    label: str,
) -> tuple[int, ...]:
    if not values:
        return tuple(int(value) for value in default)
    result: list[int] = []
    for value in values:
        for item in str(value).split(","):
            token = item.strip()
            if not token:
                continue
            try:
                parsed = int(token)
            except ValueError as exc:
                raise SystemExit(f"{label} values must be integers.") from exc
            if parsed < 1:
                raise SystemExit(f"{label} values must be >= 1.")
            if parsed not in result:
                result.append(parsed)
    if not result:
        raise SystemExit(f"at least one {label} is required.")
    return tuple(result)


def _parse_time_chunk_values(
    values: Sequence[str] | None,
    *,
    default: Sequence[str | None],
) -> tuple[str | None, ...]:
    if not values:
        return tuple(default)
    result: list[str] = []
    for value in values:
        for item in str(value).split(","):
            token = item.strip()
            if not token:
                continue
            parsed = _normalize_time_chunk_token(token)
            if parsed not in result:
                result.append(parsed)
    if not result:
        raise SystemExit("at least one time_chunk_steps value is required.")
    return tuple(result)


def _normalize_time_chunk_token(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"", "default"}:
        return "default"
    if text in {"none", "off", "unchunked", "full"}:
        return "unchunked"
    try:
        parsed = int(text)
    except ValueError as exc:
        raise SystemExit(
            "time_chunk_steps values must be default, unchunked, none, "
            "or positive integers."
        ) from exc
    if parsed < 1:
        raise SystemExit("time_chunk_steps values must be >= 1.")
    return str(parsed)


def _time_chunk_policy_from_token(value: str | None) -> str:
    if value is None or value == "default":
        return "default"
    if value == "unchunked":
        return "unchunked"
    return "explicit"


def _summary_time_chunk_steps(
    options: Mapping[str, Any],
    fallback: str | None,
) -> str:
    policy = str(
        options.get("time_chunk_policy")
        or _time_chunk_policy_from_token(fallback)
    )
    if policy == "explicit":
        value = options.get("time_chunk_steps")
        return str(value if value not in (None, "") else fallback)
    if policy == "unchunked":
        return "unchunked"
    return "default"


def _time_chunk_label_token(value: str | None) -> str:
    if value is None:
        return ""
    return f"__tc{value}"


def _label(
    *,
    script: str,
    platform: str,
    solver: str,
    block_b: int | None,
    recording: str,
    observer_state_scope: str,
    time_chunk_steps: str | None,
    n_axons: int,
    nx: int,
    precision: str,
    diameters: str,
) -> str:
    solver_token = solver if block_b is None else f"{solver}_b{block_b}"
    observer_token = (
        ""
        if observer_state_scope == "default"
        else f"__obs_{observer_state_scope}"
    )
    time_chunk_token = _time_chunk_label_token(time_chunk_steps)
    return (
        f"{script}__{platform}__{solver_token}__{recording}__"
        f"n{n_axons}__nx{nx}__{precision}__{diameters}"
        f"{time_chunk_token}{observer_token}"
    )


def _write_manifest(path: Path, *, args: argparse.Namespace, runs: Sequence[RunSpec]) -> None:
    _write_json(
        path,
        {
            "campaign": "double_cable_solver_policy",
            "preset": args.preset,
            "platform": args.platform,
            "runs": [_run_spec_json(run) for run in runs],
        },
    )


def _run_spec_json(run: RunSpec) -> dict[str, Any]:
    return {
        "label": run.label,
        "run_dir": str(run.run_dir),
        "command": list(run.command),
        "script": run.script,
        "platform": run.platform,
        "solver": run.solver,
        "tiled_thomas_block_b": run.tiled_thomas_block_b,
        "recording": run.recording,
        "observer_state_scope": run.observer_state_scope,
        "time_chunk_steps": run.time_chunk_steps,
        "n_axons": run.n_axons,
        "nx": run.nx,
        "precision": run.precision,
        "diameters": run.diameters,
    }


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Double-Cable Solver Policy Benchmark",
        "",
        "This report compares typed public double-cable solver policies through the curve workloads.",
        "Low-level solver-stage profilers should be used only to explain these results.",
        "",
    ]
    if not rows:
        lines.append("No runs.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.extend(
        [
            "## Fastest Rows",
            "",
            "| group | solver | block_b | observer_scope | time_chunk | warm mean ms | total simulate ms | variants | status |",
            "| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for group, row in _fastest_rows(rows):
        lines.append(
            "| {group} | {solver} | {block_b} | {observer_scope} | {time_chunk} | {warm} | {total} | {variants} | {status} |".format(
                group=group,
                solver=row.get("solver", ""),
                block_b=row.get("tiled_thomas_block_b", ""),
                observer_scope=row.get("observer_state_scope", ""),
                time_chunk=row.get("time_chunk_steps", ""),
                warm=_format_number(row.get("curve_simulate_warm_mean_ms")),
                total=_format_number(row.get("curve_simulate_total_ms")),
                variants=row.get("effective_variants", ""),
                status=row.get("status", ""),
            )
        )
    lines.extend(
        [
            "",
            "## All Rows",
            "",
            "| script | platform | solver | block_b | recording | observer_scope | time_chunk | n_axons | nx | precision | warm mean ms | total ms | variants | status |",
            "| --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {script} | {platform} | {solver} | {block_b} | {recording} | {observer_scope} | {time_chunk} | {n_axons} | {nx} | {precision} | {warm} | {total} | {variants} | {status} |".format(
                script=row.get("script", ""),
                platform=row.get("platform", ""),
                solver=row.get("solver", ""),
                block_b=row.get("tiled_thomas_block_b", ""),
                recording=row.get("recording", ""),
                observer_scope=row.get("observer_state_scope", ""),
                time_chunk=row.get("time_chunk_steps", ""),
                n_axons=row.get("n_axons", ""),
                nx=row.get("nx", ""),
                precision=row.get("precision", ""),
                warm=_format_number(row.get("curve_simulate_warm_mean_ms")),
                total=_format_number(row.get("curve_simulate_total_ms")),
                variants=row.get("effective_variants", ""),
                status=row.get("status", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Policy Notes",
            "",
            "- Treat failed rows as blockers before making policy decisions.",
            "- Prefer warm means for steady-state policy; inspect cold rows separately for compile/startup cost.",
            "- Do not remove solver code or tests until CPU and GPU artifacts agree with the intended support policy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fastest_rows(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "passed":
            continue
        group = "|".join(
            str(row.get(key, ""))
            for key in ("script", "platform", "recording", "n_axons", "nx", "precision", "diameters")
        )
        groups.setdefault(group, []).append(row)
    fastest: list[tuple[str, Mapping[str, Any]]] = []
    for group, items in groups.items():
        fastest.append((group, min(items, key=_policy_sort_key)))
    return sorted(fastest, key=lambda item: item[0])


def _policy_sort_key(row: Mapping[str, Any]) -> tuple[float, float]:
    warm = _float_or_none(row.get("curve_simulate_warm_mean_ms"))
    total = _float_or_none(row.get("curve_simulate_total_ms"))
    return (
        warm if warm is not None else float("inf"),
        total if total is not None else float("inf"),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _variants(events: Sequence[Mapping[str, Any]]) -> list[str]:
    variants = {
        str(_mapping(event.get("metadata")).get("variant"))
        for event in events
        if _mapping(event.get("metadata")).get("variant")
    }
    return sorted(variants)


def _duration(event: Mapping[str, Any]) -> float:
    return float(event.get("duration_ms") or 0.0)


def _mean_duration(events: Sequence[Mapping[str, Any]]) -> float | str:
    if not events:
        return ""
    return sum(_duration(event) for event in events) / len(events)


def _sum_duration(events: Sequence[Mapping[str, Any]], name: str) -> float:
    return sum(_duration(event) for event in events if str(event.get("name") or "") == name)


def _peak_memory(events: Sequence[Mapping[str, Any]], key: str) -> float | str:
    values: list[float] = []
    for event in events:
        memory = _mapping(_mapping(event.get("metadata")).get("memory"))
        value = _float_or_none(memory.get(key))
        if value is not None:
            values.append(value)
    return max(values) if values else ""


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: object) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return ""
    return f"{parsed:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
