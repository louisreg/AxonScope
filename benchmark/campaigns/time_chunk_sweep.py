"""Run a reproducible time-chunk policy sweep for P11B optimization triage."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.run import SCRIPTS
from benchmark.workloads.curve_options import PRESETS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICIES = ("default", "unchunked", "50", "250", "500", "1000")

SUMMARY_FIELDS = (
    "policy",
    "run_dir",
    "status",
    "returncode",
    "case_name",
    "script",
    "platform",
    "recording",
    "n_axons",
    "nx",
    "tsim",
    "dt",
    "precision",
    "time_chunk_policy",
    "time_chunk_steps",
    "dispatch_time_chunk_steps",
    "observer_state_scopes",
    "dispatch_chunk_steps",
    "dispatch_chunk_count_max",
    "curve_simulate_ms",
    "repeat_curve_simulate_ms",
    "kernel_enqueue_ms",
    "repeat_kernel_enqueue_ms",
    "kernel_combine_observer_chunks_ms",
    "repeat_kernel_combine_observer_chunks_ms",
    "kernel_dispatch_jax_ms",
    "repeat_kernel_dispatch_jax_ms",
    "kernel_dispatch_jax_count",
    "kernel_wait_ms",
    "repeat_kernel_wait_ms",
    "kernel_finalize_observer_ms",
    "repeat_kernel_finalize_observer_ms",
    "rss_end_mib_max",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", default="recruitment_curves", choices=tuple(SCRIPTS))
    parser.add_argument("--preset", default="quick", choices=tuple(PRESETS))
    parser.add_argument("--platform", default="cpu", choices=("cpu", "gpu", "nrv"))
    parser.add_argument(
        "--policies",
        action="append",
        help=(
            "Comma-separated policies to run. Defaults to "
            "default,unchunked,50,250,500,1000."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/p11b_time_chunk_sweep"),
        help="Campaign output directory. Each policy gets a child result directory.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args, extra = parser.parse_known_args(argv)

    try:
        policies = parse_policies(args.policies)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    runs = build_runs(args, policies, extra)
    write_manifest(args.output / "time_chunk_sweep_manifest.json", args, runs, extra)

    if args.dry_run:
        for run in runs:
            print("$", shell_join(run["command"]))
        print(f"wrote: {args.output / 'time_chunk_sweep_manifest.json'}")
        return 0

    rows: list[dict[str, Any]] = []
    failed = False
    for run in runs:
        command = [str(part) for part in run["command"]]
        run_dir = Path(run["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "campaign_command.json", run)
        result = run_command(command, cwd=REPO_ROOT)
        (run_dir / "campaign_command.log").write_text(
            result.stdout + result.stderr,
            encoding="utf-8",
        )
        status = "passed" if result.returncode == 0 else "failed"
        if result.returncode != 0:
            failed = True
            print(f"failed: {run['policy']} (returncode={result.returncode})")
        else:
            print(f"passed: {run['policy']}")
        rows.append(
            summarize_run(
                run_dir,
                policy=str(run["policy"]),
                status=status,
                returncode=result.returncode,
            )
        )
        if failed and not args.keep_going:
            break

    write_summary(args.output / "time_chunk_sweep_summary.csv", rows)
    write_report(args.output / "time_chunk_sweep_report.md", rows)
    print(f"wrote: {args.output / 'time_chunk_sweep_summary.csv'}")
    print(f"wrote: {args.output / 'time_chunk_sweep_report.md'}")
    return 1 if failed else 0


def parse_policies(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_POLICIES
    policies: list[str] = []
    for value in values:
        for item in str(value).split(","):
            policy = normalize_policy(item)
            if policy not in policies:
                policies.append(policy)
    if not policies:
        raise argparse.ArgumentTypeError("at least one time-chunk policy is required.")
    return tuple(policies)


def normalize_policy(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"", "default"}:
        return "default"
    if text in {"none", "off", "unchunked", "full"}:
        return "unchunked"
    try:
        steps = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "time-chunk policies must be default, unchunked, none, or positive integers."
        ) from exc
    if steps < 1:
        raise argparse.ArgumentTypeError("time-chunk integer policies must be >= 1.")
    return str(steps)


def build_runs(
    args: argparse.Namespace,
    policies: Sequence[str],
    extra: Sequence[str],
) -> list[dict[str, Any]]:
    runs = []
    for policy in policies:
        run_dir = args.output / policy_token(policy)
        command = [
            args.python,
            "benchmark/run.py",
            "--script",
            args.script,
            "--preset",
            args.preset,
            "--platform",
            args.platform,
            "--output",
            str(run_dir),
            "--time-chunk-steps",
            policy,
        ]
        if args.resume:
            command.append("--resume")
        command.extend(str(item) for item in extra)
        runs.append(
            {
                "policy": policy,
                "run_dir": str(run_dir),
                "command": command,
            }
        )
    return runs


def policy_token(policy: str) -> str:
    if policy in {"default", "unchunked"}:
        return policy
    return f"chunk_{policy}"


def run_command(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def summarize_run(
    run_dir: Path,
    *,
    policy: str,
    status: str = "passed",
    returncode: int = 0,
) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    options = mapping(manifest.get("options"))
    events = read_events(run_dir / "events.jsonl")
    by_id = {int(event.get("event_id", -1)): event for event in events}
    dispatch_events = [event for event in events if event.get("name") == "kernel.dispatch_jax"]
    row: dict[str, Any] = {
        "policy": policy,
        "run_dir": str(run_dir),
        "status": status,
        "returncode": returncode,
        "case_name": manifest.get("case_name", ""),
        "script": manifest.get("script", ""),
        "platform": options.get("platform", ""),
        "recording": options.get("recording", ""),
        "n_axons": options.get("n_axons", ""),
        "nx": options.get("nx", ""),
        "tsim": options.get("tsim", ""),
        "dt": options.get("dt", ""),
        "precision": options.get("precision", ""),
        "time_chunk_policy": options.get("time_chunk_policy", ""),
        "time_chunk_steps": none_to_empty(options.get("time_chunk_steps")),
        "dispatch_time_chunk_steps": unique_metadata(dispatch_events, "time_chunk_steps"),
        "observer_state_scopes": unique_metadata(dispatch_events, "observer_state_scope"),
        "dispatch_chunk_steps": unique_metadata(dispatch_events, "chunk_steps"),
        "dispatch_chunk_count_max": max_metadata(dispatch_events, "chunk_count"),
        "curve_simulate_ms": sum_duration(events, "curve.simulate"),
        "repeat_curve_simulate_ms": sum_duration(events, "curve.simulate", phase="repeat", by_id=by_id),
        "kernel_enqueue_ms": sum_duration(events, "kernel.enqueue"),
        "repeat_kernel_enqueue_ms": sum_duration(events, "kernel.enqueue", phase="repeat", by_id=by_id),
        "kernel_combine_observer_chunks_ms": sum_duration(
            events,
            "kernel.combine_observer_chunks",
        ),
        "repeat_kernel_combine_observer_chunks_ms": sum_duration(
            events,
            "kernel.combine_observer_chunks",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_dispatch_jax_ms": sum_duration(events, "kernel.dispatch_jax"),
        "repeat_kernel_dispatch_jax_ms": sum_duration(
            events,
            "kernel.dispatch_jax",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_dispatch_jax_count": len(dispatch_events),
        "kernel_wait_ms": sum_duration(events, "kernel.wait"),
        "repeat_kernel_wait_ms": sum_duration(events, "kernel.wait", phase="repeat", by_id=by_id),
        "kernel_finalize_observer_ms": sum_duration(events, "kernel.finalize_observer"),
        "repeat_kernel_finalize_observer_ms": sum_duration(
            events,
            "kernel.finalize_observer",
            phase="repeat",
            by_id=by_id,
        ),
        "rss_end_mib_max": max_event_memory(events, "rss_end_mib"),
    }
    return {field: row.get(field, "") for field in SUMMARY_FIELDS}


def sum_duration(
    events: Sequence[Mapping[str, Any]],
    name: str,
    *,
    phase: str | None = None,
    by_id: Mapping[int, Mapping[str, Any]] | None = None,
) -> float:
    total = 0.0
    for event in events:
        if event.get("name") != name:
            continue
        if phase is not None and event_phase(event, by_id or {}) != phase:
            continue
        total += float(event.get("duration_ms") or 0.0)
    return total


def event_phase(
    event: Mapping[str, Any],
    by_id: Mapping[int, Mapping[str, Any]],
) -> str:
    current: Mapping[str, Any] | None = event
    seen: set[int] = set()
    while current is not None:
        metadata = mapping(current.get("metadata"))
        phase = metadata.get("phase")
        if phase:
            return str(phase)
        parent = current.get("parent_event_id")
        if parent is None:
            return ""
        try:
            parent_id = int(parent)
        except (TypeError, ValueError):
            return ""
        if parent_id in seen:
            return ""
        seen.add(parent_id)
        current = by_id.get(parent_id)
    return ""


def unique_metadata(events: Sequence[Mapping[str, Any]], key: str) -> str:
    values = []
    for event in events:
        value = mapping(event.get("metadata")).get(key)
        if value is None:
            continue
        text = str(value)
        if text not in values:
            values.append(text)
    return ";".join(values)


def max_metadata(events: Sequence[Mapping[str, Any]], key: str) -> str:
    values = []
    for event in events:
        value = mapping(event.get("metadata")).get(key)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            pass
    if not values:
        return ""
    value = max(values)
    return str(int(value)) if value.is_integer() else str(value)


def max_event_memory(events: Sequence[Mapping[str, Any]], key: str) -> str:
    values = []
    for event in events:
        memory = mapping(mapping(event.get("metadata")).get("memory"))
        try:
            values.append(float(memory.get(key)))
        except (TypeError, ValueError):
            pass
    if not values:
        return ""
    return f"{max(values):.6g}"


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
        for value in (json.loads(line),)
        if isinstance(value, dict)
    ]


def write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def write_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Time Chunk Sweep",
        "",
        "| policy | status | curve.simulate ms | dispatch_jax ms | combine ms | finalize ms | dispatch count | scope | chunk steps |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {status} | {curve:.3f} | {dispatch:.3f} | {combine:.3f} | {finalize:.3f} | {count} | {scope} | {chunks} |".format(
                policy=row.get("policy", ""),
                status=row.get("status", ""),
                curve=float(row.get("repeat_curve_simulate_ms") or 0.0),
                dispatch=float(row.get("repeat_kernel_dispatch_jax_ms") or 0.0),
                combine=float(row.get("repeat_kernel_combine_observer_chunks_ms") or 0.0),
                finalize=float(row.get("repeat_kernel_finalize_observer_ms") or 0.0),
                count=row.get("kernel_dispatch_jax_count", ""),
                scope=row.get("observer_state_scopes", ""),
                chunks=row.get("dispatch_chunk_steps", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    runs: Sequence[Mapping[str, Any]],
    extra: Sequence[str],
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": args.script,
        "preset": args.preset,
        "platform": args.platform,
        "output": str(args.output),
        "policies": [run["policy"] for run in runs],
        "extra_args": list(extra),
        "runs": list(runs),
    }
    write_json(path, payload)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def none_to_empty(value: Any) -> Any:
    return "" if value is None else value


def shell_join(command: Sequence[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


if __name__ == "__main__":
    raise SystemExit(main())
