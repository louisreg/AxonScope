"""Compare requested/effective solver choices across benchmark traces."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROW_FIELDS = (
    "label",
    "run_dir",
    "platform",
    "requested_solver",
    "effective_solver",
    "iteration",
    "temperature",
    "simulate_ms",
    "runtime_prepare_ms",
    "kernel_enqueue_ms",
    "kernel_dispatch_jax_ms",
    "kernel_wait_ms",
    "kernel_sync_ms",
    "kernel_sync_share",
    "kernel_finalize_observer_ms",
    "kernel_finalize_to_host_ms",
    "kernel_prepare_state_ms",
    "kernel_prepare_observer_state_ms",
    "kernel_chunk_setup_ms",
    "kernel_chunk_bookkeeping_ms",
    "inputs_extracellular_ms",
    "peak_rss_end_mib",
    "peak_rss_delta_mib",
    "n_axons",
    "nx",
    "tsim",
    "dt",
    "recording",
    "precision",
    "git_commit",
    "device_models",
)


STAGES = (
    "runtime.prepare",
    "kernel.enqueue",
    "kernel.dispatch_jax",
    "kernel.wait",
    "kernel.finalize_observer",
    "kernel.finalize_observer.to_host",
    "kernel.prepare_state",
    "kernel.prepare_observer_state",
    "kernel.chunk_setup",
    "kernel.chunk_bookkeeping",
    "inputs.extracellular",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Benchmark output directory containing events.jsonl. LABEL= is optional.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/p11b_solver_choice_report"),
    )
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("at least one --run LABEL=DIR is required")

    rows: list[dict[str, Any]] = []
    for value in args.run:
        label, path = _parse_run_arg(value)
        rows.extend(_read_run(path, label=label))
    if not rows:
        print("No curve.simulate spans found.")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    rows_csv = args.output / "solver_choice_rows.csv"
    report_md = args.output / "solver_choice_report.md"
    _write_csv(rows_csv, rows)
    _write_report(report_md, rows)
    print(f"wrote: {rows_csv}")
    print(f"wrote: {report_md}")
    _print_summary(rows)
    return 0


def _parse_run_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), Path(path).expanduser()
    path = Path(value).expanduser()
    return _path_label(path), path


def _read_run(run_dir: Path, *, label: str) -> list[dict[str, Any]]:
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        raise FileNotFoundError(f"missing events.jsonl in {run_dir}")

    manifest = _read_json(run_dir / "manifest.json")
    environment = _read_json(run_dir / "environment.json")
    options = _mapping(manifest.get("options"))
    git = _mapping(environment.get("git"))
    device_models = ";".join(
        str(item) for item in _sequence(environment.get("compute_device_models"))
    )
    events = [
        _mapping(json.loads(line))
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    children: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        parent_id = _int_or_none(event.get("parent_event_id"))
        if parent_id is not None:
            children[parent_id].append(event)

    simulate_events = [
        event for event in events if str(event.get("name") or "") == "curve.simulate"
    ]
    simulate_events.sort(key=lambda event: int(_float(event.get("start_ns")) or 0))

    rows: list[dict[str, Any]] = []
    for index, root in enumerate(simulate_events):
        descendants = _descendants(root, children)
        root_meta = _mapping(root.get("metadata"))
        row = {
            "label": label,
            "run_dir": str(run_dir),
            "platform": str(options.get("platform") or root_meta.get("platform") or ""),
            "requested_solver": str(
                options.get("double_cable_block_solver")
                or root_meta.get("double_cable_block_solver")
                or ""
            ),
            "effective_solver": "/".join(_variants(descendants)),
            "iteration": str(
                root_meta.get("iteration")
                if root_meta.get("iteration") is not None
                else index
            ),
            "temperature": "cold" if index == 0 else "warm",
            "simulate_ms": _float(root.get("duration_ms")) or 0.0,
            "n_axons": str(options.get("n_axons") or ""),
            "nx": str(options.get("nx") or ""),
            "tsim": str(options.get("tsim") or ""),
            "dt": str(options.get("dt") or ""),
            "recording": str(options.get("recording") or ""),
            "precision": str(options.get("precision") or ""),
            "git_commit": str(git.get("short_commit") or git.get("commit") or ""),
            "device_models": device_models,
        }
        for stage in STAGES:
            row[_field_name(stage)] = _stage_total(descendants, stage)
        row["kernel_sync_ms"] = row["kernel_dispatch_jax_ms"] + row["kernel_wait_ms"]
        row["kernel_sync_share"] = (
            row["kernel_sync_ms"] / row["simulate_ms"] if row["simulate_ms"] else 0.0
        )
        row["peak_rss_end_mib"] = _peak_memory(descendants, "rss_end_mib")
        row["peak_rss_delta_mib"] = _peak_memory(descendants, "rss_delta_mib")
        rows.append(row)
    return rows


def _descendants(
    root: Mapping[str, Any],
    children: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    root_id = int(_float(root.get("event_id")) or 0)
    result: list[Mapping[str, Any]] = []
    stack = list(children.get(root_id, ()))
    while stack:
        event = stack.pop()
        result.append(event)
        event_id = int(_float(event.get("event_id")) or 0)
        stack.extend(children.get(event_id, ()))
    return result


def _variants(events: Sequence[Mapping[str, Any]]) -> list[str]:
    variants = {
        str(_mapping(event.get("metadata")).get("variant"))
        for event in events
        if _mapping(event.get("metadata")).get("variant")
    }
    return sorted(variants)


def _stage_total(events: Sequence[Mapping[str, Any]], stage: str) -> float:
    return sum(
        _float(event.get("duration_ms")) or 0.0
        for event in events
        if str(event.get("name") or "") == stage
    )


def _field_name(stage: str) -> str:
    return stage.replace(".", "_") + "_ms"


def _peak_memory(events: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        _float(_mapping(_mapping(event.get("metadata")).get("memory")).get(key))
        for event in events
    ]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ROW_FIELDS})


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    warm_rows = [row for row in rows if row.get("temperature") == "warm"]
    lines = [
        "# P11B Real Workflow Solver Choice Report",
        "",
        "This report compares requested and effective double-cable solver choices",
        "inside the same recruitment workflow. `cold` is the first amplitude pass",
        "and includes JAX compilation/cache misses; `warm` is the following pass.",
        "",
        "## Runs",
        "",
    ]
    lines.extend(
        _markdown_table(
            (
                "label",
                "platform",
                "requested",
                "effective",
                "temp",
                "simulate ms",
                "kernel sync ms",
                "kernel share",
                "runtime prepare ms",
                "peak RSS MiB",
            ),
            [
                (
                    row["label"],
                    row["platform"],
                    row["requested_solver"],
                    row["effective_solver"],
                    row["temperature"],
                    _fmt(row["simulate_ms"]),
                    _fmt(row["kernel_sync_ms"]),
                    _pct(row["kernel_sync_share"]),
                    _fmt(row["runtime_prepare_ms"]),
                    _fmt_optional(row.get("peak_rss_end_mib")),
                )
                for row in rows
            ],
        )
    )
    lines.extend(["", "## Warm Path", ""])
    lines.extend(
        _markdown_table(
            (
                "label",
                "platform",
                "solver",
                "simulate ms",
                "dispatch ms",
                "wait ms",
                "kernel sync ms",
                "share",
            ),
            [
                (
                    row["label"],
                    row["platform"],
                    row["effective_solver"] or row["requested_solver"],
                    _fmt(row["simulate_ms"]),
                    _fmt(row["kernel_dispatch_jax_ms"]),
                    _fmt(row["kernel_wait_ms"]),
                    _fmt(row["kernel_sync_ms"]),
                    _pct(row["kernel_sync_share"]),
                )
                for row in warm_rows
            ],
        )
    )
    notes = _comparison_notes(warm_rows)
    if notes:
        lines.extend(["", "## Solver Signals", ""])
        lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_notes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    notes: list[str] = []
    by_key = {
        (str(row.get("platform")), str(row.get("requested_solver"))): row
        for row in rows
    }
    cpu_auto = by_key.get(("cpu", "auto"))
    cpu_pcr = by_key.get(("cpu", "pcr_soa"))
    if cpu_auto and cpu_pcr:
        notes.append(
            "CPU warm forced `pcr_soa` is "
            f"{_ratio(cpu_pcr['kernel_sync_ms'], cpu_auto['kernel_sync_ms'])}x "
            "slower than CPU `auto`/Thomas on kernel sync."
        )
    gpu_auto = by_key.get(("gpu", "auto"))
    gpu_thomas = by_key.get(("gpu", "thomas"))
    if gpu_auto and gpu_thomas:
        notes.append(
            "GPU warm forced `thomas` is "
            f"{_ratio(gpu_thomas['kernel_sync_ms'], gpu_auto['kernel_sync_ms'])}x "
            "slower than GPU `auto`/PCR-SoA on kernel sync."
        )
    for row in rows:
        notes.append(
            f"{row['label']} warm kernel sync accounts for "
            f"{_pct(row['kernel_sync_share'])} of `curve.simulate`."
        )
    return notes


def _print_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    print("label,temperature,platform,requested,effective,simulate_ms,kernel_sync_ms,share")
    for row in rows:
        print(
            f"{row['label']},{row['temperature']},{row['platform']},"
            f"{row['requested_solver']},{row['effective_solver']},"
            f"{float(row['simulate_ms']):.3f},{float(row['kernel_sync_ms']):.3f},"
            f"{float(row['kernel_sync_share']):.3f}"
        )


def _path_label(path: Path) -> str:
    if path.name not in {"extracted", "outputs"}:
        return path.name
    for parent in path.parents:
        if parent.name not in {"extracted", "outputs"}:
            return parent.name
    return path.name


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    parsed = _float(value)
    return None if parsed is None else int(parsed)


def _ratio(numerator: Any, denominator: Any) -> str:
    den = float(denominator or 0.0)
    if den == 0.0:
        return "inf"
    return f"{float(numerator or 0.0) / den:.1f}"


def _fmt(value: Any) -> str:
    return f"{float(value or 0.0):.1f}"


def _fmt_optional(value: Any) -> str:
    parsed = _float(value)
    return "" if parsed is None else _fmt(parsed)


def _pct(value: Any) -> str:
    return f"{float(value or 0.0) * 100.0:.1f}%"


def _markdown_table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> list[str]:
    result = ["| " + " | ".join(str(header) for header in headers) + " |"]
    result.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        result.append(
            "| "
            + " | ".join(str(value).replace("|", "\\|") for value in row)
            + " |"
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
