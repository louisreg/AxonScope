"""Compare CPU and single-GPU evidence from runner_plan_validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu", type=Path, help="CPU validation.json or its result directory.")
    parser.add_argument("gpu", type=Path, help="GPU validation.json or its result directory.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol-mv", type=float, default=1e-3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cpu_path = _validation_path(args.cpu)
    gpu_path = _validation_path(args.gpu)
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    output = args.output or gpu_path.parent / "cpu_gpu_comparison"
    output.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    _check(checks, "cpu campaign passed", cpu.get("status") == "passed")
    _check(checks, "gpu campaign passed", gpu.get("status") == "passed")
    _check(checks, "CPU platform", cpu.get("platform") == "cpu")
    _check(checks, "GPU platform", gpu.get("platform") == "gpu")
    _check(checks, "same plan", cpu.get("plan") == gpu.get("plan"))
    _check(checks, "same scales", cpu.get("scales") == gpu.get("scales"))
    for name in ("cache_clear", "cache_invalidation", "cancellation"):
        _check(checks, f"CPU {name}", bool(cpu.get(name, {}).get("passed")))
        _check(checks, f"GPU {name}", bool(gpu.get(name, {}).get("passed")))

    cpu_signature = _run_signature(cpu, "study_cold")
    gpu_signature = _run_signature(gpu, "study_cold")
    _check(
        checks,
        "study task order",
        cpu_signature.get("keys") == gpu_signature.get("keys"),
    )
    for name in ("numeric_axis_activation", "sweep_activation", "threshold"):
        _check(
            checks,
            f"study {name}",
            cpu_signature.get(name) == gpu_signature.get(name),
        )
    _compare_voltage_rows(
        checks,
        "simple voltage",
        cpu_signature.get("simple_voltage", ()),
        gpu_signature.get("simple_voltage", ()),
        rtol=args.rtol,
        atol=args.atol_mv,
    )
    _compare_voltage_rows(
        checks,
        "mixed voltage",
        cpu_signature.get("mixed_voltage", ()),
        gpu_signature.get("mixed_voltage", ()),
        rtol=args.rtol,
        atol=args.atol_mv,
    )
    _compare_scales(checks, cpu, gpu)

    performance = _performance_rows(cpu, gpu)
    comparison = {
        "schema": "axonfleet.runner_plan_validation_comparison.v1",
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "cpu": str(cpu_path),
        "gpu": str(gpu_path),
        "rtol": args.rtol,
        "atol_mV": args.atol_mv,
        "checks": checks,
        "performance": performance,
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output, comparison)
    print(f"runner plan CPU/GPU comparison {comparison['status']}: {output}")
    return 0 if comparison["status"] == "passed" else 1


def _validation_path(path: Path) -> Path:
    resolved = path / "validation.json" if path.is_dir() else path
    if not resolved.exists():
        raise SystemExit(f"validation result does not exist: {resolved}")
    return resolved


def _check(checks: list[dict[str, Any]], name: str, passed: bool) -> None:
    checks.append({"name": name, "passed": bool(passed)})


def _run_signature(validation: dict[str, Any], label: str) -> dict[str, Any]:
    for run in validation.get("runs", ()):
        if run.get("label") == label:
            return dict(run["signature"])
    raise KeyError(f"validation has no run {label!r}")


def _compare_voltage_rows(
    checks: list[dict[str, Any]],
    label: str,
    cpu_rows: Any,
    gpu_rows: Any,
    *,
    rtol: float,
    atol: float,
) -> None:
    _check(checks, f"{label} row count", len(cpu_rows) == len(gpu_rows))
    if len(cpu_rows) != len(gpu_rows):
        return
    numeric_fields = ("minimum_mV", "maximum_mV", "final_mV", "sum_mV")
    for index, (cpu, gpu) in enumerate(zip(cpu_rows, gpu_rows, strict=True)):
        _check(checks, f"{label} row {index} shape", cpu["shape"] == gpu["shape"])
        _check(
            checks,
            f"{label} row {index} values",
            bool(
                np.allclose(
                    [cpu[field] for field in numeric_fields],
                    [gpu[field] for field in numeric_fields],
                    rtol=rtol,
                    atol=atol,
                )
            ),
        )


def _compare_scales(
    checks: list[dict[str, Any]],
    cpu: dict[str, Any],
    gpu: dict[str, Any],
) -> None:
    cpu_rows = {int(item["n_axons"]): item for item in cpu.get("scale_validation", ())}
    gpu_rows = {int(item["n_axons"]): item for item in gpu.get("scale_validation", ())}
    _check(checks, "scale case set", cpu_rows.keys() == gpu_rows.keys())
    for size in sorted(cpu_rows.keys() & gpu_rows.keys()):
        _check(checks, f"CPU scale {size}", bool(cpu_rows[size].get("passed")))
        _check(checks, f"GPU scale {size}", bool(gpu_rows[size].get("passed")))
        cpu_signature = dict(cpu_rows[size]["cold"]["signature"])
        gpu_signature = dict(gpu_rows[size]["cold"]["signature"])
        for field in ("rows", "activated_count", "first", "last"):
            _check(
                checks,
                f"scale {size} {field}",
                cpu_signature.get(field) == gpu_signature.get(field),
            )


def _performance_rows(cpu: dict[str, Any], gpu: dict[str, Any]) -> list[dict[str, Any]]:
    cpu_runs = _all_runs(cpu)
    gpu_runs = _all_runs(gpu)
    rows = []
    for label in sorted(cpu_runs.keys() & gpu_runs.keys()):
        cpu_ms = float(cpu_runs[label]["wall_ms"])
        gpu_ms = float(gpu_runs[label]["wall_ms"])
        rows.append(
            {
                "label": label,
                "cpu_wall_ms": cpu_ms,
                "gpu_wall_ms": gpu_ms,
                "gpu_speedup": cpu_ms / gpu_ms if gpu_ms > 0.0 else None,
                "cpu_stages_ms": cpu_runs[label]["stages_ms"],
                "gpu_stages_ms": gpu_runs[label]["stages_ms"],
            }
        )
    return rows


def _all_runs(validation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = {str(item["label"]): item for item in validation.get("runs", ())}
    for scale in validation.get("scale_validation", ()):
        for phase in ("cold", "warm"):
            item = scale[phase]
            runs[str(item["label"])] = item
    return runs


def _write_report(output: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# Runner Plan CPU/GPU Comparison",
        "",
        f"- status: `{comparison['status']}`",
        f"- tolerance: `rtol={comparison['rtol']}`, `atol={comparison['atol_mV']} mV`",
        "",
        "## Gates",
        "",
    ]
    for check in comparison["checks"]:
        lines.append(f"- [{'x' if check['passed'] else ' '}] {check['name']}")
    lines.extend(
        (
            "",
            "## Timing",
            "",
            "| run | CPU ms | GPU ms | GPU speedup |",
            "| --- | ---: | ---: | ---: |",
        )
    )
    for row in comparison["performance"]:
        speedup = row["gpu_speedup"]
        speedup_text = "" if speedup is None else f"{speedup:.3f}x"
        lines.append(
            f"| {row['label']} | {row['cpu_wall_ms']:.3f} | "
            f"{row['gpu_wall_ms']:.3f} | {speedup_text} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
