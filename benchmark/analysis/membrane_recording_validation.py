"""Compare CPU and GPU membrane-recording validation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.membrane_recording_validation import (
    NORMALIZED_TOLERANCES,
    TOLERANCES,
)


def compare_tensor(cpu_values: np.ndarray, gpu_values: np.ndarray, *, group: str) -> dict:
    """Compare one tensor using strict and trajectory-normalized criteria."""

    same_shape = cpu_values.shape == gpu_values.shape
    if not same_shape:
        return {
            "shape": list(cpu_values.shape),
            "gpu_shape": list(gpu_values.shape),
            "status": "fail",
            "pointwise_status": "fail",
            "trajectory_status": "fail",
        }

    difference = np.abs(cpu_values - gpu_values)
    finite = bool(np.all(np.isfinite(cpu_values)) and np.all(np.isfinite(gpu_values)))
    rtol, atol = TOLERANCES[group]
    pointwise = bool(
        finite
        and np.allclose(cpu_values, gpu_values, rtol=rtol, atol=atol, equal_nan=True)
    )
    scale = max(float(np.max(np.abs(cpu_values))), float(np.ptp(cpu_values)), 1e-12)
    rmse = float(np.sqrt(np.mean(difference**2)))
    max_abs = float(np.max(difference))
    nrmse = rmse / scale
    normalized_max = max_abs / scale
    nrmse_limit, normalized_max_limit = NORMALIZED_TOLERANCES[group]
    trajectory = bool(
        finite
        and nrmse <= nrmse_limit
        and normalized_max <= normalized_max_limit
    )
    return {
        "shape": list(cpu_values.shape),
        "rmse": rmse,
        "max_abs": max_abs,
        "scale": scale,
        "normalized_rmse": nrmse,
        "normalized_max_abs": normalized_max,
        "rtol": rtol,
        "atol": atol,
        "normalized_rmse_limit": nrmse_limit,
        "normalized_max_abs_limit": normalized_max_limit,
        "pointwise_status": "pass" if pointwise else "fail",
        "trajectory_status": "pass" if trajectory else "fail",
        "status": "pass" if pointwise or trajectory else "fail",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu", type=Path)
    parser.add_argument("gpu", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    failed = False
    for cpu_path in sorted(args.cpu.glob("*.npz")):
        gpu_path = args.gpu / cpu_path.name
        if not gpu_path.is_file():
            rows.append({"case": cpu_path.stem, "status": "fail", "reason": "GPU artifact missing"})
            failed = True
            continue
        comparisons = []
        with np.load(cpu_path) as cpu, np.load(gpu_path) as gpu:
            if set(cpu.files) != set(gpu.files):
                rows.append({
                    "case": cpu_path.stem,
                    "status": "fail",
                    "reason": "recorded tensor names differ",
                    "cpu_only": sorted(set(cpu.files) - set(gpu.files)),
                    "gpu_only": sorted(set(gpu.files) - set(cpu.files)),
                })
                failed = True
                continue
            case_failed = False
            for key in sorted(cpu.files):
                group = key.split(".", 1)[0]
                cpu_values = np.asarray(cpu[key])
                gpu_values = np.asarray(gpu[key])
                comparison = compare_tensor(cpu_values, gpu_values, group=group)
                comparison["tensor"] = key
                comparisons.append(comparison)
                passed = comparison["status"] == "pass"
                case_failed |= not passed
            rows.append({
                "case": cpu_path.stem,
                "status": "fail" if case_failed else "pass",
                "comparisons": comparisons,
            })
            failed |= case_failed

    payload = {"status": "fail" if failed else "pass", "rows": rows}
    path = args.output / "comparison.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"CPU/GPU recording comparison: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
