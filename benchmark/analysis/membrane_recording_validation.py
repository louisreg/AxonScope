"""Compare CPU and GPU membrane-recording validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.membrane_recording_validation import TOLERANCES


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
                rtol, atol = TOLERANCES[group]
                cpu_values = np.asarray(cpu[key])
                gpu_values = np.asarray(gpu[key])
                same_shape = cpu_values.shape == gpu_values.shape
                difference = np.abs(cpu_values - gpu_values) if same_shape else np.asarray([np.inf])
                passed = bool(
                    same_shape
                    and np.allclose(cpu_values, gpu_values, rtol=rtol, atol=atol, equal_nan=True)
                )
                comparisons.append({
                    "tensor": key,
                    "shape": list(cpu_values.shape),
                    "rmse": float(np.sqrt(np.mean(difference**2))),
                    "max_abs": float(np.max(difference)),
                    "rtol": rtol,
                    "atol": atol,
                    "status": "pass" if passed else "fail",
                })
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
