from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from benchmark.protocols import recruitment_amplitude_batch


def test_triton_cache_replay_uses_two_fresh_processes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = recruitment_amplitude_batch.build_parser().parse_args(
        [
            "--platform",
            "gpu",
            "--fibers-per-family",
            "2",
            "--triton-cache-replay",
            "--validate-double-cable-kernel",
            "--output",
            str(tmp_path),
        ]
    )
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, *, cwd, env, check):
        del cwd, check
        command = list(command)
        calls.append((command, dict(env)))
        child_output = Path(command[command.index("--output") + 1])
        child_output.mkdir(parents=True)
        status = "miss" if len(calls) == 1 else "hit"
        phase = {
            "trace_s": 1.0,
            "lower_s": 4.0 if status == "miss" else 0.5,
            "compile_s": 0.8,
            "first_execution_s": 0.1,
            "total_cold_s": 5.9 if status == "miss" else 2.4,
            "stablehlo_bytes": 100,
            "stablehlo_lines": 10,
            "stablehlo_custom_calls": 1,
            "triton_kernel_cache": {"status": status, "key": "same"},
        }
        (child_output / "double_cable_jit_phases.json").write_text(
            json.dumps(phase),
            encoding="utf-8",
        )
        with (child_output / "runs.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("activation_counts", "wall_ms"),
            )
            writer.writeheader()
            writer.writerow({"activation_counts": "0 1 2", "wall_ms": "7000"})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(recruitment_amplitude_batch.subprocess, "run", fake_run)

    assert recruitment_amplitude_batch._run_triton_cache_replay(args, tmp_path) == 0

    assert len(calls) == 2
    expected_cache = str(tmp_path / "triton_kernel_cache")
    assert {env["AXONSCOPE_TRITON_KERNEL_CACHE"] for _, env in calls} == {
        expected_cache
    }
    assert all("--cold-only" in command for command, _ in calls)
    assert all(
        command[command.index("--policies") + 1] == "full"
        for command, _ in calls
    )
    assert "--validate-double-cable-kernel" not in calls[0][0]
    assert "--validate-double-cable-kernel" in calls[1][0]

    replay = json.loads((tmp_path / "triton_cache_replay.json").read_text())
    assert replay["activation_counts_match"] is True
    assert replay["lower_speedup"] == 8.0
    assert [
        process["triton_kernel_cache"]["status"]
        for process in replay["processes"]
    ] == ["miss", "hit"]
