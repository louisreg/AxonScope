from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from benchmark import membrane_temporal


def test_membrane_temporal_replay_uses_fresh_processes_and_dynamic_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        output=tmp_path,
        platform="cpu",
        preset="quick",
        dt_ms=0.005,
        v_init_mv=-70.0,
    )
    case = membrane_temporal.Case("nav16", "single", "uniform", 1)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, *, cwd, env, check, stdout, stderr):
        del cwd, check, stdout, stderr
        command = list(command)
        calls.append((command, dict(env)))
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        v_init = float(command[command.index("--v-init-mv") + 1])
        checksum = "same-result" if v_init == -70.0 else "dynamic-result"
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "v_init_mV": v_init,
                    "rows": [
                        {
                            "result_checksum": checksum,
                            "cold_run_ms": 1000.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        phase_dir = output / "jax_phase_capture"
        phase_dir.mkdir()
        (phase_dir / "single.jit_phases.json").write_text(
            json.dumps(
                {
                    "trace_s": 0.1,
                    "lower_s": 0.2,
                    "compile_s": 1.0 if len(calls) == 1 else 0.1,
                    "first_execution_s": 0.05,
                    "total_cold_s": 1.35 if len(calls) == 1 else 0.45,
                    "stablehlo": {"sha256": "stable", "bytes": 100},
                    "optimized_hlo": {"sha256": "optimized", "bytes": 200},
                    "dynamic": {"Vm0_mV": [{"dtype": "float32", "shape": [1, 41]}]},
                    "static": {"record_full": False},
                }
            ),
            encoding="utf-8",
        )
        (phase_dir / "single.compiled.optimized_hlo.txt").write_text(
            "HloModule fake\n",
            encoding="utf-8",
        )
        cache = Path(env["AXONFLEET_CACHE"]) / "runtime" / "jax" / "xla"
        cache.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            (cache / "jit-cache").write_text("cached", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(membrane_temporal.subprocess, "run", fake_run)
    monkeypatch.setattr(
        membrane_temporal,
        "_write_case_hlo_analysis",
        lambda *args, **kwargs: None,
    )

    result = membrane_temporal._run_compilation_cache_replay(
        args,
        cases=[case],
        single_compartments=41,
        nodes=5,
        duration_ms=0.02,
    )

    assert result == 0
    assert len(calls) == 3
    assert all("--capture-jit-phases" in command for command, _ in calls)
    assert all("--cold-only" in command for command, _ in calls)
    assert calls[2][0][calls[2][0].index("--v-init-mv") + 1] == "-69.5"
    cache_roots = {
        environment["AXONFLEET_CACHE"]
        for _, environment in calls
    }
    assert len(cache_roots) == 1

    payload = json.loads((tmp_path / "compilation_cache_replay.json").read_text())
    record = payload["cases"][0]
    assert payload["accepted"] is True
    assert record["exact_result_match"] is True
    assert record["dynamic_stablehlo_match"] is True
    assert record["fresh_process_cache_reused"] is True
    assert record["compile_speedup"] == 10.0
