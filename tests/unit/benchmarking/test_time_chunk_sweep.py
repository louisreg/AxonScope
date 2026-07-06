from __future__ import annotations

import json
from pathlib import Path

from benchmark.campaigns.time_chunk_sweep import main as run_time_chunk_sweep
from benchmark.campaigns.time_chunk_sweep import summarize_run


def test_time_chunk_sweep_dry_run_writes_manifest(tmp_path: Path, capsys):
    assert (
        run_time_chunk_sweep(
            [
                "--script",
                "recruitment_curves",
                "--preset",
                "quick",
                "--platform",
                "cpu",
                "--policies",
                "default,unchunked,100",
                "--output",
                str(tmp_path),
                "--dry-run",
                "--recording",
                "observer_only",
                "--amplitude-count",
                "1",
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    manifest = json.loads(
        (tmp_path / "time_chunk_sweep_manifest.json").read_text(encoding="utf-8")
    )

    assert "benchmark/run.py" in out
    assert manifest["policies"] == ["default", "unchunked", "100"]
    assert manifest["recordings"] == ["observer_only"]
    assert [Path(run["run_dir"]).name for run in manifest["runs"]] == [
        "default",
        "unchunked",
        "chunk_100",
    ]
    commands = [" ".join(run["command"]) for run in manifest["runs"]]
    assert "--time-chunk-steps default" in commands[0]
    assert "--time-chunk-steps unchunked" in commands[1]
    assert "--time-chunk-steps 100" in commands[2]
    assert all("--recording observer_only" in command for command in commands)
    assert all("--amplitude-count 1" in command for command in commands)


def test_time_chunk_sweep_dry_run_builds_recording_matrix(tmp_path: Path, capsys):
    assert (
        run_time_chunk_sweep(
            [
                "--script",
                "recruitment_curves",
                "--preset",
                "quick",
                "--platform",
                "cpu",
                "--policies",
                "default,100",
                "--recordings",
                "full_vm,probe_vm",
                "--output",
                str(tmp_path),
                "--dry-run",
                "--amplitude-count",
                "1",
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    manifest = json.loads(
        (tmp_path / "time_chunk_sweep_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["policies"] == ["default", "100"]
    assert manifest["recordings"] == ["full_vm", "probe_vm"]
    rel_dirs = [
        str(Path(run["run_dir"]).relative_to(tmp_path))
        for run in manifest["runs"]
    ]
    assert rel_dirs == [
        "full_vm/default",
        "full_vm/chunk_100",
        "probe_vm/default",
        "probe_vm/chunk_100",
    ]
    commands = [" ".join(run["command"]) for run in manifest["runs"]]
    assert all("--amplitude-count 1" in command for command in commands)
    assert out.count("benchmark/run.py") == 4
    assert sum("--recording full_vm" in command for command in commands) == 2
    assert sum("--recording probe_vm" in command for command in commands) == 2


def test_time_chunk_sweep_maps_amplitude_count_for_threshold_curves(
    tmp_path: Path,
    capsys,
):
    assert (
        run_time_chunk_sweep(
            [
                "--script",
                "threshold_curves",
                "--preset",
                "quick",
                "--platform",
                "cpu",
                "--policies",
                "default",
                "--recording",
                "full_vm",
                "--output",
                str(tmp_path),
                "--dry-run",
                "--amplitude-count",
                "5",
            ]
        )
        == 0
    )

    capsys.readouterr()
    manifest = json.loads(
        (tmp_path / "time_chunk_sweep_manifest.json").read_text(encoding="utf-8")
    )
    command = " ".join(manifest["runs"][0]["command"])
    assert "--amplitude-count" not in command
    assert "--max-iterations 5" in command


def test_time_chunk_sweep_summarizes_kernel_events(tmp_path: Path):
    manifest = {
        "script": "recruitment_curves",
        "case_name": "case",
        "options": {
            "platform": "cpu",
            "recording": "observer_only",
            "n_axons": 1,
            "nx": 21,
            "tsim": 2.0,
            "dt": 0.02,
            "precision": "fp32",
            "time_chunk_policy": "explicit",
            "time_chunk_steps": 100,
        },
    }
    events = [
        {
            "event_id": 1,
            "parent_event_id": None,
            "name": "curve.simulate",
            "duration_ms": 100.0,
            "metadata": {"phase": "repeat"},
        },
        {
            "event_id": 2,
            "parent_event_id": 1,
            "name": "dispatch.group.total",
            "duration_ms": 90.0,
            "metadata": {},
        },
        {
            "event_id": 3,
            "parent_event_id": 2,
            "name": "kernel.enqueue",
            "duration_ms": 60.0,
            "metadata": {},
        },
        {
            "event_id": 4,
            "parent_event_id": 3,
            "name": "kernel.dispatch_jax",
            "duration_ms": 20.0,
            "metadata": {
                "chunk_count": 1,
                "chunk_steps": 100,
                "observer_state_scope": "chunk",
                "time_chunk_steps": 100,
                "memory": {"rss_end_mib": 321.0},
            },
        },
        {
            "event_id": 12,
            "parent_event_id": 3,
            "name": "kernel.prepare_arrays",
            "duration_ms": 11.0,
            "metadata": {},
        },
        {
            "event_id": 13,
            "parent_event_id": 3,
            "name": "kernel.prepare_state",
            "duration_ms": 1.0,
            "metadata": {},
        },
        {
            "event_id": 14,
            "parent_event_id": 3,
            "name": "kernel.prepare_observer_tables",
            "duration_ms": 1.5,
            "metadata": {},
        },
        {
            "event_id": 23,
            "parent_event_id": 3,
            "name": "kernel.prepare_inputs",
            "duration_ms": 6.0,
            "metadata": {},
        },
        {
            "event_id": 15,
            "parent_event_id": 23,
            "name": "kernel.materialize_inputs",
            "duration_ms": 2.5,
            "metadata": {},
        },
        {
            "event_id": 16,
            "parent_event_id": 3,
            "name": "kernel.prepare_factorized_forcing",
            "duration_ms": 0.5,
            "metadata": {},
        },
        {
            "event_id": 17,
            "parent_event_id": 3,
            "name": "kernel.chunk_setup",
            "duration_ms": 3.0,
            "metadata": {},
        },
        {
            "event_id": 5,
            "parent_event_id": 3,
            "name": "kernel.combine_observer_chunks",
            "duration_ms": 7.0,
            "metadata": {},
        },
        {
            "event_id": 18,
            "parent_event_id": 3,
            "name": "kernel.chunk_bookkeeping",
            "duration_ms": 4.0,
            "metadata": {},
        },
        {
            "event_id": 19,
            "parent_event_id": 3,
            "name": "kernel.concat_trace_chunks",
            "duration_ms": 5.0,
            "metadata": {},
        },
        {
            "event_id": 6,
            "parent_event_id": 2,
            "name": "kernel.wait",
            "duration_ms": 5.0,
            "metadata": {},
        },
        {
            "event_id": 7,
            "parent_event_id": 3,
            "name": "kernel.finalize_observer",
            "duration_ms": 2.0,
            "metadata": {},
        },
        {
            "event_id": 20,
            "parent_event_id": 7,
            "name": "kernel.finalize_observer.to_host",
            "duration_ms": 1.2,
            "metadata": {},
        },
        {
            "event_id": 8,
            "parent_event_id": 2,
            "name": "results.split_batch",
            "duration_ms": 9.0,
            "metadata": {},
        },
        {
            "event_id": 21,
            "parent_event_id": 8,
            "name": "results.trim_padded_batch",
            "duration_ms": 0.7,
            "metadata": {},
        },
        {
            "event_id": 9,
            "parent_event_id": 8,
            "name": "results.materialize_vm",
            "duration_ms": 3.0,
            "metadata": {},
        },
        {
            "event_id": 22,
            "parent_event_id": 9,
            "name": "results.materialize_vm.to_host",
            "duration_ms": 2.0,
            "metadata": {},
        },
        {
            "event_id": 10,
            "parent_event_id": 8,
            "name": "results.assemble_rows",
            "duration_ms": 4.0,
            "metadata": {},
        },
        {
            "event_id": 11,
            "parent_event_id": 8,
            "name": "results.assemble_cohort_record",
            "duration_ms": 1.0,
            "metadata": {},
        },
        {
            "event_id": 30,
            "parent_event_id": None,
            "name": "curve.build_pool",
            "duration_ms": 40.0,
            "metadata": {"phase": "repeat"},
        },
        {
            "event_id": 31,
            "parent_event_id": 30,
            "name": "curve.build_pool.diameter_grid",
            "duration_ms": 1.0,
            "metadata": {},
        },
        {
            "event_id": 32,
            "parent_event_id": 30,
            "name": "curve.build_pool.spatial_layout",
            "duration_ms": 2.0,
            "metadata": {},
        },
        {
            "event_id": 33,
            "parent_event_id": 30,
            "name": "curve.build_pool.rows",
            "duration_ms": 30.0,
            "metadata": {},
        },
        {
            "event_id": 34,
            "parent_event_id": 33,
            "name": "curve.build_pool.template_build",
            "duration_ms": 5.0,
            "metadata": {},
        },
        {
            "event_id": 35,
            "parent_event_id": None,
            "name": "curve.construct_simulation",
            "duration_ms": 1.5,
            "metadata": {"phase": "repeat"},
        },
        {
            "event_id": 36,
            "parent_event_id": None,
            "name": "curve.analyze_activation",
            "duration_ms": 8.0,
            "metadata": {"phase": "repeat"},
        },
        {
            "event_id": 37,
            "parent_event_id": 36,
            "name": "curve.analyze_activation.result_analyze",
            "duration_ms": 6.0,
            "metadata": {},
        },
        {
            "event_id": 38,
            "parent_event_id": 36,
            "name": "curve.analyze_activation.materialize_values",
            "duration_ms": 1.0,
            "metadata": {},
        },
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    row = summarize_run(tmp_path, policy="100")

    assert row["policy"] == "100"
    assert row["time_chunk_policy"] == "explicit"
    assert row["time_chunk_steps"] == 100
    assert row["dispatch_time_chunk_steps"] == "100"
    assert row["observer_state_scopes"] == "chunk"
    assert row["dispatch_chunk_steps"] == "100"
    assert row["dispatch_chunk_count_max"] == "1"
    assert row["repeat_curve_build_pool_ms"] == 40.0
    assert row["repeat_curve_build_pool_self_ms"] == 7.0
    assert row["repeat_curve_build_pool_diameter_grid_ms"] == 1.0
    assert row["repeat_curve_build_pool_spatial_layout_ms"] == 2.0
    assert row["repeat_curve_build_pool_rows_ms"] == 30.0
    assert row["repeat_curve_build_pool_template_build_ms"] == 5.0
    assert row["repeat_curve_construct_simulation_ms"] == 1.5
    assert row["repeat_curve_simulate_ms"] == 100.0
    assert row["repeat_curve_analyze_activation_ms"] == 8.0
    assert row["repeat_curve_analyze_activation_self_ms"] == 1.0
    assert row["repeat_curve_analyze_activation_result_analyze_ms"] == 6.0
    assert row["repeat_curve_analyze_activation_materialize_values_ms"] == 1.0
    assert row["repeat_kernel_enqueue_ms"] == 60.0
    assert row["repeat_kernel_prepare_inputs_ms"] == 6.0
    assert row["repeat_kernel_prepare_inputs_self_ms"] == 3.5
    assert row["repeat_kernel_prepare_arrays_ms"] == 11.0
    assert row["repeat_kernel_prepare_state_ms"] == 1.0
    assert row["repeat_kernel_prepare_observer_tables_ms"] == 1.5
    assert row["repeat_kernel_materialize_inputs_ms"] == 2.5
    assert row["repeat_kernel_prepare_factorized_forcing_ms"] == 0.5
    assert row["repeat_kernel_chunk_setup_ms"] == 3.0
    assert row["repeat_kernel_combine_observer_chunks_ms"] == 7.0
    assert row["repeat_kernel_dispatch_jax_ms"] == 20.0
    assert row["repeat_kernel_chunk_bookkeeping_ms"] == 4.0
    assert row["repeat_kernel_concat_trace_chunks_ms"] == 5.0
    assert row["repeat_kernel_wait_ms"] == 5.0
    assert row["repeat_kernel_finalize_observer_ms"] == 2.0
    assert row["repeat_kernel_finalize_observer_to_host_ms"] == 1.2
    assert row["repeat_results_split_batch_ms"] == 9.0
    assert row["repeat_results_trim_padded_batch_ms"] == 0.7
    assert row["repeat_results_materialize_vm_ms"] == 3.0
    assert row["repeat_results_materialize_vm_to_host_ms"] == 2.0
    assert row["repeat_results_assemble_rows_ms"] == 4.0
    assert row["repeat_results_assemble_cohort_record_ms"] == 1.0
    assert row["rss_end_mib_max"] == "321"
