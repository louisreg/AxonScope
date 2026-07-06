from __future__ import annotations

import csv
from pathlib import Path

from benchmark.analysis.time_chunk_matrix_report import main as run_matrix_report


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
    "kernel_prepare_inputs_ms",
    "repeat_kernel_prepare_inputs_ms",
    "kernel_prepare_inputs_self_ms",
    "repeat_kernel_prepare_inputs_self_ms",
    "kernel_combine_observer_chunks_ms",
    "repeat_kernel_combine_observer_chunks_ms",
    "kernel_dispatch_jax_ms",
    "repeat_kernel_dispatch_jax_ms",
    "kernel_dispatch_jax_count",
    "kernel_wait_ms",
    "repeat_kernel_wait_ms",
    "kernel_finalize_observer_ms",
    "repeat_kernel_finalize_observer_ms",
    "results_split_batch_ms",
    "repeat_results_split_batch_ms",
    "results_materialize_vm_ms",
    "repeat_results_materialize_vm_ms",
    "results_assemble_rows_ms",
    "repeat_results_assemble_rows_ms",
    "results_assemble_cohort_record_ms",
    "repeat_results_assemble_cohort_record_ms",
    "rss_end_mib_max",
)


def test_time_chunk_matrix_report_writes_best_rows(tmp_path: Path):
    root = tmp_path / "campaign"
    output = tmp_path / "report"
    _write_campaign(root)

    assert (
        run_matrix_report(
            [
                "--run",
                f"sample={root}",
                "--output",
                str(output),
                "--no-plots",
            ]
        )
        == 0
    )

    rows = list(csv.DictReader((output / "time_chunk_matrix_rows.csv").open()))
    best = list(csv.DictReader((output / "time_chunk_best_rows.csv").open()))

    assert len(rows) == 2
    assert best[0]["policy"] == "1000"
    assert best[0]["curve_s"] == "1.5"
    assert best[0]["peak_rss_end_mib"] == "222.0"
    assert "sample" in (output / "time_chunk_matrix_report.md").read_text(encoding="utf-8")


def _write_campaign(root: Path) -> None:
    root.mkdir(parents=True)
    with (root / "time_chunk_sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(_summary_row("default", 2000.0))
        writer.writerow(_summary_row("1000", 1500.0))

    _write_memory(root / "full_vm" / "default" / "memory_summary.csv", rss=111.0)
    _write_memory(root / "full_vm" / "chunk_1000" / "memory_summary.csv", rss=222.0)


def _summary_row(policy: str, curve_ms: float) -> dict[str, object]:
    return {
        "policy": policy,
        "run_dir": "",
        "status": "passed",
        "returncode": 0,
        "case_name": "case",
        "script": "threshold_curves",
        "platform": "cpu",
        "recording": "full_vm",
        "n_axons": 2,
        "nx": 11,
        "tsim": 1.0,
        "dt": 0.1,
        "precision": "fp32",
        "time_chunk_policy": "explicit" if policy == "1000" else "default",
        "time_chunk_steps": "" if policy == "default" else 1000,
        "dispatch_time_chunk_steps": "",
        "observer_state_scopes": "",
        "dispatch_chunk_steps": 1000,
        "dispatch_chunk_count_max": 1,
        "curve_simulate_ms": curve_ms,
        "repeat_curve_simulate_ms": curve_ms,
        "kernel_enqueue_ms": 900.0,
        "repeat_kernel_enqueue_ms": 900.0,
        "kernel_prepare_inputs_ms": 20.0,
        "repeat_kernel_prepare_inputs_ms": 20.0,
        "kernel_prepare_inputs_self_ms": 15.0,
        "repeat_kernel_prepare_inputs_self_ms": 15.0,
        "kernel_combine_observer_chunks_ms": 0.0,
        "repeat_kernel_combine_observer_chunks_ms": 0.0,
        "kernel_dispatch_jax_ms": 300.0,
        "repeat_kernel_dispatch_jax_ms": 300.0,
        "kernel_dispatch_jax_count": 1,
        "kernel_wait_ms": 400.0,
        "repeat_kernel_wait_ms": 400.0,
        "kernel_finalize_observer_ms": 0.0,
        "repeat_kernel_finalize_observer_ms": 0.0,
        "results_split_batch_ms": 100.0,
        "repeat_results_split_batch_ms": 100.0,
        "results_materialize_vm_ms": 10.0,
        "repeat_results_materialize_vm_ms": 10.0,
        "results_assemble_rows_ms": 80.0,
        "repeat_results_assemble_rows_ms": 80.0,
        "results_assemble_cohort_record_ms": 0.0,
        "repeat_results_assemble_cohort_record_ms": 0.0,
        "rss_end_mib_max": "",
    }


def _write_memory(path: Path, *, rss: float) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "name",
                "rss_end_mib_max",
                "rss_delta_mib_max",
                "device_bytes_in_use_end_max",
                "device_bytes_in_use_delta_max",
                "nvidia_smi_memory_used_end_mib_max",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "curve.simulate",
                "rss_end_mib_max": rss,
                "rss_delta_mib_max": 10.0,
                "device_bytes_in_use_end_max": 0,
                "device_bytes_in_use_delta_max": 0,
                "nvidia_smi_memory_used_end_mib_max": "",
            }
        )
