"""Run a reproducible time-chunk policy sweep for runtime optimization triage."""

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
DEFAULT_POLICIES = ("default", "unchunked", "128", "256", "512", "1024")
RECORDING_MODES = ("full_vm", "probe_vm", "observer_only")
DEFAULT_CABLES = ("single_cable", "double_cable")

SUMMARY_FIELDS = (
    "policy",
    "run_dir",
    "status",
    "returncode",
    "case_name",
    "script",
    "platform",
    "recording",
    "cable",
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
    "curve_build_pool_ms",
    "repeat_curve_build_pool_ms",
    "curve_build_pool_self_ms",
    "repeat_curve_build_pool_self_ms",
    "curve_build_pool_diameter_grid_ms",
    "repeat_curve_build_pool_diameter_grid_ms",
    "curve_build_pool_spatial_layout_ms",
    "repeat_curve_build_pool_spatial_layout_ms",
    "curve_build_pool_rows_ms",
    "repeat_curve_build_pool_rows_ms",
    "curve_build_pool_template_build_ms",
    "repeat_curve_build_pool_template_build_ms",
    "curve_update_amplitudes_ms",
    "repeat_curve_update_amplitudes_ms",
    "curve_update_amplitudes_rows_ms",
    "repeat_curve_update_amplitudes_rows_ms",
    "curve_update_amplitudes_stimulus_build_ms",
    "repeat_curve_update_amplitudes_stimulus_build_ms",
    "curve_activation_definition_ms",
    "repeat_curve_activation_definition_ms",
    "curve_runtime_options_ms",
    "repeat_curve_runtime_options_ms",
    "curve_construct_simulation_ms",
    "repeat_curve_construct_simulation_ms",
    "curve_simulate_ms",
    "repeat_curve_simulate_ms",
    "curve_analyze_activation_ms",
    "repeat_curve_analyze_activation_ms",
    "curve_analyze_activation_self_ms",
    "repeat_curve_analyze_activation_self_ms",
    "curve_analyze_activation_dense_values_ms",
    "repeat_curve_analyze_activation_dense_values_ms",
    "curve_analyze_activation_result_analyze_ms",
    "repeat_curve_analyze_activation_result_analyze_ms",
    "curve_analyze_activation_vm_raster_extract_ms",
    "repeat_curve_analyze_activation_vm_raster_extract_ms",
    "curve_analyze_activation_vm_raster_values_ms",
    "repeat_curve_analyze_activation_vm_raster_values_ms",
    "curve_analyze_activation_materialize_values_ms",
    "repeat_curve_analyze_activation_materialize_values_ms",
    "kernel_enqueue_ms",
    "repeat_kernel_enqueue_ms",
    "kernel_prepare_inputs_ms",
    "repeat_kernel_prepare_inputs_ms",
    "kernel_prepare_inputs_self_ms",
    "repeat_kernel_prepare_inputs_self_ms",
    "kernel_prepare_arrays_ms",
    "repeat_kernel_prepare_arrays_ms",
    "kernel_prepare_double_coefficients_ms",
    "repeat_kernel_prepare_double_coefficients_ms",
    "kernel_prepare_state_ms",
    "repeat_kernel_prepare_state_ms",
    "kernel_prepare_observer_state_ms",
    "repeat_kernel_prepare_observer_state_ms",
    "kernel_prepare_observer_tables_ms",
    "repeat_kernel_prepare_observer_tables_ms",
    "kernel_materialize_inputs_ms",
    "repeat_kernel_materialize_inputs_ms",
    "kernel_prepare_factorized_forcing_ms",
    "repeat_kernel_prepare_factorized_forcing_ms",
    "kernel_prepare_factorized_vext_ms",
    "repeat_kernel_prepare_factorized_vext_ms",
    "kernel_chunk_setup_ms",
    "repeat_kernel_chunk_setup_ms",
    "kernel_combine_observer_chunks_ms",
    "repeat_kernel_combine_observer_chunks_ms",
    "kernel_dispatch_jax_ms",
    "repeat_kernel_dispatch_jax_ms",
    "kernel_dispatch_jax_count",
    "kernel_chunk_bookkeeping_ms",
    "repeat_kernel_chunk_bookkeeping_ms",
    "kernel_concat_trace_chunks_ms",
    "repeat_kernel_concat_trace_chunks_ms",
    "kernel_wait_ms",
    "repeat_kernel_wait_ms",
    "kernel_finalize_observer_ms",
    "repeat_kernel_finalize_observer_ms",
    "kernel_finalize_observer_to_host_ms",
    "repeat_kernel_finalize_observer_to_host_ms",
    "results_split_batch_ms",
    "repeat_results_split_batch_ms",
    "results_trim_padded_batch_ms",
    "repeat_results_trim_padded_batch_ms",
    "results_materialize_vm_ms",
    "repeat_results_materialize_vm_ms",
    "results_materialize_vm_to_host_ms",
    "repeat_results_materialize_vm_to_host_ms",
    "results_assemble_rows_ms",
    "repeat_results_assemble_rows_ms",
    "results_assemble_cohort_record_ms",
    "repeat_results_assemble_cohort_record_ms",
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
            "default,unchunked,128,256,512,1024."
        ),
    )
    parser.add_argument(
        "--recording",
        choices=RECORDING_MODES,
        help=(
            "Single recording mode to forward to the curve script. Kept for "
            "compatibility with older one-mode campaign commands."
        ),
    )
    parser.add_argument(
        "--recordings",
        action="append",
        help=(
            "Comma-separated recording modes to sweep: full_vm, probe_vm, "
            "observer_only. When omitted, the preset/script default is used "
            "unless --recording is set."
        ),
    )
    parser.add_argument(
        "--cable",
        choices=DEFAULT_CABLES,
        help=(
            "Single cable formulation to forward to the curve script. Kept for "
            "compatibility with older one-cable campaign commands."
        ),
    )
    parser.add_argument(
        "--cables",
        action="append",
        help="Comma-separated cable formulations to sweep: single_cable,double_cable.",
    )
    parser.add_argument(
        "--n-axons",
        action="append",
        help=(
            "Comma-separated Naxon values to sweep. When omitted, the concrete "
            "curve preset/default is used."
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
        recordings = parse_recordings(args.recordings, args.recording)
        cables = parse_cables(args.cables, args.cable)
        n_axons_values = parse_n_axons(args.n_axons)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    runs = build_runs(args, policies, recordings, cables, n_axons_values, extra)
    write_manifest(
        args.output / "time_chunk_sweep_manifest.json",
        args,
        policies,
        recordings,
        cables,
        n_axons_values,
        runs,
        extra,
    )

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
            print(f"failed: {run_label(run)} (returncode={result.returncode})")
        else:
            print(f"passed: {run_label(run)}")
        rows.append(
            summarize_run(
                run_dir,
                policy=str(run["policy"]),
                recording=str(run.get("recording") or ""),
                cable=str(run.get("cable") or ""),
                n_axons=run.get("n_axons") or "",
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


def parse_recordings(
    values: Sequence[str] | None,
    recording: str | None,
) -> tuple[str, ...]:
    recordings: list[str] = []
    if recording:
        recordings.append(recording)
    for value in values or ():
        for item in str(value).split(","):
            mode = item.strip().lower()
            if not mode:
                continue
            if mode not in RECORDING_MODES:
                allowed = ", ".join(RECORDING_MODES)
                raise argparse.ArgumentTypeError(
                    f"recording modes must be one of: {allowed}."
                )
            if mode not in recordings:
                recordings.append(mode)
    return tuple(recordings)


def parse_cables(
    values: Sequence[str] | None,
    cable: str | None,
) -> tuple[str, ...]:
    cables: list[str] = []
    if cable:
        cables.append(cable)
    for value in values or ():
        for item in str(value).split(","):
            mode = item.strip().lower()
            if not mode:
                continue
            if mode not in DEFAULT_CABLES:
                allowed = ", ".join(DEFAULT_CABLES)
                raise argparse.ArgumentTypeError(
                    f"cable formulations must be one of: {allowed}."
                )
            if mode not in cables:
                cables.append(mode)
    return tuple(cables)


def parse_n_axons(values: Sequence[str] | None) -> tuple[int, ...]:
    parsed: list[int] = []
    for value in values or ():
        for item in str(value).split(","):
            text = item.strip()
            if not text:
                continue
            try:
                count = int(text)
            except ValueError as exc:
                raise argparse.ArgumentTypeError("--n-axons values must be integers.") from exc
            if count < 1:
                raise argparse.ArgumentTypeError("--n-axons values must be >= 1.")
            if count not in parsed:
                parsed.append(count)
    return tuple(parsed)


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
    recordings: Sequence[str],
    cables: Sequence[str],
    n_axons_values: Sequence[int],
    extra: Sequence[str],
) -> list[dict[str, Any]]:
    runs = []
    script_extra = normalize_script_extra(args.script, extra)
    recording_modes: Sequence[str | None] = tuple(recordings) or (None,)
    cable_modes: Sequence[str | None] = tuple(cables) or (None,)
    n_axons_modes: Sequence[int | None] = tuple(n_axons_values) or (None,)
    for recording in recording_modes:
        for cable in cable_modes:
            for n_axons in n_axons_modes:
                for policy in policies:
                    run_dir = _run_dir(
                        args.output,
                        recording=recording,
                        cable=cable,
                        n_axons=n_axons,
                        policy=policy,
                    )
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
                    if recording is not None:
                        command.extend(["--recording", recording])
                    if cable is not None:
                        command.extend(["--cable", cable])
                    if n_axons is not None:
                        command.extend(["--n-axons", str(n_axons)])
                    if args.resume:
                        command.append("--resume")
                    command.extend(script_extra)
                    runs.append(
                        {
                            "policy": policy,
                            "recording": "" if recording is None else recording,
                            "cable": "" if cable is None else cable,
                            "n_axons": "" if n_axons is None else n_axons,
                            "run_dir": str(run_dir),
                            "command": command,
                        }
                    )
    return runs


def _run_dir(
    output: Path,
    *,
    recording: str | None,
    cable: str | None,
    n_axons: int | None,
    policy: str,
) -> Path:
    parts = []
    if recording is not None:
        parts.append(recording)
    if cable is not None:
        parts.append(cable)
    if n_axons is not None:
        parts.append(f"n{n_axons}")
    parts.append(policy_token(policy))
    return output.joinpath(*parts)


def normalize_script_extra(script: str, extra: Sequence[str]) -> list[str]:
    """Adapt campaign-scale aliases before forwarding to a concrete curve script."""
    if script != "threshold_curves":
        return [str(item) for item in extra]

    normalized: list[str] = []
    amplitude_count: str | None = None
    has_max_iterations = False
    items = [str(item) for item in extra]
    index = 0
    while index < len(items):
        item = items[index]
        if item == "--amplitude-count":
            if index + 1 >= len(items):
                raise argparse.ArgumentTypeError("--amplitude-count requires a value.")
            amplitude_count = items[index + 1]
            index += 2
            continue
        if item.startswith("--amplitude-count="):
            amplitude_count = item.split("=", 1)[1]
            index += 1
            continue
        if item == "--max-iterations" or item.startswith("--max-iterations="):
            has_max_iterations = True
        normalized.append(item)
        index += 1

    if amplitude_count is not None and not has_max_iterations:
        normalized.extend(["--max-iterations", amplitude_count])
    return normalized


def run_label(run: Mapping[str, Any]) -> str:
    recording = str(run.get("recording") or "")
    cable = str(run.get("cable") or "")
    n_axons = str(run.get("n_axons") or "")
    policy = str(run.get("policy") or "")
    parts = [
        part
        for part in (
            recording,
            cable,
            f"n{n_axons}" if n_axons else "",
            policy,
        )
        if part
    ]
    return "/".join(parts)


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
    recording: str = "",
    cable: str = "",
    n_axons: object = "",
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
        "recording": options.get("recording", recording),
        "cable": options.get("cable", cable),
        "n_axons": options.get("n_axons", n_axons),
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
        "curve_build_pool_ms": sum_duration(events, "curve.build_pool"),
        "repeat_curve_build_pool_ms": sum_duration(
            events,
            "curve.build_pool",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_build_pool_self_ms": sum_self_duration(
            events,
            "curve.build_pool",
            by_id=by_id,
        ),
        "repeat_curve_build_pool_self_ms": sum_self_duration(
            events,
            "curve.build_pool",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_build_pool_diameter_grid_ms": sum_duration(
            events,
            "curve.build_pool.diameter_grid",
        ),
        "repeat_curve_build_pool_diameter_grid_ms": sum_duration(
            events,
            "curve.build_pool.diameter_grid",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_build_pool_spatial_layout_ms": sum_duration(
            events,
            "curve.build_pool.spatial_layout",
        ),
        "repeat_curve_build_pool_spatial_layout_ms": sum_duration(
            events,
            "curve.build_pool.spatial_layout",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_build_pool_rows_ms": sum_duration(events, "curve.build_pool.rows"),
        "repeat_curve_build_pool_rows_ms": sum_duration(
            events,
            "curve.build_pool.rows",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_build_pool_template_build_ms": sum_duration(
            events,
            "curve.build_pool.template_build",
        ),
        "repeat_curve_build_pool_template_build_ms": sum_duration(
            events,
            "curve.build_pool.template_build",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_update_amplitudes_ms": sum_duration(events, "curve.update_amplitudes"),
        "repeat_curve_update_amplitudes_ms": sum_duration(
            events,
            "curve.update_amplitudes",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_update_amplitudes_rows_ms": sum_duration(
            events,
            "curve.update_amplitudes.rows",
        ),
        "repeat_curve_update_amplitudes_rows_ms": sum_duration(
            events,
            "curve.update_amplitudes.rows",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_update_amplitudes_stimulus_build_ms": sum_duration(
            events,
            "curve.update_amplitudes.stimulus_build",
        ),
        "repeat_curve_update_amplitudes_stimulus_build_ms": sum_duration(
            events,
            "curve.update_amplitudes.stimulus_build",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_activation_definition_ms": sum_duration(events, "curve.activation_definition"),
        "repeat_curve_activation_definition_ms": sum_duration(
            events,
            "curve.activation_definition",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_runtime_options_ms": sum_duration(events, "curve.runtime_options"),
        "repeat_curve_runtime_options_ms": sum_duration(
            events,
            "curve.runtime_options",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_construct_simulation_ms": sum_duration(events, "curve.construct_simulation"),
        "repeat_curve_construct_simulation_ms": sum_duration(
            events,
            "curve.construct_simulation",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_simulate_ms": sum_duration(events, "curve.simulate"),
        "repeat_curve_simulate_ms": sum_duration(events, "curve.simulate", phase="repeat", by_id=by_id),
        "curve_analyze_activation_ms": sum_duration(events, "curve.analyze_activation"),
        "repeat_curve_analyze_activation_ms": sum_duration(
            events,
            "curve.analyze_activation",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_analyze_activation_self_ms": sum_self_duration(
            events,
            "curve.analyze_activation",
            by_id=by_id,
        ),
        "repeat_curve_analyze_activation_self_ms": sum_self_duration(
            events,
            "curve.analyze_activation",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_analyze_activation_dense_values_ms": sum_duration(
            events,
            "curve.analyze_activation.dense_values",
        ),
        "repeat_curve_analyze_activation_dense_values_ms": sum_duration(
            events,
            "curve.analyze_activation.dense_values",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_analyze_activation_result_analyze_ms": sum_duration(
            events,
            "curve.analyze_activation.result_analyze",
        ),
        "repeat_curve_analyze_activation_result_analyze_ms": sum_duration(
            events,
            "curve.analyze_activation.result_analyze",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_analyze_activation_vm_raster_extract_ms": sum_duration(
            events,
            "curve.analyze_activation.vm_raster_extract",
        ),
        "repeat_curve_analyze_activation_vm_raster_extract_ms": sum_duration(
            events,
            "curve.analyze_activation.vm_raster_extract",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_analyze_activation_vm_raster_values_ms": sum_duration(
            events,
            "curve.analyze_activation.vm_raster_values",
        ),
        "repeat_curve_analyze_activation_vm_raster_values_ms": sum_duration(
            events,
            "curve.analyze_activation.vm_raster_values",
            phase="repeat",
            by_id=by_id,
        ),
        "curve_analyze_activation_materialize_values_ms": sum_duration(
            events,
            "curve.analyze_activation.materialize_values",
        ),
        "repeat_curve_analyze_activation_materialize_values_ms": sum_duration(
            events,
            "curve.analyze_activation.materialize_values",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_enqueue_ms": sum_duration(events, "kernel.enqueue"),
        "repeat_kernel_enqueue_ms": sum_duration(events, "kernel.enqueue", phase="repeat", by_id=by_id),
        "kernel_prepare_inputs_ms": sum_duration(events, "kernel.prepare_inputs"),
        "repeat_kernel_prepare_inputs_ms": sum_duration(
            events,
            "kernel.prepare_inputs",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_prepare_inputs_self_ms": sum_self_duration(
            events,
            "kernel.prepare_inputs",
            by_id=by_id,
        ),
        "repeat_kernel_prepare_inputs_self_ms": sum_self_duration(
            events,
            "kernel.prepare_inputs",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_prepare_arrays_ms": sum_duration(events, "kernel.prepare_arrays"),
        "repeat_kernel_prepare_arrays_ms": sum_duration(
            events,
            "kernel.prepare_arrays",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_prepare_double_coefficients_ms": sum_duration(
            events,
            "kernel.prepare_double_coefficients",
        ),
        "repeat_kernel_prepare_double_coefficients_ms": sum_duration(
            events,
            "kernel.prepare_double_coefficients",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_prepare_state_ms": sum_duration(events, "kernel.prepare_state"),
        "repeat_kernel_prepare_state_ms": sum_duration(
            events,
            "kernel.prepare_state",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_prepare_observer_state_ms": sum_duration(
            events,
            "kernel.prepare_observer_state",
        ),
        "repeat_kernel_prepare_observer_state_ms": sum_duration(
            events,
            "kernel.prepare_observer_state",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_prepare_observer_tables_ms": sum_duration(
            events,
            "kernel.prepare_observer_tables",
        ),
        "repeat_kernel_prepare_observer_tables_ms": sum_duration(
            events,
            "kernel.prepare_observer_tables",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_materialize_inputs_ms": sum_duration(events, "kernel.materialize_inputs"),
        "repeat_kernel_materialize_inputs_ms": sum_duration(
            events,
            "kernel.materialize_inputs",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_prepare_factorized_forcing_ms": sum_duration(
            events,
            "kernel.prepare_factorized_forcing",
        ),
        "repeat_kernel_prepare_factorized_forcing_ms": sum_duration(
            events,
            "kernel.prepare_factorized_forcing",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_prepare_factorized_vext_ms": sum_duration(
            events,
            "kernel.prepare_factorized_vext",
        ),
        "repeat_kernel_prepare_factorized_vext_ms": sum_duration(
            events,
            "kernel.prepare_factorized_vext",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_chunk_setup_ms": sum_duration(events, "kernel.chunk_setup"),
        "repeat_kernel_chunk_setup_ms": sum_duration(
            events,
            "kernel.chunk_setup",
            phase="repeat",
            by_id=by_id,
        ),
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
        "kernel_chunk_bookkeeping_ms": sum_duration(events, "kernel.chunk_bookkeeping"),
        "repeat_kernel_chunk_bookkeeping_ms": sum_duration(
            events,
            "kernel.chunk_bookkeeping",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_concat_trace_chunks_ms": sum_duration(events, "kernel.concat_trace_chunks"),
        "repeat_kernel_concat_trace_chunks_ms": sum_duration(
            events,
            "kernel.concat_trace_chunks",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_wait_ms": sum_duration(events, "kernel.wait"),
        "repeat_kernel_wait_ms": sum_duration(events, "kernel.wait", phase="repeat", by_id=by_id),
        "kernel_finalize_observer_ms": sum_duration(events, "kernel.finalize_observer"),
        "repeat_kernel_finalize_observer_ms": sum_duration(
            events,
            "kernel.finalize_observer",
            phase="repeat",
            by_id=by_id,
        ),
        "kernel_finalize_observer_to_host_ms": sum_duration(
            events,
            "kernel.finalize_observer.to_host",
        ),
        "repeat_kernel_finalize_observer_to_host_ms": sum_duration(
            events,
            "kernel.finalize_observer.to_host",
            phase="repeat",
            by_id=by_id,
        ),
        "results_split_batch_ms": sum_duration(events, "results.split_batch"),
        "repeat_results_split_batch_ms": sum_duration(
            events,
            "results.split_batch",
            phase="repeat",
            by_id=by_id,
        ),
        "results_trim_padded_batch_ms": sum_duration(events, "results.trim_padded_batch"),
        "repeat_results_trim_padded_batch_ms": sum_duration(
            events,
            "results.trim_padded_batch",
            phase="repeat",
            by_id=by_id,
        ),
        "results_materialize_vm_ms": sum_duration(events, "results.materialize_vm"),
        "repeat_results_materialize_vm_ms": sum_duration(
            events,
            "results.materialize_vm",
            phase="repeat",
            by_id=by_id,
        ),
        "results_materialize_vm_to_host_ms": sum_duration(
            events,
            "results.materialize_vm.to_host",
        ),
        "repeat_results_materialize_vm_to_host_ms": sum_duration(
            events,
            "results.materialize_vm.to_host",
            phase="repeat",
            by_id=by_id,
        ),
        "results_assemble_rows_ms": sum_duration(events, "results.assemble_rows"),
        "repeat_results_assemble_rows_ms": sum_duration(
            events,
            "results.assemble_rows",
            phase="repeat",
            by_id=by_id,
        ),
        "results_assemble_cohort_record_ms": sum_duration(
            events,
            "results.assemble_cohort_record",
        ),
        "repeat_results_assemble_cohort_record_ms": sum_duration(
            events,
            "results.assemble_cohort_record",
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


def sum_self_duration(
    events: Sequence[Mapping[str, Any]],
    name: str,
    *,
    phase: str | None = None,
    by_id: Mapping[int, Mapping[str, Any]] | None = None,
) -> float:
    by_id = by_id or {}
    child_ms: dict[int, float] = {}
    for event in events:
        if phase is not None and event_phase(event, by_id) != phase:
            continue
        parent = event.get("parent_event_id")
        if parent is None:
            continue
        try:
            parent_id = int(parent)
        except (TypeError, ValueError):
            continue
        child_ms[parent_id] = child_ms.get(parent_id, 0.0) + float(
            event.get("duration_ms") or 0.0
        )

    total = 0.0
    for event in events:
        if event.get("name") != name:
            continue
        if phase is not None and event_phase(event, by_id) != phase:
            continue
        event_id = int(event.get("event_id", -1))
        duration = float(event.get("duration_ms") or 0.0)
        total += max(duration - child_ms.get(event_id, 0.0), 0.0)
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
        "| recording | cable | Naxon | policy | status | build pool ms | construct ms | curve.simulate ms | analyze ms | prep/chunk ms | dispatch_jax ms | wait ms | combine ms | finalize/to-host ms | result/to-host ms | dispatch count | scope | chunk steps |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        prep_ms = (
            float(row.get("repeat_kernel_prepare_arrays_ms") or 0.0)
            + float(row.get("repeat_kernel_prepare_inputs_self_ms") or 0.0)
            + float(row.get("repeat_kernel_prepare_state_ms") or 0.0)
            + float(row.get("repeat_kernel_prepare_observer_state_ms") or 0.0)
            + float(row.get("repeat_kernel_prepare_observer_tables_ms") or 0.0)
            + float(row.get("repeat_kernel_materialize_inputs_ms") or 0.0)
            + float(row.get("repeat_kernel_prepare_factorized_forcing_ms") or 0.0)
            + float(row.get("repeat_kernel_prepare_factorized_vext_ms") or 0.0)
            + float(row.get("repeat_kernel_chunk_setup_ms") or 0.0)
            + float(row.get("repeat_kernel_chunk_bookkeeping_ms") or 0.0)
            + float(row.get("repeat_kernel_concat_trace_chunks_ms") or 0.0)
        )
        finalize_ms = float(row.get("repeat_kernel_finalize_observer_ms") or 0.0)
        finalize_to_host_ms = float(
            row.get("repeat_kernel_finalize_observer_to_host_ms") or 0.0
        )
        result_ms = float(row.get("repeat_results_split_batch_ms") or 0.0)
        result_to_host_ms = float(row.get("repeat_results_materialize_vm_to_host_ms") or 0.0)
        lines.append(
            "| {recording} | {cable} | {n_axons} | {policy} | {status} | {pool:.3f} | {construct:.3f} | {curve:.3f} | {analyze:.3f} | {prep:.3f} | {dispatch:.3f} | {wait:.3f} | {combine:.3f} | {finalize:.3f}/{finalize_to_host:.3f} | {result:.3f}/{result_to_host:.3f} | {count} | {scope} | {chunks} |".format(
                recording=row.get("recording", ""),
                cable=row.get("cable", ""),
                n_axons=row.get("n_axons", ""),
                policy=row.get("policy", ""),
                status=row.get("status", ""),
                pool=float(row.get("repeat_curve_build_pool_ms") or 0.0),
                construct=float(row.get("repeat_curve_construct_simulation_ms") or 0.0),
                curve=float(row.get("repeat_curve_simulate_ms") or 0.0),
                analyze=float(row.get("repeat_curve_analyze_activation_ms") or 0.0),
                prep=prep_ms,
                dispatch=float(row.get("repeat_kernel_dispatch_jax_ms") or 0.0),
                wait=float(row.get("repeat_kernel_wait_ms") or 0.0),
                combine=float(row.get("repeat_kernel_combine_observer_chunks_ms") or 0.0),
                finalize=finalize_ms,
                finalize_to_host=finalize_to_host_ms,
                result=result_ms,
                result_to_host=result_to_host_ms,
                count=row.get("kernel_dispatch_jax_count", ""),
                scope=row.get("observer_state_scopes", ""),
                chunks=row.get("dispatch_chunk_steps", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    policies: Sequence[str],
    recordings: Sequence[str],
    cables: Sequence[str],
    n_axons_values: Sequence[int],
    runs: Sequence[Mapping[str, Any]],
    extra: Sequence[str],
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": args.script,
        "preset": args.preset,
        "platform": args.platform,
        "output": str(args.output),
        "policies": list(policies),
        "recordings": list(recordings),
        "cables": list(cables),
        "n_axons": list(n_axons_values),
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
