"""Plot CPU/GPU time-chunk benchmark matrices for P11B bottleneck triage."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.analysis.cold_path_audit import classify_stage


POLICY_ORDER = ("default", "unchunked", "50", "250", "500", "1000")
RECORDING_ORDER = ("full_vm", "probe_vm", "observer_only")
SCRIPT_ORDER = ("threshold_curves", "recruitment_curves")
PLATFORM_ORDER = ("cpu", "gpu")

STAGE_SPECS = (
    ("prepare inputs", "prepare_inputs_s", "#8CD17D"),
    ("prepare arrays", "prepare_arrays_s", "#4C78A8"),
    ("prepare state", "prepare_state_s", "#72B7B2"),
    ("observer tables", "observer_tables_s", "#54A24B"),
    ("materialize inputs", "materialize_inputs_s", "#B279A2"),
    ("chunk setup", "chunk_setup_s", "#F58518"),
    ("dispatch_jax", "dispatch_s", "#4C78A8"),
    ("wait/sync", "wait_s", "#E15759"),
    ("chunk bookkeeping", "chunk_bookkeeping_s", "#FF9DA6"),
    ("concat trace", "concat_trace_s", "#9D755D"),
    ("combine", "combine_s", "#59A14F"),
    ("finalize to-host", "finalize_to_host_s", "#F28E2B"),
    ("finalize other", "finalize_other_s", "#FFBE7D"),
    ("Vm to-host", "materialize_to_host_s", "#B07AA1"),
    ("Vm materialize other", "materialize_other_s", "#D4A6C8"),
    ("assemble rows", "assemble_rows_s", "#EDC948"),
    ("assemble cohort", "assemble_cohort_s", "#76B7B2"),
    ("result other", "result_other_s", "#9C755F"),
    ("other/setup", "unattributed_s", "#BAB0AC"),
)

GROUP_SPECS = (
    ("curve/setup", "group_curve_s", "#4C78A8"),
    ("pool build", "group_pool_build_s", "#F28E2B"),
    ("dispatch", "group_dispatch_s", "#59A14F"),
    ("runtime prepare", "group_runtime_prepare_s", "#B07AA1"),
    ("input lowering", "group_input_lowering_s", "#76B7B2"),
    ("kernel", "group_kernel_s", "#E15759"),
    ("result assembly", "group_result_assembly_s", "#EDC948"),
    ("other", "group_other_s", "#BAB0AC"),
)

ROW_FIELDS = (
    "source_label",
    "source_root",
    "script",
    "platform",
    "recording",
    "policy",
    "status",
    "case_name",
    "n_axons",
    "nx",
    "tsim",
    "dt",
    "precision",
    "time_chunk_policy",
    "time_chunk_steps",
    "dispatch_count",
    "observer_scope",
    "dispatch_chunk_steps",
    "curve_s",
    "prepare_inputs_s",
    "prepare_arrays_s",
    "prepare_state_s",
    "observer_tables_s",
    "materialize_inputs_s",
    "prepare_factorized_forcing_s",
    "chunk_setup_s",
    "dispatch_s",
    "wait_s",
    "chunk_bookkeeping_s",
    "concat_trace_s",
    "combine_s",
    "finalize_s",
    "finalize_to_host_s",
    "finalize_other_s",
    "result_s",
    "materialize_s",
    "materialize_to_host_s",
    "materialize_other_s",
    "assemble_rows_s",
    "assemble_cohort_s",
    "result_other_s",
    "unattributed_s",
    "group_curve_s",
    "group_pool_build_s",
    "group_dispatch_s",
    "group_runtime_prepare_s",
    "group_input_lowering_s",
    "group_kernel_s",
    "group_result_assembly_s",
    "group_other_s",
    "peak_rss_end_mib",
    "peak_rss_delta_mib",
    "peak_device_end_mib",
    "peak_device_delta_mib",
    "peak_nvidia_smi_end_mib",
    "git_commit",
    "device_model",
    "host_platform",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help=(
            "A time_chunk_sweep output directory. Repeat for CPU/GPU or "
            "threshold/recruitment runs. LABEL= is optional."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/p11b_time_chunk_matrix_report"),
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    if not args.run:
        parser.error("at least one --run LABEL=DIR is required")

    rows: list[dict[str, Any]] = []
    for value in args.run:
        label, root = parse_run_arg(value)
        rows.extend(read_campaign(root, source_label=label))
    if not rows:
        print("No time-chunk rows found.")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    matrix_csv = args.output / "time_chunk_matrix_rows.csv"
    best_csv = args.output / "time_chunk_best_rows.csv"
    report_md = args.output / "time_chunk_matrix_report.md"
    best_rows = select_best_rows(rows)

    write_csv(matrix_csv, ROW_FIELDS, rows)
    write_csv(best_csv, ROW_FIELDS, best_rows)
    plots: list[Path] = []
    if not args.no_plots:
        plots = write_plots(args.output / "plots", rows, best_rows)
    write_report(report_md, rows, best_rows, plots)

    print(f"wrote: {matrix_csv}")
    print(f"wrote: {best_csv}")
    print(f"wrote: {report_md}")
    for plot in plots:
        print(f"plot: {plot}")
    return 0


def parse_run_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), Path(path).expanduser()
    path = Path(value).expanduser()
    return path.name, path


def read_campaign(root: Path, *, source_label: str) -> list[dict[str, Any]]:
    summary_path = find_summary(root)
    campaign_root = summary_path.parent
    hardware = read_hardware(campaign_root)
    rows = []
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if not raw:
                continue
            run_dir = row_result_dir(campaign_root, raw)
            memory = read_row_memory(run_dir)
            stage_groups = read_row_stage_groups(run_dir)
            row = normalize_row(
                raw,
                memory,
                stage_groups,
                source_label=source_label,
                source_root=str(campaign_root),
                hardware=hardware,
            )
            rows.append(row)
    return rows


def find_summary(root: Path) -> Path:
    direct = root / "time_chunk_sweep_summary.csv"
    if direct.is_file():
        return direct
    matches = sorted(root.rglob("time_chunk_sweep_summary.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"missing time_chunk_sweep_summary.csv under {root}")


def read_hardware(root: Path) -> dict[str, str]:
    path = root / "kaggle_hardware.json"
    if not path.is_file():
        return {"git_commit": "", "device_model": "", "host_platform": ""}
    data = _mapping(json.loads(path.read_text(encoding="utf-8")))
    git = _mapping(data.get("git"))
    short_commit = _mapping(git.get("short_commit")).get("stdout")
    devices = _sequence(_mapping(data.get("jax")).get("devices"))
    device_models = [
        str(_mapping(device).get("device_kind") or _mapping(device).get("repr") or "")
        for device in devices
    ]
    return {
        "git_commit": str(short_commit or ""),
        "device_model": "; ".join(item for item in device_models if item),
        "host_platform": str(data.get("platform") or ""),
    }


def row_result_dir(campaign_root: Path, raw: Mapping[str, Any]) -> Path:
    recording = str(raw.get("recording") or "")
    policy = str(raw.get("policy") or "")
    if recording:
        nested = campaign_root / recording / policy_token(policy)
        if nested.exists():
            return nested
    return campaign_root / policy_token(policy)


def read_row_memory(run_dir: Path) -> dict[str, float | None]:
    memory_path = run_dir / "memory_summary.csv"
    if not memory_path.is_file():
        return empty_memory()

    rows = read_csv_rows(memory_path)
    return {
        "peak_rss_end_mib": max_column(rows, "rss_end_mib_max"),
        "peak_rss_delta_mib": max_column(rows, "rss_delta_mib_max"),
        "peak_device_end_mib": bytes_to_mib(max_column(rows, "device_bytes_in_use_end_max")),
        "peak_device_delta_mib": bytes_to_mib(max_column(rows, "device_bytes_in_use_delta_max")),
        "peak_nvidia_smi_end_mib": max_column(rows, "nvidia_smi_memory_used_end_mib_max"),
    }


def read_row_stage_groups(run_dir: Path) -> dict[str, float]:
    groups = {field: 0.0 for _label, field, _color in GROUP_SPECS}
    summary_path = run_dir / "summary.csv"
    if not summary_path.is_file():
        return groups
    for row in read_csv_rows(summary_path):
        group = classify_stage(str(row.get("name") or ""))
        field = f"group_{group}_s"
        if field in groups:
            groups[field] += ms_to_s(row.get("self_ms"))
    return groups


def empty_memory() -> dict[str, float | None]:
    return {
        "peak_rss_end_mib": None,
        "peak_rss_delta_mib": None,
        "peak_device_end_mib": None,
        "peak_device_delta_mib": None,
        "peak_nvidia_smi_end_mib": None,
    }


def normalize_row(
    raw: Mapping[str, Any],
    memory: Mapping[str, float | None],
    stage_groups: Mapping[str, float],
    *,
    source_label: str,
    source_root: str,
    hardware: Mapping[str, str],
) -> dict[str, Any]:
    result_s = ms_to_s(raw.get("repeat_results_split_batch_ms"))
    materialize_s = ms_to_s(raw.get("repeat_results_materialize_vm_ms"))
    materialize_to_host_s = ms_to_s(raw.get("repeat_results_materialize_vm_to_host_ms"))
    materialize_other_s = max(materialize_s - materialize_to_host_s, 0.0)
    assemble_rows_s = ms_to_s(raw.get("repeat_results_assemble_rows_ms"))
    assemble_cohort_s = ms_to_s(raw.get("repeat_results_assemble_cohort_record_ms"))
    result_other_s = max(result_s - materialize_s - assemble_rows_s - assemble_cohort_s, 0.0)
    prepare_inputs_s = ms_to_s(
        raw.get("repeat_kernel_prepare_inputs_self_ms")
        or raw.get("repeat_kernel_prepare_inputs_ms")
    )
    prepare_arrays_s = ms_to_s(raw.get("repeat_kernel_prepare_arrays_ms"))
    prepare_state_s = ms_to_s(raw.get("repeat_kernel_prepare_state_ms"))
    observer_tables_s = ms_to_s(raw.get("repeat_kernel_prepare_observer_tables_ms"))
    materialize_inputs_s = ms_to_s(raw.get("repeat_kernel_materialize_inputs_ms"))
    prepare_factorized_forcing_s = ms_to_s(
        raw.get("repeat_kernel_prepare_factorized_forcing_ms")
    )
    chunk_setup_s = ms_to_s(raw.get("repeat_kernel_chunk_setup_ms"))
    chunk_bookkeeping_s = ms_to_s(raw.get("repeat_kernel_chunk_bookkeeping_ms"))
    concat_trace_s = ms_to_s(raw.get("repeat_kernel_concat_trace_chunks_ms"))
    finalize_s = ms_to_s(raw.get("repeat_kernel_finalize_observer_ms"))
    finalize_to_host_s = ms_to_s(raw.get("repeat_kernel_finalize_observer_to_host_ms"))
    finalize_other_s = max(finalize_s - finalize_to_host_s, 0.0)
    measured_s = (
        prepare_inputs_s
        + prepare_arrays_s
        + prepare_state_s
        + observer_tables_s
        + materialize_inputs_s
        + chunk_setup_s
        + ms_to_s(raw.get("repeat_kernel_dispatch_jax_ms"))
        + ms_to_s(raw.get("repeat_kernel_wait_ms"))
        + chunk_bookkeeping_s
        + concat_trace_s
        + ms_to_s(raw.get("repeat_kernel_combine_observer_chunks_ms"))
        + finalize_s
        + materialize_s
        + assemble_rows_s
        + assemble_cohort_s
        + result_other_s
    )
    curve_s = ms_to_s(raw.get("repeat_curve_simulate_ms"))
    row: dict[str, Any] = {
        "source_label": source_label,
        "source_root": source_root,
        "script": raw.get("script", ""),
        "platform": raw.get("platform", ""),
        "recording": raw.get("recording", ""),
        "policy": raw.get("policy", ""),
        "status": raw.get("status", ""),
        "case_name": raw.get("case_name", ""),
        "n_axons": raw.get("n_axons", ""),
        "nx": raw.get("nx", ""),
        "tsim": raw.get("tsim", ""),
        "dt": raw.get("dt", ""),
        "precision": raw.get("precision", ""),
        "time_chunk_policy": raw.get("time_chunk_policy", ""),
        "time_chunk_steps": raw.get("time_chunk_steps", ""),
        "dispatch_count": raw.get("kernel_dispatch_jax_count", ""),
        "observer_scope": raw.get("observer_state_scopes", ""),
        "dispatch_chunk_steps": raw.get("dispatch_chunk_steps", ""),
        "curve_s": curve_s,
        "prepare_inputs_s": prepare_inputs_s,
        "prepare_arrays_s": prepare_arrays_s,
        "prepare_state_s": prepare_state_s,
        "observer_tables_s": observer_tables_s,
        "materialize_inputs_s": materialize_inputs_s,
        "prepare_factorized_forcing_s": prepare_factorized_forcing_s,
        "chunk_setup_s": chunk_setup_s,
        "dispatch_s": ms_to_s(raw.get("repeat_kernel_dispatch_jax_ms")),
        "wait_s": ms_to_s(raw.get("repeat_kernel_wait_ms")),
        "chunk_bookkeeping_s": chunk_bookkeeping_s,
        "concat_trace_s": concat_trace_s,
        "combine_s": ms_to_s(raw.get("repeat_kernel_combine_observer_chunks_ms")),
        "finalize_s": finalize_s,
        "finalize_to_host_s": finalize_to_host_s,
        "finalize_other_s": finalize_other_s,
        "result_s": result_s,
        "materialize_s": materialize_s,
        "materialize_to_host_s": materialize_to_host_s,
        "materialize_other_s": materialize_other_s,
        "assemble_rows_s": assemble_rows_s,
        "assemble_cohort_s": assemble_cohort_s,
        "result_other_s": result_other_s,
        "unattributed_s": max(curve_s - measured_s, 0.0),
        "git_commit": hardware.get("git_commit", ""),
        "device_model": hardware.get("device_model", ""),
        "host_platform": hardware.get("host_platform", ""),
    }
    row.update(memory)
    row.update(stage_groups)
    return row


def select_best_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("status") != "passed":
            continue
        key = (str(row.get("script")), str(row.get("platform")), str(row.get("recording")))
        current = buckets.get(key)
        if current is None or _float(row.get("curve_s")) < _float(current.get("curve_s")):
            buckets[key] = row
    return [dict(row) for row in sorted(buckets.values(), key=sort_key)]


def write_plots(output: Path, rows: Sequence[Mapping[str, Any]], best_rows: Sequence[Mapping[str, Any]]) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    try:
        mpl_config = Path("benchmark/results/.matplotlib")
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config.resolve()))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional plotting dependency.
        print(f"plots skipped: {type(exc).__name__}: {exc}")
        return []

    configure_matplotlib(plt)
    plots = [
        plot_curve_heatmaps(output, rows, plt, np),
        plot_best_stage_group_breakdown(output, best_rows, plt, np),
        plot_best_kernel_result_breakdown(output, best_rows, plt, np),
        plot_speedup(output, best_rows, plt, np),
    ]
    plots.extend(plot_stage_policy_panels(output, rows, plt, np))
    memory_plots = [
        plot_memory_panels(
            output,
            rows,
            plt,
            np,
            platform="cpu",
            metric="peak_rss_end_mib",
            filename="memory_cpu_rss_end.png",
            title="CPU peak RSS by policy",
            ylabel="RSS end [MiB]",
        ),
        plot_memory_panels(
            output,
            rows,
            plt,
            np,
            platform="gpu",
            metric="peak_device_end_mib",
            filename="memory_gpu_jax_device_end.png",
            title="GPU live JAX device memory by policy",
            ylabel="JAX device bytes in use [MiB]",
        ),
        plot_memory_panels(
            output,
            rows,
            plt,
            np,
            platform="gpu",
            metric="peak_nvidia_smi_end_mib",
            filename="memory_gpu_nvidia_smi_end.png",
            title="GPU process memory from nvidia-smi",
            ylabel="nvidia-smi used [MiB]",
        ),
    ]
    plots.extend(path for path in memory_plots if path is not None)
    return [path for path in plots if path is not None]


def configure_matplotlib(plt: Any) -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_curve_heatmaps(output: Path, rows: Sequence[Mapping[str, Any]], plt: Any, np: Any) -> Path:
    scripts = ordered_values(rows, "script", SCRIPT_ORDER)
    platforms = ordered_values(rows, "platform", PLATFORM_ORDER)
    policies = ordered_values(rows, "policy", POLICY_ORDER)
    recordings = ordered_values(rows, "recording", RECORDING_ORDER)
    fig, axes = plt.subplots(
        len(scripts),
        len(platforms),
        figsize=(max(8.0, len(platforms) * 4.8), max(4.8, len(scripts) * 3.2)),
        squeeze=False,
        constrained_layout=True,
    )
    all_values = [_float(row.get("curve_s")) for row in rows if row.get("status") == "passed"]
    vmin = min(all_values) if all_values else 0.0
    vmax = max(all_values) if all_values else 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#E5E7EB")
    for row_idx, script in enumerate(scripts):
        for col_idx, platform in enumerate(platforms):
            ax = axes[row_idx][col_idx]
            matrix = np.full((len(recordings), len(policies)), np.nan)
            for rec_idx, recording in enumerate(recordings):
                for pol_idx, policy in enumerate(policies):
                    match = first_row(
                        rows,
                        script=script,
                        platform=platform,
                        recording=recording,
                        policy=policy,
                    )
                    if match is not None:
                        matrix[rec_idx, pol_idx] = _float(match.get("curve_s"))
            image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            for rec_idx in range(len(recordings)):
                for pol_idx in range(len(policies)):
                    value = matrix[rec_idx, pol_idx]
                    if np.isnan(value):
                        continue
                    color = "white" if value > (vmin + vmax) / 2.0 else "#111827"
                    ax.text(pol_idx, rec_idx, f"{value:.1f}", ha="center", va="center", color=color, fontsize=8)
            ax.set_title(f"{script_label(script)} / {platform.upper()}")
            ax.set_xticks(range(len(policies)), policies, rotation=35, ha="right")
            ax.set_yticks(range(len(recordings)), [recording_label(item) for item in recordings])
            ax.grid(False)
    fig.colorbar(image, ax=axes, shrink=0.86, label="curve.simulate [s]")
    path = output / "curve_time_heatmaps.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_best_stage_group_breakdown(output: Path, best_rows: Sequence[Mapping[str, Any]], plt: Any, np: Any) -> Path:
    rows = sorted(best_rows, key=sort_key)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12.5, max(5.0, len(rows) * 0.42)), constrained_layout=True)
    left = np.zeros(len(rows), dtype=float)
    for label, field, color in GROUP_SPECS:
        values = np.asarray([_float(row.get(field)) for row in rows], dtype=float)
        if not np.any(values):
            continue
        ax.barh(y, values, left=left, label=label, color=color)
        left += values
    labels = [
        f"{script_short(row)} {str(row.get('platform')).upper()} {recording_label(str(row.get('recording')))} ({row.get('policy')})"
        for row in rows
    ]
    totals = [_float(row.get("curve_s")) for row in rows]
    for index, total in enumerate(totals):
        ax.text(total + max(totals) * 0.01, index, f"{total:.1f}s", va="center", fontsize=8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("seconds")
    ax.set_title("Best policy per script/platform/recording: exclusive pipeline groups")
    ax.legend(ncols=3, fontsize=8)
    path = output / "best_policy_stage_group_breakdown.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_best_kernel_result_breakdown(output: Path, best_rows: Sequence[Mapping[str, Any]], plt: Any, np: Any) -> Path:
    rows = sorted(best_rows, key=sort_key)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12.5, max(5.0, len(rows) * 0.42)), constrained_layout=True)
    left = np.zeros(len(rows), dtype=float)
    for label, field, color in STAGE_SPECS:
        values = np.asarray([_float(row.get(field)) for row in rows], dtype=float)
        if not np.any(values):
            continue
        ax.barh(y, values, left=left, label=label, color=color)
        left += values
    labels = [
        f"{script_short(row)} {str(row.get('platform')).upper()} {recording_label(str(row.get('recording')))} ({row.get('policy')})"
        for row in rows
    ]
    totals = [_float(row.get("curve_s")) for row in rows]
    for index, total in enumerate(totals):
        ax.text(total + max(totals) * 0.01, index, f"{total:.1f}s", va="center", fontsize=8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("seconds")
    ax.set_title("Best policy: kernel/result sub-stage breakdown")
    ax.legend(ncols=3, fontsize=8)
    path = output / "best_policy_kernel_result_breakdown.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_stage_policy_panels(output: Path, rows: Sequence[Mapping[str, Any]], plt: Any, np: Any) -> list[Path]:
    plots = []
    policies = ordered_values(rows, "policy", POLICY_ORDER)
    recordings = ordered_values(rows, "recording", RECORDING_ORDER)
    for script in ordered_values(rows, "script", SCRIPT_ORDER):
        for platform in ordered_values(rows, "platform", PLATFORM_ORDER):
            panel_rows = [
                row
                for row in rows
                if row.get("script") == script and row.get("platform") == platform
            ]
            if not panel_rows:
                continue
            fig, axes = plt.subplots(
                len(recordings),
                1,
                figsize=(11.5, max(6.0, len(recordings) * 2.5)),
                sharex=True,
                constrained_layout=False,
            )
            axes = list(axes if isinstance(axes, np.ndarray) else [axes])
            x = np.arange(len(policies))
            for ax, recording in zip(axes, recordings, strict=False):
                rec_rows = [
                    first_row(panel_rows, script=script, platform=platform, recording=recording, policy=policy)
                    for policy in policies
                ]
                bottom = np.zeros(len(policies), dtype=float)
                for label, field, color in GROUP_SPECS:
                    values = np.asarray([
                        _float(row.get(field)) if row is not None else 0.0
                        for row in rec_rows
                    ])
                    if not np.any(values):
                        continue
                    ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.72)
                    bottom += values
                totals = [
                    _float(row.get("curve_s")) if row is not None else 0.0
                    for row in rec_rows
                ]
                ax.plot(x, totals, color="#111827", marker="o", linewidth=1.5, label="total")
                for idx, total in enumerate(totals):
                    if total:
                        ax.text(idx, total + max(totals) * 0.02, f"{total:.1f}", ha="center", fontsize=8)
                ax.set_ylabel(recording_label(recording))
                ax.set_ylim(0.0, max(totals) * 1.18 if totals else 1.0)
            axes[-1].set_xticks(x, policies, rotation=35, ha="right")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.suptitle(
                f"{script_label(script)} / {platform.upper()} pipeline groups by time-chunk policy",
                y=0.98,
            )
            fig.legend(
                handles,
                labels,
                ncols=4,
                fontsize=8,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
            )
            fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
            path = output / f"stage_breakdown_{script_short_name(script)}_{platform}.png"
            fig.savefig(path, dpi=170)
            plt.close(fig)
            plots.append(path)
    return plots


def plot_memory_panels(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    plt: Any,
    np: Any,
    *,
    platform: str,
    metric: str,
    filename: str,
    title: str,
    ylabel: str,
) -> Path | None:
    platform_rows = [
        row
        for row in rows
        if row.get("platform") == platform and _float(row.get(metric)) > 0.0
    ]
    if not platform_rows:
        return None
    scripts = ordered_values(platform_rows, "script", SCRIPT_ORDER)
    recordings = ordered_values(platform_rows, "recording", RECORDING_ORDER)
    policies = ordered_values(platform_rows, "policy", POLICY_ORDER)
    fig, axes = plt.subplots(
        len(scripts),
        len(recordings),
        figsize=(max(10.5, len(recordings) * 3.6), max(4.0, len(scripts) * 3.0)),
        squeeze=False,
        constrained_layout=True,
    )
    x = np.arange(len(policies))
    for row_idx, script in enumerate(scripts):
        for col_idx, recording in enumerate(recordings):
            ax = axes[row_idx][col_idx]
            values = []
            for policy in policies:
                row = first_row(platform_rows, script=script, platform=platform, recording=recording, policy=policy)
                values.append(_float(row.get(metric)) if row is not None else 0.0)
            ax.bar(x, values, color="#4C78A8")
            ax.set_title(f"{script_label(script)} / {recording_label(recording)}")
            ax.set_xticks(x, policies, rotation=35, ha="right")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ymax = max(values) if values else 0.0
            if ymax:
                ax.set_ylim(0.0, ymax * 1.16)
    fig.suptitle(title, fontsize=12)
    path = output / filename
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_speedup(output: Path, best_rows: Sequence[Mapping[str, Any]], plt: Any, np: Any) -> Path:
    scripts = ordered_values(best_rows, "script", SCRIPT_ORDER)
    recordings = ordered_values(best_rows, "recording", RECORDING_ORDER)
    labels: list[str] = []
    cpu_values: list[float] = []
    gpu_values: list[float] = []
    speedups: list[float] = []
    for script in scripts:
        for recording in recordings:
            cpu = first_row(best_rows, script=script, platform="cpu", recording=recording)
            gpu = first_row(best_rows, script=script, platform="gpu", recording=recording)
            if cpu is None or gpu is None:
                continue
            cpu_s = _float(cpu.get("curve_s"))
            gpu_s = _float(gpu.get("curve_s"))
            labels.append(f"{script_short(cpu)}\n{recording_label(recording)}")
            cpu_values.append(cpu_s)
            gpu_values.append(gpu_s)
            speedups.append(cpu_s / gpu_s if gpu_s else 0.0)

    x = np.arange(len(labels))
    width = 0.36
    fig, (time_ax, speed_ax) = plt.subplots(2, 1, figsize=(11.0, 7.0), constrained_layout=True)
    time_ax.bar(x - width / 2.0, cpu_values, width, label="CPU", color="#4C78A8")
    time_ax.bar(x + width / 2.0, gpu_values, width, label="GPU", color="#F28E2B")
    time_ax.set_xticks(x, labels)
    time_ax.set_ylabel("best curve.simulate [s]")
    time_ax.set_title("Best CPU/GPU time by workflow and recording")
    time_ax.legend()
    speed_ax.bar(x, speedups, color="#59A14F")
    speed_ax.axhline(1.0, color="#111827", linewidth=1.0)
    speed_ax.set_xticks(x, labels)
    speed_ax.set_ylabel("CPU / GPU speedup")
    for idx, value in enumerate(speedups):
        speed_ax.text(idx, value + max(speedups) * 0.02, f"{value:.1f}x", ha="center", fontsize=8)
    path = output / "best_cpu_gpu_speedup.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def write_report(path: Path, rows: Sequence[Mapping[str, Any]], best_rows: Sequence[Mapping[str, Any]], plots: Sequence[Path]) -> None:
    lines = [
        "# P11B Time-Chunk Matrix Report",
        "",
        "This report compares time-chunk policies across CPU/GPU, threshold curves,",
        "recruitment curves, and recording modes. It is bottleneck cartography, not",
        "a default-policy change by itself.",
        "",
        "## Inputs",
        "",
    ]
    input_rows = []
    seen_inputs = set()
    for row in rows:
        key = (row.get("source_label"), row.get("source_root"))
        if key in seen_inputs:
            continue
        seen_inputs.add(key)
        input_rows.append(
            (
                row.get("source_label", ""),
                row.get("script", ""),
                row.get("platform", ""),
                row.get("git_commit", ""),
                _short(str(row.get("device_model") or ""), 34),
                _short(str(row.get("host_platform") or ""), 42),
                row.get("source_root", ""),
            )
        )
    lines.extend(
        markdown_table(
            ("label", "script", "platform", "git", "device", "host", "root"),
            input_rows,
        )
    )
    lines.extend(["", "## Best Policies", ""])
    lines.extend(
        markdown_table(
            (
                "script",
                "platform",
                "recording",
                "best policy",
                "curve s",
                "top group",
                "group s",
                "RSS MiB",
                "JAX MiB",
                "nvidia MiB",
            ),
            [
                (
                    script_label(str(row.get("script"))),
                    str(row.get("platform")).upper(),
                    recording_label(str(row.get("recording"))),
                    row.get("policy", ""),
                    f"{_float(row.get('curve_s')):.2f}",
                    top_group(row)[0],
                    f"{top_group(row)[1]:.2f}",
                    fmt_optional(row.get("peak_rss_end_mib")),
                    fmt_optional(row.get("peak_device_end_mib")),
                    fmt_optional(row.get("peak_nvidia_smi_end_mib")),
                )
                for row in best_rows
            ],
        )
    )
    lines.extend(["", "## Plot Files", ""])
    for plot in plots:
        lines.append(f"- `{plot.name}`")
    if any(_float(row.get("peak_nvidia_smi_end_mib")) for row in rows):
        lines.extend(
            [
                "",
                "## Memory Reading",
                "",
                "`peak_device_end_mib` is live JAX device allocation from JAX memory",
                "stats. `peak_nvidia_smi_end_mib` is process/device context from",
                "`nvidia-smi`; it often includes JAX allocator reservation and should",
                "not be read as live arrays alone.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def top_group(row: Mapping[str, Any]) -> tuple[str, float]:
    candidates = [
        (label, _float(row.get(field)))
        for label, field, _color in GROUP_SPECS
    ]
    return max(candidates, key=lambda item: item[1], default=("", 0.0))


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def max_column(rows: Sequence[Mapping[str, Any]], column: str) -> float | None:
    values = [_float(row.get(column)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def policy_token(policy: str) -> str:
    if policy in {"default", "unchunked"}:
        return policy
    return f"chunk_{policy}"


def ms_to_s(value: Any) -> float:
    return _float(value) / 1000.0


def bytes_to_mib(value: float | None) -> float | None:
    return None if value is None else value / float(1024**2)


def first_row(rows: Sequence[Mapping[str, Any]], **criteria: str) -> Mapping[str, Any] | None:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in criteria.items()):
            return row
    return None


def ordered_values(rows: Sequence[Mapping[str, Any]], field: str, preferred: Sequence[str]) -> list[str]:
    values = [str(row.get(field) or "") for row in rows]
    ordered = [value for value in preferred if value in values]
    ordered.extend(sorted(value for value in set(values) if value and value not in ordered))
    return ordered


def sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    script = str(row.get("script") or "")
    platform = str(row.get("platform") or "")
    recording = str(row.get("recording") or "")
    return (
        index_or_end(SCRIPT_ORDER, script),
        index_or_end(PLATFORM_ORDER, platform),
        index_or_end(RECORDING_ORDER, recording),
        str(row.get("policy") or ""),
    )


def index_or_end(values: Sequence[str], value: str) -> int:
    try:
        return values.index(value)
    except ValueError:
        return len(values)


def script_label(script: str) -> str:
    return {
        "threshold_curves": "threshold",
        "recruitment_curves": "recruitment",
    }.get(script, script)


def script_short(row: Mapping[str, Any]) -> str:
    return "thr" if row.get("script") == "threshold_curves" else "rec"


def script_short_name(script: str) -> str:
    return "threshold" if script == "threshold_curves" else "recruitment"


def recording_label(recording: str) -> str:
    return {
        "full_vm": "full Vm",
        "probe_vm": "probe Vm",
        "observer_only": "observer only",
    }.get(recording, recording)


def markdown_table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> list[str]:
    result = ["| " + " | ".join(str(header) for header in headers) + " |"]
    result.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        result.append("| " + " | ".join(escape_cell(value) for value in row) + " |")
    return result


def escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def fmt_optional(value: Any) -> str:
    parsed = _float(value)
    return "" if parsed == 0.0 else f"{parsed:.0f}"


def _float(value: Any) -> float:
    if value in {None, ""}:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str | bytes) else ()


def _short(value: str, limit: int = 70) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
