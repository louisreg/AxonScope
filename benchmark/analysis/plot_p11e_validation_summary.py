from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "benchmark/results/.matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SINGLE_CPU = REPO_ROOT / (
    "benchmark/results/kaggle/"
    "20260711_174839_single_cable_solver_policy_gpu_smoke_cpu_NvidiaTeslaP100_"
    "axonscope-p11e-valid-single-cpu-d09bd1f/outputs/extracted/"
    "single_cable_solver_policy_summary.csv"
)
DEFAULT_SINGLE_GPU = REPO_ROOT / (
    "benchmark/results/kaggle/"
    "20260711_180312_single_cable_solver_policy_gpu_smoke_gpu_NvidiaTeslaP100_"
    "axonscope-p11e-valid-single-gpu-r2-d09bd1f/outputs/extracted/"
    "single_cable_solver_policy_summary.csv"
)
DEFAULT_DOUBLE_GPU_OBSERVER = REPO_ROOT / (
    "benchmark/results/kaggle/"
    "20260711_175408_double_cable_solver_policy_gpu_smoke_gpu_NvidiaTeslaP100_"
    "axonscope-p11e-valid-double-gpu-obs-triton-d09bd1f/outputs/extracted/"
    "double_cable_solver_policy_summary.csv"
)
DEFAULT_DOUBLE_GPU_PROBE = REPO_ROOT / (
    "benchmark/results/kaggle/"
    "20260711_181226_double_cable_solver_policy_gpu_smoke_gpu_NvidiaTeslaP100_"
    "axs-p11e-dc-probe-triton-d09bd1f/outputs/extracted/"
    "double_cable_solver_policy_summary.csv"
)
DEFAULT_DOUBLE_CPU = REPO_ROOT / (
    "benchmark/results/kaggle/"
    "20260711_182715_double_cable_solver_policy_gpu_smoke_cpu_NvidiaTeslaP100_"
    "axs-p11e-dc-cpu-thomas-d09bd1f/outputs/extracted/"
    "double_cable_solver_policy_summary.csv"
)

SCRIPT_LABEL = {
    "threshold_curves": "threshold",
    "recruitment_curves": "recruitment",
}
DIAMETER_LABEL = {
    "same_diameter": "same diam.",
    "different_diameters": "diff. diam.",
}
RECORDING_LABEL = {
    "observer_only": "observer-only",
    "probe_vm": "probe Vm",
}
STAGE_GROUPS = (
    ("runtime_prepare_ms", "runtime prepare", "#4c78a8"),
    ("inputs_extracellular_ms", "extracellular", "#f58518"),
    ("kernel_dispatch_jax_ms", "JAX call/deferred device", "#54a24b"),
    ("kernel_wait_ms", "kernel wait", "#e45756"),
    ("kernel_finalize_observer_ms", "finalize obs.", "#ff9da6"),
    ("results_assemble_rows_ms", "assemble rows", "#b279a2"),
)


@dataclass(frozen=True, slots=True)
class Source:
    label: str
    cable: str
    path: Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot the P11E CPU/GPU single/double-cable validation summary."
    )
    parser.add_argument("--single-cpu", type=Path, default=DEFAULT_SINGLE_CPU)
    parser.add_argument("--single-gpu", type=Path, default=DEFAULT_SINGLE_GPU)
    parser.add_argument("--double-gpu-observer", type=Path, default=DEFAULT_DOUBLE_GPU_OBSERVER)
    parser.add_argument("--double-gpu-probe", type=Path, default=DEFAULT_DOUBLE_GPU_PROBE)
    parser.add_argument("--double-cpu", type=Path, default=DEFAULT_DOUBLE_CPU)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmark/results/p11e_validation_summary",
    )
    args = parser.parse_args(argv)

    sources = (
        Source("single_cpu", "single", args.single_cpu),
        Source("single_gpu", "single", args.single_gpu),
        Source("double_gpu_observer", "double", args.double_gpu_observer),
        Source("double_gpu_probe", "double", args.double_gpu_probe),
        Source("double_cpu", "double", args.double_cpu),
    )
    rows = _load_sources(sources)
    if not rows:
        raise SystemExit("no benchmark rows found")

    output = args.output
    plots = output / "plots"
    tables = output / "tables"
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    _write_csv(tables / "validation_rows.csv", rows)
    speedup_rows = _build_speedup_rows(rows)
    _write_csv(tables / "speedups.csv", speedup_rows)

    figures = [
        _plot_gpu_vs_cpu_speedup(speedup_rows, plots / "gpu_vs_cpu_warm_speedup.png"),
        _plot_double_gpu_triton_speedup(rows, plots / "double_gpu_triton_speedup.png"),
        _plot_kernel_wait_share(rows, plots / "kernel_wait_share.png"),
        _plot_stage_breakdown(rows, plots / "stage_breakdown_representative_cases.png"),
        _plot_single_recording(rows, plots / "single_gpu_recording_comparison.png"),
        _plot_warm_time_overview(rows, plots / "warm_time_overview.png"),
    ]
    report = output / "validation_summary.md"
    _write_report(report, rows=rows, speedups=speedup_rows, figures=figures, tables=(tables / "validation_rows.csv", tables / "speedups.csv"))

    print(f"wrote: {report}")
    for figure in figures:
        print(f"wrote: {figure}")
    print(f"wrote: {tables / 'validation_rows.csv'}")
    print(f"wrote: {tables / 'speedups.csv'}")
    return 0


def _load_sources(sources: Sequence[Source]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        if not source.path.is_file():
            raise FileNotFoundError(source.path)
        for row in _read_csv(source.path):
            if row.get("status") != "passed":
                continue
            item = dict(row)
            item["source"] = source.label
            item["cable"] = source.cable
            item["solver_label"] = _solver_label(item)
            item["recording_label"] = RECORDING_LABEL.get(str(item.get("recording", "")), str(item.get("recording", "")))
            item["script_label"] = SCRIPT_LABEL.get(str(item.get("script", "")), str(item.get("script", "")))
            item["diameter_label"] = DIAMETER_LABEL.get(str(item.get("diameters", "")), str(item.get("diameters", "")))
            item["warm_ms"] = _float(item.get("curve_simulate_warm_mean_ms"))
            item["cold_ms"] = _float(item.get("curve_simulate_cold_ms"))
            item["total_ms"] = _float(item.get("curve_simulate_total_ms"))
            item["kernel_wait_pct"] = _ratio_pct(
                _float(item.get("kernel_wait_ms")),
                _float(item.get("curve_simulate_total_ms")),
            )
            rows.append(item)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_speedup_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    single_cpu = {
        _base_key(row): row
        for row in rows
        if row["cable"] == "single"
        and row["source"] == "single_cpu"
        and row["recording"] == "observer_only"
        and str(row["nx"]) == "89"
    }
    single_gpu = {
        _base_key(row): row
        for row in rows
        if row["cable"] == "single"
        and row["source"] == "single_gpu"
        and row["recording"] == "observer_only"
        and str(row["nx"]) == "89"
    }
    for key, cpu in single_cpu.items():
        gpu = single_gpu.get(key)
        if gpu is None:
            continue
        output.append(_speedup_row("single GPU/CPU", key, cpu, gpu))

    double_cpu = {
        _base_key(row): row
        for row in rows
        if row["cable"] == "double"
        and row["source"] == "double_cpu"
        and row["recording"] == "observer_only"
        and str(row["nx"]) == "89"
    }
    double_gpu = {
        _base_key(row): row
        for row in rows
        if row["cable"] == "double"
        and row["source"] == "double_gpu_observer"
        and row["recording"] == "observer_only"
        and row["solver"] == "tiled_thomas"
        and str(row["nx"]) == "89"
    }
    for key, cpu in double_cpu.items():
        gpu = double_gpu.get(key)
        if gpu is None:
            continue
        output.append(_speedup_row("double GPU Triton/CPU thomas", key, cpu, gpu))
    return output


def _speedup_row(
    comparison: str,
    key: tuple[str, str, str, str],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_ms = _float(baseline["warm_ms"])
    candidate_ms = _float(candidate["warm_ms"])
    return {
        "comparison": comparison,
        "script": key[0],
        "n_axons": key[1],
        "nx": key[2],
        "diameters": key[3],
        "baseline_warm_ms": baseline_ms,
        "candidate_warm_ms": candidate_ms,
        "speedup": baseline_ms / candidate_ms if baseline_ms and candidate_ms else math.nan,
    }


def _base_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["script"]),
        str(row["n_axons"]),
        str(row["nx"]),
        str(row["diameters"]),
    )


def _solver_label(row: Mapping[str, Any]) -> str:
    solver = str(row.get("solver") or row.get("single_cable_solver") or "auto")
    block_b = str(row.get("tiled_thomas_block_b") or "")
    if solver == "tiled_thomas" and block_b:
        return f"tiled_thomas_b{block_b}"
    return solver


def _plot_gpu_vs_cpu_speedup(rows: Sequence[Mapping[str, Any]], output: Path) -> Path:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["comparison"])].append(row)

    fig, axes = plt.subplots(1, len(grouped), figsize=(7.0 * max(len(grouped), 1), 5.0), squeeze=False)
    for ax, (comparison, group_rows) in zip(axes[0], grouped.items(), strict=False):
        group_rows = sorted(group_rows, key=lambda row: (str(row["script"]), int(row["n_axons"]), str(row["diameters"])))
        labels = [_case_label(row, include_nx=False) for row in group_rows]
        values = [_float(row["speedup"]) for row in group_rows]
        colors = [_diameter_color(str(row["diameters"])) for row in group_rows]
        ax.bar(range(len(values)), values, color=colors, width=0.72)
        ax.axhline(1.0, color="#333333", linewidth=1.0)
        ax.set_title(comparison)
        ax.set_ylabel("warm speedup")
        ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
        ax.set_ylim(0, max(values) * 1.18 if values else 1)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        for x, value in enumerate(values):
            ax.text(x, value, f"{value:.1f}x", ha="center", va="bottom", fontsize=8)
    _add_diameter_legend(axes[0][-1])
    fig.suptitle("GPU warm speedup against CPU on matched observer-only Nx=89 cases")
    fig.tight_layout()
    _save(fig, output)
    return output


def _plot_double_gpu_triton_speedup(rows: Sequence[Mapping[str, Any]], output: Path) -> Path:
    groups: dict[tuple[str, str, str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["cable"] != "double" or str(row["platform"]) != "gpu":
            continue
        if row["source"] not in {"double_gpu_observer", "double_gpu_probe"}:
            continue
        key = (
            str(row["recording"]),
            str(row["script"]),
            str(row["n_axons"]),
            str(row["nx"]),
            str(row["diameters"]),
        )
        groups[key][str(row["solver"])] = row

    plot_rows: list[dict[str, Any]] = []
    for key, solver_rows in groups.items():
        auto = solver_rows.get("auto")
        tiled = solver_rows.get("tiled_thomas")
        if auto is None or tiled is None:
            continue
        auto_ms = _float(auto["warm_ms"])
        tiled_ms = _float(tiled["warm_ms"])
        plot_rows.append(
            {
                "recording": key[0],
                "script": key[1],
                "n_axons": key[2],
                "nx": key[3],
                "diameters": key[4],
                "speedup": auto_ms / tiled_ms if auto_ms and tiled_ms else math.nan,
                "variant": tiled.get("effective_variants", ""),
            }
        )

    recordings = ["observer_only", "probe_vm"]
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2), sharey=True)
    for ax, recording in zip(axes, recordings, strict=True):
        subset = [row for row in plot_rows if row["recording"] == recording]
        subset.sort(key=lambda row: (str(row["script"]), int(row["n_axons"]), int(row["nx"]), str(row["diameters"])))
        labels = [_case_label(row, include_nx=True) for row in subset]
        values = [_float(row["speedup"]) for row in subset]
        colors = [_diameter_color(str(row["diameters"])) for row in subset]
        ax.bar(range(len(values)), values, color=colors, width=0.72)
        ax.axhline(1.0, color="#333333", linewidth=1.0)
        ax.set_title(f"{RECORDING_LABEL[recording]}: auto / tiled_thomas_b64")
        ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
        ax.set_ylabel("warm speedup")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        if values:
            ax.set_ylim(0, max(max(values) * 1.18, 1.2))
        for x, value in enumerate(values):
            ax.text(x, value, f"{value:.2f}x", ha="center", va="bottom", fontsize=8)
    _add_diameter_legend(axes[-1])
    fig.suptitle("Double-cable GPU: Triton tiled_thomas_b64 speedup over auto")
    fig.tight_layout()
    _save(fig, output)
    return output


def _plot_kernel_wait_share(rows: Sequence[Mapping[str, Any]], output: Path) -> Path:
    panels = (
        ("single", "observer_only", "Single cable observer-only"),
        ("single", "probe_vm", "Single cable probe Vm"),
        ("double", "observer_only", "Double cable observer-only"),
        ("double", "probe_vm", "Double cable probe Vm"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharey=True, constrained_layout=True)
    for ax, (cable, recording, title) in zip(axes.flat, panels, strict=True):
        means = _aggregate_by_naxons(
            rows,
            cable=cable,
            recording=recording,
            metric="kernel_wait_pct",
        )
        _plot_grouped_metric_by_naxons(
            ax,
            means,
            title=title,
            ylabel="kernel.wait / total simulate (%)",
            percent=True,
        )
        ax.axhline(50, color="#888888", linestyle="--", linewidth=1.0)
        ax.axhline(80, color="#555555", linestyle=":", linewidth=1.0)
    fig.suptitle("Kernel wait share, averaged over scripts and diameter modes at Nx=89")
    _save(fig, output)
    return output


def _plot_stage_breakdown(rows: Sequence[Mapping[str, Any]], output: Path) -> Path:
    selected = _representative_stage_rows(rows)
    labels = [_stage_case_label(row) for row in selected]
    bottoms = [0.0] * len(selected)

    fig, ax = plt.subplots(figsize=(13.0, 6.0))
    for field, label, color in STAGE_GROUPS:
        values = [
            _stage_pct(row, field)
            for row in selected
        ]
        ax.bar(range(len(selected)), values, bottom=bottoms, label=label, color=color, width=0.74)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    other_values = [max(0.0, 100.0 - bottom) for bottom in bottoms]
    ax.bar(
        range(len(selected)),
        other_values,
        bottom=bottoms,
        label="other/untracked",
        color="#bab0ac",
        width=0.74,
    )
    totals = [_float(row.get("curve_simulate_total_ms")) for row in selected]
    for x, total in enumerate(totals):
        ax.text(x, 102.0, _format_ms(total), ha="center", va="bottom", fontsize=8)
    ax.set_title("Representative stage shares of total curve.simulate time")
    ax.set_ylabel("share of total simulate time (%)")
    ax.set_xticks(range(len(labels)), labels, rotation=50, ha="right")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    ax.legend(ncols=3, fontsize=8)
    fig.tight_layout()
    _save(fig, output)
    return output


def _plot_single_recording(rows: Sequence[Mapping[str, Any]], output: Path) -> Path:
    source_rows = [
        row
        for row in rows
        if row["source"] == "single_gpu"
        and str(row["nx"]) == "89"
        and str(row["script"]) == "recruitment_curves"
    ]
    groups: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in source_rows:
        groups[(str(row["n_axons"]), str(row["diameters"]))][str(row["recording"])] = row

    plot_rows = []
    for key, recording_rows in groups.items():
        obs = recording_rows.get("observer_only")
        probe = recording_rows.get("probe_vm")
        if obs and probe:
            plot_rows.append((key, obs, probe))
    plot_rows.sort(key=lambda item: (int(item[0][0]), item[0][1]))

    labels = [f"N={key[0]}\n{DIAMETER_LABEL[key[1]]}" for key, _, _ in plot_rows]
    obs_values = [_float(obs["warm_ms"]) for _, obs, _ in plot_rows]
    probe_values = [_float(probe["warm_ms"]) for _, _, probe in plot_rows]
    x = list(range(len(plot_rows)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar([value - width / 2 for value in x], obs_values, width=width, label="observer-only", color="#4c78a8")
    ax.bar([value + width / 2 for value in x], probe_values, width=width, label="probe Vm", color="#f58518")
    ax.set_title("Single-cable GPU: recording mode warm-time comparison")
    ax.set_ylabel("warm mean curve.simulate (ms)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    _save(fig, output)
    return output


def _plot_warm_time_overview(rows: Sequence[Mapping[str, Any]], output: Path) -> Path:
    panels = (
        ("single", "observer_only", "Single cable observer-only"),
        ("single", "probe_vm", "Single cable probe Vm"),
        ("double", "observer_only", "Double cable observer-only"),
        ("double", "probe_vm", "Double cable probe Vm"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    for ax, (cable, recording, title) in zip(axes.flat, panels, strict=True):
        means = _aggregate_by_naxons(rows, cable=cable, recording=recording, metric="warm_ms")
        _plot_grouped_metric_by_naxons(
            ax,
            means,
            title=title,
            ylabel="warm mean curve.simulate (ms)",
            log=True,
        )
    fig.suptitle("Warm time overview, averaged over scripts and diameter modes at Nx=89")
    _save(fig, output)
    return output


def _aggregate_by_naxons(
    rows: Sequence[Mapping[str, Any]],
    *,
    cable: str,
    recording: str,
    metric: str,
) -> dict[str, dict[int, float]]:
    buckets: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["cable"] != cable:
            continue
        if str(row.get("recording")) != recording:
            continue
        if str(row.get("nx")) != "89":
            continue
        if not _is_representative_solver(row):
            continue
        value = _float(row.get(metric))
        if math.isnan(value):
            continue
        label = _series_label(row)
        buckets[label][int(str(row["n_axons"]))].append(value)
    return {
        label: {n_axons: _mean(values) for n_axons, values in by_n.items()}
        for label, by_n in buckets.items()
    }


def _plot_grouped_metric_by_naxons(
    ax: Any,
    means: Mapping[str, Mapping[int, float]],
    *,
    title: str,
    ylabel: str,
    log: bool = False,
    percent: bool = False,
) -> None:
    n_values = sorted({n for by_n in means.values() for n in by_n})
    series = sorted(means, key=_series_sort_key)
    if not n_values or not series:
        ax.axis("off")
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_title(title)
        return
    width = min(0.8 / max(len(series), 1), 0.34)
    x_values = list(range(len(n_values)))
    offsets = [
        (index - (len(series) - 1) / 2.0) * width
        for index in range(len(series))
    ]
    for offset, label in zip(offsets, series, strict=True):
        values = [means[label].get(n, math.nan) for n in n_values]
        positions = [x + offset for x in x_values]
        color = _series_color(label)
        ax.bar(positions, values, width=width * 0.92, label=label, color=color)
        for x, value in zip(positions, values, strict=True):
            if math.isnan(value):
                continue
            text = f"{value:.0f}%" if percent else _format_ms(value)
            ax.text(x, value, text, ha="center", va="bottom", fontsize=7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values, [f"N={n}" for n in n_values])
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, which="both")
    ax.legend(fontsize=8)


def _representative_stage_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    wanted: list[tuple[str, str, str, str, str, str]] = [
        ("single", "gpu", "observer_only", "auto", "8192", "same_diameter"),
        ("single", "gpu", "probe_vm", "auto", "8192", "same_diameter"),
        ("single", "cpu", "observer_only", "auto", "8192", "same_diameter"),
        ("double", "gpu", "observer_only", "tiled_thomas", "8192", "same_diameter"),
        ("double", "gpu", "probe_vm", "tiled_thomas", "4096", "same_diameter"),
        ("double", "cpu", "observer_only", "thomas", "4096", "same_diameter"),
    ]
    selected = []
    for cable, platform, recording, solver, n_axons, diameters in wanted:
        matches = [
            row
            for row in rows
            if row["cable"] == cable
            and str(row["platform"]) == platform
            and str(row["recording"]) == recording
            and str(row.get("solver") or row.get("single_cable_solver")) == solver
            and str(row["n_axons"]) == n_axons
            and str(row["diameters"]) == diameters
            and str(row["nx"]) == "89"
            and str(row["script"]) == "recruitment_curves"
        ]
        if matches:
            selected.append(matches[0])
    return selected


def _is_representative_solver(row: Mapping[str, Any]) -> bool:
    if row["cable"] == "single":
        return str(row.get("single_cable_solver") or row.get("solver")) == "auto"
    if str(row["platform"]) == "cpu":
        return str(row.get("solver")) == "thomas"
    if str(row["recording"]) in {"observer_only", "probe_vm"}:
        return str(row.get("solver")) == "tiled_thomas"
    return False


def _write_report(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    speedups: Sequence[Mapping[str, Any]],
    figures: Sequence[Path],
    tables: Sequence[Path],
) -> None:
    double_triton_speedups = [
        _float(row["speedup"])
        for row in speedups
        if str(row["comparison"]).startswith("double")
    ]
    single_speedups = [
        _float(row["speedup"])
        for row in speedups
        if str(row["comparison"]).startswith("single")
    ]
    gpu_probe = [
        row
        for row in rows
        if row["source"] == "double_gpu_probe" and row["solver"] == "tiled_thomas"
    ]
    variants = sorted({str(row.get("effective_variants", "")) for row in rows if row.get("effective_variants")})
    lines = [
        "# P11E Benchmark Validation Summary",
        "",
        "Fresh Kaggle artifacts around commit `d09bd1f`.",
        "",
        "## Main Checks",
        "",
        f"- Loaded {len(rows)} passed summary rows.",
        f"- Effective variants observed: `{', '.join(variants)}`.",
        f"- Single-cable GPU/CPU warm speedup on matched observer-only Nx=89 cases: {_range(single_speedups)}.",
        f"- Double-cable GPU Triton/CPU thomas warm speedup on matched observer-only Nx=89 cases: {_range(double_triton_speedups)}.",
        f"- Double-cable GPU probe Triton rows: {len(gpu_probe)} passed rows.",
        "",
        "Interpretation: double-cable observer-only large populations validate the Triton route; probe Vm still shows meaningful non-solver cost. Single-cable GPU is fast versus CPU but not strongly solver-bound.",
        "",
        "## Figures",
        "",
    ]
    for figure in figures:
        lines.append(f"- [{figure.name}]({figure.relative_to(path.parent).as_posix()})")
    lines.extend(["", "## Tables", ""])
    for table in tables:
        lines.append(f"- [{table.name}]({table.relative_to(path.parent).as_posix()})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_label(row: Mapping[str, Any], *, include_nx: bool) -> str:
    parts = [
        SCRIPT_LABEL.get(str(row["script"]), str(row["script"])),
        f"N={row['n_axons']}",
    ]
    if include_nx:
        parts.append(f"Nx={row['nx']}")
    parts.append(DIAMETER_LABEL.get(str(row["diameters"]), str(row["diameters"])))
    return "\n".join(parts)


def _compact_row_label(row: Mapping[str, Any]) -> str:
    platform = str(row.get("platform", ""))
    cable = str(row.get("cable", ""))
    recording = RECORDING_LABEL.get(str(row.get("recording", "")), str(row.get("recording", "")))
    solver = str(row.get("solver_label", ""))
    return "\n".join(
        [
            f"{cable} {platform}",
            recording,
            SCRIPT_LABEL.get(str(row["script"]), str(row["script"])),
            f"N={row['n_axons']} {DIAMETER_LABEL.get(str(row['diameters']), str(row['diameters']))}",
            solver,
        ]
    )


def _stage_case_label(row: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"{row['cable']} {row['platform']}",
            RECORDING_LABEL.get(str(row["recording"]), str(row["recording"])),
            f"N={row['n_axons']}",
            str(row["solver_label"]),
        ]
    )


def _source_color(row: Mapping[str, Any]) -> str:
    source = str(row.get("source"))
    if source == "single_cpu":
        return "#9ecae9"
    if source == "single_gpu":
        return "#3182bd"
    if source == "double_cpu":
        return "#fdae6b"
    if source in {"double_gpu_observer", "double_gpu_probe"}:
        return "#e6550d"
    return "#777777"


def _series_label(row: Mapping[str, Any]) -> str:
    cable = str(row.get("cable", ""))
    platform = str(row.get("platform", ""))
    solver = str(row.get("solver_label", ""))
    if cable == "double" and platform == "gpu":
        return f"GPU {solver}"
    if cable == "double" and platform == "cpu":
        return "CPU thomas"
    if cable == "single" and platform == "gpu":
        return "GPU"
    if cable == "single" and platform == "cpu":
        return "CPU"
    return f"{platform} {solver}".strip()


def _series_sort_key(label: str) -> tuple[int, str]:
    order = {
        "CPU": 0,
        "CPU thomas": 0,
        "GPU": 1,
        "GPU tiled_thomas_b64": 1,
    }
    return (order.get(label, 10), label)


def _series_color(label: str) -> str:
    colors = {
        "CPU": "#9ecae9",
        "GPU": "#3182bd",
        "CPU thomas": "#fdae6b",
        "GPU tiled_thomas_b64": "#e6550d",
    }
    return colors.get(label, "#777777")


def _source_legend() -> list[tuple[str, str]]:
    return [
        ("single CPU", "#9ecae9"),
        ("single GPU", "#3182bd"),
        ("double CPU", "#fdae6b"),
        ("double GPU", "#e6550d"),
    ]


def _add_source_legend(ax: Any) -> None:
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(facecolor=color, label=label) for label, color in _source_legend()],
        loc="upper left",
        ncols=4,
        fontsize=8,
    )


def _diameter_color(value: str) -> str:
    return "#4c78a8" if value == "same_diameter" else "#f58518"


def _add_diameter_legend(ax: Any) -> None:
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=_diameter_color("same_diameter"), label="same diameter"),
            Patch(facecolor=_diameter_color("different_diameters"), label="different diameters"),
        ],
        loc="upper left",
        fontsize=8,
    )


def _platform_rank(row: Mapping[str, Any]) -> tuple[int, int]:
    return (0 if str(row.get("platform")) == "cpu" else 1, 0 if row["cable"] == "single" else 1)


def _float(value: Any) -> float:
    if value in {None, ""}:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _stage_pct(row: Mapping[str, Any], field: str) -> float:
    value = _float(row.get(field))
    total = _float(row.get("curve_simulate_total_ms"))
    if math.isnan(value) or math.isnan(total) or total <= 0:
        return 0.0
    return max(0.0, 100.0 * value / total)


def _ratio_pct(numerator: float, denominator: float) -> float:
    if math.isnan(numerator) or math.isnan(denominator) or denominator == 0:
        return math.nan
    return 100.0 * numerator / denominator


def _format_ms(value: float) -> str:
    if math.isnan(value):
        return ""
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    return f"{value:.0f}ms"


def _range(values: Iterable[float]) -> str:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return "n/a"
    return f"{min(finite):.1f}x to {max(finite):.1f}x"


def _mean(values: Iterable[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def _save(fig: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
