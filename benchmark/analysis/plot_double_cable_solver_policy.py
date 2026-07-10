from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch


SOLVER_ORDER = (
    "auto",
    "pcr_soa",
    "pcr",
    "thomas",
    "tiled_thomas_b32",
    "tiled_thomas_b64",
)
SOLVER_COLORS = {
    "auto": "#2f6f9f",
    "pcr_soa": "#1b9e77",
    "pcr": "#7570b3",
    "thomas": "#d95f02",
    "tiled_thomas_b32": "#e7298a",
    "tiled_thomas_b64": "#66a61e",
}
SCRIPT_LABELS = {
    "threshold_curves": "threshold",
    "recruitment_curves": "recruitment",
}
RECORDING_LABELS = {
    "observer_only": "observer",
    "probe_vm": "probe Vm",
}
DIAMETER_LABELS = {
    "same_diameter": "same diameter",
    "different_diameters": "different diameters",
}
METRIC_LABELS = {
    "curve_simulate_warm_mean_ms": "warm mean",
    "curve_simulate_cold_ms": "cold",
    "curve_simulate_total_ms": "total",
}
STAGE_FIELDS = (
    ("runtime_prepare_ms", "runtime prepare", "#4c78a8"),
    ("inputs_extracellular_ms", "extracellular", "#f58518"),
    ("kernel_dispatch_jax_ms", "dispatch/JAX", "#54a24b"),
    ("kernel_wait_ms", "kernel wait", "#e45756"),
    ("results_assemble_rows_ms", "assemble rows", "#b279a2"),
)
STAGE_GROUPS = (
    ("runtime_prepare", "runtime prepare", "#4c78a8", ("runtime_prepare_ms",)),
    ("extracellular", "extracellular", "#f58518", ("inputs_extracellular_ms",)),
    (
        "kernel_prepare",
        "kernel prep",
        "#72b7b2",
        (
            "kernel_prepare_inputs_ms",
            "kernel_prepare_arrays_ms",
            "kernel_prepare_double_coefficients_ms",
        ),
    ),
    ("dispatch_jax", "dispatch/JAX", "#54a24b", ("kernel_dispatch_jax_ms",)),
    ("kernel_wait", "kernel wait", "#e45756", ("kernel_wait_ms",)),
    (
        "finalize_to_host",
        "finalize/to_host",
        "#ff9da6",
        ("kernel_finalize_observer_ms", "kernel_finalize_observer_to_host_ms"),
    ),
    ("assemble_rows", "assemble rows", "#b279a2", ("results_assemble_rows_ms",)),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot the double-cable solver policy campaign summary."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="double_cable_solver_policy_summary.csv.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="Double-cable GPU solver policy",
        help="Title prefix for generated figures.",
    )
    args = parser.parse_args(argv)

    rows = _read_summary(args.summary)
    if not rows:
        raise SystemExit(f"no rows found in {args.summary}")
    args.output.mkdir(parents=True, exist_ok=True)

    figures = []
    figures.append(
        _plot_metric_heatmap(
            rows,
            metric="curve_simulate_warm_mean_ms",
            output=args.output / "warm_time_heatmap.png",
            title=f"{args.title}: warm time by condition and solver",
        )
    )
    figures.append(
        _plot_metric_heatmap(
            rows,
            metric="curve_simulate_cold_ms",
            output=args.output / "cold_time_heatmap.png",
            title=f"{args.title}: cold time by condition and solver",
        )
    )
    figures.append(
        _plot_metric_heatmap(
            rows,
            metric="curve_simulate_total_ms",
            output=args.output / "total_time_heatmap.png",
            title=f"{args.title}: total curve.simulate time",
        )
    )
    figures.append(
        _plot_speedup_vs_auto(
            rows,
            output=args.output / "warm_speedup_vs_auto_heatmap.png",
            title=f"{args.title}: warm speedup relative to auto",
        )
    )
    figures.append(
        _plot_winner_grid(
            rows,
            output=args.output / "warm_winner_grid.png",
            title=f"{args.title}: fastest warm solver",
        )
    )
    figures.append(
        _plot_warm_scaling(
            rows,
            output=args.output / "warm_scaling_by_naxons.png",
            title=f"{args.title}: warm scaling by population size",
        )
    )
    figures.append(
        _plot_diameter_warm_scaling(
            rows,
            output=args.output / "warm_scaling_same_vs_different_diameters.png",
            title=f"{args.title}: same vs different diameter warm scaling",
        )
    )
    figures.append(
        _plot_diameter_ratio_heatmap(
            rows,
            output=args.output / "warm_different_vs_same_diameter_ratio.png",
            title=f"{args.title}: different/same diameter warm-time ratio",
        )
    )
    figures.append(
        _plot_best_stage_share(
            rows,
            output=args.output / "best_solver_stage_share.png",
            title=f"{args.title}: best-solver total-time stage shares",
        )
    )
    stage_comparison = _write_stage_comparison_csv(
        rows,
        output=args.output / "stage_comparison_by_solver.csv",
    )
    figures.extend(
        _plot_stage_by_solver_panels(
            rows,
            output=args.output,
            title_prefix=f"{args.title}: stage time by solver",
            normalize=False,
        )
    )
    figures.extend(
        _plot_stage_by_solver_panels(
            rows,
            output=args.output,
            title_prefix=f"{args.title}: stage share by solver",
            normalize=True,
        )
    )

    _write_index(
        args.output / "plot_index.md",
        summary=args.summary,
        rows=rows,
        figures=figures,
        tables=(stage_comparison,),
    )
    print(f"wrote: {args.output / 'plot_index.md'}")
    for figure in figures:
        print(f"wrote: {figure}")
    return 0


def _read_summary(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "passed":
                continue
            rows.append(dict(row))
    return rows


def _plot_metric_heatmap(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    output: Path,
    title: str,
) -> Path:
    conditions = _conditions(rows)
    by_condition = _rows_by_condition(rows)
    matrix: list[list[float]] = []
    labels: list[str] = []
    for condition in conditions:
        labels.append(_condition_label(condition))
        solver_rows = by_condition[condition]
        matrix.append(
            [
                _positive_or_nan(_float(solver_rows.get(solver, {}).get(metric)))
                for solver in SOLVER_ORDER
            ]
        )

    color_values = [
        [math.log10(value) if value and not math.isnan(value) else math.nan for value in row]
        for row in matrix
    ]
    fig, ax = plt.subplots(figsize=(12.5, max(8.0, 0.35 * len(conditions))))
    image = ax.imshow(color_values, aspect="auto", cmap="viridis_r")
    ax.set_title(title)
    ax.set_xticks(range(len(SOLVER_ORDER)), SOLVER_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlabel("solver policy")
    ax.set_ylabel("condition")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"log10({METRIC_LABELS.get(metric, metric)} ms)")

    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if value and not math.isnan(value):
                ax.text(x, y, _format_ms(value), ha="center", va="center", fontsize=6)

    fig.tight_layout()
    _save(fig, output)
    return output


def _plot_speedup_vs_auto(
    rows: Sequence[Mapping[str, Any]],
    *,
    output: Path,
    title: str,
) -> Path:
    conditions = _conditions(rows)
    by_condition = _rows_by_condition(rows)
    matrix: list[list[float]] = []
    labels: list[str] = []
    for condition in conditions:
        labels.append(_condition_label(condition))
        solver_rows = by_condition[condition]
        auto = _float(solver_rows.get("auto", {}).get("curve_simulate_warm_mean_ms"))
        row_values: list[float] = []
        for solver in SOLVER_ORDER:
            value = _float(solver_rows.get(solver, {}).get("curve_simulate_warm_mean_ms"))
            row_values.append(auto / value if auto and value else math.nan)
        matrix.append(row_values)

    finite = [value for row in matrix for value in row if not math.isnan(value)]
    if not finite:
        fig, ax = plt.subplots(figsize=(9.0, 3.0))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No auto baseline in this summary.\nSpeedup vs auto is not defined.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title(title)
        fig.tight_layout()
        _save(fig, output)
        return output
    vmin = min(min(finite), 0.5)
    vmax = max(max(finite), 1.5)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12.5, max(8.0, 0.35 * len(conditions))))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", norm=norm)
    ax.set_title(title)
    ax.set_xticks(range(len(SOLVER_ORDER)), SOLVER_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlabel("solver policy")
    ax.set_ylabel("condition")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("auto warm time / solver warm time")

    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if not math.isnan(value):
                ax.text(x, y, f"{value:.2f}x", ha="center", va="center", fontsize=6)

    fig.tight_layout()
    _save(fig, output)
    return output


def _plot_winner_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    output: Path,
    title: str,
) -> Path:
    by_condition = _rows_by_condition(rows)
    scripts = _unique_sorted(row["script"] for row in rows)
    recordings = _unique_sorted(row["recording"] for row in rows)
    observer_scopes = _unique_sorted(_observer_scope(row) for row in rows)
    n_axons_values = sorted({int(row["n_axons"]) for row in rows})
    nx_values = sorted({int(row["nx"]) for row in rows})
    diameter_values = _unique_sorted(row["diameters"] for row in rows)
    x_values = [
        (nx, diameters, observer_scope)
        for nx in nx_values
        for diameters in diameter_values
        for observer_scope in observer_scopes
    ]
    solver_to_int = {solver: index for index, solver in enumerate(SOLVER_ORDER)}
    cmap = ListedColormap([SOLVER_COLORS[solver] for solver in SOLVER_ORDER])
    norm = BoundaryNorm(range(len(SOLVER_ORDER) + 1), len(SOLVER_ORDER))

    fig, axes = plt.subplots(
        len(scripts),
        len(recordings),
        figsize=(4.2 * len(recordings), 3.4 * len(scripts)),
        squeeze=False,
    )
    for row_index, script in enumerate(scripts):
        for col_index, recording in enumerate(recordings):
            ax = axes[row_index][col_index]
            data = [[math.nan for _ in x_values] for _ in n_axons_values]
            text = [["" for _ in x_values] for _ in n_axons_values]
            for y, n_axons in enumerate(n_axons_values):
                for x, (nx, diameters, observer_scope) in enumerate(x_values):
                    condition = (
                        script,
                        recording,
                        observer_scope,
                        str(n_axons),
                        str(nx),
                        diameters,
                    )
                    solver_rows = by_condition.get(condition, {})
                    winner = _winner(solver_rows)
                    if winner:
                        data[y][x] = solver_to_int[winner]
                        text[y][x] = _short_solver(winner)
            ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
            ax.set_title(f"{SCRIPT_LABELS.get(script, script)} / {RECORDING_LABELS.get(recording, recording)}")
            ax.set_xticks(
                range(len(x_values)),
                [
                    _condition_x_label(nx, diameters, observer_scope)
                    for nx, diameters, observer_scope in x_values
                ],
            )
            ax.set_yticks(range(len(n_axons_values)), [str(value) for value in n_axons_values])
            ax.set_xlabel("Nx")
            ax.set_ylabel("Naxons")
            for y in range(len(n_axons_values)):
                for x in range(len(x_values)):
                    if text[y][x]:
                        ax.text(x, y, text[y][x], ha="center", va="center", fontsize=8)

    handles = [
        Patch(facecolor=SOLVER_COLORS[solver], edgecolor="none", label=_short_solver(solver))
        for solver in SOLVER_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    _save(fig, output)
    return output


def _plot_warm_scaling(
    rows: Sequence[Mapping[str, Any]],
    *,
    output: Path,
    title: str,
) -> Path:
    scripts = _unique_sorted(row["script"] for row in rows)
    recordings = _unique_sorted(row["recording"] for row in rows)
    observer_scopes = _unique_sorted(_observer_scope(row) for row in rows)
    nx_values = sorted({int(row["nx"]) for row in rows})
    diameter_values = _unique_sorted(row["diameters"] for row in rows)
    by_key: dict[tuple[str, str, str, str, int, int, str], float] = {}
    for row in rows:
        by_key[
            (
                str(row["script"]),
                str(row["recording"]),
                _observer_scope(row),
                _solver_token(row),
                int(row["nx"]),
                int(row["n_axons"]),
                str(row["diameters"]),
            )
        ] = _float(row["curve_simulate_warm_mean_ms"]) or math.nan

    fig, axes = plt.subplots(
        len(scripts),
        len(recordings),
        figsize=(6.0 * len(recordings), 4.2 * len(scripts)),
        squeeze=False,
    )
    for row_index, script in enumerate(scripts):
        for col_index, recording in enumerate(recordings):
            ax = axes[row_index][col_index]
            for solver in SOLVER_ORDER:
                for observer_scope in observer_scopes:
                    for nx in nx_values:
                        for diameters in diameter_values:
                            key_prefix = (
                                script,
                                recording,
                                observer_scope,
                                solver,
                                nx,
                            )
                            points = [
                                (
                                    n_axons,
                                    by_key[
                                        (
                                            *key_prefix,
                                            n_axons,
                                            diameters,
                                        )
                                    ],
                                )
                                for n_axons in sorted(
                                    {
                                        key[5]
                                        for key in by_key
                                        if key[0] == script
                                        and key[1] == recording
                                        and key[2] == observer_scope
                                        and key[3] == solver
                                        and key[4] == nx
                                        and key[6] == diameters
                                    }
                                )
                                if not math.isnan(
                                    by_key[
                                        (
                                            *key_prefix,
                                            n_axons,
                                            diameters,
                                        )
                                    ]
                                )
                            ]
                            if not points:
                                continue
                            linestyle = _diameter_scope_linestyle(diameters, observer_scope)
                            marker = "o" if nx == min(nx_values) else "s"
                            ax.plot(
                                [point[0] for point in points],
                                [point[1] for point in points],
                                color=SOLVER_COLORS[solver],
                                linestyle=linestyle,
                                marker=marker,
                                linewidth=1.6,
                                markersize=4,
                                alpha=0.9,
                            )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks([64, 1024, 4096, 8192], ["64", "1024", "4096", "8192"])
            ax.set_title(f"{SCRIPT_LABELS.get(script, script)} / {RECORDING_LABELS.get(recording, recording)}")
            ax.set_xlabel("Naxons")
            ax.set_ylabel("warm mean ms")
            ax.grid(True, which="both", alpha=0.25)

    solver_handles = [
        Patch(facecolor=SOLVER_COLORS[solver], edgecolor="none", label=_short_solver(solver))
        for solver in SOLVER_ORDER
    ]
    nx_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", marker="o", label=f"Nx={min(nx_values)}"),
        plt.Line2D([0], [0], color="black", linestyle="--", marker="s", label=f"Nx={max(nx_values)}"),
    ]
    diameter_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", label="same diameter"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="different diameters"),
    ]
    scope_handles = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle=_scope_linestyle(observer_scope),
            label=f"obs={observer_scope}",
        )
        for observer_scope in observer_scopes
        if observer_scope != "default"
    ]
    fig.legend(
        handles=solver_handles + nx_handles + diameter_handles + scope_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    _save(fig, output)
    return output


def _plot_diameter_warm_scaling(
    rows: Sequence[Mapping[str, Any]],
    *,
    output: Path,
    title: str,
) -> Path:
    scripts = _unique_sorted(row["script"] for row in rows)
    recordings = _unique_sorted(row["recording"] for row in rows)
    observer_scopes = _unique_sorted(_observer_scope(row) for row in rows)
    nx_values = sorted({int(row["nx"]) for row in rows})
    solvers = [solver for solver in SOLVER_ORDER if any(_solver_token(row) == solver for row in rows)]
    by_key: dict[tuple[str, str, str, str, int, int, str], float] = {}
    for row in rows:
        by_key[
            (
                str(row["script"]),
                str(row["recording"]),
                _observer_scope(row),
                _solver_token(row),
                int(row["nx"]),
                int(row["n_axons"]),
                str(row["diameters"]),
            )
        ] = _float(row["curve_simulate_warm_mean_ms"]) or math.nan

    fig, axes = plt.subplots(
        len(scripts),
        len(recordings),
        figsize=(6.0 * len(recordings), 4.2 * len(scripts)),
        squeeze=False,
    )
    for row_index, script in enumerate(scripts):
        for col_index, recording in enumerate(recordings):
            ax = axes[row_index][col_index]
            for solver in solvers:
                for observer_scope in observer_scopes:
                    for nx in nx_values:
                        for diameters in ("same_diameter", "different_diameters"):
                            points = [
                                (key[5], value)
                                for key, value in by_key.items()
                                if key[0] == script
                                and key[1] == recording
                                and key[2] == observer_scope
                                and key[3] == solver
                                and key[4] == nx
                                and key[6] == diameters
                                and not math.isnan(value)
                            ]
                            points.sort()
                            if not points:
                                continue
                            ax.plot(
                                [point[0] for point in points],
                                [point[1] for point in points],
                                color=SOLVER_COLORS.get(solver, "#4c78a8"),
                                linestyle=_diameter_scope_linestyle(
                                    diameters,
                                    observer_scope,
                                ),
                                marker="o",
                                linewidth=1.8,
                                markersize=4,
                                label=(
                                    f"{_short_solver(solver)} / "
                                    f"{_short_diameters(diameters)} / "
                                    f"obs={observer_scope}"
                                ),
                            )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks([64, 1024, 4096, 8192], ["64", "1024", "4096", "8192"])
            ax.set_title(f"{SCRIPT_LABELS.get(script, script)} / {RECORDING_LABELS.get(recording, recording)}")
            ax.set_xlabel("Naxons")
            ax.set_ylabel("warm mean ms")
            ax.grid(True, which="both", alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.10 if handles else 0.0, 1, 0.96))
    _save(fig, output)
    return output


def _plot_diameter_ratio_heatmap(
    rows: Sequence[Mapping[str, Any]],
    *,
    output: Path,
    title: str,
) -> Path:
    scripts = _unique_sorted(row["script"] for row in rows)
    recordings = _unique_sorted(row["recording"] for row in rows)
    observer_scopes = _unique_sorted(_observer_scope(row) for row in rows)
    n_axons_values = sorted({int(row["n_axons"]) for row in rows})
    nx_values = sorted({int(row["nx"]) for row in rows})
    solvers = [solver for solver in SOLVER_ORDER if any(_solver_token(row) == solver for row in rows)]
    x_values = [(n_axons, nx) for n_axons in n_axons_values for nx in nx_values]
    by_key: dict[tuple[str, str, str, str, int, int, str], float] = {}
    for row in rows:
        by_key[
            (
                str(row["script"]),
                str(row["recording"]),
                _observer_scope(row),
                _solver_token(row),
                int(row["n_axons"]),
                int(row["nx"]),
                str(row["diameters"]),
            )
        ] = _float(row["curve_simulate_warm_mean_ms"]) or math.nan

    matrix: list[list[float]] = []
    labels: list[str] = []
    for script in scripts:
        for recording in recordings:
            for observer_scope in observer_scopes:
                for solver in solvers:
                    labels.append(
                        f"{SCRIPT_LABELS.get(script, script)} | "
                        f"{RECORDING_LABELS.get(recording, recording)} | "
                        f"obs={observer_scope} | {_short_solver(solver)}"
                    )
                    row_values: list[float] = []
                    for n_axons, nx in x_values:
                        same = by_key.get(
                            (
                                script,
                                recording,
                                observer_scope,
                                solver,
                                n_axons,
                                nx,
                                "same_diameter",
                            )
                        )
                        different = by_key.get(
                            (
                                script,
                                recording,
                                observer_scope,
                                solver,
                                n_axons,
                                nx,
                                "different_diameters",
                            )
                        )
                        row_values.append(
                            different / same
                            if same and different and same > 0.0 and different > 0.0
                            else math.nan
                        )
                    matrix.append(row_values)

    finite = [value for row in matrix for value in row if not math.isnan(value)]
    if not finite:
        fig, ax = plt.subplots(figsize=(9.0, 3.0))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Need both same_diameter and different_diameters rows\nfor this ratio plot.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title(title)
        fig.tight_layout()
        _save(fig, output)
        return output

    vmin = min(min(finite), 0.75)
    vmax = max(max(finite), 1.25)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(max(9.0, 1.2 * len(x_values)), max(4.0, 0.5 * len(labels))))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", norm=norm)
    ax.set_title(title)
    ax.set_xticks(
        range(len(x_values)),
        [f"N={n_axons}\nNx={nx}" for n_axons, nx in x_values],
    )
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis="both", labelsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("different diameter warm time / same diameter warm time")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if not math.isnan(value):
                ax.text(x, y, f"{value:.2f}x", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    _save(fig, output)
    return output


def _plot_best_stage_share(
    rows: Sequence[Mapping[str, Any]],
    *,
    output: Path,
    title: str,
) -> Path:
    conditions = _conditions(rows)
    by_condition = _rows_by_condition(rows)
    labels: list[str] = []
    stage_rows: list[list[float]] = []
    other_values: list[float] = []

    for condition in conditions:
        solver_rows = by_condition[condition]
        winner = _winner(solver_rows)
        if not winner:
            continue
        row = solver_rows[winner]
        total = _float(row.get("curve_simulate_total_ms")) or 0.0
        if total <= 0:
            continue
        values = [max((_float(row.get(field)) or 0.0) / total, 0.0) for field, _, _ in STAGE_FIELDS]
        other = max(1.0 - sum(values), 0.0)
        labels.append(f"{_condition_label(condition)} | {_short_solver(winner)}")
        stage_rows.append(values)
        other_values.append(other)

    fig, ax = plt.subplots(figsize=(13.5, max(8.0, 0.34 * len(labels))))
    y_positions = list(range(len(labels)))
    left = [0.0] * len(labels)
    for stage_index, (_, label, color) in enumerate(STAGE_FIELDS):
        values = [row[stage_index] for row in stage_rows]
        ax.barh(y_positions, values, left=left, label=label, color=color)
        left = [previous + value for previous, value in zip(left, values)]
    ax.barh(y_positions, other_values, left=left, label="other", color="#bab0ac")

    ax.set_title(title)
    ax.set_yticks(y_positions, labels)
    ax.tick_params(axis="y", labelsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("share of curve.simulate total time")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _save(fig, output)
    return output


def _plot_stage_by_solver_panels(
    rows: Sequence[Mapping[str, Any]],
    *,
    output: Path,
    title_prefix: str,
    normalize: bool,
) -> list[Path]:
    figures: list[Path] = []
    scripts = _unique_sorted(row["script"] for row in rows)
    recordings = _unique_sorted(row["recording"] for row in rows)
    observer_scopes = _unique_sorted(_observer_scope(row) for row in rows)
    n_axons_values = sorted({int(row["n_axons"]) for row in rows})
    nx_values = sorted({int(row["nx"]) for row in rows})
    diameter_values = _unique_sorted(row["diameters"] for row in rows)
    x_values = [(nx, diameters) for nx in nx_values for diameters in diameter_values]
    by_condition = _rows_by_condition(rows)

    for script in scripts:
        for recording in recordings:
            for observer_scope in observer_scopes:
                rows_for_scope = [
                    row
                    for row in rows
                    if str(row["script"]) == script
                    and str(row["recording"]) == recording
                    and _observer_scope(row) == observer_scope
                ]
                if not rows_for_scope:
                    continue
                fig, axes = plt.subplots(
                    len(n_axons_values),
                    len(x_values),
                    figsize=(4.9 * len(x_values), 2.55 * len(n_axons_values)),
                    squeeze=False,
                    sharey=normalize,
                )
                for row_index, n_axons in enumerate(n_axons_values):
                    for col_index, (nx, diameters) in enumerate(x_values):
                        ax = axes[row_index][col_index]
                        condition = (
                            script,
                            recording,
                            observer_scope,
                            str(n_axons),
                            str(nx),
                            diameters,
                        )
                        solver_rows = by_condition.get(condition, {})
                        x_positions = list(range(len(SOLVER_ORDER)))
                        bottoms = [0.0] * len(SOLVER_ORDER)
                        totals = [
                            _float(
                                solver_rows.get(solver, {}).get(
                                    "curve_simulate_total_ms"
                                )
                            )
                            or 0.0
                            for solver in SOLVER_ORDER
                        ]
                        for _, label, color, fields in STAGE_GROUPS:
                            values = [
                                _stage_sum(solver_rows.get(solver, {}), fields)
                                for solver in SOLVER_ORDER
                            ]
                            if normalize:
                                values = [
                                    value / total if total > 0.0 else 0.0
                                    for value, total in zip(values, totals)
                                ]
                            else:
                                values = [value / 1000.0 for value in values]
                            ax.bar(
                                x_positions,
                                values,
                                bottom=bottoms,
                                width=0.78,
                                color=color,
                                label=label,
                            )
                            bottoms = [
                                bottom + value
                                for bottom, value in zip(bottoms, values)
                            ]

                        known_totals = [
                            sum(
                                _stage_sum(solver_rows.get(solver, {}), fields)
                                for _, _, _, fields in STAGE_GROUPS
                            )
                            for solver in SOLVER_ORDER
                        ]
                        other = [
                            max(total - known, 0.0)
                            for total, known in zip(totals, known_totals)
                        ]
                        if normalize:
                            other_values = [
                                value / total if total > 0.0 else 0.0
                                for value, total in zip(other, totals)
                            ]
                        else:
                            other_values = [value / 1000.0 for value in other]
                        ax.bar(
                            x_positions,
                            other_values,
                            bottom=bottoms,
                            width=0.78,
                            color="#bab0ac",
                            label="other",
                        )
                        ax.set_title(
                            f"N={n_axons}, Nx={nx}, {_short_diameters(diameters)}",
                            fontsize=9,
                        )
                        ax.set_xticks(
                            x_positions,
                            [_short_solver(solver) for solver in SOLVER_ORDER],
                            rotation=35,
                            ha="right",
                            fontsize=7,
                        )
                        ax.grid(True, axis="y", alpha=0.25)
                        if normalize:
                            ax.set_ylim(0.0, 1.0)
                        if col_index == 0:
                            ax.set_ylabel("share" if normalize else "seconds")

                handles = [
                    Patch(facecolor=color, edgecolor="none", label=label)
                    for _, label, color, _ in STAGE_GROUPS
                ]
                handles.append(Patch(facecolor="#bab0ac", edgecolor="none", label="other"))
                fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
                fig.suptitle(
                    f"{title_prefix}: {SCRIPT_LABELS.get(script, script)} / "
                    f"{RECORDING_LABELS.get(recording, recording)}"
                    f"{_observer_title_suffix(observer_scope)}"
                )
                fig.tight_layout(rect=(0, 0.10, 1, 0.94))
                suffix = "share" if normalize else "time"
                scope_suffix = (
                    ""
                    if len(observer_scopes) == 1 and observer_scope == "default"
                    else f"_obs_{observer_scope}"
                )
                path = (
                    output
                    / f"stage_{suffix}_by_solver_{script}_{recording}{scope_suffix}.png"
                )
                _save(fig, path)
                figures.append(path)
    return figures


def _write_stage_comparison_csv(
    rows: Sequence[Mapping[str, Any]],
    *,
    output: Path,
) -> Path:
    fields = [
        "script",
        "recording",
        "observer_state_scope",
        "diameters",
        "n_axons",
        "nx",
        "solver",
        "warm_mean_ms",
        "cold_ms",
        "total_ms",
    ]
    for stage_key, _, _, _ in STAGE_GROUPS:
        fields.extend((f"{stage_key}_ms", f"{stage_key}_pct"))
    fields.append("other_pct")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                str(item["script"]),
                str(item["recording"]),
                _observer_scope(item),
                str(item["diameters"]),
                int(item["n_axons"]),
                int(item["nx"]),
                SOLVER_ORDER.index(_solver_token(item))
                if _solver_token(item) in SOLVER_ORDER
                else len(SOLVER_ORDER),
            ),
        ):
            total = _float(row.get("curve_simulate_total_ms")) or 0.0
            known = 0.0
            out: dict[str, Any] = {
                "script": row.get("script", ""),
                "recording": row.get("recording", ""),
                "observer_state_scope": _observer_scope(row),
                "diameters": row.get("diameters", ""),
                "n_axons": row.get("n_axons", ""),
                "nx": row.get("nx", ""),
                "solver": _solver_token(row),
                "warm_mean_ms": row.get("curve_simulate_warm_mean_ms", ""),
                "cold_ms": row.get("curve_simulate_cold_ms", ""),
                "total_ms": row.get("curve_simulate_total_ms", ""),
            }
            for stage_key, _, _, stage_fields in STAGE_GROUPS:
                value = _stage_sum(row, stage_fields)
                known += value
                out[f"{stage_key}_ms"] = value
                out[f"{stage_key}_pct"] = value / total if total > 0.0 else ""
            out["other_pct"] = max(total - known, 0.0) / total if total > 0.0 else ""
            writer.writerow(out)
    return output


def _write_index(
    path: Path,
    *,
    summary: Path,
    rows: Sequence[Mapping[str, Any]],
    figures: Sequence[Path],
    tables: Sequence[Path] = (),
) -> None:
    winners = defaultdict(int)
    for solver_rows in _rows_by_condition(rows).values():
        winner = _winner(solver_rows)
        if winner:
            winners[winner] += 1

    lines = [
        "# Double-Cable Solver Policy Plots",
        "",
        f"Source summary: `{summary}`",
        "",
        f"Rows: `{len(rows)}` passed benchmark rows.",
        "",
        "Warm-time winner counts:",
        "",
    ]
    for solver in SOLVER_ORDER:
        if winners.get(solver):
            lines.append(f"- `{solver}`: `{winners[solver]}` groups")
    lines.extend(["", "Figures:", ""])
    for figure in figures:
        lines.append(f"- `{figure.name}`")
    if tables:
        lines.extend(["", "Tables:", ""])
        for table in tables:
            lines.append(f"- `{table.name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _conditions(
    rows: Sequence[Mapping[str, Any]]
) -> list[tuple[str, str, str, str, str, str]]:
    return sorted(
        {
            (
                str(row["script"]),
                str(row["recording"]),
                _observer_scope(row),
                str(row["n_axons"]),
                str(row["nx"]),
                str(row["diameters"]),
            )
            for row in rows
        },
        key=lambda item: (
            item[0],
            item[1],
            _observer_scope_sort_key(item[2]),
            int(item[3]),
            int(item[4]),
            item[5],
        ),
    )


def _rows_by_condition(
    rows: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str, str, str, str, str], dict[str, Mapping[str, Any]]]:
    grouped: dict[
        tuple[str, str, str, str, str, str],
        dict[str, Mapping[str, Any]],
    ] = defaultdict(dict)
    for row in rows:
        condition = (
            str(row["script"]),
            str(row["recording"]),
            _observer_scope(row),
            str(row["n_axons"]),
            str(row["nx"]),
            str(row["diameters"]),
        )
        grouped[condition][_solver_token(row)] = row
    return grouped


def _winner(rows_by_solver: Mapping[str, Mapping[str, Any]]) -> str:
    candidates: list[tuple[float, str]] = []
    for solver, row in rows_by_solver.items():
        value = _float(row.get("curve_simulate_warm_mean_ms"))
        if value is not None:
            candidates.append((value, solver))
    return min(candidates)[1] if candidates else ""


def _solver_token(row: Mapping[str, Any]) -> str:
    solver = str(row.get("solver", ""))
    block_b = str(row.get("tiled_thomas_block_b", "") or "")
    return f"{solver}_b{block_b}" if block_b else solver


def _condition_label(condition: tuple[str, str, str, str, str, str]) -> str:
    script, recording, observer_scope, n_axons, nx, diameters = condition
    observer_label = (
        ""
        if observer_scope == "default"
        else f" | obs={observer_scope}"
    )
    return (
        f"{SCRIPT_LABELS.get(script, script)} | "
        f"{RECORDING_LABELS.get(recording, recording)} | "
        f"N={n_axons} | Nx={nx} | {_short_diameters(diameters)}"
        f"{observer_label}"
    )


def _condition_x_label(nx: int, diameters: str, observer_scope: str) -> str:
    label = f"Nx={nx}\n{_short_diameters(diameters)}"
    if observer_scope != "default":
        label = f"{label}\nobs={observer_scope}"
    return label


def _short_solver(solver: str) -> str:
    return {
        "auto": "auto",
        "pcr_soa": "PCR-SoA",
        "pcr": "PCR",
        "thomas": "Thomas",
        "tiled_thomas_b32": "Triton b32",
        "tiled_thomas_b64": "Triton b64",
    }.get(solver, solver)


def _short_diameters(diameters: str) -> str:
    return {
        "same_diameter": "same",
        "different_diameters": "different",
    }.get(diameters, DIAMETER_LABELS.get(diameters, diameters))


def _observer_scope(row: Mapping[str, Any]) -> str:
    value = str(row.get("observer_state_scope", "") or "").strip()
    return value or "default"


def _observer_title_suffix(observer_scope: str) -> str:
    return "" if observer_scope == "default" else f" / obs={observer_scope}"


def _scope_linestyle(observer_scope: str) -> str:
    return ":" if observer_scope != "default" else "-"


def _diameter_scope_linestyle(diameters: str, observer_scope: str) -> str:
    if observer_scope != "default":
        return "-." if diameters == "same_diameter" else ":"
    return "-" if diameters == "same_diameter" else "--"


def _observer_scope_sort_key(observer_scope: str) -> tuple[int, str]:
    return (0, "") if observer_scope == "default" else (1, observer_scope)


def _format_ms(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.1f}s"
    return f"{value:.0f}ms"


def _float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stage_sum(row: Mapping[str, Any], fields: Sequence[str]) -> float:
    return sum(_float(row.get(field)) or 0.0 for field in fields)


def _positive_or_nan(value: float | None) -> float:
    if value is None or value <= 0:
        return math.nan
    return value


def _unique_sorted(values: Sequence[str]) -> list[str]:
    return sorted(set(values))


if __name__ == "__main__":
    raise SystemExit(main())
