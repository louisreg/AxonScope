from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "axonscope-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "axonscope-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_REPORT_DIR = Path("benchmark/reports/nrv_performance")

METRICS = {
    "warm_solve": {
        "as": "as_warm_solve_median_s",
        "nrv": "nrv_repeat_median_s",
        "speedup": "speedup_nrv_over_as_warm",
        "label": "Warm solve",
    },
    "warm_total": {
        "as": "as_warm_total_median_s",
        "nrv": "nrv_repeat_total_median_s",
        "speedup": "speedup_nrv_over_as_total_warm",
        "label": "Warm solve + materialization",
    },
    "first_solve": {
        "as": "as_first_solve_s",
        "nrv": "nrv_simulate_s",
        "speedup": "speedup_nrv_over_as_first",
        "label": "First solve",
    },
    "first_total": {
        "as": "as_total_first_s",
        "nrv": "nrv_total_s",
        "speedup": "speedup_nrv_over_as_total_first",
        "label": "First solve + materialization",
    },
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot AxonScope-vs-NRV performance CSV files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="CSV files produced by nrv_axonscope_grid.py.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for plot outputs.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to the first input stem.")
    parser.add_argument("--metric", choices=tuple(METRICS), default="warm_total", help="Timing pair to plot.")
    parser.add_argument(
        "--x",
        choices=("axon_nx", "input_nx", "nodes"),
        default="axon_nx",
        help="X axis for the sweep.",
    )
    parser.add_argument("--title", default="AxonScope vs NRV performance", help="Plot title.")
    args = parser.parse_args(argv)

    prefix = args.prefix or args.inputs[0].stem
    report = write_performance_plot(
        args.inputs,
        args.out_dir,
        prefix=prefix,
        metric=args.metric,
        x_axis=args.x,
        title=args.title,
    )

    print("=== NRV performance plot ===")
    print(f"plot   : {report['plot']}")
    print(f"summary: {report['summary_csv']}")


def write_performance_plot(
    inputs: Sequence[Path],
    out_dir: Path,
    *,
    prefix: str = "nrv_performance",
    metric: str = "warm_total",
    x_axis: str = "axon_nx",
    title: str = "AxonScope vs NRV performance",
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_performance_csvs(inputs)
    if df.empty:
        raise ValueError("No performance rows found.")
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric {metric!r}.")

    metric_spec = METRICS[metric]
    plot_df = _select_plot_rows(df, metric_spec=metric_spec, x_axis=x_axis)
    if plot_df.empty:
        raise ValueError(f"No rows contain the required columns for metric {metric!r}.")

    summary_csv = out_dir / f"{prefix}_summary.csv"
    plot_df.to_csv(summary_csv, index=False)

    plot_path = out_dir / f"{prefix}_{metric}_vs_nrv.png"
    _write_plot(plot_df, plot_path, metric_spec=metric_spec, x_axis=x_axis, title=title)
    return {"plot": plot_path, "summary_csv": summary_csv}


def load_performance_csvs(inputs: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for index, path in enumerate(inputs):
        frame = pd.read_csv(path)
        frame["source"] = path.stem if len(inputs) == 1 else f"{index + 1:02d}_{path.stem}"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _select_plot_rows(df: pd.DataFrame, *, metric_spec: dict[str, str], x_axis: str) -> pd.DataFrame:
    required = [x_axis, "model", metric_spec["as"], metric_spec["nrv"], metric_spec["speedup"]]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}.")

    columns = [
        "source",
        "model",
        "dt_ms",
        "tsim_ms",
        "diameter_um",
        "input_nx",
        "nodes",
        "axon_nx",
        metric_spec["as"],
        metric_spec["nrv"],
        metric_spec["speedup"],
    ]
    columns = [column for column in columns if column in df.columns]
    plot_df = df[columns].copy()
    for column in (x_axis, metric_spec["as"], metric_spec["nrv"], metric_spec["speedup"]):
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_axis, metric_spec["as"], metric_spec["nrv"]])
    plot_df["case_label"] = plot_df.apply(_case_label, axis=1)
    return plot_df.sort_values(["case_label", x_axis]).reset_index(drop=True)


def _write_plot(
    df: pd.DataFrame,
    path: Path,
    *,
    metric_spec: dict[str, str],
    x_axis: str,
    title: str,
) -> None:
    groups = list(df.groupby("case_label", sort=False))
    fig, (time_ax, speed_ax) = plt.subplots(
        2,
        1,
        figsize=(9.0, max(5.4, 1.0 + 0.55 * len(groups))),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    cmap = plt.get_cmap("tab10")

    for index, (label, group) in enumerate(groups):
        color = cmap(index % 10)
        x = group[x_axis].astype(float).to_numpy()
        as_values = group[metric_spec["as"]].astype(float).to_numpy()
        nrv_values = group[metric_spec["nrv"]].astype(float).to_numpy()
        speedup = pd.to_numeric(group[metric_spec["speedup"]], errors="coerce").to_numpy()
        series_suffix = "" if len(groups) == 1 else f" ({label})"

        time_ax.plot(x, as_values, marker="o", color=color, linewidth=2.0, label=f"AxonScope{series_suffix}")
        time_ax.plot(x, nrv_values, marker="s", color=color, linewidth=1.8, linestyle="--", label=f"NRV{series_suffix}")
        speed_ax.plot(x, speedup, marker="o", color=color, linewidth=2.0, label=label)

    if _all_positive(df[metric_spec["as"]]) and _all_positive(df[metric_spec["nrv"]]):
        time_ax.set_yscale("log")
    time_ax.set_title(title)
    time_ax.set_ylabel(f"{metric_spec['label']} [s]")
    time_ax.grid(axis="y", alpha=0.25)
    time_ax.legend(loc="best", fontsize=8)

    speed_ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.55)
    speed_ax.set_ylabel("NRV / AxonScope")
    speed_ax.set_xlabel(_x_label(x_axis))
    speed_ax.grid(axis="y", alpha=0.25)
    if len(groups) > 1:
        speed_ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _case_label(row: pd.Series) -> str:
    parts = [str(row.get("source", "")), str(row.get("model", ""))]
    if pd.notna(row.get("dt_ms")):
        parts.append(f"dt={float(row['dt_ms']):g}")
    if pd.notna(row.get("tsim_ms")):
        parts.append(f"tsim={float(row['tsim_ms']):g}")
    if pd.notna(row.get("diameter_um")):
        parts.append(f"d={float(row['diameter_um']):g}")
    return " ".join(part for part in parts if part)


def _x_label(x_axis: str) -> str:
    return {
        "axon_nx": "AxonScope compartments",
        "input_nx": "Requested unmyelinated Nx",
        "nodes": "MRG nodes",
    }[x_axis]


def _all_positive(values: Any) -> bool:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return bool((numeric > 0.0).all()) if not numeric.empty else False


if __name__ == "__main__":
    main()
