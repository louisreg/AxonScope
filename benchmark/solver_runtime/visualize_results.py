from __future__ import annotations

import argparse
import html
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "axonscope-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "axonscope-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from axonscope.benchmarking import load_benchmark_results


DEFAULT_REPORT_DIR = Path("benchmark/reports/solver_runtime")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML report from AxonScope solver benchmark JSON files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON files to visualize.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for the HTML report, summary CSV, and plots.",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to the first input stem.")
    parser.add_argument("--title", default="AxonScope solver benchmark report", help="HTML report title.")
    args = parser.parse_args()

    prefix = args.prefix or _default_prefix(args.inputs)
    report = write_benchmark_report(args.inputs, args.out_dir, prefix=prefix, title=args.title)

    print("=== Solver benchmark report ===")
    print(f"html   : {report['html']}")
    print(f"summary: {report['summary_csv']}")
    for plot in report["plots"]:
        print(f"plot   : {plot}")


def write_benchmark_report(
    inputs: Sequence[Path],
    out_dir: Path,
    *,
    prefix: str = "solver_runtime",
    title: str = "AxonScope solver benchmark report",
) -> dict[str, Any]:
    """Write a static benchmark report and return generated paths."""

    out_dir.mkdir(parents=True, exist_ok=True)
    df = flatten_benchmark_files(inputs)
    if df.empty:
        raise ValueError("No benchmark rows found.")

    summary_csv = out_dir / f"{prefix}_summary.csv"
    df.to_csv(summary_csv, index=False)

    plots = _write_plots(df, out_dir, prefix)
    html_path = out_dir / f"{prefix}.html"
    _write_html_report(
        df,
        html_path,
        plots=plots,
        title=title,
        summary_csv=summary_csv,
        inputs=inputs,
    )

    return {
        "html": html_path,
        "summary_csv": summary_csv,
        "plots": plots,
    }


def flatten_benchmark_files(inputs: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_index, path in enumerate(inputs):
        results, metadata = load_benchmark_results(path)
        source = path.stem
        if len(inputs) > 1:
            source = f"{source_index + 1:02d}_{source}"
        for result in results:
            rows.append(_flatten_result(path, source, result, metadata))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    sort_cols = [col for col in ("case_name", "solver_name", "source") if col in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def _flatten_result(
    path: Path,
    source: str,
    result: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    git = metadata.get("git", {}) if isinstance(metadata, Mapping) else {}
    if not isinstance(git, Mapping):
        git = {}

    return {
        "source": source,
        "source_path": str(path),
        "created_at": metadata.get("created_at"),
        "git_sha": git.get("sha"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "jax": metadata.get("jax"),
        "case_name": result.get("case_name"),
        "solver_name": result.get("solver_name"),
        "model": _nested(result, "metadata.model"),
        "stimulation": _nested(result, "metadata.stimulation"),
        "tsim_ms": _number(result.get("tsim_ms")),
        "dt_ms": _number(result.get("dt_ms")),
        "construction_mean_s": _number(_nested(result, "construction.mean_s")),
        "first_solve_s": _number(result.get("first_solve_s")),
        "materialize_first_s": _number(result.get("materialize_first_s")),
        "total_first_s": _number(result.get("total_first_s")),
        "compile_s_estimate": _number(result.get("compile_s_estimate")),
        "warm_solve_mean_s": _number(_nested(result, "warm_solve.mean_s")),
        "warm_solve_min_s": _number(_nested(result, "warm_solve.min_s")),
        "warm_solve_max_s": _number(_nested(result, "warm_solve.max_s")),
        "warm_materialize_mean_s": _number(_nested(result, "warm_materialize.mean_s")),
        "warm_total_mean_s": _number(_nested(result, "warm_total.mean_s")),
        "rss_first_solve_delta_mb": _number(result.get("rss_first_solve_delta_mb")),
        "vm_min_mV": _number(_nested(result, "output.vm_min_mV")),
        "vm_max_mV": _number(_nested(result, "output.vm_max_mV")),
        "vm_mean_mV": _number(_nested(result, "output.vm_mean_mV")),
        "vm_shape": _nested(result, "output.vm_shape"),
    }


def _write_plots(df: pd.DataFrame, out_dir: Path, prefix: str) -> list[Path]:
    plots: list[Path] = []
    plots.append(
        _bar_plot(
            df,
            "warm_solve_mean_s",
            out_dir / f"{prefix}_warm_solve_s.png",
            title="Warm solve time",
            ylabel="seconds",
        )
    )
    plots.append(
        _first_vs_warm_plot(
            df,
            out_dir / f"{prefix}_first_vs_warm_s.png",
        )
    )
    if df["compile_s_estimate"].notna().any():
        plots.append(
            _bar_plot(
                df,
                "compile_s_estimate",
                out_dir / f"{prefix}_compile_estimate_s.png",
                title="Estimated compilation time",
                ylabel="seconds",
            )
        )
    if df["rss_first_solve_delta_mb"].notna().any():
        plots.append(
            _bar_plot(
                df,
                "rss_first_solve_delta_mb",
                out_dir / f"{prefix}_rss_delta_mb.png",
                title="First solve RSS delta",
                ylabel="MB",
            )
        )
    return plots


def _bar_plot(df: pd.DataFrame, metric: str, path: Path, *, title: str, ylabel: str) -> Path:
    plot_df = df.dropna(subset=[metric]).reset_index(drop=True)
    if plot_df.empty:
        return path

    labels = _row_labels(plot_df)
    values = plot_df[metric].astype(float).to_numpy()
    solvers = list(dict.fromkeys(plot_df["solver_name"].fillna("?").astype(str)))

    fig, ax = plt.subplots(figsize=_figure_size(len(plot_df)))
    x = np.arange(len(plot_df))
    cmap = plt.get_cmap("tab10")
    for solver_index, solver in enumerate(solvers):
        mask = plot_df["solver_name"].fillna("?").astype(str).to_numpy() == solver
        ax.bar(x[mask], values[mask], color=cmap(solver_index % 10), label=solver)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    if len(solvers) > 1:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _first_vs_warm_plot(df: pd.DataFrame, path: Path) -> Path:
    plot_df = df.dropna(subset=["first_solve_s", "warm_solve_mean_s"]).reset_index(drop=True)
    if plot_df.empty:
        return path

    labels = _row_labels(plot_df)
    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=_figure_size(len(plot_df)))
    ax.bar(x - width / 2.0, plot_df["first_solve_s"].astype(float), width, label="first solve", color="#4C78A8")
    ax.bar(x + width / 2.0, plot_df["warm_solve_mean_s"].astype(float), width, label="warm solve", color="#F58518")
    ax.set_title("First solve vs warm solve")
    ax.set_ylabel("seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_html_report(
    df: pd.DataFrame,
    html_path: Path,
    *,
    plots: Sequence[Path],
    title: str,
    summary_csv: Path,
    inputs: Sequence[Path],
) -> None:
    table_cols = [
        "source",
        "case_name",
        "solver_name",
        "model",
        "stimulation",
        "construction_mean_s",
        "first_solve_s",
        "compile_s_estimate",
        "warm_solve_mean_s",
        "warm_total_mean_s",
        "rss_first_solve_delta_mb",
        "vm_min_mV",
        "vm_max_mV",
        "vm_mean_mV",
    ]
    table_cols = [col for col in table_cols if col in df.columns]
    table_df = df[table_cols].copy()
    numeric_cols = table_df.select_dtypes(include="number").columns
    table_df[numeric_cols] = table_df[numeric_cols].round(6)

    images = "\n".join(
        f'<section class="plot"><h2>{html.escape(_plot_title(plot))}</h2>'
        f'<img src="{html.escape(plot.name)}" alt="{html.escape(_plot_title(plot))}"></section>'
        for plot in plots
        if plot.exists()
    )
    sources = "".join(f"<li>{html.escape(str(path))}</li>" for path in inputs)
    table = table_df.to_html(index=False, escape=True, classes="metrics")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
      color: #17202a;
      background: #f6f7f9;
    }}
    body {{
      margin: 0;
      padding: 32px;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 8px;
    }}
    h2 {{
      font-size: 18px;
      margin: 28px 0 12px;
    }}
    .summary {{
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      margin: 18px 0 24px;
      color: #3d4b5c;
    }}
    .summary span {{
      background: #ffffff;
      border: 1px solid #dde3ea;
      border-radius: 6px;
      padding: 8px 10px;
    }}
    .plot img {{
      display: block;
      width: 100%;
      max-width: 1120px;
      background: #ffffff;
      border: 1px solid #dde3ea;
      border-radius: 6px;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: #ffffff;
      border: 1px solid #dde3ea;
      border-radius: 6px;
    }}
    table.metrics {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    table.metrics th,
    table.metrics td {{
      padding: 8px 10px;
      border-bottom: 1px solid #e8edf3;
      text-align: right;
      white-space: nowrap;
    }}
    table.metrics th:first-child,
    table.metrics td:first-child,
    table.metrics th:nth-child(2),
    table.metrics td:nth-child(2),
    table.metrics th:nth-child(3),
    table.metrics td:nth-child(3),
    table.metrics th:nth-child(4),
    table.metrics td:nth-child(4),
    table.metrics th:nth-child(5),
    table.metrics td:nth-child(5) {{
      text-align: left;
    }}
    ul {{
      padding-left: 20px;
    }}
    a {{
      color: #2457a7;
    }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <div class="summary">
    <span>{len(df)} result rows</span>
    <span>{df["source"].nunique()} source file(s)</span>
    <span><a href="{html.escape(summary_csv.name)}">summary CSV</a></span>
  </div>
  {images}
  <h2>Metrics</h2>
  <div class="table-wrap">
    {table}
  </div>
  <h2>Inputs</h2>
  <ul>{sources}</ul>
</main>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")


def _row_labels(df: pd.DataFrame) -> list[str]:
    labels = []
    for _, row in df.iterrows():
        source = str(row.get("source", ""))
        case = str(row.get("case_name", "?"))
        solver = str(row.get("solver_name", "?"))
        labels.append(f"{case}\n{solver}\n{source}")
    return labels


def _figure_size(n_rows: int) -> tuple[float, float]:
    return max(8.0, 0.86 * float(n_rows)), 5.2


def _plot_title(path: Path) -> str:
    stem = path.stem
    for suffix in (
        "_warm_solve_s",
        "_first_vs_warm_s",
        "_compile_estimate_s",
        "_rss_delta_mb",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            label = suffix.strip("_").replace("_", " ")
            return label.capitalize()
    return stem.replace("_", " ").capitalize()


def _default_prefix(inputs: Sequence[Path]) -> str:
    if len(inputs) == 1:
        return inputs[0].stem
    return "solver_runtime_report"


def _nested(data: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = data
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


if __name__ == "__main__":
    main()
