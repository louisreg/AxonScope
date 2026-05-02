from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

from axonscope.benchmarking import compare_benchmark_results, load_benchmark_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two AxonScope solver benchmark JSON files.")
    parser.add_argument("baseline", type=Path, help="Baseline benchmark JSON.")
    parser.add_argument("current", type=Path, help="Current benchmark JSON.")
    parser.add_argument("--build-threshold", type=float, default=0.15, help="Relative construction regression threshold.")
    parser.add_argument("--first-threshold", type=float, default=0.20, help="Relative first-solve regression threshold.")
    parser.add_argument("--total-first-threshold", type=float, default=0.20, help="Relative first solve+materialize regression threshold.")
    parser.add_argument("--warm-threshold", type=float, default=0.10, help="Relative warm-solve regression threshold.")
    parser.add_argument("--warm-total-threshold", type=float, default=0.10, help="Relative warm solve+materialize regression threshold.")
    parser.add_argument("--rss-threshold", type=float, default=0.15, help="Relative RSS-delta regression threshold.")
    parser.add_argument("--output-atol", type=float, default=5e-2, help="Absolute tolerance for output min/max/mean changes.")
    parser.add_argument("--output-rtol", type=float, default=1e-6, help="Relative tolerance for output min/max/mean changes.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON comparison report.")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit non-zero if a regression is detected.")
    args = parser.parse_args()

    baseline_results, baseline_metadata = load_benchmark_results(args.baseline)
    current_results, current_metadata = load_benchmark_results(args.current)
    rows = compare_benchmark_results(
        baseline_results,
        current_results,
        thresholds={
            "construction.mean_s": args.build_threshold,
            "first_solve_s": args.first_threshold,
            "total_first_s": args.total_first_threshold,
            "warm_solve.mean_s": args.warm_threshold,
            "warm_total.mean_s": args.warm_total_threshold,
            "rss_first_solve_delta_mb": args.rss_threshold,
        },
        output_atol=args.output_atol,
        output_rtol=args.output_rtol,
    )

    _print_metadata_delta(baseline_metadata, current_metadata)
    _print_table(rows)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": str(args.baseline),
            "current": str(args.current),
            "baseline_metadata": baseline_metadata,
            "current_metadata": current_metadata,
            "rows": [
                {
                    "case_name": row.case_name,
                    "solver_name": row.solver_name,
                    "status": row.status,
                    "notes": list(row.notes),
                    "metrics": [metric.__dict__ for metric in row.metrics],
                }
                for row in rows
            ],
        }
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    has_regression = any(row.status == "regression" for row in rows)
    if has_regression and args.fail_on_regression:
        raise SystemExit(1)


def _print_metadata_delta(baseline_metadata: dict, current_metadata: dict) -> None:
    print("=== Benchmark comparison ===")
    baseline_git = baseline_metadata.get("git", {}) if isinstance(baseline_metadata, dict) else {}
    current_git = current_metadata.get("git", {}) if isinstance(current_metadata, dict) else {}
    if baseline_git or current_git:
        print(
            "git: "
            f"{baseline_git.get('sha', '?')} ({baseline_git.get('branch', '?')}) "
            "-> "
            f"{current_git.get('sha', '?')} ({current_git.get('branch', '?')})"
        )
    baseline_jax = baseline_metadata.get("jax") if isinstance(baseline_metadata, dict) else None
    current_jax = current_metadata.get("jax") if isinstance(current_metadata, dict) else None
    if baseline_jax or current_jax:
        print(f"jax: {baseline_jax or '?'} -> {current_jax or '?'}")


def _print_table(rows) -> None:
    headers = ("case", "solver", "build", "first", "total", "warm", "warm_total", "rss", "status")
    print(
        f"{headers[0]:32s} {headers[1]:20s} "
        f"{headers[2]:>9s} {headers[3]:>9s} {headers[4]:>9s} "
        f"{headers[5]:>9s} {headers[6]:>10s} {headers[7]:>9s} {headers[8]:>16s}"
    )
    print("-" * 137)
    for row in rows:
        metrics = {metric.metric: metric for metric in row.metrics}
        print(
            f"{row.case_name:32s} {row.solver_name:20s} "
            f"{_fmt_metric(metrics.get('construction.mean_s')):>9s} "
            f"{_fmt_metric(metrics.get('first_solve_s')):>9s} "
            f"{_fmt_metric(metrics.get('total_first_s')):>9s} "
            f"{_fmt_metric(metrics.get('warm_solve.mean_s')):>9s} "
            f"{_fmt_metric(metrics.get('warm_total.mean_s')):>10s} "
            f"{_fmt_metric(metrics.get('rss_first_solve_delta_mb')):>9s} "
            f"{row.status:>16s}"
        )
        for note in row.notes:
            print(f"  note: {note}")


def _fmt_metric(metric) -> str:
    if metric is None or metric.relative_delta is None:
        return "n/a"
    marker = "!" if metric.status == "regression" else ""
    return f"{100.0 * metric.relative_delta:+.1f}%{marker}"


if __name__ == "__main__":
    main()
