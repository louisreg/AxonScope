"""Summarize exact double-cable linear-solver benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


GROUP_COLUMNS = ("source", "batch_size", "nx", "dtype")
SOLVER_COLUMNS = ("requested_solver", "resolved_solver", "kernel_solver")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path, nargs="+")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV path for the summarized crossover table.",
    )
    args = parser.parse_args(argv)

    rows = []
    for path in args.summary_csv:
        rows.extend(load_rows(path))

    summary = summarize_rows(rows)
    print_summary(summary)
    if args.out is not None:
        write_summary(args.out, summary)
        print(f"\nwrote: {args.out}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load one benchmark summary CSV and attach a source label."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            row = dict(row)
            row["source"] = _source_label(path)
            row["batch_size"] = int(row["batch_size"])
            row["nx"] = int(row["nx"])
            row["steady_median_ms"] = float(row["steady_median_ms"])
            row["node_solves_per_s"] = float(row["node_solves_per_s"])
            rows.append(row)
        return rows


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one summary row per source/B/Nx/dtype group."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[column] for column in GROUP_COLUMNS)
        grouped[key].append(dict(row))

    summary = []
    for key, group in sorted(grouped.items(), key=_group_sort_key):
        fastest = min(group, key=lambda row: row["steady_median_ms"])
        thomas = _first(group, requested_solver="thomas")
        adaptive = _first(group, requested_solver="pcr_adaptive")
        pcr = _first(group, requested_solver="pcr")
        pcr_soa = _first(group, requested_solver="pcr_soa")

        fastest_ms = fastest["steady_median_ms"]
        thomas_ms = _maybe_ms(thomas)
        adaptive_ms = _maybe_ms(adaptive)
        summary.append(
            {
                "source": key[0],
                "batch_size": key[1],
                "nx": key[2],
                "dtype": key[3],
                "fastest_requested_solver": fastest["requested_solver"],
                "fastest_kernel_solver": fastest["kernel_solver"],
                "fastest_median_ms": fastest_ms,
                "thomas_median_ms": thomas_ms,
                "pcr_median_ms": _maybe_ms(pcr),
                "pcr_soa_median_ms": _maybe_ms(pcr_soa),
                "pcr_adaptive_kernel_solver": ""
                if adaptive is None
                else adaptive["kernel_solver"],
                "pcr_adaptive_median_ms": adaptive_ms,
                "thomas_over_fastest_x": _ratio(thomas_ms, fastest_ms),
                "adaptive_over_fastest_x": _ratio(adaptive_ms, fastest_ms),
                "fastest_node_solves_per_s": fastest["node_solves_per_s"],
            }
        )
    return summary


def print_summary(rows: Sequence[dict[str, Any]]) -> None:
    """Print a compact human-readable summary."""

    if not rows:
        print("No rows.")
        return

    header = (
        "source",
        "B",
        "Nx",
        "dtype",
        "fastest",
        "fast ms",
        "thomas/fast",
        "adaptive",
        "adapt/fast",
    )
    print(
        f"{header[0]:>8} {header[1]:>6} {header[2]:>4} {header[3]:>7} "
        f"{header[4]:>18} {header[5]:>9} {header[6]:>12} "
        f"{header[7]:>18} {header[8]:>11}"
    )
    print("-" * 108)
    for row in rows:
        fastest = f"{row['fastest_requested_solver']}({row['fastest_kernel_solver']})"
        adaptive = (
            ""
            if row["pcr_adaptive_median_ms"] == ""
            else f"pcr_adaptive({row['pcr_adaptive_kernel_solver']})"
        )
        print(
            f"{str(row['source'])[:8]:>8} "
            f"{row['batch_size']:6d} {row['nx']:4d} {row['dtype']:>7} "
            f"{fastest[:18]:>18} {row['fastest_median_ms']:9.3f} "
            f"{_format_ratio(row['thomas_over_fastest_x']):>12} "
            f"{adaptive[:18]:>18} {_format_ratio(row['adaptive_over_fastest_x']):>11}"
        )


def write_summary(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write summary rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "source",
        "batch_size",
        "nx",
        "dtype",
        "fastest_requested_solver",
        "fastest_kernel_solver",
        "fastest_median_ms",
        "thomas_median_ms",
        "pcr_median_ms",
        "pcr_soa_median_ms",
        "pcr_adaptive_kernel_solver",
        "pcr_adaptive_median_ms",
        "thomas_over_fastest_x",
        "adaptive_over_fastest_x",
        "fastest_node_solves_per_s",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _source_label(path: Path) -> str:
    if path.name == "summary.csv":
        return path.parent.name
    return path.stem


def _first(group: Sequence[dict[str, Any]], *, requested_solver: str) -> dict[str, Any] | None:
    for row in group:
        if row["requested_solver"] == requested_solver:
            return row
    return None


def _maybe_ms(row: dict[str, Any] | None) -> float | str:
    return "" if row is None else float(row["steady_median_ms"])


def _ratio(numerator: float | str, denominator: float | str) -> float | str:
    if numerator == "" or denominator == "":
        return ""
    denominator = float(denominator)
    if denominator == 0.0:
        return ""
    return float(numerator) / denominator


def _format_ratio(value: float | str) -> str:
    if value == "":
        return ""
    return f"{float(value):.2f}x"


def _group_sort_key(item: tuple[tuple[Any, ...], list[dict[str, Any]]]) -> tuple[Any, ...]:
    key, _ = item
    return key[0], key[3], int(key[1]), int(key[2])


if __name__ == "__main__":
    main(sys.argv[1:])

