from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


STATE_FIELDS = (
    "diag00",
    "diag01",
    "diag10",
    "diag11",
    "lower00",
    "lower01",
    "lower10",
    "lower11",
    "upper00",
    "upper01",
    "upper10",
    "upper11",
    "rhs0",
    "rhs1",
)
HLO_FIELDS = (
    "hlo_fusion_name",
    "hlo_output_arrays",
    "hlo_output_mib",
    "hlo_input_mib",
    "hlo_io_mib",
    "hlo_gather",
    "hlo_select",
)
SUMMARY_FIELDS = (
    "stage_index",
    "stride",
    "valid_left_columns",
    "valid_right_columns",
    "state_arrays",
    "state_mib",
    "state_fields",
    "explicit_neighbor_reads",
    "output_state_arrays",
    "output_state_mib",
    *HLO_FIELDS,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark-only audit of the batch-native double-cable PCR/SoA "
            "stage state. This is structural cartography, not a runtime solver."
        )
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--hlo-fusion-summary",
        type=Path,
        help="Optional hlo_fusion_summary.csv to attach compiler fusion output estimates.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    hlo_rows = _read_hlo_rows(args.hlo_fusion_summary) if args.hlo_fusion_summary else ()
    rows = pcr_soa_stage_state_rows(
        batch_size=args.batch_size,
        nx=args.nx,
        dtype=args.dtype,
        hlo_rows=hlo_rows,
    )
    write_outputs(
        args.output,
        rows=rows,
        metadata={
            "batch_size": args.batch_size,
            "nx": args.nx,
            "dtype": args.dtype,
            "hlo_fusion_summary": str(args.hlo_fusion_summary) if args.hlo_fusion_summary else None,
        },
    )
    print(f"wrote: {args.output / 'pcr_soa_stage_state_summary.csv'}")
    print(f"wrote: {args.output / 'pcr_soa_stage_state_report.md'}")
    return 0


def pcr_soa_stage_state_rows(
    *,
    batch_size: int,
    nx: int,
    dtype: str = "float32",
    hlo_rows: Sequence[dict[str, Any]] = (),
) -> tuple[dict[str, Any], ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if nx <= 0:
        raise ValueError("nx must be positive.")

    bytes_per_array = int(batch_size) * int(nx) * _dtype_nbytes(dtype)
    hlo_stage_rows = _pcr_stage_hlo_rows(hlo_rows)
    rows: list[dict[str, Any]] = []
    for stage_index, stride in enumerate(_pcr_strides(nx)):
        hlo = hlo_stage_rows[stage_index] if stage_index < len(hlo_stage_rows) else {}
        rows.append(
            {
                "stage_index": stage_index,
                "stride": stride,
                "valid_left_columns": max(nx - stride, 0),
                "valid_right_columns": max(nx - stride, 0),
                "state_arrays": len(STATE_FIELDS),
                "state_mib": _bytes_to_mib(len(STATE_FIELDS) * bytes_per_array),
                "state_fields": ";".join(STATE_FIELDS),
                "explicit_neighbor_reads": 28,
                "output_state_arrays": len(STATE_FIELDS),
                "output_state_mib": _bytes_to_mib(len(STATE_FIELDS) * bytes_per_array),
                **_hlo_projection(hlo),
            }
        )
    return tuple(rows)


def write_outputs(
    output: Path,
    *,
    rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "pcr_soa_stage_state_summary.csv", SUMMARY_FIELDS, rows)
    (output / "pcr_soa_stage_state_metrics.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "stage_count": len(rows),
                "max_state_mib": max((float(row["state_mib"]) for row in rows), default=0.0),
                "max_hlo_io_mib": max((float(row["hlo_io_mib"] or 0.0) for row in rows), default=0.0),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output / "pcr_soa_stage_state_report.md", rows, metadata)


def _pcr_strides(nx: int) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    while stride < nx:
        strides.append(stride)
        stride *= 2
    return tuple(strides)


def _pcr_stage_hlo_rows(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    selected = [
        row
        for row in rows
        if row.get("stage") == "block_solve"
        and row.get("variant") == "pcr_soa_batched"
        and "loop_select_subtract" in str(row.get("fusion_name", ""))
    ]
    return tuple(sorted(selected, key=lambda row: _float(row.get("line")) or 0.0))


def _hlo_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "hlo_fusion_name": row.get("fusion_name", ""),
        "hlo_output_arrays": _int_or_empty(row.get("output_arrays")),
        "hlo_output_mib": _bytes_to_mib(_float(row.get("output_bytes_estimate")) or 0.0)
        if row
        else "",
        "hlo_input_mib": _bytes_to_mib(_float(row.get("computation_input_bytes_estimate")) or 0.0)
        if row
        else "",
        "hlo_io_mib": _bytes_to_mib(_float(row.get("computation_io_bytes_estimate")) or 0.0)
        if row
        else "",
        "hlo_gather": _int_or_empty(row.get("count_gather")),
        "hlo_select": _int_or_empty(row.get("count_select")),
    }


def _read_hlo_rows(path: Path) -> tuple[dict[str, Any], ...]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return tuple(csv.DictReader(fh))


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_report(path: Path, rows: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> None:
    lines = [
        "# PCR/SoA Stage State Audit",
        "",
        "Benchmark-only structural audit of the batch-native double-cable PCR/SoA solver.",
        "",
        "## Context",
        "",
        f"- Batch size: `{metadata['batch_size']}`",
        f"- Nx: `{metadata['nx']}`",
        f"- Dtype: `{metadata['dtype']}`",
        f"- HLO fusion summary: `{metadata.get('hlo_fusion_summary')}`",
        "",
        "## Stage State",
        "",
        "| stage | stride | state arrays | state MiB | HLO fusion | HLO outputs | HLO I/O MiB | gather | select |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {stage_index} | {stride} | {state_arrays} | {state_mib:.2f} | {hlo_fusion_name} | "
            "{hlo_output_arrays} | {hlo_io_mib} | {hlo_gather} | {hlo_select} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Algorithmic PCR/SoA state currently carries 14 batch-space arrays: "
            "`diag*`, `lower*`, `upper*`, and `rhs*`.",
            "- The explicit stage body performs 28 neighbor indexed reads before XLA "
            "optimization. HLO rows, when attached, are compiler outputs and their "
            "mapping to PCR strides is approximate.",
            "- Use this report to choose benchmark-only state/staging variants; do not "
            "promote solver policy from this structural estimate alone.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dtype_nbytes(dtype: str) -> int:
    if dtype == "float64":
        return 8
    return 4


def _bytes_to_mib(value: float | int) -> float:
    return float(value) / (1024.0 * 1024.0)


def _float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_empty(value: Any) -> int | str:
    parsed = _float(value)
    return "" if parsed is None else int(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
