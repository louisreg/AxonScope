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


PCR_SOA_BATCHED_STATE_FIELDS = (
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
PCR_SOA_SYMMETRIC_BATCHED_STATE_FIELDS = (
    "diag00",
    "diag01",
    "diag10",
    "diag11",
    "lower00",
    "lower01",
    "lower10",
    "lower11",
    "rhs0",
    "rhs1",
)
VARIANT_SPECS = {
    "pcr_soa_batched": {
        "state_fields": PCR_SOA_BATCHED_STATE_FIELDS,
        "explicit_neighbor_reads": 28,
        "description": (
            "current batch-native PCR/SoA solver carrying diagonal, lower, "
            "upper, and RHS state"
        ),
    },
    "pcr_soa_symmetric_batched": {
        "state_fields": PCR_SOA_SYMMETRIC_BATCHED_STATE_FIELDS,
        "explicit_neighbor_reads": 24,
        "description": (
            "benchmark-only symmetric candidate carrying diagonal, lower, "
            "and RHS state while reconstructing upper couplings"
        ),
    },
}
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
    "variant",
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
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_SPECS),
        default="pcr_soa_batched",
        help="Solver variant to map against HLO fusion rows.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    hlo_rows = _read_hlo_rows(args.hlo_fusion_summary) if args.hlo_fusion_summary else ()
    rows = pcr_soa_stage_state_rows(
        batch_size=args.batch_size,
        nx=args.nx,
        dtype=args.dtype,
        variant=args.variant,
        hlo_rows=hlo_rows,
    )
    write_outputs(
        args.output,
        rows=rows,
        metadata={
            "batch_size": args.batch_size,
            "nx": args.nx,
            "dtype": args.dtype,
            "variant": args.variant,
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
    variant: str = "pcr_soa_batched",
    hlo_rows: Sequence[dict[str, Any]] = (),
) -> tuple[dict[str, Any], ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if nx <= 0:
        raise ValueError("nx must be positive.")

    spec = _variant_spec(variant)
    state_fields = spec["state_fields"]
    explicit_neighbor_reads = spec["explicit_neighbor_reads"]
    bytes_per_array = int(batch_size) * int(nx) * _dtype_nbytes(dtype)
    hlo_stage_rows = _pcr_stage_hlo_rows(hlo_rows, variant=variant)
    rows: list[dict[str, Any]] = []
    for stage_index, stride in enumerate(_pcr_strides(nx)):
        hlo = hlo_stage_rows[stage_index] if stage_index < len(hlo_stage_rows) else {}
        rows.append(
            {
                "stage_index": stage_index,
                "variant": variant,
                "stride": stride,
                "valid_left_columns": max(nx - stride, 0),
                "valid_right_columns": max(nx - stride, 0),
                "state_arrays": len(state_fields),
                "state_mib": _bytes_to_mib(len(state_fields) * bytes_per_array),
                "state_fields": ";".join(state_fields),
                "explicit_neighbor_reads": explicit_neighbor_reads,
                "output_state_arrays": len(state_fields),
                "output_state_mib": _bytes_to_mib(len(state_fields) * bytes_per_array),
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
                "max_hlo_output_mib": max(
                    (float(row["hlo_output_mib"] or 0.0) for row in rows), default=0.0
                ),
                "max_hlo_input_mib": max(
                    (float(row["hlo_input_mib"] or 0.0) for row in rows), default=0.0
                ),
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


def _pcr_stage_hlo_rows(
    rows: Sequence[dict[str, Any]], *, variant: str
) -> tuple[dict[str, Any], ...]:
    selected = [
        row
        for row in rows
        if row.get("stage") == "block_solve"
        and row.get("variant") == variant
        and "loop_select_subtract" in str(row.get("fusion_name", ""))
    ]
    return tuple(sorted(selected, key=lambda row: _float(row.get("line")) or 0.0))


def _hlo_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "hlo_fusion_name": row.get("fusion_name", ""),
        "hlo_output_arrays": _int_or_empty(row.get("output_arrays")),
        "hlo_output_mib": _mib_or_empty(row.get("output_bytes_estimate")) if row else "",
        "hlo_input_mib": _mib_or_empty(row.get("computation_input_bytes_estimate")) if row else "",
        "hlo_io_mib": _mib_or_empty(row.get("computation_io_bytes_estimate")) if row else "",
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
    variant = metadata.get("variant") or (rows[0].get("variant") if rows else "unknown")
    spec = VARIANT_SPECS.get(str(variant), {})
    lines = [
        "# PCR/SoA Stage State Audit",
        "",
        "Benchmark-only structural audit of the batch-native double-cable PCR/SoA solver.",
        "",
        "## Context",
        "",
        f"- Variant: `{variant}`",
        f"- Variant shape: {spec.get('description', 'unknown')}",
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
            "- `state arrays` is the algorithmic PCR/SoA live state for this variant; "
            "HLO output counts are compiler artifacts for the attached optimized HLO.",
            "- The explicit neighbor-read count is estimated from the Python stage "
            "body before XLA common-subexpression or fusion optimization. HLO rows, "
            "when attached, are compiler outputs and their mapping to PCR strides is "
            "approximate.",
            "- Use this report to choose benchmark-only state/staging variants; do not "
            "promote solver policy from this structural estimate alone.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _variant_spec(variant: str) -> dict[str, Any]:
    try:
        return VARIANT_SPECS[variant]
    except KeyError as exc:
        choices = ", ".join(sorted(VARIANT_SPECS))
        raise ValueError(f"unknown variant {variant!r}; choices are: {choices}") from exc


def _dtype_nbytes(dtype: str) -> int:
    if dtype == "float64":
        return 8
    return 4


def _bytes_to_mib(value: float | int) -> float:
    return float(value) / (1024.0 * 1024.0)


def _mib_or_empty(value: Any) -> float | str:
    parsed = _float(value)
    return "" if parsed is None else _bytes_to_mib(parsed)


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
