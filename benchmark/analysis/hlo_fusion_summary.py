from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


KNOWN_STAGES = (
    "membrane_conductance_terms_gated_only",
    "membrane_conductance_terms_mask_mix",
    "membrane_conductance_terms",
    "membrane_gate_update_gated_only",
    "membrane_gate_update",
    "extracellular_rhs_drive",
    "system_assembly",
    "one_step_proxy",
    "observer_write",
    "block_solve",
)

ARRAY_RE = re.compile(
    r"\b(?P<dtype>pred|bf16|f16|f32|f64|s8|s16|s32|s64|u8|u16|u32|u64)"
    r"\[(?P<dims>[^\]]*)\](?:\{(?P<layout>[^}]*)\})?"
)
FUSION_INSTRUCTION_RE = re.compile(
    r"^\s*(?:ROOT\s+)?(?P<name>%[\w.\-]+)\s*=\s*"
    r"(?P<output>.+?)\s+fusion\((?P<operands>.*)\),\s*"
    r"kind=(?P<kind>[^,]+),\s*calls=(?P<calls>%[\w.\-]+)"
)
COMPUTATION_START_RE = re.compile(
    r"^\s*(?P<name>%?[\w.\-]+)\s*\(.*\)\s*->\s*(?P<output>.+)\{\s*$"
)

FUSION_FIELDS = (
    "file",
    "stage",
    "variant",
    "module",
    "fusion_name",
    "calls",
    "fusion_kind",
    "line",
    "operand_count",
    "output_arrays",
    "output_bytes_estimate",
    "output_layouts",
    "computation_lines",
    "computation_input_arrays",
    "computation_output_arrays",
    "count_add",
    "count_bitcast",
    "count_broadcast",
    "count_compare",
    "count_concatenate",
    "count_copy",
    "count_divide",
    "count_gather",
    "count_multiply",
    "count_negate",
    "count_reduce",
    "count_select",
    "count_slice",
    "count_subtract",
    "count_tuple",
)

LAYOUT_FIELDS = (
    "file",
    "stage",
    "variant",
    "module",
    "role",
    "dtype",
    "rank",
    "dims",
    "layout",
    "count",
    "bytes_per_array",
)


@dataclass(frozen=True)
class HloFusionAnalysis:
    fusion_rows: tuple[dict[str, Any], ...]
    layout_rows: tuple[dict[str, Any], ...]
    module_rows: tuple[dict[str, Any], ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize XLA optimized HLO fusions, shape layouts, and rough "
            "memory pressure from AxonScope lowering artifacts."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="HLO files or directories to scan.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for hlo_fusion_summary.csv and hlo_fusion_report.md.",
    )
    args = parser.parse_args(argv)

    files = _collect_hlo_files(args.inputs)
    if not files:
        parser.error("no *.optimized_hlo.txt files found.")

    analysis = analyze_hlo_files(files)
    write_hlo_fusion_artifacts(args.output, analysis=analysis)
    print(f"wrote: {args.output / 'hlo_fusion_summary.csv'}")
    print(f"wrote: {args.output / 'hlo_fusion_report.md'}")
    return 0


def analyze_hlo_files(files: Sequence[Path]) -> HloFusionAnalysis:
    fusion_rows: list[dict[str, Any]] = []
    layout_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        stage, variant = _stage_variant_from_path(file)
        module = _module_name(text)
        computations = _fused_computations(text)
        fusion_rows.extend(
            _fusion_rows_for_file(
                text,
                file=file,
                stage=stage,
                variant=variant,
                module=module,
                computations=computations,
            )
        )
        layout_rows.extend(
            _layout_rows_for_file(
                text,
                file=file,
                stage=stage,
                variant=variant,
                module=module,
            )
        )
        module_rows.append(
            {
                "file": str(file),
                "stage": stage,
                "variant": variant,
                "module": module,
                "lines": len(text.splitlines()),
                "bytes": len(text.encode("utf-8")),
                "fusion_count": sum(1 for line in text.splitlines() if " fusion(" in line),
                "entry_computation_layout": _entry_computation_layout(text),
            }
        )
    return HloFusionAnalysis(
        fusion_rows=tuple(fusion_rows),
        layout_rows=tuple(layout_rows),
        module_rows=tuple(module_rows),
    )


def write_hlo_fusion_artifacts(
    output_dir: Path,
    *,
    files: Sequence[Path] | None = None,
    analysis: HloFusionAnalysis | None = None,
    metadata: dict[str, Any] | None = None,
) -> HloFusionAnalysis:
    if analysis is None:
        if files is None:
            files = _collect_hlo_files((output_dir,))
        analysis = analyze_hlo_files(tuple(files))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "hlo_fusion_summary.csv", FUSION_FIELDS, analysis.fusion_rows)
    _write_csv(output_dir / "hlo_layout_summary.csv", LAYOUT_FIELDS, analysis.layout_rows)
    _write_json(
        output_dir / "hlo_fusion_metrics.json",
        {
            "metadata": metadata or {},
            "modules": analysis.module_rows,
            "fusion_count": len(analysis.fusion_rows),
            "layout_count": len(analysis.layout_rows),
        },
    )
    _write_report(output_dir / "hlo_fusion_report.md", analysis, metadata=metadata)
    return analysis


def _collect_hlo_files(inputs: Iterable[Path]) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.optimized_hlo.txt")))
            files.extend(sorted(path.rglob("*.compiled.optimized_hlo.txt")))
        elif path.is_file():
            files.append(path)
    deduped = sorted({file.resolve(): file for file in files}.values())
    return tuple(deduped)


def _fusion_rows_for_file(
    text: str,
    *,
    file: Path,
    stage: str,
    variant: str,
    module: str,
    computations: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = FUSION_INSTRUCTION_RE.match(line)
        if not match:
            continue
        output_text = match.group("output")
        calls = match.group("calls")
        computation = computations.get(calls, "")
        computation_shapes = _array_shapes(computation)
        output_shapes = _array_shapes(output_text)
        counts = _op_counts(computation)
        rows.append(
            {
                "file": str(file),
                "stage": stage,
                "variant": variant,
                "module": module,
                "fusion_name": match.group("name"),
                "calls": calls,
                "fusion_kind": match.group("kind"),
                "line": line_number,
                "operand_count": len(re.findall(r"%[\w.\-]+", match.group("operands"))),
                "output_arrays": len(output_shapes),
                "output_bytes_estimate": sum(_shape_nbytes(shape) for shape in output_shapes),
                "output_layouts": ";".join(sorted({shape.get("layout") or "" for shape in output_shapes})),
                "computation_lines": len(computation.splitlines()),
                "computation_input_arrays": _computation_input_array_count(computation),
                "computation_output_arrays": _computation_output_array_count(computation),
                **counts,
            }
        )
    return rows


def _layout_rows_for_file(
    text: str,
    *,
    file: Path,
    stage: str,
    variant: str,
    module: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, role_text in (("entry", _entry_computation_layout(text)), ("all", text)):
        counts: Counter[tuple[str, str, str, str]] = Counter()
        bytes_by_key: dict[tuple[str, str, str, str], int] = {}
        for shape in _array_shapes(role_text):
            dims = ",".join(str(dim) for dim in shape["dims"])
            layout = shape.get("layout") or ""
            key = (shape["dtype"], str(len(shape["dims"])), dims, layout)
            counts[key] += 1
            bytes_by_key[key] = _shape_nbytes(shape)
        for (dtype, rank, dims, layout), count in sorted(counts.items()):
            rows.append(
                {
                    "file": str(file),
                    "stage": stage,
                    "variant": variant,
                    "module": module,
                    "role": role,
                    "dtype": dtype,
                    "rank": rank,
                    "dims": dims,
                    "layout": layout,
                    "count": count,
                    "bytes_per_array": bytes_by_key[(dtype, rank, dims, layout)],
                }
            )
    return rows


def _fused_computations(text: str) -> dict[str, str]:
    lines = text.splitlines()
    blocks: dict[str, str] = {}
    index = 0
    while index < len(lines):
        start = COMPUTATION_START_RE.match(lines[index])
        if not start:
            index += 1
            continue
        name = start.group("name")
        if not name.startswith("%"):
            name = f"%{name}"
        depth = lines[index].count("{") - lines[index].count("}")
        end = index
        while depth > 0 and end + 1 < len(lines):
            end += 1
            depth += lines[end].count("{") - lines[end].count("}")
        blocks[name] = "\n".join(lines[index : end + 1])
        index = end + 1
    return blocks


def _op_counts(text: str) -> dict[str, int]:
    return {
        "count_add": len(re.findall(r"\badd\(", text)),
        "count_bitcast": len(re.findall(r"\bbitcast\(", text)),
        "count_broadcast": len(re.findall(r"\bbroadcast\(", text)),
        "count_compare": len(re.findall(r"\bcompare\(", text)),
        "count_concatenate": len(re.findall(r"\bconcatenate\(", text)),
        "count_copy": len(re.findall(r"\bcopy\(", text)),
        "count_divide": len(re.findall(r"\bdivide\(", text)),
        "count_gather": len(re.findall(r"\bgather\(", text)),
        "count_multiply": len(re.findall(r"\bmultiply\(", text)),
        "count_negate": len(re.findall(r"\bnegate\(", text)),
        "count_reduce": len(re.findall(r"\breduce\(", text)),
        "count_select": len(re.findall(r"\bselect\(", text)),
        "count_slice": len(re.findall(r"\bslice\(", text)),
        "count_subtract": len(re.findall(r"\bsubtract\(", text)),
        "count_tuple": len(re.findall(r"\btuple\(", text)),
    }


def _array_shapes(text: str) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    for match in ARRAY_RE.finditer(text):
        dims = tuple(
            int(item)
            for item in match.group("dims").split(",")
            if item.strip().isdigit()
        )
        shapes.append(
            {
                "dtype": match.group("dtype"),
                "dims": dims,
                "layout": match.group("layout"),
            }
        )
    return shapes


def _shape_nbytes(shape: dict[str, Any]) -> int:
    n = 1
    for dim in shape["dims"]:
        n *= int(dim)
    return n * _dtype_nbytes(shape["dtype"])


def _dtype_nbytes(dtype: str) -> int:
    if dtype in {"f64", "s64", "u64"}:
        return 8
    if dtype in {"f32", "s32", "u32"}:
        return 4
    if dtype in {"f16", "bf16", "s16", "u16"}:
        return 2
    return 1


def _computation_input_array_count(text: str) -> int:
    first_line = text.splitlines()[0] if text else ""
    before_arrow = first_line.split("->", 1)[0]
    return len(_array_shapes(before_arrow))


def _computation_output_array_count(text: str) -> int:
    first_line = text.splitlines()[0] if text else ""
    after_arrow = first_line.split("->", 1)[1] if "->" in first_line else ""
    return len(_array_shapes(after_arrow))


def _module_name(text: str) -> str:
    first = text.splitlines()[0] if text else ""
    match = re.match(r"HloModule\s+([^,\s]+)", first)
    return match.group(1) if match else ""


def _entry_computation_layout(text: str) -> str:
    first = text.splitlines()[0] if text else ""
    match = re.search(r"entry_computation_layout=(.*?),\s+allow_spmd", first)
    if match:
        return match.group(1)
    match = re.search(r"entry_computation_layout=(.*)$", first)
    return match.group(1) if match else ""


def _stage_variant_from_path(path: Path) -> tuple[str, str]:
    name = path.name
    for suffix in (
        ".compiled.optimized_hlo.txt",
        ".optimized_hlo.txt",
        ".lowered.hlo.txt",
        ".hlo.txt",
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for stage in KNOWN_STAGES:
        prefix = f"{stage}_"
        if name.startswith(prefix):
            return stage, name[len(prefix) :]
    return "", name


def _write_report(
    path: Path,
    analysis: HloFusionAnalysis,
    *,
    metadata: dict[str, Any] | None,
) -> None:
    lines = [
        "# HLO Fusion Summary",
        "",
        "Benchmark-only summary of optimized HLO fusion bodies and array layouts.",
        "",
        "## Modules",
        "",
        "| stage | variant | module | lines | fusions |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in analysis.module_rows:
        lines.append(
            "| {stage} | {variant} | {module} | {lines} | {fusion_count} |".format(**row)
        )
    lines.extend(["", "## Largest Fusion Outputs", ""])
    lines.extend(_fusion_table(sorted(analysis.fusion_rows, key=_fusion_sort_key, reverse=True)[:12]))
    lines.extend(["", "## Top Shape Layouts", ""])
    lines.extend(_layout_table(_top_layout_rows(analysis.layout_rows)))
    if metadata:
        context = metadata.get("context") or {}
        lines.extend(
            [
                "",
                "## Context",
                "",
                f"- Platform: `{context.get('platform', metadata.get('platform'))}`",
                f"- Device: `{context.get('device', metadata.get('device'))}`",
                f"- Precision: `{context.get('precision')}`",
                f"- Target Nx: `{context.get('target_nx')}`",
                f"- Actual Nx: `{context.get('actual_nx')}`",
                f"- Naxons: `{context.get('n_axons')}`",
                f"- Diameters: `{context.get('diameters')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fusion_table(rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "| stage | variant | fusion | kind | operands | outputs | output MiB | lines | gather | select | copy |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {variant} | {fusion_name} | {fusion_kind} | {operand_count} | "
            "{output_arrays} | {output_mib:.2f} | {computation_lines} | {count_gather} | "
            "{count_select} | {count_copy} |".format(
                **row,
                output_mib=float(row["output_bytes_estimate"]) / (1024.0 * 1024.0),
            )
        )
    return lines


def _layout_table(rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "| stage | variant | role | dtype | dims | layout | count | bytes/array |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {variant} | {role} | {dtype} | {dims} | {layout} | "
            "{count} | {bytes_per_array} |".format(**row)
        )
    return lines


def _fusion_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["output_bytes_estimate"]),
        int(row["operand_count"]),
        int(row["computation_lines"]),
    )


def _top_layout_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    all_rows = [row for row in rows if row["role"] == "all"]
    return sorted(
        all_rows,
        key=lambda row: (int(row["count"]), int(row["bytes_per_array"])),
        reverse=True,
    )[:12]


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
