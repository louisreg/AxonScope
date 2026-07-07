from __future__ import annotations

import argparse
import csv
import json
import os
import platform as host_platform
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.analysis.double_cable_real_stage_profile import (  # noqa: E402
    StageCase,
    _prepare_real_stage_inputs,
    _select_device,
)
from benchmark.analysis.hlo_fusion_summary import write_hlo_fusion_artifacts  # noqa: E402
from benchmark.workloads.curve_options import PRESETS  # noqa: E402


SUMMARY_BASE_FIELDS = (
    "stage",
    "variant",
    "ir_kind",
    "dialect",
    "platform",
    "device",
    "precision",
    "target_nx",
    "actual_nx",
    "n_axons",
    "kernel_group_size",
    "diameters",
    "shared_coefficients",
    "bytes",
    "lines",
    "file",
)

OP_PATTERNS = {
    "add": (r"\bstablehlo\.add\b", r"\badd\("),
    "broadcast": (r"\bstablehlo\.broadcast_in_dim\b", r"\bbroadcast\("),
    "call": (r"\bcall\b", r"\bcall\("),
    "compare": (r"\bstablehlo\.compare\b", r"\bcompare\("),
    "concatenate": (r"\bstablehlo\.concatenate\b", r"\bconcatenate\("),
    "divide": (r"\bstablehlo\.divide\b", r"\bdivide\("),
    "dynamic_slice": (r"\bstablehlo\.dynamic_slice\b", r"\bdynamic-slice\("),
    "fusion": (r"\bfusion\b", r"\bfusion\("),
    "gather": (r"\bstablehlo\.gather\b", r"\bgather\("),
    "maximum": (r"\bstablehlo\.maximum\b", r"\bmaximum\("),
    "minimum": (r"\bstablehlo\.minimum\b", r"\bminimum\("),
    "multiply": (r"\bstablehlo\.multiply\b", r"\bmultiply\("),
    "reduce": (r"\bstablehlo\.reduce\b", r"\breduce\("),
    "reshape": (r"\bstablehlo\.reshape\b", r"\breshape\("),
    "select": (r"\bstablehlo\.select\b", r"\bselect\("),
    "slice": (r"\bstablehlo\.slice\b", r"\bslice\("),
    "subtract": (r"\bstablehlo\.subtract\b", r"\bsubtract\("),
    "transpose": (r"\bstablehlo\.transpose\b", r"\btranspose\("),
    "while": (r"\bstablehlo\.while\b", r"\bwhile\("),
}

SUMMARY_FIELDS = SUMMARY_BASE_FIELDS + tuple(f"count_{name}" for name in OP_PATTERNS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Lower real AxonScope double-cable PCR/SoA solver variants to "
            "StableHLO/HLO and summarize codegen-relevant patterns."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/p11b_double_cable_solver_lowering_audit"),
    )
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="quick")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--n-axons", type=int)
    parser.add_argument("--tsim", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--precision", choices=("fp32", "fp64"))
    parser.add_argument(
        "--diameters",
        choices=("same_diameter", "different_diameters"),
        default="different_diameters",
    )
    parser.add_argument(
        "--recording",
        choices=("observer_only", "full_vm", "probe_vm"),
        default="observer_only",
    )
    parser.add_argument("--amplitude-uA", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-chunk-steps", type=int, default=50)
    parser.add_argument(
        "--double-cable-block-solver",
        choices=("auto", "thomas", "pcr", "pcr_soa", "pcr_adaptive"),
        default="pcr_soa",
        help="Solver used by the one-step proxy. Defaults to pcr_soa for audit work.",
    )
    parser.add_argument(
        "--variants",
        default="pcr_soa_vmap,pcr_soa_batched",
        help="Comma-separated block-solve variants to lower.",
    )
    parser.add_argument(
        "--dialects",
        default="stablehlo,hlo",
        help="Comma-separated lowered compiler_ir dialects to write.",
    )
    parser.add_argument("--include-one-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-membrane-stages",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also lower generated membrane gate/conductance stages.",
    )
    parser.add_argument(
        "--include-system-stages",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also lower extracellular drive, system assembly, and observer write stages.",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-ir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--analyze-hlo-fusions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Summarize compiled optimized-HLO fusion bodies and shape layouts.",
    )
    args = parser.parse_args(argv)

    if args.nx is not None and args.nx < 3:
        parser.error("--nx must be >= 3.")
    if args.n_axons is not None and args.n_axons < 1:
        parser.error("--n-axons must be >= 1.")
    if args.time_chunk_steps < 1:
        parser.error("--time-chunk-steps must be >= 1.")

    variants = _split_csv(args.variants)
    dialects = _split_csv(args.dialects)
    if not variants:
        parser.error("--variants must select at least one variant.")
    if not dialects:
        parser.error("--dialects must select at least one dialect.")
    unknown = set(variants) - {
        "active_auto",
        "thomas_vmap",
        "pcr_matrix_vmap",
        "pcr_soa_vmap",
        "pcr_soa_batched",
        "pcr_soa_symmetric_batched",
        "pcr_soa_nomask_batched",
        "pcr_soa_shift_batched",
        "pcr_soa_transposed_batched",
        "pcr_soa_padded_batched",
        "pcr_soa_hybrid_batched",
    }
    if unknown:
        parser.error(f"unsupported --variants: {sorted(unknown)}")

    if args.precision == "fp64":
        jax.config.update("jax_enable_x64", True)
    device = _select_device(args.platform)
    args.output.mkdir(parents=True, exist_ok=True)

    with jax.default_device(device):
        inputs = _prepare_inputs(args, variants=variants)

    cases = _selected_cases(
        inputs.stage_cases,
        variants=variants,
        include_one_step=args.include_one_step,
        include_membrane_stages=args.include_membrane_stages,
        include_system_stages=args.include_system_stages,
    )
    metadata = _metadata(args=args, device=device, inputs=inputs, variants=variants, dialects=dialects)
    _write_json(args.output / "metadata.json", metadata)

    rows: list[dict[str, Any]] = []
    lowered_metrics: dict[str, Any] = {}
    with jax.default_device(device):
        for case in cases:
            case_rows, case_metrics = _lower_case(
                case,
                args=args,
                dialects=dialects,
                context=metadata["context"],
            )
            rows.extend(case_rows)
            lowered_metrics[_case_key(case)] = case_metrics

    _write_csv(args.output / "lowering_summary.csv", SUMMARY_FIELDS, rows)
    _write_json(args.output / "lowering_metrics.json", lowered_metrics)
    if args.analyze_hlo_fusions:
        _write_hlo_fusion_summary(args.output, rows, metadata)
    _write_report(args.output / "lowering_report.md", rows, metadata)
    print(f"wrote: {args.output / 'lowering_summary.csv'}")
    print(f"wrote: {args.output / 'lowering_report.md'}")
    return 0


def _prepare_inputs(args: argparse.Namespace, *, variants: Sequence[str]) -> Any:
    profile_args = argparse.Namespace(
        output=args.output,
        platform=args.platform,
        preset=args.preset,
        nx=args.nx,
        n_axons=args.n_axons,
        tsim=args.tsim,
        dt=args.dt,
        precision=args.precision,
        diameters=args.diameters,
        recording=args.recording,
        amplitude_uA=args.amplitude_uA,
        seed=args.seed,
        time_chunk_steps=args.time_chunk_steps,
        double_cable_block_solver=args.double_cable_block_solver,
        solver=list(variants),
        repeats=1,
        warmups=0,
        no_plots=True,
    )
    return _prepare_real_stage_inputs(profile_args, device=_select_device(args.platform))


def _selected_cases(
    cases: Sequence[StageCase],
    *,
    variants: Sequence[str],
    include_one_step: bool,
    include_membrane_stages: bool,
    include_system_stages: bool,
) -> tuple[StageCase, ...]:
    wanted = set(variants)
    out: list[StageCase] = [
        case for case in cases if case.stage == "block_solve" and case.variant in wanted
    ]
    if include_one_step:
        out.extend(
            case
            for case in cases
            if case.stage in {"one_step_proxy", "one_step_without_solve"}
        )
    if include_membrane_stages:
        out.extend(
            case
            for case in cases
            if case.stage.startswith("membrane_")
        )
    if include_system_stages:
        out.extend(
            case
            for case in cases
            if case.stage in {"extracellular_rhs_drive", "system_assembly", "observer_write"}
        )
    if not out:
        raise RuntimeError("No solver lowering cases were selected.")
    return tuple(out)


def _lower_case(
    case: StageCase,
    *,
    args: argparse.Namespace,
    dialects: Sequence[str],
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lowered = case.fn.lower(*case.args)
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "stage": case.stage,
        "variant": case.variant,
        "arguments": _argument_summaries(case.args),
    }
    for dialect in dialects:
        text, error = _compiler_ir_text(lowered, dialect=dialect)
        row, file_path = _record_ir_text(
            text,
            error=error,
            case=case,
            args=args,
            context=context,
            ir_kind="lowered",
            dialect=dialect,
        )
        rows.append(row)
        metrics[f"lowered_{dialect}"] = {
            "file": str(file_path) if file_path else None,
            "error": error,
            "metrics": _text_metrics(text, dialect=dialect),
        }
    if args.compile:
        compiled, error = _compile_lowered(lowered)
        if compiled is None:
            row, _ = _record_ir_text(
                "",
                error=error,
                case=case,
                args=args,
                context=context,
                ir_kind="compiled",
                dialect="optimized_hlo",
            )
            rows.append(row)
            metrics["compiled"] = {"error": error}
        else:
            text = _compiled_text(compiled)
            row, file_path = _record_ir_text(
                text,
                error=None,
                case=case,
                args=args,
                context=context,
                ir_kind="compiled",
                dialect="optimized_hlo",
            )
            rows.append(row)
            metrics["compiled"] = {
                "file": str(file_path) if file_path else None,
                "cost_analysis": _safe_cost_analysis(compiled),
                "metrics": _text_metrics(text, dialect="optimized_hlo"),
            }
    return rows, metrics


def _record_ir_text(
    text: str,
    *,
    error: str | None,
    case: StageCase,
    args: argparse.Namespace,
    context: dict[str, Any],
    ir_kind: str,
    dialect: str,
) -> tuple[dict[str, Any], Path | None]:
    file_path: Path | None = None
    if args.write_ir and text:
        suffix = {
            "stablehlo": "stablehlo.mlir",
            "hlo": "hlo.txt",
            "optimized_hlo": "optimized_hlo.txt",
        }.get(dialect, f"{dialect}.txt")
        file_path = args.output / "ir" / f"{case.stage}_{case.variant}.{ir_kind}.{suffix}"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding="utf-8")
    row = {
        **context,
        "stage": case.stage,
        "variant": case.variant,
        "ir_kind": ir_kind,
        "dialect": dialect,
        "bytes": len(text.encode("utf-8")),
        "lines": len(text.splitlines()),
        "file": str(file_path) if file_path else "",
    }
    row.update(_pattern_counts(text, dialect=dialect))
    if error:
        row["file"] = f"ERROR: {error}"
    return row, file_path


def _compiler_ir_text(lowered: Any, *, dialect: str) -> tuple[str, str | None]:
    try:
        ir = lowered.compiler_ir(dialect=dialect)
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
    return _ir_to_text(ir), None


def _ir_to_text(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return "\n\n".join(_ir_to_text(item) for item in value)
    as_hlo_text = getattr(value, "as_hlo_text", None)
    if callable(as_hlo_text):
        return str(as_hlo_text())
    return str(value)


def _compile_lowered(lowered: Any) -> tuple[Any | None, str | None]:
    try:
        return lowered.compile(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _compiled_text(compiled: Any) -> str:
    as_text = getattr(compiled, "as_text", None)
    if callable(as_text):
        return str(as_text())
    runtime_executable = getattr(compiled, "runtime_executable", None)
    if callable(runtime_executable):
        executable = runtime_executable()
        hlo_modules = getattr(executable, "hlo_modules", None)
        if callable(hlo_modules):
            return "\n\n".join(str(module.to_string()) for module in hlo_modules())
    return str(compiled)


def _safe_cost_analysis(compiled: Any) -> Any:
    cost_analysis = getattr(compiled, "cost_analysis", None)
    if not callable(cost_analysis):
        return None
    try:
        return _jsonable(cost_analysis())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _pattern_counts(text: str, *, dialect: str) -> dict[str, int]:
    index = 0 if dialect == "stablehlo" else 1
    return {
        f"count_{name}": len(re.findall(patterns[index], text))
        for name, patterns in OP_PATTERNS.items()
    }


def _text_metrics(text: str, *, dialect: str) -> dict[str, Any]:
    return {
        "bytes": len(text.encode("utf-8")),
        "lines": len(text.splitlines()),
        "counts": _pattern_counts(text, dialect=dialect),
    }


def _argument_summaries(values: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        nbytes = getattr(value, "nbytes", None)
        out.append(
            {
                "index": index,
                "type": type(value).__name__,
                "shape": tuple(int(dim) for dim in shape) if shape is not None else None,
                "dtype": str(dtype) if dtype is not None else None,
                "nbytes": int(nbytes) if nbytes is not None else None,
            }
        )
    return out


def _metadata(
    *,
    args: argparse.Namespace,
    device: Any,
    inputs: Any,
    variants: Sequence[str],
    dialects: Sequence[str],
) -> dict[str, Any]:
    group = inputs.group_metadata
    return {
        "script": "benchmark/analysis/double_cable_solver_lowering_audit.py",
        "purpose": "Lower real double-cable PCR/SoA solver variants and compare IR patterns.",
        "platform": args.platform,
        "device": str(device),
        "jax_version": jax.__version__,
        "python": host_platform.python_version(),
        "host": {
            "system": host_platform.system(),
            "release": host_platform.release(),
            "machine": host_platform.machine(),
            "processor": host_platform.processor(),
        },
        "git": _git_metadata(),
        "options": {
            "preset": args.preset,
            "target_nx": args.nx,
            "n_axons": args.n_axons,
            "diameters": args.diameters,
            "recording": args.recording,
            "double_cable_block_solver": args.double_cable_block_solver,
            "variants": tuple(variants),
            "dialects": tuple(dialects),
            "compile": bool(args.compile),
            "include_one_step": bool(args.include_one_step),
            "include_membrane_stages": bool(args.include_membrane_stages),
            "include_system_stages": bool(args.include_system_stages),
            "analyze_hlo_fusions": bool(args.analyze_hlo_fusions),
        },
        "group": group,
        "context": {
            "platform": args.platform,
            "device": str(device),
            "precision": args.precision or PRESETS[args.preset].precision,
            "target_nx": group["target_nx"],
            "actual_nx": group["actual_nx"],
            "n_axons": group["n_axons"],
            "kernel_group_size": group["kernel_group_size"],
            "diameters": group["diameters"],
            "shared_coefficients": group["shared_coefficients"],
        },
        "limitations": [
            "This is a benchmark-only lowering audit; it does not add runtime routes.",
            "StableHLO is useful for structural comparison; optimized HLO is platform-specific.",
            "Pattern counts are triage signals, not performance claims by themselves.",
        ],
    }


def _write_hlo_fusion_summary(
    output: Path,
    rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    hlo_files = [
        Path(str(row["file"]))
        for row in rows
        if row["ir_kind"] == "compiled"
        and row["dialect"] == "optimized_hlo"
        and row.get("file")
        and not str(row["file"]).startswith("ERROR:")
    ]
    existing = tuple(path for path in hlo_files if path.exists())
    if existing:
        write_hlo_fusion_artifacts(output, files=existing, metadata=metadata)


def _write_report(path: Path, rows: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> None:
    compiled_rows = [
        row for row in rows if row["ir_kind"] == "compiled" and row["dialect"] == "optimized_hlo"
    ]
    lowered_rows = [row for row in rows if row["ir_kind"] == "lowered"]
    lines = [
        "# Double-Cable Solver Lowering Audit",
        "",
        "Benchmark-only lowering/codegen cartography for real AxonScope double-cable stages.",
        "",
        "## Context",
        "",
        f"- Platform: `{metadata['platform']}`",
        f"- Device: `{metadata['device']}`",
        f"- JAX: `{metadata['jax_version']}`",
        f"- Git commit: `{metadata['git'].get('commit')}`",
        f"- Git dirty: `{metadata['git'].get('dirty')}`",
        f"- Target Nx: `{metadata['group']['target_nx']}`",
        f"- Actual kernel Nx: `{metadata['group']['actual_nx']}`",
        f"- Kernel group size: `{metadata['group']['kernel_group_size']}`",
        f"- Diameters: `{metadata['group']['diameters']}`",
        f"- Shared coefficients: `{metadata['group']['shared_coefficients']}`",
        f"- Active solver: `{metadata['group']['active_solver']}`",
        "",
    ]
    if compiled_rows:
        lines.extend(_table_section("Compiled Optimized HLO Counts", compiled_rows))
    if lowered_rows:
        lines.extend(_table_section("Lowered IR Counts", lowered_rows))
    lines.extend(
        [
            "## Notes",
            "",
            "- `pcr_soa_vmap` lowers a one-fiber PCR solve under an outer vmap.",
            "- `pcr_soa_batched` lowers the batch-native `[B, Nx]` PCR/SoA path used by the active GPU runtime.",
            "- `one_step_proxy` lowers gate update, conductance terms, assembly, and active block solve together.",
            "- Optional membrane/system stages can be included with `--include-membrane-stages` and `--include-system-stages`.",
            "- `hlo_fusion_summary.csv` and `hlo_layout_summary.csv` summarize optimized-HLO fusion bodies when compiled HLO is available.",
            "- Use the emitted IR files for detailed inspection before adding solver routes.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table_section(title: str, rows: Sequence[dict[str, Any]]) -> list[str]:
    ordered = sorted(rows, key=lambda row: (row["stage"], row["variant"], row["dialect"]))
    lines = [
        f"## {title}",
        "",
        "| stage | variant | dialect | lines | gather | broadcast | select | transpose | fusion |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            "| {stage} | {variant} | {dialect} | {lines} | {count_gather} | "
            "{count_broadcast} | {count_select} | {count_transpose} | {count_fusion} |".format(
                **row
            )
        )
    lines.append("")
    return lines


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def _git_metadata() -> dict[str, Any]:
    def run_git(*cmd: str) -> str | None:
        try:
            result = subprocess.run(
                ("git", *cmd),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status,
    }


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _case_key(case: StageCase) -> str:
    return f"{case.stage}:{case.variant}"


if __name__ == "__main__":
    raise SystemExit(main())
