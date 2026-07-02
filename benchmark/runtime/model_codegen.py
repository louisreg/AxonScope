from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

import axonscope as axs
from axonscope.membranes.model import MembraneModel
from axonscope.membranes.types import (
    ConductanceDensity,
    CurrentDensity,
    Gate,
    Rate,
    ResistanceArea,
    Voltage,
)


@dataclass(frozen=True)
class ModelCase:
    """One membrane model benchmark case."""

    name: str
    factory: Callable[[], axs.membranes.Model]
    group: str


@dataclass(frozen=True)
class CodegenRow:
    """One measured codegen/cache row."""

    model: str
    group: str
    phase: str
    repeat: int
    seconds: float
    cache_status: str
    cache_reason: str
    cache_key: str
    source_hash: str
    source_path: str
    generated_bytes: int
    generated_files: tuple[str, ...]


class BenchmarkLeak(axs.membranes.Model):
    """Small custom membrane model used by the codegen benchmark."""

    model_kind = "benchmark_leak"

    Rm: ResistanceArea = 10_000.0 * axs.ohm_cm2
    EL: Voltage = -70.0 * axs.mV

    @axs.membranes.currents
    def currents(self, Vm: Voltage):
        drive: Voltage = Vm - self.EL
        g_l: ConductanceDensity = 1.0 / self.Rm
        I_l: CurrentDensity = g_l * drive
        return I_l, g_l, drive


class BenchmarkSodiumLeak(axs.membranes.Model):
    """Small custom gated model used by the codegen benchmark."""

    model_kind = "benchmark_sodium_leak"

    gna: ConductanceDensity = 20.0 * axs.mS_per_cm2
    gl: ConductanceDensity = 0.1 * axs.mS_per_cm2
    ena: Voltage = 45.0 * axs.mV
    el: Voltage = -70.0 * axs.mV

    @axs.membranes.rates
    def rates(self, Vm: Voltage):
        alpha_m: Rate = 0.1 / (axs.ms * axs.mV) * (Vm + 35.0 * axs.mV)
        beta_m: Rate = 4.0 / axs.ms
        self.keep(alpha_m, beta_m)

    @axs.membranes.currents(outputs=("I_na", "I_l"), observables=("g_na", "g_l"))
    def currents(self, Vm: Voltage, m: Gate):
        g_na: ConductanceDensity = self.gna * m
        g_l: ConductanceDensity = self.gl
        I_na: CurrentDensity = g_na * (Vm - self.ena)
        I_l: CurrentDensity = g_l * (Vm - self.el)
        return I_na, I_l, g_na, g_l


BUILTIN_CASES: tuple[ModelCase, ...] = (
    ModelCase("passive", axs.membranes.Passive, "builtin"),
    ModelCase("hodgkin_huxley", axs.membranes.HodgkinHuxley, "builtin"),
    ModelCase("rattay_aberham", axs.membranes.RattayAberham, "builtin"),
    ModelCase("sundt", axs.membranes.Sundt, "builtin"),
    ModelCase("axnode", axs.membranes.AxNode, "builtin"),
    ModelCase("tigerholm", axs.membranes.Tigerholm, "builtin"),
    ModelCase("schild94", axs.membranes.Schild94, "builtin"),
    ModelCase("schild97", axs.membranes.Schild97, "builtin"),
)

CUSTOM_CASES: tuple[ModelCase, ...] = (
    ModelCase("benchmark_leak", BenchmarkLeak, "custom"),
    ModelCase("benchmark_sodium_leak", BenchmarkSodiumLeak, "custom"),
)

MODEL_CASES: dict[str, ModelCase] = {
    case.name: case for case in (*BUILTIN_CASES, *CUSTOM_CASES)
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark class-based membrane source/codegen cache behavior."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["builtins"],
        choices=("all", "builtins", "custom", *MODEL_CASES),
        help="Model cases to benchmark.",
    )
    parser.add_argument(
        "--warm-repeats",
        type=int,
        default=3,
        help="Warm cache-hit inspection repetitions per model.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional generated-code cache root. Defaults under --out-dir.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark/results/runtime"),
        help="Directory for JSON and CSV benchmark outputs.",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix.")
    parser.add_argument("--list", action="store_true", help="List model cases and exit.")
    args = parser.parse_args(argv)

    if args.list:
        print_cases()
        return

    if args.warm_repeats < 0:
        raise ValueError("--warm-repeats must be >= 0.")

    selected = select_cases(args.models)
    prefix = args.prefix or datetime.now().strftime("model_codegen_%Y%m%d_%H%M%S")
    cache_root = (args.cache_root or args.out_dir / f"{prefix}_codegen_cache").resolve()
    rows = run_codegen_benchmark(
        selected,
        cache_root=cache_root,
        warm_repeats=int(args.warm_repeats),
    )
    json_path, csv_path = write_rows(
        rows,
        out_dir=args.out_dir,
        prefix=prefix,
        metadata=run_metadata(selected, cache_root=cache_root, warm_repeats=args.warm_repeats),
    )

    print("=== Membrane model codegen benchmark ===")
    for row in rows:
        print(
            f"{row.model:22s} {row.phase:5s} r{row.repeat:<2d} "
            f"{row.seconds:.4f}s cache={row.cache_status}/{row.cache_reason} "
            f"bytes={row.generated_bytes}"
        )
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")
    print(f"cache: {cache_root}")


def print_cases() -> None:
    print("Model codegen cases:")
    for name, case in MODEL_CASES.items():
        print(f"  {name:22s} {case.group}")


def select_cases(names: Sequence[str]) -> tuple[ModelCase, ...]:
    requested = tuple(names)
    if "all" in requested:
        return tuple(MODEL_CASES.values())
    cases: list[ModelCase] = []
    if "builtins" in requested:
        cases.extend(BUILTIN_CASES)
    if "custom" in requested:
        cases.extend(CUSTOM_CASES)
    for name in requested:
        if name in {"builtins", "custom"}:
            continue
        cases.append(MODEL_CASES[name])
    deduped: dict[str, ModelCase] = {}
    for case in cases:
        deduped.setdefault(case.name, case)
    return tuple(deduped.values())


def run_codegen_benchmark(
    cases: Sequence[ModelCase],
    *,
    cache_root: Path,
    warm_repeats: int,
) -> tuple[CodegenRow, ...]:
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    previous_cache = os.environ.get("AXONSCOPE_MODEL_CODEGEN_CACHE")
    os.environ["AXONSCOPE_MODEL_CODEGEN_CACHE"] = str(cache_root)
    try:
        rows: list[CodegenRow] = []
        for case in cases:
            rows.append(measure_inspection(case, phase="cold", repeat=0))
            for repeat in range(warm_repeats):
                rows.append(measure_inspection(case, phase="warm", repeat=repeat))
        return tuple(rows)
    finally:
        if previous_cache is None:
            os.environ.pop("AXONSCOPE_MODEL_CODEGEN_CACHE", None)
        else:
            os.environ["AXONSCOPE_MODEL_CODEGEN_CACHE"] = previous_cache


def measure_inspection(case: ModelCase, *, phase: str, repeat: int) -> CodegenRow:
    start = time.perf_counter()
    model = case.factory()
    descriptor = MembraneModel(
        model.kind,
        source_path=model.__class__.source_path(),
        source_class=model.__class__.source_class(),
        dtype=model.dtype,
    )
    report = axs.membranes.inspect_generated_code(descriptor)
    seconds = time.perf_counter() - start
    source = report.sources[0]
    generated_files = tuple(generated.name for generated in source.files)
    generated_bytes = sum(int(generated.size_bytes) for generated in source.files)
    return CodegenRow(
        model=case.name,
        group=case.group,
        phase=phase,
        repeat=int(repeat),
        seconds=float(seconds),
        cache_status=source.cache_status,
        cache_reason=source.cache_reason,
        cache_key=source.cache_key,
        source_hash=source.source_hash,
        source_path=source.source_path,
        generated_bytes=int(generated_bytes),
        generated_files=generated_files,
    )


def run_metadata(
    cases: Sequence[ModelCase],
    *,
    cache_root: Path,
    warm_repeats: int,
) -> dict[str, Any]:
    return {
        "benchmark": "model_codegen",
        "models": [case.name for case in cases],
        "groups": sorted({case.group for case in cases}),
        "warm_repeats": int(warm_repeats),
        "cache_root": str(cache_root),
        "python": sys.version,
        "platform": platform.platform(),
        "axonscope_version": axs.__version__,
        "environment": {
            "AXONSCOPE_MODEL_CODEGEN_CACHE": os.environ.get("AXONSCOPE_MODEL_CODEGEN_CACHE"),
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        },
    }


def write_rows(
    rows: Sequence[CodegenRow],
    *,
    out_dir: Path,
    prefix: str,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"
    payload = {
        "metadata": metadata,
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "group",
                "phase",
                "repeat",
                "seconds",
                "cache_status",
                "cache_reason",
                "cache_key",
                "source_hash",
                "source_path",
                "generated_bytes",
                "generated_files",
            ),
        )
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            data["generated_files"] = ";".join(row.generated_files)
            writer.writerow(data)
    return json_path, csv_path


if __name__ == "__main__":
    main()
