from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))


DEFAULT_CASES = (
    "hh_intracellular",
    "hh_extracellular",
    "double_cable_extracellular",
    "schild_intracellular",
    "tigerholm_intracellular",
)
DEFAULT_RATE_TABLE_CASES = ("schild_intracellular", "tigerholm_intracellular")


@dataclass(frozen=True)
class CaseSpec:
    name: str
    tsim_ms: float
    dt_ms: float
    metadata: dict[str, Any]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark dtype precision and tabulated alpha/beta rates."
    )
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    parser.add_argument(
        "--rate-modes",
        nargs="+",
        choices=["exact", "rate_table", "lut"],
        default=["exact", "rate_table"],
        help="'lut' is accepted as an alias for 'rate_table'.",
    )
    parser.add_argument(
        "--rate-table-cases",
        "--lut-cases",
        nargs="+",
        default=list(DEFAULT_RATE_TABLE_CASES),
    )
    parser.add_argument("--rate-table-step-mv", "--table-step-mv", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark/results/runtime"))
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        _run_worker(args)
    else:
        _run_parent(args)


def _run_parent(args: argparse.Namespace) -> None:
    args.rate_modes = _normalize_rate_modes(args.rate_modes)
    prefix = args.prefix or datetime.now().strftime("precision_rates_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / f"{prefix}_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    worker_script = Path(__file__).resolve()
    for dtype_name in args.dtypes:
        env = os.environ.copy()
        env["AXONSCOPE_DTYPE"] = dtype_name
        if dtype_name == "float64":
            env["AXONSCOPE_ENABLE_X64"] = "1"

        cmd = [
            sys.executable,
            str(worker_script),
            "--worker",
            "--suite",
            args.suite,
            "--rate-table-step-mv",
            str(args.rate_table_step_mv),
            "--repeats",
            str(args.repeats),
            "--warmups",
            str(args.warmups),
            "--out-dir",
            str(trace_dir),
            "--prefix",
            prefix,
            "--dtypes",
            dtype_name,
            "--rate-modes",
            *args.rate_modes,
            "--rate-table-cases",
            *args.rate_table_cases,
        ]
        if args.cases:
            cmd.extend(["--cases", *args.cases])

        completed = subprocess.run(
            cmd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise SystemExit(completed.returncode)
        payload = json.loads(completed.stdout)
        rows.extend(payload["rows"])

    _attach_float64_reference_metrics(rows)
    json_path, csv_path = _write_outputs(
        rows,
        out_dir=out_dir,
        prefix=prefix,
        metadata={
            "schema_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "suite": args.suite,
            "cases": args.cases or list(DEFAULT_CASES),
            "dtypes": args.dtypes,
            "rate_modes": args.rate_modes,
            "rate_table_cases": args.rate_table_cases,
            "rate_table_step_mV": args.rate_table_step_mv,
            "repeats": args.repeats,
            "warmups": args.warmups,
        },
    )

    print("=== Precision/rate-table benchmark ===")
    for row in rows:
        print(
            f"{row['case_name']:28s} dtype={row['dtype']:7s} mode={row['rate_mode']:5s} "
            f"first={row['first_s']:.4f}s warm={row['warm_mean_s']:.4f}s "
            f"rmse_exact={_fmt(row.get('rmse_vs_exact_mV'))} mV "
            f"rmse_fp64={_fmt(row.get('rmse_vs_float64_exact_mV'))} mV "
            f"Vm={row['vm_min_mV']:.2f}/{row['vm_max_mV']:.2f} mV"
        )
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")


def _run_worker(args: argparse.Namespace) -> None:
    args.rate_modes = _normalize_rate_modes(args.rate_modes)
    dtype_name = args.dtypes[0]
    cases = args.cases or list(DEFAULT_CASES)
    if args.suite == "smoke":
        case_specs = _case_specs_smoke()
    else:
        case_specs = _case_specs_full()

    rows: list[dict[str, Any]] = []
    exact_traces: dict[str, np.ndarray] = {}
    for case_name in cases:
        if case_name not in case_specs:
            raise ValueError(f"Unknown case {case_name!r}. Available: {sorted(case_specs)}")
        modes = [
            mode
            for mode in args.rate_modes
            if mode == "exact" or case_name in set(args.rate_table_cases)
        ]
        for mode in modes:
            row, vm = _benchmark_case(
                case_specs[case_name],
                dtype_name=dtype_name,
                rate_mode=mode,
                rate_table_step_mV=float(args.rate_table_step_mv),
                repeats=int(args.repeats),
                warmups=int(args.warmups),
            )
            if mode == "exact":
                exact_traces[case_name] = vm
                row["rmse_vs_exact_mV"] = 0.0
                row["max_abs_vs_exact_mV"] = 0.0
            elif case_name in exact_traces:
                diff = vm.astype(np.float64) - exact_traces[case_name].astype(np.float64)
                row["rmse_vs_exact_mV"] = float(np.sqrt(np.mean(diff * diff)))
                row["max_abs_vs_exact_mV"] = float(np.max(np.abs(diff)))

            trace_path = _trace_path(
                Path(args.out_dir),
                prefix=args.prefix or "precision_rates",
                case_name=case_name,
                dtype_name=dtype_name,
                rate_mode=mode,
            )
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(trace_path, Vm=vm)
            row["trace_path"] = str(trace_path)
            rows.append(row)

    print(json.dumps({"rows": rows}, sort_keys=True))


def _case_specs_smoke() -> dict[str, CaseSpec]:
    return {
        "hh_intracellular": CaseSpec(
            "hh_intracellular",
            tsim_ms=2.0,
            dt_ms=0.02,
            metadata={"model": "HodgkinHuxley", "stimulation": "intracellular", "Nx": 41},
        ),
        "hh_extracellular": CaseSpec(
            "hh_extracellular",
            tsim_ms=1.5,
            dt_ms=0.01,
            metadata={"model": "HodgkinHuxley", "stimulation": "extracellular", "Nx": 81},
        ),
        "double_cable_extracellular": CaseSpec(
            "double_cable_extracellular",
            tsim_ms=1.0,
            dt_ms=0.01,
            metadata={"model": "MRG", "stimulation": "extracellular", "nodes": 5},
        ),
        "schild_intracellular": CaseSpec(
            "schild_intracellular",
            tsim_ms=2.0,
            dt_ms=0.02,
            metadata={"model": "Schild97", "stimulation": "intracellular", "Nx": 31},
        ),
        "tigerholm_intracellular": CaseSpec(
            "tigerholm_intracellular",
            tsim_ms=1.0,
            dt_ms=0.02,
            metadata={"model": "Tigerholm", "stimulation": "intracellular", "Nx": 31},
        ),
    }


def _case_specs_full() -> dict[str, CaseSpec]:
    specs = _case_specs_smoke()
    return {
        name: CaseSpec(
            spec.name,
            tsim_ms=spec.tsim_ms * 2.0,
            dt_ms=spec.dt_ms,
            metadata=dict(spec.metadata),
        )
        for name, spec in specs.items()
    }


def _benchmark_case(
    spec: CaseSpec,
    *,
    dtype_name: str,
    rate_mode: str,
    rate_table_step_mV: float,
    repeats: int,
    warmups: int,
) -> tuple[dict[str, Any], np.ndarray]:
    from axonscope.solvers import CrankNicholson

    first_s, first_vm, table_count = _solve_case_once(spec, rate_mode, rate_table_step_mV)
    for _ in range(warmups):
        _solve_case_once(spec, rate_mode, rate_table_step_mV)

    warm_times: list[float] = []
    vm = first_vm
    for _ in range(repeats):
        elapsed, vm, _ = _solve_case_once(spec, rate_mode, rate_table_step_mV)
        warm_times.append(elapsed)

    # Keep the import in this scope so the worker picks up AXONSCOPE_DTYPE first.
    _ = CrankNicholson
    row = {
        "case_name": spec.name,
        "dtype": dtype_name,
        "rate_mode": rate_mode,
        "rate_table_enabled": bool(rate_mode == "rate_table"),
        "rate_table_count": int(table_count),
        "rate_table_step_mV": float(rate_table_step_mV) if rate_mode == "rate_table" else None,
        "tsim_ms": float(spec.tsim_ms),
        "dt_ms": float(spec.dt_ms),
        "first_s": float(first_s),
        "warm_mean_s": float(statistics.fmean(warm_times)),
        "warm_median_s": float(statistics.median(warm_times)),
        "warm_min_s": float(min(warm_times)),
        "warm_max_s": float(max(warm_times)),
        "vm_shape": "x".join(str(int(x)) for x in vm.shape),
        "vm_min_mV": float(np.min(vm)),
        "vm_max_mV": float(np.max(vm)),
        "vm_mean_mV": float(np.mean(vm)),
        **spec.metadata,
    }
    return row, vm


def _solve_case_once(
    spec: CaseSpec,
    rate_mode: str,
    rate_table_step_mV: float,
) -> tuple[float, np.ndarray, int]:
    from axonscope.channel_models import enable_rate_tables
    from axonscope.solvers import CrankNicholson

    axon = _build_case(spec.name)
    table_count = 0
    if rate_mode == "rate_table":
        table_count = enable_rate_tables(
            axon.layout.sections[0].membrane,
            step_mV=rate_table_step_mV,
        )

    solver = CrankNicholson()
    start = time.perf_counter()
    result = solver.solve(axon, tsim=spec.tsim_ms, dt=spec.dt_ms)
    _block_until_ready(result)
    elapsed = time.perf_counter() - start
    return float(elapsed), np.asarray(result.Vm, dtype=np.float64), table_count


def _build_case(case_name: str):
    from axonscope import AxonSimulation, degC, um
    from axonscope.axons import HodgkinHuxley, MRG, Schild97, Tigerholm
    from axonscope.stimulation import AnalyticalExtracellularContext, PointSourceElectrode
    from axonscope.stimulation import Stimulus

    if case_name == "hh_intracellular":
        length_um = 500.0
        axon = HodgkinHuxley(length=length_um * um, diameter=0.5 * um, compartments=41, celsius=6.3 * degC)
        simulation = AxonSimulation(axon)
        simulation.add_current_clamp(position_um=length_um / 2.0,
            current=Stimulus.pulse(start=0.5, duration=0.4, amplitude=2.0),
        )
        return simulation

    if case_name == "hh_extracellular":
        length_um = 500.0
        axon = HodgkinHuxley(length=length_um * um, diameter=0.5 * um, compartments=81, celsius=6.3 * degC)
        simulation = AxonSimulation(axon)
        electrode = PointSourceElectrode(
            x0_m=(length_um / 2.0) * 1e-6,
            y0_m=100e-6,
            z0_m=0.0,
        )
        stimulus = Stimulus.biphasic(
            start=0.4,
            cathodic_amplitude=20e-6,
            cathodic_duration=0.08,
            anodic_amplitude=5e-6,
            interphase=0.04,
        )
        simulation.add_extracellular_context(
            context=AnalyticalExtracellularContext(
                electrodes=[electrode.with_stimulus(stimulus)],
                sigma=0.3,
            ),
            replace=True,
        )
        return simulation

    if case_name == "double_cable_extracellular":
        axon = MRG(diameter=10.0 * um, nodes=5)
        simulation = AxonSimulation(axon)
        x0_m = float(
            np.asarray(axon.layout.position_values(unit="micrometer"))[axon.n_compartments // 2]
        ) * 1e-6
        electrode = PointSourceElectrode(
            x0_m=x0_m,
            y0_m=100e-6,
            z0_m=0.0,
        )
        stimulus = Stimulus.biphasic(
            start=0.3,
            cathodic_amplitude=80e-6,
            cathodic_duration=0.05,
            anodic_amplitude=20e-6,
            interphase=0.02,
        )
        simulation.add_extracellular_context(
            context=AnalyticalExtracellularContext(
                electrodes=[electrode.with_stimulus(stimulus)],
                sigma=0.2,
            ),
            replace=True,
        )
        return simulation

    if case_name == "schild_intracellular":
        length_um = 1200.0
        axon = Schild97(length=length_um * um, diameter=0.8 * um, compartments=31)
        simulation = AxonSimulation(axon)
        simulation.add_current_clamp(position_um=length_um / 2.0,
            current=Stimulus.pulse(start=0.5, duration=0.5, amplitude=0.8),
        )
        return simulation

    if case_name == "tigerholm_intracellular":
        length_um = 1200.0
        axon = Tigerholm(length=length_um * um, diameter=0.8 * um, compartments=31)
        simulation = AxonSimulation(axon)
        simulation.add_current_clamp(position_um=length_um / 2.0,
            current=Stimulus.pulse(start=0.3, duration=0.4, amplitude=0.7),
        )
        return simulation

    raise ValueError(f"Unknown case {case_name!r}.")


def _block_until_ready(result: Any) -> None:
    for attr in ("Vm", "t"):
        value = getattr(result, attr, None)
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()


def _trace_path(
    out_dir: Path,
    *,
    prefix: str,
    case_name: str,
    dtype_name: str,
    rate_mode: str,
) -> Path:
    return out_dir / f"{prefix}_{case_name}_{dtype_name}_{rate_mode}.npz"


def _attach_float64_reference_metrics(rows: list[dict[str, Any]]) -> None:
    by_key = {
        (row["case_name"], row["dtype"], row["rate_mode"]): row
        for row in rows
    }
    for row in rows:
        ref = by_key.get((row["case_name"], "float64", "exact"))
        if ref is None:
            continue
        try:
            current_vm = np.load(row["trace_path"])["Vm"]
            ref_vm = np.load(ref["trace_path"])["Vm"]
        except Exception:
            continue
        if current_vm.shape != ref_vm.shape:
            continue
        diff = current_vm.astype(np.float64) - ref_vm.astype(np.float64)
        row["rmse_vs_float64_exact_mV"] = float(np.sqrt(np.mean(diff * diff)))
        row["max_abs_vs_float64_exact_mV"] = float(np.max(np.abs(diff)))


def _write_outputs(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    prefix: str,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"
    json_path.write_text(
        json.dumps({"metadata": metadata, "results": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4g}"


def _normalize_rate_modes(modes: Sequence[str]) -> list[str]:
    normalized = ["rate_table" if mode == "lut" else mode for mode in modes]
    return list(dict.fromkeys(normalized))


if __name__ == "__main__":
    main()
