from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from axonscope.benchmarking import (
    default_solver_benchmark_cases,
    default_solver_factories,
    jax_profile_trace,
    run_solver_benchmark_suite,
    write_benchmark_results,
)


def main() -> None:
    cases_by_name = default_solver_benchmark_cases()
    solvers_by_name = default_solver_factories()

    parser = argparse.ArgumentParser(description="Benchmark AxonScope Solver workloads.")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["hh_intracellular_small"],
        choices=["all", *cases_by_name.keys()],
        help="Benchmark case names. Use 'all' to run every default case.",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["crank_nicholson"],
        choices=["all", *solvers_by_name.keys()],
        help="Solver names. Use 'all' to run every default solver.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Measured warm solve repetitions.")
    parser.add_argument("--warmups", type=int, default=1, help="Warm solves before measured repetitions.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark/results/solver_runtime"),
        help="Directory for JSON and CSV benchmark outputs.",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix.")
    parser.add_argument(
        "--jax-profile-dir",
        type=Path,
        default=None,
        help="Optional base directory for a JAX profiler trace.",
    )
    parser.add_argument(
        "--jax-profile-name",
        default=None,
        help="Optional JAX profiler run directory name. Defaults to the output prefix.",
    )
    parser.add_argument(
        "--jax-profile-perfetto",
        action="store_true",
        help="Also emit a local Perfetto trace file when supported by JAX.",
    )
    parser.add_argument(
        "--jax-profile-link",
        action="store_true",
        help="Ask JAX to create a Perfetto link for the captured trace.",
    )
    parser.add_argument("--record-observables", action="store_true", help="Record gates/currents/states during solves.")
    parser.add_argument("--record-diagnostics", action="store_true", help="Record model diagnostics during solves.")
    parser.add_argument("--list", action="store_true", help="List available cases and solvers, then exit.")
    args = parser.parse_args()

    if args.list:
        print("Cases:")
        for name, case in cases_by_name.items():
            print(f"  {name:32s} tsim={case.tsim_ms:g} ms dt={case.dt_ms:g} ms")
        print("Solvers:")
        for name in solvers_by_name:
            print(f"  {name}")
        return

    selected_cases = list(cases_by_name.values()) if "all" in args.cases else [cases_by_name[name] for name in args.cases]
    selected_solvers = solvers_by_name if "all" in args.solvers else {name: solvers_by_name[name] for name in args.solvers}
    solve_kwargs = {
        "record_observables": bool(args.record_observables),
        "record_diagnostics": bool(args.record_diagnostics),
    }

    prefix = args.prefix or datetime.now().strftime("solver_benchmark_%Y%m%d_%H%M%S")
    jax_profile_path = _profile_path(args.jax_profile_dir, args.jax_profile_name, prefix)
    profile_context = (
        jax_profile_trace(
            jax_profile_path,
            create_perfetto_trace=bool(args.jax_profile_perfetto),
            create_perfetto_link=bool(args.jax_profile_link),
        )
        if jax_profile_path is not None
        else nullcontext()
    )

    with profile_context:
        results = run_solver_benchmark_suite(
            selected_cases,
            selected_solvers,
            repeats=args.repeats,
            warmups=args.warmups,
            solve_kwargs=solve_kwargs,
        )

    run_metadata = {
        "benchmark_cases": [case.name for case in selected_cases],
        "benchmark_solvers": list(selected_solvers),
        "benchmark_repeats": int(args.repeats),
        "benchmark_warmups": int(args.warmups),
        "solve_kwargs": solve_kwargs,
    }
    if jax_profile_path is not None:
        run_metadata["jax_profile_dir"] = str(jax_profile_path)
        run_metadata["jax_profile_perfetto"] = bool(args.jax_profile_perfetto)
        run_metadata["jax_profile_link"] = bool(args.jax_profile_link)

    json_path, csv_path = write_benchmark_results(
        results,
        args.out_dir,
        prefix=prefix,
        run_metadata=run_metadata,
    )

    print("=== Solver benchmark ===")
    for result in results:
        print(
            f"{result.case_name:32s} {result.solver_name:24s} "
            f"build={result.construction.mean_s:.4f}s "
            f"first={result.first_solve_s:.4f}s "
            f"compile_est={_fmt_optional(result.compile_s_estimate)}s "
            f"mat={result.materialize_first_s:.4f}s "
            f"total={result.total_first_s:.4f}s "
            f"warm={result.warm_solve.mean_s:.4f}s "
            f"warm_total={result.warm_total.mean_s:.4f}s "
            f"Vm={result.output['vm_min_mV']:.2f}/{result.output['vm_max_mV']:.2f} mV"
        )
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")
    if jax_profile_path is not None:
        print(f"jax profile: {jax_profile_path}")


def _profile_path(base_dir: Path | None, profile_name: str | None, prefix: str) -> Path | None:
    if base_dir is None:
        return None
    return base_dir / (profile_name or prefix)


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
