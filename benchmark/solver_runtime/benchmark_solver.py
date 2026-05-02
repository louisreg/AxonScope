from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from axonscope.benchmarking import (
    default_solver_benchmark_cases,
    default_solver_factories,
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

    results = run_solver_benchmark_suite(
        selected_cases,
        selected_solvers,
        repeats=args.repeats,
        warmups=args.warmups,
        solve_kwargs=solve_kwargs,
    )
    prefix = args.prefix or datetime.now().strftime("solver_benchmark_%Y%m%d_%H%M%S")
    json_path, csv_path = write_benchmark_results(results, args.out_dir, prefix=prefix)

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


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
