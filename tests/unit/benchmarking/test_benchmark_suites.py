from __future__ import annotations

from pathlib import Path

from benchmark.nrv_performance.run import suite_argv as nrv_performance_suite_argv
from benchmark.nrv_performance.suites import NRV_PERFORMANCE_SUITES
from benchmark.runtime.run import suite_argv as runtime_suite_argv
from benchmark.runtime.suites import RUNTIME_SUITES


def test_nrv_performance_suites_include_smoke_and_forward_args():
    suite = NRV_PERFORMANCE_SUITES["smoke"]

    argv = nrv_performance_suite_argv(
        suite,
        out_dir=Path("out"),
        prefix="prefix",
        dry_run=True,
        extra_args=("--", "--dt", "0.02"),
    )

    assert argv[:2] == ["--profile", "smoke"]
    assert "--dry-run" in argv
    assert argv[-2:] == ["--dt", "0.02"]
    assert argv[argv.index("--out-dir") + 1] == "out"
    assert argv[argv.index("--prefix") + 1] == "prefix"


def test_nrv_mrg_extracellular_perf_suite_has_warm_repeats():
    suite = NRV_PERFORMANCE_SUITES["mrg_extracellular_perf"]

    assert suite.runner == "nrv_axonscope_grid"
    assert suite.args[suite.args.index("--model") + 1] == "mrg_extracellular"
    assert suite.args[suite.args.index("--repeats") + 1] == "4"
    assert suite.args[suite.args.index("--warmups") + 1] == "1"
    assert "--record-gates" not in suite.args


def test_runtime_suites_forward_to_benchmark_solver():
    suite = RUNTIME_SUITES["smoke"]

    argv = runtime_suite_argv(
        suite,
        out_dir=Path("runtime-out"),
        prefix="smoke",
        extra_args=("--", "--record-diagnostics"),
    )

    assert argv[:4] == ["--cases", "hh_intracellular_small", "--repeats", "1"]
    assert argv[-1] == "--record-diagnostics"
    assert argv[argv.index("--out-dir") + 1] == "runtime-out"
    assert argv[argv.index("--prefix") + 1] == "smoke"


def test_runtime_population_memory_suite_is_registered_and_forwardable():
    suite = RUNTIME_SUITES["population_memory"]

    argv = runtime_suite_argv(
        suite,
        out_dir=Path("runtime-out"),
        prefix="population",
        extra_args=("--", "--scenarios", "full"),
    )

    assert suite.runner == "population_memory"
    assert argv[argv.index("--fibers") + 1] == "128"
    assert argv[argv.index("--nx") + 1] == "201"
    assert argv[-2:] == ["--scenarios", "full"]
