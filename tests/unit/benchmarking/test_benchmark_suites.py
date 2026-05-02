from __future__ import annotations

from pathlib import Path

from benchmark.runtime.run import suite_argv as runtime_suite_argv
from benchmark.runtime.suites import RUNTIME_SUITES
from benchmark.validation.run import suite_argv as validation_suite_argv
from benchmark.validation.suites import VALIDATION_SUITES


def test_validation_suites_include_nrv_smoke_and_forward_args():
    suite = VALIDATION_SUITES["nrv_smoke"]

    argv = validation_suite_argv(
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
