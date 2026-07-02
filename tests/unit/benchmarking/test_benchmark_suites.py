from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from benchmark.registry import BENCHMARK_SURFACES, surfaces_by_status
from benchmark.nrv_performance.realistic_fascicle_recruitment import profile_report_metrics
from benchmark.nrv_performance.run import suite_argv as nrv_performance_suite_argv
from benchmark.nrv_performance.suites import NRV_PERFORMANCE_SUITES
from benchmark.runtime.run import suite_argv as runtime_suite_argv
from benchmark.runtime.suites import RUNTIME_SUITES
from benchmark.runtime.model_codegen import (
    run_model_step_benchmark,
    select_cases,
    select_simulation_cases,
)


EXPECTED_COMMAND_KINDS = {
    "public-runtime",
    "hotpath-diagnostic",
    "model-codegen",
    "validation-only",
    "external-comparison",
    "remote-GPU",
    "archive",
    "generated-output",
}


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


def test_benchmark_surface_registry_classifies_active_archive_and_outputs():
    surfaces = {surface.path: surface for surface in BENCHMARK_SURFACES}

    assert surfaces["benchmark/runtime"].status == "active"
    assert surfaces["benchmark/hotpaths"].status == "active"
    assert surfaces["benchmark/solvers"].status == "validation-only"
    assert surfaces["benchmark/triton_solver"].status == "archive"
    assert surfaces["benchmark/results"].status == "generated-output"
    assert surfaces["benchmark/reports"].status == "generated-output"
    assert surfaces_by_status("active")
    assert surfaces_by_status("archive")

    for surface in BENCHMARK_SURFACES:
        assert surface.commands
        assert surface.command_kinds
        for command in surface.commands:
            assert command.command
            assert command.kind in EXPECTED_COMMAND_KINDS
            assert command.purpose

    assert "model-codegen" in surfaces["benchmark/runtime"].command_kinds
    assert surfaces["benchmark/hotpaths"].command_kinds == ("hotpath-diagnostic",)
    assert surfaces["benchmark/kaggle"].command_kinds == ("remote-GPU", "generated-output")
    assert surfaces["benchmark/results"].command_kinds == ("generated-output",)


def test_nrv_mrg_extracellular_perf_suite_has_warm_repeats():
    suite = NRV_PERFORMANCE_SUITES["mrg_extracellular_perf"]

    assert suite.runner == "nrv_axonscope_grid"
    assert suite.args[suite.args.index("--model") + 1] == "mrg_extracellular"
    assert suite.args[suite.args.index("--repeats") + 1] == "4"
    assert suite.args[suite.args.index("--warmups") + 1] == "1"
    assert "--record-gates" not in suite.args


def test_population_tsim_gpu_suite_is_axonscope_only_and_synthetic():
    suite = NRV_PERFORMANCE_SUITES["population_tsim_gpu"]

    assert suite.runner == "population_tsim_scaling"
    assert suite.args[suite.args.index("--runner") + 1] == "axonscope"
    assert suite.args[suite.args.index("--geometry-source") + 1] == "synthetic"
    assert suite.args[suite.args.index("--device") + 1] == "gpu"
    assert "--profile-cold-path" in suite.args
    assert "--profile-warm-path" in suite.args


def test_population_tsim_gpu_1000_suite_uses_large_synthetic_case():
    suite = NRV_PERFORMANCE_SUITES["population_tsim_gpu_1000"]

    assert suite.runner == "population_tsim_scaling"
    assert suite.args[suite.args.index("--runner") + 1] == "axonscope"
    assert suite.args[suite.args.index("--geometry-source") + 1] == "synthetic"
    assert suite.args[suite.args.index("--device") + 1] == "gpu"
    assert suite.args[suite.args.index("--fiber-counts") + 1] == "1000"


def test_realistic_fascicle_suite_keeps_runner_default_out_dir():
    suite = NRV_PERFORMANCE_SUITES["realistic_fascicle_smoke"]

    argv = nrv_performance_suite_argv(suite, out_dir=None)

    assert suite.runner == "realistic_fascicle_recruitment"
    assert "--out-dir" not in argv


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


def test_runtime_pool_memory_suite_is_registered_and_forwardable():
    suite = RUNTIME_SUITES["pool_memory"]

    argv = runtime_suite_argv(
        suite,
        out_dir=Path("runtime-out"),
        prefix="pool",
        extra_args=("--", "--scenarios", "full"),
    )

    assert suite.runner == "pool_memory"
    assert argv[argv.index("--fibers") + 1] == "128"
    assert argv[argv.index("--nx") + 1] == "201"
    assert argv[-2:] == ["--scenarios", "full"]


def test_runtime_model_codegen_suite_is_registered_and_forwardable():
    suite = RUNTIME_SUITES["model_codegen"]

    argv = runtime_suite_argv(
        suite,
        out_dir=Path("runtime-out"),
        prefix="models",
        extra_args=("--", "--warm-repeats", "1"),
    )

    assert suite.runner == "model_codegen"
    assert argv[:2] == ["--models", "builtins"]
    assert "--simulation-cases" in argv
    assert argv[argv.index("--simulation-cases") + 1] == "none"
    assert argv[argv.index("--out-dir") + 1] == "runtime-out"
    assert argv[argv.index("--prefix") + 1] == "models"
    assert argv[-2:] == ["--warm-repeats", "1"]


def test_runtime_model_codegen_simulation_suite_is_registered_and_forwardable():
    suite = RUNTIME_SUITES["model_codegen_simulations"]

    argv = runtime_suite_argv(
        suite,
        out_dir=Path("runtime-out"),
        prefix="model-sims",
        extra_args=("--", "--simulation-warm-repeats", "0"),
    )

    assert suite.runner == "model_codegen"
    assert argv[:2] == ["--models", "passive"]
    assert "--no-model-steps" in argv
    assert argv[argv.index("--simulation-cases") + 1] == "representative"
    assert argv[-2:] == ["--simulation-warm-repeats", "0"]


def test_model_codegen_case_selection_and_passive_step_benchmark(tmp_path, monkeypatch):
    monkeypatch.setenv("AXONSCOPE_MODEL_CODEGEN_CACHE", str(tmp_path / "codegen"))

    rows, correctness = run_model_step_benchmark(
        select_cases(["passive"]),
        repeats=1,
        node_count=3,
    )

    assert {row.target for row in rows} == {
        "numpy_interpreter",
        "generated_numpy",
        "generated_jax",
        "jax_runtime_lowering",
    }
    assert all(row.status == "ok" for row in rows)
    assert {row.target for row in correctness} == {
        "generated_jax_vs_numpy",
        "generated_numpy_vs_interpreter",
    }
    assert all(row.status == "ok" for row in correctness)
    assert select_simulation_cases(["smoke"])[0].name == "hh_template"


def test_realistic_fascicle_profile_metrics_flatten_nbytes_components():
    report = SimpleNamespace(
        events=[
            SimpleNamespace(
                metadata={
                    "memory_estimate_components_nbytes": {
                        "vstim_mid": 10,
                        "vm_output": 5,
                    },
                    "memory_estimate_total_nbytes": 15,
                }
            )
        ],
        summary=[],
        metadata={},
    )

    metrics = profile_report_metrics(report)

    assert metrics["profile_memory_estimate_components_nbytes"] == 15
    assert metrics["profile_memory_estimate_components_nbytes_vstim_mid"] == 10
    assert metrics["profile_memory_estimate_components_nbytes_vm_output"] == 5
    assert metrics["profile_memory_estimate_total_nbytes"] == 15
