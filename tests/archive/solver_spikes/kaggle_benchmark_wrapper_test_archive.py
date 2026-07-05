import json
import py_compile
import subprocess
from pathlib import Path

from benchmark.kaggle import axonscope_solver_benchmarks as kaggle_bench
from benchmark.kaggle import run_kernel as kaggle_runner
from benchmark.kaggle import stream_logs as kaggle_stream_logs
from benchmark.kaggle.prepare_kernel_metadata import main as prepare_metadata
from benchmark.kaggle.run_kernel import (
    default_publish_branch,
    parse_args,
    parse_kernel_status,
    poll_status,
    run_kernel,
    status_kind,
)


def test_prepare_kaggle_kernel_metadata(tmp_path, capsys):
    prepare_metadata(
        [
            "--username",
            "demo-user",
            "--slug",
            "demo-kernel",
            "--path",
            str(tmp_path),
        ]
    )

    metadata_path = tmp_path / "kernel-metadata.json"
    config_path = tmp_path / "kaggle_config.json"
    code_path = tmp_path / "axonscope_solver_benchmarks_generated.py"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert capsys.readouterr().out.splitlines() == [
        str(metadata_path),
        str(config_path),
        str(code_path),
    ]
    assert metadata["id"] == "demo-user/demo-kernel"
    assert metadata["code_file"] == "axonscope_solver_benchmarks_generated.py"
    assert metadata["kernel_type"] == "script"
    assert metadata["enable_gpu"] == "true"
    assert metadata["enable_internet"] == "true"
    assert config["benchmark"] == "smoke"
    assert config["branch"] == "bench-colab"
    assert config["require_gpu"] is True
    assert "'benchmark': 'smoke'" in code_path.read_text(encoding="utf-8")
    py_compile.compile(str(code_path), doraise=True)


def test_parse_kaggle_kernel_status():
    output = (
        "louisregnacq/axonscope-double-cable-solver-benchmarks "
        'has status "KernelWorkerStatus.ERROR"'
    )

    assert parse_kernel_status(output) == "KernelWorkerStatus.ERROR"
    assert status_kind("KernelWorkerStatus.RUNNING") == "running"
    assert status_kind("KernelWorkerStatus.COMPLETE") == "success"
    assert status_kind("KernelWorkerStatus.ERROR") == "failure"


def test_default_kaggle_publish_branch_uses_commit_sha():
    assert default_publish_branch("abc1234\n") == "kaggle-bench/abc1234"
    assert default_publish_branch("abc1234") == "kaggle-bench/abc1234"


def test_kaggle_backend_probe_escapes_runtime_fstring(monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.verify_backend()

    assert len(commands) == 2
    assert "{backend!r}" in commands[1][-1]


def test_kaggle_gpu_jax_extra_matches_installed_jax_version(monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)
    monkeypatch.setattr(kaggle_bench, "installed_package_version", lambda name: "0.10.1")
    monkeypatch.setattr(kaggle_bench, "JAX_CUDA_EXTRA", "cuda12")
    monkeypatch.setattr(kaggle_bench, "configure_cuda_library_path", lambda: None)
    monkeypatch.setattr(kaggle_bench, "print_cuda_package_diagnostics", lambda: None)

    kaggle_bench.install_jax_gpu_extra()

    assert len(commands) == 1
    command = commands[0]
    assert command[:6] == [
        kaggle_bench.sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-cache-dir",
    ]
    assert "jax==0.10.1" in command
    assert "jaxlib==0.10.1" in command
    assert "jax[cuda12]==0.10.1" in command
    assert "jax-cuda12-plugin[with_cuda]==0.10.1" in command
    assert "nvidia-cublas-cu12" in command


def test_kaggle_smoke_commands_are_small(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.run_linear(tmp_path, smoke=True)
    kaggle_bench.run_e2e(tmp_path, smoke=True)

    linear_command, e2e_command = commands
    assert linear_command[linear_command.index("--batch-sizes") + 1] == "8"
    assert linear_command[linear_command.index("--nx") + 1] == "16"
    assert linear_command[linear_command.index("--repeats") + 1] == "1"
    assert "4096" not in linear_command

    assert e2e_command[e2e_command.index("--batch-sizes") + 1] == "2"
    assert e2e_command[e2e_command.index("--nt") + 1] == "3"
    assert e2e_command[e2e_command.index("--repeats") + 1] == "1"
    assert "1000" not in e2e_command


def test_kaggle_standard_e2e_is_bounded(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.run_e2e(tmp_path, mode="standard")

    (command,) = commands
    assert command[command.index("--batch-sizes") + 1 : command.index("--nx")] == [
        "512",
        "2048",
    ]
    assert command[command.index("--nx") + 1 : command.index("--nt")] == ["51", "96"]
    assert command[command.index("--nt") + 1 : command.index("--dt")] == ["500"]
    assert command[command.index("--recordings") + 1 : command.index("--iinj-modes")] == [
        "none",
        "center",
    ]
    assert command[command.index("--repeats") + 1] == "2"
    assert "1000" not in command


def test_kaggle_linear_excludes_archived_solver_spikes(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.run_linear(tmp_path, smoke=False)

    (command,) = commands
    solvers_start = command.index("--solvers") + 1
    solvers_end = command.index("--warmups")
    assert command[solvers_start:solvers_end] == [
        "thomas",
        "pcr",
        "pcr_soa",
        "pcr_adaptive",
    ]


def test_kaggle_linear_pcr_soa_trace_profiles_focused_gpu_cases(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.run_linear_pcr_soa_trace(tmp_path)

    (command,) = commands
    assert command[command.index("--batch-sizes") + 1 : command.index("--nx")] == [
        "2048",
        "4096",
    ]
    assert command[command.index("--nx") + 1 : command.index("--dtypes")] == [
        "51",
        "96",
    ]
    solvers_start = command.index("--solvers") + 1
    solvers_end = command.index("--warmups")
    assert command[solvers_start:solvers_end] == ["pcr", "pcr_soa", "pcr_adaptive"]
    assert "--skip-reference" in command
    assert "--jax-trace" in command
    assert command[command.index("--jax-trace-dir") + 1] == str(
        tmp_path / "linear_pcr_soa_trace" / "jax_traces"
    )


def test_kaggle_runner_accepts_active_benchmark_choices():
    for benchmark in [
        "smoke",
        "linear",
        "linear_pcr_soa_trace",
        "e2e",
        "e2e_full",
        "realistic_smoke",
        "realistic",
        "realistic_stress",
        "realistic_stress_cpu",
        "realistic_stress_gpu",
        "realistic_stress_single_vm",
        "realistic_stress_single_vm_cpu",
        "realistic_stress_single_vm_gpu",
        "realistic_stress_observer",
        "realistic_stress_observer_cpu",
        "realistic_stress_observer_gpu",
        "population_tsim_gpu",
        "population_tsim_gpu_1000",
        "both",
    ]:
        args = parse_args(["--username", "owner", "--benchmark", benchmark])
        assert args.benchmark == benchmark


def test_kaggle_realistic_smoke_runs_small_workflow_matrix(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.run_realistic_examples(tmp_path, smoke=True)

    (command,) = commands
    assert command[1] == "benchmark/realistic_examples/bench_basic_examples.py"
    assert command[command.index("--preset") + 1] == "smoke"
    assert command[command.index("--repeats") + 1] == "1"
    assert command[
        command.index("--platforms") + 1 : command.index("--example08-recording")
    ] == [
        "cpu",
        "gpu",
    ]


def test_kaggle_realistic_standard_is_bounded(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.run_realistic_examples(tmp_path, smoke=False)

    (command,) = commands
    assert command[command.index("--run-counts") + 1 : command.index("--family-counts")] == [
        "2",
        "5",
    ]
    assert command[command.index("--family-counts") + 1 : command.index("--example07-max-iterations")] == [
        "5",
        "25",
    ]
    assert command[command.index("--example07-max-iterations") + 1] == "8"
    assert command[command.index("--example08-amplitude-count") + 1] == "4"


def test_kaggle_realistic_stress_uses_larger_matrix(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.run_realistic_examples(tmp_path, smoke=False, stress=True)

    (command,) = commands
    assert command[command.index("--preset") + 1] == "stress"
    assert command[command.index("--run-counts") + 1 : command.index("--family-counts")] == [
        "5",
        "10",
        "20",
    ]
    assert command[
        command.index("--family-counts") + 1 : command.index("--example07-max-iterations")
    ] == [
        "25",
        "50",
    ]
    assert command[command.index("--example07-max-iterations") + 1] == "20"
    assert command[command.index("--example08-amplitude-count") + 1] == "8"
    assert command[command.index("--repeats") + 1] == "3"
    assert "--profile" in command


def test_kaggle_realistic_single_vm_stress_uses_center_recording(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.run_realistic_examples(
        tmp_path,
        smoke=False,
        stress=True,
        example08_recording="center",
    )

    (command,) = commands
    assert command[command.index("--example08-recording") + 1] == "center"
    assert command[
        command.index("--platforms") + 1 : command.index("--example08-recording")
    ] == [
        "cpu",
        "gpu",
    ]
    assert "--profile" in command


def test_kaggle_realistic_stress_can_run_gpu_only(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.run_realistic_examples(
        tmp_path,
        smoke=False,
        stress=True,
        platforms=("gpu",),
        progress=True,
    )

    (command,) = commands
    assert command[
        command.index("--platforms") + 1 : command.index("--example08-recording")
    ] == ["gpu"]
    assert "--progress" in command
    assert "--profile" in command


def test_kaggle_population_tsim_gpu_uses_synthetic_axonscope_suite(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.run_population_tsim_gpu(tmp_path)

    (command,) = commands
    assert command[1] == "benchmark/nrv_performance/run.py"
    assert command[command.index("--suite") + 1] == "population_tsim_gpu"
    assert command[command.index("--out-dir") + 1] == str(tmp_path / "population_tsim_gpu")
    assert command[command.index("--prefix") + 1] == "population_tsim_gpu"
    assert command[command.index("--report-dir") + 1] == str(
        tmp_path / "population_tsim_gpu_profiles"
    )


def test_kaggle_population_tsim_gpu_1000_uses_large_suite(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.run_population_tsim_gpu(
        tmp_path,
        suite="population_tsim_gpu_1000",
        prefix="population_tsim_gpu_1000",
    )

    (command,) = commands
    assert command[command.index("--suite") + 1] == "population_tsim_gpu_1000"
    assert command[command.index("--out-dir") + 1] == str(
        tmp_path / "population_tsim_gpu_1000"
    )
    assert command[command.index("--prefix") + 1] == "population_tsim_gpu_1000"
    assert command[command.index("--report-dir") + 1] == str(
        tmp_path / "population_tsim_gpu_1000_profiles"
    )


def test_kaggle_runner_rejects_archived_benchmark_choices():
    for benchmark in [
        "linear_pallas_focus",
        "linear_triton_focus",
        "linear_jax_triton_focus",
        "linear_cuda_ffi_focus",
        "e2e_jax_triton_focus",
        "validate_jax_triton_focus",
    ]:
        try:
            parse_args(["--username", "owner", "--benchmark", benchmark])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"{benchmark} should not be an active Kaggle choice")


def test_kaggle_checkout_stays_out_of_persisted_working_dir():
    assert kaggle_bench.CHECKOUT_DIR == Path("/tmp/AxonScope")


def test_kaggle_output_download_filters_results(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, capture, check, cwd=None):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(kaggle_runner, "run", fake_run)

    exit_code = kaggle_runner.download_outputs(
        "kaggle",
        "owner/kernel",
        tmp_path,
        file_pattern=".*axonscope_solver_results.*",
    )

    assert exit_code == 0
    assert commands == [
        [
            "kaggle",
            "kernels",
            "output",
            "owner/kernel",
            "-p",
            str(tmp_path / "outputs"),
            "-o",
            "--file-pattern",
            ".*axonscope_solver_results.*",
        ]
    ]


def test_kaggle_attach_does_not_push(tmp_path, monkeypatch):
    calls = []

    def fake_poll_status(**kwargs):
        calls.append(("status", kwargs["kernel_ref"]))
        return "KernelWorkerStatus.COMPLETE", "success"

    def fake_fetch_logs(kaggle_bin, kernel_ref, run_dir):
        calls.append(("logs", kernel_ref))

    def fake_fetch_files(kaggle_bin, kernel_ref, run_dir):
        calls.append(("files", kernel_ref))

    def fake_download_outputs(kaggle_bin, kernel_ref, run_dir, *, file_pattern):
        calls.append(("download", kernel_ref, file_pattern))
        return 0

    def fail_unexpected(*args, **kwargs):
        raise AssertionError("attach mode should not publish metadata or push a kernel")

    monkeypatch.setattr(kaggle_runner, "poll_status", fake_poll_status)
    monkeypatch.setattr(kaggle_runner, "fetch_logs", fake_fetch_logs)
    monkeypatch.setattr(kaggle_runner, "fetch_files", fake_fetch_files)
    monkeypatch.setattr(kaggle_runner, "download_outputs", fake_download_outputs)
    monkeypatch.setattr(kaggle_runner, "publish_git_branch", fail_unexpected)
    monkeypatch.setattr(kaggle_runner, "prepare_metadata", fail_unexpected)

    args = parse_args(["--username", "owner", "--attach"])
    exit_code = run_kernel(args, "kaggle", "owner/kernel", tmp_path)

    assert exit_code == 0
    assert calls == [
        ("status", "owner/kernel"),
        ("logs", "owner/kernel"),
        ("files", "owner/kernel"),
        ("download", "owner/kernel", ".*axonscope_solver_results.*"),
    ]


def test_kaggle_status_retry_after_successful_running_status(tmp_path, monkeypatch):
    responses = [
        subprocess.CompletedProcess(
            ["kaggle", "kernels", "status", "owner/kernel"],
            0,
            'owner/kernel has status "KernelWorkerStatus.RUNNING"',
            "",
        ),
        subprocess.CompletedProcess(
            ["kaggle", "kernels", "status", "owner/kernel"],
            1,
            "",
            "Cannot access kernel 'owner/kernel' (Permission 'kernels.get' was denied).",
        ),
        subprocess.CompletedProcess(
            ["kaggle", "kernels", "status", "owner/kernel"],
            0,
            'owner/kernel has status "KernelWorkerStatus.COMPLETE"',
            "",
        ),
    ]

    def fake_run(command, *, capture, check, cwd=None):
        return responses.pop(0)

    monkeypatch.setattr(kaggle_runner, "run", fake_run)
    monkeypatch.setattr(kaggle_runner.time, "sleep", lambda _seconds: None)

    status, kind = poll_status(
        kaggle_bin="kaggle",
        kernel_ref="owner/kernel",
        poll_interval=1,
        wait_timeout=30,
        max_status_fetch_failures=2,
        run_dir=tmp_path,
    )

    assert status == "KernelWorkerStatus.COMPLETE"
    assert kind == "success"
    assert not responses
    assert "Permission 'kernels.get' was denied" in (tmp_path / "status.log").read_text()


def test_kaggle_status_retry_uses_existing_status_log_for_attach(tmp_path, monkeypatch):
    (tmp_path / "status.log").write_text(
        'owner/kernel has status "KernelWorkerStatus.RUNNING"\n',
        encoding="utf-8",
    )
    responses = [
        subprocess.CompletedProcess(
            ["kaggle", "kernels", "status", "owner/kernel"],
            1,
            "",
            "Cannot access kernel 'owner/kernel' (Permission 'kernels.get' was denied).",
        ),
        subprocess.CompletedProcess(
            ["kaggle", "kernels", "status", "owner/kernel"],
            0,
            'owner/kernel has status "KernelWorkerStatus.COMPLETE"',
            "",
        ),
    ]

    def fake_run(command, *, capture, check, cwd=None):
        return responses.pop(0)

    monkeypatch.setattr(kaggle_runner, "run", fake_run)
    monkeypatch.setattr(kaggle_runner.time, "sleep", lambda _seconds: None)

    status, kind = poll_status(
        kaggle_bin="kaggle",
        kernel_ref="owner/kernel",
        poll_interval=1,
        wait_timeout=30,
        max_status_fetch_failures=2,
        run_dir=tmp_path,
    )

    assert status == "KernelWorkerStatus.COMPLETE"
    assert kind == "success"
    assert not responses


def test_kaggle_stream_log_formatters():
    payload = json.dumps({"stream_name": "stdout", "time": 1.0, "data": "case 1/48\n"})
    persisted = json.dumps(
        [
            {"stream_name": "stdout", "time": 1.0, "data": "hello\n"},
            {"stream_name": "stderr", "time": 2.0, "data": "warn\n"},
        ]
    )

    assert kaggle_stream_logs.parse_kernel_ref("owner/kernel") == ("owner", "kernel")
    assert kaggle_stream_logs.format_stream_payload(payload) == "case 1/48\n"
    assert kaggle_stream_logs.format_persisted_log(persisted) == "hello\nwarn\n"
