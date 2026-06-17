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
    parse_kernel_status,
    parse_args,
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


def test_kaggle_split_focus_e2e_is_bounded_to_array_kernel_candidates(
    tmp_path,
    monkeypatch,
):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.run_e2e(tmp_path, mode="split_focus")

    (command,) = commands
    assert command[command.index("--batch-sizes") + 1 : command.index("--nx")] == [
        "1024",
        "2048",
        "4096",
    ]
    assert command[command.index("--nx") + 1 : command.index("--nt")] == ["51", "96"]
    assert command[command.index("--recordings") + 1 : command.index("--iinj-modes")] == [
        "center"
    ]
    assert command[command.index("--iinj-modes") + 1 : command.index("--solvers")] == [
        "none"
    ]
    assert command[command.index("--solvers") + 1 : command.index("--warmups")] == [
        "pcr_adaptive",
        "split_gs_3",
        "split_gs_4",
    ]


def test_kaggle_linear_includes_active_split_iterative_candidates(tmp_path, monkeypatch):
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
        "split_jacobi_4",
        "split_jacobi4_gs1",
        "split_gs_2",
        "split_gs_3",
        "split_gs_4",
    ]


def test_kaggle_linear_split_focus_is_bounded_to_current_candidates(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(kaggle_bench, "run", fake_run)

    kaggle_bench.run_linear_split_focus(tmp_path)

    (command,) = commands
    assert command[command.index("--batch-sizes") + 1 : command.index("--nx")] == [
        "1024",
        "2048",
        "4096",
    ]
    assert command[command.index("--nx") + 1 : command.index("--dtypes")] == [
        "32",
        "51",
        "64",
        "96",
    ]
    solvers_start = command.index("--solvers") + 1
    solvers_end = command.index("--warmups")
    assert command[solvers_start:solvers_end] == [
        "pcr_soa",
        "pcr_adaptive",
        "split_jacobi_4",
        "split_jacobi4_gs1",
        "split_gs_2",
        "split_gs_3",
        "split_gs_4",
    ]


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
