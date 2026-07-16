from __future__ import annotations

import json
from pathlib import Path

from benchmark.kaggle.kernel_entry import (
    _benchmark_environment,
    _jax_cuda_requirement,
    _redact_env_value,
)
from benchmark.kaggle.run_kernel import main as run_kaggle


def test_kaggle_cuda_install_preserves_project_jax_constraint(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\ndependencies = ["numpy>=1.26", "jax>=0.10.1,<0.11.0"]\n',
        encoding="utf-8",
    )

    assert (
        _jax_cuda_requirement("cuda12", pyproject_path=pyproject)
        == "jax[cuda12]>=0.10.1,<0.11.0"
    )


def test_kaggle_runner_dry_run_writes_kernel_package(tmp_path: Path):
    assert (
        run_kaggle(
            [
                "--username",
                "demo-user",
                "--script",
                "threshold_curves",
                "--preset",
                "gpu_smoke",
                "--platform",
                "gpu",
                "--machine-shape",
                "NvidiaTeslaP100",
                "--dry-run",
                "--no-publish-branch",
                "--output-root",
                str(tmp_path),
                "--case-filter",
                "observer_only",
                "--memory-trace",
                "all",
                "--apt-package",
                "libglu1-mesa",
                "--pip-package",
                "nrv-py",
                "--nrv-conda-env",
                "--benchmark-env",
                "XLA_FLAGS=--xla_gpu_enable_command_buffer=FUSION,WHILE,CUSTOM_CALL",
            ]
        )
        == 0
    )

    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    package = run_dirs[0] / "kernel"
    metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
    config = json.loads((package / "kaggle_config.json").read_text(encoding="utf-8"))
    kernel_source = (package / "axonscope_benchmark_kernel.py").read_text(encoding="utf-8")

    assert (package / "axonscope_benchmark_kernel.py").is_file()
    assert "EMBEDDED_CONFIG: dict[str, Any] = json.loads(" in kernel_source
    assert '"repo_url"' in kernel_source
    assert metadata["id"] == "demo-user/axonscope-p11a-benchmarks"
    assert metadata["code_file"] == "axonscope_benchmark_kernel.py"
    assert metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaP100"
    assert config["script"] == "threshold_curves"
    assert config["preset"] == "gpu_smoke"
    assert config["platform"] == "gpu"
    assert config["require_gpu"] is True
    assert config["apt_packages"] == ["libglu1-mesa"]
    assert config["nrv_conda_env"] is True
    assert config["pip_packages"] == ["nrv-py", "triton", "jax-triton"]
    assert config["benchmark_env"] == {
        "XLA_FLAGS": "--xla_gpu_enable_command_buffer=FUSION,WHILE,CUSTOM_CALL"
    }
    assert config["benchmark_args"] == [
        "--case-filter",
        "observer_only",
        "--memory-trace",
        "all",
    ]


def test_kaggle_runner_dry_run_supports_time_chunk_campaign(tmp_path: Path):
    assert (
        run_kaggle(
            [
                "--username",
                "demo-user",
                "--slug",
                "axonscope-p11b-time-chunk-sweep-cpu",
                "--campaign",
                "time_chunk_sweep",
                "--script",
                "recruitment_curves",
                "--preset",
                "quick",
                "--platform",
                "cpu",
                "--machine-shape",
                "NvidiaTeslaP100",
                "--dry-run",
                "--no-publish-branch",
                "--output-root",
                str(tmp_path),
                "--policies",
                "default,unchunked,100",
                "--recording",
                "observer_only",
            ]
        )
        == 0
    )

    package = next(path for path in tmp_path.iterdir() if path.is_dir()) / "kernel"
    metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
    config = json.loads((package / "kaggle_config.json").read_text(encoding="utf-8"))
    kernel_source = (package / "axonscope_benchmark_kernel.py").read_text(encoding="utf-8")

    assert metadata["id"] == "demo-user/axonscope-p11b-time-chunk-sweep-cpu"
    assert metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaP100"
    assert config["campaign"] == "time_chunk_sweep"
    assert config["script"] == "recruitment_curves"
    assert config["platform"] == "cpu"
    assert config["require_gpu"] is False
    assert config["benchmark_args"] == [
        "--policies",
        "default,unchunked,100",
        "--recording",
        "observer_only",
    ]
    assert "benchmark/campaigns/time_chunk_sweep.py" in kernel_source


def test_kaggle_runner_dry_run_supports_double_cable_solver_policy_campaign(
    tmp_path: Path,
):
    assert (
        run_kaggle(
            [
                "--username",
                "demo-user",
                "--slug",
                "axonscope-p11c-solver-policy-gpu",
                "--campaign",
                "double_cable_solver_policy",
                "--preset",
                "gpu_smoke",
                "--platform",
                "gpu",
                "--machine-shape",
                "NvidiaTeslaP100",
                "--dry-run",
                "--no-publish-branch",
                "--output-root",
                str(tmp_path),
                "--curve-script",
                "recruitment_curves,threshold_curves",
                "--solver",
                "auto,tiled_thomas",
                "--n-axons",
                "64,4096",
                "--nx",
                "89",
                "--recording",
                "observer_only",
                "--tiled-thomas-block-b",
                "32",
            ]
        )
        == 0
    )

    package = next(path for path in tmp_path.iterdir() if path.is_dir()) / "kernel"
    metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
    config = json.loads((package / "kaggle_config.json").read_text(encoding="utf-8"))
    kernel_source = (package / "axonscope_benchmark_kernel.py").read_text(encoding="utf-8")

    assert metadata["id"] == "demo-user/axonscope-p11c-solver-policy-gpu"
    assert metadata["enable_gpu"] == "true"
    assert config["campaign"] == "double_cable_solver_policy"
    assert config["script"] is None
    assert config["preset"] == "gpu_smoke"
    assert config["platform"] == "gpu"
    assert config["require_gpu"] is True
    assert config["benchmark_args"] == [
        "--curve-script",
        "recruitment_curves,threshold_curves",
        "--solver",
        "auto,tiled_thomas",
        "--n-axons",
        "64,4096",
        "--nx",
        "89",
        "--recording",
        "observer_only",
        "--tiled-thomas-block-b",
        "32",
    ]
    assert "benchmark/campaigns/double_cable_solver_policy.py" in kernel_source


def test_kaggle_runner_dry_run_supports_single_cable_solver_policy_campaign(
    tmp_path: Path,
):
    assert (
        run_kaggle(
            [
                "--username",
                "demo-user",
                "--slug",
                "axonscope-p11-single-policy-gpu",
                "--campaign",
                "single_cable_solver_policy",
                "--preset",
                "gpu_smoke",
                "--platform",
                "gpu",
                "--machine-shape",
                "NvidiaTeslaP100",
                "--dry-run",
                "--no-publish-branch",
                "--output-root",
                str(tmp_path),
                "--curve-script",
                "recruitment_curves,threshold_curves",
                "--solver",
                "auto,jax_tridiagonal",
                "--n-axons",
                "64,4096",
                "--nx",
                "89",
                "--recording",
                "observer_only",
            ]
        )
        == 0
    )

    package = next(path for path in tmp_path.iterdir() if path.is_dir()) / "kernel"
    metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
    config = json.loads((package / "kaggle_config.json").read_text(encoding="utf-8"))
    kernel_source = (package / "axonscope_benchmark_kernel.py").read_text(encoding="utf-8")

    assert metadata["id"] == "demo-user/axonscope-p11-single-policy-gpu"
    assert metadata["enable_gpu"] == "true"
    assert config["campaign"] == "single_cable_solver_policy"
    assert config["script"] is None
    assert config["preset"] == "gpu_smoke"
    assert config["platform"] == "gpu"
    assert config["require_gpu"] is True
    assert config["benchmark_args"] == [
        "--curve-script",
        "recruitment_curves,threshold_curves",
        "--solver",
        "auto,jax_tridiagonal",
        "--n-axons",
        "64,4096",
        "--nx",
        "89",
        "--recording",
        "observer_only",
    ]
    assert "benchmark/campaigns/single_cable_solver_policy.py" in kernel_source


def test_kaggle_runner_cpu_shape_forces_cpu_platform(tmp_path: Path):
    assert (
        run_kaggle(
            [
                "--username",
                "demo-user",
                "--script",
                "recruitment_curves",
                "--preset",
                "quick",
                "--platform",
                "gpu",
                "--machine-shape",
                "cpu",
                "--dry-run",
                "--no-publish-branch",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )

    package = next(path for path in tmp_path.iterdir() if path.is_dir()) / "kernel"
    metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
    config = json.loads((package / "kaggle_config.json").read_text(encoding="utf-8"))

    assert metadata["enable_gpu"] == "false"
    assert "machine_shape" not in metadata
    assert config["platform"] == "cpu"
    assert config["require_gpu"] is False


def test_kaggle_runner_platform_cpu_defaults_to_cpu_kernel(tmp_path: Path):
    assert (
        run_kaggle(
            [
                "--username",
                "demo-user",
                "--script",
                "threshold_curves",
                "--platform",
                "cpu",
                "--dry-run",
                "--no-publish-branch",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )

    package = next(path for path in tmp_path.iterdir() if path.is_dir()) / "kernel"
    metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
    config = json.loads((package / "kaggle_config.json").read_text(encoding="utf-8"))

    assert metadata["enable_gpu"] == "false"
    assert "machine_shape" not in metadata
    assert config["preset"] == "quick"
    assert config["platform"] == "cpu"
    assert config["require_gpu"] is False


def test_kaggle_runner_can_run_cpu_path_on_gpu_machine(tmp_path: Path):
    assert (
        run_kaggle(
            [
                "--username",
                "demo-user",
                "--script",
                "threshold_curves",
                "--preset",
                "quick",
                "--platform",
                "cpu",
                "--machine-shape",
                "NvidiaTeslaP100",
                "--dry-run",
                "--no-publish-branch",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )

    package = next(path for path in tmp_path.iterdir() if path.is_dir()) / "kernel"
    metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
    config = json.loads((package / "kaggle_config.json").read_text(encoding="utf-8"))

    assert metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaP100"
    assert config["platform"] == "cpu"
    assert config["require_gpu"] is False


def test_kaggle_runner_cpu_shortcut_uses_quick_cpu_preset(tmp_path: Path):
    assert (
        run_kaggle(
            [
                "--username",
                "demo-user",
                "--script",
                "recruitment_curves",
                "--cpu",
                "--dry-run",
                "--no-publish-branch",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )

    package = next(path for path in tmp_path.iterdir() if path.is_dir()) / "kernel"
    metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
    config = json.loads((package / "kaggle_config.json").read_text(encoding="utf-8"))

    assert metadata["enable_gpu"] == "false"
    assert config["preset"] == "quick"
    assert config["platform"] == "cpu"
    assert config["require_gpu"] is False


def test_kaggle_hardware_metadata_redacts_sensitive_environment_values():
    assert _redact_env_value("KAGGLE_DATA_PROXY_TOKEN", "secret-token") == "<redacted:12 chars>"
    assert _redact_env_value("KAGGLE_API_V1_TOKEN", "/etc/secrets/kaggle/api-v1-token") == "<redacted:32 chars>"
    assert _redact_env_value("CUDA_VERSION", "12.8.1") == "12.8.1"


def test_kaggle_benchmark_environment_overrides_only_requested_values(monkeypatch):
    monkeypatch.setenv("UNCHANGED_VALUE", "base")

    environment = _benchmark_environment(
        {"benchmark_env": {"XLA_FLAGS": "--xla_gpu_enable_command_buffer=WHILE"}}
    )

    assert environment["UNCHANGED_VALUE"] == "base"
    assert environment["XLA_FLAGS"] == "--xla_gpu_enable_command_buffer=WHILE"
