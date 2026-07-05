from __future__ import annotations

import json
from pathlib import Path

from benchmark.kaggle.kernel_entry import _redact_env_value
from benchmark.kaggle.run_kernel import main as run_kaggle


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
    assert config["benchmark_args"] == [
        "--case-filter",
        "observer_only",
        "--memory-trace",
        "all",
    ]


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


def test_kaggle_hardware_metadata_redacts_sensitive_environment_values():
    assert _redact_env_value("KAGGLE_DATA_PROXY_TOKEN", "secret-token") == "<redacted:12 chars>"
    assert _redact_env_value("KAGGLE_API_V1_TOKEN", "/etc/secrets/kaggle/api-v1-token") == "<redacted:32 chars>"
    assert _redact_env_value("CUDA_VERSION", "12.8.1") == "12.8.1"
